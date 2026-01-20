import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
from io import BytesIO

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Smart Pricing Chesnaie - Yield V4.0", page_icon="📈", layout="wide")

# ============================================================
# PARAMETRES (SIDEBAR)
# ============================================================
st.sidebar.header("⚙️ Paramètres Yield")

# Conseils automatiques (règles simples)
OCC_HIGH = st.sidebar.slider("Seuil remplissage haut (%)", 0, 100, 70, 1) / 100.0
OCC_LOW  = st.sidebar.slider("Seuil remplissage bas (%)", 0, 100, 35, 1) / 100.0

# Bandes marché
BAND_MODE = st.sidebar.selectbox("Bande marché", ["Quantiles (Q25/Q50/Q75)", "±10% autour de la moyenne marché"])
BAND_PCT  = st.sidebar.slider("Bande ± (%) (si mode ±)", 0, 30, 10, 1) / 100.0

# Cap / Floor
APPLY_MARKET_FLOOR = st.sidebar.checkbox("Appliquer plancher marché", value=True)
APPLY_MARKET_CAP   = st.sidebar.checkbox("Appliquer plafond marché", value=True)

# Cap global vs prix actuel
GLOBAL_CAP_PCT = st.sidebar.slider("Plafond global vs prix Chesnaie (%)", 0, 80, 30, 1) / 100.0

# Météo
st.sidebar.subheader("🌤️ Météo")
USE_WEATHER = st.sidebar.checkbox("Activer météo (si dates disponibles)", value=True)
BONUS_SUN_PCT  = st.sidebar.slider("Bonus soleil (%)", 0, 20, 5, 1) / 100.0
MALUS_BAD_PCT  = st.sidebar.slider("Malus pluie/neige/orage (%)", 0, 30, 5, 1) / 100.0

# Evènements
st.sidebar.subheader("📅 Evènements")
USE_EVENTS = st.sidebar.checkbox("Activer évènements", value=True)

# ============================================================
# BASE "QUALITE / AVANTAGE" -> POIDS CONCURRENTS (modifiable)
# ============================================================
DEFAULT_WEIGHT = 1.00
COMPETITOR_WEIGHTS = {
    # Equivalents / marché comparable
    "Camping de Nibelle": 1.00,

    # Un peu moins bien mais avantage localisation (proche Paris)
    "Camping Le Parc des Roches": 1.05,
    "Parc des Roches": 1.05,
    "Camping Le Parc des Roches (eurocamp)": 1.05,

    # Meilleur + marque (Capfun) + proche Paris
    "Camping Heliomonde": 1.15,
    "Camping Heliomonde (cap fun)": 1.15,
    "Heliomonde": 1.15,
    "Capfun": 1.15,

    # Destination / haut de gamme
    "Les Bois du Bardelet": 1.15,
    "Camping Sandaya Les Alicourts": 1.30,
    "Sandaya Les Alicourts": 1.30,
    "Sandaya": 1.30,

    # Pro/étape qualité faible (si jamais présent dans benchmark)
    "Le Bois de la Justice": 0.85,
    "Camping des Lilas": 0.85,
}

# ============================================================
# EVENEMENTS PAR DEFAUT (fiables / nationaux + NSK officiel)
# ============================================================
DEFAULT_EVENTS = [
    # Ponts (France)
    {"Event": "Pont Ascension", "Start": "2026-05-13", "End": "2026-05-17", "Multiplier": 1.25},
    {"Event": "Pont Pentecôte", "Start": "2026-05-22", "End": "2026-05-25", "Multiplier": 1.25},

    # Karting NSK Angerville (dates officielles NSK 2026)
    {"Event": "NSK Angerville", "Start": "2026-05-28", "End": "2026-05-31", "Multiplier": 1.20},
]

# ============================================================
# METEO (Open-Meteo)
# ============================================================
LAT, LON = 48.31, 1.99
OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LAT}&longitude={LON}"
    "&daily=weathercode,temperature_2m_max"
    "&timezone=Europe%2FParis"
)

def wmo_bucket(code: int) -> str:
    try:
        c = int(code)
    except Exception:
        return "NEUTRE"
    if c in [0, 1, 2, 3]:
        return "SOLEIL"
    if c in [45, 48]:
        return "BROUILLARD"
    if 51 <= c <= 67 or 80 <= c <= 82:
        return "PLUIE"
    if 71 <= c <= 77 or 85 <= c <= 86:
        return "NEIGE"
    if 95 <= c <= 99:
        return "ORAGE"
    return "NEUTRE"

@st.cache_data(ttl=60 * 30)
def fetch_weather_daily():
    try:
        r = requests.get(OPEN_METEO_URL, timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

def build_weather_map(dates: list[dt.date]) -> tuple[dict, bool]:
    """
    Map {date -> {type, score, desc}}, degraded=True si API indisponible/parsing KO
    """
    degraded = False
    api = fetch_weather_daily()
    api_map = {}

    if not api:
        degraded = True
    else:
        try:
            api_dates = [dt.date.fromisoformat(d) for d in api["daily"]["time"]]
            codes = api["daily"]["weathercode"]
            temps = api["daily"]["temperature_2m_max"]
            for d, c, t in zip(api_dates, codes, temps):
                b = wmo_bucket(c)
                if b == "SOLEIL":
                    api_map[d] = {"type": "SOLEIL", "score": 1.0 + BONUS_SUN_PCT, "desc": f"Beau temps ({t}°C)"}
                elif b in ["PLUIE", "ORAGE", "NEIGE"]:
                    api_map[d] = {"type": b, "score": 1.0 - MALUS_BAD_PCT, "desc": f"{b.title()} ({t}°C)"}
                elif b == "BROUILLARD":
                    api_map[d] = {"type": "BROUILLARD", "score": 1.0, "desc": f"Brouillard ({t}°C)"}
                else:
                    api_map[d] = {"type": "NEUTRE", "score": 1.0, "desc": f"Nuageux ({t}°C)"}
        except Exception:
            degraded = True
            api_map = {}

    out = {}
    for d in dates:
        out[d] = api_map.get(d, {"type": "SAISON", "score": 1.0, "desc": "Hors prévision (neutre)"})
    return out, degraded

# ============================================================
# OUTILS
# ============================================================
def parse_date_any(x):
    if pd.isna(x):
        return None
    try:
        return pd.to_datetime(x, dayfirst=True).date()
    except Exception:
        try:
            return dt.date.fromisoformat(str(x))
        except Exception:
            return None

def segment_from_chambres(x):
    """
    Si 0 ou vide => EMPL
    Sinon 1CH / 2CH / 3CH+
    """
    if pd.isna(x):
        return "EMPL"
    try:
        v = float(x)
    except Exception:
        return "EMPL"
    if v <= 0:
        return "EMPL"
    v = int(round(v))
    return "3CH+" if v >= 3 else f"{v}CH"

def weighted_quantile(values, quantiles, weights=None):
    values = np.array(values, dtype=float)
    quantiles = np.array(quantiles, dtype=float)

    if weights is None:
        weights = np.ones(len(values), dtype=float)
    else:
        weights = np.array(weights, dtype=float)

    m = np.isfinite(values) & np.isfinite(weights)
    values = values[m]
    weights = weights[m]

    if len(values) == 0:
        return [np.nan] * len(quantiles)

    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    cdf = np.cumsum(weights) - 0.5 * weights
    cdf /= np.sum(weights)
    return np.interp(quantiles, cdf, values)

def compute_market_bands(g: pd.DataFrame):
    """
    g doit contenir Prix_semaine et Poids
    """
    vals = g["Prix_semaine"].to_numpy(dtype=float)
    wts  = g["Poids"].to_numpy(dtype=float)
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan, np.nan)  # mean, floor, mid, cap

    mean_w = float(np.average(vals, weights=wts))

    if BAND_MODE.startswith("Quantiles"):
        q25, q50, q75 = weighted_quantile(vals, [0.25, 0.50, 0.75], weights=wts)
        floor = float(q25)
        mid   = float(q50)
        cap   = float(q75)
    else:
        floor = mean_w * (1.0 - BAND_PCT)
        mid   = mean_w
        cap   = mean_w * (1.0 + BAND_PCT)

    return (mean_w, floor, mid, cap)

def advice_formula(occ, ch_price, market_mean):
    """
    Conseil Auto (Excel style) :
    - si remplissage haut ET prix < moyenne -> HAUSSE POSSIBLE
    - si remplissage bas ET prix > moyenne -> BAISSE POSSIBLE
    - sinon RAS
    """
    if occ is None or pd.isna(occ) or market_mean is None or pd.isna(market_mean):
        return "RAS"
    if occ >= OCC_HIGH and ch_price < market_mean:
        return "HAUSSE POSSIBLE"
    if occ <= OCC_LOW and ch_price > market_mean:
        return "BAISSE POSSIBLE"
    return "RAS"

def apply_events_multiplier(date_start: dt.date, date_end: dt.date, events_df: pd.DataFrame):
    """
    Applique les événements si la période [date_start; date_end] overlap un event.
    Multiplie les multipliers (cumulatif).
    """
    if date_start is None or date_end is None or events_df.empty:
        return 1.0, []

    mult = 1.0
    reasons = []
    for _, ev in events_df.iterrows():
        s = ev["Start_date"]
        e = ev["End_date"]
        if s is None or e is None:
            continue
        overlap = not (date_end < s or date_start > e)
        if overlap:
            m = float(ev["Multiplier"])
            mult *= m
            reasons.append(f"{ev['Event']} (x{m})")
    return mult, reasons

# ============================================================
# UI
# ============================================================
st.title("📈 Smart Pricing Chesnaie — Yield V4.0 (Marché + Indices + Météo + Évènements)")
st.caption("Basé sur tes fichiers benchmark & Chesnaie + formules yield (moyenne marché, indice, écart, conseil, RevPAR).")

colA, colB = st.columns(2)
with colA:
    bench_file = st.file_uploader("📂 Benchmark concurrents (xlsx/csv)", type=["xlsx", "csv"])
with colB:
    ches_file = st.file_uploader("📂 Tarifs Chesnaie (xlsx/csv)", type=["xlsx", "csv"])

events_file = st.file_uploader("📂 (Option) Fichier évènements (Event,Start,End,Multiplier)", type=["xlsx", "csv"])

occ_file = st.file_uploader("📂 (Option) Occupation (Date_début ou Mois_label + Segment + Occupation%)", type=["xlsx", "csv"])

def read_any(uploaded, sheet_name=None):
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded, sep=None, engine="python")
        except Exception:
            return pd.read_csv(uploaded, sep=";")
    else:
        if sheet_name:
            return pd.read_excel(uploaded, sheet_name=sheet_name)
        return pd.read_excel(uploaded)

# ---------- READ FILES ----------
bench = read_any(bench_file, sheet_name="01_Donnees_normalisees") if bench_file else None
ches  = read_any(ches_file,  sheet_name="01_Donnees_normalisees") if ches_file else None

if bench is None or ches is None:
    st.info("Charge ton benchmark + tes tarifs Chesnaie pour lancer l’analyse.")
    st.stop()

# ---------- NORMALISATION COLONNES ----------
# Benchmark attendu: Concurrent, Chambres, Mois_label, Prix (€)
# Chesnaie attendu: Chambres, Mois_label, Prix (€) pour 7 jours  (ou Prix (€))
def pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

bench_col_conc = pick_col(bench, ["Concurrent", "Nom du Concurrent", "Nom concurrent"])
bench_col_cham = pick_col(bench, ["Chambres", "Nb chambres", "Nbr chambres"])
bench_col_mois = pick_col(bench, ["Mois_label", "Mois", "mois"])
bench_col_prix = pick_col(bench, ["Prix (€)", "Prix", "Prix Semaine", "Prix semaine"])

ches_col_cham  = pick_col(ches, ["Chambres", "Nb chambres", "Nbr chambres"])
ches_col_mois  = pick_col(ches, ["Mois_label", "Mois", "mois"])
ches_col_prix7 = pick_col(ches, ["Prix (€) pour 7 jours", "Prix (€)", "Prix", "Prix Semaine", "Prix semaine"])

# Dates optionnelles (pour météo/évènements)
ches_col_start = pick_col(ches, ["Date_début", "Date debut", "Start", "Début", "Begin"])
ches_col_end   = pick_col(ches, ["Date_fin", "Date fin", "End", "Fin"])

missing = []
for name, col in [
    ("Benchmark: Concurrent", bench_col_conc),
    ("Benchmark: Chambres", bench_col_cham),
    ("Benchmark: Mois_label", bench_col_mois),
    ("Benchmark: Prix", bench_col_prix),
    ("Chesnaie: Chambres", ches_col_cham),
    ("Chesnaie: Mois_label", ches_col_mois),
    ("Chesnaie: Prix semaine", ches_col_prix7),
]:
    if col is None:
        missing.append(name)

if missing:
    st.error("Colonnes manquantes détectées :\n- " + "\n- ".join(missing))
    st.stop()

# ---------- PREP BENCH ----------
bench2 = bench.copy()
bench2["Concurrent"]   = bench2[bench_col_conc].astype(str).str.strip()
bench2["Chambres"]     = pd.to_numeric(bench2[bench_col_cham], errors="coerce").fillna(0)
bench2["Segment"]      = bench2["Chambres"].apply(segment_from_chambres)
bench2["Mois_label"]   = bench2[bench_col_mois].astype(str).str.strip()
bench2["Prix_semaine"] = pd.to_numeric(bench2[bench_col_prix], errors="coerce")

bench2["Poids"] = bench2["Concurrent"].map(COMPETITOR_WEIGHTS).fillna(DEFAULT_WEIGHT)
bench2 = bench2.dropna(subset=["Prix_semaine", "Mois_label", "Segment", "Concurrent"])

# ---------- PREP CHESNAIE ----------
ches2 = ches.copy()
ches2["Chambres"] = pd.to_numeric(ches2[ches_col_cham], errors="coerce").fillna(0)
ches2["Segment"] = ches2["Chambres"].apply(segment_from_chambres)
ches2["Mois_label"] = ches2[ches_col_mois].astype(str).str.strip()
ches2["Prix_Chesnaie_semaine"] = pd.to_numeric(ches2[ches_col_prix7], errors="coerce")

if ches_col_start:
    ches2["Date_début_obj"] = ches2[ches_col_start].apply(parse_date_any)
else:
    ches2["Date_début_obj"] = None

if ches_col_end:
    ches2["Date_fin_obj"] = ches2[ches_col_end].apply(parse_date_any)
else:
    # si on a Date_début, on suppose 7 jours
    ches2["Date_fin_obj"] = ches2["Date_début_obj"].apply(lambda d: (d + dt.timedelta(days=6)) if d else None)

ches2 = ches2.dropna(subset=["Prix_Chesnaie_semaine", "Mois_label", "Segment"])

# ---------- EVENTS ----------
events_df = pd.DataFrame(DEFAULT_EVENTS)
if events_file is not None:
    ev = read_any(events_file)
    # on tolère colonnes au cas où
    colE = pick_col(ev, ["Event", "Evenement", "Événement", "Nom"])
    colS = pick_col(ev, ["Start", "Date_début", "Debut", "Début"])
    colF = pick_col(ev, ["End", "Date_fin", "Fin"])
    colM = pick_col(ev, ["Multiplier", "Coef", "Coefficient", "Impact"])
    if colE and colS and colF and colM:
        ev2 = ev[[colE, colS, colF, colM]].copy()
        ev2.columns = ["Event", "Start", "End", "Multiplier"]
        events_df = pd.concat([events_df, ev2], ignore_index=True)
    else:
        st.warning("Fichier évènements ignoré (colonnes attendues : Event, Start, End, Multiplier).")

events_df["Start_date"] = events_df["Start"].apply(parse_date_any)
events_df["End_date"]   = events_df["End"].apply(parse_date_any)
events_df["Multiplier"] = pd.to_numeric(events_df["Multiplier"], errors="coerce").fillna(1.0)

# ---------- OCCUPATION (OPTION) ----------
occ_df = None
if occ_file is not None:
    occ_df = read_any(occ_file)
    # attendu : soit Date_début, soit (Mois_label + Segment), et un taux
    occ_col_rate = pick_col(occ_df, ["Occupation", "Taux", "Taux_occupation", "%_Occupation", "Occ"])
    occ_col_mois = pick_col(occ_df, ["Mois_label", "Mois"])
    occ_col_seg  = pick_col(occ_df, ["Segment"])
    occ_col_date = pick_col(occ_df, ["Date_début", "Start", "Début"])

    if occ_col_rate:
        occ_df["_occ"] = pd.to_numeric(occ_df[occ_col_rate], errors="coerce")
        # si c'est 0-100 on convertit
        if occ_df["_occ"].dropna().gt(1.0).any():
            occ_df["_occ"] = occ_df["_occ"] / 100.0
    else:
        occ_df = None
        st.warning("Fichier occupation ignoré (colonne taux non trouvée).")

# ============================================================
# RUN
# ============================================================
if st.button("🚀 LANCER L'ANALYSE"):
    # 1) Bandes marché par (Mois_label, Segment)
    bands_rows = []
    for (mois, seg), g in bench2.groupby(["Mois_label", "Segment"]):
        mean_w, floor, mid, cap = compute_market_bands(g)
        bands_rows.append({
            "Mois_label": mois,
            "Segment": seg,
            "Nb_points": len(g),
            "Moyenne_marche_ponderee": round(mean_w, 2),
            "Plancher_marche": round(floor, 2),
            "Cible_marche": round(mid, 2),
            "Plafond_marche": round(cap, 2),
        })
    bands = pd.DataFrame(bands_rows)

    # 2) Merge Chesnaie + bandes
    out = ches2.merge(bands, on=["Mois_label", "Segment"], how="left")

    # 3) Ajout occupation si dispo
    out["Occupation"] = np.nan
    if occ_df is not None:
        occ_col_mois = pick_col(occ_df, ["Mois_label", "Mois"])
        occ_col_seg  = pick_col(occ_df, ["Segment"])
        occ_col_date = pick_col(occ_df, ["Date_début", "Start", "Début"])
        if occ_col_date:
            occ_df["_date_obj"] = occ_df[occ_col_date].apply(parse_date_any)
            tmp = occ_df[["_date_obj", "_occ"]].dropna()
            out = out.merge(tmp, left_on="Date_début_obj", right_on="_date_obj", how="left")
            out["Occupation"] = out["_occ"]
            out = out.drop(columns=["_date_obj", "_occ"], errors="ignore")
        elif occ_col_mois and occ_col_seg:
            tmp = occ_df[[occ_col_mois, occ_col_seg, "_occ"]].copy()
            tmp.columns = ["Mois_label", "Segment", "_occ"]
            out = out.merge(tmp, on=["Mois_label", "Segment"], how="left")
            out["Occupation"] = out["_occ"]
            out = out.drop(columns=["_occ"], errors="ignore")

    # 4) Formules Yield
    # 4.1 Moyenne du marché (pondérée) déjà: Moyenne_marche_ponderee
    # 4.2 Indice de prix
    out["Indice_prix"] = (out["Prix_Chesnaie_semaine"] / out["Moyenne_marche_ponderee"]) * 100.0
    # 4.3 Ecart nominal
    out["Ecart_nominal"] = out["Prix_Chesnaie_semaine"] - out["Moyenne_marche_ponderee"]
    # 4.4 Conseil auto
    out["Conseil_auto"] = out.apply(
        lambda r: advice_formula(r.get("Occupation"), r["Prix_Chesnaie_semaine"], r["Moyenne_marche_ponderee"]),
        axis=1
    )
    # 4.5 RevPAR simplifié (prix * occ)
    out["RevPAR_simplifie"] = out.apply(
        lambda r: (r["Prix_Chesnaie_semaine"] * r["Occupation"]) if (r.get("Occupation") is not None and pd.notna(r.get("Occupation"))) else np.nan,
        axis=1
    )

    # 5) Météo (si dates dispo)
    weather_map = {}
    degraded_weather = False
    if USE_WEATHER and out["Date_début_obj"].notna().any():
        unique_dates = sorted({d for d in out["Date_début_obj"].dropna().tolist() if d is not None})
        weather_map, degraded_weather = build_weather_map(unique_dates)

    # 6) Evènements
    # 7) Proposition de prix (météo + events + garde-fous marché + cap global)
    suggestions = []
    for _, r in out.iterrows():
        base = float(r["Prix_Chesnaie_semaine"])
        reasons = []
        mult = 1.0

        # Evènements
        if USE_EVENTS:
            em, er = apply_events_multiplier(r.get("Date_début_obj"), r.get("Date_fin_obj"), events_df)
            mult *= em
            reasons.extend(er)

        # Météo
        if USE_WEATHER and r.get("Date_début_obj") is not None and r.get("Date_début_obj") in weather_map:
            w = weather_map[r["Date_début_obj"]]
            # Bonus soleil (hors haute saison non déterminée ici -> on reste simple)
            if w["type"] == "SOLEIL":
                mult *= w["score"]
                reasons.append(w["desc"])
            elif w["type"] in ["PLUIE", "NEIGE", "ORAGE"]:
                mult *= w["score"]
                reasons.append(f"Météo défavorable ({w['type']})")

        proposed = base * mult

        # Garde-fous marché
        floor = r.get("Plancher_marche")
        cap   = r.get("Plafond_marche")

        if APPLY_MARKET_FLOOR and pd.notna(floor) and proposed < float(floor):
            proposed = float(floor)
            reasons.append("Remonté au plancher marché")

        if APPLY_MARKET_CAP and pd.notna(cap) and proposed > float(cap):
            proposed = float(cap)
            reasons.append("Plafonné au plafond marché")

        # Cap global vs prix actuel
        max_global = base * (1.0 + GLOBAL_CAP_PCT)
        if proposed > max_global:
            proposed = max_global
            reasons.append(f"Plafond global +{int(GLOBAL_CAP_PCT*100)}%")

        suggestions.append((round(proposed, 2), " + ".join(reasons) if reasons else "Standard"))

    out["Prix_suggere"] = [x[0] for x in suggestions]
    out["Justification_sugg"] = [x[1] for x in suggestions]

    # 8) Mise en forme statut (indice)
    def flag_index(x):
        if pd.isna(x):
            return ""
        if x > 115:
            return "🔴 Trop cher ?"
        if x < 85:
            return "🟢 Trop bas ?"
        return "⚪ OK"

    out["Alerte_indice"] = out["Indice_prix"].apply(flag_index)

    # ============================================================
    # AFFICHAGE
    # ============================================================
    st.success("Analyse terminée ✅")

    if USE_WEATHER and degraded_weather:
        st.warning("⚠️ Mode dégradé météo : API indisponible/parsing KO. Météo ignorée ou neutre.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prix Chesnaie moyen (sem.)", f"{out['Prix_Chesnaie_semaine'].mean():.0f}€")
    k2.metric("Prix suggéré moyen (sem.)", f"{out['Prix_suggere'].mean():.0f}€")
    k3.metric("Semaines 'trop cher' (Indice>115)", f"{(out['Indice_prix']>115).sum()}")
    k4.metric("Semaines 'trop bas' (Indice<85)", f"{(out['Indice_prix']<85).sum()}")

    show_cols = [
        "Mois_label","Segment",
        "Prix_Chesnaie_semaine","Moyenne_marche_ponderee",
        "Indice_prix","Alerte_indice","Ecart_nominal",
        "Occupation","Conseil_auto","RevPAR_simplifie",
        "Plancher_marche","Cible_marche","Plafond_marche",
        "Prix_suggere","Justification_sugg"
    ]
    # garde seulement celles qui existent
    show_cols = [c for c in show_cols if c in out.columns]

    st.dataframe(out[show_cols], use_container_width=True)

    # Graph simple
    if {"Prix_Chesnaie_semaine","Prix_suggere"}.issubset(out.columns):
        st.line_chart(out[["Prix_Chesnaie_semaine","Prix_suggere"]])

    # Export
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        bench2.to_excel(writer, index=False, sheet_name="01_Benchmark_net")
        bands.to_excel(writer, index=False, sheet_name="02_Bandes_marche")
        events_df.drop(columns=["Start_date","End_date"], errors="ignore").to_excel(writer, index=False, sheet_name="03_Evenements")
        out.to_excel(writer, index=False, sheet_name="04_Resultats")

    st.download_button("📥 Télécharger résultats Excel", output.getvalue(), "Yield_Resultats_V4.xlsx")
