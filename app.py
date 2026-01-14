import streamlit as st
import pandas as pd
import requests
import datetime as dt
from io import BytesIO

# --- CONFIGURATION ---
st.set_page_config(page_title="Smart Pricing Chesnaie - V3.4", page_icon="📈", layout="wide")

# Base de connaissance concurrents (Prix planchers/plafonds)
COMPETITOR_KNOWLEDGE_BASE = {
    "BASSE_SAISON": {"Moyenne": 45, "Max": 60, "Min": 35},
    "MOYENNE_SAISON": {"Moyenne": 65, "Max": 85, "Min": 50},
    "HAUTE_SAISON": {"Moyenne": 95, "Max": 130, "Min": 75},
}

# Configuration Météo
LAT, LON = 48.31, 1.99
OPEN_METEO_URL = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LAT}&longitude={LON}"
    "&daily=weathercode,temperature_2m_max"
    "&timezone=Europe%2FParis"
)

# --- PRÉ-TRAITEMENT DES ÉVÉNEMENTS (Optimisation Performance) ---
RAW_EVENTS = [
    {"name": "Pont Ascension", "start": "2026-05-13", "end": "2026-05-17", "mult": 1.25},
    {"name": "Karting NSK", "start": "2026-05-29", "end": "2026-05-31", "mult": 1.20},
    {"name": "Pont Pentecôte", "start": "2026-05-22", "end": "2026-05-25", "mult": 1.25},
]

PROCESSED_EVENTS = [
    {
        "name": ev["name"],
        "start": dt.date.fromisoformat(ev["start"]),
        "end": dt.date.fromisoformat(ev["end"]),
        "mult": float(ev["mult"]),
    }
    for ev in RAW_EVENTS
]

# --- PARAMÈTRES AJUSTABLES (UX PRO) ---
st.sidebar.header("⚙️ Paramètres Yield (Ajustables)")
MAX_CAP_PCT = st.sidebar.slider("Plafond max (%)", min_value=0, max_value=80, value=30, step=1)
BONUS_SUN_PCT = st.sidebar.slider("Bonus soleil (%)", min_value=0, max_value=20, value=5, step=1)
MALUS_RAIN_PCT = st.sidebar.slider("Malus pluie (%)", min_value=0, max_value=30, value=5, step=1)
APPLY_FLOOR = st.sidebar.checkbox("Appliquer plancher marché", value=True)

# --- FONCTIONS UTILITAIRES ---
def get_season_data(date_obj: dt.date):
    if date_obj.month in [7, 8]:
        return "HAUTE_SAISON", COMPETITOR_KNOWLEDGE_BASE["HAUTE_SAISON"]
    if date_obj.month in [4, 5, 6, 9]:
        return "MOYENNE_SAISON", COMPETITOR_KNOWLEDGE_BASE["MOYENNE_SAISON"]
    return "BASSE_SAISON", COMPETITOR_KNOWLEDGE_BASE["BASSE_SAISON"]


@st.cache_data(ttl=60 * 30)
def fetch_weather_daily():
    """
    Récupère la météo quotidienne (J..J+7). Mise en cache 30 min.
    Ne doit pas faire planter l’app : renvoie None en cas d'échec.
    """
    try:
        r = requests.get(OPEN_METEO_URL, timeout=6)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def wmo_to_bucket(code: int) -> str:
    """
    Classification simple mais moins grossière que '>=50 = pluie'.
    Référence WMO weather codes (approximations utiles côté yield).
    """
    try:
        c = int(code)
    except Exception:
        return "NEUTRE"

    # Beau temps
    if c in [0, 1, 2, 3]:
        return "SOLEIL"

    # Brouillard
    if c in [45, 48]:
        return "BROUILLARD"

    # Pluie / bruine
    if 51 <= c <= 67 or 80 <= c <= 82:
        return "PLUIE"

    # Neige / grésil
    if 71 <= c <= 77 or 85 <= c <= 86:
        return "NEIGE"

    # Orage
    if 95 <= c <= 99:
        return "ORAGE"

    return "NEUTRE"


def build_weather_map(dates_in_excel: list[dt.date]) -> tuple[dict, bool]:
    """
    Crée un dictionnaire {date: info_météo}.
    Retourne (weather_map, degraded_mode).
    degraded_mode=True si l'API est indisponible ou parsing KO.
    """
    weather_map = {}
    api_data = fetch_weather_daily()

    api_map = {}
    degraded_mode = False

    if not api_data:
        degraded_mode = True
    else:
        try:
            api_dates = [dt.date.fromisoformat(d) for d in api_data["daily"]["time"]]
            codes = api_data["daily"]["weathercode"]
            temps = api_data["daily"]["temperature_2m_max"]

            for d, c, t in zip(api_dates, codes, temps):
                bucket = wmo_to_bucket(c)
                if bucket == "SOLEIL":
                    api_map[d] = {
                        "type": "SOLEIL",
                        "score": 1.0 + (BONUS_SUN_PCT / 100.0),
                        "desc": f"Beau temps ({t}°C)",
                    }
                elif bucket in ["PLUIE", "ORAGE", "NEIGE"]:
                    api_map[d] = {
                        "type": bucket,
                        "score": 1.0 - (MALUS_RAIN_PCT / 100.0),
                        "desc": f"{bucket.title()} prévue ({t}°C)",
                    }
                elif bucket == "BROUILLARD":
                    api_map[d] = {"type": "BROUILLARD", "score": 1.0, "desc": f"Brouillard ({t}°C)"}
                else:
                    api_map[d] = {"type": "NEUTRE", "score": 1.0, "desc": f"Nuageux ({t}°C)"}

        except Exception:
            degraded_mode = True
            api_map = {}

    for d in dates_in_excel:
        if d in api_map:
            weather_map[d] = api_map[d]
        else:
            # Hors prévision (ou mode dégradé)
            weather_map[d] = {"type": "SAISON", "score": 1.0, "desc": "Saisonnier (Hors prévision)"}

    return weather_map, degraded_mode


def parse_date(value):
    try:
        return pd.to_datetime(value, dayfirst=True).date()
    except Exception:
        return None


def calculate_smart_price(date_obj: dt.date, base_price: float, weather_map: dict):
    """
    Retourne un tuple strict (Nouveau_Prix, Ref_Concurrents, Justification, Statut).
    """
    if date_obj is None:
        return (base_price or 0, "N/A", "ERREUR: Date invalide", "⚪")
    if base_price is None or pd.isna(base_price) or float(base_price) <= 0:
        return (0, "N/A", "Prix nul ou absent", "⚪")

    base_price = float(base_price)

    season_key, season_data = get_season_data(date_obj)
    reasons = []
    mult = 1.0

    # 1) Météo
    w = weather_map.get(date_obj, {"type": "INFO", "score": 1.0, "desc": "Météo inconnue"})
    if w["type"] == "SOLEIL" and season_key != "HAUTE_SAISON":
        mult *= w["score"]
        reasons.append(w["desc"])
    elif w["type"] in ["PLUIE", "ORAGE", "NEIGE"]:
        mult *= w["score"]
        reasons.append(f"Ajustement météo ({w['type'].title()} -{MALUS_RAIN_PCT}%)")

    # 2) Événements
    for ev in PROCESSED_EVENTS:
        if ev["start"] <= date_obj <= ev["end"]:
            mult *= ev["mult"]
            reasons.append(ev["name"])

    # 3) Calcul + garde-fous
    final_price = base_price * mult

    # Plafond
    cap_mult = 1.0 + (MAX_CAP_PCT / 100.0)
    max_price = base_price * cap_mult
    if final_price > max_price:
        final_price = max_price
        reasons.append(f"(Plafonné +{MAX_CAP_PCT}%)")

    # Plancher marché
    if APPLY_FLOOR and final_price < season_data["Min"]:
        final_price = float(season_data["Min"])
        reasons.append(f"Plancher Marché ({season_data['Min']}€)")

    # Statut visuel
    if final_price > base_price:
        status = "🟢"
    elif final_price < base_price:
        status = "🔴"
    else:
        status = "⚪"

    return (
        round(final_price, 2),
        f"{season_data['Moyenne']}€",
        " + ".join(reasons) if reasons else "Standard",
        status,
    )


# --- INTERFACE UTILISATEUR ---
st.title("📈 Smart Pricing Chesnaie - V3.4")
st.caption("V3.4 corrigée : météo batch + mode dégradé + WMO affiné + paramètres yield + gestion erreurs.")

uploaded_file = st.file_uploader("📂 Importez votre fichier CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file:
    # Lecture fichier robuste
    if uploaded_file.name.endswith(".csv"):
        try:
            df_raw = pd.read_csv(uploaded_file, sep=None, engine="python")
        except Exception:
            df_raw = pd.read_csv(uploaded_file, sep=";")
    else:
        df_raw = pd.read_excel(uploaded_file)

    df = df_raw.copy()

    # Détection colonnes
    col_date_guess = [c for c in df.columns if "date" in c.lower() or "start" in c.lower()]
    col_price_guess = [c for c in df.columns if "price" in c.lower() or "tarif" in c.lower()]

    c1, c2 = st.columns(2)
    date_col = c1.selectbox(
        "Colonne DATE", df.columns, index=df.columns.get_loc(col_date_guess[0]) if col_date_guess else 0
    )
    price_col = c2.selectbox(
        "Colonne PRIX", df.columns, index=df.columns.get_loc(col_price_guess[0]) if col_price_guess else 0
    )

    # Préparation données (sans perdre l’info : on garde un rapport d’erreurs)
    df["_date_obj"] = df[date_col].apply(parse_date)
    df["Price"] = pd.to_numeric(df[price_col], errors="coerce")

    df_errors = df[df["_date_obj"].isna() | df["Price"].isna() | (df["Price"] <= 0)].copy()
    df_valid = df.dropna(subset=["_date_obj", "Price"]).copy()
    df_valid = df_valid[df_valid["Price"] > 0].copy()

    if len(df_errors) > 0:
        st.warning(
            f"⚠️ {len(df_errors)} lignes ignorées (date invalide et/ou prix manquant/nul). "
            "Elles seront disponibles dans l’export (onglet 'Erreurs')."
        )

    if st.button("🚀 LANCER L'ANALYSE"):
        with st.spinner("Récupération Météo & Calculs Yield..."):
            # Météo batch
            unique_dates = sorted(df_valid["_date_obj"].unique())
            weather_map, degraded_mode = build_weather_map(unique_dates)

            if degraded_mode:
                st.warning("⚠️ Mode dégradé météo : API indisponible ou parsing en échec. Calcul sans météo réelle.")

            # Calcul
            results = df_valid.apply(
                lambda r: calculate_smart_price(r["_date_obj"], r["Price"], weather_map),
                axis=1,
            )
            df_valid[["Nouveau_Prix", "Ref_Concurrents", "Justification", "Statut"]] = pd.DataFrame(
                results.tolist(), index=df_valid.index
            )

            st.success("Calcul terminé !")

            # KPIs (sur lignes valides)
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            mean_initial = df_valid["Price"].mean()
            mean_optimized = df_valid["Nouveau_Prix"].mean()
            diff = mean_optimized - mean_initial

            kpi1.metric("Prix Moyen Initial", f"{mean_initial:.2f}€")
            kpi2.metric("Prix Moyen Optimisé", f"{mean_optimized:.2f}€", delta=f"{diff:.2f}€")
            kpi3.metric("Jours modifiés", f"{(df_valid['Statut'] != '⚪').sum()}")
            kpi4.metric("Opportunités (hausse)", f"{(df_valid['Statut'] == '🟢').sum()}")

            # Graphique (index sur _date_obj pour éviter tri cassé)
            df_plot = df_valid.sort_values("_date_obj").set_index("_date_obj")[["Price", "Nouveau_Prix"]]
            st.line_chart(df_plot)

            # Tableau détaillé
            def statut_color(v):
                if v == "🟢":
                    return "color: green; font-weight: 700;"
                if v == "🔴":
                    return "color: red; font-weight: 700;"
                return ""

            st.dataframe(
                df_valid[[date_col, price_col, "Nouveau_Prix", "Ref_Concurrents", "Justification", "Statut"]]
                .style.applymap(statut_color, subset=["Statut"]),
                use_container_width=True,
            )

            # Export Excel (2 onglets : Optimisation + Erreurs)
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df_export = df_valid.drop(columns=["_date_obj"], errors="ignore").copy()
                df_export.to_excel(writer, index=False, sheet_name="Optimisation")

                if len(df_errors) > 0:
                    df_errors_export = df_errors.drop(columns=["_date_obj"], errors="ignore").copy()
                    df_errors_export.to_excel(writer, index=False, sheet_name="Erreurs")

            st.download_button("📥 Télécharger Excel Optimisé", output.getvalue(), "SmartPricing_Resultats.xlsx")
