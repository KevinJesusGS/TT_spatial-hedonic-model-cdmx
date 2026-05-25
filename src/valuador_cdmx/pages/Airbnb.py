# =============================================================================
# TRABAJO TERMINAL
# Valuador Interactivo para Listings Airbnb en CDMX
# Basado en el modelo hedónico-espacial de airbnb.py
#
# Autores: Kevin Jesús González Sosa, José Manuel Torres Gutiérrez
# Institución: Escuela Superior de Cómputo
# Fecha: Mayo, 2026
# =============================================================================

import os
import warnings
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point, shape
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import NearestNeighbors
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN DE LA INTERFAZ
# ============================================================
st.set_page_config(
    page_title="Valuador Airbnb CDMX · ESCOM-IPN",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Parche sidebar — fondo sólido garantizado (idéntico al original)
import streamlit.components.v1 as _stcomp
_SIDEBAR_PATCH = (
    "<style>"
    "section[data-testid='stSidebar'],"
    "section[data-testid='stSidebar']>div,"
    "section[data-testid='stSidebar']>div>div,"
    "section[data-testid='stSidebar']>div>div>div"
    "{background-color:var(--secondary-background-color,#1e293b)!important;"
    "backdrop-filter:none!important;-webkit-backdrop-filter:none!important;"
    "opacity:1!important;}"
    "</style>"
)
_stcomp.html(_SIDEBAR_PATCH, height=0, scrolling=False)

# ============================================================
# BANNER DE NAVEGACIÓN AL PORTAL PRINCIPAL
# ============================================================
_nav_col1, _nav_col2, _nav_col3 = st.columns([1, 4, 1])
with _nav_col1:
    if st.button("← Inicio", help="Regresar al portal principal"):
        st.switch_page("Inicio.py")
with _nav_col2:
    st.markdown("""
    <div style="text-align:center;padding:0.35rem 0;font-size:0.82rem;
                color:var(--text-color);opacity:0.6;">
        📍 Módulo activo: <b>🏠 Airbnb</b> &nbsp;·&nbsp; 
        🏢 Compra/Venta &nbsp;·&nbsp; 🏘️ Rentas
    </div>
    """, unsafe_allow_html=True)
with _nav_col3:
    st.markdown("")

st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)



# ============================================================
# LOGOS INSTITUCIONALES
# ============================================================
import base64 as _b64

def _img_to_b64(path):
    try:
        with open(path, "rb") as _f:
            return _b64.b64encode(_f.read()).decode()
    except Exception:
        return ""

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
def _find_logo(filename):
    for candidate in [
        os.path.join(_SCRIPT_DIR, filename),
        os.path.join(_SCRIPT_DIR, "assets", filename),
        os.path.join(_SCRIPT_DIR, "..", "..", "..", "outputs", "assets", filename),
    ]:
        if os.path.exists(candidate):
            return candidate
    return ""

_escom_b64 = _img_to_b64(_find_logo("escom.png"))
_ipn_b64   = _img_to_b64(_find_logo("ipn.png"))

# ============================================================
# CSS GLOBAL (idéntico al original)
# ============================================================
st.markdown("""
<style>
.stApp {
    background-color: var(--background-color) !important;
    color: var(--text-color) !important;
}
.custom-card {
    background-color: var(--secondary-background-color) !important;
    border: 1px solid var(--border-color) !important;
    padding: 1.5rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.custom-card h1, .custom-card h2, .custom-card h3, .custom-card h4 {
    color: var(--text-color) !important;
    margin-top: 0 !important;
    margin-bottom: 0.5rem !important;
    font-weight: 700 !important;
}
.custom-card p, .custom-card span, .custom-card div {
    color: var(--text-color) !important;
    opacity: 0.95;
}
* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    -webkit-font-smoothing: antialiased;
}
.block-container {
    padding-top: 2rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 1600px !important;
}
h1 span, h1 code, .stApp h1 iframe {
    color: var(--text-color) !important;
}
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] *:not(img):not(svg):not(canvas) {
    background-color: var(--secondary-background-color) !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}
section[data-testid="stSidebar"] {
    border-right: 1px solid var(--border-color) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.4) !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: var(--text-color) !important;
}
[data-testid="stSelectbox"] label, [data-testid="stSlider"] label,
[data-testid="stNumberInput"] label {
    font-size: 0.875rem !important;
    font-weight: 600 !important;
    color: var(--text-color) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--background-color) !important;
    border: 1.5px solid var(--border-color) !important;
    border-radius: 10px !important;
    color: var(--text-color) !important;
}
[data-testid="stNumberInput"] input {
    background: var(--background-color) !important;
    border: 1.5px solid var(--border-color) !important;
    border-radius: 10px !important;
    color: var(--text-color) !important;
}
[data-testid="stTabs"] button {
    font-size: 1rem;
    font-weight: 600;
    padding: 0.875rem 1.75rem;
    color: var(--text-color) !important;
    opacity: 0.6;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    opacity: 1 !important;
    color: #3B82F6 !important;
    background: var(--secondary-background-color) !important;
}
[data-testid="stMetric"] {
    background: var(--secondary-background-color) !important;
    border: 1.5px solid var(--border-color) !important;
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-color) !important;
    font-weight: 600 !important;
    opacity: 0.85 !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-color) !important;
    font-weight: 800 !important;
}
[data-testid="stExpander"] {
    background: var(--background-color) !important;
    border: 1.5px solid var(--border-color) !important;
    border-radius: 16px !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-color) !important;
    background: var(--secondary-background-color) !important;
}
hr {
    border: none !important;
    height: 1px !important;
    background: var(--border-color) !important;
    margin: 2rem 0 !important;
}
h1, h2, h3, h4, h5, h6 {
    color: var(--text-color) !important;
}
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--background-color); }
::-webkit-scrollbar-thumb { background: #3B82F6; border-radius: 4px; }
.servicio-badge {
    padding: 10px 12px !important;
    background: var(--secondary-background-color) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
}
.servicio-badge .s-numero {
    color: var(--text-color) !important;
    font-weight: bold !important;
    font-size: 1.1rem !important;
    display: block !important;
    margin-bottom: 4px !important;
}
.servicios-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: 12px;
    margin: 1rem 0;
}
@media screen and (max-width: 1024px) {
    .block-container { padding-left: 1.5rem !important; padding-right: 1.5rem !important; max-width: 100% !important; }
    .authors-section { display: none !important; }
    .header-content  { gap: 1rem !important; }
    .main-title      { font-size: 1.35rem !important; }
    [data-testid="stTabs"] button { font-size: 0.85rem !important; padding: 0.6rem 1rem !important; }
}
@media screen and (max-width: 640px) {
    /* Contenedor principal: sin padding lateral excesivo */
    .block-container {
        padding-top: 0.75rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }

    /* ── Header institucional en móvil ── */
    .premium-header { padding: 1rem !important; border-radius: 14px !important; }
    .header-content { flex-direction: column !important; align-items: center !important; gap: 0.75rem !important; text-align: center !important; }
    .logo-section { justify-content: center !important; gap: 1rem !important; }
    .logo-divider { display: none !important; }
    .logo-box { width: 52px !important; height: 52px !important; }
    .logo-box img { height:48px !important; width:48px !important; }
    .title-section { text-align: center !important; }
    .main-title { font-size: 1.1rem !important; letter-spacing: -0.02em !important; }
    .subtitle { font-size: 0.78rem !important; }
    .authors-section { display: none !important; }

    /* ── Tabs: scroll horizontal en móvil ── */
    [data-testid="stTabs"] [role="tablist"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: none !important;
    }
    [data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar { display: none !important; }
    [data-testid="stTabs"] button {
        font-size: 0.8rem !important;
        padding: 0.5rem 0.85rem !important;
        white-space: nowrap !important;
        flex-shrink: 0 !important;
    }

    /* ── Sidebar móvil: fondo sólido y contraste de texto garantizado ── */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        background-color: var(--background-color, #ffffff) !important;
        color: var(--text-color, #31333F) !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        opacity: 1 !important;
    }

    /* CORRECCIÓN: Volver transparente SOLO el contenedor interno del slider */
    section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] > div {
        background-color: transparent !important;
    }

    /* Forzar la visibilidad del punto/nodo interactivo */
    section[data-testid="stSidebar"] [class*="stSlider"] [role="slider"] {
        background-color: #FF5A5F !important;
        border: 2px solid #ffffff !important;
        opacity: 1 !important;
        z-index: 9999 !important;
    }

    /* Forzar el color en la barra de progreso activa */
    section[data-testid="stSidebar"] [class*="stSlider"] div[data-track="true"] {
        background-color: #FF5A5F !important;
        opacity: 1 !important;
    }

    /* ── SOLUCIÓN NÚMERO INPUTS ── */
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] {
        background-color: transparent !important;
    }
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
        color: #31333F !important;
        background-color: #F0F2F6 !important;
    }

    /* Forzar visibilidad y color de los botones (+) y (-) */
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
        background-color: #E0E3E9 !important;
        color: #31333F !important;
        opacity: 1 !important;
        border: 1px solid #D1D5DB !important;
    }

    /* Asegurar que los iconos SVG (+ y -) se pinten oscuros */
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg {
        fill: #31333F !important;
        color: #31333F !important;
    }

    /* Forzar el color correcto en etiquetas de inputs y títulos */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] span {
        color: var(--text-color, #31333F) !important;
    }

    /* Estabilización estructural de la barra en móviles */
    section[data-testid="stSidebar"] {
        min-width: 300px !important;
        box-shadow: 8px 0 32px rgba(0, 0, 0, 0.15) !important;
        z-index: 999999 !important;
    }

    /* ── Inputs en móvil: más altura táctil ── */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stNumberInput"] input {
        min-height: 44px !important;
        font-size: 1rem !important;
    }

    /* ── Métricas: 1 por fila ── */
    [data-testid="stMetric"] { padding: 1rem !important; border-radius: 12px !important; }

    /* ── Galería de mapas: 1 columna ── */
    [data-testid="column"] { min-width: 100% !important; padding: 0 !important; }

    /* ── Badges de servicios ── */
    .servicio-badge { padding: 8px 8px !important; }
    .servicio-badge .s-numero { font-size: 0.95rem !important; }
    .servicios-grid {
        grid-template-columns: repeat(auto-fit, minmax(80px, 1fr)) !important;
        gap: 8px !important;
    }

    /* ── Tipografía global reducida ── */
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }

    /* ── Sidebar: secciones y estrella-rating ── */
    .sidebar-section-label {
        font-size: 0.72rem !important;
        padding: 0.25rem 0.6rem !important;
    }
    .star-rating-widget .star-label {
        font-size: 0.75rem !important;
    }
    .star-rating-widget .star-btn {
        font-size: 1.4rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# ENCABEZADO INSTITUCIONAL PREMIUM (idéntico al original)
# ============================================================
st.markdown("""
<style>
.premium-header {
    background: var(--secondary-background-color);
    border-radius: 24px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    border: 1px solid var(--border-color);
    position: relative;
    overflow: hidden;
}
.premium-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent, #3B82F6, #10B981, #3B82F6, transparent);
    animation: shimmer 3s ease-in-out infinite;
}
@keyframes shimmer {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
}
.header-content { display: flex; align-items: center; justify-content: space-between; gap: 2rem; }
.logo-section { display: flex; align-items: center; gap: 1.5rem; }
.logo-box { width: 70px; height: 70px; background: transparent !important; border-radius: 16px; display: flex; align-items: center; justify-content: center; }
.logo-divider { width: 1px; height: 60px; background: linear-gradient(to bottom, transparent, var(--border-color), transparent); }
.title-section { flex: 1; text-align: center; }
.main-title { color: var(--text-color) !important; font-size: 1.75rem; font-weight: 900; letter-spacing: -0.03em; margin-bottom: 0.5rem; }
.subtitle { color: var(--text-color); opacity: 0.8; font-size: 0.875rem; font-weight: 500; }
.institution-badge { display: inline-block; background: rgba(59, 130, 246, 0.15); border: 1px solid #3B82F6; color: #3B82F6; padding: 0.35rem 1rem; border-radius: 999px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; margin-top: 0.5rem; }
.authors-section { text-align: right; color: var(--text-color); opacity: 0.7; font-size: 0.8125rem; }
.author-name { color: var(--text-color); font-weight: 600; }
</style>
""", unsafe_allow_html=True)

_header_html = f"""
<div class="premium-header">
    <div class="header-content">
        <div class="logo-section">
            <div class="logo-box">
                {f'<img src="data:image/png;base64,{_ipn_b64}" style="height:66px;width:66px;object-fit:contain;border-radius:12px;">' if _ipn_b64 else '<span style="font-size:2.2rem;">🏛️</span>'}
            </div>
            <div class="logo-divider"></div>
            <div class="logo-box">
                {f'<img src="data:image/png;base64,{_escom_b64}" style="height:66px;width:66px;object-fit:contain;border-radius:12px;">' if _escom_b64 else '<span style="font-size:2.2rem;">🎓</span>'}
            </div>
        </div>
        <div class="title-section">
            <div class="main-title">🏠 Sistema de Valuación Airbnb · CDMX</div>
            <div class="subtitle">Escuela Superior de Cómputo • Instituto Politécnico Nacional • 2026</div>
            <div class="institution-badge">Trabajo Terminal</div>
        </div>
        <div class="authors-section">
            <div class="author-name">Kevin J. González Sosa</div>
            <div class="author-name">José M. Torres Gutiérrez</div>
            <div style="margin-top:0.5rem; font-size:0.75rem; opacity:0.7;">ESCOM-IPN</div>
        </div>
    </div>
</div>
"""
st.markdown(_header_html, unsafe_allow_html=True)

# ============================================================
# RUTAS
# ============================================================
PROJECT_ROOT      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH         = os.path.join(PROJECT_ROOT, "..", "..", "..", "data")
RAW_DATA_PATH     = os.path.join(DATA_PATH, "raw")
OUTPUTS_PATH_AIRBNB = os.path.join(PROJECT_ROOT, "..", "..", "..", "outputs", "airbnb", "results")

PATH_QGIS_DATA = os.path.join(OUTPUTS_PATH_AIRBNB, "resultados_qgis_airbnb.csv")
PATH_PROCESSED = os.path.join(DATA_PATH, "processed", "dataset_geocodificado_airbnb.csv")
if not os.path.exists(PATH_QGIS_DATA):
    PATH_QGIS_DATA = PATH_PROCESSED

PATH_COLONIAS         = os.path.join(RAW_DATA_PATH, "coloniascdmx", "colonias_iecm.shp")
PATH_METRO_EST        = os.path.join(RAW_DATA_PATH, "stcmetro_shp", "STC_Metro_estaciones_utm14n.shp")
PATH_METROBUS_EST     = os.path.join(RAW_DATA_PATH, "mb_shp", "Metrobus_estaciones.shp")
PATH_TREN_EST         = os.path.join(RAW_DATA_PATH, "ste_shp", "ste_tren_ligero_shp",
                                     "ste_tren_ligero_shp", "STE_TrenLigero_estaciones_utm14n.shp")
PATH_TROLE_PARADAS    = os.path.join(RAW_DATA_PATH, "ste_shp", "ste_trolebus_shp",
                                     "ste_trolebus_shp", "STE_Trolebus_Paradas.shp")
PATH_CABLE_EST        = os.path.join(RAW_DATA_PATH, "ste_shp", "ste_cablebus_shp",
                                     "ste_cablebus_shp", "STE_Cablebus_estaciones.shp")
PATH_AREAS_VERDES     = os.path.join(RAW_DATA_PATH, "inventario_areas_verdes_1",
                                     "inventario_areas_verdes_1.shp")
PATH_SALUD_BASE       = os.path.join(RAW_DATA_PATH, "hospitales_y_centros_de_salud",
                                     "hospitales_y_centros_de_salud.shp")
PATH_HOSPITALES_PUBLICOS = os.path.join(RAW_DATA_PATH, "hospitales_2020_publicos",
                                        "hospitales_2020_publicos.shp")
PATH_COMERCIO         = os.path.join(RAW_DATA_PATH, "cypc", "C_PComerciales.shp")
PATH_ESCUELAS_PRIVADAS = os.path.join(RAW_DATA_PATH, "escuelas_privadas", "escuelas_privadas.shp")
PATH_ESCUELAS_PUBLICAS = os.path.join(RAW_DATA_PATH, "escuelas_publicas", "escuelas_publicas.shp")
PATH_CICLOVIAS        = os.path.join(RAW_DATA_PATH, "infraestructura_vial_ciclista",
                                     "Infraestructura ciclista total.shp")

# ============================================================
# FEATURES (alineadas con airbnb.py)
# ============================================================
STATIC_FEATURES_AIRBNB = [
    "accommodates", "bedrooms", "bathrooms", "beds",
    "rt_entire", "rt_private", "rt_hotel",
    "host_profesional", "es_superhost", "host_antiguedad",
    "es_listing_veterano", "min_nights_log", "flexibilidad_reserva",
    "review_score_compuesto", "review_scores_rating", "review_scores_cleanliness",
    "review_scores_location", "n_reviews_log", "tasa_ocupacion", "pct_disponible",
    "n_amenidades_log", "ratio_amenidades_premium", "score_cocina", "score_seguridad",
    "score_confort", "score_premium", "score_familia", "score_trabajo", "score_accesibilidad",
    "am_wifi", "am_air_conditioning", "am_pool", "am_hot_tub", "am_gym",
    "am_elevator", "am_washer", "am_free_parking_on_premises", "am_self_check-in",
    "am_dedicated_workspace", "am_breakfast", "am_indoor_fireplace", "am_bbq_grill",
    "am_patio_or_balcony", "am_pets_allowed", "am_long_term_stays_allowed",
    "review_x_gentrif", "premium_x_15min", "superhost_x_review", "amenidades_x_pdi",
    "density_entire_500m", "profesional_x_flex", "entire_x_premium", "private_x_clean",
    "hotel_x_confort", "entire_x_area_cat",
    "dist_metro_m", "density_metro", "dist_metrobus_m", "density_metrobus",
    "dist_tren_m", "density_tren", "dist_trole_m", "density_trole",
    "dist_cable_m", "density_cable", "dist_pdi_log", "dist_pdi_cercano_m",
    "competencia_misma_cap", "marginalidad_score", "comercio_density",
    "pct_migrantes_turistas", "gentrification_index", "area_x_marginalidad",
    "area_X_gentrif", "gentrif_x_metro", "dist_ciclovia_m", "densidad_ciclovia_15m",
    "dist_area_verde_m", "densidad_parques_15m", "dist_salud_m", "acceso_salud_15m",
    "dist_escuela_m", "acceso_educacion_15m", "score_15min", "prox_ciclovia",
    "prox_parque", "prox_salud", "prox_escuela", "uso_mixto", "15min_X_gentrif",
    "listing_density_log", "verde_marginalidad_ratio", "salud_marginalidad_ratio",
    "educacion_marginalidad_ratio", "cat_mean_valor_suelo_200m", "cat_mean_vus_200m",
    "cat_mean_antiguedad_200m", "cat_std_valor_suelo_200m", "cat_mean_ratio_construccion_200m",
    "cat_density_predios_200m", "cat_mean_sup_construccion_200m", "cat_densidad_edificatoria_200m",
    "cat_area_imputable",
]

DYNAMIC_FEATURES = ["spatial_lag_price", "precio_vecinal_local", "lag_x_accommodates"]

# ============================================================
# HELPERS
# ============================================================
def clean_text(txt):
    if pd.isna(txt):
        return np.nan
    txt = str(txt).lower().strip()
    txt = txt.translate(str.maketrans("áéíóúüñ", "aeiouun"))
    return txt

def nearest_distance(pts, coords):
    if len(pts) == 0:
        return np.ones(len(coords)) * 5000
    tree = cKDTree(pts)
    d, _ = tree.query(coords)
    return d

def line_to_points(gdf_line, interval_m=100):
    if gdf_line is None or gdf_line.empty:
        return np.empty((0, 2))
    puntos = []
    for geom in gdf_line.geometry:
        if geom is None or geom.is_empty:
            continue
        length = geom.length
        if length == 0:
            continue
        num_points = max(2, int(length / interval_m) + 1)
        distances = np.linspace(0, length, num_points)
        for dist in distances:
            pt = geom.interpolate(dist)
            puntos.append((pt.x, pt.y))
    return np.array(puntos)

def density_proxy_from_points(coords_utm, pts, radio=1200):
    if len(pts) == 0:
        return np.zeros(len(coords_utm))
    tree = cKDTree(pts)
    counts = [len(tree.query_ball_point(xy, r=radio)) for xy in coords_utm]
    return np.array(counts)

def fmt_dist(m):
    return f"{m:.0f} m" if m < 1000 else f"{m/1000:.1f} km"

# ============================================================
# ENCABEZADO DE SECCIÓN (idéntico al original)
# ============================================================
def encabezado_seccion(titulo, subtitulo=None, icono=None):
    icono_html = f'<span style="margin-right:0.75rem; font-size:1.75rem;">{icono}</span>' if icono else ''
    subtitulo_html = (
        f'<div style="color:var(--text-muted); font-size:0.9375rem; margin-top:0.5rem;">{subtitulo}</div>'
        if subtitulo else ''
    )
    st.markdown(f"""
    <div style="margin:2.5rem 0 1.5rem 0;">
        <div style="display:flex; align-items:center;">
            {icono_html}
            <h3 style="margin:0; color:var(--text-primary); font-size:1.5rem; font-weight:800;
                        letter-spacing:-0.01em;">
                {titulo}
            </h3>
        </div>
        {subtitulo_html}
        <div style="width:60px; height:4px; background:linear-gradient(90deg,#3B82F6,#10B981);
                    border-radius:2px; margin-top:0.75rem;"></div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SISTEMA PRINCIPAL (CACHEADO)
# ============================================================
@st.cache_resource
def preparar_sistema_airbnb():
    # 1. Cargar datos
    df = pd.read_csv(PATH_QGIS_DATA)
    required_cols = ["price", "latitude", "longitude"]
    for col in required_cols:
        if col not in df.columns:
            st.error(f"Falta columna obligatoria: {col}")
            st.stop()

    # 2. Limpieza inicial
    df = df.dropna(subset=["latitude", "longitude", "price"]).copy()
    df = df.reset_index(drop=True)

    # 3. Segmentación territorial con KMeans en coordenadas UTM
    gdf_train = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    ).to_crs("EPSG:32614")
    df["utm_x"] = gdf_train.geometry.x
    df["utm_y"] = gdf_train.geometry.y
    X_spatial = df[["utm_x", "utm_y"]].values
    kmeans_spatial = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["submercado_cluster"] = kmeans_spatial.fit_predict(X_spatial)

    # 4. Construir features estáticas disponibles en el dataset
    disponibles = [f for f in STATIC_FEATURES_AIRBNB if f in df.columns]
    features_final = disponibles + DYNAMIC_FEATURES

    # 5. Entrenamiento de modelos por cluster
    modelos  = {}
    imputers = {}
    scaler_global = StandardScaler()
    # Rellenar dinámicas con 0 para ajustar el scaler sobre todas las filas
    X_global = df[[f for f in disponibles]].fillna(0)
    for f in DYNAMIC_FEATURES:
        X_global[f] = 0.0
    scaler_global.fit(X_global.values)

    for cluster in sorted(df["submercado_cluster"].unique()):
        df_c = df[df["submercado_cluster"] == cluster].copy()
        if len(df_c) < 20:
            df_c = df.copy()
        X_c_df = df_c[[f for f in disponibles]].fillna(0)
        for f in DYNAMIC_FEATURES:
            X_c_df[f] = 0.0
        y_c = np.log1p(df_c["price"].values)
        X_c_scaled = scaler_global.transform(X_c_df.values)
        enet = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1],
            alphas=np.logspace(-4, 1, 30),
            cv=5,
            random_state=42,
            max_iter=3000,
        )
        enet.fit(X_c_scaled, y_c)
        pipeline = Pipeline([("scaler", scaler_global), ("enet", enet)])
        modelos[cluster]  = pipeline
        imputers[cluster] = (df_c[[f for f in disponibles]].median().to_dict(), features_final)

    # 6. Cruce espacial con colonias
    colonias = gpd.read_file(PATH_COLONIAS)
    if colonias.crs != "EPSG:4326":
        colonias = colonias.to_crs("EPSG:4326")
    colonias["colonia_clean"]  = colonias["NOMUT"].apply(clean_text)
    posibles_alcaldias = ["NOMDT", "DEMARCACI", "DEMARCACION", "MUNICIPIO", "NOM_MUN"]
    alcaldia_col = next((c for c in posibles_alcaldias if c in colonias.columns), None)
    if alcaldia_col is None:
        raise ValueError("Falta identificador de alcaldía en el shapefile de colonias.")
    colonias["alcaldia_clean"] = colonias[alcaldia_col].apply(clean_text)

    gdf_geo = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )
    gdf_geo["idx_original"] = gdf_geo.index

    joined = gpd.sjoin(
        gdf_geo,
        colonias[["colonia_clean", "alcaldia_clean", "geometry"]],
        how="left",
        predicate="within"
    )
    joined = joined.sort_values("idx_original").drop_duplicates(subset="idx_original", keep="first")
    joined = joined.set_index("idx_original").reindex(gdf_geo.index)

    df["colonia_real"]  = joined["colonia_clean"].values
    df["alcaldia_real"] = joined["alcaldia_clean"].values
    df["colonia_real"]  = df["colonia_real"].fillna("Desconocida")
    df["alcaldia_real"] = df["alcaldia_real"].fillna("Otra")

    mask_valid = (df["colonia_real"] != "Desconocida") & (df["alcaldia_real"] != "Otra")
    df = df[mask_valid].copy()
    if df.empty:
        st.error("No se encontraron datos dentro de colonias conocidas. Verifica la geolocalización.")
        st.stop()

    df["colonia_key"] = df["alcaldia_real"].astype(str) + "||" + df["colonia_real"].astype(str)
    df = df.reset_index(drop=True)

    # 7. Referencias espaciales por colonia — incluir TODAS las columnas relevantes
    # FIX: asegurarse de incluir gentrif_local_presion y gentrif_map_score
    vars_agrupar = list(dict.fromkeys(
        [f for f in STATIC_FEATURES_AIRBNB + DYNAMIC_FEATURES
         + ["latitude", "longitude",
            "gentrif_local_presion", "gentrif_map_score",
            "dist_area_verde_recreativa_m", "dist_salud_m", "dist_escuela_m",
            "dist_comercio_m", "dist_ciclovia_m"]
         if f in df.columns]
    ))
    referencia_espacial  = df.groupby("colonia_key")[vars_agrupar].mean().to_dict("index")
    price_m2_col         = df.groupby("colonia_key")["price"].median().to_dict()
    price_m2_by_colonia  = df.groupby("colonia_key")["price"].mean().to_dict()
    coords_by_colonia    = df.groupby("colonia_key")[["longitude", "latitude"]].mean().to_dict("index")
    alcaldia_colonias    = (
        df.groupby("alcaldia_real")["colonia_real"]
        .unique()
        .apply(lambda x: sorted(list(x)))
        .to_dict()
    )
    alcaldias_disponibles = sorted(alcaldia_colonias.keys())

    # 8. Polígonos de colonias
    colonias_geo = colonias[["colonia_clean", "alcaldia_clean", "geometry"]].copy()
    colonias_geo = colonias_geo[colonias_geo.geometry.notnull()]
    colonias_geo["colonia_key"] = (
        colonias_geo["alcaldia_clean"].astype(str)
        + "||"
        + colonias_geo["colonia_clean"].astype(str)
    )
    colonias_geo = colonias_geo.dissolve(by="colonia_key").reset_index()
    poligonos_colonias = {
        row["colonia_key"]: row["geometry"].__geo_interface__
        for _, row in colonias_geo.iterrows()
    }

    # 9. Capas de servicios para el mapa y cálculo de distancias
    def _cargar_capa_con_nombres(path, col_nombre, fallback_prefix):
        try:
            gdf = gpd.read_file(path)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            gdf = gdf.to_crs("EPSG:32614")
            gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
            gdf["geometry"] = gdf.geometry.centroid
            nombres = []
            for i, row in enumerate(gdf.itertuples()):
                val = getattr(row, col_nombre, None) if col_nombre else None
                if val and str(val).strip() not in ("", "nan", "None"):
                    nombre = str(val).strip()
                    if len(nombre) > 35:
                        nombre = nombre[:33] + "…"
                else:
                    nombre = f"{fallback_prefix} #{i+1}"
                nombres.append(nombre)
            pts = np.array([(g.x, g.y) for g in gdf.geometry])
            return pts, nombres
        except Exception:
            return np.empty((0, 2)), []

    metro_pts,    metro_nombres    = _cargar_capa_con_nombres(PATH_METRO_EST,          "NOMBRE",     "Estación Metro")
    mb_pts,       mb_nombres       = _cargar_capa_con_nombres(PATH_METROBUS_EST,        "NOMBRE",     "Estación Metrobús")
    tren_pts,     tren_nombres     = _cargar_capa_con_nombres(PATH_TREN_EST,            "NOMBRE",     "Estación Tren Ligero")
    trole_pts,    trole_nombres    = _cargar_capa_con_nombres(PATH_TROLE_PARADAS,       "NOMBRE",     "Parada Trolebús")
    cable_pts,    cable_nombres    = _cargar_capa_con_nombres(PATH_CABLE_EST,           "NOMBRE",     "Estación Cablebús")
    av_pts,       av_nombres       = _cargar_capa_con_nombres(PATH_AREAS_VERDES,        "nombre",     "Área Verde")
    salud1_pts,   salud1_nombres   = _cargar_capa_con_nombres(PATH_SALUD_BASE,          "nombre",     "Unidad de Salud")
    salud2_pts,   salud2_nombres   = _cargar_capa_con_nombres(PATH_HOSPITALES_PUBLICOS, "NOMBRE_D_3", "Hospital")
    com_pts,      com_nombres      = _cargar_capa_con_nombres(PATH_COMERCIO,            "INSTITUCIO", "Centro Comercial")
    esc_priv_pts, esc_priv_nombres = _cargar_capa_con_nombres(PATH_ESCUELAS_PRIVADAS,   "nombre",     "Escuela Privada")
    esc_pub_pts,  esc_pub_nombres  = _cargar_capa_con_nombres(PATH_ESCUELAS_PUBLICAS,   "nombre",     "Escuela Pública")

    if len(salud1_pts) and len(salud2_pts):
        salud_pts,    salud_nombres    = np.vstack([salud1_pts, salud2_pts]), salud1_nombres + salud2_nombres
    elif len(salud1_pts):
        salud_pts,    salud_nombres    = salud1_pts, salud1_nombres
    else:
        salud_pts,    salud_nombres    = salud2_pts, salud2_nombres

    capas_servicios = {
        "metro":       (metro_pts,    metro_nombres),
        "metrobus":    (mb_pts,       mb_nombres),
        "tren":        (tren_pts,     tren_nombres),
        "trolebus":    (trole_pts,    trole_nombres),
        "cablebus":    (cable_pts,    cable_nombres),
        "parques":     (av_pts,       av_nombres),
        "salud":       (salud_pts,    salud_nombres),
        "comercio":    (com_pts,      com_nombres),
        "esc_privada": (esc_priv_pts, esc_priv_nombres),
        "esc_publica": (esc_pub_pts,  esc_pub_nombres),
    }

    # Ciclovías
    try:
        ciclo_wgs = gpd.read_file(PATH_CICLOVIAS)
        if ciclo_wgs.crs is None:
            ciclo_wgs = ciclo_wgs.set_crs("EPSG:4326")
        elif ciclo_wgs.crs.to_epsg() != 4326:
            ciclo_wgs = ciclo_wgs.to_crs("EPSG:4326")
        ciclo_wgs = ciclo_wgs[ciclo_wgs.geometry.notnull() & ~ciclo_wgs.geometry.is_empty].copy()
        ciclo_utm = ciclo_wgs.to_crs("EPSG:32614")
    except Exception:
        ciclo_wgs = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        ciclo_utm = gpd.GeoDataFrame(geometry=[], crs="EPSG:32614")

    capas_lineas = {
        "ciclovias_wgs": ciclo_wgs,
        "ciclovias_utm": ciclo_utm,
    }

    return (
        modelos, imputers, kmeans_spatial, scaler_global,
        referencia_espacial, price_m2_col, price_m2_by_colonia,
        alcaldias_disponibles, alcaldia_colonias, coords_by_colonia,
        capas_servicios, capas_lineas, poligonos_colonias,
    )

# ============================================================
# INSTANCIACIÓN
# ============================================================

_ya_cargado = "sistema_airbnb_listo" in st.session_state

if not _ya_cargado:
    with st.spinner("⏳ Cargando modelo Airbnb — solo ocurre una vez..."):
        (MODELS, IMPUTERS, KMEANS_SPATIAL, SCALER_GLOBAL,
     REF_ESPACIAL, PRICE_M2_COL, PRICE_M2_BY_COLONIA,
     ALCALDIAS_DISPONIBLES, ALCALDIA_COLONIAS, COORDS_BY_COLONIA,
     CAPAS_SERVICIOS, CAPAS_LINEAS, POLIGONOS_COLONIAS) = preparar_sistema_airbnb()
    st.session_state["sistema_airbnb_listo"] = True
else:
    (MODELS, IMPUTERS, KMEANS_SPATIAL, SCALER_GLOBAL,
     REF_ESPACIAL, PRICE_M2_COL, PRICE_M2_BY_COLONIA,
     ALCALDIAS_DISPONIBLES, ALCALDIA_COLONIAS, COORDS_BY_COLONIA,
     CAPAS_SERVICIOS, CAPAS_LINEAS, POLIGONOS_COLONIAS) = preparar_sistema_airbnb()  # Instantáneo: viene de cache_resource
# ============================================================
# RECALCULAR DISTANCIAS DESDE UN PUNTO (igual al original)
# ============================================================
def recalcular_distancias_desde_punto(lat, lon):
    gdf_punto = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326"
    ).to_crs("EPSG:32614")
    px, py = gdf_punto.geometry.iloc[0].x, gdf_punto.geometry.iloc[0].y
    punto_utm = np.array([[px, py]])

    def _dist_min(pts):
        if len(pts) == 0:
            return 5000.0
        tree = cKDTree(pts)
        d, _ = tree.query(punto_utm)
        return float(d[0])

    def _conteo_radio(pts, radio=1200):
        if len(pts) == 0:
            return 0
        tree = cKDTree(pts)
        return len(tree.query_ball_point([px, py], r=radio))

    metro_pts,    _ = CAPAS_SERVICIOS["metro"]
    mb_pts,       _ = CAPAS_SERVICIOS["metrobus"]
    tren_pts,     _ = CAPAS_SERVICIOS["tren"]
    trole_pts,    _ = CAPAS_SERVICIOS["trolebus"]
    cable_pts,    _ = CAPAS_SERVICIOS["cablebus"]
    av_pts,       _ = CAPAS_SERVICIOS["parques"]
    salud_pts,    _ = CAPAS_SERVICIOS["salud"]
    com_pts,      _ = CAPAS_SERVICIOS["comercio"]
    esc_priv_pts, _ = CAPAS_SERVICIOS["esc_privada"]
    esc_pub_pts,  _ = CAPAS_SERVICIOS["esc_publica"]

    esc_todos = (
        np.vstack([esc_priv_pts, esc_pub_pts])
        if len(esc_priv_pts) and len(esc_pub_pts)
        else (esc_priv_pts if len(esc_priv_pts) else esc_pub_pts)
    )

    ciclo_utm = CAPAS_LINEAS.get("ciclovias_utm", gpd.GeoDataFrame(geometry=[], crs="EPSG:32614"))
    ciclo_pts = line_to_points(ciclo_utm, interval_m=100) if not ciclo_utm.empty else np.empty((0, 2))

    d_metro   = _dist_min(metro_pts)
    d_mb      = _dist_min(mb_pts)
    d_tren    = _dist_min(tren_pts)
    d_trole   = _dist_min(trole_pts)
    d_cable   = _dist_min(cable_pts)
    d_verde   = _dist_min(av_pts)
    d_salud   = _dist_min(salud_pts)
    d_escuela = _dist_min(esc_todos)
    d_comercio = _dist_min(com_pts)
    d_ciclovia = _dist_min(ciclo_pts)

    n_salud    = _conteo_radio(salud_pts)
    n_escuela  = _conteo_radio(esc_todos)
    n_parques  = _conteo_radio(av_pts)
    n_ciclovia = _conteo_radio(ciclo_pts)

    prox_parque   = 1 if d_verde    <= 800  else 0
    prox_salud    = 1 if d_salud    <= 1000 else 0
    prox_escuela  = 1 if d_escuela  <= 1000 else 0
    prox_ciclovia = 1 if d_ciclovia <= 500  else 0

    score = float(np.mean([
        min(n_salud   / 6,  1.0),
        min(n_escuela / 40, 1.0),
        min(n_parques / 10, 1.0),
        prox_parque,
        prox_salud,
        prox_escuela,
        prox_ciclovia,
    ]))

    return {
        "dist_metro_m":              d_metro,
        "dist_metrobus_m":           d_mb,
        "dist_tren_m":               d_tren,
        "dist_trole_m":              d_trole,
        "dist_cable_m":              d_cable,
        "dist_area_verde_m":         d_verde,
        "dist_area_verde_recreativa_m": d_verde,
        "dist_salud_m":              d_salud,
        "dist_escuela_m":            d_escuela,
        "dist_comercio_m":           d_comercio,
        "dist_ciclovia_m":           d_ciclovia,
        "acceso_salud_15m":          float(n_salud),
        "acceso_educacion_15m":      float(n_escuela),
        "densidad_parques_15m":      float(n_parques),
        "densidad_ciclovia_15m":     float(n_ciclovia),
        "prox_parque":               float(prox_parque),
        "prox_salud":                float(prox_salud),
        "prox_escuela":              float(prox_escuela),
        "prox_ciclovia":             float(prox_ciclovia),
        "score_15min":               score,
    }

# ============================================================
# PREDICCIÓN
# ============================================================
def predict_price_airbnb(accommodates, bedrooms, bathrooms, beds, room_type, superhost,
                          amenities_dict, antiguedad_host, n_reviews,
                          colonia_key, lat, lon, distancias_externas=None,
                          review_score=4.5, review_cleanliness=4.5, review_location=4.5):
    datos_colonia = REF_ESPACIAL.get(colonia_key, list(REF_ESPACIAL.values())[0])
    median_price  = PRICE_M2_COL.get(colonia_key, 0)
    lux_cut       = np.percentile(list(PRICE_M2_COL.values()), 90) if PRICE_M2_COL else 5000
    seg_label     = "lujo" if median_price >= lux_cut else "estandar"

    input_data = datos_colonia.copy()

    # Actualizar distancias si se proporcionan (punto personalizado)
    if distancias_externas:
        input_data.update(distancias_externas)

    gentrif = input_data.get("gentrification_index", 0.5)
    d_metro_val  = input_data.get("dist_metro_m", 5000)
    density_metro_val = max(0.0, 1.0 - d_metro_val / 1200.0) if d_metro_val < 1200 else 0.0

    input_data.update({
        "accommodates":         accommodates,
        "bedrooms":             bedrooms,
        "bathrooms":            bathrooms,
        "beds":                 beds,
        "host_antiguedad":      np.log1p(antiguedad_host),
        "n_reviews_log":        np.log1p(n_reviews),
        "es_superhost":         1 if superhost else 0,
        "flexibilidad_reserva": 0.5,
        "host_profesional":     0,
        "tasa_ocupacion":       0.5,
        "pct_disponible":       0.5,
        "latitude":             lat,
        "longitude":            lon,
        "latitud":              lat,
        "longitud":             lon,
    })

    # Amenities
    for amen, val in amenities_dict.items():
        col_name = "am_" + amen.lower().replace(" ", "_").replace("/", "_")
        if col_name in STATIC_FEATURES_AIRBNB:
            input_data[col_name] = 1 if val else 0

    # Room type OHE
    input_data["rt_entire"]  = 1 if room_type == "Entire home/apt" else 0
    input_data["rt_private"] = 1 if room_type == "Private room"    else 0
    input_data["rt_hotel"]   = 1 if room_type == "Hotel room"      else 0

    # Scores compuestos por defecto
    for score_col in ["score_cocina", "score_seguridad", "score_confort", "score_premium",
                      "score_familia", "score_trabajo", "score_accesibilidad"]:
        if score_col not in input_data or input_data[score_col] == 0:
            input_data[score_col] = 0.3

    input_data["n_amenidades_log"]        = np.log1p(sum(amenities_dict.values()))
    input_data["ratio_amenidades_premium"] = 0.1

    for col, usr_val in [
        ("review_score_compuesto",    review_score),
        ("review_scores_rating",      review_score),
        ("review_scores_cleanliness", review_cleanliness),
        ("review_scores_location",    review_location),
    ]:
        if pd.isna(usr_val):
            if col not in input_data or pd.isna(input_data.get(col, np.nan)):
                input_data[col] = 4.5
        else:
            input_data[col] = float(usr_val)

    # Interacciones
    input_data["review_x_gentrif"]   = input_data.get("review_score_compuesto", 4.5) * gentrif
    input_data["premium_x_15min"]    = input_data.get("score_premium", 0.3) * input_data.get("score_15min", 0)
    input_data["superhost_x_review"] = input_data["es_superhost"] * input_data.get("review_score_compuesto", 4.5)
    input_data["amenidades_x_pdi"]   = input_data["n_amenidades_log"] / (input_data.get("dist_pdi_log", 1) + 1)
    input_data["density_entire_500m"] = 0
    input_data["profesional_x_flex"]  = 0
    input_data["entire_x_premium"]    = input_data["rt_entire"] * input_data.get("score_premium", 0.3)
    input_data["private_x_clean"]     = input_data["rt_private"] * input_data.get("review_scores_cleanliness", 4.5)
    input_data["hotel_x_confort"]     = input_data["rt_hotel"]   * input_data.get("score_confort", 0.3)
    input_data["entire_x_area_cat"]   = input_data["rt_entire"]  * np.log1p(input_data.get("cat_area_imputable", 60))
    input_data["area_x_marginalidad"] = np.log1p(accommodates) * input_data.get("marginalidad_score", 3)
    input_data["area_X_gentrif"]      = accommodates * gentrif
    input_data["gentrif_x_metro"]     = gentrif * density_metro_val
    input_data["15min_X_gentrif"]     = input_data.get("score_15min", 0) * gentrif

    # Cluster espacial (KMeans entrenado sobre coords UTM)
    gdf_pt  = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326").to_crs("EPSG:32614")
    utm_x   = gdf_pt.geometry.iloc[0].x
    utm_y   = gdf_pt.geometry.iloc[0].y
    cluster_id = KMEANS_SPATIAL.predict([[utm_x, utm_y]])[0]
    if cluster_id not in MODELS:
        cluster_id = list(MODELS.keys())[0]

    model = MODELS[cluster_id]
    imputador_dict, feat_list = IMPUTERS[cluster_id]

    X_df = pd.DataFrame([input_data])
    for f in feat_list:
        if f not in X_df.columns:
            X_df[f] = 0.0

    # Spatial lag real (en metros — mismo enfoque que el original)
    all_col_keys = list(PRICE_M2_BY_COLONIA.keys())
    if len(all_col_keys) > 1:
        col_lons   = np.array([COORDS_BY_COLONIA.get(k, {"longitude": lon})["longitude"] for k in all_col_keys])
        col_lats   = np.array([COORDS_BY_COLONIA.get(k, {"latitude":  lat})["latitude"]  for k in all_col_keys])
        col_prices = np.array([PRICE_M2_BY_COLONIA.get(k, median_price) for k in all_col_keys])
        gdf_cols   = gpd.GeoDataFrame(geometry=gpd.points_from_xy(col_lons, col_lats), crs="EPSG:4326").to_crs("EPSG:32614")
        col_coords = np.array([(g.x, g.y) for g in gdf_cols.geometry])
        tree_col   = cKDTree(col_coords)
        k_lag      = min(10, len(col_coords))
        d_lag, ix_lag = tree_col.query([[utm_x, utm_y]], k=k_lag)
        weights       = 1.0 / (d_lag[0] + 1.0)
        spatial_lag_raw = float(np.average(col_prices[ix_lag[0]], weights=weights))
    else:
        spatial_lag_raw = median_price

    _p_ref = median_price if median_price > 0 else float(np.median(list(PRICE_M2_COL.values())))
    spatial_lag_raw = float(np.clip(spatial_lag_raw, _p_ref * 0.1, _p_ref * 10.0))

    X_df["spatial_lag_price"]    = np.log1p(spatial_lag_raw)
    X_df["precio_vecinal_local"] = spatial_lag_raw
    X_df["lag_x_accommodates"]   = X_df["spatial_lag_price"] * np.log1p(accommodates)

    X_pred = X_df[feat_list].fillna(imputador_dict).fillna(0)

    scaler   = model.named_steps["scaler"]
    X_scaled = np.clip(scaler.transform(X_pred), -5, 5)
    log_pred = float(np.clip(
        model.named_steps["enet"].predict(X_scaled)[0],
        np.log1p(200), np.log1p(50000)
    ))
    precio_noche      = np.expm1(log_pred)
    precio_base_noche = np.expm1(model.named_steps["enet"].intercept_)

    return (precio_noche, precio_base_noche, seg_label, cluster_id,
            model, X_pred, X_scaled, datos_colonia)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    # ── CSS exclusivo del sidebar ──
    st.markdown("""
    <style>
    /* ── Sidebar: ancho mínimo en desktop ── */
    section[data-testid="stSidebar"] > div:first-child {
        min-width: 300px;
    }

    /* ── Etiquetas de sección dentro del sidebar ── */
    .sidebar-section-label {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #FF5A5F;
        background: rgba(255, 90, 95, 0.10);
        border: 1px solid rgba(255, 90, 95, 0.25);
        border-radius: 20px;
        padding: 0.28rem 0.75rem;
        margin: 1.1rem 0 0.6rem 0;
    }

    /* ── Divider suave ── */
    .sidebar-divider {
        height: 1px;
        background: var(--border-color);
        margin: 1rem 0;
        opacity: 0.5;
    }

    /* ── Widget de estrella-rating ── */
    .star-rating-widget {
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 0.75rem 0.9rem 0.6rem;
        margin-bottom: 0.5rem;
    }
    .star-rating-widget .star-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 0.4rem;
        display: block;
    }
    .star-rating-widget .star-row {
        display: flex;
        align-items: center;
        gap: 4px;
        flex-wrap: nowrap;
    }
    .star-rating-widget .star-btn {
        font-size: 1.6rem;
        background: none;
        border: none;
        padding: 0;
        cursor: pointer;
        line-height: 1;
        transition: transform 0.1s ease;
        flex-shrink: 0;
    }
    .star-rating-widget .star-btn:hover { transform: scale(1.15); }
    .star-rating-widget .star-value {
        font-size: 0.85rem;
        font-weight: 700;
        color: var(--text-color);
        margin-left: 6px;
        opacity: 0.9;
    }

    /* ── Inputs del sidebar: mejor contraste ── */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
        background: var(--background-color) !important;
        border: 1.5px solid var(--border-color) !important;
        border-radius: 10px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
        border-radius: 10px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stCheckbox"] label {
        font-size: 0.9rem !important;
    }

    /* ── Checkbox pill ── */
    .am-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── Header del sidebar ──
    st.markdown("""
    <div style="text-align:center; padding:1.4rem 0 1rem; border-bottom:2px solid var(--border-color);
                margin-bottom:0.5rem;">
        <div style="font-size:2.8rem; margin-bottom:0.4rem; line-height:1;">🏠</div>
        <h2 style="margin:0; color:var(--text-color); font-size:1.2rem; font-weight:900;
                   letter-spacing:-0.02em;">
            Parámetros del Airbnb
        </h2>
        <p style="color:var(--text-color); opacity:0.6; font-size:0.8rem; margin-top:0.4rem;">
            Configura tu listing paso a paso
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── SECCIÓN 1: Ubicación ──
    st.markdown('<div class="sidebar-section-label">📍 Ubicación</div>', unsafe_allow_html=True)
    alcaldia_sel       = st.selectbox("Alcaldía", ALCALDIAS_DISPONIBLES, label_visibility="collapsed")
    colonias_filtradas = ALCALDIA_COLONIAS.get(alcaldia_sel, [])
    colonia_sel = st.selectbox(
        "Colonia", colonias_filtradas,
        key=f"colonia_{alcaldia_sel}",
        label_visibility="collapsed"
    )
    colonia_key_sel = f"{alcaldia_sel}||{colonia_sel}"
    st.caption(f"📌 {alcaldia_sel.title()} · {colonia_sel.title()}")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── SECCIÓN 2: Capacidad ──
    st.markdown('<div class="sidebar-section-label">🛏️ Capacidad y habitaciones</div>', unsafe_allow_html=True)
    accommodates    = st.number_input("👥 Huéspedes", min_value=1, max_value=20, value=2)
    bedrooms        = st.number_input("🛏️ Recámaras", min_value=0, max_value=10, value=1)
    bathrooms       = st.number_input("🚿 Baños", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
    beds            = st.number_input("🛌 Camas", min_value=0, max_value=20, value=1)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── SECCIÓN 3: Tipo y host ──
    st.markdown('<div class="sidebar-section-label">🏷️ Tipo y anfitrión</div>', unsafe_allow_html=True)
    room_type = st.selectbox(
        "Tipo de espacio",
        ["Entire home/apt", "Private room", "Shared room", "Hotel room"],
        format_func=lambda x: {
            "Entire home/apt": "🏠 Departamento/Casa completa",
            "Private room":    "🚪 Habitación privada",
            "Shared room":     "🛏️ Habitación compartida",
            "Hotel room":      "🏨 Habitación de hotel",
        }[x]
    )
    c_sh, c_ib = st.columns(2)
    with c_sh:
        superhost    = st.checkbox("⭐ Superhost")
    with c_ib:
        instant_book = st.checkbox("⚡ Inst. Book")

    antiguedad_host = st.slider("🕐 Antigüedad del anfitrión (años)", 0, 20, 2)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── SECCIÓN 4: Valoración con estrellas ──
    st.markdown('<div class="sidebar-section-label">⭐ Valoración del listing</div>', unsafe_allow_html=True)

    # Rating general con estrellas interactivas (via selectbox estilizado)
    _star_opts = {
        "★★★★★  5.0 — Excepcional":    5.0,
        "★★★★☆  4.5 — Muy bueno":      4.5,
        "★★★★☆  4.0 — Bueno":          4.0,
        "★★★☆☆  3.5 — Aceptable":      3.5,
        "★★★☆☆  3.0 — Promedio":       3.0,
        "★★☆☆☆  2.5 — Regular":        2.5,
        "★★☆☆☆  2.0 — Malo":           2.0,
        "★☆☆☆☆  1.0 — Muy malo":       1.0,
    }
    star_sel = st.selectbox(
        "⭐ Calificación general",
        list(_star_opts.keys()),
        index=1,
    )
    review_score_val = _star_opts[star_sel]

    # Sub-puntuaciones por dimensión
    st.markdown("""
    <div style="font-size:0.75rem; font-weight:700; color:var(--text-color); opacity:0.6;
                text-transform:uppercase; letter-spacing:0.06em; margin:0.6rem 0 0.3rem;">
        Puntuaciones por categoría
    </div>
    """, unsafe_allow_html=True)

    _clean_opts  = ["☆☆☆☆☆ —", "★☆☆☆☆ 1", "★★☆☆☆ 2", "★★★☆☆ 3", "★★★★☆ 4", "★★★★★ 5"]
    _clean_vals  = [np.nan, 1.0, 2.0, 3.0, 4.0, 5.0]

    c_clean, c_loc = st.columns(2)
    with c_clean:
        clean_sel  = st.selectbox("🧹 Limpieza", _clean_opts, index=5)
    with c_loc:
        loc_sel    = st.selectbox("📍 Ubicación", _clean_opts, index=4)

    clean_val = _clean_vals[_clean_opts.index(clean_sel)]
    loc_val   = _clean_vals[_clean_opts.index(loc_sel)]

    n_reviews = st.number_input("💬 Número de reseñas", min_value=0, max_value=500, value=10)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── SECCIÓN 5: Amenidades ──
    st.markdown('<div class="sidebar-section-label">✨ Amenidades</div>', unsafe_allow_html=True)

    c_am1, c_am2 = st.columns(2)
    with c_am1:
        am_wifi        = st.checkbox("📶 WiFi",          value=True)
        am_ac          = st.checkbox("❄️ Aire acond.")
        am_pool        = st.checkbox("🏊 Alberca")
        am_washer      = st.checkbox("🫧 Lavadora")
    with c_am2:
        am_parking     = st.checkbox("🚗 Parking gratis")
        am_kitchen     = st.checkbox("🍳 Cocina")
        am_self_checkin = st.checkbox("🔑 Auto check-in")
        am_workspace   = st.checkbox("💻 Workspace")

    amenities_dict = {
        "Wifi":                      am_wifi,
        "Air conditioning":          am_ac,
        "Pool":                      am_pool,
        "Free parking on premises":  am_parking,
        "Kitchen":                   am_kitchen,
        "Washer":                    am_washer,
        "Self check-in":             am_self_checkin,
        "Dedicated workspace":       am_workspace,
    }

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.caption("📍 Usa el marcador (✏️) en el mapa para elegir un punto específico y confirma con el botón.")

# ============================================================
# ESTADO DE SESIÓN Y PREDICCIÓN
# ============================================================
coords_base = COORDS_BY_COLONIA.get(colonia_key_sel, {"latitude": 19.43, "longitude": -99.13})
lat_base    = float(coords_base["latitude"])
lon_base    = float(coords_base["longitude"])

_key_lat        = f"marker_lat_{colonia_key_sel}"
_key_lon        = f"marker_lon_{colonia_key_sel}"
_key_confirmado = f"punto_confirmado_{colonia_key_sel}"
_key_dist       = f"distancias_punto_{colonia_key_sel}"

for _k, _v in [(_key_lat, lat_base), (_key_lon, lon_base),
                (_key_confirmado, False), (_key_dist, None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

lat_marcador           = st.session_state[_key_lat]
lon_marcador           = st.session_state[_key_lon]
punto_confirmado       = st.session_state[_key_confirmado]
distancias_confirmadas = st.session_state[_key_dist]

# Predicción base (centroide)
(precio_noche, precio_base_noche, seg_label, cluster_id,
 model, X_pred, X_scaled_arr, datos_colonia) = predict_price_airbnb(
    accommodates, bedrooms, bathrooms, beds, room_type, superhost,
    amenities_dict, antiguedad_host, n_reviews, colonia_key_sel,
    lat_base, lon_base,
    review_score=review_score_val,
    review_cleanliness=clean_val,
    review_location=loc_val,
)

# Si el marcador se movió, recalcular con distancias reales al punto
_punto_movido = (
    abs(lat_marcador - lat_base) > 0.000001
    or abs(lon_marcador - lon_base) > 0.000001
)
if _punto_movido or punto_confirmado:
    _dist_nuevas = recalcular_distancias_desde_punto(lat_marcador, lon_marcador)
    (precio_noche, precio_base_noche, seg_label, cluster_id,
     model, X_pred, X_scaled_arr, datos_colonia) = predict_price_airbnb(
        accommodates, bedrooms, bathrooms, beds, room_type, superhost,
        amenities_dict, antiguedad_host, n_reviews, colonia_key_sel,
        lat_marcador, lon_marcador, distancias_externas=_dist_nuevas,
        review_score=review_score_val,
        review_cleanliness=clean_val,
        review_location=loc_val,
    )
    distancias_confirmadas = _dist_nuevas
    st.session_state[_key_dist] = distancias_confirmadas

datos_entorno = datos_colonia.copy()
if distancias_confirmadas:
    datos_entorno.update(distancias_confirmadas)

# ============================================================
# PESTAÑAS
# ============================================================
tab_valuador, tab_atlas = st.tabs([
    "🏠 Valuador Airbnb",
    "🗺️ Estudio Multidimensional del Mercado Airbnb"
])

with tab_valuador:

    # Métricas principales — HTML custom idéntico al original
    precio_fmt  = f"${precio_noche:,.0f} MXN/noche"
    base_fmt    = f"${precio_base_noche:,.0f} MXN"
    seg_fmt     = f"{'Lujo' if seg_label == 'lujo' else 'Estándar'} · C{cluster_id}"
    gentrif_fmt = str(round(datos_colonia.get("gentrification_index", 0), 3))

    st.markdown(f"""
    <style>
    .metricas-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }}
    @media screen and (max-width: 900px) {{
        .metricas-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    @media screen and (max-width: 500px) {{
        .metricas-grid {{ grid-template-columns: 1fr; }}
    }}
    .metrica-card {{
        background: var(--secondary-background-color);
        border: 1.5px solid var(--border-color);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        min-width: 0;
    }}
    .metrica-label {{
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--text-color);
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }}
    .metrica-valor {{
        font-size: clamp(1.1rem, 2.5vw, 1.55rem);
        font-weight: 800;
        color: var(--text-color);
        letter-spacing: -0.02em;
        line-height: 1.2;
        word-break: break-word;
        overflow-wrap: break-word;
    }}
    </style>
    <div class="metricas-grid">
        <div class="metrica-card" style="border-top:3px solid #3B82F6;">
            <div class="metrica-label">Precio Estimado por Noche</div>
            <div class="metrica-valor">{precio_fmt}</div>
        </div>
        <div class="metrica-card" style="border-top:3px solid #10B981;">
            <div class="metrica-label">Precio Base del Submercado</div>
            <div class="metrica-valor">{base_fmt}</div>
        </div>
        <div class="metrica-card" style="border-top:3px solid #8B5CF6;">
            <div class="metrica-label">Segmento / Clúster</div>
            <div class="metrica-valor">{seg_fmt}</div>
        </div>
        <div class="metrica-card" style="border-top:3px solid #F59E0B;">
            <div class="metrica-label">Índice de Gentrificación</div>
            <div class="metrica-valor">{gentrif_fmt}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ============================================================
    # PANEL DE ANÁLISIS DE ENTORNO URBANO (5 bloques)
    # ============================================================
    with st.expander("🌎 Análisis de Entorno Urbano y Socioespacial", expanded=True):

        # --- Bloque 1: Ciudad de 15 Minutos ---
        encabezado_seccion("1. Ciudad de 15 Minutos",
                           "Accesibilidad a servicios esenciales a escala peatonal", "🏬")

        with st.container(border=True):
            score_15 = float(datos_entorno.get("score_15min", 0))
            c_score, c_prog = st.columns([1, 3])
            with c_score:
                st.metric("🎯 Índice General 15 Min", round(score_15, 3))
            with c_prog:
                st.markdown("<div style='padding-top:10px;'></div>", unsafe_allow_html=True)
                st.progress(min(score_15, 1.0))
                if score_15 >= 0.7:   st.success("Alta cobertura peatonal.")
                elif score_15 >= 0.4: st.info("Cobertura urbana funcional.")
                else:                  st.warning("Dependencia de vehículo.")

            st.markdown("---")
            fila1_col1, fila1_col2 = st.columns(2)
            fila2_col1, fila2_col2 = st.columns(2)

            with fila1_col1:
                acceso_salud = float(datos_entorno.get("acceso_salud_15m", 0))
                st.metric("🏥 Servicios de Salud Cercanos", f"{acceso_salud:.0f} Unidades")
                if acceso_salud >= 6:   st.success("Alta densidad médica.")
                elif acceso_salud >= 2: st.info("Cobertura hospitalaria básica.")
                else:                   st.warning("Déficit de equipamiento médico.")
                st.caption("Estructura unificada y depurada.")

            with fila1_col2:
                acceso_edu = float(datos_entorno.get("acceso_educacion_15m", 0))
                st.metric("📚 Planteles Educativos", f"{acceso_edu:.0f} Escuelas")
                if acceso_edu >= 40:   st.success("Alta oferta escolar.")
                elif acceso_edu >= 15: st.info("Infraestructura escolar suficiente.")
                else:                  st.warning("Disponibilidad local limitada.")
                st.caption("Búfer operativo de 1.2 km.")

            with fila2_col1:
                parque      = float(datos_entorno.get("dist_area_verde_recreativa_m",
                                    datos_entorno.get("dist_area_verde_m", 9999)))
                dens_parques = float(datos_entorno.get("densidad_parques_15m", 0))
                st.metric("🌳 Espacio Público Recreativo", f"{parque:.0f} m")
                if parque <= 300:   st.success("Radio óptimo de proximidad.")
                elif parque <= 800: st.info("Distancia media de acceso.")
                else:               st.warning("Déficit de áreas verdes.")
                st.caption(f"Aprox. {dens_parques:.0f} espacios verdes detectados.")

            with fila2_col2:
                comercio_dist = float(datos_entorno.get("dist_comercio_m", 5000))
                st.metric("🛍️ Centros de Abasto / Comercio", fmt_dist(comercio_dist))
                if comercio_dist <= 600:    st.success("Abasto local inmediato.")
                elif comercio_dist <= 1500: st.info("Proximidad comercial aceptable.")
                else:                       st.warning("Distancia prolongada a zonas comerciales.")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Bloque 2: Transporte ---
        encabezado_seccion("2. Conectividad y Transporte",
                           "Red de transporte público estructurado", "🚊")

        with st.container(border=True):
            c2_izquierda, c2_derecha = st.columns([3, 1.2])

            d_metro  = float(datos_entorno.get("dist_metro_m",    9999))
            d_mb     = float(datos_entorno.get("dist_metrobus_m", 9999))
            d_tren   = float(datos_entorno.get("dist_tren_m",     9999))
            d_trole  = float(datos_entorno.get("dist_trole_m",    9999))
            d_cable  = float(datos_entorno.get("dist_cable_m",    9999))
            ciclovias = float(datos_entorno.get("densidad_ciclovia_15m", 0))

            with c2_izquierda:
                sub_c1, sub_c2, sub_c3 = st.columns(3)
                with sub_c1:
                    st.metric("🚇 STC Metro",   fmt_dist(d_metro))
                    st.markdown("<div style='padding-top:15px;'></div>", unsafe_allow_html=True)
                    st.metric("🚌 Metrobús",    fmt_dist(d_mb))
                with sub_c2:
                    st.metric("🚊 Tren Ligero", fmt_dist(d_tren))
                    st.markdown("<div style='padding-top:15px;'></div>", unsafe_allow_html=True)
                    st.metric("🚎 Trolebús",    fmt_dist(d_trole))
                with sub_c3:
                    st.metric("🚠 Cablebús",    fmt_dist(d_cable))
                    st.markdown("<div style='padding-top:15px;'></div>", unsafe_allow_html=True)
                    if ciclovias >= 5.0:    status_bici = "Excelente"
                    elif ciclovias >= 1.5:  status_bici = "Funcional"
                    elif ciclovias > 0:     status_bici = "Escasa"
                    else:                   status_bici = "Ninguna"
                    st.metric("🚲 Infraestructura Ciclista", status_bici,
                              help=f"{ciclovias:.1f} segmentos en radio 15 min.")

            with c2_derecha:
                min_dist = min([d_metro, d_mb, d_tren, d_trole])
                st.markdown("<b style='font-size:14px;color:var(--text-muted);'>Evaluación Multimodal</b>",
                            unsafe_allow_html=True)
                if min_dist <= 500:    st.success("🟢 **Conectividad Excelente**")
                elif min_dist <= 1000: st.info("🔵 **Accesibilidad Media**")
                else:                  st.warning("🟡 **Cobertura Restringida**")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Bloque 3: Gentrificación ---
        encabezado_seccion("3. Transformación Socioespacial",
                           "Análisis de gentrificación y presión inmobiliaria", "🏙️")

        with st.container(border=True):
            c3_1, c3_2, c3_3 = st.columns(3)

            # FIX: leer directamente de datos_colonia (referencia espacial con todas las columnas)
            gentrif_macro = float(datos_colonia.get("gentrification_index",  0))
            gentrif_micro = float(datos_colonia.get("gentrif_local_presion", 0))
            gentrif_map   = float(datos_colonia.get("gentrif_map_score",     0))

            with c3_1:
                st.metric("🏛️ Cambio Estructural (Alcaldía)", round(gentrif_macro, 3))
                st.progress(min(gentrif_macro, 1.0))
                st.caption("Deltas socioeconómicos intercensales (INEGI 2010–2020)")

            with c3_2:
                st.metric("🏘️ Presión de Mercado Local", round(gentrif_micro, 3))
                st.progress(min(gentrif_micro, 1.0))
                st.caption("Variaciones de valor respecto al entorno continuo")

            with c3_3:
                st.metric("🌆 Vulnerabilidad al Cambio Socioespacial", round(gentrif_map, 3))
                st.progress(min(gentrif_map, 1.0))
                if gentrif_map >= 0.7:    st.warning("Proceso acelerado de transformación.")
                elif gentrif_map >= 0.4:  st.info("Área en transición urbana activa.")
                else:                      st.success("Estabilidad sociodemográfica relativa.")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Bloque 4: Variables Catastrales ---
        encabezado_seccion("4. Contexto Catastral",
                           "Variables del entorno inmediato (radio 200m)", "🗂️")

        with st.container(border=True):
            cr1_col1, cr1_col2 = st.columns(2)

            with cr1_col1:
                vus = datos_colonia.get("cat_mean_vus_200m", None)
                if vus is not None:
                    st.metric("📐 Precio oficial del suelo (m²)", f"${vus:,.0f} MXN/m²",
                              help="Valor unitario de suelo según el Catastro CDMX 2021.")

            with cr1_col2:
                vs = datos_colonia.get("cat_mean_valor_suelo_200m", None)
                if vs is not None:
                    st.metric("🏦 Valor catastral del terreno", f"${vs:,.0f} MXN",
                              help="Valor total de suelo registrado por predio en el catastro.")

            st.markdown("---")
            cr2_col1, cr2_col2 = st.columns(2)

            with cr2_col1:
                ratio = datos_colonia.get("cat_mean_ratio_construccion_200m", None)
                if ratio is not None:
                    if ratio >= 3.0:
                        ratio_label, ratio_desc, ratio_color = "Zona muy edificada", "Predominan edificios de varios pisos.", "success"
                    elif ratio >= 1.5:
                        ratio_label, ratio_desc, ratio_color = "Zona medianamente edificada", "Mezcla de casas y edificios de baja altura.", "info"
                    else:
                        ratio_label, ratio_desc, ratio_color = "Zona poco edificada", "Predominan casas o predios con poca construcción.", "warning"
                    st.metric("🏗️ ¿Qué tan construida está la zona?", f"{ratio:.1f}× el terreno",
                              help="Metros cuadrados de construcción por m² de terreno en predios cercanos.")
                    if ratio_color == "success":   st.success(f"🏢 {ratio_label} — {ratio_desc}")
                    elif ratio_color == "info":    st.info(f"🏠 {ratio_label} — {ratio_desc}")
                    else:                          st.warning(f"🌿 {ratio_label} — {ratio_desc}")

            with cr2_col2:
                dens = datos_colonia.get("cat_density_predios_200m", None)
                if dens is not None:
                    if dens >= 80:
                        dens_label, dens_desc, dens_color = "Zona muy fraccionada", "Muchos predios pequeños, alta densidad urbana.", "success"
                    elif dens >= 30:
                        dens_label, dens_desc, dens_color = "Zona consolidada", "Tejido urbano típico de colonia establecida.", "info"
                    else:
                        dens_label, dens_desc, dens_color = "Zona poco parcelada", "Predios grandes o área en proceso de urbanización.", "warning"
                    st.metric("🏘️ ¿Cuántos predios hay cerca?", f"{dens:.0f} predios en 200 m",
                              help="Número de predios catastrados dentro de un radio de 200 metros.")
                    if dens_color == "success":    st.success(f"🏙️ {dens_label} — {dens_desc}")
                    elif dens_color == "info":     st.info(f"📌 {dens_label} — {dens_desc}")
                    else:                          st.warning(f"🌱 {dens_label} — {dens_desc}")

            st.markdown("---")
            cr3_col1, cr3_col2 = st.columns(2)

            with cr3_col1:
                ant_cat = datos_colonia.get("cat_mean_antiguedad_200m", None)
                if ant_cat is not None:
                    st.metric("📅 Antigüedad promedio de las construcciones", f"{ant_cat:.0f} años",
                              help="Edad promedio de los inmuebles cercanos según año de construcción catastral.")
                    if ant_cat <= 15:    st.success("Zona con construcciones recientes.")
                    elif ant_cat <= 40:  st.info("Zona con construcciones de edad media.")
                    else:                st.warning("Zona con construcciones antiguas.")

            with cr3_col2:
                st.metric("📊 Precio base del submercado", f"${precio_base_noche:,.0f} MXN",
                          help="Precio estimado para un listing completamente promedio dentro de este clúster.")
                delta_vs_base = precio_noche - precio_base_noche
                if delta_vs_base >= 0:
                    st.success(f"▲ +${delta_vs_base:,.0f} MXN sobre el precio base")
                else:
                    st.warning(f"▼ ${delta_vs_base:,.0f} MXN bajo el precio base")

            st.caption("Fuente: Catastro CDMX 2021 · Radio de análisis: 200 m.")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Bloque 5: Estructura urbana ---
        encabezado_seccion("5. Estructura Urbana y Desempeño del Mercado",
                           "Indicadores complementarios de centralidad y rezago", "📊")

        with st.container(border=True):
            c4_1, c4_2, c4_3 = st.columns(3)

            with c4_1:
                marginalidad = int(datos_colonia.get("marginalidad_score", 3))
                st.metric("📉 Índice de Rezago Social", f"{marginalidad} / 5")
                if marginalidad <= 2:   st.success("Estrato medio-alto / alto.")
                elif marginalidad <= 3: st.info("Estrato medio.")
                else:                   st.error("Condiciones de vulnerabilidad social.")
                st.caption("Marco metodológico CONAPO.")

            with c4_2:
                listing = float(datos_colonia.get("listing_density_log", 0))
                st.metric("🏢 Densidad Comercial Inmobiliaria", f"{listing:.2f} (Log)")
                if listing >= 3:     st.success("Mercado con alta rotación.")
                elif listing >= 1.5: st.info("Actividad de mercado regular.")
                else:                st.warning("Baja tasa de transacciones.")
                st.caption("Densidad logarítmica de listados activos.")

            with c4_3:
                centralidad_log = float(datos_colonia.get("dist_subcenter_log", 0))
                centralidad_m   = np.expm1(centralidad_log)
                st.metric("📍 Proximidad a Nodos de Empleo", fmt_dist(centralidad_m))
                if centralidad_m <= 1000:   st.success("Alta centralidad económica.")
                elif centralidad_m <= 3000: st.info("Distancia funcional a subcentros.")
                else:                       st.warning("Ubicación periférica o habitacional.")
                st.caption("Subcentros: Centro, Polanco, Santa Fe, Insurgentes, Del Valle, Reforma.")

    # ============================================================
    # GRÁFICO DE IMPACTO — mismo enfoque exacto que el original
    # ============================================================
    st.markdown("### 💰 Elasticidad y Aportación de Atributos al Precio")

    NOMBRES_VARIABLES_AIRBNB = {
        "accommodates":             "Número de huéspedes",
        "bedrooms":                 "Recámaras",
        "bathrooms":                "Baños",
        "beds":                     "Camas",
        "rt_entire":                "Alquiler completo",
        "rt_private":               "Habitación privada",
        "rt_hotel":                 "Habitación de hotel",
        "es_superhost":             "Superhost",
        "host_antiguedad":          "Antigüedad del anfitrión (log)",
        "n_reviews_log":            "Número de reseñas (log)",
        "flexibilidad_reserva":     "Flexibilidad de reserva",
        "score_premium":            "Índice de amenities premium",
        "score_confort":            "Confort",
        "score_cocina":             "Cocina equipada",
        "score_seguridad":          "Seguridad",
        "score_trabajo":            "Espacio de trabajo",
        "score_familia":            "Apto para familias",
        "score_accesibilidad":      "Accesibilidad",
        "n_amenidades_log":         "Total de amenidades (log)",
        "ratio_amenidades_premium": "Ratio amenidades premium",
        "review_score_compuesto":   "Puntuación de reseñas",
        "review_x_gentrif":        "Reseñas × Gentrificación",
        "premium_x_15min":          "Premium × Ciudad 15 Min",
        "superhost_x_review":       "Superhost × Reseñas",
        "am_wifi":                  "WiFi disponible",
        "am_air_conditioning":      "Aire acondicionado",
        "am_pool":                  "Alberca",
        "am_washer":                "Lavadora",
        "am_free_parking_on_premises": "Estacionamiento gratuito",
        "am_dedicated_workspace":   "Espacio de trabajo dedicado",
        "score_15min":              "Puntaje Ciudad 15 Minutos",
        "gentrification_index":     "Índice de gentrificación",
        "marginalidad_score":       "Rezago social",
        "listing_density_log":      "Densidad de listings",
        "dist_metro_m":             "Distancia a metro",
        "dist_metrobus_m":          "Distancia a metrobús",
        "dist_ciclovia_m":          "Distancia a ciclovía",
        "dist_area_verde_m":        "Distancia a área verde",
        "acceso_salud_15m":         "Cobertura de salud",
        "acceso_educacion_15m":     "Cobertura educativa",
        "spatial_lag_price":        "Lag espacial de precios",
        "precio_vecinal_local":     "Precio vecinal local",
        "lag_x_accommodates":       "Lag × huéspedes",
        "area_X_gentrif":           "Capacidad × Gentrificación",
        "gentrif_x_metro":          "Gentrificación × Metro",
        "15min_X_gentrif":          "Ciudad 15 Min × Gentrificación",
        "cat_mean_vus_200m":                "Valor unitario suelo catastral",
        "cat_mean_valor_suelo_200m":        "Valor catastral suelo promedio",
        "cat_mean_antiguedad_200m":         "Antigüedad catastral promedio",
        "cat_mean_ratio_construccion_200m": "Ratio construcción/terreno catastral",
        "cat_density_predios_200m":         "Densidad predios catastro 200 m",
    }

    enet_model   = model.named_steps["enet"]
    coeficientes = enet_model.coef_
    feat_names   = X_pred.columns.tolist()

    # Mismo cálculo de impacto que el original (precio * (exp(beta * val/ref) - 1))
    impactos_pesos = []
    for idx, var in enumerate(feat_names):
        val  = float(X_pred[var].values[0]) if var in X_pred.columns else 0.0
        beta = coeficientes[idx] if idx < len(coeficientes) else 0.0
        ref  = X_pred[var].values[0] if X_pred[var].values[0] != 0 else 1
        impacto = precio_noche * (np.exp(beta * (val / ref)) - 1)
        # Clamping igual que el original
        if var in ["accommodates", "area_X_gentrif"] and abs(impacto) > precio_noche:
            impacto = np.sign(impacto) * (precio_noche * 0.4)
        impactos_pesos.append(impacto)

    datos_impacto = pd.DataFrame({
        "Variable_Interna": feat_names,
        "Impacto_Pesos":    impactos_pesos
    })
    datos_impacto["Característica"] = (
        datos_impacto["Variable_Interna"]
        .map(NOMBRES_VARIABLES_AIRBNB)
        .fillna(datos_impacto["Variable_Interna"])
    )

    UMBRAL_IMPACTO = 25
    datos_filtrados = datos_impacto[
        datos_impacto["Impacto_Pesos"].abs() >= UMBRAL_IMPACTO
    ].copy().sort_values("Impacto_Pesos", ascending=True)

    if not datos_filtrados.empty:
        colores = ["#EF553B" if v < 0 else "#00CC96" for v in datos_filtrados["Impacto_Pesos"]]
        fig_impacto = go.Figure()
        fig_impacto.add_trace(go.Bar(
            y=datos_filtrados["Característica"],
            x=datos_filtrados["Impacto_Pesos"],
            orientation="h",
            marker_color=colores,
            text=datos_filtrados["Impacto_Pesos"].apply(
                lambda x: f"${x:,.0f} MXN" if x >= 0 else f"-${abs(x):,.0f} MXN"
            ),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Aportación: %{x:$,.2f} MXN<extra></extra>"
        ))
        fig_impacto.update_layout(
            title=(
                "<b>Factores Dominantes en la Formación del Precio</b>"
                "<br><span style='font-size:12px;color:gray;'>"
                f"Impacto absoluto &gt; ${UMBRAL_IMPACTO:,} MXN</span>"
            ),
            xaxis_title="Impacto Neto sobre el Valor Estimado ($ MXN)",
            yaxis_title="",
            margin=dict(l=25, r=80, t=70, b=25),
            height=len(datos_filtrados) * 38 + 110,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(128,128,128,0.15)",
                zeroline=True,
                zerolinecolor="rgba(128,128,128,0.4)"
            )
        )
        st.plotly_chart(fig_impacto, use_container_width=True)
        st.info(
            "💡 **Guía de Lectura:** Barras **verdes** → atributos que incrementan "
            "la plusvalía. Barras **rojas** → penalizaciones por obsolescencia o "
            "déficit de conectividad."
        )
    else:
        st.warning(
            f"No se identificaron variables con impacto superior a ${UMBRAL_IMPACTO:,} MXN "
            "en esta configuración de Airbnb."
        )

    # ============================================================
    # MAPA DE CONTEXTO (idéntico al original)
    # ============================================================
    RADIO_M = 1200

    CAPAS_CONFIG = {
        "metro":       {"label": "🚇 STC Metro",      "color": "blue",      "icono": "train",          "prefix": "fa", "max_puntos": 30},
        "metrobus":    {"label": "🚌 Metrobús",        "color": "red",       "icono": "bus",            "prefix": "fa", "max_puntos": 30},
        "tren":        {"label": "🚊 Tren Ligero",     "color": "cadetblue", "icono": "subway",         "prefix": "fa", "max_puntos": 20},
        "trolebus":    {"label": "🚎 Trolebús",        "color": "purple",    "icono": "bolt",           "prefix": "fa", "max_puntos": 30},
        "cablebus":    {"label": "🚠 Cablebús",        "color": "darkblue",  "icono": "cloud",          "prefix": "fa", "max_puntos": 20},
        "parques":     {"label": "🌳 Áreas Verdes",    "color": "green",     "icono": "leaf",           "prefix": "fa", "max_puntos": 25},
        "salud":       {"label": "🏥 Salud",           "color": "darkred",   "icono": "plus-square",    "prefix": "fa", "max_puntos": 20},
        "comercio":    {"label": "🛍️ Comercio",       "color": "orange",    "icono": "shopping-cart",  "prefix": "fa", "max_puntos": 15},
        "esc_privada": {"label": "🏫 Esc. Privadas",  "color": "beige",     "icono": "graduation-cap", "prefix": "fa", "max_puntos": 20},
        "esc_publica": {"label": "🏫 Esc. Públicas",  "color": "darkgreen", "icono": "graduation-cap", "prefix": "fa", "max_puntos": 20},
        "ciclovias":   {"label": "🚲 Ciclovías",       "color": "teal",      "icono": "road",           "prefix": "fa", "max_puntos": 20},
    }

    _COLOR_HEX = {
        "blue":      "#1A73E8", "red":       "#E53935", "cadetblue": "#5F9EA0",
        "purple":    "#8E24AA", "darkblue":  "#1565C0", "green":     "#43A047",
        "darkred":   "#C62828", "orange":    "#FB8C00", "beige":     "#8D6E63",
        "darkgreen": "#2E7D32", "teal":      "#00796B",
    }

    def _latlon_a_utm_mapa(lat, lon):
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326").to_crs("EPSG:32614")
        return gdf.geometry.iloc[0].x, gdf.geometry.iloc[0].y

    def _utm_a_latlon_mapa(pts_utm):
        if len(pts_utm) == 0:
            return []
        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(pts_utm[:, 0], pts_utm[:, 1]),
            crs="EPSG:32614"
        ).to_crs("EPSG:4326")
        return [(g.y, g.x) for g in gdf.geometry]

    poligono_colonia_geojson = POLIGONOS_COLONIAS.get(colonia_key_sel)

    # FIX DEL MAPA: la key incluye colonia_key_sel completo (alcaldía + colonia)
    # para que se regenere al cambiar de ubicación
    m = folium.Map(
        location=[lat_marcador, lon_marcador],
        zoom_start=15,
        tiles="OpenStreetMap",
    )

    # Polígono de colonia
    if poligono_colonia_geojson:
        grupo_colonia = folium.FeatureGroup(name="🔵 Límite de colonia", show=True)
        folium.GeoJson(
            poligono_colonia_geojson,
            style_function=lambda _: {
                "color": "#1A73E8", "weight": 2,
                "fillColor": "#1A73E8", "fillOpacity": 0.06,
                "dashArray": "6 4",
            },
            tooltip=f"Límite de colonia: {colonia_sel.title()}",
        ).add_to(grupo_colonia)
        grupo_colonia.add_to(m)

    # Círculo de radio
    folium.Circle(
        location=[lat_marcador, lon_marcador],
        radius=RADIO_M,
        color="#E53935", weight=2,
        fill=True, fill_color="#E53935", fill_opacity=0.05,
        tooltip=f"Radio de análisis: {RADIO_M/1000:.1f} km | {'Confirmado' if punto_confirmado else 'Centroide'}",
    ).add_to(m)

    # Marcador de la vivienda
    folium.Marker(
        location=[lat_marcador, lon_marcador],
        icon=folium.DivIcon(
            html='<div style="font-size:36px;line-height:1;text-align:center;'
                 'filter:drop-shadow(0 3px 6px rgba(0,0,0,0.55));'
                 'user-select:none;-webkit-user-select:none;">&#127968;</div>',
            icon_size=(40, 40), icon_anchor=(20, 36),
        ),
        tooltip="📍 Posición actual — usa el botón ✏️ para colocar un nuevo marcador",
        popup=folium.Popup(
            f"<b>📍 Airbnb</b><br>Colonia: <b>{colonia_sel.title()}</b>",
            max_width=240,
        ),
        draggable=False,
    ).add_to(m)

    # Herramienta de dibujo (solo marcadores)
    Draw(
        draw_options={
            "marker": True, "polyline": False, "polygon": False,
            "circle": False, "rectangle": False, "circlemarker": False,
        },
        edit_options={"edit": False, "remove": False},
        position="topleft",
    ).add_to(m)

    # CSS para icono de dibujo local (igual al original)
    _icon_css = folium.Element("""
    <style>
    .leaflet-layer .leaflet-marker-icon:not(.leaflet-div-icon) {
        display: none !important;
    }
    </style>
    """)
    m.get_root().html.add_child(_icon_css)

    # Capas de servicios (puntos)
    centro_utm  = _latlon_a_utm_mapa(lat_marcador, lon_marcador)
    resumen_capas = {}

    for clave, (pts_utm, nombres) in CAPAS_SERVICIOS.items():
        cfg = CAPAS_CONFIG[clave]
        if len(pts_utm) == 0:
            resumen_capas[clave] = 0
            continue
        tree = cKDTree(pts_utm)
        idx  = tree.query_ball_point(centro_utm, r=RADIO_M)
        if not idx:
            resumen_capas[clave] = 0
            continue
        pts_dentro     = pts_utm[idx]
        nombres_dentro = [nombres[i] for i in idx] if nombres else []
        total          = len(pts_dentro)
        resumen_capas[clave] = total
        max_p = cfg["max_puntos"]
        if total > max_p:
            step           = max(1, total // max_p)
            pts_dentro     = pts_dentro[::step][:max_p]
            nombres_dentro = nombres_dentro[::step][:max_p]
        coords_ll = _utm_a_latlon_mapa(pts_dentro)
        grupo     = folium.FeatureGroup(name=cfg["label"], show=True)
        for i, (lt, ln) in enumerate(coords_ll):
            nombre_punto = nombres_dentro[i] if i < len(nombres_dentro) else f"{cfg['label']} #{i+1}"
            folium.Marker(
                location=[lt, ln],
                icon=folium.Icon(color=cfg["color"], icon=cfg["icono"], prefix=cfg["prefix"]),
                tooltip=folium.Tooltip(nombre_punto, sticky=True),
                popup=folium.Popup(
                    f"<b>{cfg['label']}</b><br>{nombre_punto}<br>"
                    f"<span style='font-size:10px;color:#888'>{lt:.5f}, {ln:.5f}</span>",
                    max_width=200,
                ),
            ).add_to(grupo)
        grupo.add_to(m)

    # Capa ciclovías (líneas)
    _cic_utm = CAPAS_LINEAS.get("ciclovias_utm", gpd.GeoDataFrame(geometry=[], crs="EPSG:32614"))
    _cic_wgs = CAPAS_LINEAS.get("ciclovias_wgs", gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
    if not _cic_utm.empty:
        from shapely.geometry import Point as _Pt
        _circulo_utm  = _Pt(centro_utm[0], centro_utm[1]).buffer(RADIO_M)
        _mask         = _cic_utm.intersects(_circulo_utm)
        _cic_wgs_dentro = _cic_wgs[_mask]
        if not _cic_wgs_dentro.empty:
            resumen_capas["ciclovias"] = len(_cic_wgs_dentro)
            grupo_lineas = folium.FeatureGroup(name="🚲 Ciclovías", show=True)
            for _, row in _cic_wgs_dentro.iterrows():
                nombre = str(row.get("NOMBRE", "") or "").strip()
                if not nombre or nombre in ("nan", "None"):
                    nombre = "Ciclovía sin nombre"
                if len(nombre) > 50:
                    nombre = nombre[:47] + "…"
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda _: {"color": "#008080", "weight": 3, "opacity": 0.85},
                    tooltip=folium.Tooltip(f"🚲 {nombre}", sticky=True),
                    popup=folium.Popup(f"<b>Ciclovía</b><br>{nombre}", max_width=200),
                ).add_to(grupo_lineas)
            grupo_lineas.add_to(m)
        else:
            resumen_capas["ciclovias"] = 0
    else:
        resumen_capas["ciclovias"] = 0

    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    # Leyenda (igual que el original)
    def _fila_leyenda(clave, cfg):
        color_hex = _COLOR_HEX[cfg["color"]]
        return (
            f"<div style='display:flex;align-items:center;margin-bottom:5px;'>"
            f"<span style='background:{color_hex};width:12px;height:12px;"
            f"border-radius:50%;display:inline-block;margin-right:8px;"
            f"border:1px solid rgba(255,255,255,0.25);flex-shrink:0;'></span>"
            f"<span style='font-size:11.5px;line-height:1.3;'>{cfg['label']}</span></div>"
        )
    leyenda_filas = "".join(_fila_leyenda(k, v) for k, v in CAPAS_CONFIG.items())

    leyenda_html = f"""
    <div id="leyenda-wrapper" style="position:fixed;bottom:30px;left:30px;z-index:9999;font-family:'Segoe UI',Arial,sans-serif;">
        <button id="leyenda-toggle-btn"
            onclick="(function(){{var p=document.getElementById('leyenda-panel');var oculto=p.style.display==='none';p.style.display=oculto?'block':'none';}})()"
            title="Ocultar/Mostrar leyenda"
            style="width:36px;height:36px;border-radius:8px;border:1px solid rgba(255,255,255,0.2);
                   background:rgba(30,41,59,0.97);color:#e2e8f0;font-size:18px;cursor:pointer;
                   display:flex;align-items:center;justify-content:center;
                   box-shadow:0 2px 8px rgba(0,0,0,0.4);margin-bottom:6px;padding:0;">👁️</button>
        <div id="leyenda-panel" style="background:rgba(30,41,59,0.97);border:1px solid rgba(255,255,255,0.15);
             border-radius:12px;padding:10px 14px;box-shadow:0 4px 20px rgba(0,0,0,0.40);
             min-width:185px;color:#e2e8f0;">
            <div style='font-weight:700;font-size:12.5px;margin-bottom:8px;color:#93C5FD;
                        border-bottom:2px solid #3B82F6;padding-bottom:5px;'>
                📍 Servicios en radio 1.2 km
            </div>
            {leyenda_filas}
            <div style='margin-top:8px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.1);
                        display:flex;align-items:center;gap:6px;font-size:11px;color:#94a3b8;'>
                <span style='display:inline-block;width:22px;height:0;border-top:2px dashed #60A5FA;flex-shrink:0;'></span>
                Límite de colonia
            </div>
            <div style='display:flex;align-items:center;gap:6px;font-size:11px;color:#94a3b8;margin-top:4px;'>
                <span style='display:inline-block;width:22px;height:0;border-top:2px solid #F87171;flex-shrink:0;'></span>
                Radio de análisis
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(leyenda_html))

    # Título del mapa
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin:1.2rem 0 0.5rem;">
        <span style="font-size:1.15rem;font-weight:700;letter-spacing:-0.2px;color:var(--text-primary);">🗺️ Entorno Urbano</span>
        <span style="background:#1A73E8;color:#fff;font-size:0.7rem;font-weight:700;padding:3px 11px;border-radius:20px;letter-spacing:0.04em;">RADIO 1.2 KM</span>
        <span style="font-size:0.72rem;color:var(--text-muted);opacity:0.7;">— usa el botón 👁️ en el mapa para ocultar/mostrar la leyenda</span>
    </div>
    """, unsafe_allow_html=True)

    if punto_confirmado:
        st.success(f"✅ Punto confirmado en {lat_marcador:.5f}, {lon_marcador:.5f} — distancias recalculadas.")
    else:
        st.info("📍 Usa el ícono ✏️ del panel izquierdo del mapa para colocar un marcador, luego confirma.")

    # Badges de conteo de servicios
    iconos_txt    = {"metro":"🚇","metrobus":"🚌","tren":"🚊","trolebus":"🚎",
                     "cablebus":"🚠","parques":"🌳","salud":"🏥","comercio":"🛍️",
                     "esc_privada":"🏫","esc_publica":"🏫","ciclovias":"🚲"}
    nombres_cortos = {"metro":"Metro","metrobus":"Metrobús","tren":"Tren","trolebus":"Trolebús",
                      "cablebus":"Cablebús","parques":"Parques","salud":"Salud","comercio":"Comercio",
                      "esc_privada":"Esc. Priv.","esc_publica":"Esc. Púb.","ciclovias":"Ciclovías"}

    badges_html = '<div class="servicios-grid">'
    for clave, n in resumen_capas.items():
        icono  = iconos_txt.get(clave, "📍")
        nombre = nombres_cortos.get(clave, clave)
        color  = "#3B82F6" if n > 0 else "var(--text-muted)"
        badges_html += (
            f'<div class="servicio-badge">'
            f'  <span class="s-icono">{icono}</span>'
            f'  <span class="s-nombre">{nombre}</span>'
            f'  <span class="s-numero" style="color:{color};">{n}</span>'
            f'</div>'
        )
    badges_html += '</div>'
    st.markdown(badges_html, unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Renderizar mapa — FIX key: incluye colonia_key_sel completo
    resultado_mapa = st_folium(
        m,
        use_container_width=True,
        height=620,
        key=f"mapa_{colonia_key_sel}_{st.session_state.get(f'recenter_{colonia_key_sel}', 0)}",
        returned_objects=["last_active_drawing"],
    )

    # Detectar nuevo punto colocado con Draw
    _drawn = resultado_mapa.get("last_active_drawing")
    nuevo_lat, nuevo_lon = None, None
    if _drawn and isinstance(_drawn, dict):
        _geom = _drawn.get("geometry", {})
        if _geom.get("type") == "Point":
            _coords = _geom.get("coordinates", [])
            if len(_coords) == 2:
                nuevo_lat = _coords[1]
                nuevo_lon = _coords[0]

    if nuevo_lat is not None and nuevo_lon is not None:
        _dentro = True
        if poligono_colonia_geojson is not None:
            from shapely.geometry import Point as _Pt2, shape as _shape2
            _poly  = _shape2(poligono_colonia_geojson)
            _dentro = _poly.contains(_Pt2(nuevo_lon, nuevo_lat))

        if _dentro:
            if (
                abs(nuevo_lat - st.session_state[_key_lat]) > 0.000001
                or abs(nuevo_lon - st.session_state[_key_lon]) > 0.000001
            ):
                st.session_state[_key_lat]        = nuevo_lat
                st.session_state[_key_lon]        = nuevo_lon
                st.session_state[_key_confirmado] = False
                st.rerun()
        else:
            st.warning("⚠️ Posición fuera de la colonia — no se actualizó. Mueve el marcador dentro del límite azul y confirma.")

    lat_actual       = st.session_state[_key_lat]
    lon_actual       = st.session_state[_key_lon]
    punto_confirmado = st.session_state[_key_confirmado]

    # Botones de control
    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3, _ = st.columns([2, 2, 2, 2])

    with col_btn1:
        if st.button("✅ Confirmar ubicación", type="primary", use_container_width=True):
            st.session_state[_key_confirmado] = True
            st.session_state[_key_dist]       = None
            st.rerun()

    with col_btn2:
        if st.button("↩️ Restablecer centroide", use_container_width=True):
            st.session_state[_key_lat]        = lat_base
            st.session_state[_key_lon]        = lon_base
            st.session_state[_key_confirmado] = False
            st.session_state[_key_dist]       = None
            st.rerun()

    with col_btn3:
        if st.button("🎯 Recentrar en vivienda", use_container_width=True,
                     help="Vuelve el mapa al punto de análisis actual con zoom completo"):
            st.session_state[f"recenter_{colonia_key_sel}"] = (
                st.session_state.get(f"recenter_{colonia_key_sel}", 0) + 1
            )
            st.rerun()

    st.markdown(
        f"<div style='font-size:0.75rem;color:var(--text-muted);margin-top:0.5rem;"
        f"padding:0.5rem 0.75rem;background:var(--bg-surface-alt);"
        f"border-radius:8px;border:1px solid var(--border-subtle);'>"
        f"📐 Radio de análisis: <b style='color:var(--text-secondary);'>{RADIO_M} m</b> &nbsp;·&nbsp; "
        f"Estado: <b style='color:var(--text-secondary);'>{'✅ Confirmado' if punto_confirmado else '⏳ Pendiente'}</b> &nbsp;·&nbsp; "
        f"Coordenadas: <b style='color:var(--text-secondary);'>{lat_actual:.5f}, {lon_actual:.5f}</b>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

# ============================================================
# PESTAÑA: ATLAS
# ============================================================
with tab_atlas:
    st.markdown("""
    <div style="
        background: var(--secondary-background-color);
        border-radius: 16px;
        padding: 2.5rem 2.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color);
        border-left: 6px solid #3B82F6;
    ">
        <div style="color: var(--text-color); font-size: 1.8rem; font-weight: 800; margin: 0 0 0.5rem; letter-spacing: -0.5px; line-height: 1.2;">
            🌆 Estudio Multidimensional del Mercado Airbnb
        </div>
        <p style="color: var(--text-color); opacity: 0.7; font-size: 1rem; margin: 0; line-height: 1.6;">
            Análisis espacial integral de las dinámicas sociales, económicas y urbanas<br>
            en la Ciudad de México · Elaborado con QGIS y datos abiertos CDMX
        </p>
    </div>
    """, unsafe_allow_html=True)

    MAPAS_DIR = os.path.join(PROJECT_ROOT, "..", "..", "..", "outputs", "assets", "mapas_qgis_airbnb")
    if not os.path.exists(MAPAS_DIR):
        MAPAS_DIR = os.path.join(PROJECT_ROOT, "..", "..", "..", "outputs", "assets", "mapas_qgis")

    MAPAS_ATLAS = [
        {"archivo": "mapa_airbnb_distribucion_precio_por_noche.png",   "titulo": "Distribución del Precio por Noche (Airbnb)", "emoji": "💰", "descripcion": "Precio promedio por noche de listings en la CDMX. Gradientes de valor desde Polanco, Roma–Condesa hacia la periferia.", "categoria": "Mercado Airbnb"},
        {"archivo": "mapa_airbnb_densidad_listings.png",                  "titulo": "Densidad de Oferta Airbnb",                  "emoji": "🏗️", "descripcion": "Hotspots de concentración de anuncios activos. Mayor densidad en zonas turísticas y de alta plusvalía.",                   "categoria": "Mercado Airbnb"},
        {"archivo": "mapa_airbnb_segmentación_cluster_mayoria_colonia.png",    "titulo": "Clústeres Espaciales de Precios",            "emoji": "🧩", "descripcion": "Segmentación K-Means basada en coordenadas y precio. Identifica submercados de lujo y estándar.",                          "categoria": "Mercado Airbnb"},
        {"archivo": "mapa_airbnb_gentrificacion_nivel_colonia.png",              "titulo": "Índice de Gentrificación (Airbnb) Nivel Colonia",          "emoji": "🏙️", "descripcion": "Presión gentrificadora asociada a la concentración de alquileres de corta estancia.",                                    "categoria": "Dinámicas Sociales"},
        {"archivo": "mapa_airbnb_15_min_nivel_colonia.png",         "titulo": "Ciudad de 15 Minutos para Airbnb",           "emoji": "🚶", "descripcion": "Puntaje de accesibilidad peatonal a servicios desde cada listing.",                                                         "categoria": "Movilidad y Accesibilidad"},
    ]

    categorias = sorted(set(m["categoria"] for m in MAPAS_ATLAS))
    cat_sel    = st.multiselect("🔍 Filtrar por categoría temática", options=categorias, default=categorias,
                                 help="Selecciona una o más categorías para filtrar los mapas mostrados.")
    mapas_filtrados = [m for m in MAPAS_ATLAS if m["categoria"] in cat_sel]

    if not mapas_filtrados:
        st.warning("Selecciona al menos una categoría para ver los mapas.")
    else:
        COLORES_CAT = {
            "Mercado Airbnb":         ("#0a1e3d", "#60a5fa"),
            "Dinámicas Sociales":     ("#290a0a", "#f87171"),
            "Movilidad y Accesibilidad": ("#160a2d", "#c084fc"),
        }
        cols_galeria = st.columns(2, gap="large")
        for i, mapa in enumerate(mapas_filtrados):
            col        = cols_galeria[i % 2]
            ruta_img   = os.path.join(MAPAS_DIR, mapa["archivo"])
            _, accent  = COLORES_CAT.get(mapa["categoria"], ("#111827", "#94a3b8"))
            with col:
                st.markdown(f"""
                <div style="
                    background: var(--secondary-background-color);
                    border: 1px solid var(--border-color);
                    border-left: 5px solid {accent};
                    border-radius: 12px;
                    padding: 1rem 1.2rem 0.9rem;
                    margin-bottom: 0.5rem;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
                ">
                    <span style="background: {accent}20; color: {accent}; font-size: 0.68rem; font-weight: 800;
                                 letter-spacing: 0.1em; padding: 3px 10px; border-radius: 20px;
                                 text-transform: uppercase; border: 1px solid {accent}40; display: inline-block;
                                 margin-bottom: 0.1rem;">{mapa['categoria']}</span>
                    <h3 style="color: var(--text-color); font-size: 1.05rem; font-weight: 700;
                               margin: 0.55rem 0 0.35rem; line-height: 1.3;">{mapa['emoji']} {mapa['titulo']}</h3>
                    <p style="color: var(--text-color); opacity: 0.85; font-size: 0.84rem; margin: 0; line-height: 1.55;">{mapa['descripcion']}</p>
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists(ruta_img):
                    st.image(ruta_img, use_container_width=True)
                    st.markdown(f"""
                    <div style="text-align:center; font-size:0.78rem; color:var(--text-color); opacity:0.6;
                                margin-top:-0.25rem; margin-bottom:1rem;">
                        Mapa {i+1} de {len(mapas_filtrados)} · {mapa['titulo']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:var(--secondary-background-color); border:2px dashed var(--border-color);
                                border-radius:8px; padding:3rem 1rem; text-align:center;
                                color:var(--text-color); opacity:0.6; font-size:0.9rem; margin-bottom:1rem;">
                        📁 <code>{mapa['archivo']}</code><br>
                        <small>Coloca el PNG exportado de QGIS en<br><code>assets/mapas_qgis_airbnb/</code></small>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.caption(f"📊 {len(mapas_filtrados)} mapas mostrados · Fuentes: INEGI, ADIP CDMX, Airbnb · Elaboración propia con QGIS y Python")

# ============================================================
# FOOTER PROFESIONAL (idéntico al original)
# ============================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:2rem 0; border-top:1px solid var(--border-color);
            margin-top:3rem; background-color: transparent;">
    <div style="color: var(--text-color); opacity: 0.7; font-size:0.875rem; line-height:1.8;">
        <div style="font-weight:700; color: var(--text-color); margin-bottom:0.5rem; font-size: 1rem;">
            Sistema de Valuación Airbnb CDMX
        </div>
        <div>Escuela Superior de Cómputo • Instituto Politécnico Nacional</div>
        <div style="margin-top:0.5rem; font-weight: 600; color: var(--text-color);">
            Kevin J. González Sosa • José M. Torres Gutiérrez
        </div>
        <div style="font-size: 0.75rem; opacity: 0.6; margin-top: 0.25rem;">
            Trabajo Terminal · ESCOM 2026
        </div>
    </div>
</div>
""", unsafe_allow_html=True)