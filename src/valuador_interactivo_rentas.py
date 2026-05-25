# =============================================================================
# TRABAJO TERMINAL
# Valuador Interactivo para Rentas en CDMX
# Basado en el modelo hedónico-espacial de renta.py (versión final)
#
# Autores: Kevin Jesús González Sosa, José Manuel Torres Gutiérrez
# Institución: Escuela Superior de Cómputo
# Fecha: Mayo, 2026
#
# Descripción:
# Interfaz interactiva y desplegable (Streamlit) para la visualización,
# consulta y estimación de rentas mensuales basada en submercados
# diferenciados territorialmente y variables socioespaciales complejas.
#
# El sistema integra:
#   - Predicción hedónico-espacial mediante modelos ElasticNet locales.
#   - Análisis de contribución marginal de amenidades urbanas.
#   - Evaluación contextualizada con el índice de gentrificación.
#   - Métricas asociadas al paradigma de la Ciudad de 15 Minutos.
#   - Variables catastrales pre-calculadas (cargadas desde CSV exportado).
#
# NOTA: Las variables catastrales (cat_*) se cargan directamente del CSV
# exportado por renta.py. No se requiere cargar los shapefiles del catastro.
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString

import streamlit as st
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

import folium
from streamlit_folium import st_folium


# ============================================================
# CONFIGURACIÓN DE LA INTERFAZ DE STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Valuador de Rentas CDMX · ESCOM-IPN",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PARCHE SIDEBAR — fondo sólido garantizado
# ============================================================
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
# LOGOS INSTITUCIONALES
# ============================================================
import base64 as _b64
import os as _os

def _img_to_b64(path):
    try:
        with open(path, "rb") as _f:
            return _b64.b64encode(_f.read()).decode()
    except Exception:
        return ""

_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
def _find_logo(filename):
    for candidate in [
        _os.path.join(_SCRIPT_DIR, filename),
        _os.path.join(_SCRIPT_DIR, "assets", filename),
        _os.path.join(_SCRIPT_DIR, "..", "outputs", "assets", filename),
    ]:
        if _os.path.exists(candidate):
            return candidate
    return ""

_escom_b64 = _img_to_b64(_find_logo("escom.png"))
_ipn_b64   = _img_to_b64(_find_logo("ipn.png"))

# ============================================================
# CSS GLOBAL (igual al valuador de compra/venta)
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
/* SIDEBAR — fondo sólido absoluto */
/* Cubre el section raíz y TODOS sus descendientes */
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

/* INPUTS dinámicos */
[data-testid="stSelectbox"] label, [data-testid="stSlider"] label, [data-testid="stNumberInput"] label {
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
    .block-container {
        padding-top: 0.75rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
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
    [data-testid="stMetric"] { padding: 1rem !important; border-radius: 12px !important; }
    [data-testid="column"] { min-width: 100% !important; padding: 0 !important; }
    .servicio-badge { padding: 8px 8px !important; }
    .servicio-badge .s-numero { font-size: 0.95rem !important; }
    .servicios-grid {
        grid-template-columns: repeat(auto-fit, minmax(80px, 1fr)) !important;
        gap: 8px !important;
    }
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }

    /* ── Sidebar móvil: fondo sólido garantizado ── */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        background-color: var(--background-color, #ffffff) !important;
        color: var(--text-color, #31333F) !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
        opacity: 1 !important;
    }

    /* Sliders: sólo hacer transparente el contenedor interno */
    section[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] > div {
        background-color: transparent !important;
    }

    /* Nodo (thumb) del slider */
    section[data-testid="stSidebar"] [class*="stSlider"] [role="slider"] {
        background-color: #3B82F6 !important;
        border: 2px solid #ffffff !important;
        opacity: 1 !important;
        z-index: 9999 !important;
    }

    /* Barra de progreso activa del slider */
    section[data-testid="stSidebar"] [class*="stSlider"] div[data-track="true"] {
        background-color: #3B82F6 !important;
        opacity: 1 !important;
    }

    /* Number inputs en móvil */
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] {
        background-color: transparent !important;
    }
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
        color: #31333F !important;
        background-color: #F0F2F6 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] button {
        background-color: #E0E3E9 !important;
        color: #31333F !important;
        opacity: 1 !important;
        border: 1px solid #D1D5DB !important;
    }
    section[data-testid="stSidebar"] [data-testid="stNumberInput"] button svg {
        fill: #31333F !important;
        color: #31333F !important;
    }

    /* Textos en el sidebar móvil */
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] span {
        color: var(--text-color, #31333F) !important;
    }

    section[data-testid="stSidebar"] {
        min-width: 300px !important;
        box-shadow: 8px 0 32px rgba(0, 0, 0, 0.15) !important;
        z-index: 999999 !important;
    }

    /* Inputs táctiles más grandes */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stNumberInput"] input {
        min-height: 44px !important;
        font-size: 1rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# ENCABEZADO INSTITUCIONAL PREMIUM
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
            <div class="main-title">🏘️ Sistema de Valuación de Rentas · CDMX</div>
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
# RUTAS RELATIVAS (ESTRUCTURA DE REPOSITORIO)
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(PROJECT_ROOT, "..", "data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
OUTPUTS_PATH_RENTAS = os.path.join(PROJECT_ROOT, "..", "outputs", "rentas", "results")

PATH_QGIS_DATA = os.path.join(OUTPUTS_PATH_RENTAS, "resultados_qgis_rentas.csv")

PATH_COLONIAS            = os.path.join(RAW_DATA_PATH, "coloniascdmx", "colonias_iecm.shp")
PATH_METRO_EST           = os.path.join(RAW_DATA_PATH, "stcmetro_shp", "STC_Metro_estaciones_utm14n.shp")
PATH_METROBUS_EST        = os.path.join(RAW_DATA_PATH, "mb_shp", "Metrobus_estaciones.shp")
PATH_TREN_EST            = os.path.join(RAW_DATA_PATH, "ste_shp", "ste_tren_ligero_shp",
                                        "ste_tren_ligero_shp", "STE_TrenLigero_estaciones_utm14n.shp")
PATH_TROLE_PARADAS       = os.path.join(RAW_DATA_PATH, "ste_shp", "ste_trolebus_shp",
                                        "ste_trolebus_shp", "STE_Trolebus_Paradas.shp")
PATH_CABLE_EST           = os.path.join(RAW_DATA_PATH, "ste_shp", "ste_cablebus_shp",
                                        "ste_cablebus_shp", "STE_Cablebus_estaciones.shp")
PATH_AREAS_VERDES        = os.path.join(RAW_DATA_PATH, "inventario_areas_verdes_1",
                                        "inventario_areas_verdes_1.shp")
PATH_SALUD_BASE          = os.path.join(RAW_DATA_PATH, "hospitales_y_centros_de_salud",
                                        "hospitales_y_centros_de_salud.shp")
PATH_HOSPITALES_PUBLICOS = os.path.join(RAW_DATA_PATH, "hospitales_2020_publicos",
                                        "hospitales_2020_publicos.shp")
PATH_COMERCIO            = os.path.join(RAW_DATA_PATH, "cypc", "C_PComerciales.shp")
PATH_ESCUELAS_PRIVADAS   = os.path.join(RAW_DATA_PATH, "escuelas_privadas", "escuelas_privadas.shp")
PATH_ESCUELAS_PUBLICAS   = os.path.join(RAW_DATA_PATH, "escuelas_publicas", "escuelas_publicas.shp")
PATH_CICLOVIAS           = os.path.join(RAW_DATA_PATH, "infraestructura_vial_ciclista",
                                        "Infraestructura ciclista total.shp")

# ============================================================
# SUBCENTROS URBANOS — idénticos a renta.py
# ============================================================
SUBCENTROS = {
    "Centro":      (19.4326, -99.1332),
    "Polanco":     (19.4330, -99.1960),
    "SantaFe":     (19.3619, -99.2736),
    "Insurgentes": (19.4045, -99.1700),
    "DelValle":    (19.3738, -99.1645),
    "Reforma":     (19.4273, -99.1677),
}

# ============================================================
# FEATURES — ALINEADAS CON renta.py (versión final)
# ============================================================
_STATIC_FEATURES_FULL = [
    "rooms", "area", "bathrooms", "parking_spaces", "antiguedad",
    "dist_subcenter_log", "marginalidad_score", "comercio_density",
    "dist_metro_m", "density_metro", "dist_metrobus_m", "density_metrobus",
    "dist_tren_m", "density_tren", "dist_trole_m", "density_trole",
    "dist_cable_m", "density_cable", "pct_migrantes_turistas",
    "gentrification_index", "area_x_marginalidad", "area_X_gentrif",
    "gentrif_x_metro", "dist_ciclovia_m", "densidad_ciclovia_15m",
    "dist_area_verde_m", "densidad_parques_15m", "dist_salud_m",
    "acceso_salud_15m", "dist_escuela_m", "acceso_educacion_15m",
    "score_15min", "prox_ciclovia", "prox_parque", "prox_salud",
    "prox_escuela", "uso_mixto", "15min_X_gentrif", "listing_density_log",
    "verde_marginalidad_ratio", "salud_marginalidad_ratio",
    "educacion_marginalidad_ratio",
    "cat_mean_valor_suelo_200m", "cat_mean_vus_200m",
    "cat_mean_antiguedad_200m", "cat_std_valor_suelo_200m",
    "cat_mean_ratio_construccion_200m", "cat_density_predios_200m",
]

_DYNAMIC_FEATURES = [
    "spatial_lag_price",
    "precio_vecinal_local",
    "lag_x_area",
]

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

def compute_spatial_lag(train_coords, train_prices, target_coords, k=30):
    tree = cKDTree(train_coords)
    d, ix = tree.query(target_coords, k=min(k, len(train_coords)))
    if k == 1:
        d = d.reshape(-1, 1)
        ix = ix.reshape(-1, 1)
    weights = 1.0 / (d + 10.0)
    lag_values = [np.average(train_prices[idxs], weights=w) for idxs, w in zip(ix, weights)]
    return np.log1p(np.array(lag_values))

def compute_local_neighbor_price(train_coords, train_prices, target_coords, k=15):
    nbrs = NearestNeighbors(n_neighbors=min(k, len(train_coords))).fit(train_coords)
    _, indices = nbrs.kneighbors(target_coords)
    return np.array([train_prices[idx].mean() for idx in indices])

def density_proxy_from_points(coords_utm, pts, radio=1200):
    if len(pts) == 0:
        return np.zeros(len(coords_utm))
    tree = cKDTree(pts)
    return np.array([len(tree.query_ball_point(xy, r=radio)) for xy in coords_utm])

def fmt_dist(metros):
    if metros >= 1000:
        return f"{metros/1000:.1f} km"
    return f"{metros:.0f} m"

def encabezado_seccion(titulo, subtitulo, icono):
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.75rem;">
        <span style="font-size:1.5rem;">{icono}</span>
        <div>
            <div style="font-size:1rem; font-weight:700; color:var(--text-color);">{titulo}</div>
            <div style="font-size:0.8rem; color:var(--text-color); opacity:0.6;">{subtitulo}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# SISTEMA PRINCIPAL (cacheado) - replica la segmentación de dos etapas de renta.py
# ============================================================
@st.cache_resource
def preparar_sistema():
    """
    Carga el CSV exportado por renta.py, replica el pipeline de features
    y entrena los modelos ElasticNet locales por segmento (estándar/lujo)
    y por clúster (6 clústeres para estándar, 1 para lujo).
    También carga capas de puntos y líneas para visualización.
    """

    # 1. CARGA Y LIMPIEZA INICIAL
    df = pd.read_csv(PATH_QGIS_DATA)
    df["latitud"]  = pd.to_numeric(df["latitud"],  errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")
    df = df.dropna(subset=["latitud", "longitud", "price", "area", "price_m2_raw"]).copy()
    df = df[df["area"] >= 20].copy()

    # Filtro geografico: eliminar registros fuera del bbox de la CDMX
    _CDMX_LAT_MIN, _CDMX_LAT_MAX = 19.04, 19.60
    _CDMX_LON_MIN, _CDMX_LON_MAX = -99.37, -98.94
    df = df[
        (df["latitud"]  >= _CDMX_LAT_MIN) & (df["latitud"]  <= _CDMX_LAT_MAX) &
        (df["longitud"] >= _CDMX_LON_MIN) & (df["longitud"] <= _CDMX_LON_MAX)
    ].copy()

    df = df.reset_index(drop=True)

    # 2. DETERMINAR STATIC_FEATURES (replicando drop por colinealidad de renta.py)
    disponibles = [f for f in _STATIC_FEATURES_FULL if f in df.columns]
    corr_matrix = df[disponibles].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > 0.90)]
    static_features = [f for f in disponibles if f not in drop_cols]
    features_all    = static_features + _DYNAMIC_FEATURES

    # 3. GEOMETRÍA PARA JOINS Y DISTANCIAS
    gdf_geo = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitud"], df["latitud"]),
        crs="EPSG:4326"
    ).reset_index(drop=True)

    gdf_utm    = gdf_geo.to_crs(epsg=32614)
    coords_utm = np.array([(p.x, p.y) for p in gdf_utm.geometry])

    # 4. CRUCE ESPACIAL CON COLONIAS
    colonias = gpd.read_file(PATH_COLONIAS)
    if colonias.crs != "EPSG:4326":
        colonias = colonias.to_crs("EPSG:4326")

    colonias["colonia_clean"] = colonias["NOMUT"].apply(clean_text)
    posibles_alcaldias = ["NOMDT", "DEMARCACI", "DEMARCACION", "MUNICIPIO", "NOM_MUN"]
    alcaldia_col = next((c for c in posibles_alcaldias if c in colonias.columns), None)
    if alcaldia_col is None:
        raise ValueError("Falta identificador de alcaldía en el shapefile de colonias.")
    colonias["alcaldia_clean"] = colonias[alcaldia_col].apply(clean_text)

    joined = gpd.sjoin(
        gdf_geo,
        colonias[["colonia_clean", "alcaldia_clean", "geometry"]],
        how="left",
        predicate="within"
    ).sort_index()

    df["colonia_real"]  = joined["colonia_clean"].values
    df["alcaldia_real"] = joined["alcaldia_clean"].values
    df["colonia_key"]   = (df["alcaldia_real"].astype(str) + "||" + df["colonia_real"].astype(str))
    df = df.dropna(subset=["colonia_real", "alcaldia_real"]).copy()
    df = df.reset_index(drop=True)

    gdf_utm2    = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitud"], df["latitud"]),
        crs="EPSG:4326"
    ).to_crs(epsg=32614)
    coords_utm2 = np.array([(p.x, p.y) for p in gdf_utm2.geometry])

    # 5. RECALCULAR DIST A TRANSPORTE (usando las mismas capas que renta.py)
    def _get_pts(path):
        try:
            gdf = gpd.read_file(path)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            gdf = gdf.to_crs(epsg=32614)
            gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
            exp = gdf.geometry.explode(index_parts=False)
            return np.array([(g.x, g.y) for g in exp if g.geom_type == "Point"])
        except Exception:
            return np.empty((0, 2))

    metro_pts  = _get_pts(PATH_METRO_EST)
    mb_pts     = _get_pts(PATH_METROBUS_EST)
    tren_pts   = _get_pts(PATH_TREN_EST)
    trole_pts  = _get_pts(PATH_TROLE_PARADAS)
    cable_pts  = _get_pts(PATH_CABLE_EST)

    try:
        av = gpd.read_file(PATH_AREAS_VERDES)
        if "subcat_sed" in av.columns:
            cats_validas = ["Parques", "Deportivos", "Jardines públicos", "Arboledas", "Alamedas"]
            av = av[av["subcat_sed"].isin(cats_validas)].copy()
        av = av[av.geometry.notnull() & av.is_valid].to_crs(epsg=32614)
        av["geometry"] = av.geometry.centroid
        av_pts = np.array([(g.x, g.y) for g in av.geometry])
    except Exception:
        av_pts = np.empty((0, 2))

    df["dist_metro_m"]    = nearest_distance(metro_pts,  coords_utm2)
    df["dist_metrobus_m"] = nearest_distance(mb_pts,     coords_utm2)
    df["dist_tren_m"]     = nearest_distance(tren_pts,   coords_utm2)
    df["dist_trole_m"]    = nearest_distance(trole_pts,  coords_utm2)
    df["dist_cable_m"]    = nearest_distance(cable_pts,  coords_utm2)
    df["dist_area_verde_m"] = nearest_distance(av_pts,   coords_utm2)

    try:
        com = gpd.read_file(PATH_COMERCIO).to_crs(epsg=32614)
        com_pts = np.array([(g.centroid.x, g.centroid.y) for g in com.geometry])
        df["dist_comercio_m"] = nearest_distance(com_pts, coords_utm2)
    except Exception:
        com_pts = np.empty((0, 2))
        df["dist_comercio_m"] = 5000

    # Ciclovías
    try:
        _ciclo_raw = gpd.read_file(PATH_CICLOVIAS)
        if _ciclo_raw.crs is None:
            _ciclo_raw = _ciclo_raw.set_crs("EPSG:4326")
        elif _ciclo_raw.crs.to_epsg() != 4326:
            _ciclo_raw = _ciclo_raw.to_crs("EPSG:4326")
        ciclovias_wgs = _ciclo_raw[_ciclo_raw.geometry.notnull() & ~_ciclo_raw.geometry.is_empty].copy()
        ciclovias_utm = ciclovias_wgs.to_crs("EPSG:32614")
        ciclovias_pts = line_to_points(ciclovias_utm, interval_m=100)
        df["dist_ciclovia_m"]      = nearest_distance(ciclovias_pts, coords_utm2)
        df["densidad_ciclovia_15m"] = density_proxy_from_points(coords_utm2, ciclovias_pts, radio=1200)
    except Exception as e:
        st.warning(f"No se pudo cargar ciclovías: {e}")
        ciclovias_wgs = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        ciclovias_utm = gpd.GeoDataFrame(geometry=[], crs="EPSG:32614")
        ciclovias_pts = np.empty((0, 2))
        df["dist_ciclovia_m"]      = 5000.0
        df["densidad_ciclovia_15m"] = 0.0

    # 6. SEGMENTACIÓN DE MERCADO (top 10% = lujo)
    lux_cut = df["price_m2_raw"].quantile(0.90)
    df["is_luxury"] = (df["price_m2_raw"] >= lux_cut).astype(int)

    # 7. FUNCIÓN DE SEGMENTACIÓN EN DOS ETAPAS (IDÉNTICA A renta.py)
    def segmentar_dos_etapas(data, n_clusters=6, k_spatial=30):
        coords_xy = data[["longitud", "latitud"]].values
        prices    = data["price_m2_raw"].values

        tree  = cKDTree(coords_xy)
        fetch = min(k_spatial + 1, len(coords_xy))
        d, ix = tree.query(coords_xy, k=fetch)
        d, ix = d[:, 1:], ix[:, 1:]
        w     = 1.0 / (d + 50.0)
        precio_suavizado = np.array([
            np.average(prices[idxs], weights=ws) for idxs, ws in zip(ix, w)
        ])

        pca_dict = {
            "precio_suavizado": precio_suavizado,
            "dist_subcenter":   data["dist_subcenter_log"].values,
            "marginalidad":     data["marginalidad_score"].values,
            "gentrificacion":   data["gentrification_index"].values,
            "colonia_enc":      data["colonia_price_enc"].values,
        }
        pca_feats = pd.DataFrame(pca_dict).fillna(0)
        scores_pca = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(pca_feats))

        geo_norm  = StandardScaler().fit_transform(coords_xy)
        X_cluster = np.hstack([scores_pca * 0.6, geo_norm * 0.4])
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(X_cluster)

        medias = {lbl: precio_suavizado[labels == lbl].mean() for lbl in range(n_clusters)}
        orden  = sorted(medias, key=medias.get)
        remap  = {old: new for new, old in enumerate(orden)}
        estrato = np.array([remap[l] for l in labels])
        data["cluster"] = estrato
        return data

    # 7b. Feature colateral: colonia_price_enc (target encoding global, solo para referencia)
    global_mean = df["price_m2_raw"].mean()
    col_sum = df.groupby("colonia_key")["price_m2_raw"].transform("sum")
    col_cnt = df.groupby("colonia_key")["price_m2_raw"].transform("count")
    df["colonia_price_enc"] = np.where(
        col_cnt > 1,
        (col_sum - df["price_m2_raw"]) / (col_cnt - 1),
        global_mean
    )
    df["colonia_price_enc"] = np.log1p(df["colonia_price_enc"])

    # 8. ENTRENAMIENTO POR SEGMENTO Y CLÚSTER
    modelos  = {}
    imputers = {}
    segmentadores = {}  # (seg_label) -> (km, scaler, cluster_feat)

    for seg_label, seg_mask in [("estandar", df["is_luxury"] == 0),
                                 ("lujo",     df["is_luxury"] == 1)]:
        data_seg = df[seg_mask].copy()
        q_low, q_high = data_seg["price_m2_raw"].quantile([0.05, 0.95])
        data_seg = data_seg[
            (data_seg["price_m2_raw"] > q_low) &
            (data_seg["price_m2_raw"] < q_high)
        ].copy()

        if len(data_seg) < 50:
            continue

        if seg_label == "estandar":
            n_clusters_seg = 6
            data_seg = segmentar_dos_etapas(data_seg, n_clusters=n_clusters_seg)
            # Guardar pipeline de segmentación
            coords_xy = data_seg[["longitud", "latitud"]].values
            prices    = data_seg["price_m2_raw"].values
            tree  = cKDTree(coords_xy)
            fetch = min(30 + 1, len(coords_xy))
            d, ix = tree.query(coords_xy, k=fetch)
            d, ix = d[:, 1:], ix[:, 1:]
            w = 1.0 / (d + 50.0)
            precio_suavizado = np.array([np.average(prices[idxs], weights=ws) for idxs, ws in zip(ix, w)])
            pca_dict = {
                "precio_suavizado": precio_suavizado,
                "dist_subcenter":   data_seg["dist_subcenter_log"].values,
                "marginalidad":     data_seg["marginalidad_score"].values,
                "gentrificacion":   data_seg["gentrification_index"].values,
                "colonia_enc":      data_seg["colonia_price_enc"].values,
            }
            pca_feats = pd.DataFrame(pca_dict).fillna(0)
            pca_scaler = StandardScaler().fit(pca_feats)
            scores_pca = pca_scaler.transform(pca_feats)
            pca_obj = PCA(n_components=2).fit(scores_pca)
            scores_pca = pca_obj.transform(scores_pca)

            geo_norm = StandardScaler().fit(coords_xy)
            X_cluster = np.hstack([scores_pca * 0.6, geo_norm.transform(coords_xy) * 0.4])
            km = KMeans(n_clusters=n_clusters_seg, random_state=42, n_init=20).fit(X_cluster)
            medias = {lbl: precio_suavizado[km.labels_ == lbl].mean() for lbl in range(n_clusters_seg)}
            orden = sorted(medias, key=medias.get)
            remap = {old: new for new, old in enumerate(orden)}
            segmentadores[seg_label] = {
                "km": km, "remap": remap,
                "pca_scaler": pca_scaler, "pca": pca_obj,
                "geo_scaler": geo_norm,
                "n_clusters": n_clusters_seg,
            }
        else:  # lujo
            data_seg["cluster"] = 0
            segmentadores[seg_label] = None

        coords_seg = data_seg[["longitud", "latitud"]].values

        for c in sorted(data_seg["cluster"].unique()):
            d_c = data_seg[data_seg["cluster"] == c].copy()
            if len(d_c) < 30:
                continue

            coords_c = d_c[["longitud", "latitud"]].values
            X_base   = d_c[static_features].copy()
            y        = np.log1p(d_c["price_m2_raw"])
            areas    = d_c["area"].values

            lag = compute_spatial_lag(coords_c, d_c["price_m2_raw"].values, coords_c, k=30)
            nvp = compute_local_neighbor_price(coords_c, d_c["price_m2_raw"].values, coords_c, k=15)

            X_base["spatial_lag_price"]   = lag
            X_base["precio_vecinal_local"] = nvp
            X_base["lag_x_area"] = X_base["spatial_lag_price"] * np.log1p(X_base["area"])

            for dyn in _DYNAMIC_FEATURES:
                if dyn not in X_base.columns:
                    X_base[dyn] = 0.0

            feat_disp = [f for f in features_all if f in X_base.columns]
            imputador = X_base[feat_disp].median()
            X_clean = X_base[feat_disp].fillna(imputador).fillna(0)

            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("enet", ElasticNetCV(
                    alphas=np.logspace(-4, -1, 25),
                    l1_ratio=[0.4, 0.6, 0.8],
                    cv=5,
                    max_iter=5000
                ))
            ])
            pipeline.fit(X_clean, y)

            modelos[(seg_label, c)]  = pipeline
            imputers[(seg_label, c)] = (imputador, feat_disp)

    # 9. REFERENCIA ESPACIAL POR COLONIA
    cols_ref = static_features + [
        "latitud", "longitud", "gentrif_local_presion", "gentrif_map_score",
        "dist_comercio_m", "dist_ciclovia_m", "densidad_ciclovia_15m",
        "densidad_parques_15m", "acceso_salud_15m", "acceso_educacion_15m",
        "cat_mean_valor_suelo_200m", "cat_mean_vus_200m",
        "cat_mean_antiguedad_200m", "cat_mean_ratio_construccion_200m",
        "cat_density_predios_200m", "colonia_price_enc",
    ]
    cols_ref = [c for c in cols_ref if c in df.columns]
    referencia_espacial = df.groupby("colonia_key")[cols_ref].mean().to_dict("index")

    text_feat_cols = [c for c in df.columns if c.startswith("text_svd_")]
    if text_feat_cols:
        referencia_texto = df.groupby("colonia_key")[text_feat_cols].mean().to_dict("index")
    else:
        referencia_texto = {}

    # Enriquecer referencia_espacial: SIEMPRE usar el centroide del shapefile como
    # fuente primaria de coordenadas. El promedio de lat/lon del CSV puede ser
    # inexacto cuando hay pocos registros, registros con GPS malo o colonias con
    # poca muestra. El centroide del polígono catastral es la referencia canónica.
    _CDMX_LAT_MIN_I, _CDMX_LAT_MAX_I = 19.04, 19.60
    _CDMX_LON_MIN_I, _CDMX_LON_MAX_I = -99.37, -98.94

    # Construir mapa colonia_key -> centroide del shapefile (WGS84)
    _shapefile_centroids = {}
    try:
        _cg_temp = colonias[["colonia_clean", "alcaldia_clean", "geometry"]].copy()
        _cg_temp = _cg_temp[_cg_temp.geometry.notnull()].copy()
        _cg_temp["colonia_key"] = (_cg_temp["alcaldia_clean"].astype(str)
                                   + "||" + _cg_temp["colonia_clean"].astype(str))
        _cg_temp = _cg_temp.dissolve(by="colonia_key").reset_index()
        for _, _row in _cg_temp.iterrows():
            _c = _row["geometry"].centroid
            _slat, _slon = _c.y, _c.x
            if (_CDMX_LAT_MIN_I <= _slat <= _CDMX_LAT_MAX_I and
                    _CDMX_LON_MIN_I <= _slon <= _CDMX_LON_MAX_I):
                _shapefile_centroids[_row["colonia_key"]] = (_slat, _slon)
    except Exception:
        pass

    # Para TODAS las colonias en referencia_espacial: si el shapefile tiene un
    # centroide válido, usarlo como coordenada base (más confiable que el promedio del CSV).
    # Si el CSV también tiene coords válidas, el shapefile sigue siendo preferido
    # para la inicialización del mapa; las coords del CSV se mantienen para el modelo.
    for _ck, _datos in referencia_espacial.items():
        _lat_csv = float(_datos.get("latitud", 0) or 0)
        _lon_csv = float(_datos.get("longitud", 0) or 0)
        _csv_ok = (_CDMX_LAT_MIN_I <= _lat_csv <= _CDMX_LAT_MAX_I and
                   _CDMX_LON_MIN_I <= _lon_csv <= _CDMX_LON_MAX_I)
        if _ck in _shapefile_centroids:
            _slat, _slon = _shapefile_centroids[_ck]
            # Siempre preferir el centroide del shapefile
            referencia_espacial[_ck]["latitud"]  = _slat
            referencia_espacial[_ck]["longitud"] = _slon
        elif not _csv_ok:
            # Sin shapefile y sin coords válidas del CSV: buscar por nombre de colonia
            # en otras alcaldías del shapefile (fallback de nombre aproximado)
            _col_clean = _ck.split("||")[-1] if "||" in _ck else _ck
            for _sk, _sv in _shapefile_centroids.items():
                if _sk.endswith(f"||{_col_clean}"):
                    referencia_espacial[_ck]["latitud"]  = _sv[0]
                    referencia_espacial[_ck]["longitud"] = _sv[1]
                    break

    price_m2_col = df.groupby("colonia_key")["price_m2_raw"].median().to_dict()
    alcaldia_colonias_raw = df.groupby("alcaldia_real")["colonia_real"].unique().apply(lambda x: sorted(list(x))).to_dict()

    # Filtrar: sólo incluir colonias que tienen datos EN referencia_espacial
    # Y cuyas coordenadas son válidas (ya enriquecidas con centroides del shapefile).
    # También incluir colonias con polígono disponible aunque el CSV no tenga coords
    # válidas (el centroide del polígono es suficiente para centrar el mapa).
    alcaldia_colonias = {}
    for alcaldia, lista_colonias in alcaldia_colonias_raw.items():
        colonias_validas = []
        for c in lista_colonias:
            _ck = f"{alcaldia}||{c}"
            if _ck not in referencia_espacial:
                continue
            _lat = float(referencia_espacial[_ck].get("latitud", 0) or 0)
            _lon = float(referencia_espacial[_ck].get("longitud", 0) or 0)
            _coord_ok = (_CDMX_LAT_MIN_I <= _lat <= _CDMX_LAT_MAX_I and
                         _CDMX_LON_MIN_I <= _lon <= _CDMX_LON_MAX_I)
            # Aceptar si tiene coords válidas O si existe polígono catastral para centrar el mapa
            _tiene_poligono = _ck in _shapefile_centroids
            if _coord_ok or _tiene_poligono:
                colonias_validas.append(c)
        if colonias_validas:
            alcaldia_colonias[alcaldia] = colonias_validas

    alcaldias_disponibles = sorted(alcaldia_colonias.keys())
    price_m2_by_colonia = df.groupby("colonia_key")["price_m2_raw"].mean().to_dict()

    # Polígonos de colonias
    colonias_geo = colonias[["colonia_clean", "alcaldia_clean", "geometry"]].copy()
    colonias_geo = colonias_geo[colonias_geo.geometry.notnull()].copy()
    colonias_geo["colonia_key"] = (colonias_geo["alcaldia_clean"].astype(str) + "||" + colonias_geo["colonia_clean"].astype(str))
    colonias_geo = colonias_geo.dissolve(by="colonia_key").reset_index()
    poligonos_colonias = {row["colonia_key"]: row["geometry"].__geo_interface__ for _, row in colonias_geo.iterrows()}

    # 10. CAPAS DE SERVICIOS PARA EL MAPA
    def _cargar_capa_con_nombres(path, col_nombre, fallback_prefix):
        try:
            gdf = gpd.read_file(path)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            gdf = gdf.to_crs(epsg=32614)
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
        salud_pts     = np.vstack([salud1_pts, salud2_pts])
        salud_nombres = salud1_nombres + salud2_nombres
    elif len(salud1_pts):
        salud_pts, salud_nombres = salud1_pts, salud1_nombres
    else:
        salud_pts, salud_nombres = salud2_pts, salud2_nombres

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

    capas_lineas = {
        "ciclovias_utm": ciclovias_utm if 'ciclovias_utm' in locals() else gpd.GeoDataFrame(geometry=[], crs="EPSG:32614"),
        "ciclovias_wgs": ciclovias_wgs if 'ciclovias_wgs' in locals() else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
    }

    return (
        modelos, imputers, segmentadores,
        static_features, features_all,
        referencia_espacial, referencia_texto,
        price_m2_col,
        alcaldias_disponibles, alcaldia_colonias,
        price_m2_by_colonia,
        capas_servicios, capas_lineas,
        poligonos_colonias,
        _shapefile_centroids,
    )

# ============================================================
# INSTANCIACIÓN
# ============================================================
with st.spinner("Cargando modelo de rentas y datos urbanos..."):
    (
        MODELS, IMPUTERS, SEGMENTADORES,
        STATIC_FEATURES, FEATURES_ALL,
        REF_ESPACIAL, REF_TEXTO,
        PRICE_M2_COL,
        ALCALDIAS_DISPONIBLES, ALCALDIA_COLONIAS,
        PRICE_M2_BY_COLONIA,
        CAPAS_SERVICIOS, CAPAS_LINEAS,
        POLIGONOS_COLONIAS,
        SHAPEFILE_CENTROIDS,
    ) = preparar_sistema()

# ============================================================
# RECALCULAR DISTANCIAS DESDE UN PUNTO
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

    esc_todos = (np.vstack([esc_priv_pts, esc_pub_pts]) if len(esc_priv_pts) and len(esc_pub_pts)
                 else (esc_priv_pts if len(esc_priv_pts) else esc_pub_pts))

    ciclovias_utm = CAPAS_LINEAS.get("ciclovias_utm", gpd.GeoDataFrame(geometry=[], crs="EPSG:32614"))
    ciclovias_pts = line_to_points(ciclovias_utm, interval_m=100) if not ciclovias_utm.empty else np.empty((0, 2))

    d_metro    = _dist_min(metro_pts)
    d_mb       = _dist_min(mb_pts)
    d_tren     = _dist_min(tren_pts)
    d_trole    = _dist_min(trole_pts)
    d_cable    = _dist_min(cable_pts)
    d_verde    = _dist_min(av_pts)
    d_salud    = _dist_min(salud_pts)
    d_escuela  = _dist_min(esc_todos)
    d_comercio = _dist_min(com_pts)
    d_ciclovia = _dist_min(ciclovias_pts)

    n_salud    = _conteo_radio(salud_pts)
    n_escuela  = _conteo_radio(esc_todos)
    n_parques  = _conteo_radio(av_pts)
    n_ciclovia = _conteo_radio(ciclovias_pts)

    prox_parque   = 1 if d_verde   <= 800  else 0
    prox_salud    = 1 if d_salud   <= 1000 else 0
    prox_escuela  = 1 if d_escuela <= 1000 else 0
    prox_ciclovia = 1 if d_ciclovia <= 500  else 0

    score = float(np.mean([
        min(n_salud   / 6,  1.0),
        min(n_escuela / 40, 1.0),
        min(n_parques / 10, 1.0),
        prox_parque, prox_salud, prox_escuela, prox_ciclovia,
    ]))

    return {
        "dist_metro_m":               d_metro,
        "dist_metrobus_m":            d_mb,
        "dist_tren_m":                d_tren,
        "dist_trole_m":               d_trole,
        "dist_cable_m":               d_cable,
        "dist_area_verde_m":          d_verde,
        "dist_salud_m":               d_salud,
        "dist_escuela_m":             d_escuela,
        "dist_comercio_m":            d_comercio,
        "dist_ciclovia_m":            d_ciclovia,
        "acceso_salud_15m":           float(n_salud),
        "acceso_educacion_15m":       float(n_escuela),
        "densidad_parques_15m":       float(n_parques),
        "densidad_ciclovia_15m":      float(n_ciclovia),
        "prox_parque":                float(prox_parque),
        "prox_salud":                 float(prox_salud),
        "prox_escuela":               float(prox_escuela),
        "prox_ciclovia":              float(prox_ciclovia),
        "score_15min":                score,
    }

# ============================================================
# INFERENCIA — PREDICCIÓN DE RENTA MENSUAL (sin leakage)
# ============================================================
def predict_renta_con_punto(area, rooms, baths, parking, ant, colonia_key, lat, lon):
    # 1. Datos base de la colonia (medias históricas de todas las features estáticas)
    datos_colonia = REF_ESPACIAL.get(colonia_key, list(REF_ESPACIAL.values())[0]).copy()
    # Añadir features de texto (promedio de la colonia)
    texto_colonia = REF_TEXTO.get(colonia_key, {})
    for k, v in texto_colonia.items():
        datos_colonia[k] = v

    # 2. Precio m2 mediano de la colonia (para determinar segmento lujo/estándar)
    median_m2 = PRICE_M2_COL.get(colonia_key, 0)
    lux_global_cut = np.percentile(list(PRICE_M2_COL.values()), 90)
    seg_label = "lujo" if median_m2 >= lux_global_cut else "estandar"

    # 3. Recalcular distancias desde el punto exacto (lat, lon)
    distancias_nuevas = recalcular_distancias_desde_punto(lat, lon)
    datos_colonia.update(distancias_nuevas)

    # 4. Interacciones y variables derivadas
    ant_log = np.log1p(ant)
    gentrif = datos_colonia.get("gentrification_index", 0.5)
    d_metro = distancias_nuevas.get("dist_metro_m", 9999)
    density_metro = max(0.0, 1.0 - d_metro / 1200.0) if d_metro < 1200 else 0.0

    datos_colonia.update({
        "area": area,
        "rooms": rooms,
        "bathrooms": baths,
        "parking_spaces": parking,
        "antiguedad": ant_log,
        "latitud": lat,
        "longitud": lon,
        "area_x_marginalidad": np.log1p(area) * datos_colonia.get("marginalidad_score", 3),
        "area_X_gentrif": area * gentrif,
        "gentrif_x_metro": gentrif * density_metro,
        "15min_X_gentrif": distancias_nuevas["score_15min"] * gentrif,
        "verde_marginalidad_ratio": np.log1p(datos_colonia.get("densidad_parques_15m", 0) / (datos_colonia.get("marginalidad_score", 3) + 1)),
        "salud_marginalidad_ratio": np.log1p(datos_colonia.get("acceso_salud_15m", 0) / (datos_colonia.get("marginalidad_score", 3) + 1)),
        "educacion_marginalidad_ratio": np.log1p(datos_colonia.get("acceso_educacion_15m", 0) / (datos_colonia.get("marginalidad_score", 3) + 1)),
    })

    # 5. Asignación de clúster (solo para estándar; lujo tiene cluster fijo 0)
    if seg_label == "estandar" and SEGMENTADORES.get("estandar") is not None:
        seg = SEGMENTADORES["estandar"]
        precio_suavizado = median_m2
        dist_subcenter = datos_colonia.get("dist_subcenter_log", np.log1p(5000))
        marginalidad = datos_colonia.get("marginalidad_score", 3)
        gentrificacion = gentrif
        colonia_enc = datos_colonia.get("colonia_price_enc", 0)
        pca_input = np.array([[precio_suavizado, dist_subcenter, marginalidad, gentrificacion, colonia_enc]])
        pca_input_scaled = seg["pca_scaler"].transform(pca_input)
        scores = seg["pca"].transform(pca_input_scaled)
        coords_xy = np.array([[lat, lon]])
        geo_scaled = seg["geo_scaler"].transform(coords_xy)
        X_cluster = np.hstack([scores * 0.6, geo_scaled * 0.4])
        cluster_raw = seg["km"].predict(X_cluster)[0]
        cluster = seg["remap"].get(cluster_raw, 0)
    else:
        cluster = 0
        seg_label = "lujo"

    # 6. Obtener modelo y features
    if (seg_label, cluster) not in MODELS:
        candidates = [k for k in MODELS if k[0] == seg_label]
        if not candidates:
            seg_label, cluster = list(MODELS.keys())[0]
        else:
            seg_label, cluster = candidates[0]

    model = MODELS[(seg_label, cluster)]
    imputador, feat_disp = IMPUTERS[(seg_label, cluster)]

    # 7. Construir dataframe de entrada
    input_df = pd.DataFrame([datos_colonia])

    # Calcular spatial lag y precio vecinal usando todas las colonias como referencia
    all_col_keys = list(REF_ESPACIAL.keys())
    all_lons = np.array([REF_ESPACIAL[k].get("longitud", 0) for k in all_col_keys])
    all_lats = np.array([REF_ESPACIAL[k].get("latitud", 0) for k in all_col_keys])
    all_prices = np.array([PRICE_M2_COL.get(k, 0) for k in all_col_keys])

    gdf_all = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(all_lons, all_lats), crs="EPSG:4326"
    ).to_crs("EPSG:32614")
    all_coords_utm = np.array([(g.x, g.y) for g in gdf_all.geometry])

    gdf_pt = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326"
    ).to_crs("EPSG:32614")
    pt_utm = np.array([(gdf_pt.geometry.iloc[0].x, gdf_pt.geometry.iloc[0].y)])

    lag_val = compute_spatial_lag(all_coords_utm, all_prices, pt_utm, k=30)[0]
    nvp_val = compute_local_neighbor_price(all_coords_utm, all_prices, pt_utm, k=15)[0]

    input_df["spatial_lag_price"] = lag_val
    input_df["precio_vecinal_local"] = nvp_val
    input_df["lag_x_area"] = lag_val * np.log1p(area)

    for f in feat_disp:
        if f not in input_df.columns:
            input_df[f] = 0.0

    X_pred = input_df[feat_disp].fillna(imputador).fillna(0)

    scaler = model.named_steps["scaler"]
    Xz = scaler.transform(X_pred)
    Xz = np.clip(Xz, -5, 5)
    X_pred_safe = pd.DataFrame(scaler.inverse_transform(Xz), columns=feat_disp, index=X_pred.index)

    log_pred = np.clip(model.predict(X_pred_safe)[0], np.log1p(200), np.log1p(200_000))
    precio_m2_pred = np.expm1(log_pred)
    renta_mensual = precio_m2_pred * area

    intercepto_log = model.named_steps["enet"].intercept_
    precio_base_m2 = np.expm1(intercepto_log)
    precio_base_total = precio_base_m2 * area

    return (renta_mensual, precio_m2_pred, seg_label, cluster,
            model, X_pred_safe, datos_colonia, precio_base_total, precio_base_m2, distancias_nuevas)

# ============================================================
# CONTROLES DE ENTRADA (SIDEBAR)
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1.5rem 0; border-bottom:2px solid var(--border-color);
                margin-bottom:1.5rem;">
        <div style="font-size:3rem; margin-bottom:0.5rem;">🏘️</div>
        <h2 style="margin:0; color:var(--text-color); font-size:1.25rem; font-weight:800;">
            Parámetros de la Renta
        </h2>
        <p style="color:var(--text-color); opacity:0.6; font-size:0.875rem; margin-top:0.5rem;">
            Personaliza tu valuación de renta
        </p>
    </div>
    """, unsafe_allow_html=True)

    alcaldia_sel = st.selectbox("Alcaldía", ALCALDIAS_DISPONIBLES)
    colonias_filtradas = ALCALDIA_COLONIAS.get(alcaldia_sel, [])
    colonia_sel = st.selectbox("Colonia", colonias_filtradas, key=f"colonia_{alcaldia_sel}")
    colonia_key_sel = f"{alcaldia_sel}||{colonia_sel}"

    area = st.slider("Metros Cuadrados (m²):", min_value=20, max_value=500, value=70, step=5)
    rooms   = st.number_input("Recámaras", min_value=1, max_value=10, value=2)
    baths   = st.number_input("Baños Completos", min_value=1.0, max_value=10.0, value=1.0, step=0.5)
    parking = st.number_input("Espacios de Estacionamiento", min_value=0, max_value=10, value=1)
    antiguedad = st.slider("Antigüedad de la Estructura (Años)", 0, 80, 10)

    st.markdown("---")
    st.caption("📍 Usa la herramienta de marcador (🔵) en el mapa para elegir un punto específico dentro de tu colonia y confirma con el botón.")

# ============================================================
# ESTADO DE SESIÓN Y PREDICCIÓN
# ============================================================
_key_lat        = f"marker_lat_{colonia_key_sel}"
_key_lon        = f"marker_lon_{colonia_key_sel}"
_key_confirmado = f"punto_confirmado_{colonia_key_sel}"
_key_dist       = f"distancias_punto_{colonia_key_sel}"

# ============================================================
# COORDENADAS BASE — cadena de fallback de 5 niveles
# ============================================================
# Nivel 1: centroide del shapefile catastral (más confiable geográficamente)
# Nivel 2: coords enriquecidas en REF_ESPACIAL (pueden venir del shapefile via preparar_sistema)
# Nivel 3: centroide del polígono de POLIGONOS_COLONIAS disponible en memoria
# Nivel 4: promedio de colonias vecinas de la misma alcaldía con coords válidas
# Nivel 5: Centro Histórico CDMX como último recurso

_CDMX_LAT_MIN, _CDMX_LAT_MAX = 19.04, 19.60
_CDMX_LON_MIN, _CDMX_LON_MAX = -99.37, -98.94

def _coord_valida(lat, lon):
    return (_CDMX_LAT_MIN <= float(lat or 0) <= _CDMX_LAT_MAX and
            _CDMX_LON_MIN <= float(lon or 0) <= _CDMX_LON_MAX)

lat_base, lon_base = None, None

# Nivel 1: centroide del shapefile (ya disponible en SHAPEFILE_CENTROIDS)
if colonia_key_sel in SHAPEFILE_CENTROIDS:
    _sl, _so = SHAPEFILE_CENTROIDS[colonia_key_sel]
    if _coord_valida(_sl, _so):
        lat_base, lon_base = _sl, _so

# Nivel 2: REF_ESPACIAL (ya enriquecido con centroides en preparar_sistema)
if lat_base is None:
    _ref_col = REF_ESPACIAL.get(colonia_key_sel, {})
    _rl = float(_ref_col.get("latitud", 0) or 0)
    _ro = float(_ref_col.get("longitud", 0) or 0)
    if _coord_valida(_rl, _ro):
        lat_base, lon_base = _rl, _ro

# Nivel 3: centroide del polígono cargado en POLIGONOS_COLONIAS
if lat_base is None:
    _poly_geojson = POLIGONOS_COLONIAS.get(colonia_key_sel)
    if _poly_geojson is None:
        # Intentar match aproximado por nombre de colonia limpia
        _col_clean = colonia_key_sel.split("||")[-1] if "||" in colonia_key_sel else colonia_key_sel
        for _pk, _pv in POLIGONOS_COLONIAS.items():
            if _pk.endswith(f"||{_col_clean}"):
                _poly_geojson = _pv
                break
    if _poly_geojson is not None:
        try:
            from shapely.geometry import shape as _shape_init
            _poly = _shape_init(_poly_geojson)
            _c = _poly.centroid
            if _coord_valida(_c.y, _c.x):
                lat_base, lon_base = _c.y, _c.x
        except Exception:
            pass

# Nivel 4: promedio de colonias de la misma alcaldía con coords válidas
if lat_base is None:
    _colonias_alcaldia = [
        k for k in REF_ESPACIAL
        if k.startswith(f"{alcaldia_sel}||")
        and _coord_valida(REF_ESPACIAL[k].get("latitud", 0), REF_ESPACIAL[k].get("longitud", 0))
    ]
    if not _colonias_alcaldia:
        # Buscar también en SHAPEFILE_CENTROIDS para la alcaldía
        _colonias_alcaldia_shp = [
            k for k in SHAPEFILE_CENTROIDS
            if k.startswith(f"{alcaldia_sel}||")
        ]
        if _colonias_alcaldia_shp:
            lat_base = float(np.mean([SHAPEFILE_CENTROIDS[k][0] for k in _colonias_alcaldia_shp]))
            lon_base = float(np.mean([SHAPEFILE_CENTROIDS[k][1] for k in _colonias_alcaldia_shp]))
    else:
        lat_base = float(np.mean([float(REF_ESPACIAL[k]["latitud"]) for k in _colonias_alcaldia]))
        lon_base = float(np.mean([float(REF_ESPACIAL[k]["longitud"]) for k in _colonias_alcaldia]))

# Nivel 5: Centro Histórico CDMX
if lat_base is None or not _coord_valida(lat_base, lon_base):
    lat_base, lon_base = 19.4326, -99.1332

if _key_lat not in st.session_state:
    st.session_state[_key_lat] = lat_base
if _key_lon not in st.session_state:
    st.session_state[_key_lon] = lon_base
if _key_confirmado not in st.session_state:
    st.session_state[_key_confirmado] = False
if _key_dist not in st.session_state:
    st.session_state[_key_dist] = None

lat_marcador = st.session_state[_key_lat]
lon_marcador = st.session_state[_key_lon]
punto_confirmado = st.session_state[_key_confirmado]
distancias_confirmadas = st.session_state[_key_dist]

# Sanitizar: si el marcador guardado está fuera del bbox de CDMX, resetear al base válido
if not _coord_valida(lat_marcador, lon_marcador):
    lat_marcador = lat_base
    lon_marcador = lon_base
    st.session_state[_key_lat] = lat_base
    st.session_state[_key_lon] = lon_base
    st.session_state[_key_confirmado] = False
    st.session_state[_key_dist] = None
    punto_confirmado = False
    distancias_confirmadas = None

# Siempre predecir usando la posición actual del marcador (evita el bug de valor estable
# causado por hacer primero una predicción base con lat_base/lon_base y luego sobreescribirla)
(renta, precio_m2, seg_label, cluster_id,
 modelo_fit, x_input, datos_colonia,
 precio_base_total, precio_base_m2, distancias_nuevas) = predict_renta_con_punto(
    area, rooms, baths, parking, antiguedad, colonia_key_sel, lat_marcador, lon_marcador
)
distancias_confirmadas = distancias_nuevas
st.session_state[_key_dist] = distancias_confirmadas

datos_entorno = datos_colonia.copy()
if distancias_confirmadas:
    datos_entorno.update(distancias_confirmadas)

# ============================================================
# CUADRO DE MANDO PRINCIPAL
# ============================================================
tab_valuador, tab_atlas = st.tabs([
    "🏘️ Valuador de Rentas",
    "🗺️ Estudio Multidimensional del Mercado de Renta"
])

with tab_valuador:

    renta_fmt   = f"${renta:,.0f} MXN/mes"
    pm2_fmt     = f"${precio_m2:,.0f} MXN/m²"
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
            <div class="metrica-label">Renta Mensual Estimada</div>
            <div class="metrica-valor">{renta_fmt}</div>
        </div>
        <div class="metrica-card" style="border-top:3px solid #10B981;">
            <div class="metrica-label">Precio por m² / mes</div>
            <div class="metrica-valor">{pm2_fmt}</div>
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
    # PANEL DE ANÁLISIS DE ENTORNO URBANO (5 bloques completos)
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
            with fila1_col2:
                acceso_edu = float(datos_entorno.get("acceso_educacion_15m", 0))
                st.metric("📚 Planteles Educativos", f"{acceso_edu:.0f} Escuelas")
                if acceso_edu >= 40:   st.success("Alta oferta escolar.")
                elif acceso_edu >= 15: st.info("Infraestructura escolar suficiente.")
                else:                  st.warning("Disponibilidad local limitada.")
            with fila2_col1:
                parque = float(datos_entorno.get("dist_area_verde_m", 9999))
                dens_parques = float(datos_entorno.get("densidad_parques_15m", 0))
                st.metric("🌳 Espacio Público Recreativo", fmt_dist(parque))
                if parque <= 300:   st.success("Radio óptimo de proximidad.")
                elif parque <= 800: st.info("Distancia media de acceso.")
                else:               st.warning("Déficit de áreas verdes.")
                st.caption(f"Aprox. {dens_parques:.0f} espacios verdes detectados.")
            with fila2_col2:
                comercio_dist = float(datos_entorno.get("dist_comercio_m", 5000))
                st.metric("🛍️ Centros de Abasto / Comercio", fmt_dist(comercio_dist))
                if comercio_dist <= 600:   st.success("Abasto local inmediato.")
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
                    if ciclovias >= 5.0:        status_bici = "Excelente"
                    elif ciclovias >= 1.5:      status_bici = "Funcional"
                    elif ciclovias > 0:         status_bici = "Escasa"
                    else:                       status_bici = "Ninguna"
                    st.metric("🚲 Infraestructura Ciclista", status_bici,
                              help=f"{ciclovias:.1f} segmentos en radio 15 min.")

            with c2_derecha:
                min_dist = min([d_metro, d_mb, d_tren, d_trole])
                st.markdown("<b style='font-size:14px;color:var(--text-color);opacity:0.6;'>Evaluación Multimodal</b>", unsafe_allow_html=True)
                if min_dist <= 500:    st.success("🟢 **Conectividad Excelente**")
                elif min_dist <= 1000: st.info("🔵 **Accesibilidad Media**")
                else:                  st.warning("🟡 **Cobertura Restringida**")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Bloque 3: Gentrificación ---
        encabezado_seccion("3. Transformación Socioespacial",
                           "Análisis de gentrificación y presión inmobiliaria", "🏙️")
        with st.container(border=True):
            c3_1, c3_2, c3_3 = st.columns(3)
            gentrif_macro = float(datos_colonia.get("gentrification_index", 0))
            gentrif_micro = float(datos_colonia.get("gentrif_local_presion", 0))
            gentrif_map   = float(datos_colonia.get("gentrif_map_score", 0))
            with c3_1:
                st.metric("🏛️ Cambio Estructural (Alcaldía)", round(gentrif_macro, 3))
                st.progress(min(gentrif_macro, 1.0))
                st.caption("Deltas socioeconómicos intercensales (INEGI 2010–2020)")
            with c3_2:
                st.metric("🏘️ Presión de Mercado Local", round(gentrif_micro, 3))
                st.progress(min(gentrif_micro, 1.0))
                st.caption("Variación del precio de renta respecto al entorno continuo")
            with c3_3:
                st.metric("🌆 Vulnerabilidad al Desplazamiento", round(gentrif_map, 3))
                st.progress(min(gentrif_map, 1.0))
                if gentrif_map >= 0.7:   st.warning("Proceso acelerado de encarecimiento de renta.")
                elif gentrif_map >= 0.4: st.info("Área en transición con presión moderada.")
                else:                    st.success("Estabilidad relativa en el mercado de renta.")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Bloque 4: Variables Catastrales ---
        encabezado_seccion("4. Contexto Catastral",
                           "Variables del entorno inmediato (radio 200 m)", "🗂️")
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
                st.metric("📊 Renta base del submercado", f"${precio_base_total:,.0f} MXN/mes",
                          help="Renta estimada para una vivienda completamente promedio dentro del clúster.")
                delta_vs_base = renta - precio_base_total
                if delta_vs_base >= 0:
                    st.success(f"▲ +${delta_vs_base:,.0f} MXN sobre la renta base")
                else:
                    st.warning(f"▼ ${delta_vs_base:,.0f} MXN bajo la renta base")
            st.caption("Fuente: Catastro CDMX 2021 · Radio de análisis: 200 m.")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Bloque 5: Estructura Urbana ---
        encabezado_seccion("5. Estructura Urbana y Desempeño del Mercado de Renta",
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
                st.metric("🏢 Densidad de Oferta de Renta", f"{listing:.2f} (Log)")
                if listing >= 3:     st.success("Mercado con alta oferta de renta.")
                elif listing >= 1.5: st.info("Actividad de mercado regular.")
                else:                st.warning("Baja oferta de renta disponible.")
                st.caption("Densidad logarítmica de listados de renta activos.")
            with c4_3:
                centralidad_log = float(datos_colonia.get("dist_subcenter_log", 0))
                centralidad_m   = np.expm1(centralidad_log)
                st.metric("📍 Proximidad a Nodos de Empleo", fmt_dist(centralidad_m))
                if centralidad_m <= 1000:   st.success("Alta centralidad económica.")
                elif centralidad_m <= 3000: st.info("Distancia funcional a subcentros.")
                else:                       st.warning("Ubicación periférica o habitacional.")
                st.caption("Subcentros: Centro, Polanco, Santa Fe, Insurgentes, Del Valle, Reforma.")

    # ============================================================
    # INTERPRETABILIDAD HEDÓNICA
    # ============================================================
    st.markdown("### 💰 Elasticidad y Aportación de Atributos a la Renta")

    NOMBRES_VARIABLES = {
        "rooms": "Recámaras adicionales",
        "bathrooms": "Baños completos",
        "parking_spaces": "Espacios de estacionamiento",
        "area": "Metros cuadrados de construcción",
        "antiguedad": "Años de antigüedad (log)",
        "dist_metro_m": "Proximidad STC Metro (log)",
        "density_metro": "Concentración local de accesos al Metro",
        "dist_metrobus_m": "Proximidad Metrobús (log)",
        "density_metrobus": "Concentración de accesos al Metrobús",
        "dist_tren_m": "Proximidad Tren Ligero (log)",
        "density_tren": "Densidad Tren Ligero",
        "dist_trole_m": "Proximidad Trolebús (log)",
        "density_trole": "Densidad Trolebús",
        "dist_cable_m": "Proximidad Cablebús (log)",
        "density_cable": "Densidad Cablebús",
        "dist_ciclovia_m": "Distancia a red de ciclovías (log)",
        "densidad_ciclovia_15m": "Densidad infraestructura ciclista",
        "dist_area_verde_m": "Distancia a área verde (log)",
        "densidad_parques_15m": "Densidad de parques en 15 min",
        "dist_salud_m": "Distancia a equipamiento médico (log)",
        "acceso_salud_15m": "Acceso a salud en 15 min",
        "dist_escuela_m": "Distancia a centros educativos (log)",
        "acceso_educacion_15m": "Acceso a educación en 15 min",
        "comercio_density": "Intensidad de comercios regionales (log)",
        "marginalidad_score": "Grado de marginación social",
        "score_15min": "Puntaje 'Ciudad de 15 Minutos'",
        "dist_subcenter_log": "Distancia a distritos de empleo (log)",
        "gentrification_index": "Nivel de gentrificación (alcaldía)",
        "listing_density_log": "Presión del inventario de renta",
        "pct_migrantes_turistas": "% Migrantes y turistas en la alcaldía",
        "area_x_marginalidad": "Superficie × Marginación",
        "area_X_gentrif": "Superficie × Gentrificación",
        "gentrif_x_metro": "Gentrificación × Densidad Metro",
        "15min_X_gentrif": "Ciudad 15 Min × Gentrificación",
        "verde_marginalidad_ratio": "Parques relativo a marginación",
        "salud_marginalidad_ratio": "Salud relativo a marginación",
        "educacion_marginalidad_ratio": "Educación relativo a marginación",
        "cat_mean_vus_200m": "Valor unitario suelo catastral",
        "cat_mean_valor_suelo_200m": "Valor catastral suelo promedio",
        "cat_mean_antiguedad_200m": "Antigüedad catastral promedio",
        "cat_std_valor_suelo_200m": "Varianza catastral del suelo",
        "cat_mean_ratio_construccion_200m": "Ratio construcción/terreno catastral",
        "cat_density_predios_200m": "Densidad predios catastro 200 m",
        "spatial_lag_price": "Lag espacial de precios de renta",
        "precio_vecinal_local": "Precio vecinal local de renta",
        "lag_x_area": "Lag espacial × Superficie",
        "colonia_price_enc": "Target encoding de colonia",
        "text_svd_0": "Componente textual 0",
        "text_svd_1": "Componente textual 1",
        "text_svd_2": "Componente textual 2",
        "text_svd_3": "Componente textual 3",
        "text_svd_4": "Componente textual 4",
    }

    enet_model = modelo_fit.named_steps["enet"]
    coeficientes = enet_model.coef_
    feat_names = x_input.columns.tolist()

    impactos_pesos = []
    for idx, var in enumerate(feat_names):
        val = float(x_input[var].values[0]) if var in x_input.columns else 0.0
        beta = coeficientes[idx] if idx < len(coeficientes) else 0.0
        ref = x_input[var].values[0] if x_input[var].values[0] != 0 else 1
        impacto = renta * (np.exp(beta * (val / ref)) - 1)
        if var in ["area", "area_X_gentrif"] and abs(impacto) > renta:
            impacto = np.sign(impacto) * (renta * 0.4)
        impactos_pesos.append(impacto)

    datos_impacto = pd.DataFrame({"Variable_Interna": feat_names, "Impacto_Pesos": impactos_pesos})
    datos_impacto["Característica"] = datos_impacto["Variable_Interna"].map(NOMBRES_VARIABLES).fillna(datos_impacto["Variable_Interna"])

    UMBRAL_IMPACTO = 100  # MXN/mes (ajustado para rentas)
    datos_filtrados = datos_impacto[datos_impacto["Impacto_Pesos"].abs() >= UMBRAL_IMPACTO].copy().sort_values("Impacto_Pesos", ascending=True)

    if not datos_filtrados.empty:
        colores = ["#EF553B" if v < 0 else "#00CC96" for v in datos_filtrados["Impacto_Pesos"]]
        fig_impacto = go.Figure()
        fig_impacto.add_trace(go.Bar(
            y=datos_filtrados["Característica"],
            x=datos_filtrados["Impacto_Pesos"],
            orientation="h",
            marker_color=colores,
            text=datos_filtrados["Impacto_Pesos"].apply(lambda x: f"${x:,.0f} MXN" if x >= 0 else f"-${abs(x):,.0f} MXN"),
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Aportación: %{x:$,.2f} MXN/mes<extra></extra>"
        ))
        fig_impacto.update_layout(
            title=f"<b>Factores Dominantes en la Formación de la Renta</b><br><span style='font-size:12px;color:gray;'>Impacto absoluto > ${UMBRAL_IMPACTO:,} MXN/mes</span>",
            xaxis_title="Impacto Neto sobre la Renta Estimada ($ MXN/mes)",
            yaxis_title="",
            margin=dict(l=25, r=80, t=70, b=25),
            height=len(datos_filtrados) * 38 + 110,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=True, zerolinecolor="rgba(128,128,128,0.4)")
        )
        st.plotly_chart(fig_impacto, use_container_width=True)
        st.info("💡 **Guía de Lectura:** Barras **verdes** → atributos que incrementan la renta mensual. Barras **rojas** → penalizaciones por obsolescencia o déficit de conectividad.")
    else:
        st.warning(f"No se identificaron variables con impacto superior a ${UMBRAL_IMPACTO:,} MXN/mes en esta configuración.")

    # ============================================================
    # MAPA DE CONTEXTO
    # ============================================================
    import json

    RADIO_M = 1200

    CAPAS_CONFIG = {
        "metro": {"label": "🚇 STC Metro", "color": "blue", "icono": "train", "prefix": "fa", "max_puntos": 30},
        "metrobus": {"label": "🚌 Metrobús", "color": "red", "icono": "bus", "prefix": "fa", "max_puntos": 30},
        "tren": {"label": "🚊 Tren Ligero", "color": "cadetblue", "icono": "subway", "prefix": "fa", "max_puntos": 20},
        "trolebus": {"label": "🚎 Trolebús", "color": "purple", "icono": "bolt", "prefix": "fa", "max_puntos": 30},
        "cablebus": {"label": "🚠 Cablebús", "color": "darkblue", "icono": "cloud", "prefix": "fa", "max_puntos": 20},
        "parques": {"label": "🌳 Áreas Verdes", "color": "green", "icono": "leaf", "prefix": "fa", "max_puntos": 25},
        "salud": {"label": "🏥 Salud", "color": "darkred", "icono": "plus-square", "prefix": "fa", "max_puntos": 20},
        "comercio": {"label": "🛍️ Comercio", "color": "orange", "icono": "shopping-cart", "prefix": "fa", "max_puntos": 15},
        "esc_privada": {"label": "🏫 Esc. Privadas", "color": "beige", "icono": "graduation-cap", "prefix": "fa", "max_puntos": 20},
        "esc_publica": {"label": "🏫 Esc. Públicas", "color": "darkgreen", "icono": "graduation-cap", "prefix": "fa", "max_puntos": 20},
        "ciclovias": {"label": "🚲 Ciclovías", "color": "teal", "icono": "road", "prefix": "fa", "max_puntos": 20},
    }

    _COLOR_HEX = {
        "blue":      "#1A73E8",
        "red":       "#E53935",
        "cadetblue": "#5F9EA0",
        "purple":    "#8E24AA",
        "darkblue":  "#1565C0",
        "green":     "#43A047",
        "darkred":   "#C62828",
        "orange":    "#FB8C00",
        "beige":     "#8D6E63",
        "darkgreen": "#2E7D32",
        "teal":      "#00796B",
    }

    def _latlon_a_utm_mapa(lat, lon):
        try:
            gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326").to_crs("EPSG:32614")
            x, y = gdf.geometry.iloc[0].x, gdf.geometry.iloc[0].y
            if not (np.isfinite(x) and np.isfinite(y)):
                raise ValueError("Coordenadas UTM inválidas")
            return x, y
        except Exception:
            # Fallback: usar Centro Histórico CDMX en UTM zona 14N
            return 485340.0, 2145050.0

    def _utm_a_latlon_mapa(pts_utm):
        if len(pts_utm) == 0:
            return []
        # Filtrar puntos con NaN o infinito antes de proyectar
        _mask_valid = np.isfinite(pts_utm).all(axis=1)
        pts_utm = pts_utm[_mask_valid]
        if len(pts_utm) == 0:
            return []
        gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pts_utm[:, 0], pts_utm[:, 1]), crs="EPSG:32614").to_crs("EPSG:4326")
        return [(g.y, g.x) for g in gdf.geometry]

    poligono_colonia_geojson = POLIGONOS_COLONIAS.get(colonia_key_sel)

    # Si no hay polígono exacto, intentar búsqueda aproximada por colonia limpia
    if poligono_colonia_geojson is None:
        _colonia_clean_sel = colonia_key_sel.split("||")[-1] if "||" in colonia_key_sel else colonia_key_sel
        for _k, _v in POLIGONOS_COLONIAS.items():
            if _k.endswith(f"||{_colonia_clean_sel}"):
                poligono_colonia_geojson = _v
                break

    # ── Guardia final: asegurar coordenadas válidas para folium.Map ──
    # Si lat_marcador/lon_marcador cayeron fuera del bbox por cualquier razón,
    # derivar las coordenadas de mapa del polígono disponible o del shapefile.
    _map_lat, _map_lon = lat_marcador, lon_marcador
    if not _coord_valida(_map_lat, _map_lon):
        # Intentar polígono cargado
        if poligono_colonia_geojson is not None:
            try:
                from shapely.geometry import shape as _sg
                _pc = _sg(poligono_colonia_geojson).centroid
                if _coord_valida(_pc.y, _pc.x):
                    _map_lat, _map_lon = _pc.y, _pc.x
            except Exception:
                pass
        # Intentar shapefile_centroids
        if not _coord_valida(_map_lat, _map_lon) and colonia_key_sel in SHAPEFILE_CENTROIDS:
            _map_lat, _map_lon = SHAPEFILE_CENTROIDS[colonia_key_sel]
        # Fallback a lat_base calculado arriba
        if not _coord_valida(_map_lat, _map_lon):
            _map_lat, _map_lon = lat_base, lon_base

    m = folium.Map(location=[_map_lat, _map_lon], zoom_start=15, tiles="OpenStreetMap")

    # ── Polígono de la colonia ──
    if poligono_colonia_geojson:
        try:
            # Validar la geometría antes de enviársela a folium
            from shapely.geometry import shape as _sg_val, mapping as _sg_map
            _poly_val = _sg_val(poligono_colonia_geojson)
            if not _poly_val.is_valid:
                _poly_val = _poly_val.buffer(0)   # reparar anillos rotos
            _poly_geojson_clean = _sg_map(_poly_val)
            grupo_colonia = folium.FeatureGroup(name="🔵 Límite de colonia", show=True)
            folium.GeoJson(
                _poly_geojson_clean,
                style_function=lambda _: {"color": "#1A73E8", "weight": 2,
                                          "fillColor": "#1A73E8", "fillOpacity": 0.06,
                                          "dashArray": "6 4"},
                tooltip=f"Límite de colonia: {colonia_sel.title()}"
            ).add_to(grupo_colonia)
            grupo_colonia.add_to(m)
        except Exception as _poly_err:
            # Si el polígono falla, mostrar círculo indicativo
            folium.Circle(
                location=[_map_lat, _map_lon], radius=600,
                color="#1A73E8", weight=1.5, fill=True,
                fill_color="#1A73E8", fill_opacity=0.04,
                tooltip=f"Colonia: {colonia_sel.title()} (límite no disponible)"
            ).add_to(m)
    else:
        folium.Circle(location=[_map_lat, _map_lon], radius=600,
                      color="#1A73E8", weight=1.5, fill=True,
                      fill_color="#1A73E8", fill_opacity=0.04,
                      tooltip=f"Colonia: {colonia_sel.title()} (sin límite oficial disponible)").add_to(m)

    folium.Circle(location=[_map_lat, _map_lon], radius=RADIO_M, color="#E53935", weight=2,
                  fill=True, fill_color="#E53935", fill_opacity=0.05,
                  tooltip=f"Radio de análisis: {RADIO_M/1000:.1f} km | {'Confirmado' if punto_confirmado else 'Centroide'}").add_to(m)

    folium.Marker(location=[_map_lat, _map_lon],
                  icon=folium.DivIcon(html='<div style="font-size:36px;line-height:1;text-align:center;filter:drop-shadow(0 3px 6px rgba(0,0,0,0.55));user-select:none;">🏠</div>',
                                      icon_size=(40, 40), icon_anchor=(20, 36)),
                  tooltip="📍 Posición actual — usa el botón ✏️ para colocar un nuevo marcador",
                  popup=folium.Popup(f"<b>📍 Inmueble en renta</b><br>Colonia: <b>{colonia_sel.title()}</b><br>Renta estimada: <b>${renta:,.0f} MXN/mes</b>", max_width=260),
                  draggable=False).add_to(m)

    from folium.plugins import Draw
    Draw(draw_options={"marker": True, "polyline": False, "polygon": False, "circle": False, "rectangle": False, "circlemarker": False},
         edit_options={"edit": False, "remove": False}, position="topleft").add_to(m)

   # ── Forzado de ícono local exclusivo para la herramienta Draw ──
    _icon_css = folium.Element("""
    <style>
    /* 1. Ocultar los recursos de imagen .png externos SÓLO para las herramientas de dibujo y arrastre */
    .leaflet-draw-guide-dash ~ img,
    .leaflet-mouse-marker,
    .leaflet-draw-tooltip-marker ~ .leaflet-marker-icon,
    .leaflet-pane .leaflet-marker-icon[src*="marker-icon"] {
        background-image: none !important;
        content: "" !important;
        opacity: 0 !important;
    }

    /* 2. Activar la visualización de capas editables y el cursor fantasma de dibujo */
    .leaflet-mouse-marker,
    .leaflet-draw-tooltip-marker ~ .leaflet-marker-icon {
        opacity: 1 !important;
        background: transparent !important;
        border: none !important;
    }

    /* 3. Inyectar el emoji ÚNICAMENTE en el cursor flotante (fantasma) antes de hacer click */
    .leaflet-mouse-marker::after,
    .leaflet-draw-tooltip-marker ~ .leaflet-marker-icon::after {
        content: "🏠" !important;
        font-size: 34px !important;
        line-height: 1 !important;
        display: block !important;
        text-align: center !important;
        filter: drop-shadow(0 2px 5px rgba(0,0,0,0.55)) !important;
        margin-top: -16px !important;
        margin-left: -10px !important;
        user-select: none;
        -webkit-user-select: none;
    }

    /* 4. IMPORTANTE: Ocultar el clon temporal de Leaflet.draw para que no se duplique */
    .leaflet-layer .leaflet-marker-icon:not(.leaflet-div-icon) {
        display: none !important;
    }
    </style>
    """)
    m.get_root().html.add_child(_icon_css)

    centro_utm = _latlon_a_utm_mapa(_map_lat, _map_lon)
    resumen_capas = {}

    for clave, (pts_utm, nombres) in CAPAS_SERVICIOS.items():
        cfg = CAPAS_CONFIG[clave]
        try:
            if len(pts_utm) == 0:
                resumen_capas[clave] = 0
                continue
            tree = cKDTree(pts_utm)
            idx = tree.query_ball_point(centro_utm, r=RADIO_M)
            if not idx:
                resumen_capas[clave] = 0
                continue
            pts_dentro = pts_utm[idx]
            nombres_dentro = [nombres[i] for i in idx] if nombres else []
            total = len(pts_dentro)
            resumen_capas[clave] = total
            max_p = cfg["max_puntos"]
            if total > max_p:
                step = max(1, total // max_p)
                pts_dentro = pts_dentro[::step][:max_p]
                nombres_dentro = nombres_dentro[::step][:max_p]
            coords_ll = _utm_a_latlon_mapa(pts_dentro)
            grupo = folium.FeatureGroup(name=cfg["label"], show=True)
            for i, (lt, ln) in enumerate(coords_ll):
                try:
                    # Validar coordenadas del punto antes de añadirlo
                    if not (_coord_valida(lt, ln)):
                        continue
                    nombre_punto = nombres_dentro[i] if i < len(nombres_dentro) else f"{cfg['label']} #{i+1}"
                    # Sanitizar caracteres que pueden romper JSON/HTML
                    nombre_punto = nombre_punto.replace('"', "'").replace('<', '').replace('>', '')
                    folium.Marker(
                        location=[lt, ln],
                        icon=folium.Icon(color=cfg["color"], icon=cfg["icono"], prefix=cfg["prefix"]),
                        tooltip=folium.Tooltip(nombre_punto, sticky=True),
                        popup=folium.Popup(
                            f"<b>{cfg['label']}</b><br>{nombre_punto}<br>"
                            f"<span style='font-size:10px;color:#888'>{lt:.5f}, {ln:.5f}</span>",
                            max_width=200
                        )
                    ).add_to(grupo)
                except Exception:
                    continue
            grupo.add_to(m)
        except Exception:
            resumen_capas[clave] = 0
            continue

    _cic_utm = CAPAS_LINEAS.get("ciclovias_utm", gpd.GeoDataFrame(geometry=[], crs="EPSG:32614"))
    _cic_wgs = CAPAS_LINEAS.get("ciclovias_wgs", gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
    if not _cic_utm.empty:
        try:
            _circulo_utm = Point(centro_utm[0], centro_utm[1]).buffer(RADIO_M)
            _mask = _cic_utm.intersects(_circulo_utm)
            _cic_wgs_dentro = _cic_wgs[_mask]
            if not _cic_wgs_dentro.empty:
                resumen_capas["ciclovias"] = len(_cic_wgs_dentro)
                grupo_lineas = folium.FeatureGroup(name="🚲 Ciclovías", show=True)
                for _, row in _cic_wgs_dentro.iterrows():
                    try:
                        if row.geometry is None or row.geometry.is_empty:
                            continue
                        _geom_clean = row.geometry if row.geometry.is_valid else row.geometry.buffer(0)
                        nombre = str(row.get("NOMBRE", "") or "").strip()
                        if not nombre or nombre in ("nan", "None"):
                            nombre = "Ciclovía sin nombre"
                        if len(nombre) > 50:
                            nombre = nombre[:47] + "…"
                        folium.GeoJson(
                            _geom_clean.__geo_interface__,
                            style_function=lambda feature: {"color": "#008080", "weight": 3, "opacity": 0.85},
                            tooltip=folium.Tooltip(f"🚲 {nombre}", sticky=True),
                            popup=folium.Popup(f"<b>Ciclovía</b><br>{nombre}", max_width=200)
                        ).add_to(grupo_lineas)
                    except Exception:
                        continue
                grupo_lineas.add_to(m)
            else:
                resumen_capas["ciclovias"] = 0
        except Exception:
            resumen_capas["ciclovias"] = 0
    else:
        resumen_capas["ciclovias"] = 0

    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    # Leyenda
    def _fila_leyenda(clave, cfg):
        color_hex = _COLOR_HEX[cfg['color']]
        label     = cfg['label']
        return (
            f"<div style='display:flex;align-items:center;margin-bottom:5px;'>"
            f"<span style='background:{color_hex};width:12px;height:12px;"
            f"border-radius:50%;display:inline-block;margin-right:8px;"
            f"border:1px solid rgba(255,255,255,0.25);flex-shrink:0;'></span>"
            f"<span style='font-size:11.5px;line-height:1.3;'>{label}</span></div>"
        )
    leyenda_filas = "".join(_fila_leyenda(k, v) for k, v in CAPAS_CONFIG.items())

    leyenda_html_simple = f"""
    <div id="leyenda-wrapper" style="
        position: fixed;
        bottom: 30px; left: 30px;
        z-index: 9999;
        font-family: 'Segoe UI', Arial, sans-serif;
    ">
        <button
            id="leyenda-toggle-btn"
            onclick="(function(){{
                var panel = document.getElementById('leyenda-panel');
                var btn   = document.getElementById('leyenda-toggle-btn');
                var oculto = panel.style.display === 'none';
                panel.style.display = oculto ? 'block' : 'none';
                btn.title = oculto ? 'Ocultar leyenda' : 'Mostrar leyenda';
            }})()\"
            title="Ocultar leyenda"
            style="
                width: 36px; height: 36px;
                border-radius: 8px;
                border: 1px solid rgba(255,255,255,0.2);
                background: rgba(30,41,59,0.97);
                color: #e2e8f0;
                font-size: 18px;
                cursor: pointer;
                display: flex; align-items: center; justify-content: center;
                box-shadow: 0 2px 8px rgba(0,0,0,0.4);
                margin-bottom: 6px;
                padding: 0;
                -webkit-tap-highlight-color: transparent;
            ">👁️</button>

        <div id="leyenda-panel" style="
            background: rgba(30,41,59,0.97);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 12px;
            padding: 10px 14px 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.40);
            min-width: 185px;
            color: #e2e8f0;
        ">
            <div style='font-weight:700;font-size:12.5px;margin-bottom:8px;color:#93C5FD;
                        border-bottom:2px solid #3B82F6;padding-bottom:5px;'>
                📍 Servicios en radio 1.2 km
            </div>
            {leyenda_filas}
            <div style='margin-top:8px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.1);
                        display:flex;align-items:center;gap:6px;font-size:11px;color:#94a3b8;'>
                <span style='display:inline-block;width:22px;height:0;
                             border-top:2px dashed #60A5FA;flex-shrink:0;'></span>
                Límite de colonia
            </div>
            <div style='display:flex;align-items:center;gap:6px;font-size:11px;color:#94a3b8;margin-top:4px;'>
                <span style='display:inline-block;width:22px;height:0;
                             border-top:2px solid #F87171;flex-shrink:0;'></span>
                Radio de análisis
            </div>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(leyenda_html_simple))

    # ── Encabezado del mapa ──
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin:1.2rem 0 0.5rem;">
        <span style="font-size:1.15rem;font-weight:700;letter-spacing:-0.2px;
                     color:var(--text-color);">🗺️ Mapa de Ubicación y Entorno Urbano</span>
        <span style="background:#1A73E8;color:#fff;font-size:0.7rem;font-weight:700;
                     padding:3px 11px;border-radius:20px;letter-spacing:0.04em;">RADIO 1.2 KM</span>
        <span style="font-size:0.72rem;color:var(--text-color);opacity:0.7;">
            — usa el botón 👁️ en el mapa para ocultar/mostrar la leyenda
        </span>
    </div>
    """, unsafe_allow_html=True)
    if punto_confirmado:
        st.success(
            f"✅ Punto confirmado en {lat_marcador:.5f}, {lon_marcador:.5f} — "
            f"distancias y renta recalculados."
        )
    else:
        st.info("📍 Usa el ícono ✏️ del panel izquierdo del mapa para colocar un marcador, luego confirma.")

    # Contadores de servicios
    iconos_txt = {
        "metro": "🚇", "metrobus": "🚌", "tren": "🚊", "trolebus": "🚎",
        "cablebus": "🚠", "parques": "🌳", "salud": "🏥", "comercio": "🛍️",
        "esc_privada": "🏫", "esc_publica": "🏫", "ciclovias": "🚲",
    }
    nombres_cortos = {
        "metro": "Metro", "metrobus": "Metrobús", "tren": "Tren", "trolebus": "Trolebús",
        "cablebus": "Cablebús", "parques": "Parques", "salud": "Salud", "comercio": "Comercio",
        "esc_privada": "Esc. Priv.", "esc_publica": "Esc. Púb.", "ciclovias": "Ciclovías",
    }

    badges_html = '<div class="servicios-grid">'
    for clave, n in resumen_capas.items():
        icono  = iconos_txt[clave]
        nombre = nombres_cortos[clave]
        color  = "#3B82F6" if n > 0 else "var(--text-color)"
        badges_html += (
            f'<div class="servicio-badge">'
            f'  <span>{icono}</span>'
            f'  <span style="font-size:0.7rem;opacity:0.7;">{nombre}</span>'
            f'  <span class="s-numero" style="color:{color};">{n}</span>'
            f'</div>'
        )
    badges_html += '</div>'
    st.markdown(badges_html, unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    with st.expander("🔍 Diagnóstico de ubicación (depuración)", expanded=False):
        st.write({
            "colonia_key": colonia_key_sel,
            "lat_marcador": lat_marcador,
            "lon_marcador": lon_marcador,
            "_map_lat (usado en mapa)": _map_lat,
            "_map_lon (usado en mapa)": _map_lon,
            "lat_base": lat_base,
            "lon_base": lon_base,
            "tiene_poligono": poligono_colonia_geojson is not None,
            "en_ref_espacial": colonia_key_sel in REF_ESPACIAL,
            "en_shapefile_centroids": colonia_key_sel in SHAPEFILE_CENTROIDS,
            "lat_en_ref": REF_ESPACIAL.get(colonia_key_sel, {}).get("latitud", "N/A"),
            "lon_en_ref": REF_ESPACIAL.get(colonia_key_sel, {}).get("longitud", "N/A"),
            "shapefile_centroid": SHAPEFILE_CENTROIDS.get(colonia_key_sel, "N/A"),
        })

    try:
        resultado_mapa = st_folium(
            m,
            use_container_width=True,
            height=500,
            returned_objects=["last_active_drawing"],
            key=f"mapa_{colonia_key_sel}_{st.session_state.get(f'recenter_{colonia_key_sel}', 0)}",
        )
    except Exception as _map_err:
        # Primer intento falló — construir mapa mínimo (solo marcador y radio)
        # para garantizar que siempre se muestra algo
        st.warning(f"⚠️ El mapa completo no pudo renderizarse ({type(_map_err).__name__}). "
                   f"Mostrando versión simplificada.")
        try:
            _m_min = folium.Map(location=[_map_lat, _map_lon], zoom_start=15,
                                tiles="OpenStreetMap")
            folium.Circle(
                location=[_map_lat, _map_lon], radius=RADIO_M,
                color="#E53935", weight=2, fill=True,
                fill_color="#E53935", fill_opacity=0.05,
                tooltip=f"Radio de análisis: {RADIO_M/1000:.1f} km"
            ).add_to(_m_min)
            folium.Marker(
                location=[_map_lat, _map_lon],
                icon=folium.DivIcon(
                    html='<div style="font-size:36px;line-height:1;text-align:center;'
                         'filter:drop-shadow(0 3px 6px rgba(0,0,0,0.55));">🏠</div>',
                    icon_size=(40, 40), icon_anchor=(20, 36)
                ),
                tooltip=colonia_sel.title(),
            ).add_to(_m_min)
            resultado_mapa = st_folium(
                _m_min,
                use_container_width=True,
                height=500,
                returned_objects=["last_active_drawing"],
                key=f"mapa_min_{colonia_key_sel}_{st.session_state.get(f'recenter_{colonia_key_sel}', 0)}",
            )
        except Exception as _map_err2:
            st.error(f"❌ No se pudo renderizar el mapa: {_map_err2}")
            st.write(f"Coordenadas activas: lat={_map_lat:.5f}, lon={_map_lon:.5f}")
            resultado_mapa = {}

    # Detectar nuevo marcador con validación de límite de colonia
    _drawn = resultado_mapa.get("last_active_drawing") if resultado_mapa else None

    if _drawn and isinstance(_drawn, dict):
        geom = _drawn.get("geometry", {})
        if geom.get("type") == "Point":
            coords = geom.get("coordinates", [])
            if len(coords) == 2:
                nuevo_lat, nuevo_lon = coords[1], coords[0]
                if abs(nuevo_lat - lat_marcador) > 1e-6 or abs(nuevo_lon - lon_marcador) > 1e-6:
                    # Verificar que el punto cae dentro del polígono de la colonia
                    poly_geojson = POLIGONOS_COLONIAS.get(colonia_key_sel)
                    _dentro = True
                    if poly_geojson is not None:
                        from shapely.geometry import Point as _Pt, shape as _shape
                        _poly = _shape(poly_geojson)
                        _dentro = _poly.contains(_Pt(nuevo_lon, nuevo_lat))

                    if _dentro:
                        st.session_state[_key_lat] = nuevo_lat
                        st.session_state[_key_lon] = nuevo_lon
                        st.session_state[_key_confirmado] = False
                        st.rerun()
                    else:
                        st.warning("⚠️ Posición fuera de la colonia — no se actualizó. Mueve el marcador dentro del límite azul y confirma.")

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("✅ Confirmar ubicación", use_container_width=True, type="primary"):
            st.session_state[_key_confirmado] = True
            st.session_state[_key_dist] = None  # Forzar recálculo de distancias
            st.rerun()
    with col_btn2:
        if st.button("↩️ Restablecer centroide", use_container_width=True):
            st.session_state[_key_lat] = lat_base
            st.session_state[_key_lon] = lon_base
            st.session_state[_key_confirmado] = False
            st.session_state[_key_dist] = None
            st.rerun()
    with col_btn3:
        if st.button("🎯 Recentrar en inmueble", use_container_width=True,
                     help="Vuelve el mapa al punto de análisis actual"):
            st.session_state[f"recenter_{colonia_key_sel}"] = (
                st.session_state.get(f"recenter_{colonia_key_sel}", 0) + 1
            )
            st.rerun()

    st.markdown(f"<div style='font-size:0.75rem;color:var(--text-color);opacity:0.6;margin-top:0.5rem;padding:0.5rem 0.75rem;background:var(--secondary-background-color);border-radius:8px;border:1px solid var(--border-color);'>📐 Radio de análisis: <b>{RADIO_M} m</b> &nbsp;·&nbsp; Estado: <b>{'✅ Confirmado' if punto_confirmado else '⏳ Pendiente'}</b> &nbsp;·&nbsp; Coordenadas: <b>{lat_marcador:.5f}, {lon_marcador:.5f}</b></div>", unsafe_allow_html=True)

# ============================================================
# PESTAÑA: ATLAS (igual que en valuador de compra/venta)
# ============================================================
with tab_atlas:
    st.markdown("""
    <div style="background: var(--secondary-background-color); border-radius: 16px; padding: 2.5rem 2.5rem 2rem; margin-bottom: 2rem; box-shadow: 0 4px 24px rgba(0,0,0,0.08); border: 1px solid var(--border-color); border-left: 6px solid #3B82F6;">
        <div style="color: var(--text-color); font-size: 1.8rem; font-weight: 800; margin: 0 0 0.5rem; letter-spacing: -0.5px; line-height: 1.2;">🌆 Estudio Multidimensional del Mercado de Renta CDMX</div>
        <p style="color: var(--text-color); opacity: 0.7; font-size: 1rem; margin: 0; line-height: 1.6;">Análisis espacial integral de las dinámicas sociales, económicas y urbanas en la Ciudad de México · Elaborado con QGIS y datos abiertos CDMX</p>
    </div>
    """, unsafe_allow_html=True)

    MAPAS_DIR = os.path.join(PROJECT_ROOT, "..", "outputs", "assets", "mapas_qgis")

    MAPAS_ATLAS = [
        {"archivo": "mapa_rentas_distribucion_del_precio_unitario_m2.png", "titulo": "Distribución del Precio de Renta por m²", "emoji": "💰", "descripcion": "Precio mediano de renta mensual por m² en la CDMX. Las mayores densidades de valor se concentran en Cuauhtémoc y Miguel Hidalgo, con gradientes hacia la periferia.", "categoria": "Mercado de Renta"},
        {"archivo": "mapa_rentas_densidad_de_oferta_inmobiliaria.png", "titulo": "Densidad de Oferta de Renta", "emoji": "🏗️", "descripcion": "Hotspots de concentración de anuncios de renta activos. Mayor densidad en zonas con alta gentrificación y conectividad de transporte.", "categoria": "Mercado de Renta"},
        {"archivo": "mapa_rentas_segmentacion_estandar_lujo.png", "titulo": "Clústeres Espaciales de Precios de Renta", "emoji": "🧩", "descripcion": "Segmentación K-Means de submercados de renta: Estándar y Lujo, con 3 clústeres geográficos cada uno. Identifica territorios de renta diferenciada.", "categoria": "Mercado de Renta"},
        {"archivo": "mapa_rentas_tipologias.png", "titulo": "Tipología de Vivienda en Renta", "emoji": "🏠", "descripcion": "Distribución territorial por tipo: Studio/Micro, Familiar, Amplio con/sin estacionamiento y Residencial de Lujo. Los studios predominan en colonias centrales.", "categoria": "Mercado de Renta"},
        {"archivo": "mapa_rentas_segmentacion_transporte.png", "titulo": "Premio por Accesibilidad al Transporte", "emoji": "🚇", "descripcion": "Diferencial de renta asociado a la accesibilidad al transporte público. La alta accesibilidad conlleva un premio de hasta +29.8% sobre la renta base.", "categoria": "Movilidad y Accesibilidad"},
        {"archivo": "mapa_rentas_gentrificacion.png", "titulo": "Índice de Gentrificación — Nivel Colonia", "emoji": "🏙️", "descripcion": "Coroplético del índice de gentrificación desagregado por colonia. Roma–Condesa, Polanco y Narvarte muestran los procesos más avanzados de encarecimiento de renta.", "categoria": "Dinámicas Sociales"},
        {"archivo": "mapa_rentas_socioeconomico.png", "titulo": "Cuadrante Socioeconómico (Marginalidad × Gentrificación)", "emoji": "📊", "descripcion": "Análisis bivariado que combina marginación social y nivel de gentrificación. Las colonias BajaMarg_AltaGentrif concentran la oferta de renta premium.", "categoria": "Dinámicas Sociales"},
        {"archivo": "mapa_rentas_15_min_nivel_colonia.png", "titulo": "Ciudad de 15 Minutos para Renta", "emoji": "🚶", "descripcion": "Puntaje de accesibilidad peatonal a servicios esenciales. Roma Norte, Del Valle y Narvarte destacan como los entornos más completos para vivir en renta.", "categoria": "Movilidad y Accesibilidad"},
        {"archivo": "mapa_rentas_residuos_espaciales.png", "titulo": "Análisis de Residuos del Modelo de Renta", "emoji": "📐", "descripcion": "Residuos del modelo de renta clasificados en cinco categorías. La distribución espacial de los residuos orienta la revisión de features y mejoras al modelo.", "categoria": "Validación del Modelo"},
        {"archivo": "mapa_rentas_precio_m2_mediana.png", "titulo": "Ranking de Renta por Alcaldía", "emoji": "🏆", "descripcion": "Mediana de precio por m² de renta por demarcación. Cuauhtémoc (454 $/m²) y Miguel Hidalgo (429 $/m²) lideran; Tlalpan registra la menor mediana (225 $/m²).", "categoria": "Mercado de Renta"},
    ]

    categorias = sorted(set(m["categoria"] for m in MAPAS_ATLAS))
    cat_sel = st.multiselect("🔍 Filtrar por categoría temática", options=categorias, default=categorias)
    mapas_filtrados = [m for m in MAPAS_ATLAS if m["categoria"] in cat_sel]

    if not mapas_filtrados:
        st.warning("Selecciona al menos una categoría para ver los mapas.")
    else:
        COLORES_CAT = {
            "Mercado de Renta": ("#0a1e3d", "#60a5fa"),
            "Análisis Espacial": ("#092416", "#34d399"),
            "Catastro": ("#291408", "#fb923c"),
            "Movilidad y Accesibilidad": ("#160a2d", "#c084fc"),
            "Dinámicas Sociales": ("#290a0a", "#f87171"),
            "Validación del Modelo": ("#071929", "#22d3ee"),
        }
        cols_galeria = st.columns(2, gap="large")
        for i, mapa in enumerate(mapas_filtrados):
            col = cols_galeria[i % 2]
            ruta_img = os.path.join(MAPAS_DIR, mapa["archivo"])
            _, accent = COLORES_CAT.get(mapa["categoria"], ("#111827", "#94a3b8"))
            with col:
                st.markdown(f"""
                <div style="background: var(--secondary-background-color); border: 1px solid var(--border-color); border-left: 5px solid {accent}; border-radius: 12px; padding: 1rem 1.2rem 0.9rem; margin-bottom: 0.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                    <span style="background: {accent}20; color: {accent}; font-size: 0.68rem; font-weight: 800; letter-spacing: 0.1em; padding: 3px 10px; border-radius: 20px; text-transform: uppercase; border: 1px solid {accent}40; display: inline-block; margin-bottom: 0.1rem;">{mapa['categoria']}</span>
                    <h3 style="color: var(--text-color); font-size: 1.05rem; font-weight: 700; margin: 0.55rem 0 0.35rem; line-height: 1.3;">{mapa['emoji']} {mapa['titulo']}</h3>
                    <p style="color: var(--text-color); opacity: 0.85; font-size: 0.84rem; margin: 0; line-height: 1.55;">{mapa['descripcion']}</p>
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists(ruta_img):
                    st.image(ruta_img, use_container_width=True)
                    st.markdown(f"<div style='text-align: center; font-size: 0.78rem; color: var(--text-color); opacity: 0.6; margin-top: -0.25rem; margin-bottom: 1rem;'>Mapa {i+1} de {len(mapas_filtrados)} · {mapa['titulo']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background: var(--secondary-background-color); border: 2px dashed var(--border-color); border-radius: 8px; padding: 3rem 1rem; text-align: center; color: var(--text-color); opacity: 0.6; font-size: 0.9rem; margin-bottom: 1rem;'>📁 <code>{mapa['archivo']}</code><br><small>Coloca el PNG exportado de QGIS en<br><code>outputs/assets/mapas_qgis_rentas/</code></small></div>", unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.caption(f"📊 {len(mapas_filtrados)} mapas mostrados · Fuentes: INEGI, ADIP CDMX, STC Metro, SEMOVI · Elaboración propia con QGIS y Python")

# ============================================================
# FOOTER PROFESIONAL
# ============================================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:2rem 0; border-top:1px solid var(--border-color); margin-top:3rem; background-color: transparent;">
    <div style="color: var(--text-color); opacity: 0.7; font-size:0.875rem; line-height:1.8;">
        <div style="font-weight:700; color: var(--text-color); margin-bottom:0.5rem; font-size: 1rem;">Sistema de Valuación de Rentas CDMX</div>
        <div>Escuela Superior de Cómputo • Instituto Politécnico Nacional</div>
        <div style="margin-top:0.5rem; font-weight: 600; color: var(--text-color);">Kevin J. González Sosa • José M. Torres Gutiérrez</div>
        <div style="font-size: 0.75rem; opacity: 0.6; margin-top: 0.25rem;">Trabajo Terminal · ESCOM 2026</div>
    </div>
</div>
""", unsafe_allow_html=True)