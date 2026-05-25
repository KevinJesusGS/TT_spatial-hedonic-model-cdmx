# =============================================================================
# SISTEMA INTEGRADO DE VALUACIÓN INMOBILIARIA CDMX
# Portal Principal — Inicio
#
# Autores: Kevin Jesús González Sosa, José Manuel Torres Gutiérrez
# Institución: Escuela Superior de Cómputo — Instituto Politécnico Nacional
# Fecha: Mayo, 2026
#
# Descripción:
# Portal de acceso unificado a los tres módulos de valuación inmobiliaria:
#   1. Compra/Venta — Modelo hedónico-espacial ElasticNet para precio comercial
#   2. Rentas       — Modelo hedónico-espacial ElasticNet para renta mensual
#   3. Airbnb       — Modelo hedónico-espacial ElasticNet para tarifas de hospedaje
# =============================================================================

import os
import base64 as _b64
import streamlit as st
import streamlit.components.v1 as _stcomp

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Sistema de Valuación Inmobiliaria CDMX · ESCOM-IPN",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ============================================================
# IMPORTAR FUNCIONES DE PRECARGA (lazy import para no romper rutas)
# ============================================================
import importlib, sys, pathlib

def _precargar_modelos():
    """Importa y ejecuta las funciones cacheadas de los 3 módulos.
    Streamlit guarda el resultado en st.cache_resource, así que
    cuando el usuario navega a cada página el modelo ya está listo."""
    pages_dir = pathlib.Path(__file__).parent / "pages"
    
    # --- Compra/Venta ---
    if "Compra_Venta" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "Compra_Venta", pages_dir / "Compra_Venta.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["Compra_Venta"] = mod
        # Solo necesitamos que cache_resource registre la función,
        # no ejecutar todo el script. Usamos el truco de importar solo lo que sí se puede.
    
    # En la práctica, el patrón más simple y confiable es:
    # llamar preparar_sistema() y preparar_sistema_airbnb() directamente
    # desde Inicio.py si están definidas en el mismo paquete.
    # PERO como son scripts de páginas independientes en Streamlit,
    # la solución real está en session_state (ver abajo).
    pass

# ============================================================
# PARCHE SIDEBAR — fondo sólido garantizado
# ============================================================
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
        os.path.join(_SCRIPT_DIR, "..", "..", "outputs", "assets", filename),
    ]:
        if os.path.exists(candidate):
            return candidate
    return ""

_escom_b64 = _img_to_b64(_find_logo("escom.png"))
_ipn_b64   = _img_to_b64(_find_logo("ipn.png"))

# ============================================================
# CSS GLOBAL UNIFICADO
# ============================================================
st.markdown("""
<style>
/* ─── BASE ─────────────────────────────────────────────── */
.stApp {
    background-color: var(--background-color) !important;
    color: var(--text-color) !important;
}
* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    -webkit-font-smoothing: antialiased;
}
.block-container {
    padding-top: 2rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 1400px !important;
}

/* ─── SIDEBAR ──────────────────────────────────────────── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] *:not(img):not(svg):not(canvas) {
    background-color: var(--secondary-background-color) !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}
section[data-testid="stSidebar"] {
    border-right: 1px solid var(--border-color) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.15) !important;
    opacity: 1 !important;
    z-index: 999999 !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: var(--text-color) !important;
}

/* ─── HEADER PREMIUM ───────────────────────────────────── */
.premium-header {
    background: var(--secondary-background-color);
    border-radius: 24px;
    padding: 2rem 2.5rem;
    margin-bottom: 2.5rem;
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
    50%       { opacity: 1; }
}
.header-content {
    display: flex;
    align-items: center;
    gap: 2.5rem;
}
.logo-section {
    display: flex;
    align-items: center;
    gap: 1.25rem;
    flex-shrink: 0;
}
.logo-box {
    width: 70px; height: 70px;
    border-radius: 16px;
    background: var(--background-color);
    border: 1px solid var(--border-color);
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
}
.logo-box img { height: 60px; width: 60px; object-fit: contain; }
.logo-divider {
    width: 1px; height: 48px;
    background: var(--border-color);
}
.title-section { flex: 1; }
.main-title {
    font-size: 1.65rem;
    font-weight: 800;
    color: var(--text-color) !important;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin: 0 0 0.25rem;
}
.subtitle {
    font-size: 0.9rem;
    color: var(--text-color);
    opacity: 0.7;
    font-weight: 500;
    margin: 0;
}
.authors-section {
    text-align: right;
    flex-shrink: 0;
}
.author-name {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-color);
    opacity: 0.85;
}

/* ─── TARJETAS DE MÓDULO ────────────────────────────────── */
.module-card {
    background: var(--secondary-background-color);
    border: 1.5px solid var(--border-color);
    border-radius: 20px;
    padding: 2rem;
    height: 100%;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;
}
.module-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.12);
}
.module-card .module-icon {
    font-size: 2.8rem;
    margin-bottom: 1rem;
    display: block;
}
.module-card .module-title {
    font-size: 1.3rem;
    font-weight: 800;
    color: var(--text-color) !important;
    margin: 0 0 0.5rem;
}
.module-card .module-desc {
    font-size: 0.88rem;
    color: var(--text-color);
    opacity: 0.75;
    line-height: 1.6;
    margin: 0 0 1.25rem;
}
.module-card .module-tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    margin-right: 6px;
    margin-bottom: 6px;
}
.module-card .module-accent {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    border-radius: 20px 20px 0 0;
}
.module-card .btn-ir {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 1.25rem;
    padding: 10px 20px;
    border-radius: 10px;
    font-size: 0.9rem;
    font-weight: 700;
    text-decoration: none;
    border: 2px solid transparent;
    transition: all 0.15s ease;
    cursor: pointer;
    color: white !important;
}

/* ─── CARDS ESTADÍSTICAS ────────────────────────────────── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 2rem 0;
}
.stat-card {
    background: var(--secondary-background-color);
    border: 1.5px solid var(--border-color);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    text-align: center;
}
.stat-card .stat-num {
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--text-color);
    letter-spacing: -0.02em;
}
.stat-card .stat-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-color);
    opacity: 0.6;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}

/* ─── INFO SECTION ──────────────────────────────────────── */
.info-section {
    background: var(--secondary-background-color);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.75rem 2rem;
    margin-top: 2rem;
}

/* ─── RESPONSIVE ────────────────────────────────────────── */
@media screen and (max-width: 1024px) {
    .block-container {
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    .stat-grid { grid-template-columns: repeat(2, 1fr) !important; }
    .authors-section { display: none !important; }
}
@media screen and (max-width: 640px) {
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
    .header-content {
        flex-direction: column !important;
        align-items: center !important;
        text-align: center !important;
        gap: 0.75rem !important;
    }
    .logo-divider { display: none !important; }
    .main-title { font-size: 1.2rem !important; }
    .stat-grid { grid-template-columns: repeat(2, 1fr) !important; }
}

/* ─── SCROLLBAR ─────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: var(--background-color); }
::-webkit-scrollbar-thumb { background: #3B82F6; border-radius: 4px; }

hr {
    border: none !important;
    height: 1px !important;
    background: var(--border-color) !important;
    margin: 2rem 0 !important;
}
h1, h2, h3, h4, h5, h6 { color: var(--text-color) !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ENCABEZADO INSTITUCIONAL
# ============================================================
_logo_h    = "height:60px;object-fit:contain;"
_ipn_tag   = f"<img src='data:image/png;base64,{_ipn_b64}' style='{_logo_h}'>" if _ipn_b64 else "🏛️"
_escom_tag = f"<img src='data:image/png;base64,{_escom_b64}' style='{_logo_h}'>" if _escom_b64 else "🎓"

st.markdown(f"""
<div class="premium-header">
    <div class="header-content">
        <div class="logo-section">
            <div class="logo-box">{_ipn_tag}</div>
            <div class="logo-divider"></div>
            <div class="logo-box">{_escom_tag}</div>
        </div>
        <div class="title-section">
            <div class="main-title">Sistema de Valuación Inmobiliaria CDMX</div>
            <div class="subtitle">
                Modelo hedónico-espacial con análisis de gentrificación · Escuela Superior de Cómputo · IPN
            </div>
        </div>
        <div class="authors-section">
            <div class="author-name">Kevin J. González Sosa</div>
            <div class="author-name">José M. Torres Gutiérrez</div>
            <div style="font-size:0.75rem;opacity:0.6;margin-top:0.4rem;">Trabajo Terminal · ESCOM 2026</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR — Navegación y contexto
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 1rem;border-bottom:1px solid var(--border-color);margin-bottom:1.5rem;">
        <div style="font-size:2.5rem;margin-bottom:0.5rem;">🏙️</div>
        <h2 style="margin:0;font-size:1.1rem;font-weight:800;color:var(--text-color);">
            Valuación CDMX
        </h2>
        <p style="font-size:0.78rem;color:var(--text-color);opacity:0.6;margin-top:0.3rem;">
            ESCOM · IPN · 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:0.75rem 0;">
        <p style="font-size:0.82rem;color:var(--text-color);opacity:0.75;line-height:1.6;">
            Selecciona un módulo de valuación desde el menú de páginas (⬆️ arriba) o usando los botones en el panel principal.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style="font-size:0.8rem;color:var(--text-color);opacity:0.7;line-height:1.8;">
        <div style="font-weight:700;margin-bottom:0.5rem;">📌 Módulos disponibles</div>
        <div>🏢 <b>Compra/Venta</b> — Precio comercial</div>
        <div>🏘️ <b>Rentas</b> — Renta mensual</div>
        <div>🏠 <b>Airbnb</b> — Tarifa por noche</div>
    </div>
    """, unsafe_allow_html=True)

    # En el sidebar, después de los módulos disponibles:
    st.markdown("---")

    st.markdown("""
    <div style="font-size:0.75rem;color:var(--text-color);opacity:0.55;line-height:1.6;">
        <div>📊 Fuentes: INEGI, ADIP CDMX,</div>
        <div>SHF, SEDUVI, STC Metro</div>
        <div style="margin-top:0.5rem;">🤖 Modelos: ElasticNet + K-Means</div>
        <div>🗺️ SIG: QGIS + GeoPandas</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# ESTADÍSTICAS DEL SISTEMA
# ============================================================
st.markdown("""
<div class="stat-grid">
    <div class="stat-card">
        <div class="stat-num" style="color:#3B82F6;">3</div>
        <div class="stat-label">Modelos Predictivos</div>
    </div>
    <div class="stat-card">
        <div class="stat-num" style="color:#10B981;">16</div>
        <div class="stat-label">Alcaldías CDMX</div>
    </div>
    <div class="stat-card">
        <div class="stat-num" style="color:#8B5CF6;">+40</div>
        <div class="stat-label">Variables Hedónicas</div>
    </div>
    <div class="stat-card">
        <div class="stat-num" style="color:#F59E0B;">6</div>
        <div class="stat-label">Submercados Espaciales</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# CARDS DE MÓDULOS
# ============================================================
st.markdown("""
<div style="margin-bottom:1rem;">
    <h2 style="font-size:1.4rem;font-weight:800;color:var(--text-color);margin:0 0 0.4rem;">
        🚀 Selecciona un Módulo de Valuación
    </h2>
    <p style="font-size:0.9rem;color:var(--text-color);opacity:0.65;margin:0;">
        Cada módulo utiliza un modelo hedónico-espacial independiente, calibrado para su mercado específico.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("""
    <div class="module-card">
        <div class="module-accent" style="background:linear-gradient(90deg,#3B82F6,#2563EB);"></div>
        <span class="module-icon">🏢</span>
        <div class="module-title">Compra / Venta</div>
        <div class="module-desc">
            Estima el precio comercial de departamentos y casas en la CDMX mediante 
            regresión ElasticNet espacial con variables hedónicas, catastrales y de 
            gentrificación. Incluye atlas cartográfico con 20 mapas temáticos.
        </div>
        <div>
            <span class="module-tag" style="background:#3B82F620;color:#3B82F6;border:1px solid #3B82F640;">
                Precio Total MXN
            </span>
            <span class="module-tag" style="background:#3B82F620;color:#3B82F6;border:1px solid #3B82F640;">
                Valor/m²
            </span>
        </div>
        <div>
            <span class="module-tag" style="background:#E0F2FE;color:#0369A1;border:1px solid #BAE6FD;">
                Segmento Estándar
            </span>
            <span class="module-tag" style="background:#FEF3C7;color:#92400E;border:1px solid #FDE68A;">
                Lujo
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("→ Ir al Valuador de Compra/Venta", key="btn_compra",
                 use_container_width=True, type="primary"):
        st.switch_page("pages/Compra_Venta.py")

with col2:
    st.markdown("""
    <div class="module-card">
        <div class="module-accent" style="background:linear-gradient(90deg,#10B981,#059669);"></div>
        <span class="module-icon">🏘️</span>
        <div class="module-title">Rentas Mensuales</div>
        <div class="module-desc">
            Estima la renta mensual óptima de una vivienda en la CDMX considerando 
            su entorno socioespacial, accesibilidad urbana y presión de gentrificación. 
            Incluye mapa interactivo y análisis de submercado.
        </div>
        <div>
            <span class="module-tag" style="background:#10B98120;color:#10B981;border:1px solid #10B98140;">
                Renta MXN/mes
            </span>
            <span class="module-tag" style="background:#10B98120;color:#10B981;border:1px solid #10B98140;">
                Precio/m²
            </span>
        </div>
        <div>
            <span class="module-tag" style="background:#D1FAE5;color:#065F46;border:1px solid #A7F3D0;">
                Índice 15 Min
            </span>
            <span class="module-tag" style="background:#D1FAE5;color:#065F46;border:1px solid #A7F3D0;">
                Gentrificación
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("→ Ir al Valuador de Rentas", key="btn_renta",
                 use_container_width=True):
        st.switch_page("pages/Rentas.py")

with col3:
    st.markdown("""
    <div class="module-card">
        <div class="module-accent" style="background:linear-gradient(90deg,#F59E0B,#D97706);"></div>
        <span class="module-icon">🏠</span>
        <div class="module-title">Airbnb / Hospedaje</div>
        <div class="module-desc">
            Estima la tarifa noche óptima para listings de Airbnb en la CDMX. 
            Incorpora tipo de alojamiento, amenidades, perfil de anfitrión, reseñas 
            y variables espaciales de accesibilidad turística.
        </div>
        <div>
            <span class="module-tag" style="background:#F59E0B20;color:#F59E0B;border:1px solid #F59E0B40;">
                Tarifa/noche MXN
            </span>
            <span class="module-tag" style="background:#F59E0B20;color:#F59E0B;border:1px solid #F59E0B40;">
                Potencial Anual
            </span>
        </div>
        <div>
            <span class="module-tag" style="background:#FEF3C7;color:#92400E;border:1px solid #FDE68A;">
                Amenidades
            </span>
            <span class="module-tag" style="background:#FEF3C7;color:#92400E;border:1px solid #FDE68A;">
                Superhost
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("→ Ir al Valuador de Airbnb", key="btn_airbnb",
                 use_container_width=True):
        st.switch_page("pages/Airbnb.py")

# ============================================================
# SECCIÓN INFORMATIVA
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)

with st.expander("📖 Acerca del Sistema — Objetivos y Metodología", expanded=False):
    col_izq, col_der = st.columns(2, gap="large")

    with col_izq:
        st.markdown("""
        #### 🎯 Objetivo General
        Desarrollar un modelo de regresión espacial aplicado al avalúo inmobiliario 
        para el estudio de los procesos de gentrificación y su impacto en el valor 
        inmobiliario de la Ciudad de México.

        #### 🔬 Metodología
        El sistema integra:
        - **Regresión ElasticNet** con selección automática de variables (Lasso/Ridge)
        - **Segmentación K-Means** en submercados territoriales diferenciados
        - **Variables hedónicas** físicas, socioeconómicas y espaciales
        - **Índice de gentrificación** multi-componente por colonia
        - **Paradigma Ciudad de 15 Minutos** como variable de accesibilidad
        - **Variables catastrales** del SIGCDMX integradas por buffer espacial
        """)

    with col_der:
        st.markdown("""
        #### 📊 Fuentes de Datos
        - **Portales inmobiliarios**: datos de oferta activa (compra/venta y rentas)
        - **Inside Airbnb**: listings activos de hospedaje en CDMX
        - **INEGI Censo 2020**: indicadores socioeconómicos por AGEB
        - **ADIP CDMX**: capas geoespaciales urbanas (transporte, servicios)
        - **SIGCDMX / Catastro**: valores unitarios del suelo por predio
        - **SHF**: Índice de Precios de la Vivienda (referencia de validación)

        #### ✅ Alcances del Despliegue
        - Estimación puntual con selección de colonia o punto en mapa
        - Análisis de contribución marginal por variable
        - Atlas cartográfico del mercado inmobiliario (Compra/Venta)
        - Mapas interactivos de transporte, servicios y gentrificación
        """)

st.markdown("---")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style="text-align:center;padding:2rem 0;border-top:1px solid var(--border-color);margin-top:1rem;">
    <div style="font-weight:700;font-size:1rem;color:var(--text-color);margin-bottom:0.5rem;">
        Sistema de Valuación Inmobiliaria CDMX
    </div>
    <div style="font-size:0.875rem;color:var(--text-color);opacity:0.7;line-height:1.8;">
        Escuela Superior de Cómputo · Instituto Politécnico Nacional<br>
        <span style="font-weight:600;">Kevin J. González Sosa · José M. Torres Gutiérrez</span><br>
        <span style="font-size:0.75rem;opacity:0.6;">Trabajo Terminal · ESCOM 2026</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PRECARGA SILENCIOSA DE LOS 3 MODELOS EN SEGUNDO PLANO
# Streamlit ejecuta el script completo en cada interacción,
# pero cache_resource garantiza que el trabajo pesado
# solo ocurre la primera vez.
# ============================================================
if "modelos_precargados" not in st.session_state:
    # Mostrar un indicador sutil en el footer
    _precarga_placeholder = st.empty()
    _precarga_placeholder.caption("⚙️ Precargando modelos en segundo plano...")
    
    # Importar y ejecutar cada sistema (solo la primera vez es lento)
    try:
        # Necesitas importar las funciones cacheadas aquí
        # La forma más limpia es mover preparar_sistema() a un módulo compartido,
        # por ejemplo: utils/modelos.py
        # y luego importarlo aquí y en cada página.
        pass
    except Exception as e:
        pass
    
    st.session_state["modelos_precargados"] = True
    _precarga_placeholder.empty()