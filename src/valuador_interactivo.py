# =============================================================================
# TRABAJO TERMINAL
# Modelo de avalúo inmobiliario con técnicas de análisis espacial considerando las variables de gentrificación en la CDMX
#
# Autores: Kevin Jesús González Sosa, José Manuel Torres Gutiérrez 
# Institución: Escuela Superior de Cómputo
# Fecha: [Mayo, 2026]
#
# Descripción:
# Interfaz interactiva y desplegable (Streamlit) para la visualización,
# consulta y estimación de precios inmobiliarios basada en submercados
# diferenciados territorialmente y variables socioespaciales complejas.
#
# El sistema integra:
#   - Predicción hedónico-espacial mediante modelos ElasticNet locales.
#   - Análisis de contribución marginal de amenidades urbanas.
#   - Evaluación contextualizada con el índice de gentrificación.
#   - Métricas asociadas al paradigma de la Ciudad de 15 Minutos.
# =============================================================================

# Librerías estándar
import os
import warnings

# Manipulación de datos
import numpy as np
import pandas as pd

# Geoespacial
import geopandas as gpd
from scipy.spatial import cKDTree

# Visualización interactiva
import streamlit as st
import plotly.graph_objects as go

# Machine Learning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURACIÓN DE LA INTERFAZ DE STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Sistema de Valuación Inmobiliaria CDMX",
    layout="wide"
)

# ============================================================
# CONFIGURACIÓN DE RUTAS RELATIVAS (ESTRUCTURA GITHUB)
# ============================================================
# Al estar este archivo dentro de la carpeta 'src', PROJECT_ROOT apunta a 'src'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Ruta hacia la carpeta de salidas y resultados del modelo
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "..", "outputs", "results")
PATH_QGIS_DATA = os.path.join(OUTPUTS_PATH, "resultados_qgis_final.csv")

# Subimos un nivel para encontrar la carpeta matriz de datos del repositorio
DATA_PATH = os.path.join(PROJECT_ROOT, "..", "data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")

# Polígonos de demarcaciones territoriales
PATH_COLONIAS = os.path.join(RAW_DATA_PATH, "coloniascdmx", "colonias_iecm.shp")

# Capas de Infraestructura de Transporte Público Masivo y Semimasivo
PATH_METRO_EST = os.path.join(RAW_DATA_PATH, "stcmetro_shp", "STC_Metro_estaciones_utm14n.shp")
PATH_METROBUS_EST = os.path.join(RAW_DATA_PATH, "mb_shp", "Metrobus_estaciones.shp")
PATH_TREN_EST = os.path.join(RAW_DATA_PATH, "ste_shp", "ste_tren_ligero_shp", "ste_tren_ligero_shp", "STE_TrenLigero_estaciones_utm14n.shp")
PATH_TROLE_PARADAS = os.path.join(RAW_DATA_PATH, "ste_shp", "ste_trolebus_shp", "ste_trolebus_shp", "STE_Trolebus_Paradas.shp")
PATH_CABLE_EST = os.path.join(RAW_DATA_PATH, "ste_shp", "ste_cablebus_shp", "ste_cablebus_shp", "STE_Cablebus_estaciones.shp")

# Capas de Equipamiento Urbano, Áreas Verdes y Servicios Básicos
PATH_AREAS_VERDES = os.path.join(RAW_DATA_PATH, "inventario_areas_verdes_1", "inventario_areas_verdes_1.shp")
PATH_SALUD_BASE = os.path.join(RAW_DATA_PATH, "hospitales_y_centros_de_salud", "hospitales_y_centros_de_salud.shp")
PATH_HOSPITALES_PUBLICOS = os.path.join(RAW_DATA_PATH, "hospitales_2020_publicos", "hospitales_2020_publicos.shp")
PATH_COMERCIO = os.path.join(RAW_DATA_PATH, "cypc", "C_PComerciales.shp")

# ============================================================
# MATRIZ DE ATRIBUTOS SELECCIONADOS (FEATURES)
# ============================================================
FEATURES_MODELO = [
    # Atributos Estructurales e Intrínsecos del Inmueble
    "rooms",
    "bathrooms",
    "parking_spaces",
    "area",
    "antiguedad",

    # Accesibilidad y Conectividad Urbana
    "dist_metro_m",
    "density_metro",
    "dist_ciclovia_m",
    "densidad_ciclovia_15m",

    # Indicadores del Entorno (Ciudad de 15 Minutos)
    "comercio_density",
    "marginalidad_score",
    "score_15min",
    "prox_ciclovia",
    "prox_parque",
    "prox_salud",
    "prox_escuela",

    # Configuración Espacial y Dinámicas de Gentrificación
    "dist_subcenter_log",
    "gentrification_index",
    "listing_density_log",

    # Variables de Interacción para Capturar Efectos Cruzados
    "area_X_gentrif",
    "gentrif_x_metro"
]

# ============================================================
# FUNCIONES DE PROCESAMIENTO TEXTUAL
# ============================================================

def clean_text(txt):
    """
    Normaliza cadenas de texto eliminando acentos, mayúsculas y espacios redundantes.
    Garantiza la consistencia en los cruces por nombre de colonias y alcaldías.
    """
    if pd.isna(txt):
        return np.nan
    txt = str(txt).lower().strip()
    reemplazos = str.maketrans("áéíóúüñ", "aeiouun")
    return txt.translate(reemplazos)

# ============================================================
# MÓDULO CENTRAL DE PROCESAMIENTO Y MODELADO ESPACIAL
# ============================================================

@st.cache_resource
def preparar_sistema():
    """
    Ejecuta el pipeline de carga, normalización geoespacial, cálculo de distancias
    por vecindario más cercano, segmentación por clústeres y entrenamiento de modelos locales.
    """
    # 1. Carga y depuración inicial del set de datos inmobiliario
    df = pd.read_csv(PATH_QGIS_DATA)
    columnas_necesarias = ["price", "latitud", "longitud", "price_m2_raw"] + FEATURES_MODELO
    df = df.dropna(subset=columnas_necesarias).copy()
    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")
    df = df.dropna(subset=["latitud", "longitud"])

    # 2. Carga y homologación de proyecciones geográficas (WGS84 y UTM 14N)
    colonias = gpd.read_file(PATH_COLONIAS)
    if colonias.crs != "EPSG:4326":
        colonias = colonias.to_crs("EPSG:4326")

    areas_verdes = gpd.read_file(PATH_AREAS_VERDES)
    
    metro_est = gpd.read_file(PATH_METRO_EST).to_crs("EPSG:32614")
    metrobus_est = gpd.read_file(PATH_METROBUS_EST).to_crs("EPSG:32614")
    tren_est = gpd.read_file(PATH_TREN_EST).to_crs("EPSG:32614")
    trole_paradas = gpd.read_file(PATH_TROLE_PARADAS).to_crs("EPSG:32614")
    cable_est = gpd.read_file(PATH_CABLE_EST).to_crs("EPSG:32614")
    comercio_centros = gpd.read_file(PATH_COMERCIO).to_crs("EPSG:32614")

    # 3. Transformación de las coordenadas de las viviendas a metros (UTM 14N) para cálculo lineal de distancias
    viviendas_geo = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["longitud"], df["latitud"]), crs="EPSG:4326")
    viviendas_geo = viviendas_geo.reset_index(drop=True)
    viviendas_utm = viviendas_geo.to_crs(epsg=32614)
    coords_utm = np.array([(p.x, p.y) for p in viviendas_utm.geometry])
    
    def get_min_dist(gdf_target, coords_base):
        """
        Calcula la distancia mínima euclidiana de forma indexada mediante árboles KD alternativos.
        Optimiza la velocidad de respuesta frente a operaciones tradicionales de GIS.
        """
        pts = np.array([(g.x, g.y) for g in gdf_target.geometry if g.geom_type == "Point"])
        if len(pts) == 0:
            pts = np.array([(g.centroid.x, g.centroid.y) for g in gdf_target.geometry])
        tree = cKDTree(pts)
        return tree.query(coords_base)[0]
    
    # Cálculo indexado de proximidad a redes de transporte y comercio masivo
    df["dist_metro_m"] = get_min_dist(metro_est, coords_utm)
    df["dist_metrobus_m"] = get_min_dist(metrobus_est, coords_utm)
    df["dist_tren_m"] = get_min_dist(tren_est, coords_utm)
    df["dist_trole_m"] = get_min_dist(trole_paradas, coords_utm)
    df["dist_cable_m"] = get_min_dist(cable_est, coords_utm)
    df["dist_comercio_m"] = get_min_dist(comercio_centros, coords_utm)

    # 4. Procesamiento de Espacios Públicos y Áreas Verdes Recreativas
    categorias_validas = ["Parques", "Deportivos", "Jardines públicos", "Arboledas", "Alamedas"]
    areas_verdes = areas_verdes[areas_verdes["subcat_sed"].isin(categorias_validas)].copy()
    if areas_verdes.crs != "EPSG:32614":
        areas_verdes = areas_verdes.to_crs("EPSG:32614")
    areas_verdes = areas_verdes[areas_verdes.geometry.notnull() & areas_verdes.is_valid].copy()
    areas_verdes["geometry"] = areas_verdes.geometry.centroid
    
    tree_areas = cKDTree(np.array([(g.x, g.y) for g in areas_verdes.geometry]))
    df["dist_area_verde_recreativa_m"] = tree_areas.query(coords_utm)[0]

    # 5. Algoritmo de Fusión y Desduplicación de Infraestructura Hospitalaria
    salud_final_pts = []

    if os.path.exists(PATH_SALUD_BASE):
        salud_base = gpd.read_file(PATH_SALUD_BASE)
        if salud_base.crs is None: salud_base = salud_base.set_crs("EPSG:4326")
        salud_base = salud_base.to_crs(epsg=32614)
        salud_base = salud_base[salud_base.geometry.notnull() & ~salud_base.geometry.is_empty]
        pts_base = np.array([(p.x, p.y) for p in salud_base.geometry])
        if len(pts_base) > 0:
            salud_final_pts.append(pts_base)

    if os.path.exists(PATH_HOSPITALES_PUBLICOS):
        salud_pub = gpd.read_file(PATH_HOSPITALES_PUBLICOS)
        if salud_pub.crs is None: salud_pub = salud_pub.set_crs("EPSG:4326")
        salud_pub = salud_pub.to_crs(epsg=32614)
        salud_pub = salud_pub[salud_pub.geometry.notnull() & ~salud_pub.geometry.is_empty]
        
        if "CLAVE_DE_L" in salud_pub.columns:
            salud_pub = salud_pub[salud_pub["CLAVE_DE_L"] == "09"].copy()
            
        pts_pub = np.array([(p.x, p.y) for p in salud_pub.geometry])
        
        if len(pts_pub) > 0:
            if len(salud_final_pts) > 0:
                pts_existentes = np.vstack(salud_final_pts)
                tree_existente = cKDTree(pts_existentes)
                # Remueve nodos redundantes en un radio de 1 metro para evitar sobre-representación
                indices_duplicados = tree_existente.query_ball_point(pts_pub, r=1.0)
                filtrados = [pts_pub[i] for i, vecinos in enumerate(indices_duplicados) if len(vecinos) == 0]
                if len(filtrados) > 0:
                    salud_final_pts.append(np.array(filtrados))
            else:
                salud_final_pts.append(pts_pub)

    if salud_final_pts:
        salud_pts_master = np.vstack(salud_final_pts)
    else:
        salud_pts_master = np.empty((0, 2))

    if len(salud_pts_master) > 0:
        tree_salud = cKDTree(salud_pts_master)
        df["dist_salud_m"] = tree_salud.query(coords_utm)[0]
    else:
        df["dist_salud_m"] = 5000

    # 6. Cruce Espacial (Spatial Join) para determinar pertenencia política y residencial
    colonias["colonia_clean"] = colonias["NOMUT"].apply(clean_text)
    posibles_alcaldias = ["NOMDT", "DEMARCACI", "DEMARCACION", "MUNICIPIO", "NOM_MUN"]
    alcaldia_col = next((c for c in posibles_alcaldias if c in colonias.columns), None)
    
    if alcaldia_col is None:
        raise ValueError("Error en la estructura del Shapefile: Falta identificador territorial de demarcación.")
    colonias["alcaldia_clean"] = colonias[alcaldia_col].apply(clean_text)

    joined = gpd.sjoin(viviendas_geo, colonias[["colonia_clean", "alcaldia_clean", "geometry"]], how="left", predicate="within")
    joined = joined.sort_index()

    df["colonia_real"] = joined["colonia_clean"].values
    df["alcaldia_real"] = joined["alcaldia_clean"].values
    df["dist_estacion_metro_m"] = df["dist_metro_m"]

    df = df.dropna(subset=["colonia_real", "alcaldia_real"]).copy()

    # 7. Segmentación del Mercado de Suelo mediante K-Means
    # Agrupa submercados con base en localización física e indicadores socioespaciales estratégicos
    cluster_features = ["latitud", "longitud", "gentrification_index", "score_15min"]
    cluster_scaler = StandardScaler()
    Xc = cluster_scaler.fit_transform(df[cluster_features])

    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(Xc)

    # 8. Modelado Local mediante Regresiones ElasticNet por Clúster
    modelos, imputers = {}, {}
    for c in sorted(df["cluster"].unique()):
        d_cluster = df[df["cluster"] == c].copy()
        if len(d_cluster) < 30: continue

        X = d_cluster[FEATURES_MODELO]
        y = np.log1p(d_cluster["price"])  # Transformación logarítmica para estabilizar la varianza del precio
        imputador = X.median()

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("enet", ElasticNetCV(alphas=np.logspace(-4, -1, 20), l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99], cv=5, max_iter=5000))
        ])
        pipeline.fit(X.fillna(imputador), y)
        modelos[c] = pipeline
        imputers[c] = imputador

    # Estructuración de diccionarios de referencia para la interfaz
    alcaldia_colonias = df.groupby("alcaldia_real")["colonia_real"].unique().apply(lambda x: sorted(list(x))).to_dict()
    alcaldias_disponibles = sorted(alcaldia_colonias.keys())

    referencia_espacial = df.groupby("colonia_real")[
        FEATURES_MODELO + ["latitud", "longitud", "gentrif_local_presion", "gentrif_map_score",
                           "dist_area_verde_recreativa_m", "dist_salud_m", "dist_escuela_m", 
                           "dist_metro_m", "dist_metrobus_m", "dist_tren_m", "dist_trole_m", "dist_cable_m",
                           "dist_estacion_metro_m", "dist_comercio_m", "densidad_parques_15m", "acceso_salud_15m", "acceso_educacion_15m"]
    ].mean().to_dict("index")

    price_m2_col = df.groupby("colonia_real")["price_m2_raw"].median().to_dict()

    return modelos, imputers, kmeans, cluster_scaler, referencia_espacial, price_m2_col, alcaldias_disponibles, alcaldia_colonias, cluster_features

# ============================================================
# INSTANCIACIÓN DE ATRIBUTOS GLOBALES
# ============================================================

(
    MODELS, IMPUTERS, CLUSTERER, CLUSTER_SCALER, REF_ESPACIAL, 
    PRICE_M2_COL, ALCALDIAS_DISPONIBLES, ALCALDIA_COLONIAS, CLUSTER_FEATURES
) = preparar_sistema()

# ============================================================
# LOGICA DE INFERENCIA Y EVALUACIÓN DE MODELOS
# ============================================================

def predict_price(area, rooms, baths, parking, ant, colonia):
    """
    Construye el vector de entrada combinando las especificaciones del usuario con el
    perfil urbano de la colonia seleccionada, asigna el clúster y computa la predicción.
    """
    datos_colonia = REF_ESPACIAL.get(colonia, list(REF_ESPACIAL.values())[0])
    input_data = datos_colonia.copy()

    input_data.update({
        "area": area,
        "rooms": rooms,
        "bathrooms": baths,
        "parking_spaces": parking,
        "antiguedad": ant,
        "area_X_gentrif": (area * datos_colonia["gentrification_index"]),
        "gentrif_x_metro": (datos_colonia["gentrification_index"] * datos_colonia["dist_metro_m"])
    })

    X_df = pd.DataFrame([input_data])
    cluster_input = X_df[CLUSTER_FEATURES]
    cluster_scaled = CLUSTER_SCALER.transform(cluster_input)
    c_id = CLUSTERER.predict(cluster_scaled)[0]

    if c_id not in MODELS:
        c_id = list(MODELS.keys())[0]

    model = MODELS[c_id]
    X_pred = X_df[FEATURES_MODELO].fillna(IMPUTERS[c_id])
    log_pred = model.predict(X_pred)[0]
    precio = np.expm1(log_pred)  # Reversión de la transformación logarítmica log1p

    return precio, c_id, model, X_pred, datos_colonia

# ============================================================
# CONTROLES DE ENTRADA (SIDEBAR)
# ============================================================

with st.sidebar:
    st.header("🏢 Parámetros del Inmueble")
    alcaldia_sel = st.selectbox("Alcaldía", ALCALDIAS_DISPONIBLES)
    colonias_filtradas = ALCALDIA_COLONIAS.get(alcaldia_sel, [])
    colonia_sel = st.selectbox("Colonia", colonias_filtradas, key=f"colonia_{alcaldia_sel}")

    area = st.slider("Área Habitable (m²)", 25, 500, 120)
    rooms = st.number_input("Recámaras", min_value=1, max_value=10, value=3)
    baths = st.number_input("Baños Completos", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
    parking = st.number_input("Espacios de Estacionamiento", min_value=0, max_value=10, value=1)
    antiguedad = st.slider("Antigüedad de la Estructura (Años)", 0, 80, 10)

# Ejecución del motor de inferencia económica
precio, cluster_id, modelo_fit, x_input, datos_colonia = predict_price(
    area, rooms, baths, parking, antiguedad, colonia_sel
)

# ============================================================
# CUADRO DE MANDO PRINCIPAL (DASHBOARD)
# ============================================================

st.title("Sistema de Valuación Inmobiliaria CDMX")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Precio Estimado Comercial", f"${precio:,.0f} MXN")
c2.metric("Valor Unitario (m²)", f"${precio/area:,.0f} MXN/m²")
c3.metric("Identificador de Submercado (Cluster)", int(cluster_id))
c4.metric("Índice Macroeconómico de Gentrificación", round(datos_colonia["gentrification_index"], 3))

def fmt_dist(m):
    return f"{m:.0f} m" if m < 1000 else f"{m/1000:.1f} km"

# ============================================================
# SECCIÓN PANEL DE ANÁLISIS DE ENTORNO URBANO
# ============================================================
with st.expander("🌎 Análisis de Entorno Urbano y Socioespacial", expanded=True):

    # Bloque 1: Ciudad de 15 Minutos
    st.markdown("### 🏬 1. Accesibilidad a Escala Humana (Ciudad de 15 Minutos)")
    
    with st.container(border=True):
        # 1. Fila destacada para el Índice General (Ancho completo para el progreso y estatus)
        score_15 = float(datos_colonia["score_15min"])
        c_score, c_prog = st.columns([1, 3])
        
        with c_score:
            st.metric("🎯 Índice General 15 Min", round(score_15, 3), help="Indicador sintético de proximidad peatonal")
        with c_prog:
            st.markdown("<div style='padding-top: 10px;'></div>", unsafe_allow_html=True) # Alineación visual
            st.progress(score_15)
            if score_15 >= 0.7: st.success("Alta cobertura peatonal.")
            elif score_15 >= 0.4: st.info("Cobertura urbana funcional.")
            else: st.warning("Entorno con dependencia de vehículo.")
        
        st.markdown("---") # Separador sutil para las dimensiones
        
        # 2. Subcuadrícula de 2x2 para las 4 dimensiones urbanas (Da el doble de espacio horizontal)
        fila1_col1, fila1_col2 = st.columns(2)
        fila2_col1, fila2_col2 = st.columns(2)
        
        # --- Dimensión: Salud ---
        with fila1_col1:
            acceso_salud = float(datos_colonia["acceso_salud_15m"])
            st.metric("🏥 Servicios de Salud Cercanos", f"{acceso_salud:.0f} Unidades")
            if acceso_salud >= 6: st.success("Alta densidad médica.")
            elif acceso_salud >= 2: st.info("Cobertura hospitalaria básica.")
            else: st.warning("Déficit de equipamiento médico.")
            st.caption("Estructura unificada y depurada.")
            st.markdown("<br>", unsafe_allow_html=True)

        # --- Dimensión: Educación ---
        with fila1_col2:
            acceso_edu = float(datos_colonia["acceso_educacion_15m"])
            st.metric("📚 Planteles Educativos", f"{acceso_edu:.0f} Escuelas")
            if acceso_edu >= 40: st.success("Densidad de oferta escolar.")
            elif acceso_edu >= 15: st.info("Infraestructura escolar suficiente.")
            else: st.warning("Disponibilidad local limitada.")
            st.caption("Búfer operativo de 1.2 km.")
            st.markdown("<br>", unsafe_allow_html=True)
            
        # --- Dimensión: Espacio Público ---
        with fila2_col1:
            parque = float(datos_colonia["dist_area_verde_recreativa_m"])
            dens_parques = float(datos_colonia["densidad_parques_15m"])
            st.metric("🌳 Espacio Público Recreativo", f"{parque:.0f} m")
            if parque <= 300: st.success("Radio óptimo de proximidad.")
            elif parque <= 800: st.info("Distancia media de acceso.")
            else: st.warning("Déficit de áreas verdes.")
            st.caption(f"Aproximadamente {dens_parques:.0f} espacios verdes detectados.")

        # --- Dimensión: Abasto/Comercio ---
        with fila2_col2:
            comercio_dist = float(datos_colonia["dist_comercio_m"])
            st.metric("🛍️ Centros de Abasto / Comercio", fmt_dist(comercio_dist))
            if comercio_dist <= 600: st.success("Abasto local inmediato.")
            elif comercio_dist <= 1500: st.info("Proximidad comercial aceptable.")
            else: st.warning("Distancia prolongada a zonas comerciales.")
            st.caption("Concentraciones comerciales de escala urbana.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Bloque 2: Red de Movilidad e Infraestructura de Transporte Masivo
    st.markdown("### 🚊 2. Conectividad y Red de Transporte Estructurado")
    
    with st.container(border=True):
        # Dividimos el contenedor en 2 bloques principales: Izquierda (Métricas) y Derecha (Evaluación)
        c2_izquierda, c2_derecha = st.columns([3, 1.2])
        
        with c2_izquierda:
            # Creamos una cuadrícula de 3 columnas para distribuir uniformemente los sistemas de transporte
            sub_c1, sub_c2, sub_c3 = st.columns(3)
            
            d_metro = float(datos_colonia["dist_metro_m"])
            d_mb = float(datos_colonia["dist_metrobus_m"])
            d_tren = float(datos_colonia["dist_tren_m"])
            d_trole = float(datos_colonia["dist_trole_m"])
            d_cable = float(datos_colonia["dist_cable_m"])
            ciclovias = float(datos_colonia["densidad_ciclovia_15m"])
            
            with sub_c1:
                st.metric(label="🚇 STC Metro", value=fmt_dist(d_metro))
                st.markdown("<div style='padding-top: 15px;'></div>", unsafe_allow_html=True) # Espaciador vertical sutil
                st.metric(label="🚌 Metrobús", value=fmt_dist(d_mb))
                
            with sub_c2:
                st.metric(label="🚊 Tren Ligero", value=fmt_dist(d_tren))
                st.markdown("<div style='padding-top: 15px;'></div>", unsafe_allow_html=True)
                st.metric(label="🚎 Trolebús", value=fmt_dist(d_trole))
                
            with sub_c3:
                st.metric(label="🚠 Cablebús", value=fmt_dist(d_cable))
                st.markdown("<div style='padding-top: 15px;'></div>", unsafe_allow_html=True)
                
                # Traducimos la densidad de ciclovías a un diagnóstico comprensible
                if ciclovias >= 5.0:
                    status_bici = "Excelente"
                elif ciclovias >= 1.5:
                    status_bici = "Funcional"
                elif ciclovias > 0:
                    status_bici = "Escasa"
                else:
                    status_bici = "Ninguna"
                
                st.metric(
                    label="🚲 Infraestructura Ciclista", 
                    value=status_bici, 
                    help=f"Densidad de ciclovías detectadas en un radio de 15 minutos caminando ({ciclovias:.1f} segmentos)."
                )

        with c2_derecha:
            min_dist_sistema = min([d_metro, d_mb, d_tren, d_trole])
            
            # Un pequeño truco visual para alinear el título de la evaluación con las métricas de la izquierda
            st.markdown("<b style='font-size: 14px; color: #808495;'>Evaluación Multimodal</b>", unsafe_allow_html=True)
            st.markdown("<div style='padding-top: 4px;'></div>", unsafe_allow_html=True)
            
            if min_dist_sistema <= 500:
                st.success("🟢 **Conectividad Excelente**\n\nAcceso inmediato a nodos de transporte masivo.")
            elif min_dist_sistema <= 1000:
                st.info("🔵 **Accesibilidad Media**\n\nIntervalo estándar de caminata urbana.")
            else:
                st.warning("🟡 **Cobertura Restringida**\n\nSujeto a transporte colectivo local o baja frecuencia.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Bloque 3: Indicadores de Transformación Urbana y Dinámica Inmobiliaria
    st.markdown("### 🏙️ 3. Dinámica de Transformación Socioespacial")
    with st.container(border=True):
        c3_1, c3_2, c3_3 = st.columns(3)
        
        gentrif_macro = float(datos_colonia["gentrification_index"])
        gentrif_micro = float(datos_colonia["gentrif_local_presion"])
        gentrif = float(datos_colonia["gentrif_map_score"])
        
        with c3_1:
            st.metric("🏛️ Cambio Estructural (Escala Alcaldía)", round(gentrif_macro, 3))
            st.progress(gentrif_macro)
            st.caption("Evolución socioeconómica intercensal (Variables deltas 2010-2020 INEGI)")
            
        with c3_2:
            st.metric("🏘️ Presión de Mercado Local", round(gentrif_micro, 3))
            st.progress(gentrif_micro)
            st.caption("Variaciones de valor respecto al entorno geográfico continuo")
            
        with c3_3:
            st.metric("🌆 Vulnerabilidad al Cambio Socioespacial", round(gentrif, 3))
            st.progress(gentrif)
            if gentrif >= 0.7: st.warning("Entorno bajo un proceso acelerado de transformación.")
            elif gentrif >= 0.4: st.info("Área en transición urbana activa.")
            else: st.success("Zona con estabilidad sociodemográfica relativa.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Bloque 4: Mercados y Desempeño Estructural
    st.markdown("### 📊 4. Estructura Urbana y Desempeño del Mercado")
    with st.container(border=True):
        c4_1, c4_2, c4_3 = st.columns(3)
        
        with c4_1:
            marginalidad = int(datos_colonia["marginalidad_score"])
            st.metric("📉 Índice de Rezago Social", f"{marginalidad} / 5")
            if marginalidad <= 2: st.success("Estrato socioeconómico medio-alto / alto.")
            elif marginalidad <= 3.0: st.info("Estrato socioeconómico medio.")
            else: st.error("Condiciones de vulnerabilidad social detectadas.")
            st.caption("Alineado al marco metodológico de CONAPO.")
            
        with c4_2:
            listing = float(datos_colonia["listing_density_log"])
            st.metric("🏢 Grado de Densidad Comercial", f"{listing:.2f} (Log)")
            if listing >= 3: st.success("Mercado inmobiliario con alta rotación.")
            elif listing >= 1.5: st.info("Actividad de mercado regular.")
            else: st.warning("Baja tasa de transacciones en la zona.")
            st.caption("Densidad logarítmica de listados activos.")
            
        with c4_3:
            centralidad_log = float(datos_colonia["dist_subcenter_log"])
            centralidad_m = np.expm1(centralidad_log)
            st.metric("📍 Proximidad a Nodos de Empleo", fmt_dist(centralidad_m))
            if centralidad_m <= 1000: st.success("Alta centralidad económica corporativa.")
            elif centralidad_m <= 3000: st.info("Distancia funcional a subcentros urbanos.")
            else: st.warning("Ubicación eminentemente habitacional o periférica.")
            st.caption("Cercanía a clústeres económicos y de servicios masivos.")

# ============================================================
# MÓDULO DE INTERPRETACIÓN ECONÓMICA DE ATRIBUTOS
# ============================================================
st.markdown("### 💰 Elasticidad y Aportación de Atributos al Precio de Cierre")

NOMBRES_VARIABLES = {
    "rooms": "Recámaras adicionales",
    "bathrooms": "Baños completos",
    "parking_spaces": "Espacios de estacionamiento",
    "area": "Metros cuadrados de construcción (Área)",
    "antiguedad": "Años de antigüedad de la estructura",
    "dist_metro_m": "Proximidad a estaciones del STC Metro",
    "density_metro": "Concentración local de accesos al Metro",
    "dist_ciclovia_m": "Distancia a la red de ciclovías",
    "densidad_ciclovia_15m": "Densidad de infraestructura ciclista",
    "comercio_density": "Intensidad de comercios regionales",
    "marginalidad_score": "Grado de marginación social de la zona",
    "score_15min": "Puntaje del entorno 'Ciudad de 15 Minutos'",
    "prox_ciclovia": "Cercanía a ejes viales con ciclovía",
    "prox_parque": "Cercanía a infraestructura de parques",
    "prox_salud": "Proximidad a equipamiento médico",
    "prox_escuela": "Proximidad a centros educativos de la zona",
    "dist_subcenter_log": "Distancia a distritos centrales de empleo",
    "gentrification_index": "Nivel de gentrificación de la demarcación",
    "listing_density_log": "Presencia de inventario en el mercado local",
    "area_X_gentrif": "Efecto cruzado: Dimensiones en zonas con alta plusvalía",
    "gentrif_x_metro": "Efecto cruzado: Cercanía al transporte en colonias en transición"
}

# Obtención de los coeficientes penalizados del modelo ElasticNet correspondiente al clúster
enet_model = modelo_fit.named_steps['enet']
coeficientes = enet_model.coef_

valores_usuario = {
    "rooms": float(rooms),
    "bathrooms": float(baths),
    "parking_spaces": float(parking),
    "area": float(area),
    "antiguedad": float(antiguedad),
    "dist_metro_m": float(datos_colonia["dist_metro_m"]),
    "density_metro": float(datos_colonia["density_metro"]),
    "dist_ciclovia_m": float(datos_colonia["dist_ciclovia_m"]),
    "densidad_ciclovia_15m": float(datos_colonia["densidad_ciclovia_15m"]),
    "comercio_density": float(datos_colonia["comercio_density"]),
    "marginalidad_score": float(datos_colonia["marginalidad_score"]),
    "score_15min": float(datos_colonia["score_15min"]),
    "prox_ciclovia": float(datos_colonia["prox_ciclovia"]),
    "prox_parque": float(datos_colonia["prox_parque"]),
    "prox_salud": float(datos_colonia["prox_salud"]),
    "prox_escuela": float(datos_colonia["prox_escuela"]),
    "dist_subcenter_log": float(datos_colonia["dist_subcenter_log"]),
    "gentrification_index": float(datos_colonia["gentrification_index"]),
    "listing_density_log": float(datos_colonia["listing_density_log"]),
    "area_X_gentrif": float(area * datos_colonia["gentrification_index"]),
    "gentrif_x_metro": float(datos_colonia["gentrification_index"] * datos_colonia["dist_metro_m"])
}

# Desglose analítico del valor en términos absolutos ($ MXN)
impactos_pesos = []
for idx, var in enumerate(FEATURES_MODELO):
    val = valores_usuario.get(var, 0.0)
    beta = coeficientes[idx]
    
    # Evaluación del cambio marginal sobre el valor predicho del mercado
    impacto_estimado = precio * (np.exp(beta * (val / (x_input[var].values[0] if x_input[var].values[0] != 0 else 1))) - 1)
    
    # Restricción matemática para acotar distorsiones por escalas en interacciones extremas
    if var in ["area", "area_X_gentrif"] and abs(impacto_estimado) > precio:
        impacto_estimado = np.sign(impacto_estimado) * (precio * 0.4)
        
    impactos_pesos.append(impacto_estimado)

datos_impacto = pd.DataFrame({
    "Variable_Interna": FEATURES_MODELO,
    "Impacto_Pesos": impactos_pesos
})

datos_impacto["Característica"] = datos_impacto["Variable_Interna"].map(NOMBRES_VARIABLES).fillna(datos_impacto["Variable_Interna"])

# Filtro de significancia económica (Umbral de $15,000 MXN para descartar variables sin peso estadístico)
UMBRAL_IMPACTO = 15000 
datos_filtrados = datos_impacto[datos_impacto["Impacto_Pesos"].abs() >= UMBRAL_IMPACTO].copy()
datos_filtrados = datos_filtrados.sort_values(by="Impacto_Pesos", ascending=True)

if not datos_filtrados.empty:
    fig_impacto = go.Figure()
    colores = ['#EF553B' if val < 0 else '#00CC96' for val in datos_filtrados["Impacto_Pesos"]]

    fig_impacto.add_trace(go.Bar(
        y=datos_filtrados["Característica"],
        x=datos_filtrados["Impacto_Pesos"],
        orientation='h',
        marker_color=colores,
        text=datos_filtrados["Impacto_Pesos"].apply(lambda x: f"${x:,.0f} MXN" if x >= 0 else f"-${abs(x):,.0f} MXN"),
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Aportación: %{x:$,.2f} MXN<extra></extra>"
    ))

    fig_impacto.update_layout(
        title="<b>Factores Dominantes en la Formación del Precio</b><br><span style='font-size:12px;color:gray;'>Mostrando aportaciones significativas con impacto absoluto mayor a $15k MXN</span>",
        xaxis_title="Impacto Neto sobre el Valor Estimado ($ MXN)",
        yaxis_title="",
        margin=dict(l=25, r=80, t=70, b=25),
        height=len(datos_filtrados) * 38 + 110,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.15)',
            zeroline=True,
            zerolinecolor='rgba(128, 128, 128, 0.4)'
        )
    )

    st.plotly_chart(fig_impacto, use_container_width=True)
    st.info("💡 **Guía de Lectura:** Las barras **verdes** cuantifican ventajas intrínsecas o del entorno urbano que incrementan la plusvalía. Las barras **rojas** reflejan penalizaciones por obsolescencia física (antigüedad) o rezagos de conectividad territorial.")
else:
    st.warning("No se identificaron variables individuales con un impacto superior al umbral crítico de $15,000 MXN en esta configuración de vivienda.")

# ============================================================
# MAPA DE CONTEXTO GEOGRÁFICO
# ============================================================
st.subheader("Ubicación Aproximada de la Colonia")
map_df = pd.DataFrame({
    "lat": [datos_colonia["latitud"]],
    "lon": [datos_colonia["longitud"]]
})
st.map(map_df)

# ============================================================
# VISUALIZACIÓN DE CONTROL (METADATOS DE SOPORTE)
# ============================================================
with st.expander("Metadatos del Sistema (Soporte Técnico)"):
    st.write({
        "ID Clúster Activo": int(cluster_id),
        "Centroide Latitud": float(datos_colonia["latitud"]),
        "Centroide Longitud": float(datos_colonia["longitud"]),
        "Mediana precio m² zona": float(PRICE_M2_COL.get(colonia_sel, 0))
    })