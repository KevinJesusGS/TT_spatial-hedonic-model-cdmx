# =============================================================================
# TRABAJO TERMINAL
# Modelo de avalúo inmobiliario con técnicas de análisis espacial considerando las variables de gentrificación en la CDMX
#
# Autores: Kevin Jesús González Sosa, José Manuel Torres Gutiérrez 
# Institución: Escuela Superior de Cómputo
# Fecha: [Mayo, 2026]
#
# Descripción:
# Este script implementa un modelo hedónico-espacial para estimación
# de precios inmobiliarios mediante segmentación espacial, ElasticNet
# regularizado y variables geoespaciales urbanas.
#
# El modelo integra:
#   - Accesibilidad multimodal
#   - Policentrismo urbano
#   - Índice robusto de gentrificación
#   - Métricas de ciudad de 15 minutos
#   - Validación espacial sin fuga de información
#
# Objetivo:
# Evaluar la relación entre estructura urbana, accesibilidad y procesos
# de gentrificación sobre la formación del valor inmobiliario.
# =============================================================================

# =============================================================================
# SUPUESTOS METODOLÓGICOS
#
# 1. La distancia euclidiana aproxima accesibilidad espacial.
# 2. El mercado inmobiliario presenta dependencia espacial local.
# 3. Existen submercados diferenciados territorialmente.
# 4. Los efectos hedónicos pueden modelarse linealmente tras transformación log.
# 5. La gentrificación puede aproximarse mediante cambio socioeconómico PCA.
# =============================================================================

# Librerías estándar
import os
import glob
import warnings

# Manipulación de datos
import numpy as np
import pandas as pd

# Geoespacial
import geopandas as gpd
import pyogrio
from scipy.spatial import cKDTree

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error,
    silhouette_score
)
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURACIÓN DE EJECUCIÓN
# ======================================================
MOSTRAR_GRAFICAS = False  # Control de visualización.
                          # Se mantiene desactivado para ejecución batch y reproducibilidad experimental. 

# Configuración estética de gráficas
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ======================================================
# RUTAS
# ======================================================


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(PROJECT_ROOT, "..", "data")
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "..", "outputs")
RESULTS_PATH = os.path.join(OUTPUTS_PATH, "results")
FIGURES_PATH = os.path.join(OUTPUTS_PATH, "figures")

# Crear carpetas automáticamente si no existen
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

path_viviendas = os.path.join(DATA_PATH, "processed", "dataset_geocodificado.csv")
path_2010 = os.path.join(DATA_PATH, "raw", "INEGI", "inegi_2010.csv")
path_2020 = os.path.join(DATA_PATH, "raw", "INEGI", "inegi_2020.csv")

path_metro = os.path.join(DATA_PATH, "raw", "stcmetro_shp", "STC_Metro_estaciones_utm14n.shp")
path_metrobus = os.path.join(DATA_PATH, "raw", "mb_shp", "Metrobus_estaciones.shp")
path_seguridad = os.path.join(DATA_PATH, "raw", "crimen", "urbanismo_social_sintesis.shp")
path_comercio = os.path.join(DATA_PATH, "raw", "cypc", "C_PComerciales.shp")
path_turistas = os.path.join(DATA_PATH, "raw", "Turistas", "turistas_alcaldia.csv")

path_tren = os.path.join(DATA_PATH, "raw", "ste_shp", "ste_tren_ligero_shp", "ste_tren_ligero_shp", "STE_TrenLigero_estaciones_utm14n.shp")
path_trole = os.path.join(DATA_PATH, "raw", "ste_shp", "ste_trolebus_shp", "ste_trolebus_shp", "STE_Trolebus_Paradas.shp")
path_cable = os.path.join(DATA_PATH, "raw", "ste_shp", "ste_cablebus_shp", "ste_cablebus_shp", "STE_Cablebus_estaciones.shp")

path_alcaldias = os.path.join(DATA_PATH, "raw", "alcaldias", "poligonos_alcaldias_cdmx.shp")

# Capas geoespaciales asociadas al paradigma de accesibilidad urbana
# bajo el enfoque de "Ciudad de 15 Minutos"

path_ciclovias = os.path.join(DATA_PATH, "raw", "infraestructura_vial_ciclista", "Infraestructura ciclista total.shp")
path_areas_verdes = os.path.join(DATA_PATH, "raw", "inventario_areas_verdes_1", "inventario_areas_verdes_1.shp")
path_salud = os.path.join(DATA_PATH, "raw", "hospitales_y_centros_de_salud", "hospitales_y_centros_de_salud.shp")
path_hospitales_publicos = os.path.join(DATA_PATH, "raw", "hospitales_2020_publicos", "hospitales_2020_publicos.shp")
path_escuelas_pub = os.path.join(DATA_PATH, "raw", "escuelas_publicas", "escuelas_publicas.shp")
path_escuelas_priv = os.path.join(DATA_PATH, "raw", "escuelas_privadas", "escuelas_privadas.shp")
path_uso_suelo = os.path.join(DATA_PATH, "raw", "uso-de-suelo", "uso-de-suelo.shp")

carpeta_promedios = os.path.join(DATA_PATH, "raw", "Promedios")


# ======================================================
# SUBCENTROS URBANOS CDMX (POLICENTRISMO)
# ======================================================

SUBCENTROS = {
    "Centro": (19.4326, -99.1332),
    "Polanco": (19.4330, -99.1960),
    "SantaFe": (19.3619, -99.2736),
    "Insurgentes": (19.4045, -99.1700),
    "DelValle": (19.3738, -99.1645),
    "Reforma": (19.4273, -99.1677)
}

# ======================================================
# HELPERS
# ======================================================
def clean_text(txt):
    
    """
    Normaliza cadenas de texto para homologación semántica.

    Este procedimiento reduce inconsistencias derivadas de diferencias
    ortográficas, uso de mayúsculas/minúsculas y acentuación en variables
    categóricas provenientes de múltiples fuentes institucionales.

    La normalización resulta necesaria para garantizar correspondencia
    correcta durante procesos de integración, agrupamiento y uniones
    entre bases heterogéneas.

    Parámetros
    ----------
    txt : str or object
        Cadena de entrada susceptible de normalización.

    Retorna
    -------
    str
        Texto estandarizado en minúsculas y sin caracteres acentuados.
    """

    if pd.isna(txt): return txt
    t = str(txt).lower().strip()
    t = t.translate(str.maketrans("áéíóú", "aeiou"))
    return t

def density_proxy(target_gdf, pts, r=1000):
    if len(pts) == 0: return np.zeros(len(target_gdf))
    tree = cKDTree(pts)
    coords = np.array([(p.x, p.y) for p in target_gdf.geometry])
    counts = tree.query_ball_point(coords, r)
    return np.array([len(c) for c in counts])

def nearest_distance(pts, coords):
    
    """
    Calcula la distancia euclidiana mínima entre observaciones y
    equipamiento urbano.

    Esta métrica operacionaliza accesibilidad espacial bajo el supuesto
    de fricción isotrópica, permitiendo aproximar el costo locacional
    asociado al acceso a infraestructura urbana.

    En ausencia de puntos de referencia, se asigna una penalización
    uniforme de distancia máxima.

    Parámetros
    ----------
    pts : ndarray
        Coordenadas del equipamiento urbano.
    coords : ndarray
        Coordenadas de observaciones inmobiliarias.

    Retorna
    -------
    ndarray
        Distancias mínimas expresadas en metros.
    """

    if len(pts) == 0: return np.ones(len(coords)) * 5000
    tree = cKDTree(pts)
    d, _ = tree.query(coords)
    return d

def line_to_points(gdf, step):
    
    """
    Discretiza geometrías lineales en una nube de puntos equidistantes.

    Este procedimiento transforma infraestructura lineal continua
    (como redes de transporte) en representaciones puntuales aptas
    para cálculo eficiente de proximidad y densidad espacial.

    La discretización preserva estructura geométrica general al tiempo
    que reduce complejidad computacional.

    Parámetros
    ----------
    gdf : GeoDataFrame
        Capa geoespacial de geometrías lineales.
    step : float
        Intervalo de muestreo en metros.

    Retorna
    -------
    ndarray
        Coordenadas discretizadas.
    """

    pts = []
    for geom in gdf.geometry:
        if geom.geom_type == "LineString":
            for d in np.arange(0, geom.length, step):
                p = geom.interpolate(d)
                pts.append((p.x, p.y))
    return np.array(pts)

def validar_y_relocalizar(df, path_alcaldias):
    
    """
    Detecta y corrige inconsistencias geoespaciales en registros
    inmobiliarios.

    El algoritmo identifica observaciones fuera de los límites
    geográficos del área de estudio y las relocaliza al centroide
    de su alcaldía declarada.

    Esta estrategia minimiza pérdida de información derivada de errores
    de geocodificación, preservando representatividad territorial.

    Parámetros
    ----------
    df : DataFrame
        Base de datos inmobiliaria.
    path_alcaldias : str
        Ruta al shapefile de delimitaciones administrativas.

    Retorna
    -------
    DataFrame
        Conjunto validado y espacialmente corregido.
    """

    print("="*80)
    print("VALIDACIÓN ESPACIAL CDMX")
    print("="*80)

    alcaldias = gpd.read_file(path_alcaldias).to_crs("EPSG:4326")

    col_alc = "NOMGEO"
    alcaldias["alc_clean"] = alcaldias[col_alc].apply(clean_text)
    df["alc_clean"] = df["alcaldia"].apply(clean_text)

# Delimitación geográfica de control para detección de registros
# georreferenciados fuera del polígono funcional de estudio.

    LAT_MIN, LAT_MAX = 19.04, 19.60
    LON_MIN, LON_MAX = -99.37, -98.94

    fuera_cdmx = (
        (df["latitud"] < LAT_MIN) |
        (df["latitud"] > LAT_MAX) |
        (df["longitud"] < LON_MIN) |
        (df["longitud"] > LON_MAX)
    )

    n_fuera = fuera_cdmx.sum()

    print(f"Registros realmente fuera de CDMX: {n_fuera}")

    if n_fuera > 0:
        centroides = alcaldias.copy()
        centroides["centroide"] = centroides.geometry.centroid

        mapa_centroides = dict(
            zip(centroides["alc_clean"], centroides["centroide"])
        )

        for idx in df[fuera_cdmx].index:
            alc = df.loc[idx, "alc_clean"]

            if alc in mapa_centroides:
                c = mapa_centroides[alc]
                df.loc[idx, "longitud"] = c.x
                df.loc[idx, "latitud"] = c.y

        print(f"✓ {n_fuera} registros relocalizados")
    else:
        print("✓ No se detectaron outliers espaciales")

    df.drop(columns=["alc_clean"], inplace=True, errors="ignore")

    return df

# ======================================================
# 1 CARGA
# ======================================================
print("="*80)
print("1/7 CARGA Y PROCESAMIENTO")
print("="*80)

df = pd.read_csv(path_viviendas)
df = df.dropna(subset=["latitud", "longitud", "price", "area", "alcaldia", "antiguedad"])
df = df[df.area >= 20].copy()

print("Registros iniciales:", len(df))

# Validación espacial
df = validar_y_relocalizar(df, path_alcaldias)

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitud, df.latitud), crs="EPSG:4326")
gdf_utm = gdf.to_crs(epsg=32614)
coords = np.array([(p.x, p.y) for p in gdf_utm.geometry])

# ======================================================
# 2 GEOFEATURES
# ======================================================
print("="*80)
print("2/7 GEOFEATURES")
print("="*80)

metro = gpd.read_file(path_metro).to_crs(epsg=32614)
metro_pts = line_to_points(metro, 100)
df["dist_metro_m"] = nearest_distance(metro_pts, coords)
df["density_metro"] = density_proxy(gdf_utm, metro_pts, 500)

tren = gpd.read_file(path_tren).to_crs(epsg=32614)
tren_pts = line_to_points(tren, 50)
df["dist_tren_m"] = nearest_distance(tren_pts, coords)
df["density_tren"] = density_proxy(gdf_utm, tren_pts, 500)

trole = gpd.read_file(path_trole).to_crs(epsg=32614)
trole_pts = np.array([(g.x, g.y) for g in trole.geometry if g.geom_type == "Point"])
df["dist_trole_m"] = nearest_distance(trole_pts, coords)
df["density_trole"] = density_proxy(gdf_utm, trole_pts, 300)

cable = gpd.read_file(path_cable).to_crs(epsg=32614)
cable_pts = line_to_points(cable, 100)
df["dist_cable_m"] = nearest_distance(cable_pts, coords)
df["density_cable"] = density_proxy(gdf_utm, cable_pts, 400)

# --- NUEVO BLOQUE: METROBÚS (ESTACIONES) ---
print("Procesando infraestructura de Metrobús...")
if os.path.exists(path_metrobus):
    mb_gdf = gpd.read_file(path_metrobus)
    if mb_gdf.crs is None: 
        mb_gdf = mb_gdf.set_crs("EPSG:4326")
    mb_gdf = mb_gdf.to_crs(epsg=32614)
    mb_gdf = mb_gdf[mb_gdf.geometry.notnull() & ~mb_gdf.geometry.is_empty]
    
    # Extraer puntos de las estaciones
    mb_pts = np.array([(g.x, g.y) for g in mb_gdf.geometry if g.geom_type == "Point"])
    
    if len(mb_pts) > 0:
        df["dist_metrobus_m"] = nearest_distance(mb_pts, coords)
        df["density_metrobus"] = density_proxy(gdf_utm, mb_pts, 600) # Radio de 600m para estaciones
        print(f"✓ Metrobús integrado ({len(mb_pts)} estaciones evaluadas)")
    else:
        df["dist_metrobus_m"] = 5000
        df["density_metrobus"] = 0
else:
    df["dist_metrobus_m"] = 5000
    df["density_metrobus"] = 0

com = gpd.read_file(path_comercio).to_crs(epsg=32614)
com_pts = np.array([(g.centroid.x, g.centroid.y) for g in com.geometry])
df["comercio_density"] = density_proxy(gdf_utm, com_pts, 1500)

seg = gpd.read_file(path_seguridad).to_crs("EPSG:4326")
tmp = gpd.sjoin(gdf, seg[['C_US', 'geometry']], how='left', predicate='covered_by')
df["marginalidad_score"] = pd.to_numeric(tmp["C_US"], errors="coerce").fillna(3)

if os.path.exists(path_turistas):
    tur = pd.read_csv(path_turistas)
    df["alc"] = df["alcaldia"].apply(clean_text)
    tur["alc"] = tur.iloc[:, 0].apply(clean_text)
    col = tur.columns[1]
    tur[col] = tur[col].astype(str).str.replace(",", "").str.replace("$", "")
    tur[col] = pd.to_numeric(tur[col], errors="coerce")
    tur = tur.groupby("alc")[col].mean().reset_index()
    tur.columns = ["alc", "total_turistas"]
    df = df.merge(tur, on="alc", how="left")
    df["total_turistas"] = df["total_turistas"].fillna(0)
    print("[OK] Variable de intensidad turística incorporada")
else:
    df["total_turistas"] = 0

import pyogrio

# Configuración para restaurar archivos .shx faltantes automáticamente
os.environ["SHAPE_RESTORE_SHX"] = "YES"

def safe_load_geodata(path, label, epsg=32614):
    
    """
    Realiza carga robusta de capas geoespaciales.

    El procedimiento incorpora validación de existencia, restauración
    automática de índices faltantes, homologación de sistema de referencia
    espacial y depuración de geometrías inválidas.

    Su propósito es asegurar consistencia topológica y continuidad
    operativa del pipeline analítico.

    Parámetros
    ----------
    path : str
        Ruta del archivo geoespacial.
    label : str
        Nombre descriptivo de la capa.
    epsg : int, optional
        Sistema de referencia espacial objetivo.

    Retorna
    -------
    GeoDataFrame or None
        Capa validada o valor nulo si la carga falla.
    """

    if not os.path.exists(path):
        print(f"Omitiendo {label}: Archivo no encontrado en {path}")
        return None
    try:
        # Cargamos el archivo
        gdf = gpd.read_file(path)

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        gdf = gdf.to_crs(epsg=epsg)
        # Limpieza de geometrías (Previene el error de 'NoneType' object has no attribute 'centroid')
        gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
        print(f"✓ {label} cargado correctamente ({len(gdf)} registros)")
        return gdf
    except Exception as e:
        print(f"⚠ Error al cargar {label}: {e}")
        return None

# Carga robusta de capas espaciales complementarias.
print("="*80)
print("2/7 GEOFEATURES: REGLA DE 15 MINUTOS (MODO ROBUSTO)")
print("="*80)

# Carga segura de cada capa
ciclo = safe_load_geodata(path_ciclovias, "Ciclovías")
verdes = safe_load_geodata(path_areas_verdes, "Áreas Verdes")
salud_base = safe_load_geodata(path_salud, "Salud")
salud_pub = safe_load_geodata(path_hospitales_publicos, "Hospitales Públicos 2020")
esc_pub = safe_load_geodata(path_escuelas_pub, "Escuelas Públicas")
esc_priv = safe_load_geodata(path_escuelas_priv, "Escuelas Privadas")

salud_final_pts = []

# --- PROCESAMIENTO DE PUNTOS ---
# Solo calculamos si la carga fue exitosa
if ciclo is not None:
    ciclo_pts = line_to_points(ciclo, 100)
    df["dist_ciclovia_m"] = nearest_distance(ciclo_pts, coords)
    df["densidad_ciclovia_15m"] = density_proxy(gdf_utm, ciclo_pts, 1200)

if verdes is not None:
    # Usamos centroid directamente para evitar el error previo
    verdes_pts = np.array([(p.x, p.y) for p in verdes.geometry.centroid])
    df["dist_area_verde_m"] = nearest_distance(verdes_pts, coords)
    df["densidad_parques_15m"] = density_proxy(gdf_utm, verdes_pts, 1200)

# Caso 1: Procesar la capa base si existe
if salud_base is not None:
    pts_base = np.array([(p.x, p.y) for p in salud_base.geometry])
    if len(pts_base) > 0:
        salud_final_pts.append(pts_base)

# Caso 2: Filtrar por entidad (CDMX) y remover duplicados espaciales de la nueva capa
if salud_pub is not None:
    # Filtrar estrictamente los registros correspondientes a la CDMX ("09")
    if "CLAVE_DE_L" in salud_pub.columns:
        salud_pub = salud_pub[salud_pub["CLAVE_DE_L"] == "09"].copy()
        print(f"✓ Hospitales Públicos filtrados para CDMX: {len(salud_pub)} registros")
    
    pts_pub = np.array([(p.x, p.y) for p in salud_pub.geometry])
    
    if len(pts_pub) > 0:
        # Si ya tenemos puntos en la lista base, removemos duplicados coexistentes por proximidad
        if len(salud_final_pts) > 0:
            pts_existentes = np.vstack(salud_final_pts)
            tree_existente = cKDTree(pts_existentes)
            
            # Consultamos si los nuevos puntos existen en un radio de 1 metro
            # (Aproximación de coincidencia exacta de coordenadas)
            indices_duplicados = tree_existente.query_ball_point(pts_pub, r=1.0)
            
            # Mantener solo los puntos cuyo conteo de vecinos sea 0 (sin coexistencia previa)
            filtrados = [pts_pub[i] for i, vecinos in enumerate(indices_duplicados) if len(vecinos) == 0]
            
            if len(filtrados) > 0:
                print(f"✓ Se añadieron {len(filtrados)} hospitales públicos únicos (se descartaron {len(pts_pub) - len(filtrados)} duplicados)")
                salud_final_pts.append(np.array(filtrados))
            else:
                print("⚠ Todos los puntos de la nueva capa eran coexistentes o duplicados.")
        else:
            # Si la capa base falló por completo, usamos la nueva directo
            salud_final_pts.append(pts_pub)

# Consolidar los arrays resultantes e integrarlos al dataframe principal
if salud_final_pts:
    salud_pts = np.vstack(salud_final_pts)
    df["dist_salud_m"] = nearest_distance(salud_pts, coords)
    df["acceso_salud_15m"] = density_proxy(gdf_utm, salud_pts, 1200)
    print(f"✓ Dataset unificado de salud listo. Total de nodos evaluados: {len(salud_pts)}")
else:
    # Fallback preventivo en caso de que ningún SHP se haya cargado con éxito
    df["dist_salud_m"] = 5000
    df["acceso_salud_15m"] = 0
    print("⚠ No se pudo construir la infraestructura de salud. Asignando valores default.")

# Para escuelas, combinamos solo si existen
esc_list = []
if esc_pub is not None: esc_list.append(np.array([(p.x, p.y) for p in esc_pub.geometry]))
if esc_priv is not None: esc_list.append(np.array([(p.x, p.y) for p in esc_priv.geometry]))

if esc_list:
    esc_pts = np.vstack(esc_list)
    df["dist_escuela_m"] = nearest_distance(esc_pts, coords)
    df["acceso_educacion_15m"] = density_proxy(gdf_utm, esc_pts, 1200)

# 5. CÁLCULO DEL SCORE DE 15 MINUTOS (Normalizado)
# Este índice suma la presencia de transporte, salud, educación y parques en el radio de 15 min
componentes_15min = [
    "density_metro", "densidad_ciclovia_15m", 
    "densidad_parques_15m", "acceso_salud_15m", "acceso_educacion_15m"
]
df["score_15min"] = df[componentes_15min].sum(axis=1)
df["score_15min"] = (df["score_15min"] - df["score_15min"].min()) / (df["score_15min"].max() - df["score_15min"].min())

print("✓ Score de 15 minutos calculado")

# Construcción de variables derivadas de proximidad urbana.
# 1. Transformación de proximidad percibida (Decaimiento Exponencial)
# Transformación exponencial de distancia.
# Modela el principio de decaimiento espacial, donde la utilidad
# marginal de la proximidad disminuye no linealmente conforme aumenta
# la distancia al equipamiento urbano.

df["prox_ciclovia"] = np.exp(-df["dist_ciclovia_m"] / 500)
df["prox_parque"] = np.exp(-df["dist_area_verde_m"] / 500)
df["prox_salud"] = np.exp(-df["dist_salud_m"] / 800)
df["prox_escuela"] = np.exp(-df["dist_escuela_m"] / 600)

# 2. Integración de Uso de Suelo (HM = Habitacional Mixto)
if os.path.exists(path_uso_suelo):
    uso = safe_load_geodata(path_uso_suelo, "Uso de Suelo")
    uso['es_mixto'] = uso['us_dscr'].str.contains('HM', na=False).astype(int)
    
    # Unión espacial para saber si la vivienda está en zona mixta
    gdf_temp = gpd.sjoin(gdf_utm, uso[['es_mixto', 'geometry']], how='left', predicate='within')
    # Eliminamos duplicados por si una vivienda cae en dos polígonos
    df['uso_mixto'] = gdf_temp.groupby(gdf_temp.index)['es_mixto'].max().fillna(0)
    print("✓ Variable de Uso Mixto integrada")

# ======================================================
# 3 FEATURES & SPATIAL LAG
# ======================================================
print("="*80)
print("3/7 FEATURES")
print("="*80)

df["price_m2_raw"] = df["price"] / df["area"]


def compute_local_listing_density(coords, radius=0.01):

    """
    Estima densidad espacial local de oferta inmobiliaria.

    La métrica captura intensidad de concentración del mercado
    mediante conteo de observaciones vecinas dentro de un radio
    predefinido.

    Esta variable funciona como aproximación a presión de mercado
    y competencia espacial.

    Parámetros
    ----------
    coords : ndarray
        Coordenadas geográficas.
    radius : float, optional
        Radio de vecindad.

    Retorna
    -------
    ndarray
        Conteo local de propiedades vecinas.
    """

    tree = cKDTree(coords)

    neighbors = tree.query_ball_point(coords, r=radius)

    density = np.array([
        len(n) - 1 for n in neighbors
    ])

    return density

coords_geo = df[["longitud", "latitud"]].values

df["listing_density_local"] = compute_local_listing_density(
    coords_geo,
    radius=0.015
)

df["listing_density_log"] = np.log1p(
    df["listing_density_local"]
)

df["precio_vecinal_local"] = np.nan

subcentros_gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(
        [lon for lat, lon in SUBCENTROS.values()],
        [lat for lat, lon in SUBCENTROS.values()]
    ),
    crs="EPSG:4326"
).to_crs(epsg=32614)

sub_pts = np.array([(p.x, p.y) for p in subcentros_gdf.geometry])

tree_sub = cKDTree(sub_pts)

dist_sub, _ = tree_sub.query(coords)

df["dist_nearest_subcenter_m"] = dist_sub
df["dist_subcenter_log"] = np.log1p(df["dist_nearest_subcenter_m"])

# ======================================================
# ÍNDICE ROBUSTO DE GENTRIFICACIÓN (PCA MULTIVARIADO)
# ======================================================

print("Calculando índice robusto de gentrificación...")

try:
    c10 = pd.read_csv(path_2010)
    c20 = pd.read_csv(path_2020)

    vars_soc = [
        "pct_educ_sup",   # escolaridad superior
        "pct_internet",   # conectividad
        "vph_pc",         # acceso digital
        "vph_autom",      # motorización
        "graproes",       # grado promedio escolar
        "pea",            # población económicamente activa
        "prom_ocup",      # ocupación por vivienda
        "pder_ss"         # derechohabiencia
    ]

    c10_g = c10.groupby("nom_mun")[vars_soc].mean().reset_index()
    c20_g = c20.groupby("nom_mun")[vars_soc].mean().reset_index()

    c10_g["alc"] = c10_g["nom_mun"].apply(clean_text)
    c20_g["alc"] = c20_g["nom_mun"].apply(clean_text)

    delta = c10_g.merge(c20_g, on="alc", suffixes=("_10", "_20"))

    for v in vars_soc:
        delta[f"d_{v}"] = delta[f"{v}_20"] - delta[f"{v}_10"]

    cols_delta = [f"d_{v}" for v in vars_soc]

    X_pca = StandardScaler().fit_transform(
        delta[cols_delta].fillna(0)
    )

    gentrif = PCA(n_components=1).fit_transform(X_pca).flatten()

    delta["gentrification_index"] = (
        gentrif - gentrif.min()
    ) / (
        gentrif.max() - gentrif.min()
    )

    delta["alc"] = delta["alc"].apply(clean_text)

    df = df.merge(
        delta[["alc", "gentrification_index"]],
        on="alc",
        how="left"
    )

    df["gentrification_index"] = df["gentrification_index"].fillna(
        df["gentrification_index"].median()
    )

    print("✓ Índice robusto calculado (8 variables + PCA)")

except Exception as e:
    print(f"⚠ Error en índice de gentrificación: {e}")
    df["gentrification_index"] = 0.5

print("Spatial lag será calculado dentro de cada fold (sin leakage)")
df["spatial_lag_price"] = np.nan

lux_cut = df["price_m2_raw"].quantile(.90)
df["is_luxury"] = (df["price_m2_raw"] >= lux_cut).astype(int)

# Interacciones
df["lag_x_area"] = df["spatial_lag_price"] * np.log1p(df["area"])

df["area_x_marginalidad"] = np.log1p(df["area"]) * df["marginalidad_score"]

df["area_X_gentrif"] = df["area"] * df["gentrification_index"]
df["gentrif_x_metro"] = df["gentrification_index"] * df["density_metro"]

# Nuevas interacciones que el modelo debe aprender
df["15min_X_gentrif"] = df["score_15min"] * df["gentrification_index"]

log_cols = [
    "dist_metro_m",
    "dist_metrobus_m",
    "dist_tren_m",
    "dist_trole_m",
    "dist_cable_m",
    "dist_ciclovia_m",
    "dist_area_verde_m",
    "dist_salud_m",
    "dist_escuela_m",
    "comercio_density",
    "total_turistas",
    "antiguedad"
]

for c in log_cols:
    if c in df.columns:
        df[c] = np.log1p(df[c])


df["verde_marginalidad_ratio"] = (
    df["densidad_parques_15m"] /
    (df["marginalidad_score"] + 1)
)

df["salud_marginalidad_ratio"] = (
    df["acceso_salud_15m"] /
    (df["marginalidad_score"] + 1)
)

df["educacion_marginalidad_ratio"] = (
    df["acceso_educacion_15m"] /
    (df["marginalidad_score"] + 1)
)
for c in [
    "verde_marginalidad_ratio",
    "salud_marginalidad_ratio",
    "educacion_marginalidad_ratio"
]:
    df[c] = np.log1p(df[c])
features = [
    "rooms", "area", "bathrooms", "parking_spaces", "antiguedad", "dist_subcenter_log", 
    "marginalidad_score", "comercio_density",
    "spatial_lag_price", "dist_metro_m", "density_metro", "dist_metrobus_m", "density_metrobus", "dist_tren_m", 
    "density_tren", "dist_trole_m", "density_trole", "dist_cable_m", 
    "density_cable", "total_turistas", "gentrification_index", 
    "lag_x_area", "area_x_marginalidad", 
    "area_X_gentrif", "gentrif_x_metro"
]

features += [
    "dist_ciclovia_m", "densidad_ciclovia_15m", 
    "dist_area_verde_m", "densidad_parques_15m",
    "dist_salud_m", "acceso_salud_15m", 
    "dist_escuela_m", "acceso_educacion_15m",
    "score_15min", 
    # Proximidades Transformadas (Estas son más potentes que las distancias puras)
    "prox_ciclovia", "prox_parque", "prox_salud", "prox_escuela",
    "uso_mixto",
    # Interacciones clave
    "15min_X_gentrif",
    "listing_density_log",
    "precio_vecinal_local",
    "verde_marginalidad_ratio",
    "salud_marginalidad_ratio",
    "educacion_marginalidad_ratio"
]


print("\n" + "="*80)
print("DEPURACIÓN AUTOMÁTICA DE FEATURES")
print("="*80)

# ----------------------------------------
# Eliminación automática por alta correlación
# ----------------------------------------

corr_matrix = df[features].corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

drop_cols = [
    column for column in upper.columns
    if any(upper[column] > 0.90)
]

if drop_cols:
    print(f"\nVariables eliminadas por colinealidad (>0.90):")
    for c in drop_cols:
        print(f" - {c}")

features = [f for f in features if f not in drop_cols]

print(f"\nFeatures finales: {len(features)}")

print("\nLISTA FINAL:")
for i, f in enumerate(features, 1):
    print(f"{i}. {f}")


def compute_spatial_lag(train_coords, train_prices, target_coords, k=10):
    
    """
    Calcula rezago espacial ponderado por distancia inversa.

    El rezago espacial aproxima externalidades locacionales y
    dependencia espacial del mercado inmobiliario mediante una
    media ponderada de precios vecinos.

    La ponderación decrece con la distancia, reflejando el principio
    de autocorrelación espacial.

    Su cálculo se restringe al conjunto de entrenamiento para evitar
    fuga de información.

    Parámetros
    ----------
    train_coords : ndarray
        Coordenadas de entrenamiento.
    train_prices : ndarray
        Valores observados.
    target_coords : ndarray
        Coordenadas objetivo.
    k : int, optional
        Número de vecinos considerados.

    Retorna
    -------
    ndarray
        Rezago espacial transformado logarítmicamente.
    """
    tree = cKDTree(train_coords)

    d, ix = tree.query(
    target_coords,
    k=min(k, len(train_coords))
    )  

    if k == 1:
        d = d.reshape(-1, 1)
        ix = ix.reshape(-1, 1)

    weights = 1 / (d + 1e-5)

    lag_values = []

    for idxs, w in zip(ix, weights):
        lag_values.append(
            np.average(train_prices[idxs], weights=w)
        )

    return np.log1p(lag_values)

def compute_local_neighbor_price(
    train_coords,
    train_prices,
    target_coords,
    k=15
):
    """
    Estima precio promedio vecinal local.

    Esta variable sintetiza condiciones microespaciales del entorno
    inmediato y captura patrones de homogeneidad intrabarrial.

    Constituye una aproximación al valor de referencia implícito
    observado por proximidad.

    Parámetros
    ----------
    train_coords : ndarray
        Coordenadas de entrenamiento.
    train_prices : ndarray
        Precios observados.
    target_coords : ndarray
        Coordenadas objetivo.
    k : int, optional
        Número de vecinos.

    Retorna
    -------
    ndarray
        Precio promedio local.
    """

    nbrs = NearestNeighbors(
        n_neighbors=min(k, len(train_coords))
    ).fit(train_coords)

    _, indices = nbrs.kneighbors(target_coords)

    return np.array([
        train_prices[idx].mean()
        for idx in indices
    ])

# ======================================================
# 4 FUNCIONES DE MODELADO (K-MEANS + ELASTICNET)
# ======================================================
def run_market_optimized(data, label):

    """
    Ejecuta ajuste segmentado del modelo ElasticNet espacial.

    El procedimiento divide el mercado inmobiliario en submercados
    homogéneos mediante clustering espacial y ajusta modelos
    regularizados independientes.

    Esta estrategia reconoce heterogeneidad estructural entre segmentos
    de mercado y mejora capacidad predictiva.

    La validación cruzada se realiza preservando integridad espacial
    de variables derivadas.

    Parámetros
    ----------
    data : DataFrame
        Datos del segmento.
    label : str
        Identificador del submercado.

    Retorna
    -------
    tuple
        Valores reales, predicciones y dataframe enriquecido.
    """

    print("\n" + "="*80)
    print(f"PROCESANDO ELASTICNET POR CLUSTERS: {label}")
    print("="*80)

    q_low, q_high = data["price_m2_raw"].quantile([0.05, 0.95])
    data = data[
        (data["price_m2_raw"] > q_low) &
        (data["price_m2_raw"] < q_high)
    ].copy()

    coords_for_cluster = StandardScaler().fit_transform(
        data[["latitud", "longitud"]]
    )

    if label == "MERCADO ESTÁNDAR":
        data["cluster"] = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        ).fit_predict(coords_for_cluster)

    else:
        enriched_cluster = StandardScaler().fit_transform(
            data[[
                "latitud",
                "longitud",
                "gentrification_index",
                "score_15min"
            ]]
        )

        data["cluster"] = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        ).fit_predict(enriched_cluster)

    data["precio_predicho"] = np.nan

    global_real = []
    global_pred = []

    for c in sorted(data.cluster.unique()):
        d_cluster = data[data.cluster == c].copy()

        if len(d_cluster) < 100:
            continue

        X_base = d_cluster[features].copy()
        y = np.log1p(d_cluster["price_m2_raw"])

        areas = d_cluster["area"].values
        coords_cluster = d_cluster[["longitud", "latitud"]].values

        kf = KFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        c_real = []
        c_pred = []

        for tr, te in kf.split(X_base):

            # =====================================================
            # Spatial lag SIN leakage
            # =====================================================
            lag_train = compute_spatial_lag(
                coords_cluster[tr],
                d_cluster.iloc[tr]["price_m2_raw"].values,
                coords_cluster[tr]
            )

            lag_test = compute_spatial_lag(
                coords_cluster[tr],
                d_cluster.iloc[tr]["price_m2_raw"].values,
                coords_cluster[te]
            )

            # =====================================================
            # Precio vecinal local SIN leakage
            # =====================================================
            neighbor_price_train = compute_local_neighbor_price(
                coords_cluster[tr],
                d_cluster.iloc[tr]["price_m2_raw"].values,
                coords_cluster[tr]
            )

            neighbor_price_test = compute_local_neighbor_price(
                coords_cluster[tr],
                d_cluster.iloc[tr]["price_m2_raw"].values,
                coords_cluster[te]
            )

            X_train = X_base.iloc[tr].copy()
            X_test = X_base.iloc[te].copy()

            # Asignar variables espaciales
            X_train["spatial_lag_price"] = lag_train
            X_test["spatial_lag_price"] = lag_test

            X_train["precio_vecinal_local"] = neighbor_price_train
            X_test["precio_vecinal_local"] = neighbor_price_test

            # Recalcular interacciones dependientes
            if "lag_x_area" in X_train.columns:
                X_train["lag_x_area"] = (
                    X_train["spatial_lag_price"] *
                    np.log1p(X_train["area"])
                )

                X_test["lag_x_area"] = (
                    X_test["spatial_lag_price"] *
                    np.log1p(X_test["area"])
                )

            # Manejo de NA
            X_train = X_train.fillna(X_train.median())
            X_test = X_test.fillna(X_train.median())

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("enet", ElasticNetCV(
                    alphas=np.logspace(-4, -1, 25),
                    l1_ratio=[0.4, 0.6, 0.8],
                    cv=5
                ))
            ])

            model.fit(X_train, y.iloc[tr])

            pred_m2 = np.expm1(
                model.predict(X_test)
            )

            pred_total = pred_m2 * areas[te]

            idx_test = d_cluster.iloc[te].index

            data.loc[idx_test, "precio_predicho"] = pred_total

            c_pred.extend(pred_total)
            c_real.extend(
                d_cluster["price"].iloc[te]
            )

        print(
            f"Cluster {c} | "
            f"R²: {r2_score(c_real, c_pred):.4f}"
        )

        global_real.extend(c_real)
        global_pred.extend(c_pred)

    return global_real, global_pred, data

# ======================================================
# 5 FUNCIÓN COMPARATIVA (k-NN)
# ======================================================
def run_knn_comparison(data, label):

    """
    Ejecuta modelo de referencia basado en vecindad espacial.

    Este procedimiento implementa un estimador no paramétrico
    como línea base comparativa frente al modelo hedónico regularizado.

    Su inclusión permite evaluar la ganancia explicativa derivada
    de la incorporación estructurada de teoría urbana.

    Parámetros
    ----------
    data : DataFrame
        Segmento de mercado.
    label : str
        Nombre del segmento.

    Retorna
    -------
    tuple
        Métricas de desempeño y predicciones.
    """

    print("\n" + "="*80)
    print(f"PROCESANDO COMPARATIVA k-NN: {label}")
    print("="*80)

    q_low, q_high = data["price_m2_raw"].quantile([0.05, 0.95])
    data = data[
        (data["price_m2_raw"] > q_low) &
        (data["price_m2_raw"] < q_high)
    ].copy()

    # ======================================
    # Prevención de fuga de información (data leakage).
    # Todas las variables espaciales dependientes de vecindad se recalculan
    # exclusivamente con información del conjunto de entrenamiento para
    # preservar validez inferencial.
    # quitar precio_vecinal precomputado
    # ======================================
    knn_features = [
        f for f in features
        if f != "precio_vecinal_local"
    ]

    X = data[knn_features].copy()
    y = np.log1p(data["price_m2_raw"])
    areas = data["area"].values
    coords = data[["longitud", "latitud"]].values

    # ======================================
    # BLOQUES ESPACIALES
    # ======================================
    spatial_blocks = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    ).fit_predict(coords)

    real, pred = [], []

    for fold in np.unique(spatial_blocks):

        tr = np.where(spatial_blocks != fold)[0]
        te = np.where(spatial_blocks == fold)[0]

        X_train = X.iloc[tr].copy()
        X_test = X.iloc[te].copy()

        # --------------------------------------
        # Spatial lag sin leakage
        # --------------------------------------
        lag_train = compute_spatial_lag(
            coords[tr],
            data.iloc[tr]["price_m2_raw"].values,
            coords[tr]
        )

        lag_test = compute_spatial_lag(
            coords[tr],
            data.iloc[tr]["price_m2_raw"].values,
            coords[te]
        )

        # --------------------------------------
        # Precio vecinal sin leakage
        # --------------------------------------
        neighbor_train = compute_local_neighbor_price(
            coords[tr],
            data.iloc[tr]["price_m2_raw"].values,
            coords[tr]
        )

        neighbor_test = compute_local_neighbor_price(
            coords[tr],
            data.iloc[tr]["price_m2_raw"].values,
            coords[te]
        )

        if "spatial_lag_price" in X_train.columns:
            X_train["spatial_lag_price"] = lag_train
            X_test["spatial_lag_price"] = lag_test

        X_train["precio_vecinal_local"] = neighbor_train
        X_test["precio_vecinal_local"] = neighbor_test

        if "lag_x_area" in X_train.columns:
            X_train["lag_x_area"] = (
                X_train["spatial_lag_price"] *
                np.log1p(X_train["area"])
            )

            X_test["lag_x_area"] = (
                X_test["spatial_lag_price"] *
                np.log1p(X_test["area"])
            )

        # imputación robusta
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())

        knn_model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(
                n_neighbors=12,
                weights="distance"
            ))
        ])

        knn_model.fit(X_train, y.iloc[tr])

        pred_m2 = np.expm1(
            knn_model.predict(X_test)
        )

        pred_total = pred_m2 * areas[te]

        pred.extend(pred_total)
        real.extend(data["price"].iloc[te])

    r2 = r2_score(real, pred)
    mape = mean_absolute_percentage_error(real, pred)

    print(f"   k-NN -> R²: {r2:.4f} | MAPE: {mape:.2%}")

    return r2, mape, real, pred

# ======================================================
# EJECUCIÓN FINAL
# ======================================================
re_est, pe_est, df_est = run_market_optimized(
    df[df.is_luxury == 0].copy(),
    "MERCADO ESTÁNDAR"
)

re_lux, pe_lux, df_lux = run_market_optimized(
    df[df.is_luxury == 1].copy(),
    "MERCADO LUJO"
)

rk_est, mk_est, real_est_kn, pred_est_kn = run_knn_comparison(
    df_est, "MERCADO ESTÁNDAR"
)

rk_lux, mk_lux, real_lux_kn, pred_lux_kn = run_knn_comparison(
    df_lux, "MERCADO LUJO"
)

# Consolidación ElasticNet
y_true_en = np.array(re_est + re_lux)
y_pred_en = np.array(pe_est + pe_lux)

y_true_kn = np.concatenate([real_est_kn, real_lux_kn])
y_pred_kn = np.concatenate([pred_est_kn, pred_lux_kn])

rmse_kn = np.sqrt(
    mean_squared_error(
        np.log1p(y_true_kn),
        np.log1p(y_pred_kn)
    )
)
# ======================================================
# VISUALIZACIÓN DE RESULTADOS
# ======================================================

# ---------- ElasticNet: Real vs Predicho ----------
fig, ax = plt.subplots(figsize=(10, 7))

sns.regplot(
    x=y_true_en,
    y=y_pred_en,
    ax=ax,
    scatter_kws={'alpha': 0.3, 'color': 'teal'},
    line_kws={'color': 'red'}
)

ax.set_title(
    f"ElasticNet + Clusters (Global)\nR²: {r2_score(y_true_en, y_pred_en):.3f}",
    fontsize=14
)
ax.set_xlabel("Precio Real", fontsize=12)
ax.set_ylabel("Precio Predicho", fontsize=12)

plt.savefig(
    os.path.join(FIGURES_PATH, "elasticnet_real_vs_predicho.png"),
    dpi=300,
    bbox_inches="tight"
)

if MOSTRAR_GRAFICAS:
    plt.tight_layout()
    plt.show()
else:
    plt.close('all')


# ---------- Distribución del error ----------
plt.figure(figsize=(10, 6))

error_en = ((y_pred_en - y_true_en) / y_true_en) * 100

sns.kdeplot(
    error_en,
    label="ElasticNet Clusters",
    fill=True,
    color="teal"
)

plt.axvline(0, color='black', linestyle='--')
plt.title("Distribución del Error Porcentual (%)", fontsize=14)
plt.xlabel("Error (%)")
plt.xlim(-100, 100)
plt.legend()

plt.savefig(
    os.path.join(FIGURES_PATH, "distribucion_error_elasticnet.png"),
    dpi=300,
    bbox_inches="tight"
)

if MOSTRAR_GRAFICAS:
    plt.tight_layout()
    plt.show()
else:
    plt.close('all')


# ======================================================
# RESUMEN COMPARATIVO FINAL
# ======================================================

r2_kn_global = np.mean([rk_est, rk_lux])
mape_kn_global = np.mean([mk_est, mk_lux])

print("\n" + "#"*80)
print("RESUMEN COMPARATIVO FINAL")
print("#"*80)
print(f"{'MÉTRICA':<20} | {'CLUSTER + ENET':<18} | {'k-NN (Vecindad)':<18}")
print("-" * 65)

print(f"{'R² Global':<20} | {r2_score(y_true_en, y_pred_en):.4f}{'':<13} | {r2_kn_global:.4f}")

print(f"{'MAPE Global':<20} | "
      f"{mean_absolute_percentage_error(y_true_en, y_pred_en)*100:.2f}%{'':<11} | "
      f"{mape_kn_global*100:.2f}%")
rmse_en = np.sqrt(
    mean_squared_error(np.log1p(y_true_en), np.log1p(y_pred_en))
)

print(f"{'RMSE Logarítmico':<20} | "
      f"{rmse_en:.4f}{'':<13} | {rmse_kn:.4f}")

print("#"*80)

# ======================================================
# EJEMPLO DE INTERPRETABILIDAD HEDÓNICA
# ======================================================
print("\n" + "="*80)
print("EJEMPLO DE INTERPRETABILIDAD HEDÓNICA")
print("="*80)
print("""
El modelo hedónico descompone el precio de la vivienda en sus atributos individuales. 
Dado que usamos ElasticNet con variables estandarizadas, el peso (coeficiente) 
representa el impacto marginal de cada característica.

EJEMPLO PRÁCTICO EN CDMX:
-------------------------
Si el coeficiente de 'metro_accessibility' fuera 0.15 y el de 'seguridad_score' fuera 0.20:

1. Atributo Locacional: Estar un 10% más cerca de una estación de metro incrementa el 
   valor de la propiedad en un ratio determinado, validando la teoría de Renta de Suelo.
   
2. Atributo Ambiental: La seguridad actúa como un 'bien superior'. En el Mercado de Lujo, 
   un incremento en el score de seguridad tiene un impacto porcentual mayor que en el 
   Mercado Estándar, lo que sugiere una alta disposición a pagar por certidumbre física.

3. Interacciones: La variable 'area_X_gentrif' permite observar si el tamaño de la 
   vivienda se premia más en colonias con procesos de gentrificación activos, 
   capturando la plusvalía especulativa de la zona.

Esta capacidad de separar 'cuánto vale el baño' vs 'cuánto vale la zona' es lo que 
fundamenta científicamente este Trabajo Terminal.
""")

# ======================================================
# ANÁLISIS AMPLIADO DE GENTRIFICACIÓN Y EVOLUCIÓN 2022-2025
# ======================================================
print("\n" + "="*80)
print("ANÁLISIS AMPLIADO DE GENTRIFICACIÓN")
print("="*80)

import glob


# ------------------------------------------------------
# 1. CARGA DE DATOS 2022
# ------------------------------------------------------
archivos = glob.glob(os.path.join(carpeta_promedios, "*.csv"))
data_2022 = []

for archivo in archivos:
    nombre_archivo = os.path.basename(archivo)

    alcaldia = (
        nombre_archivo.replace("Promedios_id_", "")
        .replace(".csv", "")
        .replace("_", " ")
        .strip()
        .title()
    )

    df_temp = pd.read_csv(archivo)

    if "precio por metro cuadrado" not in df_temp.columns:
        continue

    df_temp = df_temp[df_temp["precio por metro cuadrado"] > 0]

    data_2022.append({
        "alcaldia": alcaldia,
        "precio_2022": df_temp["precio por metro cuadrado"].mean(),
        "n_2022": df_temp["precio por metro cuadrado"].count()
    })

df_2022 = pd.DataFrame(data_2022)

# ------------------------------------------------------
# 2. NORMALIZACIÓN DE ALCALDÍAS
# ------------------------------------------------------
mapeo_alcaldias = {
    "Alvaro Obregon": "Álvaro Obregón",
    "Benito Juarez": "Benito Juárez",
    "Coyoacan": "Coyoacán",
    "Cuajimalpa": "Cuajimalpa de Morelos",
    "Cuauhtemoc": "Cuauhtémoc",
    "Gam": "Gustavo A. Madero",
    "Magdalena Contreras": "La Magdalena Contreras",
    "Milpa": "Milpa Alta",
    "Tlahuac": "Tláhuac"
}

df_2022["alcaldia"] = df_2022["alcaldia"].replace(mapeo_alcaldias)

# ------------------------------------------------------
# 3. PROMEDIOS 2025
# ------------------------------------------------------
df_2025 = df.copy()

df_2025["precio_m2_2025"] = df_2025["price"] / df_2025["area"]

df_2025_prom = df_2025.groupby("alcaldia", as_index=False).agg(
    precio_m2_2025=("precio_m2_2025", "mean"),
    n_2025=("precio_m2_2025", "count")
)

# ------------------------------------------------------
# 4. COMPARACIÓN
# ------------------------------------------------------
comparacion = pd.merge(df_2022, df_2025_prom, on="alcaldia", how="inner")

comparacion["incremento_%"] = (
    (comparacion["precio_m2_2025"] - comparacion["precio_2022"])
    / comparacion["precio_2022"]
) * 100

comparacion["alc"] = comparacion["alcaldia"].apply(clean_text)

comparacion = comparacion.merge(
    delta[["alc", "gentrification_index"]],
    on="alc",
    how="left"
)

comparacion = comparacion.sort_values(
    "gentrification_index",
    ascending=False
)

# ------------------------------------------------------
# 5. IMPRESIÓN EN CONSOLA
# ------------------------------------------------------
print("\nRANKING DE GENTRIFICACIÓN + EVOLUCIÓN DE PRECIOS\n")
print(comparacion[[
    "alcaldia",
    "precio_2022",
    "precio_m2_2025",
    "incremento_%",
    "gentrification_index",
    "n_2022",
    "n_2025"
]].to_string(index=False))

# ------------------------------------------------------
# 6. GRÁFICA 1: RANKING GENTRIFICACIÓN
# ------------------------------------------------------
plt.figure(figsize=(12,8))

sns.barplot(
    data=comparacion,
    x="gentrification_index",
    y="alcaldia"
)

plt.title("Ranking de Gentrificación por Alcaldía")
plt.xlabel("Índice Normalizado")
plt.ylabel("Alcaldía")
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_PATH, "ranking_gentrificacion.png"),
    dpi=300,
    bbox_inches="tight"
)

if MOSTRAR_GRAFICAS:
    plt.tight_layout()
    plt.show()
else:
    print("\n[INFO] Modo silencioso: Gráficas omitidas. Cambia MOSTRAR_GRAFICAS a True para visualizarlas.")
    plt.close('all') # Cerramos las figuras en memoria para ahorrar RAM

# ------------------------------------------------------
# 7. GRÁFICA 2: GENTRIFICACIÓN VS INCREMENTO
# ------------------------------------------------------
plt.figure(figsize=(10,7))

sns.regplot(
    data=comparacion,
    x="gentrification_index",
    y="incremento_%"
)

for _, row in comparacion.iterrows():
    plt.text(
        row["gentrification_index"],
        row["incremento_%"],
        row["alcaldia"],
        fontsize=8
    )

plt.title("Relación entre Gentrificación e Incremento de Precio")
plt.xlabel("Índice de Gentrificación")
plt.ylabel("Incremento % 2022-2025")
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_PATH, "impacto_gentrificacion.png"),
    dpi=300,
    bbox_inches="tight"
)

if MOSTRAR_GRAFICAS:
    plt.tight_layout()
    plt.show()
else:
    print("\n[INFO] Modo silencioso: Gráficas omitidas. Cambia MOSTRAR_GRAFICAS a True para visualizarlas.")
    plt.close('all') # Cerramos las figuras en memoria para ahorrar RAM
# ------------------------------------------------------
# 8. GRÁFICA 3: EVOLUCIÓN INTERNA
# ------------------------------------------------------
plt.figure(figsize=(11,8))

scatter = plt.scatter(
    comparacion["precio_2022"],
    comparacion["precio_m2_2025"],
    c=comparacion["gentrification_index"],
    s=180,
    alpha=0.85
)

for _, row in comparacion.iterrows():
    plt.text(
        row["precio_2022"],
        row["precio_m2_2025"],
        row["alcaldia"],
        fontsize=8
    )

mx = max(
    comparacion["precio_2022"].max(),
    comparacion["precio_m2_2025"].max()
)

plt.plot([0, mx], [0, mx], 'r--')

plt.colorbar(scatter, label="Índice de Gentrificación")

plt.title("Evolución Interna de Precios por Alcaldía")
plt.xlabel("Precio promedio m² 2022")
plt.ylabel("Precio promedio m² 2025")
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_PATH, "evolucion_precios_2022_2025.png"),
    dpi=300,
    bbox_inches="tight"
)

if MOSTRAR_GRAFICAS:
    plt.tight_layout()
    plt.show()
else:
    print("\n[INFO] Modo silencioso: Gráficas omitidas. Cambia MOSTRAR_GRAFICAS a True para visualizarlas.")
    plt.close('all') # Cerramos las figuras en memoria para ahorrar RAM

# ------------------------------------------------------
# 9. GRÁFICA 4: INCREMENTO ORDENADO
# ------------------------------------------------------
comparacion_sorted = comparacion.sort_values("incremento_%", ascending=False)

plt.figure(figsize=(12,8))

colores = np.where(
    comparacion_sorted["incremento_%"] >= 0,
    "forestgreen",
    "firebrick"
)

plt.barh(
    comparacion_sorted["alcaldia"],
    comparacion_sorted["incremento_%"],
    color=colores
)

plt.title("Incremento % del Precio por m² (2022–2025)")
plt.xlabel("Incremento (%)")
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_PATH, "incremento_precio_alcaldia.png"),
    dpi=300,
    bbox_inches="tight"
)

if MOSTRAR_GRAFICAS:
    plt.tight_layout()
    plt.show()
else:
    print("\n[INFO] Modo silencioso: Gráficas omitidas. Cambia MOSTRAR_GRAFICAS a True para visualizarlas.")
    plt.close('all') # Cerramos las figuras en memoria para ahorrar RAM

# ------------------------------------------------------
# 10. CORRELACIÓN
# ------------------------------------------------------
corr = comparacion["gentrification_index"].corr(
    comparacion["incremento_%"]
)

print("\n" + "="*80)
print(f"CORRELACIÓN GENTRIFICACIÓN vs INCREMENTO PRECIO: {corr:.4f}")
print("="*80)

# ------------------------------------------------------
# 11. EXPORTACIÓN
# ------------------------------------------------------
ruta_salida = os.path.join(OUTPUTS_PATH, "results", "comparacion_gentrificacion.csv")
comparacion.to_csv(ruta_salida, index=False)

print(f"\nArchivo exportado en:\n{ruta_salida}")

# ======================================================
# INTERPRETABILIDAD HEDÓNICA REAL DEL ELASTICNET
# ======================================================
print("\n" + "="*80)
print("INTERPRETABILIDAD HEDÓNICA CUANTITATIVA")
print("="*80)

def fit_interpretable_elasticnet(data, label):

    """
    Ajusta versión interpretable del modelo ElasticNet.

    El objetivo es extraer coeficientes hedónicos directamente
    interpretables para análisis económico marginal.

    Este ajuste permite identificar elasticidades implícitas
    asociadas a atributos físicos, locacionales y socioespaciales.

    Parámetros
    ----------
    data : DataFrame
        Datos del segmento.
    label : str
        Identificador del mercado.

    Retorna
    -------
    tuple
        Coeficientes, modelo ajustado y escalador.
    """

    print("\n" + "="*80)
    print(f"TOP 15 VARIABLES HEDÓNICAS: {label}")
    print("="*80)

    data = data.copy()

    # --------------------------------------------------
    # Recalcular variables espaciales estáticas
    # --------------------------------------------------
    coords = data[["longitud", "latitud"]].values

    data["spatial_lag_price"] = compute_spatial_lag(
        coords,
        data["price_m2_raw"].values,
        coords
    )

    data["precio_vecinal_local"] = compute_local_neighbor_price(
        coords,
        data["price_m2_raw"].values,
        coords
    )

    if "lag_x_area" in data.columns:
        data["lag_x_area"] = (
            data["spatial_lag_price"] *
            np.log1p(data["area"])
        )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    X = data[features].copy()
    y = np.log1p(data["price_m2_raw"])

    # Limpieza robusta
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # Si alguna columna sigue completamente NaN
    X = X.fillna(0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = ElasticNetCV(
        alphas=np.logspace(-4, -1, 25),
        l1_ratio=[0.4, 0.6, 0.8],
        cv=5
    )

    model.fit(Xs, y)

    coef_df = pd.DataFrame({
        "feature": features,
        "coef": model.coef_
    })

    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values(
        "abs_coef",
        ascending=False
    )

    print(coef_df[["feature", "coef"]].head(15).to_string(index=False))

    return coef_df, model, scaler


# ------------------------------------------------------
# 1. AJUSTAR MODELOS INTERPRETABLES
# ------------------------------------------------------
# ======================================================
# PRUEBA RÁPIDA SIN SPATIAL LAG (SOLO INTERPRETABILIDAD)
# ======================================================

features_original = features.copy()

features_sin_spatial = [
    f for f in features
    if f not in ["spatial_lag_price", "lag_x_area"]
]

print("\n" + "="*80)
print("ANÁLISIS AUXILIAR DE SENSIBILIDAD: EVALUACIÓN INTERPRETATIVA EXCLUYENDO REZAGOS ESPACIALES")
print("="*80)
print(f"Features activas: {len(features_sin_spatial)}")

# cambiar temporalmente
features = features_sin_spatial

coef_est, model_est, scaler_est = fit_interpretable_elasticnet(
    df[df.is_luxury == 0],
    "MERCADO ESTÁNDAR (SIN SPATIAL)"
)

coef_lux, model_lux, scaler_lux = fit_interpretable_elasticnet(
    df[df.is_luxury == 1],
    "MERCADO LUJO (SIN SPATIAL)"
)

# restaurar
features = features_original

coef_est, model_est, scaler_est = fit_interpretable_elasticnet(
    df[df.is_luxury == 0],
    "MERCADO ESTÁNDAR"
)

coef_lux, model_lux, scaler_lux = fit_interpretable_elasticnet(
    df[df.is_luxury == 1],
    "MERCADO LUJO"
)

# ------------------------------------------------------
# 2. GRÁFICA DE IMPORTANCIA
# ------------------------------------------------------
def plot_coef(coef_df, titulo):
    top = coef_df.head(15)

    plt.figure(figsize=(12,8))

    colores = ["forestgreen" if x > 0 else "firebrick" for x in top["coef"]]

    plt.barh(
        top["feature"],
        top["coef"],
        color=colores
    )

    plt.title(titulo)
    plt.xlabel("Coeficiente ElasticNet")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    nombre_archivo = titulo.lower().replace(" ", "_") + ".png"

    plt.savefig(
        os.path.join(FIGURES_PATH, nombre_archivo),
        dpi=300,
        bbox_inches="tight"
    )
    if MOSTRAR_GRAFICAS:
        plt.tight_layout()
        plt.show()
    else:
        print("\n[INFO] Modo silencioso: Gráficas omitidas. Cambia MOSTRAR_GRAFICAS a True para visualizarlas.")
        plt.close('all') # Cerramos las figuras en memoria para ahorrar RAM

plot_coef(
    coef_est,
    "Impacto Hedónico - Mercado Estándar"
)

plot_coef(
    coef_lux,
    "Impacto Hedónico - Mercado Lujo"
)

# ------------------------------------------------------
# 3. TRADUCCIÓN ECONÓMICA
# ------------------------------------------------------
def interpret_feature_impact(coef_df, market_name):

    """
    Traduce coeficientes estandarizados a interpretación económica.

    Convierte efectos lineales estimados en variaciones porcentuales
    aproximadas del valor inmobiliario.

    Esto facilita comunicación de hallazgos hacia interpretación
    hedónica aplicada.

    Parámetros
    ----------
    coef_df : DataFrame
        Coeficientes del modelo.
    market_name : str
        Segmento analizado.
    """

    print("\n" + "-"*80)
    print(f"INTERPRETACIÓN ECONÓMICA: {market_name}")
    print("-"*80)

    top = coef_df.head(10)

    for _, row in top.iterrows():
        feat = row["feature"]
        beta = row["coef"]

        pct = (np.exp(beta) - 1) * 100

        if beta > 0:
            efecto = "incrementa"
        else:
            efecto = "reduce"

        print(
            f"{feat:<25} → {efecto:10} ≈ {abs(pct):.2f}% "
            f"por +1 desviación estándar"
        )

interpret_feature_impact(coef_est, "MERCADO ESTÁNDAR")
interpret_feature_impact(coef_lux, "MERCADO LUJO")

# ------------------------------------------------------
# 4. SIMULADOR HEDÓNICO
# ------------------------------------------------------
print("\n" + "="*80)
print("SIMULADOR DE IMPACTO MARGINAL")
print("="*80)

variables_clave = [
    
    "seguridad_score",
    "area",
    "gentrification_index",
    "spatial_lag_price"
]

for var in variables_clave:
    if var in coef_est["feature"].values:
        beta = coef_est.loc[
            coef_est["feature"] == var,
            "coef"
        ].values[0]

        impacto_10 = (np.exp(beta * 0.1) - 1) * 100

        print(
            f"Si '{var}' mejora 10%, "
            f"el precio esperado cambia ≈ {impacto_10:.2f}%"
        )

# ------------------------------------------------------
# 5. COMPARATIVA LUJO VS ESTÁNDAR
# ------------------------------------------------------
comp_coef = coef_est[["feature", "coef"]].merge(
    coef_lux[["feature", "coef"]],
    on="feature",
    suffixes=("_estandar", "_lujo")
)

comp_coef["diferencia"] = (
    comp_coef["coef_lujo"] - comp_coef["coef_estandar"]
)

comp_coef = comp_coef.sort_values(
    "diferencia",
    key=np.abs,
    ascending=False
)

print("\n" + "="*80)
print("VARIABLES MÁS DIFERENTES ENTRE MERCADOS")
print("="*80)

print(comp_coef.head(15).to_string(index=False))

# ------------------------------------------------------
# 6. VISUALIZACIÓN COMPARATIVA
# ------------------------------------------------------
top_diff = comp_coef.head(12)

x = np.arange(len(top_diff))
width = 0.35

plt.figure(figsize=(14,8))

plt.bar(
    x - width/2,
    top_diff["coef_estandar"],
    width,
    label="Mercado Estándar"
)

plt.bar(
    x + width/2,
    top_diff["coef_lujo"],
    width,
    label="Mercado Lujo"
)

plt.xticks(
    x,
    top_diff["feature"],
    rotation=45,
    ha="right"
)

plt.title("Comparativa Hedónica entre Segmentos")
plt.ylabel("Coeficiente")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_PATH, "comparativa_hedonica_segmentos.png"),
    dpi=300,
    bbox_inches="tight"
)


if MOSTRAR_GRAFICAS:
    plt.tight_layout()
    plt.show()
else:
    print("\n[INFO] Modo silencioso: Gráficas omitidas. Cambia MOSTRAR_GRAFICAS a True para visualizarlas.")
    plt.close('all') # Cerramos las figuras en memoria para ahorrar RAM

# ------------------------------------------------------
# 7. TEXTO AUTOMÁTICO 
# ------------------------------------------------------
print("\n" + "="*80)
print("INTERPRETACIÓN AUTOMÁTICA")
print("="*80)

top_pos = coef_est.loc[coef_est["coef"].idxmax()]
top_neg = coef_est.loc[coef_est["coef"].idxmin()]

print(f"""
En el mercado estándar, la variable con mayor efecto positivo fue 
'{top_pos['feature']}', sugiriendo que incrementos en este atributo
se asocian con valorizaciones importantes del inmueble.

Por otro lado, '{top_neg['feature']}' presentó el mayor efecto
depreciativo, evidenciando una penalización hedónica significativa.

La comparación entre segmentos muestra que ciertos atributos urbanos
poseen distinta elasticidad marginal según el estrato de mercado,
lo que confirma la existencia de submercados inmobiliarios con
estructuras de valoración diferenciadas.
""")

# CÁLCULO DE PRESIÓN DE GENTRIFICACIÓN LOCAL (K-Neighbors)

def calcular_gentrificacion_local(df_puntos):

    """
    Estima presión local de gentrificación a escala microterritorial.

    El indicador compara el valor unitario de cada propiedad frente
    al promedio de su vecindad inmediata.

    Valores superiores a la unidad sugieren presión alcista local
    potencialmente asociada a procesos de sustitución socioespacial.

    Esta métrica complementa el índice macro de transformación
    socioeconómica a nivel alcaldía.

    Parámetros
    ----------
    df_puntos : DataFrame
        Observaciones inmobiliarias georreferenciadas.

    Retorna
    -------
    list
        Índices locales de presión de valorización.
    """
    
    coords = df_puntos[['longitud', 'latitud']].values
    
    # Buscamos los 20 vecinos más cercanos para cada punto
    nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    gentrificacion_local = []
    
    for i in range(len(df_puntos)):
        vecinos_idx = indices[i]
        # El precio de la propiedad vs el promedio de sus 20 vecinos
        precio_propio = df_puntos.iloc[i]['price_m2_raw']
        precio_vecindario = df_puntos.iloc[vecinos_idx]['price_m2_raw'].mean()
        
        # Ratio de presión: Si es > 1, la propiedad está "pujando" el precio al alza
        ratio_presion = precio_propio / precio_vecindario
        gentrificacion_local.append(ratio_presion)
        
    return gentrificacion_local

# ======================================================
# GENTRIFICACIÓN LOCAL Y EXPORTACIÓN PARA QGIS
# ======================================================
print("\n" + "="*80)
print("PROCESANDO MÉTRICAS LOCALES Y EXPORTACIÓN")
print("="*80)

# 1. Consolidar los dataframes de los dos mercados
df_qgis = pd.concat([df_est, df_lux])

# 2. Función de Gentrificación Local (Mejorada para Tesis)
def calcular_gentrificacion_local(df_puntos):
    coords = df_puntos[['longitud', 'latitud']].values

    # Buscamos los 20 vecinos más cercanos
    nbrs = NearestNeighbors(
        n_neighbors=21,
        algorithm='ball_tree'
    ).fit(coords)

    distances, indices = nbrs.kneighbors(coords)

    gentrif_local = []

    for i in range(len(df_puntos)):

        # Excluimos el propio punto
        vecinos_idx = indices[i][1:]

        precio_propio = df_puntos.iloc[i]['price_m2_raw']

        precio_vecindario = (
            df_puntos.iloc[vecinos_idx]['price_m2_raw']
            .mean()
        )

        # Ratio relativo
        ratio = precio_propio / (precio_vecindario + 1e-5)

        # Limitar extremos para estabilidad
        ratio = np.clip(ratio, 0, 5)

        # Transformación estable 0-1
        score = ratio / (1 + ratio)

        gentrif_local.append(score)

    return gentrif_local


# Aplicar cálculo
df_qgis['gentrif_local_presion'] = (
    calcular_gentrificacion_local(df_qgis)
)

# 4. Crear métrica combinada (Macro + Micro)
# Esto une el índice de la alcaldía con la presión del punto específico
df_qgis['gentrif_map_score'] = (
    0.75 * df_qgis['gentrification_index']
    + 0.25 * df_qgis['gentrif_local_presion']
)

# 5. Limpieza y formato final
df_qgis["error_pct"] = ((df_qgis["precio_predicho"] - df_qgis["price"]) / df_qgis["price"]) * 100
df_qgis["segmento"] = np.where(df_qgis["is_luxury"] == 1, "Lujo", "Estandar")

ruta_qgis = os.path.join(OUTPUTS_PATH, "results", "resultados_qgis_final.csv")
df_qgis.to_csv(ruta_qgis, index=False, encoding="utf-8-sig")

print(f"✓ Archivo con métrica local generado en: {ruta_qgis}")
print(f"✓ Total de puntos exportados: {len(df_qgis)}")

# 1. Visualización de Precisión Global
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_true_kn, y=y_pred_kn, alpha=0.5, color='royalblue')
plt.plot([y_true_kn.min(), y_true_kn.max()], [y_true_kn.min(), y_true_kn.max()], '--r', linewidth=2)
plt.title(f'Desempeño del Modelo k-NN (R²: {r2_kn_global:.4f})', fontsize=14)
plt.xlabel('Precio Real (m²)', fontsize=12)
plt.ylabel('Precio Predicho (m²)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(FIGURES_PATH, "precision_modelo.png"))

# 2. Impacto de Gentrificación
plt.figure(figsize=(12, 7))
sns.regplot(data=comparacion, x="gentrification_index", y="incremento_%", 
            scatter_kws={'s':100, 'color':'teal'}, line_kws={'color':'orange'})
for i in range(comparacion.shape[0]):
    plt.text(comparacion.gentrification_index[i]+0.01, comparacion["incremento_%"][i], 
             comparacion.alcaldia[i], fontsize=9)
plt.title('Impacto de la Gentrificación en la Plusvalía (2022-2025)', fontsize=14)
plt.xlabel('Índice de Gentrificación (Normalizado)', fontsize=12)
plt.ylabel('Incremento de Precio (%)', fontsize=12)
plt.savefig(os.path.join(FIGURES_PATH, "impacto_gentrificacion.png"))

# 3. Comparativa de Importancia de Variables (Estandar vs Lujo)
top_10_diff = comp_coef.head(10)
top_10_diff.plot(x='feature', y=['coef_estandar', 'coef_lujo'], kind='barh', figsize=(12, 8))
plt.title('Diferencia de Pesos Hedónicos por Segmento de Mercado', fontsize=14)
plt.xlabel('Coeficiente (Impacto en Precio)', fontsize=12)
plt.grid(axis='x', alpha=0.3)
plt.savefig(os.path.join(FIGURES_PATH, "comparativa_mercados.png"))

print(f"\n✓ Gráficas de defensa generadas exitosamente en: {FIGURES_PATH}")

