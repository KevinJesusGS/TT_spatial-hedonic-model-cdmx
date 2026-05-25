# =============================================================================
# TRABAJO TERMINAL
# Modelo Hedónico-Espacial de Listings Airbnb — CDMX
#
# Autores: Kevin Jesús González Sosa, José Manuel Torres Gutiérrez
# Institución: Escuela Superior de Cómputo
# Fecha: Mayo, 2026
#
# Descripción:
#   Pipeline completo que:
#     1. Limpia y estandariza el dataset de Airbnb (listings.csv)
#     2. Expande la columna 'amenities' en features binarias y de score
#     3. Construye features geoespaciales urbanas (metro, 15 min, etc.)
#     4. Ajusta el modelo hedónico-espacial (ElasticNet + k-NN)
#        con la misma lógica que el modelo de ventas/rentas
#
# Variable objetivo principal: precio_noche (MXN/noche)
# Variable secundaria opcional: estimated_revenue_l365d (MXN/año)
#
# Diferencias respecto al modelo de ventas:
#   • No se requiere geocodificación (lat/lon ya disponibles)
#   • 'amenities' se transforma en features binarias + índice compuesto
#   • Se incorporan features propias de Airbnb:
#       - host_is_superhost, instant_bookable
#       - review_scores_* (6 dimensiones de calidad percibida)
#       - occupancy_rate, availability_365
#       - room_type / property_type codificados
#   • La segmentación lujo se basa en percentil 90 del precio/noche
#   • No hay datos INEGI de gentrificación por ageb (se usa a nivel alcaldía)
# =============================================================================

# =============================================================================
# SUPUESTOS METODOLÓGICOS
#
# 1. La distancia euclidiana aproxima accesibilidad espacial.
# 2. El mercado Airbnb presenta dependencia espacial local de precios.
# 3. Existen submercados diferenciados por tipo de propiedad y zona.
# 4. Los efectos hedónicos pueden modelarse linealmente tras log-transform.
# 5. Las amenidades actúan como bienes de diferenciación con efectos marginales.
# 6. Las reseñas proxy calidad percibida y afectan el precio de equilibrio.
# =============================================================================

import ast
import os
import glob
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import libpysal
from esda.moran import Moran

warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURACIÓN
# ======================================================
MOSTRAR_GRAFICAS = False   # True para ver gráficas en pantalla

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# ======================================================
# RUTAS
# ======================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(PROJECT_ROOT, "..", "data")
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "..", "outputs", "airbnb")
RESULTS_PATH = os.path.join(OUTPUTS_PATH, "results")
FIGURES_PATH = os.path.join(OUTPUTS_PATH, "figures")

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

# Dataset fuente (Inside Airbnb / scraping propio)
PATH_LISTINGS = os.path.join(DATA_PATH, "raw", "airbnb", "listings.csv")

# Capas geoespaciales — mismas que modelos de ventas y rentas
path_metro      = os.path.join(DATA_PATH, "raw", "stcmetro_shp",
                               "STC_Metro_estaciones_utm14n.shp")
path_metrobus   = os.path.join(DATA_PATH, "raw", "mb_shp", "Metrobus_estaciones.shp")
path_seguridad  = os.path.join(DATA_PATH, "raw", "crimen", "urbanismo_social_sintesis.shp")
path_comercio   = os.path.join(DATA_PATH, "raw", "cypc", "C_PComerciales.shp")
path_turistas   = os.path.join(DATA_PATH, "raw", "Turistas", "turistas_alcaldia.csv")
path_tren       = os.path.join(DATA_PATH, "raw", "ste_shp", "ste_tren_ligero_shp",
                               "ste_tren_ligero_shp", "STE_TrenLigero_estaciones_utm14n.shp")
path_trole      = os.path.join(DATA_PATH, "raw", "ste_shp", "ste_trolebus_shp",
                               "ste_trolebus_shp", "STE_Trolebus_Paradas.shp")
path_cable      = os.path.join(DATA_PATH, "raw", "ste_shp", "ste_cablebus_shp",
                               "ste_cablebus_shp", "STE_Cablebus_estaciones.shp")
path_alcaldias  = os.path.join(DATA_PATH, "raw", "alcaldias",
                               "poligonos_alcaldias_cdmx.shp")
path_ciclovias  = os.path.join(DATA_PATH, "raw", "infraestructura_vial_ciclista",
                               "Infraestructura ciclista total.shp")
path_areas_verdes = os.path.join(DATA_PATH, "raw", "inventario_areas_verdes_1",
                                 "inventario_areas_verdes_1.shp")
path_salud      = os.path.join(DATA_PATH, "raw", "hospitales_y_centros_de_salud",
                               "hospitales_y_centros_de_salud.shp")
path_hospitales_publicos = os.path.join(DATA_PATH, "raw", "hospitales_2020_publicos",
                                        "hospitales_2020_publicos.shp")
path_escuelas_pub  = os.path.join(DATA_PATH, "raw", "escuelas_publicas",
                                  "escuelas_publicas.shp")
path_escuelas_priv = os.path.join(DATA_PATH, "raw", "escuelas_privadas",
                                  "escuelas_privadas.shp")
path_uso_suelo     = os.path.join(DATA_PATH, "raw", "uso-de-suelo", "uso-de-suelo.shp")
path_catastro      = os.path.join(DATA_PATH, "raw", "Catastrales")
path_2010          = os.path.join(DATA_PATH, "raw", "INEGI", "inegi_2010.csv")
path_2020          = os.path.join(DATA_PATH, "raw", "INEGI", "inegi_2020.csv")

# ======================================================
# SUBCENTROS URBANOS (POLICENTRISMO)
# ======================================================
SUBCENTROS = {
    "Centro":      (19.4326, -99.1332),
    "Polanco":     (19.4330, -99.1960),
    "SantaFe":     (19.3619, -99.2736),
    "Insurgentes": (19.4045, -99.1700),
    "DelValle":    (19.3738, -99.1645),
    "Reforma":     (19.4273, -99.1677),
}

# ======================================================
# AMENIDADES DE INTERÉS
# Agrupadas por categoría para construir índices compuestos
# ======================================================

# Amenidades que se codifican como feature binaria individual
# (alta frecuencia y relevancia hedónica)
AMENITIES_BINARIAS = [
    "Wifi",
    "Kitchen",
    "Air conditioning",
    "Pool",
    "Hot tub",
    "Gym",
    "Elevator",
    "Washer",
    "Dryer",
    "Free parking on premises",
    "Free street parking",
    "EV charger",
    "Self check-in",
    "Smoke alarm",
    "Carbon monoxide alarm",
    "Fire extinguisher",
    "First aid kit",
    "Dedicated workspace",
    "Long term stays allowed",
    "Instant Bookable",        # redundante con campo, pero puede venir en amenities
    "Breakfast",
    "Indoor fireplace",
    "BBQ grill",
    "Patio or balcony",
    "Garden or backyard",
    "Beachfront",
    "Waterfront",
    "TV",
    "Heating",
    "Private entrance",
    "Luggage dropoff allowed",
    "Bed linens",
    "Extra pillows and blankets",
    "Room-darkening shades",
    "Baby monitor",            # proxy familias con bebés
    "Crib",
    "High chair",
    "Pets allowed",
    "Lockbox",
    "Keypad",
    "Smart lock",
    "Security cameras on property",
    "Exterior security cameras on property",
    "Safe",
]

# Grupos para índices compuestos (scores de categoría)
AMENITY_GROUPS = {
    "score_cocina": [
        "Kitchen", "Refrigerator", "Microwave", "Oven", "Stove",
        "Dishes and silverware", "Cooking basics", "Coffee maker",
        "Blender", "Freezer", "Wine glasses", "Coffee",
    ],
    "score_seguridad": [
        "Smoke alarm", "Carbon monoxide alarm", "Fire extinguisher",
        "First aid kit", "Exterior security cameras on property",
        "Security cameras on property", "Safe",
    ],
    "score_confort": [
        "Air conditioning", "Heating", "Hot water", "Bed linens",
        "Extra pillows and blankets", "Room-darkening shades",
        "Portable fans", "Hair dryer", "Iron",
    ],
    "score_premium": [
        "Pool", "Hot tub", "Gym", "Sauna", "EV charger",
        "Indoor fireplace", "BBQ grill", "Patio or balcony",
        "Garden or backyard", "Beachfront", "Waterfront",
        "Resort access", "Pocket wifi",
    ],
    "score_familia": [
        "Crib", "High chair", "Baby monitor", "Baby bath",
        "Children's books and toys", "Children's dinnerware",
        "Pets allowed", "Pack 'n play/travel crib",
    ],
    "score_trabajo": [
        "Dedicated workspace", "Wifi", "Pocket wifi",
        "Long term stays allowed", "Printer",
    ],
    "score_accesibilidad": [
        "Elevator", "Wheelchair accessible", "Wide entrance",
        "Wide hallways", "Roll-in shower",
    ],
}

# ======================================================
# HELPERS
# ======================================================

def clean_text(txt):
    if pd.isna(txt):
        return ""
    t = str(txt).lower().strip()
    t = t.translate(str.maketrans("áéíóú", "aeiou"))
    return t


def parse_price(raw):
    """Convierte '$1,234.00' → 1234.0"""
    if pd.isna(raw):
        return np.nan
    cleaned = str(raw).replace("$", "").replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def parse_amenities(raw) -> list:
    """Parsea el campo amenities (JSON-like string) → lista de strings."""
    if pd.isna(raw) or str(raw).strip() in ("", "[]"):
        return []
    try:
        return ast.literal_eval(raw)
    except Exception:
        # Fallback: limpiar comillas y separar por coma
        raw = str(raw).strip("[]").replace('"', "").replace("'", "")
        return [x.strip() for x in raw.split(",") if x.strip()]


def tf_bool(val, true_val="t"):
    """Convierte 't'/'f' o True/False → 0/1"""
    if pd.isna(val):
        return np.nan
    return 1 if str(val).strip().lower() in ("t", "true", "1", "yes") else 0


def nearest_distance(pts, coords):
    if len(pts) == 0:
        return np.ones(len(coords)) * 5000
    tree = cKDTree(pts)
    d, _ = tree.query(coords)
    return d


def density_proxy(target_gdf, pts, r=1000):
    if len(pts) == 0:
        return np.zeros(len(target_gdf))
    tree = cKDTree(pts)
    coords = np.array([(p.x, p.y) for p in target_gdf.geometry])
    counts = tree.query_ball_point(coords, r)
    return np.array([len(c) for c in counts])


def line_to_points(gdf, step):
    pts = []
    for geom in gdf.geometry:
        if geom.geom_type == "LineString":
            for d in np.arange(0, geom.length, step):
                p = geom.interpolate(d)
                pts.append((p.x, p.y))
    return np.array(pts)


def safe_load_geodata(path, label, epsg=32614):
    if not os.path.exists(path):
        print(f"  ⚠ Saltando {label}: no encontrado")
        return None
    try:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf = gdf.to_crs(epsg=epsg)
        gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
        print(f"  ✓ {label} ({len(gdf)} registros)")
        return gdf
    except Exception as e:
        print(f"  ⚠ Error {label}: {e}")
        return None


def validar_y_relocalizar(df, path_alcaldias):
    print("=" * 80)
    print("VALIDACIÓN ESPACIAL CDMX")
    print("=" * 80)
    alcaldias = gpd.read_file(path_alcaldias).to_crs("EPSG:4326")
    alcaldias["alc_clean"] = alcaldias["NOMGEO"].apply(clean_text)
    df["alc_clean"] = df["alcaldia"].apply(clean_text)

    LAT_MIN, LAT_MAX = 19.04, 19.60
    LON_MIN, LON_MAX = -99.37, -98.94

    fuera = (
        (df["latitud"] < LAT_MIN) | (df["latitud"] > LAT_MAX) |
        (df["longitud"] < LON_MIN) | (df["longitud"] > LON_MAX)
    )
    n = fuera.sum()
    print(f"  Registros fuera de CDMX: {n}")

    if n > 0:
        centroides = dict(zip(
            alcaldias["alc_clean"],
            alcaldias.geometry.centroid
        ))
        for idx in df[fuera].index:
            alc = df.loc[idx, "alc_clean"]
            if alc in centroides:
                c = centroides[alc]
                df.loc[idx, "longitud"] = c.x
                df.loc[idx, "latitud"]  = c.y
        print(f"  ✓ {n} registros relocalizados")
    else:
        print("  ✓ Sin outliers espaciales")

    df.drop(columns=["alc_clean"], inplace=True, errors="ignore")
    return df


def cargar_catastro_completo(path_catastro):
    print("=" * 80)
    print("CARGANDO CATASTRO (opcional)")
    print("=" * 80)
    shp_files = glob.glob(os.path.join(path_catastro, "*.shp"))
    frames = []
    for shp in shp_files:
        try:
            nb = os.path.splitext(os.path.basename(shp))[0]
            csv_path = os.path.join(path_catastro, nb + ".csv")
            geo = gpd.read_file(shp, engine="fiona")
            if not os.path.exists(csv_path):
                continue
            attrs = pd.read_csv(csv_path, low_memory=False)
            if "fid" not in geo.columns or "fid" not in attrs.columns:
                continue
            for d in [geo, attrs]:
                d["fid"] = d["fid"].astype(str).str.strip().str.replace(".0", "", regex=False)
            temp = geo.merge(attrs, on="fid", how="left")
            if temp.crs is None:
                temp = temp.set_crs("EPSG:4326")
            temp = temp.to_crs(epsg=32614)
            frames.append(temp)
            print(f"  ✓ {nb} ({len(temp)} polígonos)")
        except Exception as e:
            print(f"  ⚠ {e}")
    if not frames:
        return gpd.GeoDataFrame()
    catastro = pd.concat(frames, ignore_index=True)
    for c in ["sup_terreno", "sup_construccion", "anio_construccion",
              "valor_unitario_suelo", "valor_suelo"]:
        if c in catastro.columns:
            catastro[c] = pd.to_numeric(catastro[c], errors="coerce")
    print(f"  ✓ Catastro: {len(catastro)} polígonos")
    return catastro


# ======================================================
# 1. CARGA Y LIMPIEZA BÁSICA
# ======================================================
print("=" * 80)
print("1/7  CARGA Y LIMPIEZA — AIRBNB CDMX")
print("=" * 80)

df_raw = pd.read_csv(PATH_LISTINGS)
print(f"  Registros cargados: {len(df_raw)}")

df = df_raw.copy()

# ── Precio ──
df["precio_noche"] = df["price"].apply(parse_price)

# Rango válido para CDMX: $100 – $50,000 MXN/noche
# (percentil 99 ~ $10,000; precio máximo real ~$50,000 para propiedades de lujo extremo)
n_antes = len(df)
df = df[(df["precio_noche"] >= 100) & (df["precio_noche"] <= 50_000)].copy()
print(f"  Registros tras filtro de precio: {len(df)}  (eliminados: {n_antes - len(df)})")

# ── Alias geográficos ──
df["latitud"]  = df["latitude"]
df["longitud"] = df["longitude"]
df["alcaldia"] = df["neighbourhood_cleansed"]

# ── Variables booleanas Airbnb ──
df["es_superhost"]       = df["host_is_superhost"].apply(tf_bool)
df["es_instant_book"]    = df["instant_bookable"].apply(tf_bool)
df["tiene_perfil_pic"]   = df["host_has_profile_pic"].apply(tf_bool)
df["identidad_verificada"] = df["host_identity_verified"].apply(tf_bool)

# ── Room type → OHE con referencia "Shared room" ──────────────────────────
# Justificación: room_type es una variable nominal (no ordinal).
# La codificación ordinal (0-3) impone distancias equidistantes entre
# categorías que no tienen sustento hedónico:
#   un Hotel room puede superar en precio a un Entire home/apt barato.
# OHE con categoría de referencia (Shared room, la de menor precio mediano)
# permite que el modelo estime un efecto independiente por tipo,
# consistente con la teoría hedónica de Rosen (1974).
#
# Se mantiene room_type_ord SOLO como feature ordinal de respaldo
# por si OHE genera multicolinealidad en muestras pequeñas (lo filtra
# la depuración automática de >0.90 de correlación).

ROOM_TYPE_MAP = {
    "Entire home/apt": 3,
    "Hotel room":      2,
    "Private room":    1,
    "Shared room":     0,
}
df["room_type_ord"] = df["room_type"].map(ROOM_TYPE_MAP).fillna(1)

# OHE — referencia: Shared room (omitida para evitar multicolinealidad perfecta)
df["rt_entire"]  = (df["room_type"] == "Entire home/apt").astype(int)
df["rt_private"] = (df["room_type"] == "Private room").astype(int)
df["rt_hotel"]   = (df["room_type"] == "Hotel room").astype(int)
# Shared room = rt_entire=0, rt_private=0, rt_hotel=0 → categoría base

# ── Interacciones room_type × variables de calidad ──────────────────────────
# Hipótesis: el efecto de las amenidades premium, el rating de limpieza y la
# localización sobre el precio NO es el mismo para un cuarto privado que para
# un inmueble completo. Un pool o un gym solo eleva significativamente el
# precio en "Entire home/apt"; en "Private room" el huésped no lo usa igual.
# Estas interacciones capturan esa elasticidad diferenciada.
# (Se calcularán después de tener review_score_compuesto y score_premium)


# ── Property type → binarias de alto nivel ──
df["es_entire"]  = df["property_type"].str.lower().str.contains("entire").astype(int)
df["es_privado"] = df["property_type"].str.lower().str.contains("private").astype(int)
df["es_hotel"]   = df["property_type"].str.lower().str.contains("hotel").astype(int)
df["es_serviced"]= df["property_type"].str.lower().str.contains("serviced").astype(int)

# ── Fechas → antigüedad del host (años) ──
df["host_since_dt"]    = pd.to_datetime(df["host_since"], errors="coerce")
df["host_antiguedad"]  = (pd.Timestamp("2025-06-01") - df["host_since_dt"]).dt.days / 365.25
df["host_antiguedad"]  = df["host_antiguedad"].clip(0, 20)

# ── Tasa de respuesta / aceptación → float ──
for col in ["host_response_rate", "host_acceptance_rate"]:
    df[col] = (df[col].astype(str)
               .str.replace("%", "", regex=False)
               .apply(pd.to_numeric, errors="coerce")) / 100.0

# ── Ocupación estimada (proxy demanda) ──
# estimated_occupancy_l365d = noches ocupadas estimadas en los últimos 365 días
# Solo tiene valor para ~55% de registros; se imputa después
df["ocupacion_estimada"]  = pd.to_numeric(df["estimated_occupancy_l365d"], errors="coerce")
df["revenue_estimado"]    = pd.to_numeric(df["estimated_revenue_l365d"],   errors="coerce")

# Tasa de ocupación normalizada [0, 1]
# (máx teórico = 365 noches)
df["tasa_ocupacion"] = (df["ocupacion_estimada"] / 365).clip(0, 1)

# ── Disponibilidad ──
df["disponibilidad_365"] = pd.to_numeric(df["availability_365"], errors="coerce")
df["pct_disponible"]     = (df["disponibilidad_365"] / 365).clip(0, 1)

# ── Reviews ──
review_cols = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]
for c in review_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Score compuesto de calidad (promedio ponderado)
# Ponderaciones revisadas a partir de López-Tamayo & Ramírez-Álvarez (2021) CDMX:
# - cleanliness y location son los más significativos en precio hedónico CDMX
# - review_scores_value se EXCLUYE del compuesto: el paper muestra que
#   "value" captura percepción precio/calidad y tiene efecto negativo en precio
#   (listings caros son calificados como "bajo valor") — incluirlo distorsiona
#   el score de calidad hacia abajo en el segmento lujo.
# - Se redistribuye su peso entre rating y cleanliness.
df["review_score_compuesto"] = (
    df["review_scores_rating"]        * 0.30 +   # ↑ de 0.25
    df["review_scores_cleanliness"]   * 0.25 +   # ↑ de 0.20
    df["review_scores_location"]      * 0.20 +
    df["review_scores_accuracy"]      * 0.10 +
    df["review_scores_checkin"]       * 0.10 +
    df["review_scores_communication"] * 0.05    # ↓ de 0.10; review_value excluido
    # review_scores_value EXCLUIDO — efecto negativo en precio hedónico (paper CDMX)
)

# Listing "veterano" (>50 reviews → señal de demanda probada)
df["es_listing_veterano"] = (df["number_of_reviews"] >= 50).astype(int)

# ── Host profesional (>2 listings = ≥3) ──
# López-Tamayo & Ramírez-Álvarez (2021) CDMX: anfitrión profesional cobra 16.2%
# más por un listing equivalente. Umbral >2 listados, igual que Arvanitidis (2020)
# y Cai (2019); es el criterio más estricto de la literatura.
df["host_profesional"] = (df["host_total_listings_count"] >= 3).astype(int)

# ── Mínimo de noches ──
df["min_nights_log"] = np.log1p(df["minimum_nights"].clip(1, 365))

# ── Índice de flexibilidad de reservación ──────────────────────────────────
# El mayor coeficiente extrínseco en el paper CDMX: +11.05% en precio.
# Hipótesis: en el mercado mexicano el turista valora no tener fricciones
# administrativas para reservar — opuesto a Hong Kong y Barcelona donde el
# huésped valora anfitriones estrictos (Cai 2019, Lladós-Masllorens 2020).
# Índice aditivo de 4 componentes (0 = muy estricto, 4 = muy flexible):
#   1. Sin depósito de garantía (sin security_deposit o =0)
#   2. Sin mínimo de noches (minimum_nights <= 1)
#   3. Reserva instantánea activada (instant_bookable = 't')
#   4. Sin requerimiento de documentos (host_identity_verified ≠ obligatorio)
#      — proxy: cancellation_policy flexible o moderate
df["sin_deposito"] = (
    pd.to_numeric(df.get("security_deposit", pd.Series("0", index=df.index))
                  .astype(str).str.replace("[$,]", "", regex=True), errors="coerce")
    .fillna(0) == 0
).astype(int)

df["sin_minimo_noches"] = (df["minimum_nights"].clip(1, 365) <= 1).astype(int)

df["cancelacion_flexible"] = df.get(
    "cancellation_policy", pd.Series("moderate", index=df.index)
).astype(str).str.lower().str.contains("flexible|moderate").astype(int)

# Componente ya construido: es_instant_book
df["flexibilidad_reserva"] = (
    df["sin_deposito"] +
    df["sin_minimo_noches"] +
    df["es_instant_book"] +
    df["cancelacion_flexible"]
) / 4.0   # normalizado [0, 1]

print(f"\n  Variables Airbnb construidas.")
print(f"  Distribución de room_type:")
print(df["room_type"].value_counts().to_string())

# ======================================================
# 2. EXPANSIÓN DE AMENITIES
# ======================================================
print("\n" + "=" * 80)
print("2/7  EXPANSIÓN DE AMENIDADES")
print("=" * 80)

# Parsear amenities
df["amenities_list"] = df["amenities"].apply(parse_amenities)

# Total de amenidades (proxy de "equipamiento general")
df["n_amenidades"] = df["amenities_list"].apply(len)
df["n_amenidades_log"] = np.log1p(df["n_amenidades"])

# ── Features binarias ──
print("  Construyendo features binarias de amenidades...")
for amenity in AMENITIES_BINARIAS:
    col_name = "am_" + amenity.lower().replace(" ", "_").replace("/", "_")
    df[col_name] = df["amenities_list"].apply(
        lambda lst: 1 if amenity in lst else 0
    )

# ── Scores compuestos por categoría ──
print("  Construyendo scores de categoría de amenidades...")
for score_name, amenity_list in AMENITY_GROUPS.items():
    df[score_name] = df["amenities_list"].apply(
        lambda lst: sum(1 for a in amenity_list if a in lst)
    )
    # Normalizar por el máximo teórico del grupo
    max_val = len(amenity_list)
    df[score_name] = df[score_name] / max_val

# ── Índice de amenidades premium ──
# Ratio de amenidades de lujo respecto al total
df["ratio_amenidades_premium"] = np.where(
    df["n_amenidades"] > 0,
    df["score_premium"] * len(AMENITY_GROUPS["score_premium"]) / df["n_amenidades"],
    0
)

print(f"  ✓ {len(AMENITIES_BINARIAS)} features binarias + {len(AMENITY_GROUPS)} scores de grupo")
print(f"  Mediana de amenidades por listing: {df['n_amenidades'].median():.0f}")

# ======================================================
# 3. IMPUTACIÓN MULTIVARIADA
# ======================================================
print("\n" + "=" * 80)
print("3/7  IMPUTACIÓN MULTIVARIADA (MICE)")
print("=" * 80)

# Columnas a imputar
COLS_IMPUTE = [
    "precio_noche",
    "bedrooms",
    "bathrooms",
    "beds",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "review_scores_value",
    "review_score_compuesto",
    "tasa_ocupacion",
    "host_response_rate",
    "host_acceptance_rate",
    "es_superhost",
    "host_antiguedad",
]

# Columnas auxiliares como features para guiar la imputación
COLS_AUX_IMP = [
    "accommodates",
    "room_type_ord",
    "es_entire",
    "n_amenidades",
    "score_confort",
    "score_premium",
    "es_listing_veterano",
    "number_of_reviews",
]

cols_to_fit = COLS_IMPUTE + COLS_AUX_IMP

print(f"  Nulos antes de imputación:")
for c in COLS_IMPUTE:
    pct = df[c].isna().mean() * 100
    print(f"    {c:<40} {pct:5.1f}%")

imputer = IterativeImputer(
    max_iter=10,
    random_state=42,
    min_value=0,
    imputation_order="ascending",
)

# Fit solo con registros que tienen precio (el target)
mask_precio = df["precio_noche"].notna()
imputer.fit(df.loc[mask_precio, cols_to_fit])
df[cols_to_fit] = imputer.transform(df[cols_to_fit])

# Post-procesamiento tras imputación
df["bedrooms"]  = df["bedrooms"].round().clip(0, 20).astype(int)
df["bathrooms"] = df["bathrooms"].round(1).clip(0, 20)
df["beds"]      = df["beds"].round().clip(0, 30).astype(int)
df["precio_noche"] = df["precio_noche"].round(0).clip(100, 50_000)

# Reconstruir review compuesto tras imputación (misma ponderación que arriba)
df["review_score_compuesto"] = (
    df["review_scores_rating"]        * 0.30 +
    df["review_scores_cleanliness"]   * 0.25 +
    df["review_scores_location"]      * 0.20 +
    df["review_scores_accuracy"].fillna(df["review_scores_rating"]) * 0.10 +
    df["review_scores_checkin"].fillna(df["review_scores_rating"])  * 0.10 +
    df["review_scores_communication"].fillna(df["review_scores_rating"]) * 0.05
    # review_scores_value excluido — ver justificación arriba
)

print(f"\n  ✓ Imputación completada.")

# ── Aliases de compatibilidad con el esquema del modelo espacial ──
df["price"]          = df["precio_noche"]
# En el modelo espacial: price_m2_raw = price / area  →  area = accommodates (capacidad funcional)
# No se multiplica por 12: el "área" en Airbnb son los huéspedes, no los m²
df["rooms"]          = df["bedrooms"]
df["parking_spaces"] = df.get("am_free_parking_on_premises", pd.Series(0, index=df.index))
df["antiguedad"]     = (2025 - df["host_since_dt"].dt.year).clip(0, 15).fillna(5)
df["alc"]            = df["alcaldia"].apply(clean_text)

# Remover listings sin coordenadas (no debería haber ninguno)
df = df.dropna(subset=["latitud", "longitud", "precio_noche"]).reset_index(drop=True)
print(f"  Registros finales: {len(df)}")

# ── Precio por huésped: análogo a precio/m² en el modelo espacial ──
# El modelo espacial entrena sobre price_m2_raw (precio por unidad de área)
# y reconstruye el precio total como: pred_total = pred_m2 * area
# En Airbnb: "área funcional" = accommodates → se normaliza por huésped
# Así los coeficientes son comparables entre listings de distinto tamaño

# ======================================================
# 4. CAPAS GEOESPACIALES
# ======================================================
print("\n" + "=" * 80)
print("4/7  GEOFEATURES")
print("=" * 80)

# Validación espacial
df = validar_y_relocalizar(df, path_alcaldias)

gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.longitud, df.latitud), crs="EPSG:4326"
)
gdf_utm = gdf.to_crs(epsg=32614)
coords  = np.array([(p.x, p.y) for p in gdf_utm.geometry])

# ── Catastro ──
# Justificación metodológica del uso catastral en Airbnb:
#   Inside Airbnb NO reporta superficie en m² del inmueble.
#   El catastro CDMX 2021 contiene sup_construccion (m² construidos) de
#   cada predio. El promedio de sup_construccion en radio 200m (≈ manzana)
#   es el mejor proxy disponible de la escala típica de los inmuebles
#   de la zona, y se usa para construir:
#     (a) cat_area_imputable: área estimada del listing según zona
#     (b) cat_densidad_edificatoria: sup_construccion / sup_terreno local
#         → captura si la colonia es de alta densidad (departamentos) o
#           baja densidad (casas), lo que afecta el perfil del huésped
#     (c) cat_mean_vus_200m: valor unitario de suelo vecinal
#         → proxy de plusvalía del suelo que no captura ninguna
#           otra variable del dataset
#
# Teoría hedónica: el precio de un Airbnb no solo refleja las
# características del inmueble sino la "calidad del lugar" (Rosen, 1974).
# El valor catastral del suelo vecinal captura justamente eso.

catastro = cargar_catastro_completo(path_catastro)

cat_cols = [
    "cat_mean_valor_suelo_200m",
    "cat_mean_vus_200m",
    "cat_mean_antiguedad_200m",
    "cat_std_valor_suelo_200m",
    "cat_mean_ratio_construccion_200m",
    "cat_density_predios_200m",
    # NUEVAS — área y densidad edificatoria
    "cat_mean_sup_construccion_200m",   # m² construidos promedio vecinal
    "cat_densidad_edificatoria_200m",   # ratio_construccion medio (intensidad de uso)
    "cat_area_imputable",               # área estimada del listing (ver abajo)
]

if not catastro.empty:
    catastro_pts = catastro.copy()
    catastro_pts["geometry"] = catastro_pts.geometry.centroid
    catastro_pts = catastro_pts[catastro_pts["valor_suelo"] > 0]
    catastro_pts = catastro_pts[catastro_pts["valor_unitario_suelo"] > 0]

    for c in ["sup_terreno", "sup_construccion", "anio_construccion",
              "valor_unitario_suelo", "valor_suelo"]:
        catastro_pts[c] = pd.to_numeric(catastro_pts[c], errors="coerce")

    # Ratio de construcción: qué tanto del terreno está edificado
    catastro_pts["ratio_construccion"] = np.where(
        catastro_pts["sup_terreno"] > 0,
        catastro_pts["sup_construccion"] / catastro_pts["sup_terreno"], np.nan
    )
    catastro_pts["antiguedad_cat"] = 2025 - catastro_pts["anio_construccion"]
    catastro_pts = catastro_pts[
        (catastro_pts["antiguedad_cat"] >= 0) & (catastro_pts["antiguedad_cat"] <= 150)
    ]

    # sup_construccion: filtrar valores extremos (< 10 m² o > 50,000 m²)
    catastro_pts = catastro_pts[
        catastro_pts["sup_construccion"].between(10, 50_000)
    ]

    catastro_pts = catastro_pts[[
        "geometry", "valor_suelo", "valor_unitario_suelo",
        "ratio_construccion", "antiguedad_cat", "sup_construccion", "sup_terreno"
    ]].copy()

    sindex = catastro_pts.sindex
    radio  = 200  # metros ≈ radio de manzana

    (mean_vs, mean_vus, mean_ant, std_vs, mean_ratio, dens,
     mean_sup_const, mean_dens_edif) = [], [], [], [], [], [], [], []

    for geom in gdf_utm.geometry:
        try:
            posibles = list(sindex.intersection(geom.buffer(radio).bounds))
            vecinos  = catastro_pts.iloc[posibles]
            vecinos  = vecinos[vecinos.geometry.distance(geom) <= radio]
            if len(vecinos) == 0:
                mean_vs.append(np.nan);  mean_vus.append(np.nan)
                mean_ant.append(np.nan); std_vs.append(np.nan)
                mean_ratio.append(np.nan); dens.append(0)
                mean_sup_const.append(np.nan); mean_dens_edif.append(np.nan)
            else:
                mean_vs.append(vecinos["valor_suelo"].mean())
                mean_vus.append(vecinos["valor_unitario_suelo"].mean())
                mean_ant.append(vecinos["antiguedad_cat"].mean())
                std_vs.append(vecinos["valor_suelo"].std())
                mean_ratio.append(vecinos["ratio_construccion"].mean())
                dens.append(len(vecinos))
                mean_sup_const.append(vecinos["sup_construccion"].mean())
                # Densidad edificatoria: sup_construccion total / sup_terreno total
                sup_t = vecinos["sup_terreno"].sum()
                sup_c = vecinos["sup_construccion"].sum()
                mean_dens_edif.append(sup_c / sup_t if sup_t > 0 else np.nan)
        except Exception:
            mean_vs.append(np.nan);  mean_vus.append(np.nan)
            mean_ant.append(np.nan); std_vs.append(np.nan)
            mean_ratio.append(np.nan); dens.append(np.nan)
            mean_sup_const.append(np.nan); mean_dens_edif.append(np.nan)


    df["cat_mean_valor_suelo_200m"] = mean_vs
    df["cat_mean_vus_200m"] = mean_vus
    df["cat_mean_antiguedad_200m"] = mean_ant
    df["cat_std_valor_suelo_200m"] = std_vs
    df["cat_mean_ratio_construccion_200m"] = mean_ratio
    df["cat_density_predios_200m"] = dens
    df["cat_mean_sup_construccion_200m"] = mean_sup_const
    df["cat_densidad_edificatoria_200m"] = mean_dens_edif

    # ── cat_area_imputable ──────────────────────────────────────────────────
    # Estimación del área del listing a partir de la superficie catastral vecinal.
    sup_base = pd.Series(mean_sup_const).fillna(pd.Series(mean_sup_const).median())
    divisor_room_type = df["room_type_ord"].map({
        3: 1.0, # Entire home/apt → predio completo
        2: 1.0, # Hotel room → aplica igual
        1: 3.5, # Private room → fracción del predio
        0: 5.0, # Shared room → fracción menor
    }).fillna(3.5)

    # Ajuste por número de habitaciones (1 habitación = divisor base)
    bedrooms_adj = df["bedrooms"].clip(1, 10).values
    df["cat_area_imputable"] = (
        sup_base.values / divisor_room_type.values * bedrooms_adj
    ).clip(8, 500) # límites físicos plausibles: 8 m² – 500 m²

    
    # price_per_guest = precio por huésped (accommodates)
    # NOTA: en la literatura hedónica de Airbnb se normaliza por capacidad de personas,
    # no por m², porque el área del listing raramente está disponible directamente.
    # Se mantiene el alias price_m2_raw por compatibilidad con el resto del pipeline.
    df["price_per_guest"] = df["price"] / df["accommodates"]
    df["price_per_guest"] = df["price_per_guest"].clip(50, 50000)
    df["price_m2_raw"]    = df["price_per_guest"]   # alias para compatibilidad

    if "area" not in df.columns:
        df["area"] = df["cat_area_imputable"].astype(float)
    print("  ✓ Features catastrales (incluye área imputable y densidad edificatoria asignadas a 'area')")

else:
    print("  ⚠ Catastro no disponible — usando medianas globales")
    # Bloque de seguridad en caso de que falle la lectura del catastro completo
    for c in cat_cols:
        if c not in df.columns:
            df[c] = 0.0
    df["cat_area_imputable"] = df["cat_area_imputable"].fillna(60.0) # 60m² promedio por defecto
    df["area"] = df["cat_area_imputable"].astype(float)
    df["price_m2_raw"] = df["price"] / df["area"]
    
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].median() if df[cat_cols].notna().any() else 0.0)

# ── Metro ──
metro = gpd.read_file(path_metro).to_crs(epsg=32614)
metro_pts = np.array([(g.x, g.y) for g in metro.geometry.explode(index_parts=False)
                       if g.geom_type == "Point"])
df["dist_metro_m"]   = nearest_distance(metro_pts, coords)
df["density_metro"]  = density_proxy(gdf_utm, metro_pts, 500)
print(f"  ✓ Metro ({len(metro_pts)} estaciones)")

# ── Tren Ligero ──
tren = gpd.read_file(path_tren).to_crs(epsg=32614)
tren_pts = np.array([(g.x, g.y) for g in tren.geometry.explode(index_parts=False)
                      if g.geom_type == "Point"])
df["dist_tren_m"]    = nearest_distance(tren_pts, coords)
df["density_tren"]   = density_proxy(gdf_utm, tren_pts, 500)
print(f"  ✓ Tren Ligero ({len(tren_pts)} estaciones)")

# ── Trolebús ──
trole = gpd.read_file(path_trole).to_crs(epsg=32614)
trole_pts = np.array([(g.x, g.y) for g in trole.geometry.explode(index_parts=False)
                       if g.geom_type == "Point"])
df["dist_trole_m"]   = nearest_distance(trole_pts, coords)
df["density_trole"]  = density_proxy(gdf_utm, trole_pts, 300)
print(f"  ✓ Trolebús ({len(trole_pts)} paradas)")

# ── Cablebús ──
cable = gpd.read_file(path_cable).to_crs(epsg=32614)
cable_pts = np.array([(g.x, g.y) for g in cable.geometry.explode(index_parts=False)
                       if g.geom_type == "Point"])
df["dist_cable_m"]   = nearest_distance(cable_pts, coords)
df["density_cable"]  = density_proxy(gdf_utm, cable_pts, 400)
print(f"  ✓ Cablebús ({len(cable_pts)} estaciones)")

# ── Metrobús ──
if os.path.exists(path_metrobus):
    mb_gdf = gpd.read_file(path_metrobus)
    if mb_gdf.crs is None:
        mb_gdf = mb_gdf.set_crs("EPSG:4326")
    mb_gdf = mb_gdf.to_crs(epsg=32614)
    mb_gdf = mb_gdf[mb_gdf.geometry.notnull() & ~mb_gdf.geometry.is_empty]
    mb_pts = np.array([(g.x, g.y) for g in mb_gdf.geometry.explode(index_parts=False)
                        if g.geom_type == "Point"])
    if len(mb_pts) > 0:
        df["dist_metrobus_m"]  = nearest_distance(mb_pts, coords)
        df["density_metrobus"] = density_proxy(gdf_utm, mb_pts, 600)
        print(f"  ✓ Metrobús ({len(mb_pts)} estaciones)")
    else:
        df["dist_metrobus_m"] = 5000; df["density_metrobus"] = 0
else:
    df["dist_metrobus_m"] = 5000; df["density_metrobus"] = 0

# ── Comercio ──
com = gpd.read_file(path_comercio).to_crs(epsg=32614)
com_pts = np.array([(g.centroid.x, g.centroid.y) for g in com.geometry])
df["comercio_density"] = density_proxy(gdf_utm, com_pts, 1500)

# ── Seguridad ──
seg = gpd.read_file(path_seguridad).to_crs("EPSG:4326")
tmp = gpd.sjoin(gdf, seg[["C_US", "geometry"]], how="left", predicate="covered_by")
df["marginalidad_score"] = pd.to_numeric(tmp["C_US"], errors="coerce").fillna(3)

# ── Turistas / migrantes ──
if os.path.exists(path_turistas):
    tur = pd.read_csv(path_turistas)
    tur = tur[~tur["Municipio"].astype(str).str.contains("000")]
    tur["Municipio_limpio"] = tur["Municipio"].astype(str).str.replace(
        r"^\d{3}\s+", "", regex=True
    )
    tur["alc"] = tur["Municipio_limpio"].apply(clean_text)
    col_t = "Población_migrantes_porcentaje"
    tur[col_t] = (tur[col_t].astype(str).str.replace("%", "", regex=False))
    tur[col_t] = pd.to_numeric(tur[col_t], errors="coerce") / 100.0
    tur_g = tur.groupby("alc")[col_t].mean().reset_index()
    tur_g.columns = ["alc", "pct_migrantes_turistas"]
    df["alc"] = df["alcaldia"].apply(clean_text)
    df = df.merge(tur_g, on="alc", how="left")
    df["pct_migrantes_turistas"] = df["pct_migrantes_turistas"].fillna(0)
    print("  ✓ Intensidad turística incorporada")
else:
    df["pct_migrantes_turistas"] = 0

# ── Ciudad de 15 Minutos ──
print("=" * 80)
print("4/7  CIUDAD DE 15 MINUTOS")
print("=" * 80)

ciclo    = safe_load_geodata(path_ciclovias,           "Ciclovías")
verdes   = safe_load_geodata(path_areas_verdes,        "Áreas Verdes")
salud_b  = safe_load_geodata(path_salud,               "Salud Base")
salud_p  = safe_load_geodata(path_hospitales_publicos, "Hospitales Públicos")
esc_pub  = safe_load_geodata(path_escuelas_pub,        "Escuelas Públicas")
esc_priv = safe_load_geodata(path_escuelas_priv,       "Escuelas Privadas")

if ciclo is not None:
    ciclo_pts = line_to_points(ciclo, 100)
    df["dist_ciclovia_m"]       = nearest_distance(ciclo_pts, coords)
    df["densidad_ciclovia_15m"] = density_proxy(gdf_utm, ciclo_pts, 1200)
else:
    df["dist_ciclovia_m"] = 5000; df["densidad_ciclovia_15m"] = 0

if verdes is not None:
    verdes_pts = np.array([(p.x, p.y) for p in verdes.geometry.centroid])
    df["dist_area_verde_m"]    = nearest_distance(verdes_pts, coords)
    df["densidad_parques_15m"] = density_proxy(gdf_utm, verdes_pts, 1200)
else:
    df["dist_area_verde_m"] = 5000; df["densidad_parques_15m"] = 0

salud_pts_list = []
for cap in [salud_b, salud_p]:
    if cap is not None:
        pts = np.array([(p.x, p.y) for p in cap.geometry])
        if len(pts) > 0:
            salud_pts_list.append(pts)

if salud_pts_list:
    salud_pts = np.vstack(salud_pts_list)
    df["dist_salud_m"]     = nearest_distance(salud_pts, coords)
    df["acceso_salud_15m"] = density_proxy(gdf_utm, salud_pts, 1200)
else:
    df["dist_salud_m"] = 5000; df["acceso_salud_15m"] = 0

esc_list = []
if esc_pub  is not None: esc_list.append(np.array([(p.x, p.y) for p in esc_pub.geometry]))
if esc_priv is not None: esc_list.append(np.array([(p.x, p.y) for p in esc_priv.geometry]))

if esc_list:
    esc_pts = np.vstack(esc_list)
    df["dist_escuela_m"]       = nearest_distance(esc_pts, coords)
    df["acceso_educacion_15m"] = density_proxy(gdf_utm, esc_pts, 1200)
else:
    df["dist_escuela_m"] = 5000; df["acceso_educacion_15m"] = 0

componentes_15min = ["density_metro", "densidad_ciclovia_15m",
                     "densidad_parques_15m", "acceso_salud_15m", "acceso_educacion_15m"]
df["score_15min"] = df[componentes_15min].sum(axis=1)
rng = df["score_15min"].max() - df["score_15min"].min()
df["score_15min"] = (df["score_15min"] - df["score_15min"].min()) / (rng if rng > 0 else 1)

df["prox_ciclovia"] = np.exp(-df["dist_ciclovia_m"]  / 500)
df["prox_parque"]   = np.exp(-df["dist_area_verde_m"] / 500)
df["prox_salud"]    = np.exp(-df["dist_salud_m"]      / 800)
df["prox_escuela"]  = np.exp(-df["dist_escuela_m"]    / 600)

df["uso_mixto"] = 0
if os.path.exists(path_uso_suelo):
    uso = safe_load_geodata(path_uso_suelo, "Uso de Suelo")
    if uso is not None:
        uso["es_mixto"] = uso["us_dscr"].str.contains("HM", na=False).astype(int)
        gdf_tmp = gpd.sjoin(gdf_utm, uso[["es_mixto", "geometry"]], how="left", predicate="within")
        df["uso_mixto"] = gdf_tmp.groupby(gdf_tmp.index)["es_mixto"].max().fillna(0)

# ======================================================
# 5. FEATURES DERIVADAS Y ESPACIALES
# ======================================================
print("\n" + "=" * 80)
print("5/7  FEATURES DERIVADAS")
print("=" * 80)

# ── Puntos de Interés Turísticos (PDI) — paper CDMX ──────────────────────
# López-Tamayo & Ramírez-Álvarez (2021): distancia media a 10 PDI más visitados
# según TripAdvisor CDMX. Coeficiente: -0.33% por cada 100m adicionales.
# DIFERENCIA CRÍTICA respecto a subcentros de negocios (Polanco, Santa Fe):
# Airbnb es turismo/estancia temporal, no actividad laboral corporativa.
# El turista en CDMX valora Chapultepec, el Zócalo, Coyoacán — no Santa Fe.
PDI_TURISTICOS = {
    "Museo_Antropologia":    (19.42612, -99.18628),
    "Castillo_Chapultepec":  (19.42067, -99.18165),
    "Basilica_Guadalupe":    (19.48513, -99.11741),
    "Bellas_Artes":          (19.43563, -99.14117),
    "Coyoacan_Centro":       (19.34839, -99.16317),
    "Museo_Frida_Kahlo":     (19.35546, -99.16205),
    "Zocalo":                (19.43282, -99.13292),
    "Museo_Soumaya":         (19.44097, -99.20467),
    "Templo_Mayor":          (19.43488, -99.13132),
    "Bosque_Chapultepec":    (19.41982, -99.18914),
}

pdi_gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(
        [lon for lat, lon in PDI_TURISTICOS.values()],
        [lat for lat, lon in PDI_TURISTICOS.values()],
    ),
    crs="EPSG:4326",
).to_crs(epsg=32614)
pdi_pts = np.array([(p.x, p.y) for p in pdi_gdf.geometry])

# Distancia MEDIA a todos los PDI (en 100m, igual que el paper)
tree_pdi = cKDTree(pdi_pts)
dists_pdi, _ = tree_pdi.query(coords, k=len(pdi_pts))   # distancia a cada PDI
df["dist_media_pdi_100m"] = dists_pdi.mean(axis=1) / 100   # en unidades de 100m
df["dist_pdi_log"]        = np.log1p(df["dist_media_pdi_100m"])

# Mantener también distancia al PDI más cercano (feature adicional)
df["dist_pdi_cercano_m"]  = dists_pdi[:, 0]

# Compatibilidad: alias para features que referencian subcentros
df["dist_nearest_subcenter_m"] = df["dist_pdi_cercano_m"]
df["dist_subcenter_log"]       = df["dist_pdi_log"]

# ── Señales de demanda basadas en reseñas ──────────────────────────────────
# Paper CDMX: el número de reseñas tiene efecto NEGATIVO en precio (-0.21%).
# Mecanismo (Gibbs et al. 2017): más reseñas → menor asimetría de información
# → menor capacidad de cobrar sobreprecio. También: listings baratos
# acumulan más reservas y por ende más reseñas.
# Se transforma en log para comprimir la cola derecha (hay listings con 500+).
df["n_reviews_log"] = np.log1p(df["number_of_reviews"])

# Competencia directa en colonia: listings con igual accommodates (paper CDMX)
# Coeficiente: +0.16% por competidor adicional (zonas turísticas congestionadas)
# → efecto positivo porque la competencia concentra la demanda en zonas atractivas
coords_geo = df[["longitud", "latitud"]].values
tree_listings = cKDTree(coords_geo)

# Radio ~800m aproxima colonia en CDMX (promedio ~0.5 km²)
nbrs_listings = tree_listings.query_ball_point(coords_geo, r=0.008)
df["listing_density_local"] = np.array([len(n) - 1 for n in nbrs_listings])
df["listing_density_log"]   = np.log1p(df["listing_density_local"])

# Competencia directa: misma capacidad de huéspedes (paper: "listados con
# mismo número de huéspedes permitidos en la colonia")
accomm_arr = df["accommodates"].values
def _comp_misma_cap(idx, nbrs, accomm):
    return sum(1 for i in nbrs if i != idx and accomm[i] == accomm[idx])

nbrs_800 = tree_listings.query_ball_point(coords_geo, r=0.008)
df["competencia_misma_cap"] = np.array(
    [_comp_misma_cap(idx, nbrs, accomm_arr)
     for idx, nbrs in enumerate(nbrs_800)]
)
df["competencia_log"] = np.log1p(df["competencia_misma_cap"])

# Concentración de listings completos en el radio (proxy de competencia directa)
es_entire_arr = df["es_entire"].values
nbrs_entire = tree_listings.query_ball_point(coords_geo, r=0.005)
df["density_entire_500m"] = np.array(
    [sum(es_entire_arr[i] for i in nbrs if i != idx)
     for idx, nbrs in enumerate(nbrs_entire)]
)

# ── Índice de gentrificación ──
print("  Calculando índice de gentrificación...")
try:
    c10 = pd.read_csv(path_2010)
    c20 = pd.read_csv(path_2020)
    vars_soc = ["pct_educ_sup", "pct_internet", "vph_pc", "vph_autom",
                "graproes", "pea", "prom_ocup", "pder_ss"]
    c10_g = c10.groupby("nom_mun")[vars_soc].mean().reset_index()
    c20_g = c20.groupby("nom_mun")[vars_soc].mean().reset_index()
    c10_g["alc"] = c10_g["nom_mun"].apply(clean_text)
    c20_g["alc"] = c20_g["nom_mun"].apply(clean_text)
    delta = c10_g.merge(c20_g, on="alc", suffixes=("_10", "_20"))
    for v in vars_soc:
        delta[f"d_{v}"] = delta[f"{v}_20"] - delta[f"{v}_10"]
    cols_delta = [f"d_{v}" for v in vars_soc]
    X_pca = StandardScaler().fit_transform(delta[cols_delta].fillna(0))
    gentrif = PCA(n_components=1).fit_transform(X_pca).flatten()
    delta["gentrification_index"] = (
        (gentrif - gentrif.min()) / (gentrif.max() - gentrif.min())
    )
    delta["alc"] = delta["alc"].apply(clean_text)
    df["alc"] = df["alcaldia"].apply(clean_text)
    df = df.merge(delta[["alc", "gentrification_index"]], on="alc", how="left")
    df["gentrification_index"] = df["gentrification_index"].fillna(
        df["gentrification_index"].median()
    )
    print("  ✓ Índice de gentrificación (PCA 8 variables)")
except Exception as e:
    print(f"  ⚠ Gentrificación: {e} — asignando 0.5")
    df["gentrification_index"] = 0.5

# ── Moran ──
print("\n  Calculando índice de Moran...")
try:
    gdf_moran = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitud, df.latitud), crs="EPSG:4326"
    ).to_crs(epsg=32614)
    gdf_moran = gdf_moran[np.isfinite(gdf_moran["price_m2_raw"])].reset_index(drop=True)
    y_moran = gdf_moran["price_m2_raw"].values.astype(float)
    w_knn = libpysal.weights.KNN.from_dataframe(gdf_moran, k=min(8, len(gdf_moran) - 1))
    w_knn.transform = "R"
    moran_global = Moran(y_moran, w_knn)
    i_val = float(np.asarray(moran_global.I).flat[0])
    p_val = float(np.asarray(moran_global.p_sim).flat[0])
    z_val = float(np.asarray(moran_global.z_norm).flat[0])
    print(f"  I de Moran: {i_val:.4f}  |  p: {p_val:.4f}  |  Z: {z_val:.4f}")
except Exception as e:
    print(f"  ⚠ Moran no calculado: {e}")

# ── Transformaciones log ──
log_cols = [
    "dist_metro_m", "dist_metrobus_m", "dist_tren_m", "dist_trole_m", "dist_cable_m",
    "dist_ciclovia_m", "dist_area_verde_m", "dist_salud_m", "dist_escuela_m",
    "comercio_density", "pct_migrantes_turistas", "antiguedad",
]
for c in log_cols:
    if c in df.columns:
        df[c] = np.log1p(df[c])

# ── Interacciones ──
df["marginalidad_score"]         = df["marginalidad_score"].fillna(3)
df["area_x_marginalidad"]        = np.log1p(df["accommodates"]) * df["marginalidad_score"]
df["area_X_gentrif"]             = df["accommodates"] * df["gentrification_index"]
df["gentrif_x_metro"]            = df["gentrification_index"] * df["density_metro"]
df["15min_X_gentrif"]            = df["score_15min"] * df["gentrification_index"]

# Interacciones específicas Airbnb
df["review_x_gentrif"]       = df["review_score_compuesto"] * df["gentrification_index"]
df["premium_x_15min"]        = df["score_premium"] * df["score_15min"]
df["superhost_x_review"]     = df["es_superhost"] * df["review_score_compuesto"]
# amenidades × proximidad PDI turístico: un listing con muchas amenidades
# cerca de atractivos turísticos puede cobrar mayor prima que uno igualmente
# equipado pero lejos — hipótesis: amenidades son complementarias a ubicación turística
df["amenidades_x_pdi"]       = df["n_amenidades_log"] / (df["dist_pdi_log"] + 1)
df["amenidades_x_subcenter"] = df["amenidades_x_pdi"]   # alias de compatibilidad

# Interacción profesional × flexibilidad: el paper muestra que la flexibilidad
# tiene mayor impacto para anfitriones profesionales (+19.4% vs +6.4%)
df["profesional_x_flex"]     = df["host_profesional"] * df["flexibilidad_reserva"]

# ── Interacciones room_type × calidad ──────────────────────────────────────
# rt_entire × score_premium: el efecto de las amenidades de lujo (pool, jacuzzi,
#   chimenea) solo se materializa plenamente en inmuebles completos donde el
#   huésped tiene uso exclusivo — hipótesis verificable con el signo del coeficiente.
df["entire_x_premium"]  = df["rt_entire"]  * df["score_premium"]

# rt_private × review_cleanliness: en cuartos privados (baño compartido o
#   espacios comunes) la limpieza percibida es más crítica para el precio
#   que en entire homes donde el huésped controla el espacio.
df["private_x_clean"]   = df["rt_private"] * df["review_scores_cleanliness"]

# rt_hotel × score_confort: los hotel rooms compiten directamente con la
#   hostelería tradicional, por lo que el confort (AC, calefacción, ropa de cama)
#   tiene mayor peso hedónico que en listings residenciales.
df["hotel_x_confort"]   = df["rt_hotel"]   * df["score_confort"]

# rt_entire × cat_area_imputable: el área construida solo es relevante como
#   proxy de tamaño en inmuebles completos; en cuartos privados o compartidos
#   el área total del predio no captura el espacio disponible al huésped.
if "cat_area_imputable" in df.columns:
    df["entire_x_area_cat"] = df["rt_entire"] * np.log1p(df["cat_area_imputable"])
else:
    df["entire_x_area_cat"] = 0.0


df["verde_marginalidad_ratio"] = np.log1p(
    df["densidad_parques_15m"] / (df["marginalidad_score"] + 1)
)
df["salud_marginalidad_ratio"] = np.log1p(
    df["acceso_salud_15m"] / (df["marginalidad_score"] + 1)
)
df["educacion_marginalidad_ratio"] = np.log1p(
    df["acceso_educacion_15m"] / (df["marginalidad_score"] + 1)
)

# ── Segmento lujo: basado en price_m2_raw = precio/huésped ──
# Consistente con el modelo espacial (is_luxury sobre price_m2_raw)
lux_cut = df["price_m2_raw"].quantile(0.90)
df["is_luxury"] = (df["price_m2_raw"] >= lux_cut).astype(int)

# ── Placeholders dinámicos ──
df["spatial_lag_price"]    = np.nan
df["precio_vecinal_local"] = np.nan
df["lag_x_accommodates"] = df["spatial_lag_price"] * np.log1p(df["area"])
# ======================================================
# FEATURES VECTOR
# ======================================================

# Features Airbnb-específicas
airbnb_features = [
    "accommodates", "bedrooms", "bathrooms", "beds",
    # room_type: OHE (referencia = Shared room) + ordinal de respaldo
    # La depuración automática de colinealidad eliminará room_type_ord
    # si correlaciona >0.90 con las dummies — lo cual es esperable.
    "room_type_ord",
    "rt_entire", "rt_private", "rt_hotel",          # OHE room_type
    "es_hotel", "es_serviced",                       # property_type adicionales

    # ── Factores de plataforma (paper CDMX) ──
    # host_profesional: +16.2% en precio (mayor efecto de plataforma en CDMX)
    "host_profesional",
    # es_superhost: no significativo en CDMX (Chen & Xie 2017, López-Tamayo 2021)
    # — se mantiene porque la depuración lo descartará si no aporta
    "es_superhost",
    "host_antiguedad",          # experiencia del anfitrión en meses → log
    "es_listing_veterano",
    "min_nights_log",

    # flexibilidad_reserva: mayor coeficiente extrínseco en CDMX (+11.05%)
    # El mercado mexicano valora no tener fricciones para reservar
    "flexibilidad_reserva",
    # es_instant_book: ya está dentro de flexibilidad_reserva, pero se mantiene
    # como feature independiente — la depuración eliminará si hay colinealidad
    "es_instant_book",

    # ── Scores de reviews ──
    # review_score_compuesto: calidad percibida (sin review_scores_value)
    "review_score_compuesto",
    "review_scores_rating",
    "review_scores_cleanliness",  # mayor peso hedónico en CDMX
    "review_scores_location",
    # review_scores_value EXCLUIDO del feature set: efecto negativo en precio
    # (listings caros son calificados como "bajo valor" — introduce sesgo)

    # n_reviews_log: efecto NEGATIVO confirmado en paper CDMX (-0.21%)
    # Mecanismo: más reseñas → menos asimetría de info → menos capacidad de sobreprecio
    "n_reviews_log",

    "tasa_ocupacion", "pct_disponible",
    "n_amenidades_log", "ratio_amenidades_premium",
    "score_cocina", "score_seguridad", "score_confort",
    "score_premium", "score_familia", "score_trabajo", "score_accesibilidad",

    # Binarias de amenidades más relevantes hedónicamente
    "am_wifi", "am_air_conditioning", "am_pool", "am_hot_tub", "am_gym",
    "am_elevator", "am_washer", "am_free_parking_on_premises",
    "am_self_check-in", "am_dedicated_workspace", "am_breakfast",
    "am_indoor_fireplace", "am_bbq_grill", "am_patio_or_balcony",
    "am_pets_allowed", "am_long_term_stays_allowed",

    # Interacciones Airbnb
    "review_x_gentrif", "premium_x_15min",
    "superhost_x_review",
    # amenidades_x_pdi — NO incluir amenidades_x_subcenter: es el mismo valor (alias),
    # incluir ambos produce correlación=1.0 y la depuración elimina las dos
    "amenidades_x_pdi",
    "density_entire_500m",
    # profesional × flexibilidad: paper muestra efecto diferencial por tipo de anfitrión
    "profesional_x_flex",
    # Interacciones room_type × calidad
    "entire_x_premium", "private_x_clean", "hotel_x_confort", "entire_x_area_cat",
]

# Features geoespaciales
geo_features = [
    # ── Localización turística (paper CDMX) ──
    # dist_media_pdi_100m: distancia media a 10 PDI turísticos TripAdvisor
    # Coeficiente esperado: -0.33% por 100m adicionales (López-Tamayo 2021)
    "dist_media_pdi_100m",
    # dist_pdi_log: versión log — NO incluir dist_subcenter_log (alias idéntico),
    # incluir ambos produce correlación=1.0 y la depuración elimina las dos
    "dist_pdi_log",
    "dist_pdi_cercano_m",      # distancia al PDI más cercano (variable distinta)

    # ── Competencia (paper CDMX) ──
    # competencia_misma_cap: listings con igual accommodates en ~800m radio
    # Coeficiente esperado: +0.16% (efecto positivo, aglomeración turística)
    # NO incluir competencia_log: correlaciona >0.95 con competencia_misma_cap
    "competencia_misma_cap",

    "marginalidad_score", "comercio_density",
    "dist_metro_m", "density_metro",
    "dist_metrobus_m", "density_metrobus",
    "dist_tren_m", "density_tren",
    "dist_trole_m", "density_trole",
    "dist_cable_m", "density_cable",
    "pct_migrantes_turistas", "gentrification_index",
    "area_x_marginalidad", "area_X_gentrif", "gentrif_x_metro",
    "dist_ciclovia_m", "densidad_ciclovia_15m",
    "dist_area_verde_m", "densidad_parques_15m",
    "dist_salud_m", "acceso_salud_15m",
    "dist_escuela_m", "acceso_educacion_15m",
    "score_15min", "prox_ciclovia", "prox_parque", "prox_salud", "prox_escuela",
    "uso_mixto", "15min_X_gentrif",
    "listing_density_log",
    "verde_marginalidad_ratio", "salud_marginalidad_ratio", "educacion_marginalidad_ratio",
    # Features catastrales originales
    "cat_mean_valor_suelo_200m", "cat_mean_vus_200m", "cat_mean_antiguedad_200m",
    "cat_std_valor_suelo_200m", "cat_mean_ratio_construccion_200m", "cat_density_predios_200m",
    # Features catastrales nuevas
    "cat_mean_sup_construccion_200m",
    "cat_densidad_edificatoria_200m",
    "cat_area_imputable",
]

# Solo incluir columnas que existan en el dataframe
static_features = [f for f in (airbnb_features + geo_features) if f in df.columns]
# Eliminar duplicados preservando orden (puede haber features repetidas entre listas)
seen = set()
static_features = [f for f in static_features if not (f in seen or seen.add(f))]
dynamic_features = ["spatial_lag_price", "precio_vecinal_local", "lag_x_accommodates"]

# ── Depuración por colinealidad — filtro orientado al target ──────────────
# Problema del filtro anterior: eliminaba por posición en la lista, no por
# relevancia. Si A y B correlacionan >umbral, siempre caía B (la que aparece
# después), aunque B correlacionara más con el target.
#
# Solución: para cada par correlacionado, conservar la variable con mayor
# correlación absoluta con log(price_m2_raw). El ElasticNet con L1 maneja
# colinealidad residual moderada — el umbral 0.92 solo filtra cuasi-duplicados.
print("\nDEPURACIÓN AUTOMÁTICA DE FEATURES")

y_target = np.log1p(df["price_m2_raw"].fillna(df["price_m2_raw"].median()))
target_corr = df[static_features].corrwith(pd.Series(y_target, index=df.index)).abs()

corr_matrix = df[static_features].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

drop_cols = set()
for col in upper.columns:
    high_corr_partners = upper.index[upper[col] > 0.92].tolist()
    for partner in high_corr_partners:
        if col in drop_cols or partner in drop_cols:
            continue
        # Conservar la que tiene mayor correlación con el target
        if target_corr.get(col, 0) >= target_corr.get(partner, 0):
            drop_cols.add(partner)
        else:
            drop_cols.add(col)

if drop_cols:
    print(f"  Variables eliminadas: {sorted(drop_cols)}")
static_features = [f for f in static_features if f not in drop_cols]
features = static_features + dynamic_features
print(f"  Features finales: {len(features)}")

# ======================================================
# FUNCIONES ESPACIALES (idénticas a los otros modelos)
# ======================================================

def compute_spatial_lag(train_coords, train_prices, target_coords, k=10):
    tree = cKDTree(train_coords)
    d, ix = tree.query(target_coords, k=min(k, len(train_coords)))
    if k == 1:
        d = d.reshape(-1, 1); ix = ix.reshape(-1, 1)
    weights = 1 / (d + 1e-5)
    lag_values = [np.average(train_prices[idxs], weights=w) for idxs, w in zip(ix, weights)]
    return np.log1p(np.array(lag_values))


def compute_local_neighbor_price(train_coords, train_prices, target_coords, k=15):
    nbrs = NearestNeighbors(n_neighbors=min(k, len(train_coords))).fit(train_coords)
    _, indices = nbrs.kneighbors(target_coords)
    return np.array([train_prices[idx].mean() for idx in indices])

# ======================================================
# 6. MODELO ELASTICNET + CLUSTERS
# ======================================================
def run_market_optimized(data, label):
    print("\n" + "=" * 80)
    print(f"PROCESANDO ELASTICNET + CLUSTERS: {label}")
    print("=" * 80)

    q_low, q_high = data["price_m2_raw"].quantile([0.05, 0.95])
    data = data[(data["price_m2_raw"] > q_low) & (data["price_m2_raw"] < q_high)].copy()

    if len(data) < 50:
        print(f"  ⚠ Datos insuficientes ({len(data)})")
        return [], [], data

    # Clustering espacial: estándar → solo coordenadas (como modelo espacial)
    # Lujo → enriquecido con variables de calidad Airbnb y gentrificación
    if label == "MERCADO ESTÁNDAR":
        enriched = StandardScaler().fit_transform(data[["latitud", "longitud"]])
    else:
        enriched = StandardScaler().fit_transform(
            data[["latitud", "longitud", "gentrification_index",
                  "score_15min", "score_premium", "review_score_compuesto"]]
        )

    n_clusters = min(3, len(data) // 100 + 1)
    data["cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(enriched)

    data["precio_predicho"] = np.nan
    global_real, global_pred = [], []

    for c in sorted(data.cluster.unique()):
        d_c = data[data.cluster == c].copy()
        if len(d_c) < 50:
            print(f"  ⚠ Cluster {c}: {len(d_c)} registros — omitido")
            continue

        X_base        = d_c[static_features].copy()
        y             = np.log1p(d_c["price_m2_raw"])        # log(precio/huésped)
        accommodates  = d_c["accommodates"].values            # para reconstruir precio_noche
        coords_c      = d_c[["longitud", "latitud"]].values
        n_splits      = min(5, len(d_c) // 30 + 1)
        kf            = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        c_real, c_pred = [], []

        for tr, te in kf.split(X_base):
            X_train = X_base.iloc[tr].copy()
            X_test  = X_base.iloc[te].copy()

            # Variables espaciales dinámicas — sin data leakage
            lag_tr = compute_spatial_lag(coords_c[tr], d_c.iloc[tr]["price_m2_raw"].values, coords_c[tr])
            lag_te = compute_spatial_lag(coords_c[tr], d_c.iloc[tr]["price_m2_raw"].values, coords_c[te])
            np_tr  = compute_local_neighbor_price(coords_c[tr], d_c.iloc[tr]["price_m2_raw"].values, coords_c[tr])
            np_te  = compute_local_neighbor_price(coords_c[tr], d_c.iloc[tr]["price_m2_raw"].values, coords_c[te])

            X_train["spatial_lag_price"]    = lag_tr;  X_test["spatial_lag_price"]    = lag_te
            X_train["precio_vecinal_local"]  = np_tr;   X_test["precio_vecinal_local"]  = np_te
            # lag_x_area: lag × log(accommodates) — análogo a lag × log(area) en ventas
            X_train["lag_x_accommodates"] = lag_tr * np.log1p(X_train["accommodates"])
            X_test["lag_x_accommodates"]  = lag_te * np.log1p(X_test["accommodates"])

            # Guardar variables dinámicas en el dataframe global
            data.loc[d_c.iloc[tr].index, "spatial_lag_price"]    = lag_tr
            data.loc[d_c.iloc[te].index, "spatial_lag_price"]    = lag_te
            data.loc[d_c.iloc[tr].index, "precio_vecinal_local"] = np_tr
            data.loc[d_c.iloc[te].index, "precio_vecinal_local"] = np_te
            data.loc[d_c.iloc[tr].index, "lag_x_accommodates"] = lag_tr * np.log1p(d_c.iloc[tr]["accommodates"].values)
            data.loc[d_c.iloc[te].index, "lag_x_accommodates"] = lag_te * np.log1p(d_c.iloc[te]["accommodates"].values)

            X_train = X_train.fillna(X_train.median()).fillna(0)
            X_test  = X_test.fillna(X_train.median()).fillna(0)

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("enet", ElasticNetCV(
                    alphas=np.logspace(-4, -1, 25),
                    l1_ratio=[0.4, 0.6, 0.8],
                    cv=5,
                )),
            ])
            model.fit(X_train, y.iloc[tr])

            # Predicción: pred_por_huesped × accommodates = precio_noche predicho
            pred_por_huesped = np.expm1(model.predict(X_test))
            pred_precio_noche = pred_por_huesped * accommodates[te]

            data.loc[d_c.iloc[te].index, "precio_predicho"] = pred_precio_noche
            c_pred.extend(pred_precio_noche)
            c_real.extend(d_c["price"].iloc[te])   # price = precio_noche real

        if c_real:
            print(f"  Cluster {c} | n={len(d_c)} | R²: {r2_score(c_real, c_pred):.4f} "
                  f"| MAPE: {mean_absolute_percentage_error(c_real, c_pred):.2%}")
            global_real.extend(c_real)
            global_pred.extend(c_pred)

    return global_real, global_pred, data


def run_knn_comparison(data, label):
    print("\n" + "=" * 80)
    print(f"PROCESANDO k-NN: {label}")
    print("=" * 80)

    d = data.copy()
    q_low, q_high = d["price_m2_raw"].quantile([0.05, 0.95])
    d = d[(d["price_m2_raw"] > q_low) & (d["price_m2_raw"] < q_high)].copy()

    if len(d) < 50:
        return 0, 1, [], []

    X            = d[static_features].copy()
    y            = np.log1p(d["price_m2_raw"])          # log(precio/huésped)
    accommodates = d["accommodates"].values
    coords       = d[["longitud", "latitud"]].values

    # Bloques espaciales para validación cruzada sin data leakage geográfico
    spatial_blocks = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(coords)
    real, pred = [], []

    for fold in np.unique(spatial_blocks):
        tr = np.where(spatial_blocks != fold)[0]
        te = np.where(spatial_blocks == fold)[0]
        if len(tr) < 10 or len(te) < 5:
            continue

        X_train = X.iloc[tr].copy(); X_test = X.iloc[te].copy()
        lag_tr = compute_spatial_lag(coords[tr], d.iloc[tr]["price_m2_raw"].values, coords[tr])
        lag_te = compute_spatial_lag(coords[tr], d.iloc[tr]["price_m2_raw"].values, coords[te])
        np_tr  = compute_local_neighbor_price(coords[tr], d.iloc[tr]["price_m2_raw"].values, coords[tr])
        np_te  = compute_local_neighbor_price(coords[tr], d.iloc[tr]["price_m2_raw"].values, coords[te])

        X_train["spatial_lag_price"]    = lag_tr; X_test["spatial_lag_price"]    = lag_te
        X_train["precio_vecinal_local"]  = np_tr;  X_test["precio_vecinal_local"]  = np_te
        X_train["lag_x_accommodates"] = lag_tr * np.log1p(X_train["accommodates"])
        X_test["lag_x_accommodates"]  = lag_te * np.log1p(X_test["accommodates"])

        X_train = X_train.fillna(X_train.median()).fillna(0)
        X_test  = X_test.fillna(X_train.median()).fillna(0)

        knn = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=min(12, len(tr)), weights="distance")),
        ])
        knn.fit(X_train, y.iloc[tr])

        # Reconstruir precio_noche: pred_por_huesped × accommodates
        pred_por_huesped = np.expm1(knn.predict(X_test))
        pred_precio_noche = pred_por_huesped * accommodates[te]
        pred.extend(pred_precio_noche)
        real.extend(d["price"].iloc[te])         # price = precio_noche real

    if not real:
        return 0, 1, [], []
    r2   = r2_score(real, pred)
    mape = mean_absolute_percentage_error(real, pred)
    print(f"  k-NN → R²: {r2:.4f} | MAPE: {mape:.2%}")
    return r2, mape, real, pred

# ======================================================
# EJECUCIÓN
# ======================================================
re_est, pe_est, df_est = run_market_optimized(df[df.is_luxury == 0].copy(), "MERCADO ESTÁNDAR")
re_lux, pe_lux, df_lux = run_market_optimized(df[df.is_luxury == 1].copy(), "MERCADO LUJO")

rk_est, mk_est, rk_est_r, rk_est_p = run_knn_comparison(df_est, "MERCADO ESTÁNDAR")
rk_lux, mk_lux, rk_lux_r, rk_lux_p = run_knn_comparison(df_lux, "MERCADO LUJO")

y_true_en = np.array(re_est + re_lux)
y_pred_en = np.array(pe_est + pe_lux)
y_true_kn = np.concatenate([rk_est_r, rk_lux_r])
y_pred_kn = np.concatenate([rk_est_p, rk_lux_p])

# ======================================================
# CÁLCULO DE MÉTRICAS — debe ir ANTES de visualizaciones
# ======================================================

if len(y_true_en) > 0:
    # R² en escala logarítmica — la escala en que el modelo fue entrenado;
    # no amplifica errores por accommodates (métrica principal)
    log_true_en   = np.log1p(y_true_en)
    log_pred_en   = np.log1p(y_pred_en)
    r2_en_log     = r2_score(log_true_en, log_pred_en)
    # R² en escala precio_noche — útil para comunicación pero inflado
    # porque listings baratos dominan SS_res y SS_tot
    r2_en_nominal = r2_score(y_true_en, y_pred_en)
    mape_en       = mean_absolute_percentage_error(y_true_en, y_pred_en)
    rmse_en       = np.sqrt(mean_squared_error(log_true_en, log_pred_en))
else:
    r2_en_log = r2_en_nominal = mape_en = rmse_en = float("nan")

# k-NN: R² global sobre el pool completo (no promedio de R² por segmento)
# np.nanmean([r2_est, r2_lux]) sobrepondera lujo por asimetría de tamaños
if len(y_true_kn) > 0:
    log_true_kn   = np.log1p(y_true_kn)
    log_pred_kn   = np.log1p(y_pred_kn)
    r2_kn_log     = r2_score(log_true_kn, log_pred_kn)
    r2_kn_nominal = r2_score(y_true_kn, y_pred_kn)
    mape_kn       = mean_absolute_percentage_error(y_true_kn, y_pred_kn)
    rmse_kn       = np.sqrt(mean_squared_error(log_true_kn, log_pred_kn))
else:
    r2_kn_log = r2_kn_nominal = mape_kn = rmse_kn = float("nan")

# ======================================================
# VISUALIZACIONES
# ======================================================

if len(y_true_en) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel izquierdo: escala log (escala real del modelo)
    sns.regplot(x=log_true_en, y=log_pred_en, ax=axes[0],
                scatter_kws={"alpha": 0.15, "color": "steelblue"},
                line_kws={"color": "red"})
    axes[0].set_title(
        f"Escala log (modelo)\nR² = {r2_en_log:.3f}",
        fontsize=13,
    )
    axes[0].set_xlabel("log(Precio Real + 1)"); axes[0].set_ylabel("log(Precio Predicho + 1)")

    # Panel derecho: escala nominal (para comunicación)
    sns.regplot(x=y_true_en, y=y_pred_en, ax=axes[1],
                scatter_kws={"alpha": 0.15, "color": "darkorange"},
                line_kws={"color": "red"})
    axes[1].set_title(
        f"Escala precio_noche (MXN)\nR² = {r2_en_nominal:.3f}  [ver nota]",
        fontsize=13,
    )
    axes[1].set_xlabel("Precio Real (MXN/noche)"); axes[1].set_ylabel("Precio Predicho (MXN/noche)")

    fig.suptitle("ElasticNet + Clusters — AIRBNB CDMX", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "elasticnet_airbnb.png"), dpi=300)
    plt.close("all")

# ── Resumen ──
print("\n" + "#" * 80)
print("RESUMEN COMPARATIVO FINAL — AIRBNB CDMX")
print("#" * 80)
print(f"{'MÉTRICA':<26} | {'CLUSTER + ENET':<18} | {'k-NN':<18}")
print("-" * 70)
print(f"{'R² (escala log — modelo)':<26} | {r2_en_log:.4f}{'':<13} | {r2_kn_log:.4f}")
print(f"{'R² (escala precio_noche)':<26} | {r2_en_nominal:.4f}{'':<13} | {r2_kn_nominal:.4f}")
print(f"{'MAPE Global':<26} | {mape_en*100:.2f}%{'':<11} | {mape_kn*100:.2f}%")
print(f"{'RMSE Logarítmico':<26} | {rmse_en:.4f}{'':<13} | {rmse_kn:.4f}")
print("#" * 80)
print()
print("NOTA: El R² en escala log es el indicador principal de ajuste del modelo.")
print("      El R² en precio_noche se ve inflado porque accommodates amplifica")
print("      varianza en listings de alta capacidad que son minoría.")
print("#" * 80)

# ======================================================
# INTERPRETABILIDAD HEDÓNICA
# ======================================================
def fit_interpretable_elasticnet(data, label):
    """
    Ajusta ElasticNet con validación cruzada KFold para extraer coeficientes hedónicos.
    Los spatial lags se computan dentro del fold para evitar data leakage.
    Los coeficientes se promedian a través de los folds (metodología consistente
    con run_market_optimized).
    """
    print("\n" + "=" * 80)
    print(f"TOP 15 VARIABLES HEDÓNICAS — {label} (AIRBNB)")
    print("=" * 80)

    data     = data.copy()
    coords_l = data[["longitud", "latitud"]].values
    accomm   = data["accommodates"].values
    X_base   = data[static_features].copy()
    y        = np.log1p(data["price_m2_raw"])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    coef_accumulator = np.zeros(len(features))

    for tr, te in kf.split(X_base):
        X_train = X_base.iloc[tr].copy()
        X_test  = X_base.iloc[te].copy()

        lag_tr = compute_spatial_lag(coords_l[tr], data.iloc[tr]["price_m2_raw"].values, coords_l[tr])
        lag_te = compute_spatial_lag(coords_l[tr], data.iloc[tr]["price_m2_raw"].values, coords_l[te])
        np_tr  = compute_local_neighbor_price(coords_l[tr], data.iloc[tr]["price_m2_raw"].values, coords_l[tr])
        np_te  = compute_local_neighbor_price(coords_l[tr], data.iloc[tr]["price_m2_raw"].values, coords_l[te])

        X_train["spatial_lag_price"]    = lag_tr;  X_test["spatial_lag_price"]    = lag_te
        X_train["precio_vecinal_local"]  = np_tr;   X_test["precio_vecinal_local"]  = np_te
        X_train["lag_x_accommodates"] = lag_tr * np.log1p(X_train["accommodates"])
        X_test["lag_x_accommodates"]  = lag_te * np.log1p(X_test["accommodates"])

        X_all = pd.concat([X_train, X_test])
        X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(X_train.median()).fillna(0)
        X_train = X_all.iloc[:len(tr)]

        scaler = StandardScaler()
        Xs     = scaler.fit_transform(X_train[features])
        model  = ElasticNetCV(alphas=np.logspace(-4, -1, 25), l1_ratio=[0.4, 0.6, 0.8], cv=5)
        model.fit(Xs, y.iloc[tr])
        coef_accumulator += model.coef_

    mean_coefs = coef_accumulator / 5
    coef_df = (pd.DataFrame({"feature": features, "coef": mean_coefs})
               .assign(abs_coef=lambda d: d["coef"].abs())
               .sort_values("abs_coef", ascending=False))
    print(coef_df[["feature", "coef"]].head(15).to_string(index=False))
    return coef_df


if len(df_est) >= 50:
    coef_est = fit_interpretable_elasticnet(df_est, "MERCADO ESTÁNDAR")
if len(df_lux) >= 50:
    coef_lux = fit_interpretable_elasticnet(df_lux, "MERCADO LUJO")

# ======================================================
# EXPORTACIÓN PARA QGIS
# ======================================================
print("\n" + "=" * 80)
print("EXPORTACIÓN PARA QGIS")
print("=" * 80)

df_qgis = pd.concat([df_est, df_lux], ignore_index=True)

# Presión local: ratio precio_propio / precio_vecindad (en escala precio/huésped)
# Valores > 0.5 → el listing tiene precio por encima de su microentorno
def calcular_gentrificacion_local(df_pts):
    coords_l = df_pts[["longitud", "latitud"]].values
    nbrs = NearestNeighbors(n_neighbors=min(21, len(df_pts)), algorithm="ball_tree").fit(coords_l)
    _, indices = nbrs.kneighbors(coords_l)
    scores = []
    for i in range(len(df_pts)):
        vecinos_idx = indices[i][1:]
        p_propio = df_pts.iloc[i]["price_m2_raw"]
        p_vec    = df_pts.iloc[vecinos_idx]["price_m2_raw"].mean()
        ratio    = np.clip(p_propio / (p_vec + 1e-5), 0, 5)
        scores.append(ratio / (1 + ratio))
    return scores

df_qgis["gentrif_local_presion"] = calcular_gentrificacion_local(df_qgis)
df_qgis["gentrif_map_score"] = (
    0.75 * df_qgis["gentrification_index"]
    + 0.25 * df_qgis["gentrif_local_presion"]
)

if "precio_predicho" in df_qgis.columns:
    df_qgis["error_pct"] = (
        (df_qgis["precio_predicho"] - df_qgis["price"]) / (df_qgis["price"] + 1e-5) * 100
    )
df_qgis["segmento"] = np.where(df_qgis["is_luxury"] == 1, "Lujo", "Estandar")

# Exportar dataset limpio (sin columnas de texto largas)
cols_drop_export = [
    "listing_url", "description", "neighborhood_overview", "picture_url",
    "host_url", "host_about", "host_thumbnail_url", "host_picture_url",
    "amenities", "amenities_list",
]
df_export = df_qgis.drop(columns=[c for c in cols_drop_export if c in df_qgis.columns])

ruta_qgis = os.path.join(RESULTS_PATH, "resultados_qgis_airbnb.csv")
df_export.to_csv(ruta_qgis, index=False, encoding="utf-8-sig")
print(f"  ✓ {len(df_export)} listings exportados → {ruta_qgis}")

# Dataset procesado completo (para análisis externo)
ruta_processed = os.path.join(
    DATA_PATH, "processed", "dataset_geocodificado_airbnb.csv"
)
os.makedirs(os.path.dirname(ruta_processed), exist_ok=True)
df_export.to_csv(ruta_processed, index=False, encoding="utf-8-sig")
print(f"  ✓ Dataset procesado → {ruta_processed}")

print("\n" + "=" * 80)
print("PIPELINE AIRBNB COMPLETADO")
print("=" * 80)