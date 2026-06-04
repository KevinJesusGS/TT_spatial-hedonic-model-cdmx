# =============================================================================
# TRABAJO TERMINAL
# Modelo de avalúo inmobiliario con técnicas de análisis espacial
# ESCALA: Alcaldía Gustavo A. Madero (GAM) — v5 (validación espacial estricta)
#
# Cambios respecto a v4:
#   - Filtro espacial estricto usando polígono de GAM (shapefile de alcaldías)
#   - Eliminación de puntos que caen fuera de GAM o en colonias no pertenecientes
#   - Mejora en la asignación de colonia_oficial: solo se conservan puntos con sjoin dentro de colonias_gam
# =============================================================================

import os
import glob
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error,
    silhouette_score,
)
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import libpysal
from esda.moran import Moran

warnings.filterwarnings("ignore")

# ======================================================
# CONFIGURACIÓN DE EJECUCIÓN
# ======================================================
MOSTRAR_GRAFICAS = False
GUARDAR_GRAFICAS = True
GEOCODIFICAR     = False   # False si ya existe venta_gam_geocodificado.csv

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["figure.dpi"]     = 120

# ======================================================
# RUTAS
# ======================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(PROJECT_ROOT, "..", "data")
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "..", "outputs")
RESULTS_PATH = os.path.join(OUTPUTS_PATH, "results")
FIGURES_PATH = os.path.join(OUTPUTS_PATH, "figures")

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

# Datasets principales
path_viviendas = os.path.join(DATA_PATH, "raw", "venta_gam_limpio.csv")
path_geo_cache = os.path.join(DATA_PATH, "venta_gam_geocodificado.csv")

# INEGI 2010 y 2020
path_inegi_2010 = os.path.join(DATA_PATH, "raw", "INEGI", "inegi_2010.csv")
path_inegi_2020 = os.path.join(DATA_PATH, "raw", "INEGI", "inegi_2020.csv")

# Transporte masivo
path_metro    = os.path.join(DATA_PATH, "raw", "stcmetro_shp", "STC_Metro_estaciones_utm14n.shp")
path_metrobus = os.path.join(DATA_PATH, "raw", "mb_shp", "Metrobus_estaciones.shp")
path_tren     = os.path.join(DATA_PATH, "raw", "ste_shp", "ste_tren_ligero_shp", "ste_tren_ligero_shp", "STE_TrenLigero_estaciones_utm14n.shp")
path_trole    = os.path.join(DATA_PATH, "raw", "ste_shp", "ste_trolebus_shp", "ste_trolebus_shp", "STE_Trolebus_Paradas.shp")
path_cable    = os.path.join(DATA_PATH, "raw", "ste_shp", "ste_cablebus_shp", "ste_cablebus_shp", "STE_Cablebus_estaciones.shp")

# Socioeconómico / comercio / seguridad
path_seguridad = os.path.join(DATA_PATH, "raw", "crimen", "urbanismo_social_sintesis.shp")
path_comercio  = os.path.join(DATA_PATH, "raw", "cypc", "C_PComerciales.shp")

# Alcaldías y colonias
path_alcaldias = os.path.join(DATA_PATH, "raw", "alcaldias", "poligonos_alcaldias_cdmx.shp")
path_colonias  = os.path.join(DATA_PATH, "raw", "coloniascdmx", "colonias_iecm.shp")

# Ciudad de 15 minutos
path_ciclovias           = os.path.join(DATA_PATH, "raw", "infraestructura_vial_ciclista", "Infraestructura ciclista total.shp")
path_areas_verdes        = os.path.join(DATA_PATH, "raw", "inventario_areas_verdes_1", "inventario_areas_verdes_1.shp")
path_salud               = os.path.join(DATA_PATH, "raw", "hospitales_y_centros_de_salud", "hospitales_y_centros_de_salud.shp")
path_hospitales_publicos = os.path.join(DATA_PATH, "raw", "hospitales_2020_publicos", "hospitales_2020_publicos.shp")
path_escuelas_pub        = os.path.join(DATA_PATH, "raw", "escuelas_publicas", "escuelas_publicas.shp")
path_escuelas_priv       = os.path.join(DATA_PATH, "raw", "escuelas_privadas", "escuelas_privadas.shp")
path_uso_suelo           = os.path.join(DATA_PATH, "raw", "uso-de-suelo", "uso-de-suelo.shp")

# Catastro (solo GAM)
path_catastro = os.path.join(DATA_PATH, "raw", "Catastrales")

# ======================================================
# SUBCENTROS URBANOS INTERNOS DE GAM
# ======================================================
SUBCENTROS_GAM = {
    "Indios_Verdes": (19.4963, -99.1178),
    "Lindavista":    (19.4797, -99.1386),
    "La_Raza":       (19.4559, -99.1439),
    "Vallejo":       (19.4643, -99.1633),
    "Aragon":        (19.4586, -99.0950),
    "CentroGAM":     (19.4750, -99.1100),
}

# Centro Histórico (para proxy turístico)
CENTRO_HISTORICO = (19.4326, -99.1332)

# ======================================================
# HELPERS (unificados con modelo_regresion_espacial.py)
# ======================================================

def clean_text(txt):
    if pd.isna(txt):
        return txt
    t = str(txt).lower().strip()
    t = t.translate(str.maketrans("áéíóúüñ", "aeiouun"))
    return t

def nearest_distance(pts, coords):
    if len(pts) == 0:
        return np.ones(len(coords)) * 5000
    tree = cKDTree(pts)
    d, _ = tree.query(coords)
    return d

def density_proxy(gdf_utm, pts, r=1000):
    if len(pts) == 0:
        return np.zeros(len(gdf_utm))
    tree   = cKDTree(pts)
    obs    = np.array([(p.x, p.y) for p in gdf_utm.geometry])
    counts = tree.query_ball_point(obs, r)
    return np.array([len(c) for c in counts])

def line_to_points(gdf, step=100):
    pts = []
    for geom in gdf.geometry:
        if geom.geom_type == "LineString":
            for d in np.arange(0, geom.length, step):
                p = geom.interpolate(d)
                pts.append((p.x, p.y))
    return np.array(pts) if pts else np.empty((0, 2))

def safe_load_geodata(path, label, epsg=32614):
    if not os.path.exists(path):
        print(f"  ⚠ {label}: archivo no encontrado → fallback")
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
        print(f"  ⚠ Error al cargar {label}: {e}")
        return None

def mostrar_o_guardar(nombre_archivo):
    if GUARDAR_GRAFICAS:
        ruta = os.path.join(FIGURES_PATH, nombre_archivo)
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
    if MOSTRAR_GRAFICAS:
        plt.show()
    else:
        print(f"  [INFO] Gráfica guardada: {nombre_archivo}")
    plt.close("all")

# ======================================================
# CATASTRO EXCLUSIVO PARA GAM
# ======================================================

def cargar_catastro_gam(path_catastro, gdf_utm_ref=None):
    print("\n" + "=" * 80)
    print("CARGANDO CATASTRO — GUSTAVO A. MADERO (solo alcaldía)")
    print("=" * 80)

    if not os.path.isdir(path_catastro):
        print("  ⚠ Carpeta catastro no encontrada → se omiten features catastrales")
        return gpd.GeoDataFrame()

    shp_files = glob.glob(os.path.join(path_catastro, "*.shp"))
    gam_shp = None
    for shp in shp_files:
        if "GUSTAVO_A_MADERO" in os.path.basename(shp).upper():
            gam_shp = shp
            break

    if gam_shp is None:
        print("  ⚠ No se encontró shapefile de GUSTAVO_A_MADERO en la carpeta Catastrales")
        return gpd.GeoDataFrame()

    nombre_base = os.path.splitext(os.path.basename(gam_shp))[0]
    csv_path = os.path.join(path_catastro, nombre_base + ".csv")

    if not os.path.exists(csv_path):
        print(f"  ⚠ CSV no encontrado para {nombre_base}")
        return gpd.GeoDataFrame()

    try:
        geo = gpd.read_file(gam_shp, engine="fiona")
        attrs = pd.read_csv(csv_path, low_memory=False)

        if "fid" not in geo.columns or "fid" not in attrs.columns:
            print(f"  ⚠ Sin columna fid en {nombre_base}")
            return gpd.GeoDataFrame()

        geo["fid"] = geo["fid"].astype(str).str.strip().str.replace(".0", "", regex=False)
        attrs["fid"] = attrs["fid"].astype(str).str.strip().str.replace(".0", "", regex=False)

        catastro = geo.merge(attrs, on="fid", how="left")
        if catastro.crs is None:
            catastro = catastro.set_crs("EPSG:4326")
        catastro = catastro.to_crs(epsg=32614)

        for c in ["sup_terreno", "sup_construccion", "anio_construccion",
                  "valor_unitario_suelo", "valor_suelo"]:
            if c in catastro.columns:
                catastro[c] = pd.to_numeric(catastro[c], errors="coerce")

        catastro["antiguedad"] = 2025 - catastro.get("anio_construccion", 2000)
        if "anio_construccion" in catastro.columns:
            catastro = catastro[(catastro["antiguedad"] >= 0) & (catastro["antiguedad"] <= 150)]

        if "sup_terreno" in catastro.columns and "sup_construccion" in catastro.columns:
            catastro["ratio_construccion"] = np.where(
                catastro["sup_terreno"] > 0,
                catastro["sup_construccion"] / catastro["sup_terreno"],
                np.nan,
            )

        if "valor_suelo" in catastro.columns:
            catastro = catastro[catastro["valor_suelo"] > 0]

        print(f"  ✓ Catastro GAM consolidado: {len(catastro)} polígonos")
        return catastro

    except Exception as e:
        print(f"  ⚠ Error al cargar catastro GAM: {e}")
        return gpd.GeoDataFrame()


def agregar_features_catastrales(df_main, gdf_utm_main, catastro, radio=200):
    cat_cols = [
        "cat_mean_valor_suelo_200m",
        "cat_mean_vus_200m",
        "cat_mean_antiguedad_200m",
        "cat_std_valor_suelo_200m",
        "cat_mean_ratio_construccion_200m",
        "cat_density_predios_200m",
    ]

    if catastro.empty:
        for c in cat_cols:
            df_main[c] = np.nan
        return df_main

    catastro_pts = catastro.copy()
    catastro_pts["geometry"] = catastro_pts.geometry.centroid
    cols_cat = [c for c in ["valor_suelo", "valor_unitario_suelo",
                             "ratio_construccion", "antiguedad"]
                if c in catastro_pts.columns]
    catastro_pts = catastro_pts[["geometry"] + cols_cat].dropna(
        subset=["valor_suelo"] if "valor_suelo" in cols_cat else []
    )
    sindex = catastro_pts.sindex

    res = {c: [] for c in cat_cols}

    for geom in gdf_utm_main.geometry:
        try:
            bounds   = geom.buffer(radio).bounds
            posibles = list(sindex.intersection(bounds))
            vecinos  = catastro_pts.iloc[posibles]
            vecinos  = vecinos[vecinos.geometry.distance(geom) <= radio]
            if len(vecinos) == 0:
                for c in cat_cols:
                    res[c].append(np.nan)
                continue
            res["cat_mean_valor_suelo_200m"].append(
                vecinos["valor_suelo"].mean() if "valor_suelo" in vecinos.columns else np.nan
            )
            res["cat_mean_vus_200m"].append(
                vecinos["valor_unitario_suelo"].mean() if "valor_unitario_suelo" in vecinos.columns else np.nan
            )
            res["cat_mean_antiguedad_200m"].append(
                vecinos["antiguedad"].mean() if "antiguedad" in vecinos.columns else np.nan
            )
            res["cat_std_valor_suelo_200m"].append(
                vecinos["valor_suelo"].std() if "valor_suelo" in vecinos.columns else np.nan
            )
            res["cat_mean_ratio_construccion_200m"].append(
                vecinos["ratio_construccion"].mean() if "ratio_construccion" in vecinos.columns else np.nan
            )
            res["cat_density_predios_200m"].append(len(vecinos))
        except Exception:
            for c in cat_cols:
                res[c].append(np.nan)

    for c in cat_cols:
        df_main[c] = res[c]

    for c in cat_cols:
        df_main[c] = df_main[c].fillna(df_main[c].median())

    print("  ✓ Features catastrales agregadas (solo GAM)")
    return df_main

# ======================================================
# VALIDACIÓN ESPACIAL ESTRICTA: FILTRAR PUNTOS DENTRO DE GAM
# ======================================================
def filtrar_dentro_de_gam(gdf, path_alcaldias):
    """
    Filtra el GeoDataFrame de puntos para conservar solo aquellos que están
    dentro del polígono de la alcaldía Gustavo A. Madero.
    Retorna el GeoDataFrame filtrado y actualiza el DataFrame original.
    """
    print("\n" + "=" * 80)
    print("VALIDACIÓN ESPACIAL ESTRICTA: PUNTOS DENTRO DE GAM")
    print("=" * 80)

    if not os.path.exists(path_alcaldias):
        print("  ⚠ Archivo de alcaldías no encontrado. No se puede filtrar espacialmente.")
        return gdf

    alcaldias = gpd.read_file(path_alcaldias).to_crs(gdf.crs)
    # Buscar el polígono de Gustavo A. Madero (puede llamarse NOMGEO, NOMBRE, etc.)
    col_nombre = None
    for col in ["NOMGEO", "NOMBRE", "NOM_MUN", "DEMARCACION"]:
        if col in alcaldias.columns:
            col_nombre = col
            break
    if col_nombre is None:
        print("  ⚠ No se pudo identificar columna con nombre de alcaldía. Se usará bbox.")
        # fallback: usar bbox como antes
        return gdf

    alcaldias["nombre_clean"] = alcaldias[col_nombre].apply(clean_text)
    gam_poly = alcaldias[alcaldias["nombre_clean"].str.contains("gustavo", na=False)]
    if gam_poly.empty:
        print("  ⚠ No se encontró el polígono de Gustavo A. Madero en el shapefile.")
        return gdf

    # Realizar sjoin para verificar puntos dentro
    joined = gpd.sjoin(gdf, gam_poly[["geometry"]], how="inner", predicate="within")
    n_dentro = len(joined)
    n_total = len(gdf)
    print(f"  Puntos dentro del polígono de GAM: {n_dentro} de {n_total} ({100*n_dentro/n_total:.1f}%)")
    if n_dentro < n_total:
        print(f"  Eliminando {n_total - n_dentro} puntos fuera de GAM.")
        gdf_filtrado = gdf.iloc[joined.index].copy()
        return gdf_filtrado
    else:
        return gdf

# ======================================================
# 1. CARGA Y PREPROCESAMIENTO
# ======================================================
print("=" * 80)
print("1/8  CARGA Y PREPROCESAMIENTO")
print("=" * 80)

df_raw = pd.read_csv(path_viviendas)
print(f"Registros totales cargados : {len(df_raw)}")

df = df_raw.rename(columns={
    "precio_mxn":    "price",
    "recamaras":     "rooms",
    "banos":         "bathrooms",
    "m2":            "area",
    "colonia":       "colonia",
    "ubicacion":     "ubicacion_raw",
    "tipo_inmueble": "tipo_inmueble",
})

df["price"]     = pd.to_numeric(df["price"],     errors="coerce")
df["area"]      = pd.to_numeric(df["area"],      errors="coerce")
df["rooms"]     = pd.to_numeric(df["rooms"],     errors="coerce").fillna(0)
df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce").fillna(0)
df = df.dropna(subset=["price", "area"]).copy()
df = df[df["price"] > 100_000].copy()
df = df[df["area"]  >= 20].copy()

df["antiguedad"]     = 0
df["parking_spaces"] = 0
df["alcaldia"]       = "Gustavo A. Madero"

df["tipo_clean"] = df["tipo_inmueble"].apply(clean_text)
df["alc"]        = df["alcaldia"].apply(clean_text)

print(f"Registros tras filtrado básico : {len(df)}")

df["price_m2_raw"] = df["price"] / df["area"]
p3  = df["price_m2_raw"].quantile(0.03)
p97 = df["price_m2_raw"].quantile(0.97)
mask_atipicos = (
    (df["price_m2_raw"] < p3)  |
    (df["price_m2_raw"] > p97) |
    (df["area"]  > 600)        |
    (df["price"] > 50_000_000)
)
df = df[~mask_atipicos].reset_index(drop=True)
df["price_m2_raw"] = df["price"] / df["area"]
print(f"[FIX-5] Atípicos eliminados    : {mask_atipicos.sum()}")
print(f"        Registros residenciales: {len(df)}")

# Geocodificación (cache)
print("\n" + "=" * 80)
print("1B/8  GEOCODIFICACIÓN (Nominatim / cache)")
print("=" * 80)

os.makedirs(os.path.dirname(path_geo_cache), exist_ok=True)

if not GEOCODIFICAR and os.path.exists(path_geo_cache):
    geo_cache = pd.read_csv(path_geo_cache)
    df = df.merge(
        geo_cache[["ubicacion_raw", "latitud", "longitud"]],
        on="ubicacion_raw", how="left"
    )
    print(f"  ✓ Cache cargado ({len(geo_cache)} registros geocodificados)")
else:
    try:
        from geopy.geocoders import Nominatim
        from geopy.extra.rate_limiter import RateLimiter

        geolocator = Nominatim(user_agent="gam_tt_2026", timeout=10)
        geocode    = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

        df["query_geo"] = (
            df["colonia"].fillna("").str.strip()
            + ", Gustavo A. Madero, Ciudad de México, México"
        )
        lats, lons = [], []
        total = len(df)
        for i, (_, row) in enumerate(df.iterrows(), 1):
            try:
                loc = geocode(row["query_geo"])
                lats.append(loc.latitude  if loc else np.nan)
                lons.append(loc.longitude if loc else np.nan)
            except Exception:
                lats.append(np.nan)
                lons.append(np.nan)
            if i % 50 == 0:
                print(f"  Geocodificados: {i}/{total}")

        df["latitud"]  = lats
        df["longitud"] = lons
        df[["ubicacion_raw", "latitud", "longitud"]].drop_duplicates().to_csv(
            path_geo_cache, index=False
        )
        print(f"  ✓ Cache guardado en: {path_geo_cache}")

    except ImportError:
        print("  ⚠ geopy no instalado → FALLBACK coordenadas aleatorias (demo)")
        np.random.seed(42)
        n = len(df)
        df["latitud"]  = np.random.uniform(19.42, 19.52, n)
        df["longitud"] = np.random.uniform(-99.17, -99.07, n)

df = df.dropna(subset=["latitud", "longitud"]).copy()

# Filtro bbox preliminar (amplio)
LAT_MIN, LAT_MAX = 19.35, 19.60
LON_MIN, LON_MAX = -99.25, -99.00
fuera_bbox = (
    (df["latitud"]  < LAT_MIN) | (df["latitud"]  > LAT_MAX) |
    (df["longitud"] < LON_MIN) | (df["longitud"] > LON_MAX)
)
print(f"  Fuera del bbox preliminar: {fuera_bbox.sum()} → eliminados")
df = df[~fuera_bbox].reset_index(drop=True)
print(f"  Registros tras bbox: {len(df)}")

gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["longitud"], df["latitud"]),
    crs="EPSG:4326",
)

# Validación espacial estricta con polígono de GAM
gdf = filtrar_dentro_de_gam(gdf, path_alcaldias)
df = gdf.drop(columns="geometry").copy()
print(f"  Registros dentro de GAM: {len(df)}")

gdf_utm    = gdf.to_crs(epsg=32614)
coords_utm = np.array([(p.x, p.y) for p in gdf_utm.geometry])

# ======================================================
# NEW-1 | ÍNDICE DE GENTRIFICACIÓN PCA (INEGI 2010-2020)
# ======================================================
print("\n" + "=" * 80)
print("NEW-1 | ÍNDICE DE GENTRIFICACIÓN PCA — GAM (INEGI 2010-2020)")
print("=" * 80)

GENTRIFICATION_INDEX_GAM = 0.83

try:
    c10 = pd.read_csv(path_inegi_2010)
    c20 = pd.read_csv(path_inegi_2020)

    vars_soc = [
        "pct_educ_sup", "pct_internet", "vph_pc", "vph_autom",
        "graproes", "pea", "prom_ocup", "pder_ss",
    ]

    c10_g = c10.groupby("nom_mun")[vars_soc].mean().reset_index()
    c20_g = c20.groupby("nom_mun")[vars_soc].mean().reset_index()

    c10_g["alc"] = c10_g["nom_mun"].apply(clean_text)
    c20_g["alc"] = c20_g["nom_mun"].apply(clean_text)

    delta = c10_g.merge(c20_g, on="alc", suffixes=("_10", "_20"))

    for v in vars_soc:
        delta[f"d_{v}"] = delta[f"{v}_20"] - delta[f"{v}_10"]

    cols_delta = [f"d_{v}" for v in vars_soc]
    X_pca  = StandardScaler().fit_transform(delta[cols_delta].fillna(0))
    gentrif_raw = PCA(n_components=1).fit_transform(X_pca).flatten()

    gmin, gmax = gentrif_raw.min(), gentrif_raw.max()
    delta["gentrification_index"] = (
        (gentrif_raw - gmin) / (gmax - gmin)
        if gmax > gmin else np.ones(len(delta)) * 0.5
    )
    delta["alc"] = delta["alc"].apply(clean_text)

    gam_row = delta[delta["alc"].str.contains("gustavo", na=False)]
    if len(gam_row) > 0:
        GENTRIFICATION_INDEX_GAM = float(gam_row["gentrification_index"].values[0])
        print(f"  ✓ Índice calculado (PCA 8 vars): GAM = {GENTRIFICATION_INDEX_GAM:.4f}")
    else:
        print(f"  ⚠ No se encontró GAM en delta → usando fallback {GENTRIFICATION_INDEX_GAM}")

except Exception as e:
    print(f"  ⚠ Error en PCA INEGI: {e}  → fallback {GENTRIFICATION_INDEX_GAM}")

df["gentrification_index"] = GENTRIFICATION_INDEX_GAM

# ======================================================
# 2. GRÁFICAS EXPLORATORIAS (resumidas)
# ======================================================
print("\n" + "=" * 80)
print("2/8  GRÁFICAS EXPLORATORIAS")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Distribución de Variables Clave — GAM", fontsize=14, fontweight="bold")
axes[0].hist(np.log1p(df["price"]),        bins=40, color="steelblue", edgecolor="white")
axes[0].set_title("log(Precio MXN)")
axes[1].hist(np.log1p(df["price_m2_raw"]), bins=40, color="teal",      edgecolor="white")
axes[1].set_title("log(Precio/m²)")
axes[2].hist(np.log1p(df["area"]),         bins=40, color="coral",     edgecolor="white")
axes[2].set_title("log(Área m²)")
plt.tight_layout()
mostrar_o_guardar("01_distribucion_variables.png")

tipo_counts = df["tipo_inmueble"].value_counts()
fig, axes   = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Composición del Dataset — GAM", fontsize=14, fontweight="bold")

nombres_cortos = [(n[:16] + "…") if len(n) > 18 else n for n in tipo_counts.index]
wedges, texts, autotexts = axes[0].pie(
    tipo_counts.values, labels=nombres_cortos,
    autopct="%1.1f%%", pctdistance=0.78, labeldistance=1.18,
    startangle=140, colors=sns.color_palette("Set2", len(tipo_counts)),
    wedgeprops={"edgecolor": "white", "linewidth": 1.2},
)
for t in texts:    t.set_fontsize(9)
for at in autotexts: at.set_fontsize(8); at.set_fontweight("bold")
axes[0].set_title("Distribución por Tipo de Inmueble", pad=18)

mask_res = df["tipo_inmueble"].isin(["Casa", "Departamento"])
if mask_res.sum() > 0:
    sns.boxplot(
        data=df[mask_res], x="tipo_inmueble", y="price_m2_raw",
        ax=axes[1], palette="Set2", showfliers=False,
    )
    axes[1].set_title("Precio/m² por Tipo (sin outliers extremos)")
    axes[1].set_xlabel("Tipo de Inmueble")
    axes[1].set_ylabel("Precio/m² (MXN)")

plt.tight_layout()
mostrar_o_guardar("02_tipos_inmueble.png")

# ======================================================
# COBERTURA DE COLONIAS (FIX-2) Y ASIGNACIÓN DE COLONIA_OFICIAL
# ======================================================
print("\n--- Análisis de Cobertura de Colonias ---")

colonias_gdf = None
COLONIAS_OFICIALES_GAM = 193

if os.path.exists(path_colonias):
    try:
        colonias_shp = gpd.read_file(path_colonias)
        if colonias_shp.crs != "EPSG:4326":
            colonias_shp = colonias_shp.to_crs("EPSG:4326")

        alc_col = next((c for c in ["NOMDT", "DEMARCACI", "DEMARCACION", "MUNICIPIO", "NOM_MUN"]
                        if c in colonias_shp.columns), None)
        col_col = next((c for c in ["NOMUT", "NOMBRE", "NOM_COL", "COLONIA"]
                        if c in colonias_shp.columns), None)

        if alc_col and col_col:
            colonias_shp["alc_clean"] = colonias_shp[alc_col].apply(clean_text)
            colonias_shp["col_clean"] = colonias_shp[col_col].apply(clean_text)

            colonias_gam = colonias_shp[
                colonias_shp["alc_clean"].str.contains("gustavo", na=False)
            ].copy()

            COLONIAS_OFICIALES_GAM = colonias_gam["col_clean"].nunique()

            # Spatial join de puntos con colonias GAM
            joined_col = gpd.sjoin(
                gdf, colonias_gam[["col_clean", "geometry"]],
                how="left", predicate="within",
            )
            # Asignar colonia_oficial solo si el punto cayó dentro de alguna colonia GAM
            df["colonia_oficial"] = joined_col["col_clean"].values
            # Los puntos que no cayeron en ninguna colonia (NaN) se eliminarán más adelante
            colonias_gdf = colonias_gam[["col_clean", "geometry"]].copy()
            colonias_gdf = colonias_gdf.rename(columns={"col_clean": "colonia_oficial"})

            colonias_con_datos = df["colonia_oficial"].nunique()
            print(f"  ✓ Colonias oficiales en GAM  : {COLONIAS_OFICIALES_GAM}")
            print(f"  ✓ Colonias con ≥1 registro   : {colonias_con_datos}")
        else:
            raise ValueError("Columnas NOMUT/NOMDT no encontradas")

    except Exception as e:
        print(f"  ⚠ Error colonias: {e}")
        df["colonia_oficial"]  = df["colonia"].apply(clean_text)
        colonias_con_datos     = df["colonia_oficial"].nunique()
else:
    print("  ⚠ path_colonias no encontrado")
    df["colonia_oficial"]  = df["colonia"].apply(clean_text)
    colonias_con_datos     = df["colonia_oficial"].nunique()

# Eliminar puntos que no tienen colonia_oficial (quedaron fuera de las colonias de GAM)
n_sin_colonia = df["colonia_oficial"].isna().sum()
if n_sin_colonia > 0:
    print(f"  Eliminando {n_sin_colonia} puntos que no pertenecen a ninguna colonia de GAM.")
    df = df.dropna(subset=["colonia_oficial"]).reset_index(drop=True)
    # Actualizar gdf y gdf_utm
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitud"], df["latitud"]),
        crs="EPSG:4326",
    )
    gdf_utm = gdf.to_crs(epsg=32614)
    coords_utm = np.array([(p.x, p.y) for p in gdf_utm.geometry])
    print(f"  Registros tras filtro de colonia: {len(df)}")

pct_cobertura = (df["colonia_oficial"].nunique() / COLONIAS_OFICIALES_GAM) * 100
print(f"  Cobertura estimada          : {pct_cobertura:.1f}%")

colonia_counts = df["colonia_oficial"].value_counts().dropna()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Cobertura de Colonias — Gustavo A. Madero",
             fontsize=14, fontweight="bold")
colonia_counts.head(20).plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="white")
axes[0].set_title("Top 20 Colonias por Número de Anuncios")
axes[0].set_xlabel("Colonia")
axes[0].set_ylabel("Registros")
axes[0].tick_params(axis="x", rotation=52)

axes[1].hist(colonia_counts.values, bins=30, color="teal", edgecolor="white")
axes[1].axvline(colonia_counts.mean(), color="red", linestyle="--",
                label=f"Media: {colonia_counts.mean():.1f}")
axes[1].set_title("Distribución de Registros por Colonia")
axes[1].set_xlabel("Registros por colonia")
axes[1].set_ylabel("Número de colonias")
axes[1].legend()
plt.tight_layout()
mostrar_o_guardar("03_cobertura_colonias.png")

# ======================================================
# 3. VALIDACIÓN ESPACIAL (mapa base)
# ======================================================
print("\n" + "=" * 80)
print("3/8  VALIDACIÓN ESPACIAL")
print("=" * 80)

fig, ax = plt.subplots(figsize=(12, 10))
scatter = ax.scatter(
    df["longitud"], df["latitud"],
    c=np.log1p(df["price_m2_raw"]),
    cmap="plasma", alpha=0.6, s=18, linewidths=0,
)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("log(Precio/m²)", fontsize=11)
ax.set_title(
    f"Distribución Espacial — GAM (n={len(df)})\n"
    f"Índ. Gentrificación GAM = {GENTRIFICATION_INDEX_GAM:.4f}",
    fontsize=13, fontweight="bold",
)
ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud")
for nombre, (lat, lon) in SUBCENTROS_GAM.items():
    ax.scatter(lon, lat, marker="*", s=220, color="white",
               edgecolors="black", linewidths=0.8, zorder=5)
    ax.annotate(nombre.replace("_", " "), xy=(lon, lat),
                xytext=(5, 5), textcoords="offset points", fontsize=8,
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))
plt.tight_layout()
mostrar_o_guardar("05_mapa_anuncios_gam.png")

# Reconstruir GeoDataFrame (por si hubo cambios)
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["longitud"], df["latitud"]),
    crs="EPSG:4326",
)
gdf_utm = gdf.to_crs(epsg=32614)
coords_utm = np.array([(p.x, p.y) for p in gdf_utm.geometry])

# ======================================================
# 4. GEOFEATURES ESPACIALES (usando safe_load_geodata)
# ======================================================
print("\n" + "=" * 80)
print("4/8  GEOFEATURES ESPACIALES")
print("=" * 80)

os.environ["SHAPE_RESTORE_SHX"] = "YES"

# METRO
metro = safe_load_geodata(path_metro, "Metro CDMX")
metro_pts = (
    np.array([(g.x, g.y) for g in metro.geometry.explode(index_parts=False)
              if g.geom_type == "Point"])
    if metro is not None else np.empty((0, 2))
)
df["dist_metro_m"]  = nearest_distance(metro_pts, coords_utm)
df["density_metro"] = density_proxy(gdf_utm, metro_pts, 500)

# METROBÚS
metrobus = safe_load_geodata(path_metrobus, "Metrobús")
mb_pts = (
    np.array([(g.x, g.y) for g in metrobus.geometry.explode(index_parts=False)
              if g.geom_type == "Point"])
    if metrobus is not None else np.empty((0, 2))
)
df["dist_metrobus_m"]  = nearest_distance(mb_pts, coords_utm)
df["density_metrobus"] = density_proxy(gdf_utm, mb_pts, 600)

# CABLEBÚS
cable = safe_load_geodata(path_cable, "Cablebús")
cable_pts = (
    np.array([(g.x, g.y) for g in cable.geometry.explode(index_parts=False)
              if g.geom_type == "Point"])
    if cable is not None else np.empty((0, 2))
)
df["dist_cable_m"]  = nearest_distance(cable_pts, coords_utm)
df["density_cable"] = density_proxy(gdf_utm, cable_pts, 400)

# TROLEBÚS
trole = safe_load_geodata(path_trole, "Trolebús")
trole_pts = (
    np.array([(g.x, g.y) for g in trole.geometry.explode(index_parts=False)
              if g.geom_type == "Point"])
    if trole is not None else np.empty((0, 2))
)
df["dist_trole_m"]  = nearest_distance(trole_pts, coords_utm)
df["density_trole"] = density_proxy(gdf_utm, trole_pts, 300)

# TREN LIGERO
tren = safe_load_geodata(path_tren, "Tren Ligero")
tren_pts = (
    np.array([(g.x, g.y) for g in tren.geometry.explode(index_parts=False)
              if g.geom_type == "Point"])
    if tren is not None else np.empty((0, 2))
)
df["dist_tren_m"]  = nearest_distance(tren_pts, coords_utm)
df["density_tren"] = density_proxy(gdf_utm, tren_pts, 500)

# COMERCIO
com = safe_load_geodata(path_comercio, "Centros Comerciales")
if com is not None:
    com_pts = np.array([(g.centroid.x, g.centroid.y) for g in com.geometry])
    df["comercio_density"] = density_proxy(gdf_utm, com_pts, 1500)
else:
    df["comercio_density"] = 0

# ======================================================
# MARGINALIDAD (por colonia mediante unión de polígonos)
# ======================================================
print("\n--- Cálculo de Marginalidad por Colonia ---")
seg = safe_load_geodata(path_seguridad, "Urbanismo Social (Marginalidad)", epsg=4326)
if seg is not None and colonias_gdf is not None:
    seg = seg.to_crs(colonias_gdf.crs)
    colonias_con_marg = gpd.sjoin(
        colonias_gdf,
        seg[["C_US", "geometry"]],
        how="left",
        predicate="intersects"
    )
    moda_por_colonia = (
        colonias_con_marg.groupby("colonia_oficial")["C_US"]
        .agg(lambda x: x.mode()[0] if not x.mode().empty else 3)
        .reset_index()
        .rename(columns={"C_US": "marginalidad_score"})
    )
    df = df.merge(moda_por_colonia, on="colonia_oficial", how="left")
    df["marginalidad_score"] = df["marginalidad_score"].fillna(3)
    print(f"  ✓ Marginalidad asignada a {df['colonia_oficial'].nunique()} colonias (moda de C_US)")
else:
    print("  ⚠ No se pudo calcular marginalidad por colonia → fallback con score_15min")
    df["marginalidad_score"] = np.nan

# CIUDAD DE 15 MINUTOS
print("\n  Cargando capas de Ciudad de 15 Minutos...")
ciclo    = safe_load_geodata(path_ciclovias,           "Ciclovías")
verdes   = safe_load_geodata(path_areas_verdes,        "Áreas Verdes")
salud_b  = safe_load_geodata(path_salud,               "Salud Base")
salud_p  = safe_load_geodata(path_hospitales_publicos, "Hospitales Públicos 2020")
esc_pub  = safe_load_geodata(path_escuelas_pub,        "Escuelas Públicas")
esc_priv = safe_load_geodata(path_escuelas_priv,       "Escuelas Privadas")

if ciclo is not None:
    ciclo_pts = line_to_points(ciclo, step=80)
    df["dist_ciclovia_m"]       = nearest_distance(ciclo_pts, coords_utm)
    df["densidad_ciclovia_15m"] = density_proxy(gdf_utm, ciclo_pts, 1000)
else:
    df["dist_ciclovia_m"]       = 5000
    df["densidad_ciclovia_15m"] = 0

if verdes is not None:
    verdes_pts = np.array([(p.x, p.y) for p in verdes.geometry.centroid])
    df["dist_area_verde_m"]    = nearest_distance(verdes_pts, coords_utm)
    df["densidad_parques_15m"] = density_proxy(gdf_utm, verdes_pts, 1000)
else:
    df["dist_area_verde_m"]    = 5000
    df["densidad_parques_15m"] = 0

salud_final_pts = []
if salud_b is not None:
    pts_base = np.array([(p.x, p.y) for p in salud_b.geometry])
    if len(pts_base) > 0:
        salud_final_pts.append(pts_base)
if salud_p is not None:
    if "CLAVE_DE_L" in salud_p.columns:
        salud_p = salud_p[salud_p["CLAVE_DE_L"] == "09"].copy()
    pts_pub = np.array([(p.x, p.y) for p in salud_p.geometry])
    if len(pts_pub) > 0:
        if salud_final_pts:
            t_exist   = cKDTree(np.vstack(salud_final_pts))
            dup_idx   = t_exist.query_ball_point(pts_pub, r=1.0)
            filtrados = [pts_pub[i] for i, v in enumerate(dup_idx) if len(v) == 0]
            if filtrados:
                salud_final_pts.append(np.array(filtrados))
        else:
            salud_final_pts.append(pts_pub)

if salud_final_pts:
    salud_pts = np.vstack(salud_final_pts)
    df["dist_salud_m"]     = nearest_distance(salud_pts, coords_utm)
    df["acceso_salud_15m"] = density_proxy(gdf_utm, salud_pts, 1000)
    print(f"  ✓ Salud unificada: {len(salud_pts)} nodos")
else:
    df["dist_salud_m"]     = 5000
    df["acceso_salud_15m"] = 0

esc_list = []
if esc_pub  is not None: esc_list.append(np.array([(p.x, p.y) for p in esc_pub.geometry]))
if esc_priv is not None: esc_list.append(np.array([(p.x, p.y) for p in esc_priv.geometry]))
if esc_list:
    esc_pts = np.vstack(esc_list)
    df["dist_escuela_m"]       = nearest_distance(esc_pts, coords_utm)
    df["acceso_educacion_15m"] = density_proxy(gdf_utm, esc_pts, 1000)
else:
    df["dist_escuela_m"]       = 5000
    df["acceso_educacion_15m"] = 0

# Score 15 minutos
componentes_15min = [
    "density_metro", "densidad_ciclovia_15m",
    "densidad_parques_15m", "acceso_salud_15m", "acceso_educacion_15m",
]
df["score_15min"] = df[componentes_15min].sum(axis=1)
rng = df["score_15min"].max() - df["score_15min"].min()
df["score_15min"] = (df["score_15min"] - df["score_15min"].min()) / (rng if rng > 0 else 1)
print("  ✓ Score de Ciudad de 15 Minutos calculado")

# Fallback de marginalidad si aún no se asignó
if df["marginalidad_score"].isna().all():
    df["marginalidad_score"] = 1 - df["score_15min"]
    print("  ⚠ Marginalidad = 1 − score_15min (fallback, sin capa C_US)")
else:
    df["marginalidad_score"] = df["marginalidad_score"].fillna(
        df["marginalidad_score"].median()
    )

# Proximidades exponenciales
df["prox_ciclovia"] = np.exp(-df["dist_ciclovia_m"]  / 500)
df["prox_parque"]   = np.exp(-df["dist_area_verde_m"] / 500)
df["prox_salud"]    = np.exp(-df["dist_salud_m"]      / 800)
df["prox_escuela"]  = np.exp(-df["dist_escuela_m"]    / 600)

# ======================================================
# USO DE SUELO (corregido con buffer de 1m)
# ======================================================
df["uso_mixto"] = 0
if os.path.exists(path_uso_suelo):
    uso = safe_load_geodata(path_uso_suelo, "Uso de Suelo", epsg=32614)
    if uso is not None:
        col_us = next((c for c in ["us_dscr", "USO", "DESCRIPCION", "CLAVE"]
                       if c in uso.columns), None)
        if col_us:
            uso["es_mixto"] = uso[col_us].str.contains("HM", na=False).astype(int)
            uso_buffer = uso.copy()
            uso_buffer["geometry"] = uso_buffer.geometry.buffer(1)
            gdf_tmp = gpd.sjoin(
                gdf_utm, uso_buffer[["es_mixto", "geometry"]],
                how="left", predicate="within"
            )
            df["uso_mixto"] = (
                gdf_tmp.groupby(gdf_tmp.index)["es_mixto"].max().fillna(0)
            )
            print(f"  ✓ Uso Mixto integrado (buffer 1m, col={col_us}, "
                  f"mixto={df['uso_mixto'].sum():.0f} puntos)")
        else:
            print("  ⚠ Columna de descripción en uso-de-suelo no encontrada → uso_mixto=0")
    else:
        print("  ⚠ No se pudo cargar uso-de-suelo → uso_mixto=0")
else:
    print("  ⚠ Archivo uso-de-suelo no encontrado → uso_mixto=0")

# ======================================================
# PROXY TURÍSTICO: DISTANCIA AL CENTRO HISTÓRICO
# ======================================================
centro_geom = gpd.GeoSeries(
    [gpd.points_from_xy([CENTRO_HISTORICO[1]], [CENTRO_HISTORICO[0]])[0]],
    crs="EPSG:4326"
).to_crs(epsg=32614)
centro_pt = np.array([(centro_geom.x.iloc[0], centro_geom.y.iloc[0])])
df["dist_centro_historico_m"] = nearest_distance(centro_pt, coords_utm)
df["dist_centro_historico_m"] = np.log1p(df["dist_centro_historico_m"])
print("  ✓ Distancia al Centro Histórico (proxy turístico) incorporada")

# Subcentros GAM
sub_gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(
        [lon for _, lon in SUBCENTROS_GAM.values()],
        [lat for lat, _ in SUBCENTROS_GAM.values()],
    ), crs="EPSG:4326",
).to_crs(epsg=32614)
sub_pts  = np.array([(p.x, p.y) for p in sub_gdf.geometry])
tree_sub = cKDTree(sub_pts)
dist_sub, _ = tree_sub.query(coords_utm)
df["dist_nearest_subcenter_m"] = dist_sub
df["dist_subcenter_log"]       = np.log1p(dist_sub)
print(f"  Dist. media a subcentro GAM: {dist_sub.mean():.0f} m")

# ======================================================
# CATASTRO — FEATURES ESPACIALES (solo GAM)
# ======================================================
catastro = cargar_catastro_gam(path_catastro, gdf_utm)
df = agregar_features_catastrales(df, gdf_utm, catastro)

cat_cols_feat = [
    "cat_mean_valor_suelo_200m",
    "cat_mean_vus_200m",
    "cat_mean_antiguedad_200m",
    "cat_std_valor_suelo_200m",
    "cat_mean_ratio_construccion_200m",
    "cat_density_predios_200m",
]

# ======================================================
# 5. FEATURES DERIVADAS
# ======================================================
print("\n" + "=" * 80)
print("5/8  FEATURES DERIVADAS")
print("=" * 80)

coords_geo = df[["longitud", "latitud"]].values
tree_local = cKDTree(coords_geo)
neighbors  = tree_local.query_ball_point(coords_geo, r=0.012)
df["listing_density_local"] = np.array([len(n) - 1 for n in neighbors])
df["listing_density_log"]   = np.log1p(df["listing_density_local"])

df["area_x_marginalidad"]          = np.log1p(df["area"]) * df["marginalidad_score"]
df["area_X_gentrif"]               = df["area"] * df["gentrification_index"]
df["gentrif_x_metro"]              = df["gentrification_index"] * df["density_metro"]
df["15min_X_gentrif"]              = df["score_15min"] * df["gentrification_index"]
df["verde_marginalidad_ratio"]     = np.log1p(
    df["densidad_parques_15m"]  / (df["marginalidad_score"] + 1)
)
df["salud_marginalidad_ratio"]     = np.log1p(
    df["acceso_salud_15m"]      / (df["marginalidad_score"] + 1)
)
df["educacion_marginalidad_ratio"] = np.log1p(
    df["acceso_educacion_15m"]  / (df["marginalidad_score"] + 1)
)

if "cat_mean_vus_200m" in df.columns:
    df["cat_vus_x_gentrif"] = (
        np.log1p(df["cat_mean_vus_200m"]) * df["gentrification_index"]
    )

log_cols = [
    "dist_metro_m", "dist_metrobus_m", "dist_tren_m", "dist_trole_m",
    "dist_cable_m", "dist_ciclovia_m", "dist_area_verde_m",
    "dist_salud_m", "dist_escuela_m", "comercio_density",
    "dist_centro_historico_m", "antiguedad",
]
for c in log_cols:
    if c in df.columns:
        df[c] = np.log1p(df[c])

df["spatial_lag_price"]    = np.nan
df["precio_vecinal_local"] = np.nan
df["lag_x_area"]           = np.nan

# ======================================================
# 6. AUTOCORRELACIÓN ESPACIAL — I DE MORAN
# ======================================================
print("\n" + "=" * 80)
print("6/8  AUTOCORRELACIÓN ESPACIAL — I DE MORAN")
print("=" * 80)

try:
    gdf_moran = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitud"], df["latitud"]),
        crs="EPSG:4326",
    ).to_crs(epsg=32614)
    gdf_moran = gdf_moran[np.isfinite(gdf_moran["price_m2_raw"])].reset_index(drop=True)
    y_moran   = gdf_moran["price_m2_raw"].values.astype(float)

    w_knn = libpysal.weights.KNN.from_dataframe(
        gdf_moran, k=min(8, len(gdf_moran) - 1)
    )
    w_knn.transform = "R"
    moran_global = Moran(y_moran, w_knn)

    i_val = float(np.asarray(moran_global.I).flat[0])
    p_val = float(np.asarray(moran_global.p_sim).flat[0])
    z_val = abs(float(np.asarray(moran_global.z_norm).flat[0]))

    print(f"  I de Moran Global : {i_val:.4f}")
    print(f"  p-valor           : {p_val:.4f}")
    print(f"  Z-score           : {z_val:.4f}")
    if p_val <= 0.05:
        print(f"  → Autocorrelación {'POSITIVA' if i_val > 0 else 'NEGATIVA'} significativa")

    lag_y = libpysal.weights.lag_spatial(w_knn, y_moran)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_moran, lag_y, alpha=0.4, s=14, color="steelblue")
    ax.axhline(lag_y.mean(),   color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(y_moran.mean(), color="gray", linestyle="--", linewidth=0.8)
    m_m, b_m = np.polyfit(y_moran, lag_y, 1)
    xline = np.linspace(y_moran.min(), y_moran.max(), 100)
    ax.plot(xline, m_m * xline + b_m, color="red", linewidth=1.5,
            label=f"I = {i_val:.4f}")
    ax.set_xlabel("Precio/m² (MXN)")
    ax.set_ylabel("Lag Espacial del Precio/m²")
    ax.set_title(
        f"Diagrama de Moran — GAM\n"
        f"p = {p_val:.4f} · Índ. Gentrificación = {GENTRIFICATION_INDEX_GAM:.4f}",
        fontsize=13,
    )
    ax.legend()
    plt.tight_layout()
    mostrar_o_guardar("08_moran_scatter.png")
except Exception as e:
    print(f"  ⚠ Error al calcular Moran: {e}")

# ======================================================
# 7. FEATURES, SEGMENTACIÓN Y MODELO
# ======================================================
print("\n" + "=" * 80)
print("7/8  FEATURES, SEGMENTACIÓN Y MODELO")
print("=" * 80)

static_features = [
    "rooms", "area", "bathrooms", "antiguedad",
    "dist_subcenter_log",
    "dist_metro_m",    "density_metro",
    "dist_metrobus_m", "density_metrobus",
    "dist_tren_m",     "density_tren",
    "dist_trole_m",    "density_trole",
    "dist_cable_m",    "density_cable",
    "dist_ciclovia_m",    "densidad_ciclovia_15m",
    "dist_area_verde_m",  "densidad_parques_15m",
    "dist_salud_m",       "acceso_salud_15m",
    "dist_escuela_m",     "acceso_educacion_15m",
    "score_15min",
    "prox_ciclovia", "prox_parque", "prox_salud", "prox_escuela",
    "comercio_density",
    "dist_centro_historico_m",
    "gentrification_index",
    "marginalidad_score",
    "uso_mixto",
    "area_x_marginalidad", "area_X_gentrif", "gentrif_x_metro",
    "15min_X_gentrif",
    "verde_marginalidad_ratio", "salud_marginalidad_ratio",
    "educacion_marginalidad_ratio",
    "listing_density_log",
    "cat_mean_valor_suelo_200m",
    "cat_mean_vus_200m",
    "cat_mean_antiguedad_200m",
    "cat_std_valor_suelo_200m",
    "cat_mean_ratio_construccion_200m",
    "cat_density_predios_200m",
    "cat_vus_x_gentrif",
]

dynamic_features = ["spatial_lag_price", "precio_vecinal_local", "lag_x_area"]

valid_static = []
for f in static_features:
    if f not in df.columns:
        print(f"  ⚠ Feature ausente: {f}")
        continue
    if df[f].std() < 1e-9:
        print(f"  ⚠ Varianza ~0, omitida: {f}")
        continue
    valid_static.append(f)
static_features = valid_static

corr_matrix = df[static_features].corr().abs()
upper       = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
drop_cols = [col for col in upper.columns if any(upper[col] > 0.90)]
if drop_cols:
    print(f"\n  Variables eliminadas por colinealidad (>0.90):")
    for c in drop_cols:
        print(f"    - {c}")
    static_features = [f for f in static_features if f not in drop_cols]

features_all = static_features + dynamic_features
print(f"\n  Features estáticas finales : {len(static_features)}")
print(f"  Features dinámicas         : {len(dynamic_features)}")

print("\n  Evaluando k óptimo para segmentación espacial...")
df_model_tmp = df[np.isfinite(df["price_m2_raw"]) & (df["price_m2_raw"] > 0)].copy()
coords_cluster_base = StandardScaler().fit_transform(
    df_model_tmp[["latitud", "longitud"]]
)
sil_scores = {}
for k in [2, 3, 4]:
    if len(df_model_tmp) < k * 25:
        continue
    km  = KMeans(n_clusters=k, random_state=42, n_init=10).fit(coords_cluster_base)
    if len(np.unique(km.labels_)) < 2:
        continue
    sil = silhouette_score(coords_cluster_base, km.labels_)
    sil_scores[k] = sil
    print(f"    k={k}: silhouette = {sil:.4f}")

N_CLUSTERS = max(sil_scores, key=sil_scores.get) if sil_scores else 2
print(f"\n  → N_CLUSTERS óptimo para GAM: {N_CLUSTERS}")

if sil_scores:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(list(sil_scores.keys()), list(sil_scores.values()),
           color="steelblue", edgecolor="white")
    ax.axvline(N_CLUSTERS, color="red", linestyle="--",
               label=f"Óptimo k={N_CLUSTERS}")
    ax.set_title("Silhouette Score por Número de Clústeres — GAM", fontsize=12)
    ax.set_xlabel("k clústeres")
    ax.set_ylabel("Silhouette Score")
    ax.legend()
    plt.tight_layout()
    mostrar_o_guardar("09_silhouette_clusters.png")

# ======================================================
# FUNCIONES ESPACIALES (SIN LEAKAGE)
# ======================================================

def compute_spatial_lag(train_coords, train_prices, target_coords, k=15):
    tree   = cKDTree(train_coords)
    fetch  = min(k + 1, len(train_coords))
    d, ix  = tree.query(target_coords, k=fetch)
    weights = 1.0 / (d + 10.0)
    lag     = [
        np.average(train_prices[idxs], weights=w)
        for idxs, w in zip(ix, weights)
    ]
    return np.log1p(np.array(lag))

def compute_local_neighbor_price(train_coords, train_prices, target_coords, k=12):
    nbrs = NearestNeighbors(
        n_neighbors=min(k + 1, len(train_coords)), algorithm="ball_tree"
    ).fit(train_coords)
    _, indices = nbrs.kneighbors(target_coords)
    if np.array_equal(train_coords, target_coords):
        indices = indices[:, 1:]
    return np.array([train_prices[idx].mean() for idx in indices])

# ======================================================
# MODELO PRINCIPAL — ELASTICNET + CLÚSTERES
# ======================================================

def run_model_gam(data, label="GAM", n_clusters=N_CLUSTERS, n_splits=3):
    print(f"\n{'='*80}")
    print(f"  ELASTICNET + CLÚSTERES: {label}")
    print(f"  N={len(data)} · k={n_clusters} · K-Fold={n_splits}")
    print(f"  Índ. Gentrificación GAM = {GENTRIFICATION_INDEX_GAM:.4f} (PCA dinámico)")
    print(f"{'='*80}")

    q_low, q_high = data["price_m2_raw"].quantile([0.05, 0.95])
    data = data[
        (data["price_m2_raw"] > q_low) & (data["price_m2_raw"] < q_high)
    ].copy()
    print(f"  Tras filtrado P5-P95 interno: {len(data)} registros")

    enriched_cluster = StandardScaler().fit_transform(
        data[["latitud", "longitud", "gentrification_index", "score_15min"]]
    )
    data["cluster"] = KMeans(
        n_clusters=n_clusters, random_state=42, n_init=10
    ).fit_predict(enriched_cluster)

    data["precio_predicho"] = np.nan
    global_real, global_pred = [], []

    for c in sorted(data["cluster"].unique()):
        d_c = data[data["cluster"] == c].copy()
        print(f"\n  → Clúster {c} | n = {len(d_c)}")
        if len(d_c) < n_splits * 8:
            print(f"    ⚠ Muy pocos datos → omitido")
            continue

        X_base   = d_c[static_features].copy()
        y        = np.log1p(d_c["price_m2_raw"])
        areas    = d_c["area"].values
        coords_c = d_c[["longitud", "latitud"]].values

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        c_real, c_pred = [], []
        train_r2_list  = []

        for fold, (tr, te) in enumerate(kf.split(X_base)):
            X_train = X_base.iloc[tr].copy()
            X_test  = X_base.iloc[te].copy()

            lag_tr = compute_spatial_lag(
                coords_c[tr], d_c.iloc[tr]["price_m2_raw"].values, coords_c[tr])
            lag_te = compute_spatial_lag(
                coords_c[tr], d_c.iloc[tr]["price_m2_raw"].values, coords_c[te])
            nvp_tr = compute_local_neighbor_price(
                coords_c[tr], d_c.iloc[tr]["price_m2_raw"].values, coords_c[tr])
            nvp_te = compute_local_neighbor_price(
                coords_c[tr], d_c.iloc[tr]["price_m2_raw"].values, coords_c[te])

            X_train["spatial_lag_price"]    = lag_tr
            X_test["spatial_lag_price"]     = lag_te
            X_train["precio_vecinal_local"] = nvp_tr
            X_test["precio_vecinal_local"]  = nvp_te
            X_train["lag_x_area"] = X_train["spatial_lag_price"] * np.log1p(X_train["area"])
            X_test["lag_x_area"]  = X_test["spatial_lag_price"]  * np.log1p(X_test["area"])

            data.loc[d_c.iloc[tr].index, "spatial_lag_price"]    = lag_tr
            data.loc[d_c.iloc[te].index, "spatial_lag_price"]    = lag_te
            data.loc[d_c.iloc[tr].index, "precio_vecinal_local"] = nvp_tr
            data.loc[d_c.iloc[te].index, "precio_vecinal_local"] = nvp_te
            data.loc[d_c.iloc[tr].index, "lag_x_area"] = (
                lag_tr * np.log1p(d_c.iloc[tr]["area"].values)
            )
            data.loc[d_c.iloc[te].index, "lag_x_area"] = (
                lag_te * np.log1p(d_c.iloc[te]["area"].values)
            )

            X_train = X_train.fillna(X_train.median()).fillna(0)
            X_test  = X_test.fillna(X_train.median()).fillna(0)

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("enet", ElasticNetCV(
                    alphas=np.logspace(-4, -1, 25),
                    l1_ratio=[0.4, 0.6, 0.8],
                    cv=3, max_iter=5000,
                )),
            ])
            model.fit(X_train, y.iloc[tr])

            r2_train = r2_score(y.iloc[tr], model.predict(X_train))
            train_r2_list.append(r2_train)

            pred_m2    = np.expm1(model.predict(X_test))
            pred_total = pred_m2 * areas[te]

            data.loc[d_c.iloc[te].index, "precio_predicho"] = pred_total
            c_pred.extend(pred_total)
            c_real.extend(d_c["price"].iloc[te])

        if c_real:
            r2_test      = r2_score(c_real, c_pred)
            r2_train_mean = float(np.mean(train_r2_list))
            mape_c        = mean_absolute_percentage_error(c_real, c_pred)

            print(f"    R² prueba (out-of-fold) : {r2_test:.4f}")
            print(f"    R² entrenamiento (media): {r2_train_mean:.4f}")
            print(f"    MAPE                    : {mape_c:.2%}")

            if r2_train_mean - r2_test > 0.10:
                print(f"    ⚠ [FIX-4] Brecha train-test = "
                      f"{r2_train_mean - r2_test:.3f} → posible sobreajuste en clúster {c}")

            global_real.extend(c_real)
            global_pred.extend(c_pred)

    return global_real, global_pred, data

def run_knn_gam(data, n_splits=3):
    print(f"\n{'='*80}")
    print(f"  k-NN COMPARATIVO — GAM  (K-Fold espacial = {n_splits})")
    print(f"{'='*80}")

    q_low, q_high = data["price_m2_raw"].quantile([0.05, 0.95])
    data   = data[
        (data["price_m2_raw"] > q_low) & (data["price_m2_raw"] < q_high)
    ].copy()
    X      = data[static_features].copy()
    y      = np.log1p(data["price_m2_raw"])
    areas  = data["area"].values
    coords = data[["longitud", "latitud"]].values

    blks = KMeans(n_clusters=n_splits, random_state=42, n_init=10).fit_predict(coords)
    real, pred = [], []

    for fold in np.unique(blks):
        tr = np.where(blks != fold)[0]
        te = np.where(blks == fold)[0]
        if len(te) < 5:
            continue

        X_train, X_test = X.iloc[tr].copy(), X.iloc[te].copy()
        lag_tr = compute_spatial_lag(
            coords[tr], data.iloc[tr]["price_m2_raw"].values, coords[tr])
        lag_te = compute_spatial_lag(
            coords[tr], data.iloc[tr]["price_m2_raw"].values, coords[te])
        nvp_tr = compute_local_neighbor_price(
            coords[tr], data.iloc[tr]["price_m2_raw"].values, coords[tr])
        nvp_te = compute_local_neighbor_price(
            coords[tr], data.iloc[tr]["price_m2_raw"].values, coords[te])

        X_train["spatial_lag_price"]    = lag_tr
        X_test["spatial_lag_price"]     = lag_te
        X_train["precio_vecinal_local"] = nvp_tr
        X_test["precio_vecinal_local"]  = nvp_te
        X_train["lag_x_area"] = X_train["spatial_lag_price"] * np.log1p(X_train["area"])
        X_test["lag_x_area"]  = X_test["spatial_lag_price"]  * np.log1p(X_test["area"])

        X_train = X_train.fillna(X_train.median()).fillna(0)
        X_test  = X_test.fillna(X_train.median()).fillna(0)

        knn = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(
                n_neighbors=min(12, len(tr)), weights="distance"
            )),
        ])
        knn.fit(X_train, y.iloc[tr])

        pred_m2 = np.expm1(knn.predict(X_test))
        pred.extend(pred_m2 * areas[te])
        real.extend(data["price"].iloc[te])

    if real:
        r2   = r2_score(real, pred)
        mape = mean_absolute_percentage_error(real, pred)
        print(f"  k-NN Global → R² (out-of-fold): {r2:.4f} | MAPE: {mape:.2%}")
        return r2, mape, real, pred
    return 0, 1, [], []

# ======================================================
# 8. EJECUCIÓN Y VISUALIZACIÓN DE RESULTADOS
# ======================================================
print("\n" + "=" * 80)
print("8/8  EJECUCIÓN DEL MODELO")
print("=" * 80)

tipos_validos = ["Casa", "Departamento", "Edificio", "Local Comercial", "Bodega"]
df_model = df[
    df["tipo_inmueble"].isin(tipos_validos)
    & np.isfinite(df["price_m2_raw"])
    & (df["price_m2_raw"] > 0)
].copy().reset_index(drop=True)

print(f"\n  Registros para modelado: {len(df_model)}")

y_real_en, y_pred_en, df_result = run_model_gam(df_model)
r2_knn, mape_knn, y_real_kn, y_pred_kn = run_knn_gam(df_result)

if y_real_en:
    y_true_arr = np.array(y_real_en)
    y_pred_arr = np.array(y_pred_en)
    r2_global  = r2_score(y_true_arr, y_pred_arr)
    mape_global = mean_absolute_percentage_error(y_true_arr, y_pred_arr)

    print(f"\n  R² Global (out-of-fold): {r2_global:.4f}")
    print(f"  MAPE Global            : {mape_global:.2%}")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.regplot(
        x=y_true_arr, y=y_pred_arr, ax=ax,
        scatter_kws={"alpha": 0.3, "color": "teal", "s": 18},
        line_kws={"color": "red", "linewidth": 1.5},
    )
    ax.set_title(
        f"ElasticNet + Clústeres Espaciales — GAM\n"
        f"R²={r2_global:.4f} (out-of-fold) · MAPE={mape_global:.2%} · "
        f"Índ. Gentrif.={GENTRIFICATION_INDEX_GAM:.4f}",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Precio Real (MXN)")
    ax.set_ylabel("Precio Predicho (MXN)")
    plt.tight_layout()
    mostrar_o_guardar("10_real_vs_predicho.png")

    error_pct = ((y_pred_arr - y_true_arr) / y_true_arr) * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(error_pct, fill=True, color="teal", ax=ax)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(np.median(error_pct), color="red", linestyle="--",
               label=f"Mediana: {np.median(error_pct):.1f}%")
    ax.set_title("Distribución del Error Porcentual — GAM", fontsize=13)
    ax.set_xlabel("Error (%)")
    ax.set_xlim(-150, 150)
    ax.legend()
    plt.tight_layout()
    mostrar_o_guardar("11_distribucion_error.png")

if "precio_predicho" in df_result.columns:
    df_result["error_pct"] = (
        (df_result["precio_predicho"] - df_result["price"]) / df_result["price"]
    ) * 100
    col_agrup = "colonia_oficial" if "colonia_oficial" in df_result.columns else "colonia"
    error_col = (
        df_result.groupby(col_agrup)["error_pct"]
        .mean().abs().sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    error_col.dropna().head(20).plot(kind="bar", ax=ax, color="coral", edgecolor="white")
    ax.set_title("Top 20 Colonias con Mayor Error Medio Absoluto — GAM", fontsize=12)
    ax.set_xlabel("Colonia")
    ax.set_ylabel("Error Absoluto Medio (%)")
    ax.tick_params(axis="x", rotation=50)
    plt.tight_layout()
    mostrar_o_guardar("12_error_por_colonia.png")

def fit_interpretable_elasticnet(data, label):
    print(f"\n{'='*60}")
    print(f"  TOP VARIABLES HEDÓNICAS: {label}")
    print(f"{'='*60}")

    data = data.copy()
    coords_i = data[["longitud", "latitud"]].values
    data["spatial_lag_price"]    = compute_spatial_lag(
        coords_i, data["price_m2_raw"].values, coords_i)
    data["precio_vecinal_local"] = compute_local_neighbor_price(
        coords_i, data["price_m2_raw"].values, coords_i)
    data["lag_x_area"] = data["spatial_lag_price"] * np.log1p(data["area"])

    X = data[features_all].copy().replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0)
    y = np.log1p(data["price_m2_raw"])

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)
    model  = ElasticNetCV(
        alphas=np.logspace(-4, -1, 25),
        l1_ratio=[0.4, 0.6, 0.8],
        cv=3, max_iter=5000,
    )
    model.fit(Xs, y)

    coef_df = pd.DataFrame({"feature": features_all, "coef": model.coef_})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    print(coef_df[["feature", "coef"]].head(15).to_string(index=False))
    return coef_df, model, scaler

coef_df, model_interp, scaler_interp = fit_interpretable_elasticnet(
    df_model, "GAM — Modelo Unificado"
)

top_coef = coef_df.head(15)
fig, ax  = plt.subplots(figsize=(12, 7))
colores  = ["#2ecc71" if c > 0 else "#e74c3c" for c in top_coef["coef"]]
ax.barh(top_coef["feature"], top_coef["coef"], color=colores, edgecolor="white")
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title(
    f"Impacto Hedónico — ElasticNet — GAM\n"
    f"(Verde = positivo · Rojo = negativo · "
    f"Índ. Gentrif.={GENTRIFICATION_INDEX_GAM:.4f})",
    fontsize=13, fontweight="bold",
)
ax.set_xlabel("Coeficiente Estandarizado")
plt.tight_layout()
mostrar_o_guardar("13_coeficientes_hedonicos.png")

print("\n" + "-" * 60)
print("  INTERPRETACIÓN ECONÓMICA — GAM")
print("-" * 60)
for _, row in coef_df.head(10).iterrows():
    pct    = (np.exp(row["coef"]) - 1) * 100
    efecto = "incrementa" if row["coef"] > 0 else "reduce"
    print(
        f"  {row['feature']:<38} → {efecto:10} ≈ {abs(pct):.2f}% "
        f"por +1 desv. estándar"
    )

# ======================================================
# GENTRIF_MAP_SCORE PONDERADO POR COLONIA (QGIS)
# ======================================================
print("\n" + "=" * 80)
print("NEW-2 | GENTRIF_MAP_SCORE PONDERADO — NIVEL COLONIA (QGIS)")
print("=" * 80)

def calcular_gentrif_local(df_puntos):
    coords = df_puntos[["longitud", "latitud"]].values
    nbrs   = NearestNeighbors(n_neighbors=21, algorithm="ball_tree").fit(coords)
    _, indices = nbrs.kneighbors(coords)
    scores = []
    for i in range(len(df_puntos)):
        vecinos_idx   = indices[i][1:]
        precio_propio = df_puntos.iloc[i]["price_m2_raw"]
        precio_vecind = df_puntos.iloc[vecinos_idx]["price_m2_raw"].mean()
        ratio = precio_propio / (precio_vecind + 1e-5)
        ratio = np.clip(ratio, 0, 5)
        scores.append(ratio / (1 + ratio))
    return scores

if "precio_predicho" in df_result.columns:
    df_qgis = df_result.copy()
else:
    df_qgis = df_model.copy()

df_qgis["gentrif_local_presion"] = calcular_gentrif_local(df_qgis)
df_qgis["gentrif_map_score"] = (
    0.75 * df_qgis["gentrification_index"]
    + 0.25 * df_qgis["gentrif_local_presion"]
)

print(f"  gentrif_map_score calculado ({len(df_qgis)} puntos)")
print(f"  Rango: [{df_qgis['gentrif_map_score'].min():.4f}, "
      f"{df_qgis['gentrif_map_score'].max():.4f}]")

if "colonia_oficial" in df_qgis.columns:
    gentrif_por_colonia = (
        df_qgis.groupby("colonia_oficial")
        .agg(
            gentrif_map_score_mean=("gentrif_map_score", "mean"),
            gentrif_local_mean=("gentrif_local_presion", "mean"),
            price_m2_median=("price_m2_raw", "median"),
            marginalidad_mean=("marginalidad_score", "mean"),
            score_15min_mean=("score_15min", "mean"),
            n_registros=("price_m2_raw", "count"),
            lat_centroide=("latitud", "mean"),
            lon_centroide=("longitud", "mean"),
        )
        .reset_index()
    )
    for cat_c in ["cat_mean_vus_200m", "cat_mean_antiguedad_200m",
                  "cat_mean_ratio_construccion_200m"]:
        if cat_c in df_qgis.columns:
            gentrif_por_colonia = gentrif_por_colonia.merge(
                df_qgis.groupby("colonia_oficial")[cat_c]
                .mean().reset_index().rename(columns={cat_c: f"{cat_c}_col"}),
                on="colonia_oficial", how="left",
            )
    gentrif_por_colonia["gentrification_index_alcaldia"] = GENTRIFICATION_INDEX_GAM
    ruta_col = os.path.join(RESULTS_PATH, "gentrif_colonias_gam.csv")
    gentrif_por_colonia.to_csv(ruta_col, index=False, encoding="utf-8-sig")
    print(f"  ✓ gentrif_colonias_gam.csv exportado: {len(gentrif_por_colonia)} colonias")

    top_gentrif = (
        gentrif_por_colonia[gentrif_por_colonia["n_registros"] >= 3]
        .sort_values("gentrif_map_score_mean", ascending=False)
        .head(25)
    )
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle("Índice de Gentrificación por Colonia — GAM",
                 fontsize=14, fontweight="bold")
    colores_g = plt.cm.RdYlGn_r(
        (top_gentrif["gentrif_map_score_mean"] - top_gentrif["gentrif_map_score_mean"].min()) /
        (top_gentrif["gentrif_map_score_mean"].max() - top_gentrif["gentrif_map_score_mean"].min() + 1e-9)
    )
    axes[0].barh(
        top_gentrif["colonia_oficial"],
        top_gentrif["gentrif_map_score_mean"],
        color=colores_g, edgecolor="white",
    )
    axes[0].invert_yaxis()
    axes[0].set_title("Top 25 Colonias por gentrif_map_score (media)")
    axes[0].set_xlabel("gentrif_map_score (0–1)")

    sc = axes[1].scatter(
        gentrif_por_colonia["price_m2_median"],
        gentrif_por_colonia["gentrif_map_score_mean"],
        c=gentrif_por_colonia["marginalidad_mean"],
        cmap="RdYlGn_r", s=60, alpha=0.7,
    )
    plt.colorbar(sc, ax=axes[1], label="Marginalidad Media")
    axes[1].set_xlabel("Precio Mediano de Mercado/m² (MXN)")
    axes[1].set_ylabel("gentrif_map_score Medio")
    axes[1].set_title("Precio de Mercado vs Gentrificación por Colonia\n"
                      "(Color: Marginalidad — rojo = mayor marginalidad)")
    plt.tight_layout()
    mostrar_o_guardar("17_gentrif_ranking_colonia.png")

# ======================================================
# RESUMEN FINAL
# ======================================================
y_true_en_a = np.array(y_real_en) if y_real_en else np.array([])
y_pred_en_a = np.array(y_pred_en) if y_pred_en else np.array([])

if len(y_true_en_a) > 0:
    r2_en   = r2_score(y_true_en_a, y_pred_en_a)
    mape_en = mean_absolute_percentage_error(y_true_en_a, y_pred_en_a)
    rmse_en = np.sqrt(mean_squared_error(
        np.log1p(y_true_en_a), np.log1p(y_pred_en_a)
    ))
else:
    r2_en = mape_en = rmse_en = float("nan")

if y_real_kn:
    y_true_kn_a = np.array(y_real_kn)
    y_pred_kn_a = np.array(y_pred_kn)
    rmse_kn     = np.sqrt(mean_squared_error(
        np.log1p(y_true_kn_a), np.log1p(y_pred_kn_a)
    ))
else:
    rmse_kn = float("nan")

print("\n" + "#" * 80)
print("  RESUMEN COMPARATIVO FINAL — GUSTAVO A. MADERO")
print(f"  Índ. Gentrificación GAM  : {GENTRIFICATION_INDEX_GAM:.4f} (PCA INEGI 2010-2020)")
print(f"  [FIX-4] Métricas SOLO en datos de prueba out-of-fold")
print("#" * 80)
print(f"\n  {'MÉTRICA':<22} | {'ENET + Clústeres':<18} | {'k-NN Vecindad'}")
print("  " + "-" * 62)
print(f"  {'R² (out-of-fold)':<22} | {r2_en:<18.4f} | {r2_knn:.4f}")
print(f"  {'MAPE (out-of-fold)':<22} | {mape_en*100:<17.2f}% | {mape_knn*100:.2f}%")
print(f"  {'RMSE Logarítmico':<22} | {rmse_en:<18.4f} | {rmse_kn:.4f}")
print("#" * 80)

# ======================================================
# CSV FINAL PARA QGIS (con puntos ya validados espacialmente)
# ======================================================
print("\n" + "=" * 80)
print("NEW-7 | EXPORTACIÓN FINAL PARA ANÁLISIS GEOGRÁFICO (QGIS)")
print("=" * 80)

cols_qgis_base = [
    "latitud", "longitud",
    "colonia_oficial",
    "tipo_inmueble",
    "price", "area", "rooms", "bathrooms",
    "price_m2_raw",
    "gentrification_index",
    "gentrif_local_presion",
    "gentrif_map_score",
    "marginalidad_score",
    "score_15min",
    "uso_mixto",
    "dist_metro_m", "dist_cable_m",
    "density_metro", "density_cable",
    "acceso_salud_15m", "acceso_educacion_15m",
    "densidad_parques_15m",
    "comercio_density",
    "dist_centro_historico_m",
    "dist_nearest_subcenter_m",
    "cluster",
]

for c in cat_cols_feat:
    if c in df_qgis.columns:
        cols_qgis_base.append(c)

if "precio_predicho" in df_qgis.columns:
    cols_qgis_base.append("precio_predicho")
if "error_pct" in df_qgis.columns:
    cols_qgis_base.append("error_pct")

cols_export = [c for c in cols_qgis_base if c in df_qgis.columns]

ruta_qgis = os.path.join(RESULTS_PATH, "resultados_gam_v2.csv")
df_qgis[cols_export].to_csv(ruta_qgis, index=False, encoding="utf-8-sig")

print(f"  ✓ resultados_gam_v2.csv exportado: {len(df_qgis)} puntos, {len(cols_export)} cols")
print(f"  ✓ Columnas QGIS incluidas:")
for c in cols_export:
    print(f"      · {c}")

print(f"\n  ✓ Figuras guardadas en  : {FIGURES_PATH}")
print(f"  ✓ Resultados en         : {RESULTS_PATH}")
print("\n  ✓ Pipeline GAM v5 completado (validación espacial estricta).")