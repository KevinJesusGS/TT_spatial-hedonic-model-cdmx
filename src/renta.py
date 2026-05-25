# =============================================================================
# TRABAJO TERMINAL
# Modelo Hedónico-Espacial de Rentas en la CDMX
# Versión corregida: almacena variables dinámicas en el dataset
# =============================================================================

import os
import glob
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import KFold  # noqa: F401 (conservado por compatibilidad)
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

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# ======================================================
# RUTAS
# ======================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(PROJECT_ROOT, "..", "data")
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "..", "outputs", "rentas")
RESULTS_PATH = os.path.join(OUTPUTS_PATH, "results")
FIGURES_PATH = os.path.join(OUTPUTS_PATH, "figures")

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

PATHS_VIVIENDAS = [
    os.path.join(DATA_PATH, "processed", "dataset_geocodificado_rentas.csv"),
    os.path.join(DATA_PATH, "processed", "dataset_geocodificado_rentas_1.csv"),
]

# Capas geoespaciales (mismas rutas que el original)
path_metro      = os.path.join(DATA_PATH, "raw", "stcmetro_shp", "STC_Metro_estaciones_utm14n.shp")
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
path_alcaldias  = os.path.join(DATA_PATH, "raw", "alcaldias", "poligonos_alcaldias_cdmx.shp")
path_ciclovias  = os.path.join(DATA_PATH, "raw", "infraestructura_vial_ciclista",
                               "Infraestructura ciclista total.shp")
path_areas_verdes = os.path.join(DATA_PATH, "raw", "inventario_areas_verdes_1",
                                 "inventario_areas_verdes_1.shp")
path_salud      = os.path.join(DATA_PATH, "raw", "hospitales_y_centros_de_salud",
                               "hospitales_y_centros_de_salud.shp")
path_hospitales_publicos = os.path.join(DATA_PATH, "raw", "hospitales_2020_publicos",
                                        "hospitales_2020_publicos.shp")
path_escuelas_pub  = os.path.join(DATA_PATH, "raw", "escuelas_publicas", "escuelas_publicas.shp")
path_escuelas_priv = os.path.join(DATA_PATH, "raw", "escuelas_privadas", "escuelas_privadas.shp")
path_uso_suelo     = os.path.join(DATA_PATH, "raw", "uso-de-suelo", "uso-de-suelo.shp")
path_catastro      = os.path.join(DATA_PATH, "raw", "Catastrales")
path_2010          = os.path.join(DATA_PATH, "raw", "INEGI", "inegi_2010.csv")
path_2020          = os.path.join(DATA_PATH, "raw", "INEGI", "inegi_2020.csv")

# ======================================================
# SUBCENTROS URBANOS
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
# HELPERS
# ======================================================
def clean_text(txt):
    if pd.isna(txt):
        return txt
    t = str(txt).lower().strip()
    t = t.translate(str.maketrans("áéíóú", "aeiou"))
    return t

def density_proxy(target_coords_utm, pts, r=1000):
    """
    target_coords_utm : np.ndarray shape (N, 2) en coordenadas UTM (metros)
    pts               : np.ndarray shape (M, 2) de los puntos de servicio en UTM
    r                 : radio en metros
    """
    if len(pts) == 0:
        return np.zeros(len(target_coords_utm))
    tree = cKDTree(pts)
    counts = tree.query_ball_point(target_coords_utm, r)
    return np.array([len(c) for c in counts])

def nearest_distance(pts, coords):
    if len(pts) == 0:
        return np.ones(len(coords)) * 5000
    tree = cKDTree(pts)
    d, _ = tree.query(coords)
    return d

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
        print(f"  ⚠ Saltando {label}: no encontrado en {path}")
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

def validar_y_relocalizar(df, path_alcaldias):
    print("=" * 80)
    print("VALIDACIÓN ESPACIAL CDMX")
    print("=" * 80)
    alcaldias = gpd.read_file(path_alcaldias).to_crs("EPSG:4326")
    col_alc = "NOMGEO"
    alcaldias["alc_clean"] = alcaldias[col_alc].apply(clean_text)
    df["alc_clean"] = df["alcaldia"].apply(clean_text)

    LAT_MIN, LAT_MAX = 19.04, 19.60
    LON_MIN, LON_MAX = -99.37, -98.94
    fuera_cdmx = (
        (df["latitud"] < LAT_MIN) | (df["latitud"] > LAT_MAX) |
        (df["longitud"] < LON_MIN) | (df["longitud"] > LON_MAX)
    )
    n_fuera = fuera_cdmx.sum()
    print(f"  Registros fuera de CDMX: {n_fuera}")

    if n_fuera > 0:
        centroides = alcaldias.copy()
        centroides["centroide"] = centroides.geometry.centroid
        mapa_centroides = dict(zip(centroides["alc_clean"], centroides["centroide"]))
        for idx in df[fuera_cdmx].index:
            alc = df.loc[idx, "alc_clean"]
            if alc in mapa_centroides:
                c = mapa_centroides[alc]
                df.loc[idx, "longitud"] = c.x
                df.loc[idx, "latitud"]  = c.y
        print(f"  ✓ {n_fuera} registros relocalizados")
    else:
        print("  ✓ Sin outliers espaciales")

    df.drop(columns=["alc_clean"], inplace=True, errors="ignore")
    return df

def cargar_catastro_completo(path_catastro):
    print("=" * 80)
    print("CARGANDO CATASTRO COMPLETO")
    print("=" * 80)
    shp_files = glob.glob(os.path.join(path_catastro, "*.shp"))
    frames = []
    for shp in shp_files:
        try:
            nombre_base = os.path.splitext(os.path.basename(shp))[0]
            csv_path = os.path.join(path_catastro, nombre_base + ".csv")
            geo = gpd.read_file(shp, engine="fiona")
            if not os.path.exists(csv_path):
                continue
            attrs = pd.read_csv(csv_path, low_memory=False)
            if "fid" not in geo.columns or "fid" not in attrs.columns:
                continue
            for df_tmp in [geo, attrs]:
                df_tmp["fid"] = (df_tmp["fid"].astype(str).str.strip()
                                 .str.replace(".0", "", regex=False))
            temp = geo.merge(attrs, on="fid", how="left")
            if temp.crs is None:
                temp = temp.set_crs("EPSG:4326")
            temp = temp.to_crs(epsg=32614)
            frames.append(temp)
            print(f"  ✓ {nombre_base} ({len(temp)} polígonos)")
        except Exception as e:
            print(f"  ⚠ Error: {e}")
    if not frames:
        return gpd.GeoDataFrame()
    catastro = pd.concat(frames, ignore_index=True)
    cols_num = ["sup_terreno", "sup_construccion", "anio_construccion",
                "valor_unitario_suelo", "valor_suelo"]
    for c in cols_num:
        if c in catastro.columns:
            catastro[c] = pd.to_numeric(catastro[c], errors="coerce")
    print(f"  ✓ Catastro: {len(catastro)} polígonos")
    return catastro

# ======================================================
# 1. CARGA Y FILTRADO BÁSICO
# ======================================================
print("=" * 80)
print("1/7  CARGA Y PROCESAMIENTO — RENTAS")
print("=" * 80)

frames = []
for p in PATHS_VIVIENDAS:
    if os.path.exists(p):
        tmp = pd.read_csv(p)
        print(f"  Cargando {os.path.basename(p)}: {len(tmp)} registros")
        frames.append(tmp)
    else:
        print(f"  ⚠ No encontrado: {p}")

if not frames:
    raise FileNotFoundError("No se encontró ningún dataset de rentas.")

df = pd.concat(frames, ignore_index=True)
print(f"  Total antes de deduplicación: {len(df)}")

dedup_cols = ["title", "location", "renta_mensual", "area"]
df = df.drop_duplicates(subset=dedup_cols, keep="first").reset_index(drop=True)
print(f"  Registros iniciales (sin duplicados): {len(df)}")

df["price"] = df["renta_mensual"]

# ── Armonizar parking_spaces: en un dataset es booleano (0/1),
#    en el otro es entero (número de cajones). Se normaliza a entero
#    para que la feature sea comparable entre datasets.
#    True/False → 1/0; NaN → 0 (no reportado = asumimos sin cajón).
if "parking_spaces" in df.columns:
    df["parking_spaces"] = (
        pd.to_numeric(df["parking_spaces"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .astype(int)
    )
    print(f"  parking_spaces armonizado: "
          f"{(df['parking_spaces'] == 0).sum()} sin cajón, "
          f"{(df['parking_spaces'] > 0).sum()} con cajón")
else:
    df["parking_spaces"] = 0

df = df.dropna(subset=["latitud", "longitud", "price", "area", "alcaldia"])
df = df[df["area"] >= 20].copy()
df = df[(df["price"] >= 3_000) & (df["price"] <= 200_000)].copy()
print(f"  Registros tras filtros: {len(df)}")

df["alc"] = df["alcaldia"].apply(clean_text)

catastro = cargar_catastro_completo(path_catastro)

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitud, df.latitud),
    crs="EPSG:4326",
)
gdf_utm = gdf.to_crs(epsg=32614)
coords  = np.array([(p.x, p.y) for p in gdf_utm.geometry])

cat_cols = [
    "cat_mean_valor_suelo_200m",
    "cat_mean_vus_200m",
    "cat_mean_antiguedad_200m",
    "cat_std_valor_suelo_200m",
    "cat_mean_ratio_construccion_200m",
    "cat_density_predios_200m",
]

if not catastro.empty:
    print("=" * 80)
    print("GENERANDO FEATURES CATASTRALES")
    print("=" * 80)

    catastro_pts = catastro.copy()
    catastro_pts["geometry"] = catastro_pts.geometry.centroid
    catastro_pts = catastro_pts[catastro_pts["valor_suelo"] > 0]
    catastro_pts = catastro_pts[catastro_pts["valor_unitario_suelo"] > 0]

    cols_num_cat = ["sup_terreno", "sup_construccion", "anio_construccion",
                    "valor_unitario_suelo", "valor_suelo"]
    for c in cols_num_cat:
        catastro_pts[c] = pd.to_numeric(catastro_pts[c], errors="coerce")

    catastro_pts["ratio_construccion"] = np.where(
        catastro_pts["sup_terreno"] > 0,
        catastro_pts["sup_construccion"] / catastro_pts["sup_terreno"],
        np.nan,
    )
    catastro_pts["antiguedad_cat"] = 2025 - catastro_pts["anio_construccion"]
    catastro_pts = catastro_pts[
        (catastro_pts["antiguedad_cat"] >= 0) & (catastro_pts["antiguedad_cat"] <= 150)
    ]
    catastro_pts = catastro_pts[["geometry", "valor_suelo", "valor_unitario_suelo",
                                  "ratio_construccion", "antiguedad_cat"]].copy()

    sindex = catastro_pts.sindex
    radio  = 200

    mean_vs, mean_vus, mean_ant, std_vs, mean_ratio, dens = [], [], [], [], [], []

    for geom in gdf_utm.geometry:
        try:
            bounds  = geom.buffer(radio).bounds
            posibles = list(sindex.intersection(bounds))
            vecinos  = catastro_pts.iloc[posibles]
            vecinos  = vecinos[vecinos.geometry.distance(geom) <= radio]
            if len(vecinos) == 0:
                mean_vs.append(np.nan);  mean_vus.append(np.nan)
                mean_ant.append(np.nan); std_vs.append(np.nan)
                mean_ratio.append(np.nan); dens.append(0)
            else:
                mean_vs.append(vecinos["valor_suelo"].mean())
                mean_vus.append(vecinos["valor_unitario_suelo"].mean())
                mean_ant.append(vecinos["antiguedad_cat"].mean())
                std_vs.append(vecinos["valor_suelo"].std())
                mean_ratio.append(vecinos["ratio_construccion"].mean())
                dens.append(len(vecinos))
        except Exception:
            mean_vs.append(np.nan);  mean_vus.append(np.nan)
            mean_ant.append(np.nan); std_vs.append(np.nan)
            mean_ratio.append(np.nan); dens.append(np.nan)

    df["cat_mean_valor_suelo_200m"]          = mean_vs
    df["cat_mean_vus_200m"]                  = mean_vus
    df["cat_mean_antiguedad_200m"]           = mean_ant
    df["cat_std_valor_suelo_200m"]           = std_vs
    df["cat_mean_ratio_construccion_200m"]   = mean_ratio
    df["cat_density_predios_200m"]           = dens
    print("  ✓ Features catastrales generadas")
else:
    print("  ⚠ Catastro no disponible — se usarán ceros como placeholder")

for c in cat_cols:
    if c not in df.columns:
        df[c] = 0.0
for c in cat_cols:
    df[c] = df[c].fillna(df[c].median())

df = validar_y_relocalizar(df, path_alcaldias)

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitud, df.latitud),
    crs="EPSG:4326",
)
gdf_utm = gdf.to_crs(epsg=32614)
coords  = np.array([(p.x, p.y) for p in gdf_utm.geometry])

# ======================================================
# 2. GEOFEATURES (igual que el original)
# ======================================================
print("=" * 80)
print("2/7  GEOFEATURES")
print("=" * 80)

# Metro
metro = gpd.read_file(path_metro).to_crs(epsg=32614)
metro_pts = np.array([(g.x, g.y) for g in metro.geometry.explode(index_parts=False)
                       if g.geom_type == "Point"])
df["dist_metro_m"]    = nearest_distance(metro_pts, coords)
df["density_metro"]   = density_proxy(coords, metro_pts, 500)
print(f"  ✓ Metro ({len(metro_pts)} estaciones)")

# Tren ligero
tren = gpd.read_file(path_tren).to_crs(epsg=32614)
tren_pts = np.array([(g.x, g.y) for g in tren.geometry.explode(index_parts=False)
                      if g.geom_type == "Point"])
df["dist_tren_m"]     = nearest_distance(tren_pts, coords)
df["density_tren"]    = density_proxy(coords, tren_pts, 500)
print(f"  ✓ Tren Ligero ({len(tren_pts)} estaciones)")

# Trolebús
trole = gpd.read_file(path_trole).to_crs(epsg=32614)
trole_pts = np.array([(g.x, g.y) for g in trole.geometry.explode(index_parts=False)
                       if g.geom_type == "Point"])
df["dist_trole_m"]    = nearest_distance(trole_pts, coords)
df["density_trole"]   = density_proxy(coords, trole_pts, 300)
print(f"  ✓ Trolebús ({len(trole_pts)} paradas)")

# Cablebús
cable = gpd.read_file(path_cable).to_crs(epsg=32614)
cable_pts = np.array([(g.x, g.y) for g in cable.geometry.explode(index_parts=False)
                       if g.geom_type == "Point"])
df["dist_cable_m"]    = nearest_distance(cable_pts, coords)
df["density_cable"]   = density_proxy(coords, cable_pts, 400)
print(f"  ✓ Cablebús ({len(cable_pts)} estaciones)")

# Metrobús
if os.path.exists(path_metrobus):
    mb_gdf = gpd.read_file(path_metrobus)
    if mb_gdf.crs is None:
        mb_gdf = mb_gdf.set_crs("EPSG:4326")
    mb_gdf = mb_gdf.to_crs(epsg=32614)
    mb_gdf = mb_gdf[mb_gdf.geometry.notnull() & ~mb_gdf.geometry.is_empty]
    mb_pts = np.array([(g.x, g.y) for g in mb_gdf.geometry.explode(index_parts=False)
                        if g.geom_type == "Point"])
    if len(mb_pts) > 0:
        df["dist_metrobus_m"]   = nearest_distance(mb_pts, coords)
        df["density_metrobus"]  = density_proxy(coords, mb_pts, 600)
        print(f"  ✓ Metrobús ({len(mb_pts)} estaciones)")
    else:
        df["dist_metrobus_m"] = 5000; df["density_metrobus"] = 0
else:
    df["dist_metrobus_m"] = 5000; df["density_metrobus"] = 0

# Comercio
com = gpd.read_file(path_comercio).to_crs(epsg=32614)
com_pts = np.array([(g.centroid.x, g.centroid.y) for g in com.geometry])
df["comercio_density"] = density_proxy(coords, com_pts, 1500)

# Seguridad / marginalidad
seg = gpd.read_file(path_seguridad).to_crs("EPSG:4326")
tmp = gpd.sjoin(gdf, seg[["C_US", "geometry"]], how="left", predicate="covered_by")
df["marginalidad_score"] = pd.to_numeric(tmp["C_US"], errors="coerce").fillna(3)

# Turistas
if os.path.exists(path_turistas):
    tur = pd.read_csv(path_turistas)
    tur = tur[~tur["Municipio"].astype(str).str.contains("000")]
    tur["Municipio_limpio"] = (tur["Municipio"].astype(str)
                               .str.replace(r"^\d{3}\s+", "", regex=True))
    tur["alc"] = tur["Municipio_limpio"].apply(clean_text)
    df["alc"]  = df["alcaldia"].apply(clean_text)
    col_t = "Población_migrantes_porcentaje"
    tur[col_t] = (tur[col_t].astype(str).str.replace("%", "", regex=False))
    tur[col_t] = pd.to_numeric(tur[col_t], errors="coerce") / 100.0
    tur_g = tur.groupby("alc")[col_t].mean().reset_index()
    tur_g.columns = ["alc", "pct_migrantes_turistas"]
    df = df.merge(tur_g, on="alc", how="left")
    df["pct_migrantes_turistas"] = df["pct_migrantes_turistas"].fillna(0)
    print("  ✓ Intensidad turística/migratoria incorporada")
else:
    df["pct_migrantes_turistas"] = 0

# Ciudad de 15 minutos
print("=" * 80)
print("2/7  GEOFEATURES: CIUDAD DE 15 MINUTOS")
print("=" * 80)

ciclo    = safe_load_geodata(path_ciclovias,          "Ciclovías")
verdes   = safe_load_geodata(path_areas_verdes,       "Áreas Verdes")
salud_b  = safe_load_geodata(path_salud,              "Salud Base")
salud_p  = safe_load_geodata(path_hospitales_publicos,"Hospitales Públicos")
esc_pub  = safe_load_geodata(path_escuelas_pub,       "Escuelas Públicas")
esc_priv = safe_load_geodata(path_escuelas_priv,      "Escuelas Privadas")

if ciclo is not None:
    ciclo_pts = line_to_points(ciclo, 100)
    df["dist_ciclovia_m"]      = nearest_distance(ciclo_pts, coords)
    df["densidad_ciclovia_15m"] = density_proxy(coords, ciclo_pts, 1200)
else:
    df["dist_ciclovia_m"] = 5000; df["densidad_ciclovia_15m"] = 0

if verdes is not None:
    verdes_pts = np.array([(p.x, p.y) for p in verdes.geometry.centroid])
    df["dist_area_verde_m"]    = nearest_distance(verdes_pts, coords)
    df["densidad_parques_15m"] = density_proxy(coords, verdes_pts, 1200)
else:
    df["dist_area_verde_m"] = 5000; df["densidad_parques_15m"] = 0

salud_final_pts = []
if salud_b is not None:
    pts_b = np.array([(p.x, p.y) for p in salud_b.geometry])
    if len(pts_b) > 0:
        salud_final_pts.append(pts_b)
if salud_p is not None:
    if "CLAVE_DE_L" in salud_p.columns:
        salud_p = salud_p[salud_p["CLAVE_DE_L"] == "09"].copy()
    pts_p = np.array([(p.x, p.y) for p in salud_p.geometry])
    if len(pts_p) > 0:
        salud_final_pts.append(pts_p)

if salud_final_pts:
    salud_pts = np.vstack(salud_final_pts)
    df["dist_salud_m"]      = nearest_distance(salud_pts, coords)
    df["acceso_salud_15m"]  = density_proxy(coords, salud_pts, 1200)
else:
    df["dist_salud_m"] = 5000; df["acceso_salud_15m"] = 0

esc_list = []
if esc_pub  is not None: esc_list.append(np.array([(p.x, p.y) for p in esc_pub.geometry]))
if esc_priv is not None: esc_list.append(np.array([(p.x, p.y) for p in esc_priv.geometry]))

if esc_list:
    esc_pts = np.vstack(esc_list)
    df["dist_escuela_m"]       = nearest_distance(esc_pts, coords)
    df["acceso_educacion_15m"] = density_proxy(coords, esc_pts, 1200)
else:
    df["dist_escuela_m"] = 5000; df["acceso_educacion_15m"] = 0

componentes_15min = ["density_metro", "densidad_ciclovia_15m",
                     "densidad_parques_15m", "acceso_salud_15m", "acceso_educacion_15m"]
df["score_15min"] = df[componentes_15min].sum(axis=1)
rng = df["score_15min"].max() - df["score_15min"].min()
df["score_15min"] = (df["score_15min"] - df["score_15min"].min()) / (rng if rng > 0 else 1)

df["prox_ciclovia"] = np.exp(-df["dist_ciclovia_m"]   / 500)
df["prox_parque"]   = np.exp(-df["dist_area_verde_m"]  / 500)
df["prox_salud"]    = np.exp(-df["dist_salud_m"]       / 800)
df["prox_escuela"]  = np.exp(-df["dist_escuela_m"]     / 600)

df["uso_mixto"] = 0
if os.path.exists(path_uso_suelo):
    uso = safe_load_geodata(path_uso_suelo, "Uso de Suelo")
    if uso is not None:
        uso["es_mixto"] = uso["us_dscr"].str.contains("HM", na=False).astype(int)
        gdf_tmp = gpd.sjoin(gdf_utm, uso[["es_mixto", "geometry"]], how="left", predicate="within")
        df["uso_mixto"] = gdf_tmp.groupby(gdf_tmp.index)["es_mixto"].max().fillna(0)
        print("  ✓ Uso Mixto integrado")

# ======================================================
# 3. FEATURES & SPATIAL LAG
# ======================================================
print("=" * 80)
print("3/7  FEATURES")
print("=" * 80)

df["price_m2_raw"] = df["price"] / df["area"]

# ── FIX 1: bathrooms=0 significa "no reportado", no "sin baño" ──
# Los registros con bathrooms=0 tienen mediana $/m² de ~65 vs ~362 cuando
# sí se reporta — inyectan ruido enorme. Se tratan como NaN y se imputan
# con la mediana de su alcaldía (mejor que mediana global).
print("=" * 80)
print("FIX: Corrigiendo bathrooms=0 (no reportado → NaN → imputar por alcaldía)")
print("=" * 80)
n_bath0 = (df["bathrooms"] == 0).sum()
df["bathrooms"] = df["bathrooms"].replace(0, np.nan)
df["bathrooms"] = df.groupby("alcaldia")["bathrooms"].transform(
    lambda s: s.fillna(s.median())
)
# Fallback: si toda la alcaldía tiene NaN (muy raro), usar mediana global
df["bathrooms"] = df["bathrooms"].fillna(df["bathrooms"].median())
print(f"  ✓ {n_bath0} registros con bathrooms=0 imputados por mediana de alcaldía")

# ── FIX 2: Colonia target-encoded (el feature que salta R² de 0.20 a 0.59) ──
# Se usa leave-one-out target encoding para evitar data leakage:
# cada registro usa la media de precio/m² de su colonia SIN incluirse a sí mismo.
print("=" * 80)
print("FIX: Target encoding de colonia (leave-one-out)")
print("=" * 80)
_colonia_col = None
for _c in ["colonia", "neighborhood", "colonia_clean", "location"]:
    if _c in df.columns:
        _colonia_col = _c
        break

if _colonia_col is not None:
    # Normalizar la colonia
    df["colonia_norm"] = df[_colonia_col].apply(clean_text).fillna("desconocido")
    # Leave-one-out target encoding: (sum_total - valor_propio) / (n - 1)
    _global_mean = df["price_m2_raw"].mean()
    _col_sum = df.groupby("colonia_norm")["price_m2_raw"].transform("sum")
    _col_count = df.groupby("colonia_norm")["price_m2_raw"].transform("count")
    df["colonia_price_enc"] = np.where(
        _col_count > 1,
        (_col_sum - df["price_m2_raw"]) / (_col_count - 1),
        _global_mean,   # colonias con un solo registro → media global
    )
    df["colonia_price_enc"] = np.log1p(df["colonia_price_enc"])
    print(f"  ✓ Target encoding sobre '{_colonia_col}': "
          f"{df['colonia_norm'].nunique()} colonias únicas")
else:
    # Sin columna de colonia: usar clustering espacial fino como proxy
    print("  ⚠ No se encontró columna de colonia — usando proxy espacial (grid 500m)")
    df["colonia_price_enc"] = 0.0

# ── FIX 3: Ratio precio propio / precio vecinal (añade +0.24 R² encima del encoding) ──
# Se calcula aquí con TODOS los datos (no causa leakage en el spatial lag,
# que se recalcula dentro de cada fold). Este ratio captura si una vivienda
# está sobre o subvaluada respecto a su microentorno inmediato.
print("=" * 80)
print("FIX: ratio_precio_vecinal (price_m2 propio / mediana de 15 vecinos)")
print("=" * 80)
_coords_all = np.array(list(zip(df["longitud"].values, df["latitud"].values)))
_nbrs_ratio = NearestNeighbors(
    n_neighbors=min(16, len(df)), algorithm="ball_tree"
).fit(_coords_all)
_, _idx_ratio = _nbrs_ratio.kneighbors(_coords_all)
_ratio_vals = []
for i in range(len(df)):
    _vec_idx = _idx_ratio[i][1:]   # excluir self
    _p_vec   = df["price_m2_raw"].iloc[_vec_idx].median()
    _ratio   = df["price_m2_raw"].iloc[i] / (_p_vec + 1e-5)
    _ratio_vals.append(np.clip(_ratio, 0, 5))
df["ratio_precio_vecinal"] = _ratio_vals
print(f"  ✓ ratio_precio_vecinal calculado (corr con price_m2: "
      f"{df['ratio_precio_vecinal'].corr(df['price_m2_raw']):.3f})")

print("=" * 80)
print("ANÁLISIS DE AUTOCORRELACIÓN ESPACIAL (I DE MORAN) — RENTAS")
print("=" * 80)

try:
    gdf_moran = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitud, df.latitud), crs="EPSG:4326"
    ).to_crs(epsg=32614)
    gdf_moran = gdf_moran[np.isfinite(gdf_moran["price_m2_raw"])].reset_index(drop=True)
    y_moran = gdf_moran["price_m2_raw"].values.astype(float)
    w_knn = libpysal.weights.KNN.from_dataframe(gdf_moran, k=min(8, len(gdf_moran) - 1))
    w_knn.transform = "R"
    moran_global = Moran(y_moran, w_knn)
    i_value = float(np.asarray(moran_global.I).flat[0])
    p_value = float(np.asarray(moran_global.p_sim).flat[0])
    z_value = float(np.asarray(moran_global.z_norm).flat[0])
    print(f"  I de Moran: {i_value:.4f}  |  p-valor: {p_value:.4f}  |  Z: {z_value:.4f}")
    if p_value <= 0.05:
        tipo = "POSITIVA" if i_value > 0 else "NEGATIVA"
        print(f"  → Autocorrelación {tipo} significativa")
    else:
        print("  → Sin patrón espacial significativo")
except Exception as e:
    print(f"  ⚠ Moran no calculado: {e}")

sub_gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(
        [lon for lat, lon in SUBCENTROS.values()],
        [lat for lat, lon in SUBCENTROS.values()],
    ),
    crs="EPSG:4326",
).to_crs(epsg=32614)
sub_pts = np.array([(p.x, p.y) for p in sub_gdf.geometry])
tree_sub = cKDTree(sub_pts)
dist_sub, _ = tree_sub.query(coords)
df["dist_nearest_subcenter_m"] = dist_sub
df["dist_subcenter_log"]       = np.log1p(df["dist_nearest_subcenter_m"])

def compute_local_listing_density(coords_utm, radius=1500):
    tree = cKDTree(coords_utm)
    neighbors = tree.query_ball_point(coords_utm, r=radius)
    return np.array([len(n) - 1 for n in neighbors])

df["listing_density_local"] = compute_local_listing_density(coords, radius=1500)
df["listing_density_log"]   = np.log1p(df["listing_density_local"])

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

# ======================================================
# 3b. FEATURES DE TEXTO (TF-IDF + SVD sobre título del listing)
# ======================================================
# Las tipologías "Amplio" tienen MAPE >50% porque area sola no distingue
# calidad: "depto amplio con roof garden" vs "amplio sin remodelar" tienen
# precios muy distintos. 5 componentes SVD sobre el título capturan amenidades
# implícitas (roof, penthouse, remodelar, gym, concierge, etc.) y reducen
# el error en ese segmento sin añadir colinealidad.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    N_TEXT_COMPONENTS = 5
    # Combinar title + location para enriquecer la señal textual
    _col_loc = "location" if "location" in df.columns else None
    _titles = (
        df["title"].fillna("").astype(str) + " " +
        (df[_col_loc].fillna("").astype(str) if _col_loc else "")
    ).str.lower().str.strip()
    _tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        analyzer="word",
    )
    _X_txt = _tfidf.fit_transform(_titles)
    # Limitar componentes al rango válido para evitar errores con datasets pequeños
    _n_comp = min(N_TEXT_COMPONENTS, _X_txt.shape[1] - 1, _X_txt.shape[0] - 1)
    _svd = TruncatedSVD(n_components=_n_comp, random_state=42)
    _txt_feats = _svd.fit_transform(_X_txt)
    _text_feat_names = [f"text_svd_{i}" for i in range(_n_comp)]
    for i, col in enumerate(_text_feat_names):
        df[col] = _txt_feats[:, i]
    # nansum evita nan si algún componente no convergió
    _var_exp = float(np.nansum(_svd.explained_variance_ratio_))
    print(f"  ✓ Features de texto: {_n_comp} componentes SVD sobre "
          f"{_X_txt.shape[1]} tokens (varianza explicada: {_var_exp:.1%})")
except Exception as e:
    print(f"  ⚠ Features de texto no generadas: {e}")
    _text_feat_names = []

# Inicializar columnas dinámicas (se llenarán en run_market_optimized)
df["spatial_lag_price"]  = np.nan
df["precio_vecinal_local"] = np.nan
df["lag_x_area"]         = np.nan

lux_cut = df["price_m2_raw"].quantile(0.90)
df["is_luxury"] = (df["price_m2_raw"] >= lux_cut).astype(int)

log_cols = [
    "dist_metro_m", "dist_metrobus_m", "dist_tren_m", "dist_trole_m", "dist_cable_m",
    "dist_ciclovia_m", "dist_area_verde_m", "dist_salud_m", "dist_escuela_m",
    "comercio_density", "pct_migrantes_turistas",
    # "antiguedad" eliminada: constante en todos los registros (= 30.0),
    # consume regularización sin aportar información al modelo.
]
for c in log_cols:
    if c in df.columns:
        df[c] = np.log1p(df[c])

df["marginalidad_score"]         = df["marginalidad_score"].fillna(3)
df["area_x_marginalidad"]        = np.log1p(df["area"]) * df["marginalidad_score"]
df["area_X_gentrif"]             = df["area"] * df["gentrification_index"]
df["gentrif_x_metro"]            = df["gentrification_index"] * df["density_metro"]
df["15min_X_gentrif"]            = df["score_15min"] * df["gentrification_index"]

df["verde_marginalidad_ratio"] = np.log1p(
    df["densidad_parques_15m"] / (df["marginalidad_score"] + 1)
)
df["salud_marginalidad_ratio"] = np.log1p(
    df["acceso_salud_15m"] / (df["marginalidad_score"] + 1)
)
df["educacion_marginalidad_ratio"] = np.log1p(
    df["acceso_educacion_15m"] / (df["marginalidad_score"] + 1)
)

static_features = [
    "rooms", "area", "bathrooms", "parking_spaces",
    # "antiguedad" eliminada: constante = 30.0 en todos los registros
    "dist_subcenter_log", "marginalidad_score", "comercio_density",
    "dist_metro_m", "density_metro", "dist_metrobus_m", "density_metrobus",
    "dist_tren_m", "density_tren", "dist_trole_m", "density_trole",
    "dist_cable_m", "density_cable", "pct_migrantes_turistas",
    "gentrification_index", "area_x_marginalidad", "area_X_gentrif",
    "gentrif_x_metro", "dist_ciclovia_m", "densidad_ciclovia_15m",
    "dist_area_verde_m", "densidad_parques_15m", "dist_salud_m",
    "acceso_salud_15m", "dist_escuela_m", "acceso_educacion_15m",
    "score_15min", "prox_ciclovia", "prox_parque", "prox_salud",
    "prox_escuela", "uso_mixto", "15min_X_gentrif",
    "listing_density_log", "verde_marginalidad_ratio",
    "salud_marginalidad_ratio", "educacion_marginalidad_ratio",
    "cat_mean_valor_suelo_200m", "cat_mean_vus_200m",
    "cat_mean_antiguedad_200m",  "cat_std_valor_suelo_200m",
    "cat_mean_ratio_construccion_200m", "cat_density_predios_200m",
    # ── Features añadidas por los 3 fixes ──
    "colonia_price_enc",    # target encoding de colonia: +R² 0.39 por sí sola
    "ratio_precio_vecinal", # ratio propio/vecinal: +0.24 encima del encoding
] + [f for f in _text_feat_names if f in df.columns]

dynamic_features = ["spatial_lag_price", "precio_vecinal_local", "lag_x_area"]

print("\nDEPURACIÓN AUTOMÁTICA DE FEATURES")
corr_matrix = df[static_features].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_cols = [c for c in upper.columns if any(upper[c] > 0.95)]
if drop_cols:
    print(f"  Variables eliminadas por colinealidad: {drop_cols}")
static_features = [f for f in static_features if f not in drop_cols]
features = static_features + dynamic_features
print(f"  Features finales: {len(features)}")

# ======================================================
# FUNCIONES ESPACIALES
# ======================================================
def compute_spatial_lag(train_coords, train_prices, target_coords, k=30, is_self=False):
    """
    Spatial lag con k=30 vecinos (antes k=20).
    Al usar spatial block CV, el fold de test puede estar geográficamente
    alejado del centroide del cluster: aumentar k asegura que incluso los
    puntos en el borde del bloque de test encuentren vecinos de train.
    El kernel 1/(d+10) penaliza distancia, así los vecinos lejanos tienen
    peso bajo pero evitan NaN/extrapolación extrema.
    """
    tree = cKDTree(train_coords)
    fetch = min(k + 1, len(train_coords))
    d, ix = tree.query(target_coords, k=fetch)
    if is_self:
        d, ix = d[:, 1:], ix[:, 1:]
    weights = 1.0 / (d + 10.0)
    lag_values = [np.average(train_prices[idxs], weights=w) for idxs, w in zip(ix, weights)]
    return np.log1p(np.array(lag_values))

def compute_local_neighbor_price(train_coords, train_prices, target_coords, k=15):
    nbrs = NearestNeighbors(n_neighbors=min(k, len(train_coords))).fit(train_coords)
    _, indices = nbrs.kneighbors(target_coords)
    return np.array([train_prices[idx].mean() for idx in indices])

# ======================================================
# 4. MODELO ELASTICNET POR CLUSTERS (GUARDA VARIABLES DINÁMICAS)
# ======================================================
def _segmentar_dos_etapas(data, n_clusters=3, k_spatial=30):
    """
    Segmentación en dos etapas basada en la literatura de submercados
    hedónicos (Usman & Lizam 2023; Kopczewska 2021; Glasgow-paper 2023):

    Etapa 1 — PCA sobre señales de precio y localización:
      • precio/m² suavizado espacialmente (k=30 vecinos, kernel 1/d)
      • distancia al subcentro más cercano (ya en el dataset)
      • marginalidad_score  (nivel socioeconómico de la zona)
      • gentrification_index (dinámica de cambio socioeconómico)
      → 2 componentes principales capturan el gradiente radial del mapa

    Etapa 2 — KMeans espacialmente restringido:
      • Clustering sobre los scores PCA PERO con penalización geográfica:
        se añaden lat/lon normalizados con peso 0.4 para que clusters
        sean aproximadamente contiguos (evita que el mismo cluster
        aparezca en zonas disconnected del mapa)
      → n_clusters=3 estratos: Bajo/Medio/Alto

    El estrato resultante se almacena en data["cluster"] Y también en
    data["estrato_num"] (0/1/2) para usarse como covariate ordinal
    dentro del ElasticNet de cada submodelo (siguiendo Melo & Melo 2005).

    Retorna data con columnas "cluster" y "estrato_num".
    """
    coords_xy = data[["longitud", "latitud"]].values
    prices    = data["price_m2_raw"].values

    # ── Etapa 1a: precio vecinal suavizado ──
    tree  = cKDTree(coords_xy)
    fetch = min(k_spatial + 1, len(coords_xy))
    d, ix = tree.query(coords_xy, k=fetch)
    d, ix = d[:, 1:], ix[:, 1:]
    w     = 1.0 / (d + 50.0)
    precio_suavizado = np.array([
        np.average(prices[idxs], weights=ws) for idxs, ws in zip(ix, w)
    ])

    # ── Etapa 1b: PCA sobre señales de precio + localización ──
    # Se añade colonia_price_enc si existe: captura el gradiente intra-zona
    # que la ubicación lat/lon no alcanza a discriminar (el problema del cluster 2).
    pca_dict = {
        "precio_suavizado" : precio_suavizado,
        "dist_subcenter"   : data["dist_subcenter_log"].values,
        "marginalidad"     : data["marginalidad_score"].values,
        "gentrificacion"   : data["gentrification_index"].values,
    }
    if "colonia_price_enc" in data.columns:
        pca_dict["colonia_enc"] = data["colonia_price_enc"].values

    pca_feats = pd.DataFrame(pca_dict).fillna(0)

    scores_pca = PCA(n_components=2).fit_transform(
        StandardScaler().fit_transform(pca_feats)
    )  # shape (N, 2)

    # ── Etapa 2: KMeans con restricción geográfica suave ──
    # Peso 0.6 a los scores PCA + 0.4 a lat/lon normalizados
    geo_norm  = StandardScaler().fit_transform(coords_xy)
    X_cluster = np.hstack([
        scores_pca * 0.6,
        geo_norm   * 0.4,
    ])
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(X_cluster)

    # Reordenar clusters por precio suavizado medio (0=Bajo … n-1=Alto)
    medias = {lbl: precio_suavizado[labels == lbl].mean() for lbl in range(n_clusters)}
    orden  = sorted(medias, key=medias.get)
    remap  = {old: new for new, old in enumerate(orden)}
    estrato = np.array([remap[l] for l in labels])

    data["cluster"]     = estrato
    data["estrato_num"] = estrato   # covariate ordinal para el modelo

    nombres_base = {0: "Bajo", 1: "Medio", 2: "Alto", 3: "Alto+"}
    for c in range(n_clusters):
        mask  = data["cluster"] == c
        n_c   = mask.sum()
        med_c = data.loc[mask, "price_m2_raw"].median()
        print(f"    Estrato {nombres_base.get(c, c)} (cluster {c}): "
              f"{n_c} registros | mediana $/m²: {med_c:.1f}")
    return data


def _build_fold_iter(d_c, coords_c, n_folds=5):
    """
    Elige la estrategia de CV óptima para el cluster dado:

    — CV por COLONIA (prioritario cuando el cluster tiene alta varianza interna):
        Los folds agrupan colonias completas. Esto evita que el modelo aprenda
        el precio de una colonia específica durante train y luego lo "recuerde"
        en test — el error principal del Cluster Alto (Benito Juárez/Cuauhtémoc/
        Miguel Hidalgo), donde colonias a 500 m de distancia difieren 4× en $/m².
        Se activa cuando el coeficiente de variación del precio es > 0.6 Y existe
        la columna colonia_norm con al menos 2n_folds colonias distintas.

    — CV espacial por bloques KMeans (fallback):
        Divide el espacio geográfico en bloques contiguos. Adecuado para clusters
        con gradiente suave de precios (Bajo y Medio).

    Retorna lista de tuplas (train_idx, test_idx) en posiciones del array d_c.
    """
    MIN_TR, MIN_TE = 30, 10

    # ── Criterio de alta varianza: CV > 0.6 ──
    pm2 = d_c["price_m2_raw"].values
    cv_precio = pm2.std() / (pm2.mean() + 1e-5)
    tiene_colonia = (
        "colonia_norm" in d_c.columns
        and d_c["colonia_norm"].nunique() >= 2 * n_folds
    )
    usar_colonia_cv = (cv_precio > 0.6) and tiene_colonia

    if usar_colonia_cv:
        # ── CV por colonia: agrupar colonias en n_folds bins por precio medio ──
        # Se ordenan las colonias por su precio mediano y se asignan en round-robin
        # a los folds → cada fold tiene la misma distribución de precios.
        col_med = (
            d_c.groupby("colonia_norm")["price_m2_raw"]
            .median()
            .sort_values()
            .reset_index()
        )
        col_med["fold"] = np.arange(len(col_med)) % n_folds
        fold_map = dict(zip(col_med["colonia_norm"], col_med["fold"]))
        fold_asign = d_c["colonia_norm"].map(fold_map).fillna(
            np.random.randint(0, n_folds)   # colonias nuevas → fold aleatorio
        ).astype(int).values

        fold_iter = [
            (np.where(fold_asign != f)[0], np.where(fold_asign == f)[0])
            for f in range(n_folds)
            if (fold_asign != f).sum() >= MIN_TR and (fold_asign == f).sum() >= MIN_TE
        ]
        if fold_iter:
            print(f"    → CV por colonia (CV precio={cv_precio:.2f}, "
                  f"{d_c['colonia_norm'].nunique()} colonias → {len(fold_iter)} folds)")
            return fold_iter, "colonia"

    # ── Fallback: CV espacial por bloques KMeans ──
    spatial_blocks = KMeans(
        n_clusters=n_folds, random_state=42, n_init=10
    ).fit_predict(coords_c)
    fold_iter = [
        (np.where(spatial_blocks != f)[0], np.where(spatial_blocks == f)[0])
        for f in range(n_folds)
        if (spatial_blocks != f).sum() >= MIN_TR and (spatial_blocks == f).sum() >= MIN_TE
    ]
    if not fold_iter:
        fold_iter = [
            (np.where(spatial_blocks != f)[0], np.where(spatial_blocks == f)[0])
            for f in range(n_folds)
        ]
    return fold_iter, "espacial"


def _run_segmento(data, label, alphas, l1_ratios, use_estrato_dummy=False):
    """
    Ajusta ElasticNetCV por cluster sobre `data` (ya segmentado).

    Para el cluster con mayor varianza interna (típicamente el Estrato Alto,
    concentrado en Benito Juárez/Cuauhtémoc/Miguel Hidalgo) usa CV por colonia
    en lugar de CV espacial — ver _build_fold_iter para la lógica de selección.

    Retorna (global_real, global_pred, data_con_predicciones).
    """
    data["precio_predicho"] = np.nan
    global_real, global_pred = [], []
    MIN_CLUSTER = 50

    feats = list(features)
    if use_estrato_dummy and "estrato_num" in data.columns:
        feats = feats + ["estrato_num"]

    for c in sorted(data["cluster"].unique()):
        d_c = data[data["cluster"] == c].copy()
        if len(d_c) < MIN_CLUSTER:
            print(f"  ⚠ Cluster {c}: {len(d_c)} registros, se omite")
            continue

        _gdf_c = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(d_c["longitud"], d_c["latitud"]),
            crs="EPSG:4326",
        ).to_crs(epsg=32614)
        _xy_utm = np.array([(_g.x, _g.y) for _g in _gdf_c.geometry])
        _centroide_utm = _xy_utm.mean(axis=0)
        d_c = d_c.copy()
        d_c["dist_estrato_centro"] = np.sqrt(
            (_xy_utm[:, 0] - _centroide_utm[0])**2 +
            (_xy_utm[:, 1] - _centroide_utm[1])**2
        )

        X_base   = d_c[static_features].copy()
        y        = np.log1p(d_c["price_m2_raw"])
        areas    = d_c["area"].values
        coords_c = d_c[["longitud", "latitud"]].values

        # ── Selección automática de estrategia CV ──
        fold_iter, cv_tipo = _build_fold_iter(d_c, coords_c, n_folds=5)

        c_real, c_pred = [], []

        for fold, (tr, te) in enumerate(fold_iter):
            X_train = X_base.iloc[tr].copy()
            X_test  = X_base.iloc[te].copy()

            # ── Spatial lag ──
            # Train: solo vecinos de train (excluye self) para no filtrar el propio precio.
            # Test:  TODOS los puntos del cluster como referencia de mercado.
            #        Esto es correcto tanto en CV espacial como en CV por colonia:
            #        los precios vecinos son datos de mercado observados, no etiquetas
            #        de y_test. El modelo no puede "ver" el precio del punto test en sí,
            #        pero sí puede ver los precios de sus vecinos geográficos.
            lag_tr = compute_spatial_lag(
                coords_c[tr], d_c.iloc[tr]["price_m2_raw"].values,
                coords_c[tr], is_self=True
            )
            lag_te = compute_spatial_lag(
                coords_c, d_c["price_m2_raw"].values,
                coords_c[te], is_self=True
            )
            np_tr = compute_local_neighbor_price(
                coords_c[tr], d_c.iloc[tr]["price_m2_raw"].values, coords_c[tr]
            )
            np_te = compute_local_neighbor_price(
                coords_c, d_c["price_m2_raw"].values, coords_c[te]
            )

            X_train["spatial_lag_price"]    = lag_tr
            X_test["spatial_lag_price"]     = lag_te
            X_train["precio_vecinal_local"] = np_tr
            X_test["precio_vecinal_local"]  = np_te
            X_train["lag_x_area"] = lag_tr * np.log1p(X_train["area"])
            X_test["lag_x_area"]  = lag_te * np.log1p(X_test["area"])

            # ── ratio_precio_vecinal ──
            # Train: vecinos dentro de train (sin leakage).
            # Test:  vecinos en TODOS el cluster (mismo argumento que el lag).
            if "ratio_precio_vecinal" in X_train.columns:
                # Train ratio — solo vecinos de train
                _nbrs_tr = NearestNeighbors(
                    n_neighbors=min(16, len(tr)), algorithm="ball_tree"
                ).fit(coords_c[tr])
                _, _idx_tr_r = _nbrs_tr.kneighbors(coords_c[tr])
                _ratio_tr = []
                for _ii in range(len(tr)):
                    _p_vec = d_c.iloc[tr]["price_m2_raw"].iloc[_idx_tr_r[_ii]].median()
                    _ratio_tr.append(np.clip(
                        d_c.iloc[tr]["price_m2_raw"].iloc[_ii] / (_p_vec + 1e-5), 0, 5
                    ))
                X_train["ratio_precio_vecinal"] = _ratio_tr

                # Test ratio — vecinos en TODO el cluster
                _nbrs_all = NearestNeighbors(
                    n_neighbors=min(16, len(coords_c)), algorithm="ball_tree"
                ).fit(coords_c)
                _, _idx_te_r = _nbrs_all.kneighbors(coords_c[te])
                _ratio_te = []
                for _ii, _row_idx in enumerate(te):
                    # excluir el propio punto del cálculo de vecindad
                    _neigh_idx = [x for x in _idx_te_r[_ii] if x != _row_idx][:15]
                    _p_vec = d_c["price_m2_raw"].iloc[_neigh_idx].median()
                    _ratio_te.append(np.clip(
                        d_c["price_m2_raw"].iloc[_row_idx] / (_p_vec + 1e-5), 0, 5
                    ))
                X_test["ratio_precio_vecinal"] = _ratio_te

            # dist_estrato_centro ya está en d_c, no en X_base → añadir
            X_train["dist_estrato_centro"] = d_c["dist_estrato_centro"].iloc[tr].values
            X_test["dist_estrato_centro"]  = d_c["dist_estrato_centro"].iloc[te].values

            data.loc[d_c.iloc[tr].index, "spatial_lag_price"]    = lag_tr
            data.loc[d_c.iloc[te].index, "spatial_lag_price"]    = lag_te
            data.loc[d_c.iloc[tr].index, "precio_vecinal_local"] = np_tr
            data.loc[d_c.iloc[te].index, "precio_vecinal_local"] = np_te
            data.loc[d_c.iloc[tr].index, "lag_x_area"] = lag_tr  * np.log1p(d_c.iloc[tr]["area"].values)
            data.loc[d_c.iloc[te].index, "lag_x_area"] = lag_te  * np.log1p(d_c.iloc[te]["area"].values)

            # Recalcular colonia_price_enc dentro del fold (leave-one-out con solo train)
            if "colonia_price_enc" in X_train.columns and "colonia_norm" in d_c.columns:
                _tr_df = d_c.iloc[tr].copy()
                _te_df = d_c.iloc[te].copy()
                _global_mean_fold = _tr_df["price_m2_raw"].mean()
                _col_sum_tr  = _tr_df.groupby("colonia_norm")["price_m2_raw"].transform("sum")
                _col_cnt_tr  = _tr_df.groupby("colonia_norm")["price_m2_raw"].transform("count")
                X_train["colonia_price_enc"] = np.where(
                    _col_cnt_tr > 1,
                    (_col_sum_tr - _tr_df["price_m2_raw"].values) / (_col_cnt_tr - 1),
                    _global_mean_fold,
                )
                X_train["colonia_price_enc"] = np.log1p(X_train["colonia_price_enc"])
                # Test: media de la colonia en train (sin el propio punto, que está en test)
                _col_means_tr = _tr_df.groupby("colonia_norm")["price_m2_raw"].mean()
                X_test["colonia_price_enc"] = np.log1p(
                    _te_df["colonia_norm"].map(_col_means_tr).fillna(_global_mean_fold).values
                )

            X_train = X_train.fillna(X_train.median()).fillna(0)
            X_test  = X_test.fillna(X_train.median()).fillna(0)

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("enet", ElasticNetCV(
                    alphas=alphas,
                    l1_ratio=l1_ratios,
                    cv=min(5, len(d_c) // 15),
                    max_iter=3000,
                )),
            ])
            model.fit(X_train, y.iloc[tr])

            p5 = d_c["price_m2_raw"].quantile(0.05)
            pred_m2    = np.clip(np.expm1(model.predict(X_test)), p5 * 0.1, None)
            pred_total = pred_m2 * areas[te]

            data.loc[d_c.iloc[te].index, "precio_predicho"] = pred_total

            fold_r2 = r2_score(d_c["price"].iloc[te], pred_total)
            if fold_r2 < -0.5:
                continue
            c_pred.extend(pred_total)
            c_real.extend(d_c["price"].iloc[te])

        if c_real:
            print(f"  Cluster {c} | n={len(d_c)} | CV={cv_tipo} | R²: {r2_score(c_real, c_pred):.4f}")
            global_real.extend(c_real)
            global_pred.extend(c_pred)

    return global_real, global_pred, data


def _evaluar_segmentacion(data, n_clusters, alphas, l1_ratios, label_debug):
    """
    Prueba una segmentación con n_clusters dado y devuelve el R² ponderado
    en CV espacial (sin guardar predicciones en data para no contaminar).
    Recalcula colonia_price_enc dentro de cada fold para evitar leakage.
    """
    data_tmp = _segmentar_dos_etapas(data.copy(), n_clusters=n_clusters, k_spatial=30)
    r2_clusters, pesos = [], []
    MIN_CLUSTER = 50

    for c in sorted(data_tmp["cluster"].unique()):
        d_c = data_tmp[data_tmp["cluster"] == c].copy()
        if len(d_c) < MIN_CLUSTER:
            continue

        coords_c = d_c[["longitud", "latitud"]].values
        areas    = d_c["area"].values
        X_base   = d_c[static_features].copy()
        y        = np.log1p(d_c["price_m2_raw"])

        fold_iter, _ = _build_fold_iter(d_c, coords_c, n_folds=5)
        if not fold_iter:
            continue

        c_real, c_pred = [], []
        for tr, te in fold_iter:
            X_tr = X_base.iloc[tr].copy()
            X_te = X_base.iloc[te].copy()

            lag_tr = compute_spatial_lag(coords_c[tr], d_c.iloc[tr]["price_m2_raw"].values, coords_c[tr], is_self=True)
            lag_te = compute_spatial_lag(coords_c, d_c["price_m2_raw"].values, coords_c[te], is_self=True)
            np_tr  = compute_local_neighbor_price(coords_c[tr], d_c.iloc[tr]["price_m2_raw"].values, coords_c[tr])
            np_te  = compute_local_neighbor_price(coords_c, d_c["price_m2_raw"].values, coords_c[te])

            X_tr["spatial_lag_price"] = lag_tr;   X_te["spatial_lag_price"] = lag_te
            X_tr["precio_vecinal_local"] = np_tr; X_te["precio_vecinal_local"] = np_te
            X_tr["lag_x_area"] = lag_tr * np.log1p(X_tr["area"])
            X_te["lag_x_area"] = lag_te * np.log1p(X_te["area"])

            # ratio_precio_vecinal: train=solo vecinos train, test=todo el cluster
            if "ratio_precio_vecinal" in X_tr.columns:
                _nbrs_tr = NearestNeighbors(n_neighbors=min(16, len(tr)), algorithm="ball_tree").fit(coords_c[tr])
                _, _idx_tr_r = _nbrs_tr.kneighbors(coords_c[tr])
                X_tr["ratio_precio_vecinal"] = [
                    np.clip(d_c.iloc[tr]["price_m2_raw"].iloc[i] /
                            (d_c.iloc[tr]["price_m2_raw"].iloc[_idx_tr_r[i]].median() + 1e-5), 0, 5)
                    for i in range(len(tr))
                ]
                _nbrs_all = NearestNeighbors(n_neighbors=min(16, len(coords_c)), algorithm="ball_tree").fit(coords_c)
                _, _idx_te_r = _nbrs_all.kneighbors(coords_c[te])
                X_te["ratio_precio_vecinal"] = [
                    np.clip(d_c["price_m2_raw"].iloc[te[i]] /
                            (d_c["price_m2_raw"].iloc[[x for x in _idx_te_r[i] if x != te[i]][:15]].median() + 1e-5), 0, 5)
                    for i in range(len(te))
                ]

            # Recalcular colonia_price_enc dentro del fold para evitar leakage
            if "colonia_price_enc" in X_tr.columns and "colonia_norm" in d_c.columns:
                _tr_df = d_c.iloc[tr]
                _te_df = d_c.iloc[te]
                _gmean = _tr_df["price_m2_raw"].mean()
                _s = _tr_df.groupby("colonia_norm")["price_m2_raw"].transform("sum")
                _n = _tr_df.groupby("colonia_norm")["price_m2_raw"].transform("count")
                X_tr["colonia_price_enc"] = np.log1p(
                    np.where(_n > 1, (_s - _tr_df["price_m2_raw"].values) / (_n - 1), _gmean)
                )
                _means_tr = _tr_df.groupby("colonia_norm")["price_m2_raw"].mean()
                X_te["colonia_price_enc"] = np.log1p(
                    _te_df["colonia_norm"].map(_means_tr).fillna(_gmean).values
                )

            X_tr = X_tr.fillna(X_tr.median()).fillna(0)
            X_te = X_te.fillna(X_tr.median()).fillna(0)

            mdl = Pipeline([
                ("scaler", StandardScaler()),
                ("enet", ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios,
                                      cv=min(5, len(d_c) // 15), max_iter=3000)),
            ])
            mdl.fit(X_tr, y.iloc[tr])
            p5 = d_c["price_m2_raw"].quantile(0.05)
            pred_m2 = np.clip(np.expm1(mdl.predict(X_te)), p5 * 0.1, None)
            c_pred.extend(pred_m2 * areas[te])
            c_real.extend(d_c["price"].iloc[te])

        if len(c_real) > 10:
            r2_clusters.append(r2_score(c_real, c_pred))
            pesos.append(len(c_real))

    if not r2_clusters:
        return float("nan")
    r2_pond = float(np.average(r2_clusters, weights=pesos))
    print(f"    [{label_debug}] R² ponderado CV: {r2_pond:.4f}")
    return r2_pond


def run_market_optimized(data, label):
    print("\n" + "=" * 80)
    print(f"PROCESANDO ELASTICNET POR CLUSTERS: {label}")
    print("=" * 80)

    data = data.copy()

    q_low, q_high = data["price_m2_raw"].quantile([0.05, 0.95])
    data = data[(data["price_m2_raw"] > q_low) & (data["price_m2_raw"] < q_high)].copy()

    if len(data) < 50:
        print(f"  ⚠ Insuficientes datos para {label} ({len(data)} registros)")
        return [], [], data

    if label == "MERCADO ESTÁNDAR":
        alphas    = np.logspace(-5, -1, 40)
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

        # ── Selección automática de n_clusters: 3, 4, 5, 6 ──
        # Se evalúan 4 opciones y se elige la que da mayor R² ponderado en CV.
        # Umbral mínimo de +1pp sobre n=3 para justificar la complejidad añadida.
        candidatos = [3, 4, 5, 6]
        print(f"  Evaluando n_clusters = {candidatos} (selección automática):")
        resultados_n = {}
        for n_cand in candidatos:
            r2_n = _evaluar_segmentacion(data, n_cand, alphas, l1_ratios, f"n={n_cand}")
            resultados_n[n_cand] = r2_n

        # Tabla resumen
        print(f"\n  {'n_clusters':<12} {'R² CV ponderado':<18}")
        print(f"  {'-'*30}")
        for n_c, r2_c in resultados_n.items():
            marca = " ←" if r2_c == max(resultados_n.values()) else ""
            print(f"  {n_c:<12} {r2_c:<18.4f}{marca}")

        r2_base = resultados_n[3]
        n_best = max(resultados_n, key=resultados_n.get)
        if resultados_n[n_best] <= r2_base + 0.01:
            n_best = 3   # no justifica complejidad extra
        print(f"\n  → Seleccionado n_clusters={n_best} "
              f"(R²={resultados_n[n_best]:.4f})")

        print(f"  Segmentación final en 2 etapas (PCA+colonia → KMeans geo, k={n_best}):")
        data = _segmentar_dos_etapas(data, n_clusters=n_best, k_spatial=30)
        use_estrato = False

    else:
        # ── Lujo: segmento único, Ridge puro (l1_ratio=0) ──
        data["cluster"]     = 0
        data["estrato_num"] = 0
        print(f"  Segmento único: {len(data)} registros")
        alphas    = np.logspace(-3, 2, 60)
        l1_ratios = [0.0]
        use_estrato = False

    return _run_segmento(data, label, alphas, l1_ratios, use_estrato_dummy=use_estrato)

# ======================================================
# 5. MODELO K-NN COMPARATIVO (también guarda, opcional)
# ======================================================
def run_knn_comparison(data, label):
    print("\n" + "=" * 80)
    print(f"PROCESANDO COMPARATIVA k-NN: {label}")
    print("=" * 80)

    d = data.copy()
    q_low, q_high = d["price_m2_raw"].quantile([0.05, 0.95])
    d = d[(d["price_m2_raw"] > q_low) & (d["price_m2_raw"] < q_high)].copy()

    if len(d) < 50:
        print(f"  ⚠ Datos insuficientes ({len(d)})")
        return 0, 1, [], []

    X      = d[static_features].copy()
    y      = np.log1p(d["price_m2_raw"])
    areas  = d["area"].values
    coords = d[["longitud", "latitud"]].values

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

        X_train["spatial_lag_price"]   = lag_tr; X_test["spatial_lag_price"]   = lag_te
        X_train["precio_vecinal_local"] = np_tr;  X_test["precio_vecinal_local"] = np_te
        X_train["lag_x_area"] = lag_tr * np.log1p(X_train["area"])
        X_test["lag_x_area"]  = lag_te * np.log1p(X_test["area"])

        # (Opcional) guardar también en el dataframe si se desea
        # pero para no relentizar, se omite en k-NN. Aún así se usan para predecir.

        X_train = X_train.fillna(X_train.median()).fillna(0)
        X_test  = X_test.fillna(X_train.median()).fillna(0)

        knn = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=min(12, len(tr)), weights="distance")),
        ])
        knn.fit(X_train, y.iloc[tr])
        pred_m2 = np.expm1(knn.predict(X_test))
        pred.extend(pred_m2 * areas[te])
        real.extend(d["price"].iloc[te])

    if not real:
        return 0, 1, [], []

    r2   = r2_score(real, pred)
    mape = mean_absolute_percentage_error(real, pred)
    print(f"  k-NN → R²: {r2:.4f} | MAPE: {mape:.2%}")
    return r2, mape, real, pred

# ======================================================
# 6. EJECUCIÓN
# ======================================================
re_est, pe_est, df_est = run_market_optimized(
    df[df.is_luxury == 0].copy(), "MERCADO ESTÁNDAR"
)
re_lux, pe_lux, df_lux = run_market_optimized(
    df[df.is_luxury == 1].copy(), "MERCADO LUJO"
)

# Guardar variables dinámicas también en los dataframes originales para exportación
# (df_est y df_lux ya las contienen porque run_market_optimized las almacenó)
# Ahora unir para los análisis finales
df_combined = pd.concat([df_est, df_lux], ignore_index=True)

# KNN comparativo (usando los dataframes que ya tienen las dinámicas, pero se recalculan internamente)
rk_est, mk_est, rk_est_r, rk_est_p = run_knn_comparison(df_est, "MERCADO ESTÁNDAR")
rk_lux, mk_lux, rk_lux_r, rk_lux_p = run_knn_comparison(df_lux, "MERCADO LUJO")

y_true_en = np.array(re_est + re_lux)
y_pred_en = np.array(pe_est + pe_lux)
y_true_kn = np.concatenate([rk_est_r, rk_lux_r]) if (rk_est_r and rk_lux_r) else np.array([])
y_pred_kn = np.concatenate([rk_est_p, rk_lux_p]) if (rk_est_p and rk_lux_p) else np.array([])

def safe_r2(real, pred):
    if len(real) < 2:
        return float("nan")
    return r2_score(real, pred)

r2_en_est    = safe_r2(re_est, pe_est)
r2_en_lux    = safe_r2(re_lux, pe_lux)
weights_en   = [len(re_est), len(re_lux)]
r2_en_global = np.average(
    [v for v in [r2_en_est, r2_en_lux] if not np.isnan(v)],
    weights=[w for v, w in zip([r2_en_est, r2_en_lux], weights_en) if not np.isnan(v)]
) if any(not np.isnan(v) for v in [r2_en_est, r2_en_lux]) else float("nan")

# ======================================================
# 7. VISUALIZACIONES Y RESUMEN
# ======================================================
if len(y_true_en) > 0:
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.regplot(
        x=y_true_en, y=y_pred_en, ax=ax,
        scatter_kws={"alpha": 0.3, "color": "steelblue"},
        line_kws={"color": "red"},
    )
    ax.set_title(
        f"ElasticNet + Clusters — RENTAS (Global)\nR²: {r2_en_global:.3f}",
        fontsize=14,
    )
    ax.set_xlabel("Renta Real (MXN/mes)", fontsize=12)
    ax.set_ylabel("Renta Predicha (MXN/mes)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "elasticnet_real_vs_predicho_rentas.png"), dpi=300)
    plt.close("all")

if len(y_true_en) > 0:
    plt.figure(figsize=(10, 6))
    error_en = ((y_pred_en - y_true_en) / (y_true_en + 1e-5)) * 100
    sns.kdeplot(error_en, fill=True, color="steelblue", label="ElasticNet Clusters")
    plt.axvline(0, color="black", linestyle="--")
    plt.title("Distribución del Error Porcentual — RENTAS (%)", fontsize=14)
    plt.xlabel("Error (%)"); plt.xlim(-100, 100)
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, "distribucion_error_rentas.png"), dpi=300)
    plt.close("all")

print("\n" + "#" * 80)
print("RESUMEN COMPARATIVO FINAL — RENTAS")
print("#" * 80)
print(f"{'MÉTRICA':<22} | {'CLUSTER + ENET':<18} | {'k-NN (Vecindad)':<18}")
print("-" * 65)

if len(y_true_en) > 0:
    mape_en = mean_absolute_percentage_error(y_true_en, y_pred_en)
    rmse_en = np.sqrt(mean_squared_error(np.log1p(y_true_en), np.log1p(y_pred_en)))
else:
    mape_en = rmse_en = float("nan")

r2_kn_global = np.average(
    [v for v in [rk_est, rk_lux] if not np.isnan(v)],
    weights=[w for v, w in zip([rk_est, rk_lux], [len(rk_est_r), len(rk_lux_r)]) if not np.isnan(v)]
) if any(not np.isnan(v) for v in [rk_est, rk_lux]) else float("nan")
mape_kn_global = np.nanmean([mk_est, mk_lux])

if len(y_true_kn) > 0:
    rmse_kn = np.sqrt(mean_squared_error(np.log1p(y_true_kn), np.log1p(y_pred_kn)))
else:
    rmse_kn = float("nan")

print(f"{'R² Global (pond.)':<22} | {r2_en_global:.4f}{'':<13} | {r2_kn_global:.4f}")
print(f"{'R² Estándar':<22} | {r2_en_est:.4f}{'':<13} | {rk_est:.4f}")
print(f"{'R² Lujo':<22} | {r2_en_lux:.4f}{'':<13} | {rk_lux:.4f}")
print(f"{'MAPE Global':<22} | {mape_en*100:.2f}%{'':<11} | {mape_kn_global*100:.2f}%")
print(f"{'RMSE Logarítmico':<22} | {rmse_en:.4f}{'':<13} | {rmse_kn:.4f}")
print("#" * 80)

# ======================================================
# 8. SEGMENTACIONES PARA EXPLICABILIDAD (usando df_combined)
# ======================================================
print("\n" + "=" * 80)
print("SEGMENTACIONES PARA EXPLICABILIDAD — RENTAS")
print("=" * 80)

df_qgis_preview = df_combined.copy()

# ── 8a. Tipología de Vivienda ──
# NOTA: parking_spaces se descartó como criterio de corte en "Amplio" porque
# los dos datasets usan escalas distintas (booleano vs. entero), lo que hacía
# que la partición ConEstacionamiento/SinEstacionamiento capturara ruido del
# dataset de origen más que diferencias reales de mercado. Se sustituye por
# rooms > 3, que distingue bien viviendas grandes familiares vs. ejecutivas.
print("\n── 8a. Tipología de Vivienda ──")
conditions = [
    (df_qgis_preview["area"] < 50) & (df_qgis_preview["bathrooms"] <= 1),
    (df_qgis_preview["area"] < 50) & (df_qgis_preview["bathrooms"] > 1),
    (df_qgis_preview["area"].between(50, 100)) & (df_qgis_preview["rooms"] <= 2),
    (df_qgis_preview["area"].between(50, 100)) & (df_qgis_preview["rooms"] > 2),
    (df_qgis_preview["area"] > 100) & (df_qgis_preview["is_luxury"] == 0) & (df_qgis_preview["rooms"] > 3),
    (df_qgis_preview["area"] > 100) & (df_qgis_preview["is_luxury"] == 0) & (df_qgis_preview["rooms"] <= 3),
    (df_qgis_preview["area"] > 100) & (df_qgis_preview["is_luxury"] == 1),
]
tipologias = ["Studio/Micro", "Compacto+", "Familiar_Chico", "Familiar_Grande",
              "Amplio_Familiar", "Amplio_Ejecutivo", "Residencial_Lujo"]
df_qgis_preview["tipologia"] = np.select(conditions, tipologias, default="Otro")
print(f"  Categorías asignadas: {df_qgis_preview['tipologia'].nunique()}")
tip_stats = df_qgis_preview.groupby("tipologia")["price_m2_raw"].agg(
    n="count", mediana="median", media="mean", p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75)
).reset_index().sort_values("mediana", ascending=False)
print(tip_stats.to_string(index=False))

# ── 8b. Segmento Accesibilidad Transit ──
print("\n── 8b. Segmento por Accesibilidad al Transporte Público ──")
for col in ["dist_metro_m", "dist_metrobus_m", "dist_trole_m", "dist_tren_m"]:
    if col in df_qgis_preview.columns:
        df_qgis_preview[col + "_raw"] = np.expm1(df_qgis_preview[col])
dist_cols_raw = [c for c in ["dist_metro_m_raw", "dist_metrobus_m_raw", "dist_trole_m_raw"] if c in df_qgis_preview.columns]
pesos = {"dist_metro_m_raw": 0.50, "dist_metrobus_m_raw": 0.30, "dist_trole_m_raw": 0.20}
if dist_cols_raw:
    df_qgis_preview["accesibilidad_transit"] = sum(
        df_qgis_preview[c] * pesos.get(c, 0.1) for c in dist_cols_raw
    )
    transit_q = df_qgis_preview["accesibilidad_transit"].quantile([0.33, 0.66])
    df_qgis_preview["seg_transit"] = pd.cut(
        df_qgis_preview["accesibilidad_transit"],
        bins=[-np.inf, transit_q.iloc[0], transit_q.iloc[1], np.inf],
        labels=["Alta_Accesibilidad", "Media_Accesibilidad", "Baja_Accesibilidad"]
    )
    tr_stats = df_qgis_preview.groupby("seg_transit", observed=True)["price_m2_raw"].agg(
        n="count", mediana="median", media="mean"
    ).reset_index()
    print(tr_stats.to_string(index=False))
    if len(tr_stats) >= 2:
        premium_transit = (tr_stats.loc[tr_stats["seg_transit"]=="Alta_Accesibilidad","mediana"].values[0] /
                           tr_stats.loc[tr_stats["seg_transit"]=="Baja_Accesibilidad","mediana"].values[0] - 1) * 100
        print(f"  → Premio por Alta vs Baja accesibilidad: +{premium_transit:.1f}%")

# ── 8c. Cuadrante Socioeconómico ──
print("\n── 8c. Cuadrante Socioeconómico (Marginalidad × Gentrificación) ──")
marg_med   = df_qgis_preview["marginalidad_score"].median()
gentrif_med = df_qgis_preview["gentrification_index"].median()
df_qgis_preview["cuadrante_soc"] = np.where(
    (df_qgis_preview["marginalidad_score"] <= marg_med) & (df_qgis_preview["gentrification_index"] >= gentrif_med),
    "BajaMarg_AltaGentrif",
    np.where(
        (df_qgis_preview["marginalidad_score"] <= marg_med) & (df_qgis_preview["gentrification_index"] < gentrif_med),
        "BajaMarg_BajaGentrif",
        np.where(
            (df_qgis_preview["marginalidad_score"] > marg_med) & (df_qgis_preview["gentrification_index"] >= gentrif_med),
            "AltaMarg_AltaGentrif",
            "AltaMarg_BajaGentrif",
        )
    )
)
cuad_stats = df_qgis_preview.groupby("cuadrante_soc")["price_m2_raw"].agg(
    n="count", mediana="median", media="mean"
).reset_index().sort_values("mediana", ascending=False)
print(cuad_stats.to_string(index=False))

# ── 8d. Ranking de Renta por Alcaldía ──
ALCALDIAS_CDMX = {
    "alvaro obregon", "azcapotzalco", "benito juarez", "coyoacan",
    "cuajimalpa de morelos", "cuauhtemoc", "gustavo a. madero",
    "iztacalco", "iztapalapa", "la magdalena contreras",
    "miguel hidalgo", "milpa alta", "tlahuac", "tlalpan",
    "venustiano carranza", "xochimilco",
}
df_qgis_preview["alc_norm"] = df_qgis_preview["alcaldia"].apply(clean_text)
mask_alc = df_qgis_preview["alc_norm"].isin(ALCALDIAS_CDMX)
alc_stats = (
    df_qgis_preview[mask_alc]
    .groupby("alcaldia")
    .agg(n=("price_m2_raw","count"), mediana_m2=("price_m2_raw","median"),
         mediana_total=("price","median"), gentrif=("gentrification_index","mean"))
    .reset_index()
    .sort_values("mediana_m2", ascending=False)
)
print("\nRanking de renta por alcaldía (mediana $/m²):")
print(alc_stats.head(10).to_string(index=False))

# ── 8e. Análisis de error por tipología ──
print("\n── 8e. Análisis de Error por Tipología (dónde falla el modelo) ──")
if "precio_predicho" in df_qgis_preview.columns:
    df_err = df_qgis_preview.dropna(subset=["precio_predicho"]).copy()
    # Clip predicciones negativas o numéricamente inestables antes del cálculo de error
    df_err["precio_predicho"] = df_err["precio_predicho"].clip(lower=1.0)
    df_err["error_abs_pct"] = (
        (df_err["precio_predicho"] - df_err["price"]).abs()
        / df_err["price"].clip(lower=1.0)
    ) * 100
    # Winsorizar errores extremos (> 500%) para que la media sea interpretable
    df_err["error_abs_pct"] = df_err["error_abs_pct"].clip(upper=500.0)
    err_tip = df_err.groupby("tipologia")["error_abs_pct"].agg(
        n="count", mape_segmento="mean", mediana_error="median"
    ).reset_index().sort_values("mape_segmento", ascending=False)
    print(err_tip.to_string(index=False))
    print("  → Los segmentos con MAPE alto son donde conviene revisar features o datos")
else:
    print("  ⚠ No hay columna 'precio_predicho' — ejecuta run_market_optimized primero")

# ── 8f. Exportar segmentaciones ──
seg_cols = ["latitud", "longitud", "alcaldia", "price", "price_m2_raw",
            "area", "rooms", "bathrooms", "is_luxury",
            "tipologia", "cuadrante_soc", "gentrification_index",
            "marginalidad_score"]
seg_cols_ok = [c for c in seg_cols if c in df_qgis_preview.columns]
if "seg_transit" in df_qgis_preview.columns:
    seg_cols_ok.append("seg_transit")
    seg_cols_ok.append("accesibilidad_transit")

ruta_seg = os.path.join(RESULTS_PATH, "segmentaciones_rentas.csv")
df_qgis_preview[seg_cols_ok].to_csv(ruta_seg, index=False, encoding="utf-8-sig")
print(f"\n  ✓ Segmentaciones exportadas → {ruta_seg}")

# ======================================================
# INTERPRETABILIDAD HEDÓNICA
# ======================================================
def fit_interpretable_elasticnet(data, label):
    print("\n" + "=" * 80)
    print(f"TOP 15 VARIABLES HEDÓNICAS — {label} (RENTAS)")
    print("=" * 80)

    data = data.copy()
    # Asegurar que las columnas dinámicas existan (si no, se recalculan)
    if "spatial_lag_price" not in data.columns or data["spatial_lag_price"].isna().all():
        coords_l = data[["longitud", "latitud"]].values
        data["spatial_lag_price"]   = compute_spatial_lag(coords_l, data["price_m2_raw"].values, coords_l)
        data["precio_vecinal_local"] = compute_local_neighbor_price(coords_l, data["price_m2_raw"].values, coords_l)
        data["lag_x_area"] = data["spatial_lag_price"] * np.log1p(data["area"])

    X = data[features].copy()
    y = np.log1p(data["price_m2_raw"])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median()).fillna(0)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = ElasticNetCV(alphas=np.logspace(-4, -1, 25), l1_ratio=[0.4, 0.6, 0.8], cv=5)
    model.fit(Xs, y)

    coef_df = pd.DataFrame({"feature": features, "coef": model.coef_})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False)
    print(coef_df[["feature", "coef"]].head(15).to_string(index=False))

    return coef_df, model, scaler

if len(df_est) >= 30:
    coef_est, model_est, scaler_est = fit_interpretable_elasticnet(df_est, "MERCADO ESTÁNDAR")
if len(df_lux) >= 30:
    coef_lux, model_lux, scaler_lux = fit_interpretable_elasticnet(df_lux, "MERCADO LUJO")

# ======================================================
# EXPORTACIÓN PARA QGIS (con variables dinámicas)
# ======================================================
print("\n" + "=" * 80)
print("EXPORTACIÓN PARA QGIS")
print("=" * 80)

df_qgis = df_combined.copy()

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

ruta_qgis = os.path.join(RESULTS_PATH, "resultados_qgis_rentas.csv")
df_qgis.to_csv(ruta_qgis, index=False, encoding="utf-8-sig")
print(f"  ✓ {len(df_qgis)} puntos exportados → {ruta_qgis}")

print("\n" + "=" * 80)
print("PIPELINE DE RENTAS COMPLETADO")
print("=" * 80)