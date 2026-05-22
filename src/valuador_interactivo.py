# =============================================================================
# TRABAJO TERMINAL
# Modelo de avalúo inmobiliario con técnicas de análisis espacial considerando
# las variables de gentrificación en la CDMX
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
#   - Variables catastrales pre-calculadas (cargadas desde CSV exportado).
#
# NOTA: Las variables catastrales (cat_*) se cargan directamente del CSV
# exportado por modelo.py. No se requiere cargar los shapefiles del catastro.
# =============================================================================

# ============================================================
# LIBRERÍAS
# ============================================================
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
    page_title="Sistema de Valuación Inmobiliaria CDMX",
    layout="wide"
)

# ============================================================
# RUTAS RELATIVAS (ESTRUCTURA DE REPOSITORIO)
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

OUTPUTS_PATH   = os.path.join(PROJECT_ROOT, "..", "outputs", "results")
PATH_QGIS_DATA = os.path.join(OUTPUTS_PATH, "resultados_qgis_final.csv")

DATA_PATH     = os.path.join(PROJECT_ROOT, "..", "data")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")

# Polígonos de demarcaciones territoriales (colonias)
PATH_COLONIAS = os.path.join(RAW_DATA_PATH, "coloniascdmx", "colonias_iecm.shp")

# Capas de transporte (sólo para recalcular distancias al nuevo punto)
PATH_METRO_EST   = os.path.join(RAW_DATA_PATH, "stcmetro_shp", "STC_Metro_estaciones_utm14n.shp")
PATH_METROBUS_EST = os.path.join(RAW_DATA_PATH, "mb_shp", "Metrobus_estaciones.shp")
PATH_TREN_EST    = os.path.join(RAW_DATA_PATH, "ste_shp", "ste_tren_ligero_shp",
                                "ste_tren_ligero_shp", "STE_TrenLigero_estaciones_utm14n.shp")
PATH_TROLE_PARADAS = os.path.join(RAW_DATA_PATH, "ste_shp", "ste_trolebus_shp",
                                  "ste_trolebus_shp", "STE_Trolebus_Paradas.shp")
PATH_CABLE_EST   = os.path.join(RAW_DATA_PATH, "ste_shp", "ste_cablebus_shp",
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

# Ciclovías (líneas)
PATH_CICLOVIAS = os.path.join(RAW_DATA_PATH, "infraestructura_vial_ciclista", "Infraestructura ciclista total.shp")

# ============================================================
# SUBCENTROS URBANOS — idénticos a modelo.py
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
# FEATURES — ALINEADAS CON modelo.py
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
    # Variables catastrales (pre-calculadas en modelo.py, cargadas desde CSV)
    "cat_mean_valor_suelo_200m", "cat_mean_vus_200m",
    "cat_mean_antiguedad_200m", "cat_std_valor_suelo_200m",
    "cat_mean_ratio_construccion_200m", "cat_density_predios_200m",
]

_DYNAMIC_FEATURES = [
    "spatial_lag_price",
    "precio_vecinal_local",
    "lag_x_area",
]

_COLINEAR_DROP = []

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
    """Convierte una capa de líneas a puntos espaciados cada interval_m (metros)."""
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

def compute_spatial_lag(train_coords, train_prices, target_coords, k=10):
    tree = cKDTree(train_coords)
    d, ix = tree.query(target_coords, k=min(k, len(train_coords)))
    if k == 1:
        d = d.reshape(-1, 1)
        ix = ix.reshape(-1, 1)
    weights = 1 / (d + 1e-5)
    lag_values = []
    for idxs, w in zip(ix, weights):
        lag_values.append(np.average(train_prices[idxs], weights=w))
    return np.log1p(np.array(lag_values))

def compute_local_neighbor_price(train_coords, train_prices,
                                 target_coords, k=15):
    nbrs = NearestNeighbors(
        n_neighbors=min(k, len(train_coords))
    ).fit(train_coords)
    _, indices = nbrs.kneighbors(target_coords)
    return np.array([train_prices[idx].mean() for idx in indices])

# ============================================================
# SISTEMA PRINCIPAL (cacheado)
# ============================================================
@st.cache_resource
def preparar_sistema():
    """
    Carga el CSV exportado por modelo.py, replica el pipeline de features
    y entrena los modelos ElasticNet locales por segmento (estándar/lujo)
    y por clúster.
    Además carga capas de puntos y líneas (ciclovías) para visualización.
    """

    # ----------------------------------------------------------
    # 1. CARGA Y LIMPIEZA INICIAL
    # ----------------------------------------------------------
    df = pd.read_csv(PATH_QGIS_DATA)

    df["latitud"]  = pd.to_numeric(df["latitud"],  errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")
    df = df.dropna(subset=["latitud", "longitud", "price",
                            "area", "price_m2_raw"]).copy()
    df = df[df["area"] >= 20].copy()
    df = df.reset_index(drop=True)

    # ----------------------------------------------------------
    # 2. DETERMINAR STATIC_FEATURES (replicando drop por colinealidad)
    # ----------------------------------------------------------
    disponibles = [f for f in _STATIC_FEATURES_FULL if f in df.columns]
    corr_matrix = df[disponibles].corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    drop_cols = [
        col for col in upper.columns if any(upper[col] > 0.90)
    ]
    static_features = [f for f in disponibles if f not in drop_cols]
    features_all    = static_features + _DYNAMIC_FEATURES

    # ----------------------------------------------------------
    # 3. GEOMETRÍA PARA JOINS Y DISTANCIAS
    # ----------------------------------------------------------
    gdf_geo = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitud"], df["latitud"]),
        crs="EPSG:4326"
    ).reset_index(drop=True)

    gdf_utm     = gdf_geo.to_crs(epsg=32614)
    coords_utm  = np.array([(p.x, p.y) for p in gdf_utm.geometry])
    coords_geo  = df[["longitud", "latitud"]].values

    # ----------------------------------------------------------
    # 4. CRUCE ESPACIAL CON COLONIAS
    # ----------------------------------------------------------
    colonias = gpd.read_file(PATH_COLONIAS)
    if colonias.crs != "EPSG:4326":
        colonias = colonias.to_crs("EPSG:4326")

    colonias["colonia_clean"] = colonias["NOMUT"].apply(clean_text)

    posibles_alcaldias = ["NOMDT", "DEMARCACI", "DEMARCACION",
                          "MUNICIPIO", "NOM_MUN"]
    alcaldia_col = next(
        (c for c in posibles_alcaldias if c in colonias.columns), None
    )
    if alcaldia_col is None:
        raise ValueError(
            "Falta identificador de alcaldía en el shapefile de colonias."
        )
    colonias["alcaldia_clean"] = colonias[alcaldia_col].apply(clean_text)

    joined = gpd.sjoin(
        gdf_geo,
        colonias[["colonia_clean", "alcaldia_clean", "geometry"]],
        how="left",
        predicate="within"
    ).sort_index()

    df["colonia_real"]  = joined["colonia_clean"].values
    df["alcaldia_real"] = joined["alcaldia_clean"].values

    # ============================================================
    # CLAVE ÚNICA ALCALDÍA + COLONIA
    # Evita confusión entre colonias homónimas
    # ============================================================
    df["colonia_key"] = (
        df["alcaldia_real"].astype(str)
        + "||" +
        df["colonia_real"].astype(str)
    )

    df = df.dropna(subset=["colonia_real", "alcaldia_real"]).copy()
    df = df.reset_index(drop=True)

    gdf_utm2    = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitud"], df["latitud"]),
        crs="EPSG:4326"
    ).to_crs(epsg=32614)
    coords_utm2 = np.array([(p.x, p.y) for p in gdf_utm2.geometry])
    coords_geo2 = df[["longitud", "latitud"]].values

    # ----------------------------------------------------------
    # 5. RECALCULAR DIST A TRANSPORTE
    # ----------------------------------------------------------
    def _get_pts(path):
        try:
            gdf = gpd.read_file(path)
            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            gdf = gdf.to_crs(epsg=32614)
            gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
            exp = gdf.geometry.explode(index_parts=False)
            return np.array(
                [(g.x, g.y) for g in exp if g.geom_type == "Point"]
            )
        except Exception:
            return np.empty((0, 2))

    metro_pts   = _get_pts(PATH_METRO_EST)
    mb_pts      = _get_pts(PATH_METROBUS_EST)
    tren_pts    = _get_pts(PATH_TREN_EST)
    trole_pts   = _get_pts(PATH_TROLE_PARADAS)
    cable_pts   = _get_pts(PATH_CABLE_EST)

    # Áreas verdes recreativas
    try:
        av = gpd.read_file(PATH_AREAS_VERDES)
        cats_validas = ["Parques", "Deportivos", "Jardines públicos",
                        "Arboledas", "Alamedas"]
        if "subcat_sed" in av.columns:
            av = av[av["subcat_sed"].isin(cats_validas)].copy()
        av = av[av.geometry.notnull() & av.is_valid]
        av = av.to_crs(epsg=32614)
        av["geometry"] = av.geometry.centroid
        av_pts = np.array([(g.x, g.y) for g in av.geometry])
    except Exception:
        av_pts = np.empty((0, 2))

    df["dist_metro_m"]   = nearest_distance(metro_pts,  coords_utm2)
    df["dist_metrobus_m"]= nearest_distance(mb_pts,     coords_utm2)
    df["dist_tren_m"]    = nearest_distance(tren_pts,   coords_utm2)
    df["dist_trole_m"]   = nearest_distance(trole_pts,  coords_utm2)
    df["dist_cable_m"]   = nearest_distance(cable_pts,  coords_utm2)
    df["dist_area_verde_recreativa_m"] = nearest_distance(av_pts, coords_utm2)

    # Comercio
    try:
        com = gpd.read_file(PATH_COMERCIO).to_crs(epsg=32614)
        com_pts = np.array([(g.centroid.x, g.centroid.y)
                            for g in com.geometry])
        df["dist_comercio_m"] = nearest_distance(com_pts, coords_utm2)
    except Exception:
        df["dist_comercio_m"] = 5000

    # ----------------------------------------------------------
    # CICLOVÍAS (líneas y puntos para distancia/densidad)
    # ----------------------------------------------------------
    try:
        _ciclo_raw = gpd.read_file(PATH_CICLOVIAS)
        if _ciclo_raw.crs is None:
            _ciclo_raw = _ciclo_raw.set_crs("EPSG:4326")
        elif _ciclo_raw.crs.to_epsg() != 4326:
            _ciclo_raw = _ciclo_raw.to_crs("EPSG:4326")
        # WGS84 para Folium
        ciclovias_wgs = _ciclo_raw[_ciclo_raw.geometry.notnull() & ~_ciclo_raw.geometry.is_empty].copy()
        # UTM para cálculos de distancia/densidad
        ciclovias_utm = ciclovias_wgs.to_crs("EPSG:32614")
        ciclovias_pts = line_to_points(ciclovias_utm, interval_m=100)
        df["dist_ciclovia_m"] = nearest_distance(ciclovias_pts, coords_utm2)
        df["densidad_ciclovia_15m"] = density_proxy_from_points(coords_utm2, ciclovias_pts, radio=1200)
    except Exception as e:
        st.warning(f"No se pudo cargar el archivo de ciclovías: {e}")
        ciclovias_wgs = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        ciclovias_utm = gpd.GeoDataFrame(geometry=[], crs="EPSG:32614")
        ciclovias_pts = np.empty((0, 2))
        df["dist_ciclovia_m"] = 5000.0
        df["densidad_ciclovia_15m"] = 0.0

    # ----------------------------------------------------------
    # 6. SEGMENTACIÓN DE MERCADO
    # ----------------------------------------------------------
    lux_cut = df["price_m2_raw"].quantile(0.90)
    df["is_luxury"] = (df["price_m2_raw"] >= lux_cut).astype(int)

    # ----------------------------------------------------------
    # 7. ENTRENAMIENTO POR SEGMENTO Y CLÚSTER
    # ----------------------------------------------------------
    modelos  = {}
    imputers = {}
    kmeans_models    = {}
    cluster_scalers  = {}
    cluster_features_by_seg = {}

    for seg_label, seg_mask in [("estandar", df["is_luxury"] == 0),
                                 ("lujo",     df["is_luxury"] == 1)]:

        data_seg = df[seg_mask].copy()
        q_low, q_high = data_seg["price_m2_raw"].quantile([0.05, 0.95])
        data_seg = data_seg[
            (data_seg["price_m2_raw"] > q_low) &
            (data_seg["price_m2_raw"] < q_high)
        ].copy()

        if len(data_seg) < 100:
            continue

        if seg_label == "estandar":
            cluster_feat = ["latitud", "longitud"]
            enriched = StandardScaler().fit_transform(
                data_seg[cluster_feat]
            )
        else:
            cluster_feat = [
                "latitud", "longitud",
                "gentrification_index", "score_15min"
            ]
            enriched = StandardScaler().fit_transform(
                data_seg[cluster_feat]
            )

        cscaler = StandardScaler()
        Xc = cscaler.fit_transform(data_seg[cluster_feat])
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        data_seg["cluster"] = km.fit_predict(Xc)

        kmeans_models[seg_label]    = km
        cluster_scalers[seg_label]  = cscaler
        cluster_features_by_seg[seg_label] = cluster_feat

        coords_seg = data_seg[["longitud", "latitud"]].values

        for c in sorted(data_seg["cluster"].unique()):
            d_c = data_seg[data_seg["cluster"] == c].copy()
            if len(d_c) < 50:
                continue

            coords_c = d_c[["longitud", "latitud"]].values
            X_base   = d_c[static_features].copy()
            y        = np.log1p(d_c["price_m2_raw"])

            lag   = compute_spatial_lag(
                coords_c, d_c["price_m2_raw"].values, coords_c
            )
            nvp   = compute_local_neighbor_price(
                coords_c, d_c["price_m2_raw"].values, coords_c
            )

            X_base["spatial_lag_price"]   = lag
            X_base["precio_vecinal_local"] = nvp
            X_base["lag_x_area"] = (
                X_base["spatial_lag_price"] *
                np.log1p(X_base["area"])
            )

            feat_disponibles = [
                f for f in features_all if f in X_base.columns
            ]

            imputador = X_base[feat_disponibles].median()
            X_clean = X_base[feat_disponibles].fillna(imputador).fillna(0)

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
            imputers[(seg_label, c)] = (imputador, feat_disponibles)

    # ----------------------------------------------------------
    # 8. REFERENCIA ESPACIAL POR COLONIA
    # ----------------------------------------------------------
    cols_ref = (
        static_features +
        [
            "latitud", "longitud",
            "gentrif_local_presion", "gentrif_map_score",
            "dist_area_verde_m",
            "dist_area_verde_recreativa_m",
            "dist_salud_m", "dist_escuela_m",
            "dist_metro_m", "dist_metrobus_m",
            "dist_tren_m", "dist_trole_m", "dist_cable_m",
            "dist_comercio_m",
            "dist_ciclovia_m", "densidad_ciclovia_15m",
            "densidad_parques_15m", "acceso_salud_15m",
            "acceso_educacion_15m",
            "cat_mean_valor_suelo_200m", "cat_mean_vus_200m",
            "cat_mean_antiguedad_200m",
            "cat_mean_ratio_construccion_200m",
            "cat_density_predios_200m",
        ]
    )
    cols_ref = list(dict.fromkeys(
        [c for c in cols_ref if c in df.columns]
    ))

    referencia_espacial = (
        df.groupby("colonia_key")[cols_ref]
        .mean()
        .to_dict("index")
    )

    price_m2_col = (
        df.groupby("colonia_key")["price_m2_raw"]
        .median()
        .to_dict()
    )   

    alcaldia_colonias    = (
        df.groupby("alcaldia_real")["colonia_real"]
        .unique()
        .apply(lambda x: sorted(list(x)))
        .to_dict()
    )
    alcaldias_disponibles = sorted(alcaldia_colonias.keys())

    price_m2_by_colonia = (
        df.groupby("colonia_key")["price_m2_raw"]
        .mean()
        .to_dict()
    )
    coords_by_colonia = (
        df.groupby("colonia_key")[["longitud", "latitud"]]
        .mean()
        .to_dict("index")
    )

    # Polígonos de colonias
    colonias_geo = colonias[
        ["colonia_clean", "alcaldia_clean", "geometry"]
    ].copy()

    colonias_geo = colonias_geo[
        colonias_geo.geometry.notnull()
    ].copy()

    colonias_geo["colonia_key"] = (
        colonias_geo["alcaldia_clean"].astype(str)
        + "||" +
        colonias_geo["colonia_clean"].astype(str)
    )

    colonias_geo = (
        colonias_geo
        .dissolve(by="colonia_key")
        .reset_index()
    )

    poligonos_colonias = {
        row["colonia_key"]: row["geometry"].__geo_interface__
        for _, row in colonias_geo.iterrows()
    }
    
    # ----------------------------------------------------------
    # 9. CARGA DE CAPAS DE PUNTOS PARA VISUALIZACIÓN
    # ----------------------------------------------------------
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

    metro_pts,    metro_nombres    = _cargar_capa_con_nombres(PATH_METRO_EST,          "NOMBRE",    "Estación Metro")
    mb_pts,       mb_nombres       = _cargar_capa_con_nombres(PATH_METROBUS_EST,        "NOMBRE",    "Estación Metrobús")
    tren_pts,     tren_nombres     = _cargar_capa_con_nombres(PATH_TREN_EST,            "NOMBRE",    "Estación Tren Ligero")
    trole_pts,    trole_nombres    = _cargar_capa_con_nombres(PATH_TROLE_PARADAS,       "NOMBRE",    "Parada Trolebús")
    cable_pts,    cable_nombres    = _cargar_capa_con_nombres(PATH_CABLE_EST,           "NOMBRE",    "Estación Cablebús")
    av_pts,       av_nombres       = _cargar_capa_con_nombres(PATH_AREAS_VERDES,        "nombre",    "Área Verde")
    salud1_pts,   salud1_nombres   = _cargar_capa_con_nombres(PATH_SALUD_BASE,          "nombre",    "Unidad de Salud")
    salud2_pts,   salud2_nombres   = _cargar_capa_con_nombres(PATH_HOSPITALES_PUBLICOS, "NOMBRE_D_3","Hospital")
    com_pts,      com_nombres      = _cargar_capa_con_nombres(PATH_COMERCIO,            "INSTITUCIO","Centro Comercial")
    esc_priv_pts, esc_priv_nombres = _cargar_capa_con_nombres(PATH_ESCUELAS_PRIVADAS,   "nombre",    "Escuela Privada")
    esc_pub_pts,  esc_pub_nombres  = _cargar_capa_con_nombres(PATH_ESCUELAS_PUBLICAS,   "nombre",    "Escuela Pública")

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

    # Capa de líneas (ciclovías)
    capas_lineas = {
        "ciclovias_utm": ciclovias_utm if 'ciclovias_utm' in locals() else gpd.GeoDataFrame(geometry=[], crs="EPSG:32614"),
        "ciclovias_wgs": ciclovias_wgs if 'ciclovias_wgs' in locals() else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"),
    }

    return (
        modelos, imputers,
        kmeans_models, cluster_scalers, cluster_features_by_seg,
        static_features, features_all,
        referencia_espacial, price_m2_col,
        alcaldias_disponibles, alcaldia_colonias,
        price_m2_by_colonia, coords_by_colonia,
        capas_servicios, capas_lineas,
        poligonos_colonias,
    )

def density_proxy_from_points(coords_utm, pts, radio=1200):
    """Conteo de puntos dentro del radio para cada coordenada."""
    if len(pts) == 0:
        return np.zeros(len(coords_utm))
    tree = cKDTree(pts)
    counts = []
    for xy in coords_utm:
        count = len(tree.query_ball_point(xy, r=radio))
        counts.append(count)
    return np.array(counts)

# ============================================================
# INSTANCIACIÓN
# ============================================================
(
    MODELS, IMPUTERS,
    KMEANS_MODELS, CLUSTER_SCALERS, CLUSTER_FEAT_BY_SEG,
    STATIC_FEATURES, FEATURES_ALL,
    REF_ESPACIAL, PRICE_M2_COL,
    ALCALDIAS_DISPONIBLES, ALCALDIA_COLONIAS,
    PRICE_M2_BY_COLONIA, COORDS_BY_COLONIA,
    CAPAS_SERVICIOS, CAPAS_LINEAS,
    POLIGONOS_COLONIAS,
) = preparar_sistema()

# ============================================================
# INFERENCIA
# ============================================================
def predict_price(area, rooms, baths, parking, ant, colonia):
    datos_colonia = REF_ESPACIAL.get(
        colonia, list(REF_ESPACIAL.values())[0]
    )

    median_m2 = PRICE_M2_COL.get(colonia, 0)
    lux_global_cut = np.percentile(
        list(PRICE_M2_COL.values()), 90
    )
    seg_label = "lujo" if median_m2 >= lux_global_cut else "estandar"

    input_data = datos_colonia.copy()
    ant_log = np.log1p(ant)

    input_data.update({
        "area":           area,
        "rooms":          rooms,
        "bathrooms":      baths,
        "parking_spaces": parking,
        "antiguedad":     ant_log,
        "area_x_marginalidad": np.log1p(area) * datos_colonia.get("marginalidad_score", 3),
        "area_X_gentrif":      area * datos_colonia.get("gentrification_index", 0.5),
        "gentrif_x_metro":     datos_colonia.get("gentrification_index", 0.5) *
                                datos_colonia.get("density_metro", 0),
        "15min_X_gentrif":     datos_colonia.get("score_15min", 0) *
                                datos_colonia.get("gentrification_index", 0.5),
    })

    X_df = pd.DataFrame([input_data])

    cluster_feat = CLUSTER_FEAT_BY_SEG.get(seg_label, ["latitud", "longitud"])
    cscaler      = CLUSTER_SCALERS.get(seg_label)
    km           = KMEANS_MODELS.get(seg_label)

    if cscaler is None or km is None:
        seg_label = list({k[0] for k in MODELS.keys()})[0]
        cluster_feat = CLUSTER_FEAT_BY_SEG[seg_label]
        cscaler      = CLUSTER_SCALERS[seg_label]
        km           = KMEANS_MODELS[seg_label]

    X_cluster = X_df[cluster_feat].fillna(0)
    c_id      = km.predict(cscaler.transform(X_cluster))[0]

    if (seg_label, c_id) not in MODELS:
        candidates = [k for k in MODELS if k[0] == seg_label]
        if not candidates:
            candidates = list(MODELS.keys())
        seg_label, c_id = candidates[0]

    model = MODELS[(seg_label, c_id)]
    imputador, feat_disponibles = IMPUTERS[(seg_label, c_id)]

    pm2_colonia = PRICE_M2_COL.get(colonia, median_m2)
    X_df["spatial_lag_price"]   = np.log1p(pm2_colonia)
    X_df["precio_vecinal_local"] = pm2_colonia
    X_df["lag_x_area"] = (
        X_df["spatial_lag_price"] * np.log1p(area)
    )

    X_pred = (
        X_df[feat_disponibles]
        .fillna(imputador)
        .fillna(0)
    )

    # Clip Z-scores a ±5σ para evitar precios absurdos por datos atípicos o colonias
    # sin representación en entrenamiento con variables catastrales fuera de rango.
    _scaler = model.named_steps["scaler"]
    _Xz = np.clip(_scaler.transform(X_pred), -5, 5)
    X_pred_safe = pd.DataFrame(_scaler.inverse_transform(_Xz),
                               columns=X_pred.columns, index=X_pred.index)
    log_pred = float(np.clip(model.predict(X_pred_safe)[0],
                             np.log1p(200), np.log1p(200_000)))
    precio_m2_pred = np.expm1(log_pred)
    precio_total   = precio_m2_pred * area

    intercepto_log   = model.named_steps["enet"].intercept_
    precio_base_m2   = np.expm1(intercepto_log)
    precio_base_total = precio_base_m2 * area

    return (precio_total, precio_m2_pred, seg_label, c_id,
            model, X_pred_safe, datos_colonia, precio_base_total, precio_base_m2)

def recalcular_distancias_desde_punto(lat, lon):
    """
    Dado un punto (lat, lon) en WGS84, recalcula todas las distancias
    a servicios usando los arrays de puntos ya cargados en CAPAS_SERVICIOS
    y también a ciclovías (usando puntos a lo largo de las líneas).
    """
    from scipy.spatial import cKDTree

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

    # Ciclovías: generar puntos a lo largo de las líneas (cada 100 m) para distancia
    ciclovias_utm = CAPAS_LINEAS.get("ciclovias_utm", gpd.GeoDataFrame(geometry=[], crs="EPSG:32614"))
    if not ciclovias_utm.empty:
        ciclovias_pts = line_to_points(ciclovias_utm, interval_m=100)
    else:
        ciclovias_pts = np.empty((0, 2))

    d_metro   = _dist_min(metro_pts)
    d_mb      = _dist_min(mb_pts)
    d_tren    = _dist_min(tren_pts)
    d_trole   = _dist_min(trole_pts)
    d_cable   = _dist_min(cable_pts)
    d_verde   = _dist_min(av_pts)
    d_salud   = _dist_min(salud_pts)
    d_escuela = _dist_min(esc_todos)
    d_comercio = _dist_min(com_pts)
    d_ciclovia = _dist_min(ciclovias_pts)

    n_salud   = _conteo_radio(salud_pts)
    n_escuela = _conteo_radio(esc_todos)
    n_parques = _conteo_radio(av_pts)
    n_ciclovia = _conteo_radio(ciclovias_pts)  # proxy de densidad

    prox_parque = 1 if d_verde   <= 800  else 0
    prox_salud  = 1 if d_salud   <= 1000 else 0
    prox_escuela= 1 if d_escuela <= 1000 else 0
    prox_ciclovia = 1 if d_ciclovia <= 500 else 0

    score = np.mean([
        min(n_salud   / 6,  1.0),
        min(n_escuela / 40, 1.0),
        min(n_parques / 10, 1.0),
        prox_parque,
        prox_salud,
        prox_escuela,
        prox_ciclovia,
    ])

    return {
        "dist_metro_m":            d_metro,
        "dist_metrobus_m":         d_mb,
        "dist_tren_m":             d_tren,
        "dist_trole_m":            d_trole,
        "dist_cable_m":            d_cable,
        "dist_area_verde_m":       d_verde,
        "dist_area_verde_recreativa_m": d_verde,
        "dist_salud_m":            d_salud,
        "dist_escuela_m":          d_escuela,
        "dist_comercio_m":         d_comercio,
        "dist_ciclovia_m":         d_ciclovia,
        "acceso_salud_15m":        float(n_salud),
        "acceso_educacion_15m":    float(n_escuela),
        "densidad_parques_15m":    float(n_parques),
        "densidad_ciclovia_15m":   float(n_ciclovia),
        "prox_parque":             float(prox_parque),
        "prox_salud":              float(prox_salud),
        "prox_escuela":            float(prox_escuela),
        "prox_ciclovia":           float(prox_ciclovia),
        "score_15min":             float(score),
    }

def predict_price_con_punto(area, rooms, baths, parking, ant, colonia, lat, lon):
    datos_colonia = REF_ESPACIAL.get(colonia, list(REF_ESPACIAL.values())[0])

    median_m2      = PRICE_M2_COL.get(colonia, 0)
    lux_global_cut = np.percentile(list(PRICE_M2_COL.values()), 90)
    seg_label      = "lujo" if median_m2 >= lux_global_cut else "estandar"

    input_data = datos_colonia.copy()
    ant_log    = np.log1p(ant)

    distancias_nuevas = recalcular_distancias_desde_punto(lat, lon)
    input_data.update(distancias_nuevas)

    gentrif = datos_colonia.get("gentrification_index", 0.5)

    # Recalcular density_metro desde la distancia real al nuevo punto
    d_metro_nuevo = distancias_nuevas.get("dist_metro_m", 9999)
    density_metro_nuevo = max(0.0, 1.0 - d_metro_nuevo / 1200.0) if d_metro_nuevo < 1200 else 0.0

    input_data.update({
        "area":            area,
        "rooms":           rooms,
        "bathrooms":       baths,
        "parking_spaces":  parking,
        "antiguedad":      ant_log,
        # Coordenadas reales del punto (no del centroide) para clúster
        "latitud":         lat,
        "longitud":        lon,
        "area_x_marginalidad": np.log1p(area) * datos_colonia.get("marginalidad_score", 3),
        "area_X_gentrif":      area * gentrif,
        "gentrif_x_metro":     gentrif * density_metro_nuevo,
        "15min_X_gentrif":     distancias_nuevas["score_15min"] * gentrif,
    })

    X_df = pd.DataFrame([input_data])

    cluster_feat = CLUSTER_FEAT_BY_SEG.get(seg_label, ["latitud", "longitud"])
    cscaler      = CLUSTER_SCALERS.get(seg_label)
    km           = KMEANS_MODELS.get(seg_label)

    if cscaler is None or km is None:
        seg_label    = list({k[0] for k in MODELS.keys()})[0]
        cluster_feat = CLUSTER_FEAT_BY_SEG[seg_label]
        cscaler      = CLUSTER_SCALERS[seg_label]
        km           = KMEANS_MODELS[seg_label]

    X_cluster = X_df[cluster_feat].fillna(0)
    c_id      = km.predict(cscaler.transform(X_cluster))[0]

    if (seg_label, c_id) not in MODELS:
        candidates     = [k for k in MODELS if k[0] == seg_label] or list(MODELS.keys())
        seg_label, c_id = candidates[0]

    model                   = MODELS[(seg_label, c_id)]
    imputador, feat_disp    = IMPUTERS[(seg_label, c_id)]

    pm2_colonia = PRICE_M2_COL.get(colonia, median_m2)

    # Spatial lag real: precio ponderado por distancia a colonias vecinas.
    # IMPORTANTE: convertir coords a UTM (metros) antes de calcular distancias
    # para evitar pesos inflados al usar grados decimales (~0.001) que generan
    # precios absurdos (billones de MXN).
    _col_lons = np.array([v.get("longitud", lon) for v in REF_ESPACIAL.values()])
    _col_lats = np.array([v.get("latitud",  lat) for v in REF_ESPACIAL.values()])
    _col_prices = np.array([
        PRICE_M2_BY_COLONIA.get(k, pm2_colonia)
        for k in REF_ESPACIAL.keys()
    ])

    if len(_col_lons) >= 5:
        from scipy.spatial import cKDTree as _cKDT
        # Proyectar colonias a UTM para calcular distancias en metros
        _gdf_cols = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(_col_lons, _col_lats), crs="EPSG:4326"
        ).to_crs("EPSG:32614")
        _col_coords_utm = np.array([(g.x, g.y) for g in _gdf_cols.geometry])

        # Proyectar el punto objetivo a UTM
        _gdf_pt = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326"
        ).to_crs("EPSG:32614")
        _px_utm = _gdf_pt.geometry.iloc[0].x
        _py_utm = _gdf_pt.geometry.iloc[0].y

        _tree_col = _cKDT(_col_coords_utm)
        _k_lag = min(10, len(_col_coords_utm))
        _d_lag, _ix_lag = _tree_col.query([[_px_utm, _py_utm]], k=_k_lag)

        # Distancias en metros → pesos inversos razonables
        _w_lag = 1.0 / (_d_lag[0] + 1.0)   # +1 metro evita división por cero
        _spatial_lag_raw = float(np.average(_col_prices[_ix_lag[0]], weights=_w_lag))
    else:
        _spatial_lag_raw = pm2_colonia

    # Sanitizar: el lag no debe alejarse más de 10× del precio mediano de la colonia
    # para atrapar cualquier valor corrupto remanente.
    _p_ref = pm2_colonia if pm2_colonia > 0 else float(np.median(list(PRICE_M2_COL.values())))
    _spatial_lag_raw = float(np.clip(_spatial_lag_raw, _p_ref * 0.1, _p_ref * 10.0))

    X_df["spatial_lag_price"]    = np.log1p(_spatial_lag_raw)
    X_df["precio_vecinal_local"] = _spatial_lag_raw
    X_df["lag_x_area"]           = X_df["spatial_lag_price"] * np.log1p(area)

    X_pred         = X_df[feat_disp].fillna(imputador).fillna(0)

    # Clip Z-scores a ±5σ para evitar precios absurdos por datos atípicos o nueva ubicación
    _scaler2 = model.named_steps["scaler"]
    _Xz2 = np.clip(_scaler2.transform(X_pred), -5, 5)
    X_pred_safe = pd.DataFrame(_scaler2.inverse_transform(_Xz2),
                               columns=X_pred.columns, index=X_pred.index)
    log_pred       = float(np.clip(model.predict(X_pred_safe)[0],
                                   np.log1p(200), np.log1p(200_000)))
    precio_m2_pred = np.expm1(log_pred)
    precio_total   = precio_m2_pred * area

    intercepto_log    = model.named_steps["enet"].intercept_
    precio_base_m2    = np.expm1(intercepto_log)
    precio_base_total = precio_base_m2 * area

    return (
        precio_total, precio_m2_pred, seg_label, c_id,
        model, X_pred_safe, datos_colonia,
        precio_base_total, precio_base_m2,
        distancias_nuevas,
    )

# ============================================================
# CONTROLES DE ENTRADA (SIDEBAR)
# ============================================================
with st.sidebar:
    st.header("🏢 Parámetros del Inmueble")
    alcaldia_sel       = st.selectbox("Alcaldía", ALCALDIAS_DISPONIBLES)
    colonias_filtradas = ALCALDIA_COLONIAS.get(alcaldia_sel, [])
    
    colonia_sel = st.selectbox(
    "Colonia", colonias_filtradas,
    key=f"colonia_{alcaldia_sel}"
)
    
    # Llave única colonia + alcaldía
    colonia_key_sel = (
        f"{alcaldia_sel}||{colonia_sel}"
    )

    area       = st.slider("Área Habitable (m²)", 25, 500, 120)
    rooms      = st.number_input("Recámaras", min_value=1, max_value=10, value=3)
    baths      = st.number_input("Baños Completos", min_value=1.0,
                                 max_value=10.0, value=2.0, step=0.5)
    parking    = st.number_input("Espacios de Estacionamiento",
                                 min_value=0, max_value=10, value=1)
    antiguedad = st.slider("Antigüedad de la Estructura (Años)", 0, 80, 10)

    st.markdown("---")
    st.caption("📍 Usa la herramienta de marcador (🔵) en el mapa para elegir un punto específico dentro de tu colonia y confirma con el botón.")

# ============================================================
# PREDICCIÓN
# ============================================================
_key_lat        = f"marker_lat_{colonia_key_sel}"
_key_lon        = f"marker_lon_{colonia_key_sel}"
_key_confirmado = f"punto_confirmado_{colonia_key_sel}"
_key_dist       = f"distancias_punto_{colonia_key_sel}"

lat_base = float(
    REF_ESPACIAL.get(colonia_key_sel, {})
    .get("latitud", 19.43)
)

lon_base = float(
    REF_ESPACIAL.get(colonia_key_sel, {})
    .get("longitud", -99.13)
)

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

# Predicción base (siempre se ejecuta primero como fallback)
(
    precio, precio_m2, seg_label, cluster_id,
    modelo_fit, x_input, datos_colonia,
    precio_base_total, precio_base_m2
) = predict_price(area, rooms, baths, parking, antiguedad, colonia_key_sel)

# Si el marcador se movió del centroide (confirmado o no), recalcular con el punto real.
# Esto corrige el bug de doble clic: antes solo se recalculaba si punto_confirmado=True.
_punto_movido = (
    abs(lat_marcador - lat_base) > 0.000001
    or abs(lon_marcador - lon_base) > 0.000001
)

if _punto_movido or punto_confirmado:
    (
        precio, precio_m2, seg_label, cluster_id,
        modelo_fit, x_input, datos_colonia,
        precio_base_total, precio_base_m2,
        distancias_confirmadas,
    ) = predict_price_con_punto(
        area, rooms, baths, parking, antiguedad, colonia_key_sel,
        lat_marcador, lon_marcador
    )
    st.session_state[_key_dist] = distancias_confirmadas

datos_entorno = datos_colonia.copy()
if distancias_confirmadas:
    datos_entorno.update(distancias_confirmadas)

# ============================================================
# CUADRO DE MANDO PRINCIPAL
# ============================================================
st.title("Sistema de Valuación Inmobiliaria CDMX")

tab_valuador, tab_atlas = st.tabs([
    "🏢 Valuador Interactivo",
    "🗺️ Estudio Multidimensional del Mercado Inmobiliario"
])

with tab_valuador:

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio Estimado Comercial",  f"${precio:,.0f} MXN")
    c2.metric("Valor Unitario (m²)",        f"${precio_m2:,.0f} MXN/m²")
    c3.metric("Segmento / Clúster",
            f"{'Lujo' if seg_label == 'lujo' else 'Estándar'} · C{cluster_id}")
    c4.metric("Índice de Gentrificación",
            round(datos_colonia.get("gentrification_index", 0), 3))

    def fmt_dist(m):
        return f"{m:.0f} m" if m < 1000 else f"{m/1000:.1f} km"

    # ============================================================
    # PANEL DE ANÁLISIS DE ENTORNO URBANO
    # ============================================================
    with st.expander("🌎 Análisis de Entorno Urbano y Socioespacial",
                    expanded=True):

        # --- Bloque 1: Ciudad de 15 Minutos ---
        st.markdown("### 🏬 1. Accesibilidad a Escala Humana (Ciudad de 15 Minutos)")

        with st.container(border=True):
            score_15   = float(datos_entorno.get("score_15min", 0))
            c_score, c_prog = st.columns([1, 3])
            with c_score:
                st.metric("🎯 Índice General 15 Min", round(score_15, 3))
            with c_prog:
                st.markdown("<div style='padding-top:10px;'></div>",
                            unsafe_allow_html=True)
                st.progress(min(score_15, 1.0))
                if score_15 >= 0.7:   st.success("Alta cobertura peatonal.")
                elif score_15 >= 0.4: st.info("Cobertura urbana funcional.")
                else:                  st.warning("Dependencia de vehículo.")

            st.markdown("---")

            fila1_col1, fila1_col2 = st.columns(2)
            fila2_col1, fila2_col2 = st.columns(2)

            with fila1_col1:
                acceso_salud = float(datos_entorno.get("acceso_salud_15m", 0))
                st.metric("🏥 Servicios de Salud Cercanos",
                        f"{acceso_salud:.0f} Unidades")
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
                parque     = float(datos_entorno.get("dist_area_verde_recreativa_m",
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
                if comercio_dist <= 600:   st.success("Abasto local inmediato.")
                elif comercio_dist <= 1500: st.info("Proximidad comercial aceptable.")
                else:                       st.warning("Distancia prolongada a zonas comerciales.")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Bloque 2: Transporte ---
        st.markdown("### 🚊 2. Conectividad y Red de Transporte Estructurado")

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
                    st.metric("🚇 STC Metro",  fmt_dist(d_metro))
                    st.markdown("<div style='padding-top:15px;'></div>",
                                unsafe_allow_html=True)
                    st.metric("🚌 Metrobús",   fmt_dist(d_mb))
                with sub_c2:
                    st.metric("🚊 Tren Ligero", fmt_dist(d_tren))
                    st.markdown("<div style='padding-top:15px;'></div>",
                                unsafe_allow_html=True)
                    st.metric("🚎 Trolebús",   fmt_dist(d_trole))
                with sub_c3:
                    st.metric("🚠 Cablebús",   fmt_dist(d_cable))
                    st.markdown("<div style='padding-top:15px;'></div>",
                                unsafe_allow_html=True)
                    if ciclovias >= 5.0:        status_bici = "Excelente"
                    elif ciclovias >= 1.5:      status_bici = "Funcional"
                    elif ciclovias > 0:         status_bici = "Escasa"
                    else:                       status_bici = "Ninguna"
                    st.metric("🚲 Infraestructura Ciclista", status_bici,
                            help=f"{ciclovias:.1f} segmentos en radio 15 min.")

            with c2_derecha:
                min_dist = min([d_metro, d_mb, d_tren, d_trole])
                st.markdown(
                    "<b style='font-size:14px;color:#808495;'>Evaluación Multimodal</b>",
                    unsafe_allow_html=True
                )
                if min_dist <= 500:
                    st.success("🟢 **Conectividad Excelente**")
                elif min_dist <= 1000:
                    st.info("🔵 **Accesibilidad Media**")
                else:
                    st.warning("🟡 **Cobertura Restringida**")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Bloque 3: Gentrificación ---
        st.markdown("### 🏙️ 3. Dinámica de Transformación Socioespacial")

        with st.container(border=True):
            c3_1, c3_2, c3_3 = st.columns(3)

            gentrif_macro = float(datos_colonia.get("gentrification_index", 0))
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
                st.metric("🌆 Vulnerabilidad al Cambio Socioespacial",
                        round(gentrif_map, 3))
                st.progress(min(gentrif_map, 1.0))
                if gentrif_map >= 0.7:
                    st.warning("Proceso acelerado de transformación.")
                elif gentrif_map >= 0.4:
                    st.info("Área en transición urbana activa.")
                else:
                    st.success("Estabilidad sociodemográfica relativa.")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Bloque 4: Variables Catastrales ---
        st.markdown("### 🗂️ 4. Contexto Catastral del Entorno (Radio 200 m)")

        with st.container(border=True):

            # Fila 1: precio oficial del suelo | valor catastral del terreno
            cr1_col1, cr1_col2 = st.columns(2)

            with cr1_col1:
                vus = datos_colonia.get("cat_mean_vus_200m", None)
                if vus is not None:
                    st.metric(
                        "📐 Precio oficial del suelo (m²)",
                        f"${vus:,.0f} MXN/m²",
                        help="Valor unitario de suelo según el Catastro CDMX 2021."
                    )

            with cr1_col2:
                vs = datos_colonia.get("cat_mean_valor_suelo_200m", None)
                if vs is not None:
                    st.metric(
                        "🏦 Valor catastral del terreno",
                        f"${vs:,.0f} MXN",
                        help="Valor total de suelo registrado por predio en el catastro."
                    )

            st.markdown("---")

            # Fila 2: ¿qué tan construida está la zona? | ¿cuántos predios hay cerca?
            cr2_col1, cr2_col2 = st.columns(2)

            with cr2_col1:
                ratio = datos_colonia.get("cat_mean_ratio_construccion_200m", None)
                if ratio is not None:
                    if ratio >= 3.0:
                        ratio_label = "Zona muy edificada"
                        ratio_desc  = "Predominan edificios de varios pisos."
                        ratio_color = "success"
                    elif ratio >= 1.5:
                        ratio_label = "Zona medianamente edificada"
                        ratio_desc  = "Mezcla de casas y edificios de baja altura."
                        ratio_color = "info"
                    else:
                        ratio_label = "Zona poco edificada"
                        ratio_desc  = "Predominan casas o predios con poca construcción."
                        ratio_color = "warning"

                    st.metric(
                        "🏗️ ¿Qué tan construida está la zona?",
                        f"{ratio:.1f}× el terreno",
                        help=(
                            "Indica cuántos metros cuadrados de construcción existen "
                            "por cada metro cuadrado de terreno en los predios cercanos. "
                            "Ejemplo: un valor de 2.5 significa que se construyó 2.5 veces "
                            "la superficie del terreno (típico de edificios de 3 o más pisos). "
                            "Un valor cercano a 1 indica casas de un solo nivel."
                        )
                    )
                    if ratio_color == "success":
                        st.success(f"🏢 {ratio_label} — {ratio_desc}")
                    elif ratio_color == "info":
                        st.info(f"🏠 {ratio_label} — {ratio_desc}")
                    else:
                        st.warning(f"🌿 {ratio_label} — {ratio_desc}")

            with cr2_col2:
                dens = datos_colonia.get("cat_density_predios_200m", None)
                if dens is not None:
                    if dens >= 80:
                        dens_label = "Zona muy fraccionada"
                        dens_desc  = "Muchos predios pequeños, alta densidad urbana."
                        dens_color = "success"
                    elif dens >= 30:
                        dens_label = "Zona consolidada"
                        dens_desc  = "Tejido urbano típico de colonia establecida."
                        dens_color = "info"
                    else:
                        dens_label = "Zona poco parcelada"
                        dens_desc  = "Predios grandes o área en proceso de urbanización."
                        dens_color = "warning"

                    st.metric(
                        "🏘️ ¿Cuántos predios hay cerca?",
                        f"{dens:.0f} predios en 200 m",
                        help=(
                            "Número de predios catastrados dentro de un radio de 200 metros. "
                            "Un número alto indica una zona muy urbanizada con lotes pequeños "
                            "(como el Centro Histórico o Tepito). Un número bajo puede indicar "
                            "predios grandes, zonas industriales o colonias en desarrollo."
                        )
                    )
                    if dens_color == "success":
                        st.success(f"🏙️ {dens_label} — {dens_desc}")
                    elif dens_color == "info":
                        st.info(f"📌 {dens_label} — {dens_desc}")
                    else:
                        st.warning(f"🌱 {dens_label} — {dens_desc}")

            st.markdown("---")

            # Fila 3: antigüedad catastral | precio base del submercado
            cr3_col1, cr3_col2 = st.columns(2)

            with cr3_col1:
                ant_cat = datos_colonia.get("cat_mean_antiguedad_200m", None)
                if ant_cat is not None:
                    st.metric(
                        "📅 Antigüedad promedio de las construcciones",
                        f"{ant_cat:.0f} años",
                        help="Edad promedio de los inmuebles cercanos según el año de construcción catastral."
                    )
                    if ant_cat <= 15:
                        st.success("Zona con construcciones recientes.")
                    elif ant_cat <= 40:
                        st.info("Zona con construcciones de edad media.")
                    else:
                        st.warning("Zona con construcciones antiguas.")

            with cr3_col2:
                st.metric(
                    "📊 Precio base del submercado",
                    f"${precio_base_total:,.0f} MXN",
                    help=(
                        "Precio estimado para una vivienda completamente promedio "
                        "dentro de este clúster de mercado, calculado a partir del "
                        "intercepto del modelo ElasticNet. "
                        "No incluye ningún ajuste por atributos específicos del inmueble. "
                        "Sirve como referencia: si el Precio Estimado es mayor, "
                        "la vivienda tiene características que la valorizan por encima "
                        "del promedio de su submercado."
                    )
                )
                delta_vs_base = precio - precio_base_total
                if delta_vs_base >= 0:
                    st.success(f"▲ +${delta_vs_base:,.0f} MXN sobre el precio base")
                else:
                    st.warning(f"▼ ${delta_vs_base:,.0f} MXN bajo el precio base")

            st.caption("Fuente: Catastro CDMX 2021 · Radio de análisis: 200 m.")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- Bloque 5: Estructura urbana y desempeño ---
        st.markdown("### 📊 5. Estructura Urbana y Desempeño del Mercado")

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
                if listing >= 3:   st.success("Mercado con alta rotación.")
                elif listing >= 1.5: st.info("Actividad de mercado regular.")
                else:              st.warning("Baja tasa de transacciones.")
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
    # INTERPRETABILIDAD HEDÓNICA
    # ============================================================
    st.markdown("### 💰 Elasticidad y Aportación de Atributos al Precio")

    NOMBRES_VARIABLES = {
        "rooms":                        "Recámaras adicionales",
        "bathrooms":                    "Baños completos",
        "parking_spaces":               "Espacios de estacionamiento",
        "area":                         "Metros cuadrados de construcción",
        "antiguedad":                   "Años de antigüedad (log)",
        "dist_metro_m":                 "Proximidad STC Metro (log)",
        "density_metro":                "Concentración local de accesos al Metro",
        "dist_metrobus_m":              "Proximidad Metrobús (log)",
        "density_metrobus":             "Concentración de accesos al Metrobús",
        "dist_tren_m":                  "Proximidad Tren Ligero (log)",
        "density_tren":                 "Densidad Tren Ligero",
        "dist_trole_m":                 "Proximidad Trolebús (log)",
        "density_trole":                "Densidad Trolebús",
        "dist_cable_m":                 "Proximidad Cablebús (log)",
        "density_cable":                "Densidad Cablebús",
        "dist_ciclovia_m":              "Distancia a red de ciclovías (log)",
        "densidad_ciclovia_15m":        "Densidad infraestructura ciclista",
        "dist_area_verde_m":            "Distancia a área verde (log)",
        "densidad_parques_15m":         "Densidad de parques en 15 min",
        "dist_salud_m":                 "Distancia a equipamiento médico (log)",
        "acceso_salud_15m":             "Acceso a salud en 15 min",
        "dist_escuela_m":               "Distancia a centros educativos (log)",
        "acceso_educacion_15m":         "Acceso a educación en 15 min",
        "comercio_density":             "Intensidad de comercios regionales (log)",
        "marginalidad_score":           "Grado de marginación social",
        "score_15min":                  "Puntaje 'Ciudad de 15 Minutos'",
        "dist_subcenter_log":           "Distancia a distritos de empleo (log)",
        "gentrification_index":         "Nivel de gentrificación (alcaldía)",
        "listing_density_log":          "Presión del inventario inmobiliario",
        "area_X_gentrif":               "Tamaño en zonas con alta plusvalía",
        "15min_X_gentrif":              "Accesibilidad × Gentrificación",
        "verde_marginalidad_ratio":      "Parques / Marginación (log)",
        "educacion_marginalidad_ratio":  "Educación / Marginación (log)",
        "uso_mixto":                    "Zona de uso de suelo mixto (HM)",
        "pct_migrantes_turistas":       "% Migrantes / turistas (INEGI, log)",
        "spatial_lag_price":            "Lag espacial de precios vecinos",
        "precio_vecinal_local":         "Precio vecinal local promedio",
        "lag_x_area":                   "Lag espacial × Área",
        "cat_mean_valor_suelo_200m":        "Valor catastral suelo promedio 200 m",
        "cat_mean_vus_200m":                "Valor unitario suelo catastral 200 m",
        "cat_mean_antiguedad_200m":         "Antigüedad catastral promedio 200 m",
        "cat_std_valor_suelo_200m":         "Varianza valor suelo catastral 200 m",
        "cat_mean_ratio_construccion_200m": "Ratio construcción/terreno catastral",
        "cat_density_predios_200m":         "Densidad predios catastro 200 m",
    }

    enet_model  = modelo_fit.named_steps["enet"]
    coeficientes = enet_model.coef_
    feat_names   = x_input.columns.tolist()

    impactos_pesos = []
    for idx, var in enumerate(feat_names):
        val  = float(x_input[var].values[0]) if var in x_input.columns else 0.0
        beta = coeficientes[idx] if idx < len(coeficientes) else 0.0
        ref  = x_input[var].values[0] if x_input[var].values[0] != 0 else 1
        impacto = precio * (np.exp(beta * (val / ref)) - 1)
        if var in ["area", "area_X_gentrif"] and abs(impacto) > precio:
            impacto = np.sign(impacto) * (precio * 0.4)
        impactos_pesos.append(impacto)

    datos_impacto = pd.DataFrame({
        "Variable_Interna": feat_names,
        "Impacto_Pesos":    impactos_pesos
    })
    datos_impacto["Característica"] = (
        datos_impacto["Variable_Interna"]
        .map(NOMBRES_VARIABLES)
        .fillna(datos_impacto["Variable_Interna"])
    )

    UMBRAL_IMPACTO = 15000
    datos_filtrados = datos_impacto[
        datos_impacto["Impacto_Pesos"].abs() >= UMBRAL_IMPACTO
    ].copy()
    datos_filtrados = datos_filtrados.sort_values("Impacto_Pesos", ascending=True)

    if not datos_filtrados.empty:
        colores = [
            "#EF553B" if v < 0 else "#00CC96"
            for v in datos_filtrados["Impacto_Pesos"]
        ]
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
                "Impacto absoluto &gt; $15k MXN</span>"
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
            "No se identificaron variables con impacto superior a $15,000 MXN "
            "en esta configuración de vivienda."
        )

    # ============================================================
    # MAPA DE CONTEXTO
    # ============================================================
    import json

    RADIO_M = 1200

    CAPAS_CONFIG = {
        "metro":       {"label": "🚇 STC Metro",     "color": "blue",      "icono": "train",          "prefix": "fa", "max_puntos": 30},
        "metrobus":    {"label": "🚌 Metrobús",       "color": "red",       "icono": "bus",            "prefix": "fa", "max_puntos": 30},
        "tren":        {"label": "🚊 Tren Ligero",    "color": "cadetblue", "icono": "subway",         "prefix": "fa", "max_puntos": 20},
        "trolebus":    {"label": "🚎 Trolebús",       "color": "purple",    "icono": "bolt",           "prefix": "fa", "max_puntos": 30},
        "cablebus":    {"label": "🚠 Cablebús",       "color": "darkblue",  "icono": "cloud",          "prefix": "fa", "max_puntos": 20},
        "parques":     {"label": "🌳 Áreas Verdes",   "color": "green",     "icono": "leaf",           "prefix": "fa", "max_puntos": 25},
        "salud":       {"label": "🏥 Salud",          "color": "darkred",   "icono": "plus-square",    "prefix": "fa", "max_puntos": 20},
        "comercio":    {"label": "🛍️ Comercio",      "color": "orange",    "icono": "shopping-cart",  "prefix": "fa", "max_puntos": 15},
        "esc_privada": {"label": "🏫 Esc. Privadas", "color": "beige",     "icono": "graduation-cap", "prefix": "fa", "max_puntos": 20},
        "esc_publica": {"label": "🏫 Esc. Públicas", "color": "darkgreen", "icono": "graduation-cap", "prefix": "fa", "max_puntos": 20},
        "ciclovias":   {"label": "🚲 Ciclovías",      "color": "teal",      "icono": "road",           "prefix": "fa", "max_puntos": 20},
    }

    _COLOR_HEX = {
        "blue": "#1A73E8", "red": "#E53935", "cadetblue": "#5F9EA0",
        "purple": "#7B1FA2", "darkblue": "#1565C0", "green": "#388E3C",
        "darkred": "#B71C1C", "orange": "#F57C00", "beige": "#A1887F",
        "darkgreen": "#1B5E20", "teal": "#008080",
    }

    def _latlon_a_utm_mapa(lat, lon):
        gdf = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([lon], [lat]), crs="EPSG:4326"
        ).to_crs("EPSG:32614")
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
    poligono_js = json.dumps(poligono_colonia_geojson) if poligono_colonia_geojson else "null"

    # Construcción del mapa
    m = folium.Map(
        location=[lat_marcador, lon_marcador],
        zoom_start=15,
        tiles="OpenStreetMap",
    )

    # Polígono de la colonia
    if poligono_colonia_geojson:
        folium.GeoJson(
            poligono_colonia_geojson,
            style_function=lambda _: {
                "color":       "#1A73E8",
                "weight":      2.5,
                "fillColor":   "#1A73E8",
                "fillOpacity": 0.06,
                "dashArray":   "6 4",
            },
            tooltip="Límite de la colonia",
        ).add_to(m)

    lat_actual = st.session_state[_key_lat]
    lon_actual = st.session_state[_key_lon]

    # Círculo de radio (actualizado con el punto actual)
    folium.Circle(
        location=[lat_actual, lon_actual],
        radius=RADIO_M,
        color="#E53935",
        weight=2,
        fill=True,
        fill_color="#E53935",
        fill_opacity=0.05,
        tooltip=f"Radio de análisis: {RADIO_M/1000:.1f} km | {'Confirmado' if punto_confirmado else 'Centroide'}",
    ).add_to(m)


    # Marcador fijo que muestra la posición actual confirmada (no arrastrable)
    folium.Marker(
        location=[lat_actual, lon_actual],
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
        tooltip="📍 Posición actual — usa el botón ✏️ (izquierda) para colocar un nuevo marcador",
        popup=folium.Popup(
            f"<b>📍 Punto de análisis</b><br>"
            f"Colonia: <b>{colonia_sel.title()}</b><br>"
            f"<span style='font-size:10px;color:#888'>"
            f"Usa la herramienta de marcador (🔵 en el panel izquierdo)<br>"
            f"para colocar un nuevo punto, luego presiona <b>Confirmar ubicación</b></span>",
            max_width=240,
        ),
        draggable=False,
    ).add_to(m)

    # Plugin Draw: solo marcadores, para capturar nueva posición confiablemente
    # via last_active_drawing con geometría tipo Point
    from folium.plugins import Draw
    Draw(
        draw_options={
            "marker":       True,
            "polyline":     False,
            "polygon":      False,
            "circle":       False,
            "rectangle":    False,
            "circlemarker": False,
        },
        edit_options={"edit": False, "remove": False},
        position="topleft",
    ).add_to(m)

    # Capas de servicios (puntos)
    centro_utm = _latlon_a_utm_mapa(lat_actual, lon_actual)
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
                icon=folium.Icon(
                    color=cfg["color"],
                    icon=cfg["icono"],
                    prefix=cfg["prefix"],
                ),
                tooltip=folium.Tooltip(nombre_punto, sticky=True),
                popup=folium.Popup(
                    f"<b>{cfg['label']}</b><br>{nombre_punto}<br>"
                    f"<span style='font-size:10px;color:#888'>{lt:.5f}, {ln:.5f}</span>",
                    max_width=200,
                ),
            ).add_to(grupo)

        grupo.add_to(m)

    # Capa de líneas: ciclovías
    _cic_utm = CAPAS_LINEAS.get("ciclovias_utm", gpd.GeoDataFrame(geometry=[], crs="EPSG:32614"))
    _cic_wgs = CAPAS_LINEAS.get("ciclovias_wgs", gpd.GeoDataFrame(geometry=[], crs="EPSG:4326"))
    if not _cic_utm.empty:
        _circulo_utm = Point(centro_utm[0], centro_utm[1]).buffer(RADIO_M)
        # Usar mismo índice para ambos GDFs (tienen filas idénticas)
        _mask = _cic_utm.intersects(_circulo_utm)
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
                    style_function=lambda feature: {"color": "#008080", "weight": 3, "opacity": 0.85},
                    tooltip=folium.Tooltip(f"🚲 {nombre}", sticky=True),
                    popup=folium.Popup(f"<b>Ciclovía</b><br>{nombre}", max_width=200),
                ).add_to(grupo_lineas)
            grupo_lineas.add_to(m)
        else:
            resumen_capas["ciclovias"] = 0
    else:
        resumen_capas["ciclovias"] = 0
        
    # LayerControl
    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    # Leyenda
    leyenda_filas = "".join([
        f"<div style='display:flex;align-items:center;margin-bottom:6px;'>"
        f"<span style='background:{_COLOR_HEX[cfg['color']]};width:13px;height:13px;"
        f"border-radius:50%;display:inline-block;margin-right:8px;"
        f"border:1px solid rgba(0,0,0,0.15);flex-shrink:0;'></span>"
        f"<span style='font-size:12px;color:#222;line-height:1.3;'>{cfg['label']}</span></div>"
        for clave, cfg in CAPAS_CONFIG.items()
    ])

    leyenda_html = f"""
    <div style="
        position: fixed;
        bottom: 30px; left: 30px;
        z-index: 9999;
        background: rgba(255,255,255,0.97);
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.18);
        font-family: 'Segoe UI', Arial, sans-serif;
        min-width: 175px;
    ">
        <div style='font-weight:700;font-size:13px;margin-bottom:9px;color:#111;
                    border-bottom:1px solid #eee;padding-bottom:6px;'>
            Servicios en radio 1.2 km
        </div>
        {leyenda_filas}
        <div style='margin-top:8px;padding-top:6px;border-top:1px solid #eee;
                    font-size:10px;color:#666;'>
            🔴 Límite de la colonia
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(leyenda_html))

    # Render en Streamlit
    st.subheader("🗺️ Entorno Urbano — Radio 1.2 km")

    if punto_confirmado:
        st.success(
            f"✅ Punto confirmado en {lat_marcador:.5f}, {lon_marcador:.5f} — "
            f"distancias y precio recalculados."
        )
    else:
        st.info("📍 Arrastra el ícono 🏠 dentro de la colonia y confirma la ubicación.")

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

    cols_resumen = st.columns(len(resumen_capas))
    for col, (clave, n) in zip(cols_resumen, resumen_capas.items()):
        col.metric(f"{iconos_txt[clave]} {nombres_cortos[clave]}", n)

    # ============================================================
    # RESULTADO DEL MAPA
    # ============================================================

    resultado_mapa = st_folium(
        m,
        use_container_width=True,
        height=540,
        key=f"mapa_{colonia_sel}",
        returned_objects=["last_active_drawing"],  # SOLO dibujo, no clics en otros marcadores
    )

    # ============================================================
    # DETECTAR NUEVO PUNTO COLOCADO CON LA HERRAMIENTA DRAW
    # ============================================================

    from shapely.geometry import Point, shape

    _drawn = resultado_mapa.get("last_active_drawing")

    nuevo_lat, nuevo_lon = None, None

    # last_active_drawing → {"geometry": {"type": "Point", "coordinates": [lon, lat]}}
    if _drawn and isinstance(_drawn, dict):
        _geom = _drawn.get("geometry", {})
        if _geom.get("type") == "Point":
            _coords = _geom.get("coordinates", [])
            if len(_coords) == 2:
                nuevo_lat = _coords[1]
                nuevo_lon = _coords[0]

    if nuevo_lat is not None and nuevo_lon is not None:
        poly_geojson = POLIGONOS_COLONIAS.get(colonia_key_sel)
        _dentro = True
        if poly_geojson is not None:
            from shapely.geometry import Point as _Pt, shape as _shape
            _poly = _shape(poly_geojson)
            _dentro = _poly.contains(_Pt(nuevo_lon, nuevo_lat))

        if _dentro:
            if (
                abs(nuevo_lat - st.session_state[_key_lat]) > 0.000001
                or abs(nuevo_lon - st.session_state[_key_lon]) > 0.000001
            ):
                st.session_state[_key_lat] = nuevo_lat
                st.session_state[_key_lon] = nuevo_lon
                st.session_state[_key_confirmado] = False
                st.rerun()
        else:
            # NO hacer rerun — solo avisar. El marcador se redibujará en la posición
            # válida de session_state en el próximo rerun natural (al confirmar).
            st.warning("⚠️ Posición fuera de la colonia — no se actualizó. Mueve el marcador dentro del límite azul y confirma.")

    # ============================================================
    # POSICIÓN ACTUAL REAL DEL MARCADOR
    # SIEMPRE LEER DESDE SESSION_STATE
    # ============================================================

    lat_actual = st.session_state[_key_lat]
    lon_actual = st.session_state[_key_lon]

    punto_confirmado = st.session_state[_key_confirmado]

    # ============================================================
    # BOTONES
    # ============================================================

    col_btn1, col_btn2, _ = st.columns([2, 2, 4])

    with col_btn1:

        if st.button(
            "✅ Confirmar ubicación",
            type="primary",
            use_container_width=True
        ):
            st.session_state[_key_confirmado] = True
            # Limpiar distancias cacheadas para que se recalculen con el punto actual
            st.session_state[_key_dist] = None
            st.rerun()

    with col_btn2:

        if st.button(
            "↩️ Restablecer centroide",
            use_container_width=True
        ):

            st.session_state[_key_lat] = lat_base
            st.session_state[_key_lon] = lon_base

            st.session_state[_key_confirmado] = False
            st.session_state[_key_dist] = None

            st.rerun()

    # ============================================================
    # TEXTO INFORMATIVO
    # ============================================================

    st.caption(
        f"📐 Radio: {RADIO_M} m · "
        f"Punto: {'confirmado ✅' if punto_confirmado else 'pendiente de confirmar'} · "
        f"{lat_actual:.5f}, {lon_actual:.5f}"
    )

# ============================================================
# PESTAÑA: ESTUDIO MULTIDIMENSIONAL
# ============================================================
with tab_atlas:

    # ── Encabezado ──────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2.5rem 2.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    ">
        <h1 style="
            color: #e2e8f0;
            font-size: 1.8rem;
            font-weight: 800;
            margin: 0 0 0.5rem;
            letter-spacing: -0.5px;
        ">🌆 Estudio Multidimensional del Mercado Inmobiliario</h1>
        <p style="
            color: #94a3b8;
            font-size: 1rem;
            margin: 0;
            line-height: 1.6;
        ">Análisis espacial integral de las dinámicas sociales, económicas y urbanas<br>
        en la Ciudad de México · Elaborado con QGIS y datos abiertos CDMX</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Definición de mapas ──────────────────────────────────
    # Ajusta la ruta base a donde guardes tus PNGs exportados de QGIS
    MAPAS_DIR = os.path.join(PROJECT_ROOT, "..", "outputs", "assets", "mapas_qgis")

    MAPAS_ATLAS = [
        {
            "archivo": "mapa_1_distribucion_del_precio_unitario_m2.png",
            "titulo":  "Precio Real por Metro Cuadrado de Viviendas",
            "emoji":   "💰",
            "descripcion": "Distribución del precio por m² en el mercado de departamentos de la CDMX (rangos de $9k a $148k MXN). Revela los gradientes de valor desde el centro hacia la periferia y la concentración de oferta premium en Benito Juárez, Cuauhtémoc y Miguel Hidalgo.",
            "categoria": "Mercado Inmobiliario",
        },
        {
            "archivo": "mapa_2_densidad_de_oferta_inmobiliaria.png",
            "titulo":  "Hotspot de Oferta Inmobiliaria (Publicaciones de Venta)",
            "emoji":   "🏗️",
            "descripcion": "Estimación de densidad kernel de los anuncios de venta activos. Los hotspots más intensos se concentran en Álvaro Obregón, Benito Juárez y Cuauhtémoc, evidenciando los submercados con mayor actividad transaccional.",
            "categoria": "Mercado Inmobiliario",
        },
        {
            "archivo": "mapa_3_identificacion_de_propiedades_de_lujo.png",
            "titulo":  "Identificación de Propiedades de Lujo",
            "emoji":   "🏆",
            "descripcion": "Clasificación binaria del inventario analizado: vivienda convencional vs. vivienda de lujo (percentil ≥ 90 de precio por m²). La vivienda de lujo se concentra en el corredor Polanco–Lomas–Coyoacán.",
            "categoria": "Mercado Inmobiliario",
        },
        {
            "archivo": "mapa_4_rezago_espacial_de_precios.png",
            "titulo":  "Rezago Espacial de Precios del Modelo",
            "emoji":   "📊",
            "descripcion": "Precio promedio ponderado de los vecinos más cercanos (spatial lag), desde entornos de precios bajos hasta hotspots inmobiliarios. Es el componente dinámico central del modelo hedónico-espacial ElasticNet.",
            "categoria": "Análisis Espacial",
        },
        {
            "archivo": "mapa_5_distribucion_espacial_del_valor_del_suelo_y_rezago_territorial.png",
            "titulo":  "Comparación: Precio por m² vs Rezago Espacial",
            "emoji":   "🗺️",
            "descripcion": "Análisis bivariado que yuxtapone la distribución real del precio por m² (izquierda) con el rezago espacial del modelo (derecha). Permite detectar zonas donde el entorno vecinal presiona al alza o a la baja el valor individual.",
            "categoria": "Análisis Espacial",
        },
        {
            "archivo": "mapa_6_LISA.png",
            "titulo":  "Clústeres Espaciales Significativos del Mercado Inmobiliario (LISA)",
            "emoji":   "🔬",
            "descripcion": "Local Indicators of Spatial Association (Moran's I local). Identifica clústeres HH (hotspot inmobiliario), LL (coldspot), HL (alta presión aislada) y LH (rezago aislado) con significancia estadística.",
            "categoria": "Análisis Espacial",
        },
        {
            "archivo": "mapa_7_el_entorno_catastral_activo.png",
            "titulo":  "El Entorno Catastral Activo",
            "emoji":   "🏛️",
            "descripcion": "Nivel de actividad catastral por predio (desde muy baja hasta muy alta), cruzado con datos del SIGCDMX. La actividad catastral alta en Cuauhtémoc y Benito Juárez correlaciona con los submercados de mayor precio.",
            "categoria": "Catastro",
        },
        {
            "archivo": "mapa_8_indice_proximidad_15_minutos.png",
            "titulo":  "Índice de Accesibilidad Territorial — Enfoque 15 Minutos",
            "emoji":   "🚶",
            "descripcion": "Score compuesto de accesibilidad peatonal que clasifica cada propiedad desde 'Desconectado / Periferia Crítica' hasta 'Entorno 15 Minutos Óptimo'. Las zonas verdes (óptimas) se concentran en Benito Juárez y Cuauhtémoc.",
            "categoria": "Movilidad y Accesibilidad",
        },
        {
            "archivo": "mapa_9_infrastructura_servicios_de_movilidad.png",
            "titulo":  "Isócronas en Proximidad a Modos de Transporte Colectivo",
            "emoji":   "🚇",
            "descripcion": "Cuatro submapas (Metro, Metrobús, Trolebús, Cablebús) con isócronas de proximidad desde < 5 min caminando hasta entorno desconectado. El Metro y Metrobús muestran la mayor cobertura territorial.",
            "categoria": "Movilidad y Accesibilidad",
        },
        {
            "archivo": "mapa_10_indice_de_gentrificacion.png",
            "titulo":  "Índice de Gentrificación",
            "emoji":   "🏙️",
            "descripcion": "Presencia del proceso de gentrificación por propiedad, desde 'Sin presión' hasta 'Gentrificación crítica'. La presión más intensa se concentra en Benito Juárez, Roma–Condesa y corredores de Coyoacán.",
            "categoria": "Dinámicas Sociales",
        },
        {
            "archivo": "mapa_11_analisis_bivariante_gentrificacion_vs_marginalidad.png",
            "titulo":  "Análisis Bivariante: Gentrificación vs Marginalidad",
            "emoji":   "📉",
            "descripcion": "Matriz 3×3 que cruza nivel de gentrificación (Baja/Media/Alta) con nivel de marginalidad (Baja/Media/Alta). Las zonas con Alta Gentrificación + Alta Marginalidad representan los territorios de mayor vulnerabilidad socioespacial.",
            "categoria": "Dinámicas Sociales",
        },
        {
            "archivo": "mapa_12_presencia_de_migrantes.png",
            "titulo":  "Presión por Población Migrante (≥ 5 años de residencia)",
            "emoji":   "🌍",
            "descripcion": "Clasificación de propiedades según la concentración de población migrante de larga estadía (Censo INEGI 2020): desde presencia insignificante hasta presión internacional crítica. Variable incorporada como factor de demanda en el modelo.",
            "categoria": "Dinámicas Sociales",
        },
        {
            "archivo": "mapa_13_segmentacion_de_submercados_kmeans.png",
            "titulo":  "Clústeres Espaciales de Mercado (K-Means)",
            "emoji":   "🧩",
            "descripcion": "Segmentación mediante K-Means en 6 submercados diferenciados: 3 clústeres estándar (azules) y 3 de lujo (cálidos). Base del sistema de predicción segmentada — cada submercado entrena su propio modelo ElasticNet.",
            "categoria": "Mercado Inmobiliario",
        },
        {
            "archivo": "mapa_14_mapa_error_relativo.png",
            "titulo":  "Análisis de Error Relativo — Nivel Colonia",
            "emoji":   "⚠️",
            "descripcion": "Error porcentual del modelo por colonia: subestimación (azul), error bajo, sobreestimación moderada/alta/extrema (rojo). Las zonas de mayor error coinciden con colonias de transición o con escasa muestra de entrenamiento.",
            "categoria": "Validación del Modelo",
        },
        {
            "archivo": "mapa_15_precio_predicho_del_modelo.png",
            "titulo":  "Promedio de Precios Estimados del Modelo — Nivel Colonia",
            "emoji":   "🎯",
            "descripcion": "Precio total estimado por el modelo ElasticNet espacial agregado a nivel colonia (rangos de 1.4 M a 40 M MXN). Lomas de Chapultepec, Polanco y Bosques concentran las estimaciones más altas.",
            "categoria": "Validación del Modelo",
        },
        {
            "archivo": "mapa_16_gentrificacion_nivel_colonia.png",
            "titulo":  "Índice de Gentrificación — Nivel Colonia",
            "emoji":   "🏘️",
            "descripcion": "Coroplético del índice de gentrificación desagregado por colonia: desde muy baja presión hasta consolidación. Iztaccihuatl, Morales/Polanco y las colonias Roma–Condesa muestran los procesos más avanzados.",
            "categoria": "Dinámicas Sociales",
        },
        {
            "archivo": "mapa_17_residuos_espaciales.png",
            "titulo":  "Análisis de Residuos Espaciales: Diagnóstico de Error de Predicción",
            "emoji":   "📐",
            "descripcion": "Residuos del modelo clasificados en cinco categorías (Sobreestimación Significativa → Rango Óptimo → Subestimación Significativa). La distribución espacial de los residuos orienta mejoras futuras del modelo.",
            "categoria": "Validación del Modelo",
        },
        {
            "archivo": "mapa_18_gap_valor_plusvalia.png",
            "titulo":  "Análisis Bivariado: Brecha de Valor y Accesibilidad a Servicios",
            "emoji":   "📈",
            "descripcion": "Interacción entre la brecha de valor (precio mercado vs. modelo base) y la cobertura de servicios. Identifica Nodos Consolidados (alto valor + alta cobertura), Oportunidades Emergentes y Zonas de Menor Interés.",
            "categoria": "Análisis Espacial",
        },
        {
            "archivo": "mapa_19_isocronas_servicios.png",
            "titulo":  "Isócronas de Proximidad 15 Minutos",
            "emoji":   "⏱️",
            "descripcion": "Grado de servicios por proximidad peatonal: desde Déficit de Accesibilidad (morado oscuro) hasta Accesibilidad Excelente (amarillo). Roma Norte, Del Valle y Narvarte destacan como los entornos más completos.",
            "categoria": "Movilidad y Accesibilidad",
        },
        {
            "archivo": "mapa_20_accesibilidad_critica.png",
            "titulo":  "Mapa de Desiertos Urbanos",
            "emoji":   "🚨",
            "descripcion": "Déficit de accesibilidad urbana por colonia, desde muy baja saturación hasta muy alta saturación de servicios. Las colonias en rojo oscuro concentran la mayor exclusión urbana, con impacto directo en la depresión del valor inmobiliario.",
            "categoria": "Movilidad y Accesibilidad",
        },
    ]

    # ── Filtro por categoría ─────────────────────────────────
    categorias = sorted(set(m["categoria"] for m in MAPAS_ATLAS))
    cat_sel = st.multiselect(
        "🔍 Filtrar por categoría temática",
        options=categorias,
        default=categorias,
        help="Selecciona una o más categorías para filtrar los mapas mostrados."
    )

    mapas_filtrados = [m for m in MAPAS_ATLAS if m["categoria"] in cat_sel]

    if not mapas_filtrados:
        st.warning("Selecciona al menos una categoría para ver los mapas.")
    else:
        # ── Colores por categoría ────────────────────────────
        COLORES_CAT = {
            "Mercado Inmobiliario":     ("#1e3a5f", "#3b82f6"),
            "Análisis Espacial":        ("#1a3a2a", "#22c55e"),
            "Catastro":                 ("#3a2a1a", "#f97316"),
            "Movilidad y Accesibilidad":("#2a1a3a", "#a855f7"),
            "Dinámicas Sociales":       ("#3a1a1a", "#ef4444"),
            "Validación del Modelo":    ("#1a2a3a", "#06b6d4"),
        }

        # ── Galería en cuadrícula de 2 columnas ─────────────
        cols_galeria = st.columns(2, gap="large")

        for i, mapa in enumerate(mapas_filtrados):
            col = cols_galeria[i % 2]
            ruta_img = os.path.join(MAPAS_DIR, mapa["archivo"])

            bg_dark, accent = COLORES_CAT.get(mapa["categoria"], ("#1e293b", "#64748b"))

            with col:
                # Badge de categoría + título
                st.markdown(f"""
                <div style="
                    background: {bg_dark};
                    border: 1px solid {accent}40;
                    border-left: 4px solid {accent};
                    border-radius: 12px;
                    padding: 1rem 1.2rem 0.7rem;
                    margin-bottom: 0.5rem;
                ">
                    <span style="
                        background: {accent}25;
                        color: {accent};
                        font-size: 0.7rem;
                        font-weight: 700;
                        letter-spacing: 0.08em;
                        padding: 2px 10px;
                        border-radius: 20px;
                        text-transform: uppercase;
                    ">{mapa['categoria']}</span>
                    <h3 style="
                        color: #f1f5f9;
                        font-size: 1.05rem;
                        font-weight: 700;
                        margin: 0.5rem 0 0.3rem;
                    ">{mapa['emoji']} {mapa['titulo']}</h3>
                    <p style="
                        color: #94a3b8;
                        font-size: 0.85rem;
                        margin: 0;
                        line-height: 1.5;
                    ">{mapa['descripcion']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Imagen
                if os.path.exists(ruta_img):
                    st.image(
                        ruta_img,
                        use_container_width=True,
                        caption=f"Mapa {i+1} de {len(mapas_filtrados)} · {mapa['titulo']}"
                    )
                else:
                    st.markdown(f"""
                    <div style="
                        background: #1e293b;
                        border: 2px dashed #334155;
                        border-radius: 8px;
                        padding: 3rem 1rem;
                        text-align: center;
                        color: #64748b;
                        font-size: 0.9rem;
                        margin-bottom: 1rem;
                    ">
                        📁 <code>{mapa['archivo']}</code><br>
                        <small>Coloca el PNG exportado de QGIS en<br>
                        <code>assets/mapas_qgis/</code></small>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='margin-bottom:1.5rem;'></div>",
                            unsafe_allow_html=True)

        # ── Pie de sección ───────────────────────────────────
        st.markdown("---")
        st.caption(
            f"📊 {len(mapas_filtrados)} mapas mostrados · "
            f"Fuentes: INEGI, ADIP CDMX, STC Metro, SEMOVI · "
            f"Elaboración propia con QGIS y Python"
        )