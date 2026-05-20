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

import streamlit as st
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

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
# Lista completa de static_features (antes del drop por colinealidad)
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

# Variables dinámicas calculadas en el K-Fold de modelo.py
_DYNAMIC_FEATURES = [
    "spatial_lag_price",
    "precio_vecinal_local",
    "lag_x_area",
]

# Variables eliminadas por colinealidad (>0.90) reproduciendo el mismo
# proceso que modelo.py ejecuta sobre el dataset de entrenamiento.
# Se recalculan en preparar_sistema() con los datos reales del CSV.
_COLINEAR_DROP = []   # Se llenará dinámicamente en preparar_sistema()


# ============================================================
# HELPERS
# ============================================================
def clean_text(txt):
    """Normaliza texto: minúsculas, sin acentos, sin espacios extra."""
    if pd.isna(txt):
        return np.nan
    txt = str(txt).lower().strip()
    txt = txt.translate(str.maketrans("áéíóúüñ", "aeiouun"))
    return txt


def nearest_distance(pts, coords):
    """Distancia euclidiana mínima (metros, EPSG:32614)."""
    if len(pts) == 0:
        return np.ones(len(coords)) * 5000
    tree = cKDTree(pts)
    d, _ = tree.query(coords)
    return d


def compute_spatial_lag(train_coords, train_prices, target_coords, k=10):
    """Spatial lag por distancia inversa — mismo algoritmo que modelo.py."""
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
    """Precio vecinal local — mismo algoritmo que modelo.py."""
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
    y por clúster, exactamente igual que modelo.py.

    Las variables catastrales (cat_*) ya vienen calculadas en el CSV.
    No se cargan shapefiles del catastro.
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
    # Verificar qué columnas están disponibles
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
    # 4. CRUCE ESPACIAL CON COLONIAS (para referencia_espacial)
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
    df = df.dropna(subset=["colonia_real", "alcaldia_real"]).copy()
    df = df.reset_index(drop=True)

    # Recalcular coords tras el drop de nulos
    gdf_utm2    = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["longitud"], df["latitud"]),
        crs="EPSG:4326"
    ).to_crs(epsg=32614)
    coords_utm2 = np.array([(p.x, p.y) for p in gdf_utm2.geometry])
    coords_geo2 = df[["longitud", "latitud"]].values

    # ----------------------------------------------------------
    # 5. RECALCULAR DIST A TRANSPORTE (para referencia_espacial actualizada)
    #    Se usa la capa ya disponible — es rápido porque solo son puntos.
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
    # 6. SEGMENTACIÓN DE MERCADO — igual que modelo.py
    #    is_luxury ya viene calculado en el CSV
    # ----------------------------------------------------------
    lux_cut = df["price_m2_raw"].quantile(0.90)
    df["is_luxury"] = (df["price_m2_raw"] >= lux_cut).astype(int)

    # ----------------------------------------------------------
    # 7. ENTRENAMIENTO POR SEGMENTO (estándar/lujo) Y CLÚSTER
    #    Replica run_market_optimized de modelo.py
    # ----------------------------------------------------------
    modelos  = {}   # key: (segmento, cluster_id)
    imputers = {}
    kmeans_models    = {}
    cluster_scalers  = {}
    cluster_features_by_seg = {}

    for seg_label, seg_mask in [("estandar", df["is_luxury"] == 0),
                                 ("lujo",     df["is_luxury"] == 1)]:

        data_seg = df[seg_mask].copy()
        # Filtro de outliers igual que modelo.py
        q_low, q_high = data_seg["price_m2_raw"].quantile([0.05, 0.95])
        data_seg = data_seg[
            (data_seg["price_m2_raw"] > q_low) &
            (data_seg["price_m2_raw"] < q_high)
        ].copy()

        if len(data_seg) < 100:
            continue

        # Cluster features — igual que modelo.py para cada segmento
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

            # Calcular variables dinámicas sobre todos los datos
            # del clúster (sin K-Fold aquí: sirve para inferencia)
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

            # Aseguramos que features_all no incluya columnas ausentes
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
            "dist_area_verde_m",         # distancia usada en modelo.py
            "dist_area_verde_recreativa_m",
            "dist_salud_m", "dist_escuela_m",
            "dist_metro_m", "dist_metrobus_m",
            "dist_tren_m", "dist_trole_m", "dist_cable_m",
            "dist_comercio_m",
            "densidad_parques_15m", "acceso_salud_15m",
            "acceso_educacion_15m",
            # Variables catastrales (para mostrar en interfaz)
            "cat_mean_valor_suelo_200m", "cat_mean_vus_200m",
            "cat_mean_antiguedad_200m",
            "cat_mean_ratio_construccion_200m",
            "cat_density_predios_200m",
        ]
    )
    # Filtrar sólo columnas existentes
    cols_ref = list(dict.fromkeys(
        [c for c in cols_ref if c in df.columns]
    ))

    referencia_espacial = (
        df.groupby("colonia_real")[cols_ref]
        .mean()
        .to_dict("index")
    )

    price_m2_col = (
        df.groupby("colonia_real")["price_m2_raw"]
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

    # Guardar precio_m2_raw por colonia para el spatial lag de inferencia
    price_m2_by_colonia = (
        df.groupby("colonia_real")["price_m2_raw"].mean().to_dict()
    )
    coords_by_colonia = (
        df.groupby("colonia_real")[["longitud", "latitud"]].mean().to_dict("index")
    )

    return (
        modelos, imputers,
        kmeans_models, cluster_scalers, cluster_features_by_seg,
        static_features, features_all,
        referencia_espacial, price_m2_col,
        alcaldias_disponibles, alcaldia_colonias,
        price_m2_by_colonia, coords_by_colonia,
    )


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
) = preparar_sistema()


# ============================================================
# INFERENCIA
# ============================================================
def predict_price(area, rooms, baths, parking, ant, colonia):
    """
    Construye el vector de entrada combinando los parámetros del usuario
    con el perfil urbano de la colonia, determina el segmento (lujo/estándar),
    asigna el clúster y genera la predicción con el modelo correspondiente.
    """
    datos_colonia = REF_ESPACIAL.get(
        colonia, list(REF_ESPACIAL.values())[0]
    )

    # Determinar segmento por precio m² mediano de la colonia
    median_m2 = PRICE_M2_COL.get(colonia, 0)
    precio_m2_raw_estimado = median_m2  # aproximación para clasificar
    lux_global_cut = np.percentile(
        list(PRICE_M2_COL.values()), 90
    )
    seg_label = "lujo" if precio_m2_raw_estimado >= lux_global_cut else "estandar"

    # --- Construir input ---
    input_data = datos_colonia.copy()

    # Aplicar transformaciones log idénticas a modelo.py
    ant_log = np.log1p(ant)

    input_data.update({
        "area":           area,
        "rooms":          rooms,
        "bathrooms":      baths,
        "parking_spaces": parking,
        "antiguedad":     ant_log,
        # Interacciones estáticas (modelo.py las precalcula con log de ant)
        "area_x_marginalidad": np.log1p(area) * datos_colonia.get("marginalidad_score", 3),
        "area_X_gentrif":      area * datos_colonia.get("gentrification_index", 0.5),
        "gentrif_x_metro":     datos_colonia.get("gentrification_index", 0.5) *
                                datos_colonia.get("density_metro", 0),
        "15min_X_gentrif":     datos_colonia.get("score_15min", 0) *
                                datos_colonia.get("gentrification_index", 0.5),
    })

    X_df = pd.DataFrame([input_data])

    # --- Asignar clúster ---
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
        # Fallback al primer modelo disponible del segmento
        candidates = [k for k in MODELS if k[0] == seg_label]
        if not candidates:
            candidates = list(MODELS.keys())
        seg_label, c_id = candidates[0]

    model = MODELS[(seg_label, c_id)]
    imputador, feat_disponibles = IMPUTERS[(seg_label, c_id)]

    # --- Variables dinámicas para inferencia ---
    # Usamos el centroide de la colonia como punto de referencia
    coord_colonia = np.array([
        [datos_colonia.get("longitud", -99.13),
         datos_colonia.get("latitud",   19.43)]
    ])

    # Precio m² vecinal promedio de la colonia como proxy
    pm2_colonia = PRICE_M2_COL.get(colonia, median_m2)
    # Para el lag usamos el precio m² de la colonia misma
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

    log_pred = model.predict(X_pred)[0]
    precio_m2_pred = np.expm1(log_pred)
    precio_total   = precio_m2_pred * area

    # ------------------------------------------------------------------
    # PRECIO BASE DEL SUBMERCADO (intercepto hedónico)
    # Representa el precio de una vivienda "promedio" del clúster,
    # sin ventajas ni penalizaciones de ninguna variable.
    # Fórmula: expm1(intercepto_enet) * area
    # El intercepto del ElasticNet opera en escala log y sobre datos
    # estandarizados, por lo que ya incorpora la media del submercado.
    # ------------------------------------------------------------------
    intercepto_log   = model.named_steps["enet"].intercept_
    precio_base_m2   = np.expm1(intercepto_log)
    precio_base_total = precio_base_m2 * area

    return (precio_total, precio_m2_pred, seg_label, c_id,
            model, X_pred, datos_colonia, precio_base_total, precio_base_m2)


# ============================================================
# CONTROLES DE ENTRADA (SIDEBAR)
# ============================================================
with st.sidebar:
    st.header("🏢 Parámetros del Inmueble")
    alcaldia_sel      = st.selectbox("Alcaldía", ALCALDIAS_DISPONIBLES)
    colonias_filtradas = ALCALDIA_COLONIAS.get(alcaldia_sel, [])
    colonia_sel       = st.selectbox(
        "Colonia", colonias_filtradas, key=f"colonia_{alcaldia_sel}"
    )

    area      = st.slider("Área Habitable (m²)", 25, 500, 120)
    rooms     = st.number_input("Recámaras", min_value=1, max_value=10, value=3)
    baths     = st.number_input("Baños Completos", min_value=1.0,
                                max_value=10.0, value=2.0, step=0.5)
    parking   = st.number_input("Espacios de Estacionamiento",
                                min_value=0, max_value=10, value=1)
    antiguedad = st.slider("Antigüedad de la Estructura (Años)", 0, 80, 10)

# ============================================================
# PREDICCIÓN
# ============================================================
( precio, precio_m2, seg_label, cluster_id,
  modelo_fit, x_input, datos_colonia,
  precio_base_total, precio_base_m2
) = predict_price(area, rooms, baths, parking, antiguedad, colonia_sel)

# ============================================================
# CUADRO DE MANDO PRINCIPAL
# ============================================================
st.title("Sistema de Valuación Inmobiliaria CDMX")

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
        score_15 = float(datos_colonia.get("score_15min", 0))
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
            acceso_salud = float(datos_colonia.get("acceso_salud_15m", 0))
            st.metric("🏥 Servicios de Salud Cercanos",
                      f"{acceso_salud:.0f} Unidades")
            if acceso_salud >= 6:   st.success("Alta densidad médica.")
            elif acceso_salud >= 2: st.info("Cobertura hospitalaria básica.")
            else:                   st.warning("Déficit de equipamiento médico.")
            st.caption("Estructura unificada y depurada.")

        with fila1_col2:
            acceso_edu = float(datos_colonia.get("acceso_educacion_15m", 0))
            st.metric("📚 Planteles Educativos", f"{acceso_edu:.0f} Escuelas")
            if acceso_edu >= 40:   st.success("Alta oferta escolar.")
            elif acceso_edu >= 15: st.info("Infraestructura escolar suficiente.")
            else:                  st.warning("Disponibilidad local limitada.")
            st.caption("Búfer operativo de 1.2 km.")

        with fila2_col1:
            parque       = float(datos_colonia.get("dist_area_verde_recreativa_m",
                                  datos_colonia.get("dist_area_verde_m", 9999)))
            dens_parques = float(datos_colonia.get("densidad_parques_15m", 0))
            st.metric("🌳 Espacio Público Recreativo", f"{parque:.0f} m")
            if parque <= 300:   st.success("Radio óptimo de proximidad.")
            elif parque <= 800: st.info("Distancia media de acceso.")
            else:               st.warning("Déficit de áreas verdes.")
            st.caption(f"Aprox. {dens_parques:.0f} espacios verdes detectados.")

        with fila2_col2:
            comercio_dist = float(datos_colonia.get("dist_comercio_m", 5000))
            st.metric("🛍️ Centros de Abasto / Comercio", fmt_dist(comercio_dist))
            if comercio_dist <= 600:   st.success("Abasto local inmediato.")
            elif comercio_dist <= 1500: st.info("Proximidad comercial aceptable.")
            else:                       st.warning("Distancia prolongada a zonas comerciales.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Bloque 2: Transporte ---
    st.markdown("### 🚊 2. Conectividad y Red de Transporte Estructurado")

    with st.container(border=True):
        c2_izquierda, c2_derecha = st.columns([3, 1.2])

        d_metro  = float(datos_colonia.get("dist_metro_m",    9999))
        d_mb     = float(datos_colonia.get("dist_metrobus_m", 9999))
        d_tren   = float(datos_colonia.get("dist_tren_m",     9999))
        d_trole  = float(datos_colonia.get("dist_trole_m",    9999))
        d_cable  = float(datos_colonia.get("dist_cable_m",    9999))
        ciclovias = float(datos_colonia.get("densidad_ciclovia_15m", 0))

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
    # Catastrales
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
st.subheader("Ubicación Aproximada de la Colonia")
map_df = pd.DataFrame({
    "lat": [datos_colonia.get("latitud",  19.43)],
    "lon": [datos_colonia.get("longitud", -99.13)]
})
st.map(map_df)

# ============================================================
# METADATOS DE SOPORTE
# ============================================================
with st.expander("Metadatos del Sistema (Soporte Técnico)"):
    st.write({
        "Segmento de Mercado":     seg_label.capitalize(),
        "ID Clúster Activo":       int(cluster_id),
        "Centroide Latitud":       float(datos_colonia.get("latitud",  19.43)),
        "Centroide Longitud":      float(datos_colonia.get("longitud", -99.13)),
        "Mediana precio m² zona":  float(PRICE_M2_COL.get(colonia_sel, 0)),
        "Features en el modelo":   len(feat_names),
        "Features estáticas":      len(STATIC_FEATURES),
    })