# =============================================================================
# TRABAJO TERMINAL
# Pipeline de Limpieza, Parseo y Geocodificación — Rentas CDMX
#
# Autores: Kevin Jesús González Sosa, José Manuel Torres Gutiérrez
# Institución: Escuela Superior de Cómputo
# Fecha: Mayo, 2026
#
# Descripción:
#   Lee los dos datasets crudos de scraping (Lamudi y MercadoLibre),
#   parsea los campos desde texto libre, unifica el esquema al estándar
#   del dataset_geocodificado.csv e imputa/geocodifica coordenadas mediante
#   Nominatim (OSM) con fallback a centroides de alcaldía.
#
# Salida:
#   data/processed/dataset_geocodificado_rentas.csv
#   (mismo esquema que dataset_geocodificado.csv + columna "renta_mensual")
# =============================================================================

import re
import time
import warnings

import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from sklearn.experimental import enable_iterative_imputer   # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# RUTAS 
# ──────────────────────────────────────────────────────────────────────────────
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(PROJECT_ROOT, "..", "..", "data")

PATH_ML     = os.path.join(DATA_PATH, "raw", "rentas", "datos_raw_rentas_cdmx.csv")
PATH_LAM    = os.path.join(DATA_PATH, "raw", "rentas", "rentas_cdmx_lamudi.csv")
PATH_OUT    = os.path.join(DATA_PATH, "processed", "dataset_geocodificado_rentas.csv")

os.makedirs(os.path.dirname(PATH_OUT), exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
# Geocodificación: cuántas peticiones hacer antes de confiar en el caché.
# Nominatim requiere ≥1 s entre llamadas; aumenta GEOCODE_DELAY si recibes errores.
GEOCODE_DELAY     = 1.2   # segundos entre peticiones
GEOCODE_MAX_TRIES = 3     # reintentos por dirección

# Bbox de CDMX para validar coordenadas devueltas por el geocoder
LAT_MIN, LAT_MAX = 19.04, 19.60
LON_MIN, LON_MAX = -99.37, -98.94

# ──────────────────────────────────────────────────────────────────────────────
# CENTROIDES DE ALCALDÍA (fallback cuando el geocoder falla)
# ──────────────────────────────────────────────────────────────────────────────
CENTROIDES_ALCALDIA = {
    "álvaro obregón":          (19.3603, -99.2040),
    "azcapotzalco":            (19.4864, -99.1853),
    "benito juárez":           (19.3984, -99.1586),
    "coyoacán":                (19.3467, -99.1617),
    "cuajimalpa de morelos":   (19.3597, -99.2980),
    "cuauhtémoc":              (19.4326, -99.1332),
    "gustavo a. madero":       (19.4966, -99.1162),
    "iztacalco":               (19.3951, -99.0972),
    "iztapalapa":              (19.3552, -99.0603),
    "la magdalena contreras":  (19.3109, -99.2297),
    "miguel hidalgo":          (19.4133, -99.1930),
    "milpa alta":              (19.1920, -98.9910),
    "tláhuac":                 (19.2857, -99.0072),
    "tlalpan":                 (19.2961, -99.1733),
    "venustiano carranza":     (19.4278, -99.0876),
    "xochimilco":              (19.2648, -99.1062),
    "cuajimalpa":              (19.3597, -99.2980),
    "la magdalena":            (19.3109, -99.2297),
    "alvaro obregon":          (19.3603, -99.2040),
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(txt: str) -> str:
    """Normaliza cadenas: minúsculas, sin acentos, sin espacios extremos."""
    if pd.isna(txt):
        return ""
    t = str(txt).lower().strip()
    t = t.translate(str.maketrans("áéíóúü", "aeiouu"))
    return t


def parse_precio(raw: str) -> float | None:
    """Extrae precio numérico de cadenas como 'MXN47,000' o '47000'."""
    if pd.isna(raw):
        return None
    raw = str(raw).replace(",", "").replace("MXN", "").replace("$", "").strip()
    m = re.search(r"[\d]+(?:\.\d+)?", raw)
    return float(m.group()) if m else None


def parse_datos_brutos(texto: str) -> dict:
    """
    Parsea el campo 'datos_brutos' del dataset de MercadoLibre.

    Ejemplo de entrada:
      "PUBLICADO ESTA SEMANA | Departamento en renta | Renta Depto Condesa |
       MXN | 47,000 | 2 recámaras | 2 baños | 107 m² construidos |
       Roma Norte, Cuauhtémoc, Distrito Federal | Destacado"
    """
    out = {
        "precio":     None,
        "recamaras":  None,
        "banos":      None,
        "area":       None,
        "ubicacion":  None,
        "colonia":    None,
        "alcaldia":   None,
    }

    if pd.isna(texto):
        return out

    partes = [p.strip() for p in str(texto).split("|")]

    for p in partes:
        pl = p.lower()

        # Precio
        if re.search(r"\d[\d,]+", p) and ("mxn" in pl or "$" in pl
                                           or any(q in pl for q in ["000", "500"])):
            val = parse_precio(p)
            if val and 1_000 < val < 500_000:
                out["precio"] = val

        # Recámaras
        m = re.search(r"(\d+)\s*rec[aá]mara", pl)
        if m:
            out["recamaras"] = int(m.group(1))

        # Baños
        m = re.search(r"(\d+)\s*ba[ñn]o", pl)
        if m:
            out["banos"] = int(m.group(1))

        # Área
        m = re.search(r"(\d+(?:\.\d+)?)\s*m[²2]", pl)
        if m:
            out["area"] = float(m.group(1))

    # Ubicación: última parte antes de "Destacado" / "Premium"
    for p in reversed(partes):
        pl = p.lower()
        if any(skip in pl for skip in ["destacado", "premium", "publicado",
                                        "departamento", "renta", "mxn"]):
            continue
        if "," in p:
            partes_ub = [x.strip() for x in p.split(",")]
            out["colonia"]   = partes_ub[0] if len(partes_ub) > 0 else None
            out["alcaldia"]  = partes_ub[1] if len(partes_ub) > 1 else None
            out["ubicacion"] = p
            break

    return out


# ──────────────────────────────────────────────────────────────────────────────
# 1. CARGA Y PARSEO
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 80)
print("1/5  CARGA Y PARSEO DE DATOS CRUDOS")
print("=" * 80)

# ---- MercadoLibre ----
df_ml = pd.read_csv(PATH_ML)
print(f"  MercadoLibre crudo: {len(df_ml)} registros")

parsed_rows = [parse_datos_brutos(row) for row in df_ml["datos_brutos"]]
df_ml_parsed = pd.DataFrame(parsed_rows)

# El campo precio crudo puede tener valor; úsalo como respaldo
precio_crudo = df_ml["precio"].apply(parse_precio)
df_ml_parsed["precio"] = df_ml_parsed["precio"].combine_first(precio_crudo)

df_ml_parsed["source"]       = "mercadolibre_rentas"
df_ml_parsed["url"]          = df_ml["url"]

# ---- Lamudi ----
df_lam = pd.read_csv(PATH_LAM)
print(f"  Lamudi crudo: {len(df_lam)} registros")

# Mapeo de columnas Lamudi → esquema unificado
df_lam_parsed = pd.DataFrame({
    "precio":     df_lam["Renta_MXN"].apply(pd.to_numeric, errors="coerce"),
    "recamaras":  df_lam["Recamaras"].apply(pd.to_numeric, errors="coerce"),
    "banos":      df_lam["Banos"].apply(pd.to_numeric, errors="coerce"),
    "area":       df_lam["Area_m2"].apply(pd.to_numeric, errors="coerce"),
    "ubicacion":  df_lam["Ubicacion"],
    "colonia":    df_lam["Ubicacion"].str.split(",").str[0].str.strip(),
    "alcaldia":   df_lam["Alcaldia"],
    "parking_spaces": df_lam["Estacionamientos"].apply(pd.to_numeric, errors="coerce"),
    "source":     "lamudi_rentas",
    "url":        df_lam["URL"],
})

# ── Unificar ──
df = pd.concat([df_ml_parsed, df_lam_parsed], ignore_index=True)
print(f"  Total unificado (antes de limpieza): {len(df)} registros")


# ──────────────────────────────────────────────────────────────────────────────
# 2. LIMPIEZA Y ESTANDARIZACIÓN
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("2/5  LIMPIEZA Y ESTANDARIZACIÓN")
print("=" * 80)

# --- Precio ---
# Solo rentas razonables para CDMX (mínimo $3 000, máximo $200 000 MXN/mes)
df["renta_mensual"] = pd.to_numeric(df["precio"], errors="coerce")
n_antes = len(df)
df = df[(df["renta_mensual"] >= 3_000) & (df["renta_mensual"] <= 200_000)].copy()
print(f"  Registros tras filtro de precio: {len(df)}  (eliminados: {n_antes - len(df)})")

# --- Área ---
df["area"] = pd.to_numeric(df["area"], errors="coerce")
# Área mínima razonable para un depto en CDMX: 20 m²
df = df[df["area"].isna() | (df["area"] >= 20)].copy()

# --- Recámaras / baños ---
for col in ["recamaras", "banos"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 10)

# --- Estacionamiento ---
if "parking_spaces" not in df.columns:
    df["parking_spaces"] = np.nan
df["parking_spaces"] = pd.to_numeric(df["parking_spaces"], errors="coerce").clip(0, 10)

# --- Alcaldía: normalizar texto ---
MAPEO_ALCALDIA = {
    "cuauhtemoc":                     "Cuauhtémoc",
    "cuauhtémoc":                     "Cuauhtémoc",
    "miguel hidalgo":                 "Miguel Hidalgo",
    "benito juarez":                  "Benito Juárez",
    "benito juárez":                  "Benito Juárez",
    "alvaro obregon":                 "Álvaro Obregón",
    "álvaro obregón":                 "Álvaro Obregón",
    "coyoacan":                       "Coyoacán",
    "coyoacán":                       "Coyoacán",
    "tlalpan":                        "Tlalpan",
    "iztapalapa":                     "Iztapalapa",
    "iztacalco":                      "Iztacalco",
    "gustavo a. madero":              "Gustavo A. Madero",
    "gustavo a madero":               "Gustavo A. Madero",
    "azcapotzalco":                   "Azcapotzalco",
    "venustiano carranza":            "Venustiano Carranza",
    "xochimilco":                     "Xochimilco",
    "tlahuac":                        "Tláhuac",
    "tláhuac":                        "Tláhuac",
    "milpa alta":                     "Milpa Alta",
    "la magdalena contreras":         "La Magdalena Contreras",
    "magdalena contreras":            "La Magdalena Contreras",
    "cuajimalpa de morelos":          "Cuajimalpa de Morelos",
    "cuajimalpa":                     "Cuajimalpa de Morelos",
    "distrito federal":               None,   # eliminamos referencias al DF genérico
    "ciudad de méxico":               None,
}

def normalizar_alcaldia(val):
    if pd.isna(val):
        return None
    key = clean_text(val)
    # Primero búsqueda exacta
    for k, v in MAPEO_ALCALDIA.items():
        if key == clean_text(k):
            return v
    # Búsqueda parcial
    for k, v in MAPEO_ALCALDIA.items():
        if clean_text(k) in key:
            return v
    return val.strip().title()

df["alcaldia"] = df["alcaldia"].apply(normalizar_alcaldia)

# Intentar extraer alcaldía de la ubicación cuando falta
def extraer_alcaldia_de_ubicacion(row):
    if pd.notna(row["alcaldia"]):
        return row["alcaldia"]
    ub = str(row.get("ubicacion", ""))
    for k, v in MAPEO_ALCALDIA.items():
        if clean_text(k) in clean_text(ub):
            return v
    return None

df["alcaldia"] = df.apply(extraer_alcaldia_de_ubicacion, axis=1)

# --- Colonia / municipio ---
df["colonia_o_municipio"] = df["colonia"].fillna("").str.strip()

# --- Renombrar para que coincida con el esquema objetivo ---
df = df.rename(columns={
    "recamaras": "rooms",
    "banos":     "bathrooms",
    "area":      "area",
})

# Columna 'title' (título del anuncio)
df["title"]    = "Departamento en Renta"

# 'antiguedad' — no disponible en scraping, se imputará después
df["antiguedad"] = np.nan

# Eliminar duplicados por URL
df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
print(f"  Registros tras deduplicación: {len(df)}")

# Valores negativos / cero en área ya filtrados; rellenamos NaN
# mediante imputación multivariada (Sección 4)

print(f"\n  Distribución de nulos antes de imputación:")
cols_check = ["renta_mensual", "rooms", "bathrooms", "parking_spaces", "area", "alcaldia"]
for c in cols_check:
    pct = df[c].isna().mean() * 100
    print(f"    {c:<20} {pct:5.1f}% nulos")


# ──────────────────────────────────────────────────────────────────────────────
# 3. IMPUTACIÓN MULTIVARIADA DE VARIABLES NUMÉRICAS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("3/5  IMPUTACIÓN MULTIVARIADA (IterativeImputer — MICE)")
print("=" * 80)

# Codificamos alcaldía para usarla como feature en la imputación
le = LabelEncoder()
alcaldia_notnull = df["alcaldia"].fillna("Desconocida")
df["_alc_enc"] = le.fit_transform(alcaldia_notnull)

COLS_IMP = ["renta_mensual", "rooms", "bathrooms", "parking_spaces", "area", "_alc_enc"]

# Solo filas con al menos renta_mensual (el target) para guiar la imputación
mask_tiene_precio = df["renta_mensual"].notna()

imputer = IterativeImputer(
    max_iter=10,
    random_state=42,
    min_value=0,
    imputation_order="ascending",
)
imputer.fit(df.loc[mask_tiene_precio, COLS_IMP])

df_imp = df.copy()
df_imp[COLS_IMP] = imputer.transform(df[COLS_IMP])

# Redondear variables discretas
for col in ["rooms", "bathrooms", "parking_spaces"]:
    df_imp[col] = df_imp[col].round().clip(0, 10).astype(int)

df_imp["area"]          = df_imp["area"].round(1).clip(20, 1_000)
df_imp["renta_mensual"] = df_imp["renta_mensual"].round(-2)  # redondeo a centenas

# Imputamos antiguedad con la mediana del dataset de ventas (≈ 30 años)
# como referencia inicial; el modelo puede refinar esto con catastro.
df_imp["antiguedad"] = df_imp["antiguedad"].fillna(30).round(0)

df_imp.drop(columns=["_alc_enc"], inplace=True)

print(f"  Imputación completada.")
print(f"  Nulos residuales en columnas numéricas:")
for c in ["renta_mensual", "rooms", "bathrooms", "parking_spaces", "area"]:
    print(f"    {c:<20} {df_imp[c].isna().sum()} nulos")


# ──────────────────────────────────────────────────────────────────────────────
# 4. GEOCODIFICACIÓN
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("4/5  GEOCODIFICACIÓN")
print("=" * 80)

# Inicializar columnas
df_imp["latitud"]  = np.nan
df_imp["longitud"] = np.nan

geolocator = Nominatim(user_agent="tt_escom_rentas_cdmx_2026", timeout=10)

def coordenadas_centroide(alcaldia: str):
    """Devuelve las coordenadas del centroide de la alcaldía como fallback."""
    key = clean_text(str(alcaldia))
    return CENTROIDES_ALCALDIA.get(key, (19.4326, -99.1332))  # default: Centro CDMX


def geocodificar(ubicacion: str, colonia: str, alcaldia: str) -> tuple[float, float]:
    """
    Intenta geocodificar con Nominatim usando la dirección más completa posible.
    Si falla, usa el centroide de la alcaldía.
    """
    candidatos = []

    # Estrategia 1: colonia + alcaldía + CDMX
    if colonia and alcaldia:
        candidatos.append(f"{colonia}, {alcaldia}, Ciudad de México, México")
    # Estrategia 2: solo alcaldía
    if alcaldia:
        candidatos.append(f"{alcaldia}, Ciudad de México, México")
    # Estrategia 3: ubicación libre
    if ubicacion and len(str(ubicacion)) > 5:
        candidatos.append(f"{ubicacion}, Ciudad de México, México")

    for query in candidatos:
        for intento in range(GEOCODE_MAX_TRIES):
            try:
                time.sleep(GEOCODE_DELAY)
                loc = geolocator.geocode(query)
                if loc:
                    lat, lon = loc.latitude, loc.longitude
                    if LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX:
                        return lat, lon
            except (GeocoderTimedOut, GeocoderUnavailable):
                time.sleep(GEOCODE_DELAY * 2)
            except Exception:
                break

    # Fallback: centroide de alcaldía
    return coordenadas_centroide(alcaldia)


# Cache para evitar geocodificar la misma colonia+alcaldía varias veces
cache_geo: dict[str, tuple[float, float]] = {}

print(f"  Geocodificando {len(df_imp)} registros "
      f"(Nominatim + fallback a centroide)...")

for idx, row in df_imp.iterrows():
    colonia  = str(row.get("colonia_o_municipio", "") or "")
    alcaldia = str(row.get("alcaldia", "") or "")
    ubicacion = str(row.get("ubicacion", "") or "")

    cache_key = f"{clean_text(colonia)}|{clean_text(alcaldia)}"

    if cache_key in cache_geo:
        lat, lon = cache_geo[cache_key]
    else:
        lat, lon = geocodificar(ubicacion, colonia, alcaldia)
        cache_geo[cache_key] = (lat, lon)

    df_imp.at[idx, "latitud"]  = lat
    df_imp.at[idx, "longitud"] = lon

    if (idx + 1) % 100 == 0:
        print(f"    ... {idx + 1}/{len(df_imp)} geocodificados")

print(f"  ✓ Geocodificación completada. "
      f"Cache hits: {len(df_imp) - len(cache_geo)}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. CONSTRUCCIÓN DEL DATASET FINAL (ESQUEMA UNIFICADO)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("5/5  EXPORTACIÓN")
print("=" * 80)

# Selección y orden de columnas idéntico a dataset_geocodificado.csv
# + columna extra "renta_mensual" como variable objetivo
df_final = df_imp[[
    "source",
    "title",
    "ubicacion",          # → "location" en el esquema destino
    "renta_mensual",      # precio mensual de renta (MXN)
    "rooms",
    "bathrooms",
    "parking_spaces",
    "area",
    "antiguedad",
    "alcaldia",
    "colonia_o_municipio",
    "latitud",
    "longitud",
    "url",
]].copy()

# Renombrar para paridad total con el esquema
df_final = df_final.rename(columns={"ubicacion": "location"})

# Quitar filas sin coordenadas útiles (ambas NaN es imposible dado el fallback,
# pero por si acaso)
df_final = df_final.dropna(subset=["latitud", "longitud"]).reset_index(drop=True)

# Estadísticas finales
print(f"\n  Registros exportados: {len(df_final)}")
print(f"  Renta mensual — mediana: ${df_final['renta_mensual'].median():,.0f} MXN")
print(f"  Área          — mediana: {df_final['area'].median():.0f} m²")
print(f"  Recámaras     — moda:    {df_final['rooms'].mode()[0]}")
print(f"\n  Distribución por alcaldía:")
print(df_final["alcaldia"].value_counts().to_string())

df_final.to_csv(PATH_OUT, index=False, encoding="utf-8-sig")
print(f"\n  ✓ Dataset geocodificado guardado en:\n    {PATH_OUT}")

print("\n" + "=" * 80)
print("PIPELINE COMPLETADO")
print("=" * 80)