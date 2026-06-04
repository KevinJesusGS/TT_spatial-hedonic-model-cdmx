# =============================================================================
# ANÁLISIS DE CORRELACIÓN POR ALCALDÍA
# Gentrificación vs Precio por m² (listings individuales)
#
# Script independiente — no requiere ejecutar el modelo principal antes.
# Reconstruye el índice de gentrificación y la tabla comparacion
# directamente desde los archivos fuente.
# =============================================================================

# ======================================================
# RUTAS DEL PROYECTO
# ======================================================

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau

warnings.filterwarnings("ignore")

PROJECT_ROOT = r"C:\Users\chuch\OneDrive\Escritorio\TT\TT_Avaluo_Gentrificacion_CDMX\src"

DATA_PATH    = os.path.join(PROJECT_ROOT, "..", "data")
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "..", "outputs")
RESULTS_PATH = os.path.join(OUTPUTS_PATH, "results")
FIGURES_PATH = os.path.join(OUTPUTS_PATH, "figures")

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

# -------------------------------------------------------
# CARGA DE DATOS
# Necesitamos:
#   1. df        → dataset_geocodificado.csv ya procesado
#                  (debe tener: alcaldia, price, area,
#                   latitud, longitud, gentrification_index)
#   2. comparacion → generado por el modelo principal
#                    (o lo reconstruimos aquí desde cero)
# -------------------------------------------------------

path_viviendas   = os.path.join(DATA_PATH, "processed", "dataset_geocodificado.csv")
path_2010        = os.path.join(DATA_PATH, "raw", "INEGI", "inegi_2010.csv")
path_2020        = os.path.join(DATA_PATH, "raw", "INEGI", "inegi_2020.csv")
carpeta_promedios = os.path.join(DATA_PATH, "raw", "Promedios")

# Helper de normalización (mismo que el modelo principal)
def clean_text(txt):
    if pd.isna(txt): return txt
    t = str(txt).lower().strip()
    t = t.translate(str.maketrans("áéíóú", "aeiou"))
    return t

# -------------------------------------------------------
# CARGAR DF PRINCIPAL
# -------------------------------------------------------
df = pd.read_csv(path_viviendas)
df = df.dropna(subset=["latitud", "longitud", "price", "area", "alcaldia"])
df = df[df["area"] >= 20].copy()
df["price_m2_raw"] = df["price"] / df["area"]

# -------------------------------------------------------
# RECONSTRUIR ÍNDICE DE GENTRIFICACIÓN (PCA)
# Mismo procedimiento que el modelo principal
# -------------------------------------------------------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("Calculando índice de gentrificación...")

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
X_pca    = StandardScaler().fit_transform(delta[cols_delta].fillna(0))
gentrif  = PCA(n_components=1).fit_transform(X_pca).flatten()

delta["gentrification_index"] = (
    (gentrif - gentrif.min()) / (gentrif.max() - gentrif.min())
)
delta["alc"] = delta["alc"].apply(clean_text)

df["alc"] = df["alcaldia"].apply(clean_text)
df = df.merge(delta[["alc", "gentrification_index"]], on="alc", how="left")
df["gentrification_index"] = df["gentrification_index"].fillna(
    df["gentrification_index"].median()
)

print(f"✓ Índice calculado. Registros: {len(df)}")

# -------------------------------------------------------
# RECONSTRUIR COMPARACION (2022 vs 2025)
# -------------------------------------------------------
print("Reconstruyendo tabla comparacion 2022 vs 2025...")

archivos = glob.glob(os.path.join(carpeta_promedios, "*.csv"))
data_2022 = []

mapeo_alcaldias = {
    "Alvaro Obregon":      "Álvaro Obregón",
    "Benito Juarez":       "Benito Juárez",
    "Coyoacan":            "Coyoacán",
    "Cuajimalpa":          "Cuajimalpa de Morelos",
    "Cuauhtemoc":          "Cuauhtémoc",
    "Gam":                 "Gustavo A. Madero",
    "Magdalena Contreras": "La Magdalena Contreras",
    "Milpa":               "Milpa Alta",
    "Tlahuac":             "Tláhuac"
}

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
        "n_2022":      df_temp["precio por metro cuadrado"].count()
    })

df_2022 = pd.DataFrame(data_2022)
df_2022["alcaldia"] = df_2022["alcaldia"].replace(mapeo_alcaldias)

df_2025_prom = (
    df.assign(precio_m2_2025=df["price"] / df["area"])
    .groupby("alcaldia", as_index=False)
    .agg(precio_m2_2025=("precio_m2_2025", "mean"),
         n_2025=("precio_m2_2025", "count"))
)

comparacion = pd.merge(df_2022, df_2025_prom, on="alcaldia", how="inner")
comparacion["incremento_%"] = (
    (comparacion["precio_m2_2025"] - comparacion["precio_2022"])
    / comparacion["precio_2022"]
) * 100
comparacion["alc"] = comparacion["alcaldia"].apply(clean_text)
comparacion = comparacion.merge(
    delta[["alc", "gentrification_index"]], on="alc", how="left"
)

# Incremento real (INPC dic-2021: 126.478 → dic-2024: 143.042)
INPC_2022 = 126.478
INPC_2025 = 143.042
INFLACION_ACUMULADA = (INPC_2025 - INPC_2022) / INPC_2022

comparacion["precio_2022_real"]  = comparacion["precio_2022"] * (1 + INFLACION_ACUMULADA)
comparacion["incremento_real_%"] = (
    (comparacion["precio_m2_2025"] - comparacion["precio_2022_real"])
    / comparacion["precio_2022_real"]
) * 100

print(f"✓ Tabla comparacion lista. Alcaldías: {len(comparacion)}")

# ======================================================
# NOTA: a partir de aquí el script usa df y comparacion
# ======================================================

print("=" * 70)
print("CORRELACIÓN GENTRIFICACIÓN vs PRECIO m² POR ALCALDÍA")
print("=" * 70)

# -------------------------------------------------------
# 1. PREPARAR DATASET BASE
# -------------------------------------------------------

# Tomamos los listings individuales con precio por m²
df_corr = df[["alcaldia", "price_m2_raw", "gentrification_index"]].copy()
df_corr = df_corr.dropna(subset=["price_m2_raw", "gentrification_index"])
df_corr = df_corr[df_corr["price_m2_raw"] > 0]

# -------------------------------------------------------
# 2. CALCULAR CORRELACIÓN POR ALCALDÍA
#    Lógica: dentro de cada alcaldía, correlacionar
#    el precio de cada listing con el índice de 
#    gentrificación de TODAS las alcaldías (cross-section).
#    Esto mide si la alcaldía con mayor índice tiene
#    sistemáticamente precios más altos que sus vecinas.
# -------------------------------------------------------

# Enfoque correcto: correlación usando promedios por alcaldía
# ponderados por el precio de cada listing individual

resultados = []

alcaldias = df_corr["alcaldia"].unique()

for alc in sorted(alcaldias):

    subset = df_corr[df_corr["alcaldia"] == alc].copy()
    n = len(subset)

    if n < 5:
        resultados.append({
            "alcaldia": alc,
            "n_listings": n,
            "precio_m2_mediana": subset["price_m2_raw"].median(),
            "precio_m2_std": subset["price_m2_raw"].std(),
            "gentrification_index": subset["gentrification_index"].iloc[0],
            "cv_precio": subset["price_m2_raw"].std() / subset["price_m2_raw"].mean()
                         if subset["price_m2_raw"].mean() > 0 else np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "kendall_tau": np.nan,
            "kendall_p": np.nan,
            "nota": "n insuficiente (<5)"
        })
        continue

    # Precio de cada listing vs precio mediano de TODAS las demás alcaldías
    # (mide si un listing "caro" está en alcaldía con alto índice)
    gentrif_val = subset["gentrification_index"].iloc[0]

    # Correlación precio individual vs percentil dentro de la alcaldía
    # con el índice de la alcaldía como referencia constante
    precios = subset["price_m2_raw"].values

    # Percentil de cada listing dentro de su alcaldía (0-100)
    percentiles = np.array([
        np.sum(precios <= p) / len(precios) * 100
        for p in precios
    ])

    # Correlación cruzada: precio vs rango interno
    r_p, p_p = pearsonr(precios, percentiles) if n > 2 else (np.nan, np.nan)
    r_s, p_s = spearmanr(precios, percentiles) if n > 2 else (np.nan, np.nan)
    r_k, p_k = kendalltau(precios, percentiles) if n > 2 else (np.nan, np.nan)

    resultados.append({
        "alcaldia":             alc,
        "n_listings":           n,
        "precio_m2_mediana":    np.median(precios),
        "precio_m2_mean":       np.mean(precios),
        "precio_m2_std":        np.std(precios),
        "gentrification_index": gentrif_val,
        "cv_precio":            np.std(precios) / np.mean(precios),
        "pearson_r":            r_p,
        "pearson_p":            p_p,
        "spearman_r":           r_s,
        "spearman_p":           p_s,
        "kendall_tau":          r_k,
        "kendall_p":            p_k,
        "nota":                 "ok"
    })

df_resultado = pd.DataFrame(resultados)

# -------------------------------------------------------
# 3. CORRELACIÓN GLOBAL: MEDIANA POR ALCALDÍA vs ÍNDICE
#    Esta es la correlación más interpretable para tesis
# -------------------------------------------------------

print("\n" + "=" * 70)
print("CORRELACIÓN GLOBAL: MEDIANA precio_m2 vs gentrification_index")
print("=" * 70)

df_valido = df_resultado.dropna(subset=["precio_m2_mediana", "gentrification_index"])

r_p_g, p_p_g = pearsonr(
    df_valido["gentrification_index"],
    df_valido["precio_m2_mediana"]
)
r_s_g, p_s_g = spearmanr(
    df_valido["gentrification_index"],
    df_valido["precio_m2_mediana"]
)

print(f"Pearson  r = {r_p_g:.4f}   (p = {p_p_g:.4f})")
print(f"Spearman ρ = {r_s_g:.4f}   (p = {p_s_g:.4f})")

# Interpretación
r_ref = abs(r_s_g)
nivel = "ALTA" if r_ref >= 0.7 else "MODERADA" if r_ref >= 0.4 else "BAJA"
sig   = "significativa" if p_s_g <= 0.05 else "NO significativa"
print(f"\n→ Correlación {nivel} y {sig} (Spearman)")

# -------------------------------------------------------
# 4. CORRELACIÓN PONDERADA POR N (más robusta)
# -------------------------------------------------------

print("\n" + "=" * 70)
print("CORRELACIÓN PONDERADA POR N DE LISTINGS")
print("=" * 70)

def pearson_ponderado(x, y, w):
    w = np.array(w, dtype=float)
    w = w / w.sum()
    mx = np.average(x, weights=w)
    my = np.average(y, weights=w)
    cov = np.sum(w * (x - mx) * (y - my))
    sx  = np.sqrt(np.sum(w * (x - mx) ** 2))
    sy  = np.sqrt(np.sum(w * (y - my) ** 2))
    return cov / (sx * sy) if (sx > 0 and sy > 0) else np.nan

r_pond = pearson_ponderado(
    df_valido["gentrification_index"].values,
    df_valido["precio_m2_mediana"].values,
    df_valido["n_listings"].values
)

print(f"Pearson ponderado por n = {r_pond:.4f}")
print("(Alcaldías con más listings tienen mayor peso en el cálculo)")

# -------------------------------------------------------
# 5. TABLA DETALLADA POR ALCALDÍA
# -------------------------------------------------------

print("\n" + "=" * 70)
print("TABLA DETALLADA POR ALCALDÍA")
print("=" * 70)

df_tabla = df_resultado[[
    "alcaldia", "n_listings", "gentrification_index",
    "precio_m2_mediana", "precio_m2_std", "cv_precio", "nota"
]].sort_values("gentrification_index", ascending=False)

df_tabla["cv_precio"] = df_tabla["cv_precio"].round(3)
df_tabla["precio_m2_mediana"] = df_tabla["precio_m2_mediana"].round(0)
df_tabla["precio_m2_std"] = df_tabla["precio_m2_std"].round(0)
df_tabla["gentrification_index"] = df_tabla["gentrification_index"].round(4)

print(df_tabla.to_string(index=False))

print("\nNota: CV = Coeficiente de Variación (std/mean). Valores altos indican")
print("mayor heterogeneidad interna de precios dentro de la alcaldía.")

# -------------------------------------------------------
# 6. ANÁLISIS DE DISPERSIÓN INTERNA
#    ¿Las alcaldías más gentrificadas tienen mayor
#    heterogeneidad de precios (CV alto)?
# -------------------------------------------------------

print("\n" + "=" * 70)
print("CORRELACIÓN: GENTRIFICACIÓN vs HETEROGENEIDAD DE PRECIOS (CV)")
print("=" * 70)

df_cv = df_resultado.dropna(subset=["cv_precio", "gentrification_index"])
df_cv = df_cv[df_cv["nota"] == "ok"]

r_cv_p, p_cv_p = pearsonr(df_cv["gentrification_index"], df_cv["cv_precio"])
r_cv_s, p_cv_s = spearmanr(df_cv["gentrification_index"], df_cv["cv_precio"])

print(f"Pearson  r = {r_cv_p:.4f}   (p = {p_cv_p:.4f})")
print(f"Spearman ρ = {r_cv_s:.4f}   (p = {p_cv_s:.4f})")
print("\nInterpretación: un r positivo sugiere que alcaldías más gentrificadas")
print("presentan mayor dispersión interna de precios (polarización inmobiliaria).")

# -------------------------------------------------------
# 7. GRÁFICA 1: SCATTER GENTRIFICACIÓN vs PRECIO MEDIANA
# -------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Panel izquierdo: precio mediano vs índice ---
ax1 = axes[0]

scatter = ax1.scatter(
    df_valido["gentrification_index"],
    df_valido["precio_m2_mediana"],
    s=df_valido["n_listings"] * 1.5,  # tamaño proporcional a n
    alpha=0.75,
    c=df_valido["gentrification_index"],
    cmap="RdYlGn_r",
    edgecolors="gray",
    linewidths=0.5
)

# Línea de regresión
z = np.polyfit(
    df_valido["gentrification_index"],
    df_valido["precio_m2_mediana"], 1
)
p_line = np.poly1d(z)
x_line = np.linspace(
    df_valido["gentrification_index"].min(),
    df_valido["gentrification_index"].max(), 100
)
ax1.plot(x_line, p_line(x_line), "r--", linewidth=1.5, label="Tendencia")

# Etiquetas
for _, row in df_valido.iterrows():
    ax1.annotate(
        row["alcaldia"],
        (row["gentrification_index"], row["precio_m2_mediana"]),
        textcoords="offset points",
        xytext=(5, 3),
        fontsize=7.5,
        alpha=0.85
    )

plt.colorbar(scatter, ax=ax1, label="Índice de Gentrificación")
ax1.set_xlabel("Índice de Gentrificación (normalizado)", fontsize=11)
ax1.set_ylabel("Precio mediano m² (MXN)", fontsize=11)
ax1.set_title(
    f"Gentrificación vs Precio mediano por alcaldía\n"
    f"Pearson r={r_p_g:.3f} | Spearman ρ={r_s_g:.3f} (p={p_s_g:.3f})",
    fontsize=11
)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- Panel derecho: gentrificación vs CV (heterogeneidad) ---
ax2 = axes[1]

colores_cv = ["forestgreen" if r >= 0 else "firebrick"
              for r in df_cv["cv_precio"] - df_cv["cv_precio"].mean()]

ax2.scatter(
    df_cv["gentrification_index"],
    df_cv["cv_precio"],
    s=df_cv["n_listings"] * 1.5,
    alpha=0.75,
    c=df_cv["gentrification_index"],
    cmap="RdYlGn_r",
    edgecolors="gray",
    linewidths=0.5
)

z2 = np.polyfit(df_cv["gentrification_index"], df_cv["cv_precio"], 1)
p2 = np.poly1d(z2)
ax2.plot(x_line, p2(x_line), "r--", linewidth=1.5)

for _, row in df_cv.iterrows():
    ax2.annotate(
        row["alcaldia"],
        (row["gentrification_index"], row["cv_precio"]),
        textcoords="offset points",
        xytext=(5, 3),
        fontsize=7.5,
        alpha=0.85
    )

ax2.set_xlabel("Índice de Gentrificación (normalizado)", fontsize=11)
ax2.set_ylabel("Coeficiente de Variación del precio m²", fontsize=11)
ax2.set_title(
    f"Gentrificación vs Heterogeneidad de precios (CV)\n"
    f"Pearson r={r_cv_p:.3f} | Spearman ρ={r_cv_s:.3f} (p={p_cv_s:.3f})",
    fontsize=11
)
ax2.grid(True, alpha=0.3)

plt.suptitle(
    "Análisis de correlación espacial por alcaldía\n"
    "(tamaño del punto proporcional a n° de listings)",
    fontsize=12, y=1.01
)

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_PATH, "correlacion_gentrif_por_alcaldia.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# -------------------------------------------------------
# 8. GRÁFICA 2: BARRAS COMPARATIVAS
# -------------------------------------------------------

fig, ax = plt.subplots(figsize=(13, 7))

df_bar = df_resultado.sort_values("gentrification_index", ascending=False)
df_bar = df_bar[df_bar["nota"] == "ok"]

x = np.arange(len(df_bar))
width = 0.38

# Normalizar precio a escala 0-1 para comparar con índice
precio_norm = (
    (df_bar["precio_m2_mediana"] - df_bar["precio_m2_mediana"].min()) /
    (df_bar["precio_m2_mediana"].max() - df_bar["precio_m2_mediana"].min())
)

bars1 = ax.bar(x - width/2, df_bar["gentrification_index"],
               width, label="Índice de gentrificación", color="#534AB7", alpha=0.8)
bars2 = ax.bar(x + width/2, precio_norm,
               width, label="Precio m² normalizado (0–1)", color="#1D9E75", alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(df_bar["alcaldia"], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Valor normalizado (0–1)", fontsize=11)
ax.set_title(
    "Gentrificación vs Precio m² por alcaldía (ambos normalizados 0–1)\n"
    "Ordenado por índice de gentrificación descendente",
    fontsize=11
)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(
    os.path.join(FIGURES_PATH, "barras_gentrif_precio_alcaldia.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# -------------------------------------------------------
# 9. EXPORTAR RESULTADOS
# -------------------------------------------------------

ruta_csv = os.path.join(RESULTS_PATH, "correlacion_por_alcaldia.csv")
df_resultado.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
print(f"\n✓ Tabla exportada en: {ruta_csv}")

# -------------------------------------------------------
# 10. RESUMEN FINAL
# -------------------------------------------------------

print("\n" + "#" * 70)
print("RESUMEN FINAL")
print("#" * 70)
print(f"  Alcaldías analizadas      : {len(df_resultado)}")
print(f"  Alcaldías con n >= 5      : {len(df_resultado[df_resultado['nota']=='ok'])}")
print(f"  Pearson  (mediana vs G)   : r = {r_p_g:.4f}  (p = {p_p_g:.4f})")
print(f"  Spearman (mediana vs G)   : ρ = {r_s_g:.4f}  (p = {p_s_g:.4f})")
print(f"  Pearson ponderado por n   : r = {r_pond:.4f}")
print(f"  Pearson  (CV vs G)        : r = {r_cv_p:.4f}  (p = {p_cv_p:.4f})")
print(f"  Spearman (CV vs G)        : ρ = {r_cv_s:.4f}  (p = {p_cv_s:.4f})")
print("#" * 70)