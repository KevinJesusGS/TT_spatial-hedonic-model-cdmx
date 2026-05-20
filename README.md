# Spatial Hedonic Model for Real Estate Valuation and Gentrification Analysis in Mexico City

## Overview
This repository contains the computational implementation developed for the undergraduate thesis focused on spatial real estate valuation and gentrification analysis in Mexico City.

The proposed methodology integrates spatial econometrics, machine learning, urban accessibility indicators, and dimensionality reduction techniques to estimate residential property values and identify territorial dynamics associated with gentrification.

---

## Research Objective
To develop a spatial regression model for real estate valuation and analyze the impact of gentrification processes on housing prices in Mexico City.

---

## Methodological Components
* **Hedonic pricing modeling:** Structuring property values based on intrinsic and extrinsic attributes.
* **ElasticNet regularized regression:** Robust linear modeling with L1 and L2 regularization to manage multicollinearity ($R^2$: 0.735, Log RMSE: 0.3576).
* **Spatial clustering (K-Means):** Regional segmentation based on socio-economic and geographic characteristics.
* **Principal Component Analysis (PCA):** Construction of a localized Gentrification Index analyzing census deltas (INEGI 2010 vs 2020).
* **Urban accessibility metrics:** Spatial buffers and proximity calculations for public transport networks (Metro, Metrobús) and Health Infrastructure.
* **Neighborhood-based spatial lag variables:** Capture spatial dependency and localized real estate spillovers.
* **15-minute city indicators:** Evaluation of localized urban self-sufficiency.

---

## Project Structure
```text
.
├── data/                             # Spatial shapefiles (SHP) and census data
├── docs/                             # Thesis documentation and references
├── outputs/                          # Datasets exported
└── src/                              # Source code
    ├── modelo_regresion_espacial.py  # Main spatial ML pipeline (ETL, features, training)
    └── valuador_interactivo.py       # Streamlit web dashboard application
```

---

## Installation
Ensure you have Python 3.9+ installed. Clone the repository and install the required dependencies:

```bash
git clone https://github.com/KevinJesusGS/TT_spatial-hedonic-model-cdmx.git
cd TT_spatial-hedonic-model-cdmx
pip install -r requirements.txt
```

---

## Execution

### 1. Run the Spatial ML Pipeline
To process spatial features, calculate urban indicators, train the ElasticNet model, and generate evaluation outputs:
```bash
python src/modelo_regresion_espacial.py
```

### 2. Launch the Interactive Dashboard
To explore valuation predictions, the impact of
geospatial and socioeconomic features via the Streamlit UI:
```bash
streamlit run src/valuador_interactivo.py
```

---

## Interactive Interface
The `valuador_interactivo.py` dashboard leverages Streamlit to provide an intuitive graphical interface for recruiters, researchers, and synodals:
* **Interactive Map Exploration:** Visualize geographic data layers and urban indicators interactively.
* **Real-Time Property Valuation:** Input specific structural characteristics (e.g., area, rooms) to obtain immediate market value estimates from the trained ElasticNet model.
* **Gentrification Index Analytics:** Analyze local gentrification pressure scores across different areas of Mexico City using dynamic charts.

---

## Key Outputs & Performance
The execution of the pipeline generates and updates the following assets inside the `outputs/` directory:
* **Appraisal Accuracy:** Real estate valuation predictions achieving an $R^2$ of **0.7352** and a Logarithmic RMSE of **0.3566**.
* **Spatial Segments:** Regional categorization maps powered by the K-Means algorithm.
* **Gentrification Indices:** Multi-temporal socioeconomic change metrics scaled from 0 to 1.
* **GIS Assets:** Vector data and processed shapefiles fully compatible with software like QGIS or ArcGIS.
* **Statistical Visualizations:** Dynamic and static charts evaluating feature importances and residual distributions.

---

## Data Availability
Due to repository storage constraints, large raw datasets are omitted from the version control system. These public datasets can be downloaded directly from their official portals:
* **INEGI:** *Censo de Población y Vivienda* (2010 and 2020 iterations).
* **Portal de Datos Abiertos de la Ciudad de México:** Urban mobility layers, public transport networks, and health infrastructure frameworks.

---

## Author
* **Kevin Jesús González Sosa** - *Data Science Student* - [GitHub](https://github.com/KevinJesusGS)
* **José Manuel Torres Gutiérrez** - *Data Science Student*

## Academic Context
Developed as a final degree project and undergraduate thesis for the Data Science program at **Escuela Superior de Cómputo (ESCOM)**, belonging to the **Instituto Politécnico Nacional (IPN)**.

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.