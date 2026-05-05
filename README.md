# Spatial Hedonic Model for Real Estate Valuation and Gentrification Analysis in Mexico City

## Overview

This repository contains the computational implementation developed for the undergraduate thesis focused on spatial real estate valuation and gentrification analysis in Mexico City.

The proposed methodology integrates spatial econometrics, machine learning, urban accessibility indicators, and dimensionality reduction techniques to estimate residential property values and identify territorial dynamics associated with gentrification.

---

## Research Objective

To develop a spatial regression model for real estate valuation and analyze the impact of gentrification processes on housing prices in Mexico City.

---

## Methodological Components

* Hedonic pricing modeling
* ElasticNet regularized regression
* Spatial clustering
* Principal Component Analysis (PCA)
* Urban accessibility metrics
* Neighborhood-based spatial lag variables
* 15-minute city indicators

---

## Project Structure

```text
src/
data/
outputs/
docs/
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Execution

```bash
python src/modelo_regresion_espacial.py
```

---

## Outputs

The pipeline generates:

* Real estate valuation predictions
* Spatial segmentation analysis
* Gentrification indicators
* GIS-compatible outputs
* Statistical visualizations

---

## Data Availability

Some large raw datasets are not included due to repository storage constraints.

These datasets can be obtained from official sources:

* INEGI
* Gobierno de la Ciudad de México

---

## Author

Kevin González

---

## Academic Context

Developed as part of an undergraduate thesis in data analytics and spatial modeling.

---

## License

MIT
