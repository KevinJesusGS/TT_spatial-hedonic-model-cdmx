# Sistema de Valuación Inmobiliaria CDMX
## Escuela Superior de Cómputo · Instituto Politécnico Nacional

**Autores:** Kevin Jesús González Sosa · José Manuel Torres Gutiérrez  
**Trabajo Terminal · ESCOM 2026**

---

## Estructura del proyecto

```
valuador_cdmx/
├── Inicio.py                      ← Portal principal (página de inicio)
├── requirements.txt               ← Dependencias Python
├── .streamlit/
│   └── config.toml                ← Configuración de tema
├── pages/
│   ├── 1_🏢_Compra_Venta.py      ← Módulo: Compra/Venta + Atlas Cartográfico
│   ├── 2_🏘️_Rentas.py            ← Módulo: Renta mensual
│   └── 3_🏠_Airbnb.py            ← Módulo: Hospedaje Airbnb
└── README.md
```

> **Nota:** Los datos de entrenamiento, shapefiles y assets (logos, mapas QGIS) 
> deben colocarse en la misma estructura de rutas relativas que los scripts originales.
> Ver cada módulo para las rutas exactas de datos.

## Ejecución

```bash
pip install -r requirements.txt
streamlit run Inicio.py
```

## Módulos

| Módulo | Descripción | Variable objetivo |
|--------|-------------|-------------------|
| 🏢 Compra/Venta | Precio comercial de vivienda | Precio total MXN / m² |
| 🏘️ Rentas | Renta mensual de vivienda | Renta MXN/mes |
| 🏠 Airbnb | Tarifa de hospedaje | Tarifa MXN/noche |

Todos los módulos integran:
- Modelo hedónico-espacial **ElasticNet** con segmentación **K-Means**
- Índice de **gentrificación** por colonia
- Paradigma **Ciudad de 15 Minutos**
- Variables catastrales del **SIGCDMX**
- Mapa interactivo con servicios urbanos
- Atlas cartográfico (mapas QGIS)
