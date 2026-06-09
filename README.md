# Drought Impact on Crop Yields: Panel Regression Analysis

**Author:** Khodanov Georgii  
**Course:** Economics and Statistics, HSE  
**Topic:** Assessing Economic Losses in Agriculture from Climate Extremes

## Description

This project estimates the effect of drought on wheat and maize yields
using a global panel dataset of 150 countries over 1979–2020.
Drought is measured via the Standardised Precipitation-Evapotranspiration
Index (SPEI-12). The empirical model is a panel fixed-effects regression
with country and year fixed effects.

## Data

| Dataset | Source | Coverage |
|---|---|---|
| Crop yields (wheat, maize) | FAOSTAT | 151 countries, 1979–2020 |
| Drought index (SPEI-12) | SPEIbase v2.11, CSIC | Global, 0.5° grid |

> Data files are not included in this repository due to size.  
> Download from [FAOSTAT](https://www.fao.org/faostat) and [SPEIbase](https://spei.csic.es).

## Files

| File | Description |
|---|---|
| `kt3_analysis.py` | Main analysis script |

## How to Run

1. Install dependencies:
2. Place data files in the same folder:
   - `FAOSTAT_data_en_4-19-2026.csv`
   - `spei12.nc`

3. Run:
## Results

Panel FE regression shows negative coefficients for SPEI-12 on both
wheat and maize yields, consistent with drought reducing productivity.
Estimates are not statistically significant at conventional levels,
likely due to centroid-based SPEI extraction rather than
cropland-weighted aggregation.

## Key References

- Jarrett et al. (2023) — *Ecological Economics*
- Araneda-Cabrera et al. (2021) — drought index benchmarking
- Schmitt et al. (2022) — farm-level drought impacts
