"""
Drought Impact on Crop Yields: Panel Regression Analysis
=========================================================
Khodanov Georgii | OP Economics and Statistics, HSE
Coursework: Assessing Economic Losses in Agriculture from Climate Extremes

Description:
    This script constructs a global country-year panel dataset by merging
    FAOSTAT crop yield data with SPEI-12 drought index values from SPEIbase
    v2.11. It estimates panel fixed-effects regressions and produces four
    figures used in the final report.

Inputs:
    - FAOSTAT_data_en_4-19-2026.csv  (wheat and maize yields, kg/ha)
    - spei12.nc                       (SPEIbase v2.11, global 0.5 degree grid)

Outputs:
    - panel_final.csv                 (merged panel dataset)
    - regression_results.csv          (regression coefficients)
    - fig1_timeseries.png
    - fig2_scatter.png
    - fig3_regression.png
    - fig4_heatmap.png
"""

import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD FAOSTAT YIELD DATA
# ─────────────────────────────────────────────

fao = pd.read_csv("FAOSTAT_data_en_4-19-2026.csv")

# Keep wheat and maize, 1979-2020
fao = fao[fao["Element"] == "Yield"]
fao = fao[fao["Item"].isin(["Wheat", "Maize"])]
fao = fao[(fao["Year"] >= 1979) & (fao["Year"] <= 2020)]
fao = fao.rename(columns={"Area": "country", "Year": "year",
                           "Value": "yield_kgha", "Item": "crop"})
fao = fao[["country", "year", "crop", "yield_kgha"]].dropna()

# Pivot to wide format
fao_wide = fao.pivot_table(index=["country", "year"],
                           columns="crop",
                           values="yield_kgha").reset_index()
fao_wide.columns.name = None
fao_wide = fao_wide.rename(columns={"Wheat": "wheat_yield",
                                     "Maize": "maize_yield"})

# ─────────────────────────────────────────────
# 2. LOAD SPEI-12 DATA
# ─────────────────────────────────────────────

ds = nc.Dataset("spei12.nc")
spei_data = ds.variables["spei"][:]      # shape: (time, lat, lon)
lats = ds.variables["lat"][:]
lons = ds.variables["lon"][:]

# Build time index (monthly from 1901-01)
n_months = spei_data.shape[0]
dates = pd.date_range(start="1901-01", periods=n_months, freq="MS")

# Country centroids (approximate)
country_centroids = {
    "Afghanistan": (33.9, 67.7), "Albania": (41.2, 20.2),
    "Algeria": (28.0, 1.7), "Angola": (-11.2, 17.9),
    "Argentina": (-34.0, -64.0), "Armenia": (40.1, 45.0),
    "Australia": (-25.0, 133.0), "Austria": (47.5, 14.6),
    "Azerbaijan": (40.1, 47.6), "Bangladesh": (23.7, 90.4),
    "Belarus": (53.7, 27.9), "Belgium": (50.8, 4.5),
    "Benin": (9.3, 2.3), "Bolivia": (-16.3, -63.6),
    "Bosnia and Herzegovina": (44.2, 17.9), "Brazil": (-10.0, -55.0),
    "Bulgaria": (42.7, 25.5), "Burkina Faso": (12.4, -1.6),
    "Cambodia": (12.6, 104.9), "Cameroon": (3.9, 11.5),
    "Canada": (56.1, -106.3), "Chad": (15.5, 18.7),
    "Chile": (-30.0, -71.0), "China": (35.9, 104.2),
    "Colombia": (4.0, -72.0), "Congo": (-0.2, 15.8),
    "Costa Rica": (9.7, -83.8), "Cote d'Ivoire": (7.5, -5.6),
    "Croatia": (45.1, 15.2), "Cuba": (21.5, -80.0),
    "Czech Republic": (49.8, 15.5), "Denmark": (56.3, 9.5),
    "Ecuador": (-1.8, -78.2), "Egypt": (26.8, 30.8),
    "El Salvador": (13.8, -88.9), "Ethiopia": (9.1, 40.5),
    "Finland": (61.9, 25.7), "France": (46.2, 2.2),
    "Georgia": (42.3, 43.4), "Germany": (51.2, 10.5),
    "Ghana": (7.9, -1.0), "Greece": (39.1, 21.8),
    "Guatemala": (15.8, -90.2), "Guinea": (9.9, -11.4),
    "Honduras": (15.2, -86.2), "Hungary": (47.2, 19.5),
    "India": (20.6, 78.9), "Indonesia": (-2.5, 117.9),
    "Iran": (32.4, 53.7), "Iraq": (33.2, 43.7),
    "Ireland": (53.4, -8.2), "Italy": (41.9, 12.6),
    "Japan": (36.2, 138.3), "Jordan": (31.2, 36.5),
    "Kazakhstan": (48.0, 66.9), "Kenya": (-0.0, 37.9),
    "Kyrgyzstan": (41.2, 74.8), "Laos": (18.2, 103.9),
    "Latvia": (56.9, 24.6), "Lebanon": (33.9, 35.9),
    "Lithuania": (55.2, 23.9), "Madagascar": (-18.8, 46.9),
    "Malawi": (-13.3, 34.3), "Malaysia": (2.5, 112.5),
    "Mali": (17.6, -2.0), "Mexico": (23.6, -102.6),
    "Moldova": (47.4, 28.4), "Morocco": (31.8, -7.1),
    "Mozambique": (-18.7, 35.5), "Myanmar": (17.1, 96.1),
    "Nepal": (28.4, 84.1), "Netherlands": (52.1, 5.3),
    "New Zealand": (-40.9, 172.7), "Nicaragua": (12.9, -85.2),
    "Niger": (17.6, 8.1), "Nigeria": (9.1, 8.7),
    "Norway": (60.5, 8.5), "Pakistan": (30.4, 69.3),
    "Panama": (8.5, -80.8), "Paraguay": (-23.4, -58.4),
    "Peru": (-9.2, -75.0), "Philippines": (12.9, 121.8),
    "Poland": (51.9, 19.1), "Portugal": (39.4, -8.2),
    "Romania": (45.9, 24.9), "Russia": (61.5, 105.3),
    "Rwanda": (-1.9, 29.9), "Saudi Arabia": (23.9, 45.1),
    "Senegal": (14.5, -14.5), "Serbia": (44.0, 21.0),
    "Slovakia": (48.7, 19.7), "Slovenia": (46.1, 14.8),
    "Somalia": (5.2, 46.2), "South Africa": (-29.0, 25.1),
    "Spain": (40.5, -3.7), "Sri Lanka": (7.9, 80.8),
    "Sudan": (12.9, 30.2), "Sweden": (62.2, 17.6),
    "Switzerland": (46.8, 8.2), "Syria": (34.8, 38.9),
    "Tajikistan": (38.9, 71.3), "Tanzania": (-6.4, 34.9),
    "Thailand": (15.9, 100.9), "Togo": (8.6, 0.8),
    "Tunisia": (33.9, 9.6), "Turkey": (38.9, 35.2),
    "Turkmenistan": (39.0, 59.6), "Uganda": (1.4, 32.3),
    "Ukraine": (48.4, 31.2), "United Kingdom": (55.4, -3.4),
    "United States of America": (37.1, -95.7),
    "Uruguay": (-32.5, -55.8), "Uzbekistan": (41.4, 64.6),
    "Venezuela": (6.4, -66.6), "Viet Nam": (14.1, 108.3),
    "Yemen": (15.6, 48.5), "Zambia": (-13.1, 27.9),
    "Zimbabwe": (-20.0, 30.0),
}

# Extract annual SPEI-12 for each country (box average ±3 degrees)
def get_country_spei(lat, lon, box=3):
    lat_mask = (lats >= lat - box) & (lats <= lat + box)
    lon_mask = (lons >= lon - box) & (lons <= lon + box)
    subset = spei_data[:, lat_mask, :][:, :, lon_mask]
    # Mean over spatial box, then annual mean
    monthly = np.nanmean(subset.reshape(len(dates), -1), axis=1)
    df = pd.DataFrame({"date": dates, "spei12": monthly})
    df["year"] = df["date"].dt.year
    annual = df.groupby("year")["spei12"].mean().reset_index()
    return annual

print("Extracting SPEI by country...")
spei_records = []
for country, (lat, lon) in country_centroids.items():
    annual = get_country_spei(lat, lon)
    annual["country"] = country
    spei_records.append(annual)

spei_df = pd.concat(spei_records, ignore_index=True)
spei_df = spei_df[(spei_df["year"] >= 1979) & (spei_df["year"] <= 2020)]

# ─────────────────────────────────────────────
# 3. MERGE AND DETREND
# ─────────────────────────────────────────────

panel = fao_wide.merge(spei_df, on=["country", "year"], how="inner")
panel = panel.dropna(subset=["spei12"])

# Detrend yields (remove linear country-specific trend)
def detrend(df, col):
    result = df.copy()
    result[col + "_dt"] = np.nan
    for country in df["country"].unique():
        mask = df["country"] == country
        sub = df[mask].dropna(subset=[col])
        if len(sub) < 5:
            continue
        coef = np.polyfit(sub["year"], sub[col], 1)
        trend = np.polyval(coef, sub["year"])
        result.loc[sub.index, col + "_dt"] = sub[col].values - trend
    return result

panel = detrend(panel, "wheat_yield")
panel = detrend(panel, "maize_yield")
panel.to_csv("panel_final.csv", index=False)
print(f"Panel saved: {len(panel)} observations, {panel['country'].nunique()} countries")

# ─────────────────────────────────────────────
# 4. PANEL REGRESSIONS
# ─────────────────────────────────────────────

results_list = []

def run_fe(data, dep_var, label):
    sub = data[["country", "year", dep_var, "spei12"]].dropna()
    sub = sub.set_index(["country", "year"])
    mod = PanelOLS.from_formula(
        f"{dep_var} ~ spei12 + EntityEffects + TimeEffects",
        data=sub
    )
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    results_list.append({
        "crop": label.split()[0],
        "model": label.split()[1] + " " + label.split()[2] if len(label.split()) > 2 else label.split()[1],
        "beta": round(res.params["spei12"], 1),
        "se": round(res.std_errors["spei12"], 1),
        "tstat": round(res.tstats["spei12"], 2),
        "pvalue": round(res.pvalues["spei12"], 2),
        "rsq": round(res.rsquared, 3),
    })
    return res

r1 = run_fe(panel, "wheat_yield", "Wheat Raw yield")
r2 = run_fe(panel, "wheat_yield_dt", "Wheat Detrended yield")
r3 = run_fe(panel, "maize_yield", "Maize Raw yield")
r4 = run_fe(panel, "maize_yield_dt", "Maize Detrended yield")

results_df = pd.DataFrame(results_list)
results_df.to_csv("regression_results.csv", index=False)
print(results_df)

# ─────────────────────────────────────────────
# 5. FIGURES
# ─────────────────────────────────────────────

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

# --- Fig 1: Time series ---
global_avg = panel.groupby("year")[["wheat_yield", "maize_yield", "spei12"]].mean()

fig, ax1 = plt.subplots(figsize=(11, 4.5))
ax2 = ax1.twinx()
colors = ["#1f77b4", "#ff7f0e"]
ax1.plot(global_avg.index, global_avg["wheat_yield"] / 1000,
         color=colors[0], lw=2, label="Wheat yield")
ax1.plot(global_avg.index, global_avg["maize_yield"] / 1000,
         color=colors[1], lw=2, label="Maize yield")
bar_colors = ["#d62728" if v < 0 else "#aec7e8" for v in global_avg["spei12"]]
ax2.bar(global_avg.index, global_avg["spei12"], color=bar_colors,
        alpha=0.5, width=0.8, label="SPEI-12")
ax2.axhline(0, color="grey", lw=0.8, ls="--")
ax1.set_xlabel("Year")
ax1.set_ylabel("Yield (t/ha)")
ax2.set_ylabel("SPEI-12")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
plt.title("Global Average Crop Yields and SPEI-12 Drought Index, 1979–2020")
plt.tight_layout()
plt.savefig("fig1_timeseries.png")
plt.close()

# --- Fig 2: Scatter detrended yield vs SPEI ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, crop, label, color in zip(
        axes,
        ["wheat_yield_dt", "maize_yield_dt"],
        ["Wheat", "Maize"],
        ["#1f77b4", "#ff7f0e"]):
    sub = panel[["spei12", crop]].dropna()
    ax.scatter(sub["spei12"], sub[crop], alpha=0.15, s=8, color=color)
    m, b = np.polyfit(sub["spei12"], sub[crop], 1)
    xs = np.linspace(sub["spei12"].min(), sub["spei12"].max(), 100)
    ax.plot(xs, m * xs + b, color="black", lw=1.5)
    ax.set_xlabel("SPEI-12")
    ax.set_ylabel("Detrended yield (kg/ha)")
    ax.set_title(f"{label}: Detrended Yield vs SPEI-12")
plt.tight_layout()
plt.savefig("fig2_scatter.png")
plt.close()

# --- Fig 3: Coefficient plot ---
labels = ["Wheat\n(raw)", "Wheat\n(detrended)", "Maize\n(raw)", "Maize\n(detrended)"]
betas = [r.params["spei12"] for r in [r1, r2, r3, r4]]
ses   = [r.std_errors["spei12"] for r in [r1, r2, r3, r4]]
ci95  = [1.96 * s for s in ses]

fig, ax = plt.subplots(figsize=(8, 4.5))
colors_bar = ["#d62728" if b < 0 else "#2ca02c" for b in betas]
ax.barh(labels, betas, xerr=ci95, color=colors_bar, alpha=0.7,
        error_kw={"elinewidth": 1.5, "capsize": 4})
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Coefficient on SPEI-12 (kg/ha per unit SPEI)")
ax.set_title("Panel FE Regression: Effect of SPEI-12 on Crop Yields\n(95% CI, SE clustered by country)")
plt.tight_layout()
plt.savefig("fig3_regression.png")
plt.close()

# --- Fig 4: Heatmap SPEI by country and decade ---
panel["decade"] = (panel["year"] // 10) * 10
heatmap_data = panel.groupby(["country", "decade"])["spei12"].mean().unstack()
heatmap_data = heatmap_data.dropna(thresh=3)
# Keep top 40 countries by data coverage for readability
heatmap_data = heatmap_data.loc[heatmap_data.notna().sum(axis=1).nlargest(40).index]

fig, ax = plt.subplots(figsize=(9, 12))
sns.heatmap(heatmap_data, cmap="RdBu", center=0, vmin=-1.5, vmax=1.5,
            linewidths=0.3, ax=ax, cbar_kws={"label": "SPEI-12"})
ax.set_title("Average SPEI-12 by Country and Decade\n(Red = drought, Blue = wet)")
ax.set_xlabel("Decade")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("fig4_heatmap.png")
plt.close()

print("\nAll figures saved.")
print("Analysis complete.")
