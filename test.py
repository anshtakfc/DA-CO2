import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
sns.set(style='whitegrid', context='talk')

import json
from entsoe import EntsoePandasClient
import requests
import os
import sys

year = 2023 # Set required year
start = pd.Timestamp(f'{year}-01-01 00:00', tz='Europe/Berlin')
end = pd.Timestamp(f'{year+1}-01-01 00:00', tz='Europe/Berlin')
country_code = "DE_LU"
ggc_key = "H4VkwhRFaI5FFxQf8uMSyOUG67pcaSUo9inpGiVEh9UN8K5t"
entsoe_key = 'fe5c8d5e-376c-47ad-8301-fa293edd6893'

os.makedirs("data", exist_ok=True)
combined_path = f"data/{country_code}_{year}_co2_price.csv"

# Function to retrieve ENTSO-E Data
def get_entsoe_data(year, country_code, entsoe_key) -> pd.DataFrame:
    client = EntsoePandasClient(api_key=entsoe_key)
    monthly_data = []
    for month in range(1, 13):
        start = pd.Timestamp(f"{year}-{month:02d}-01", tz="Europe/Berlin")
        if month < 12:
            end = pd.Timestamp(f"{year}-{month+1:02d}-01", tz="Europe/Berlin")
        else:
            end = pd.Timestamp(f"{year+1}-01-01", tz="Europe/Berlin")
        print(f"[ENTSO-E] Fetching {start} to {end} ...")
        try:
            df = client.query_day_ahead_prices(
                country_code=country_code,
                start=start,
                end=end,
            )
            monthly_data.append(df)
        except Exception as e:
            print(f"Error fetching {start} to {end}: {e}")

    if not monthly_data:
        print("No ENTSO-E data fetched.")
        return pd.DataFrame()

    entsoe = pd.concat(monthly_data)
    entsoe = entsoe[~entsoe.index.duplicated()]
    if entsoe.index.tz is None:
        entsoe.index = entsoe.index.tz_localize('Europe/Berlin', ambiguous='infer', nonexistent='shift_forward')
    entsoe = entsoe.sort_index()
    if isinstance(entsoe, pd.Series):
        entsoe = entsoe.to_frame()
    entsoe.columns = ['day_ahead_price']
    return entsoe

def load_entsoe_from_csv(year, country_code, entsoe_key, csv_path: str) -> pd.DataFrame:
    if os.path.exists(csv_path):
        df_raw = pd.read_csv(csv_path, sep=',', encoding='utf-8-sig', engine='python')

        mtu_col = "MTU (CET/CEST)"
        area_col = "Area"
        seq_col = "Sequence"
        price_col = "Day-ahead Price (EUR/MWh)"

        df = df_raw[df_raw[area_col] == "BZN|DE-LU"].copy()
        df = df[df[seq_col].str.contains("Sequence 1", na=False)]

        start_strings = df[mtu_col].astype(str).str.split(" - ").str[0]
        dt_start = pd.to_datetime(start_strings, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        dt_start = dt_start.dt.tz_localize('Europe/Berlin', ambiguous='infer', nonexistent='shift_forward')

        price_str = df[price_col].astype(str)
        price_str = price_str.str.strip()
        price_str = (
            price_str
            .str.replace('\u00A0', '', regex=False)   # non-breaking space
            .str.replace(',', '', regex=False)        # remove thousands separators if any
            )
        price_str = price_str.str.extract(r'^\s*([+-]?\d+(?:\.\d+)?)\s*$', expand=False)
        price_vals = pd.to_numeric(price_str, errors='coerce')
        s_15m = pd.Series(price_vals.values, index=dt_start).sort_index()
        s_hourly = s_15m.resample("h").mean()

        df_prices = s_hourly.to_frame(name="day_ahead_price")
        df_prices.index.name = "interval"
        df_prices = df_prices.dropna(how="all")

        return df_prices
    print("[ENTSOE] Cache not found, calling API...")
    df = get_entsoe_data(year, country_code, entsoe_key)
    if df.empty:
        print("[ENTSOE] Warning: API returned empty ENTSOE data.")
        return df
    # Save cleaned, timezone-consistent CSV with UTC index for portability
    df_to_save = df.copy()
    df_to_save.to_csv(csv_path)
    print(f"[ENTSOE] Cleaned GGC data cached at {csv_path}")
    return df
    

# Function to retrieve GGC Data

def get_ggc_data(year, country_code, ggc_key, limit=3139):
    print("Fetching GGC data")
    requestUrl = f"https://eco2grid.com/green-grid-compass/co2intensity/co2/detailed/hourly?apikey={ggc_key}"
    requestHeaders = {"Accept": "application/json"}
    rows=[]
    for month in range(1, 13):
      start_month = pd.Timestamp(f'{year}-{month:02d}-01 00:00', tz='Europe/Berlin')
      if month < 12:
          end_month = pd.Timestamp(f'{year}-{month+1:02d}-01 00:00', tz='Europe/Berlin')
      else:
          end_month = pd.Timestamp(f'{year+1}-01-01 00:00', tz='Europe/Berlin')
      start_utc = start_month.tz_convert('UTC')
      end_utc = end_month.tz_convert('UTC')
      print(f"Fetching {start_month} to {end_month} ...")
      queryParams = {
          "zone_code": country_code,
          "start": start_utc.isoformat(),
          "end": end_utc.isoformat(),
          "emission_scope": "operational",
          "limit": limit
      }
      response = requests.get(requestUrl, headers=requestHeaders, params=queryParams)
      if response.status_code == 200:
          data = response.json()
  
          for entry in data:
            timestamp = entry.get("interval")
            if not timestamp:
                continue
            rows.append({
                "interval": timestamp,
                "production_co2_intensity": entry.get("production_co2_intensity"),
                "production_co2_emitted": entry.get("production_co2_emitted")
                })
      else:
          print("Error:", response.status_code)
          print(response.text)
          break
    ggc = pd.DataFrame(rows)
    if not ggc.empty:
        ggc["interval"] = pd.to_datetime(ggc["interval"], errors="coerce", utc=True)
        ggc = ggc.dropna(subset=["interval"]).set_index("interval")
        ggc = ggc.tz_convert('Europe/Berlin')
        ggc.index.name = "interval"
        ggc = ggc.sort_index()
        return ggc

# Cached GGC loader/saver: use cache if present, else call API and save cleaned CSV
def load_or_fetch_ggc(year, country_code, ggc_key, cache_path: str) -> pd.DataFrame:
    if os.path.exists(cache_path):
        print(f"[GGC] Loading cached GGC data from {cache_path}")
        df = pd.read_csv(cache_path, index_col=0)
        # Restore timezone-aware index (cache stores ISO strings)
        idx = pd.to_datetime(df.index, utc=True, errors='coerce').tz_convert('Europe/Berlin')
        df.index = idx
        df.index.name = 'interval'
        # Ensure numeric types
        for col in ["production_co2_intensity", "production_co2_emitted"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.sort_index()
        return df
    print("[GGC] Cache not found, calling API...")
    df = get_ggc_data(year, country_code, ggc_key)
    if df.empty:
        print("[GGC] Warning: API returned empty GGC data.")
        return df
    # Save cleaned, timezone-consistent CSV with UTC index for portability
    df_to_save = df.copy()
    df_to_save.index = df_to_save.index.tz_convert('UTC')  # store as UTC in CSV
    df_to_save.to_csv(cache_path)
    print(f"[GGC] Cleaned GGC data cached at {cache_path}")
    return df

if os.path.exists(combined_path):
    print("Combined file already exists. Loading...")
    combined_df = pd.read_csv(combined_path, index_col=0, parse_dates=True)
    combined_df.index.name = "interval"
else:
    # Fetch GGC first
    df_ggc = load_or_fetch_ggc(year, country_code, ggc_key, cache_path=f"data/DE_LU_{year}_ggc.csv")
    if df_ggc.empty:
        raise RuntimeError("No GGC data retrieved; cannot build combined file.")
    df_entsoe = load_entsoe_from_csv(year, country_code, entsoe_key, csv_path=f"data/DE_LU_{year}_entsoe.csv")
    if df_entsoe.empty:
        raise RuntimeError("No ENTSOE data retrieved; cannot build combined file.")
    
    # Join ENTSO-E prices onto GGC intervals (left join keeps all GGC rows)
    combined_df = df_ggc.join(df_entsoe, how="left")

    # Ensure index name
    combined_df.index.name = "interval"

    # Save single CSV
    combined_df.to_csv(combined_path)
    print(f"Combined data saved to {combined_path}")
    
# Ensure the index is a timezone-aware DatetimeIndex in Europe/Berlin
if not isinstance(combined_df.index, pd.DatetimeIndex):
    # Try parsing with UTC, then convert to Europe/Berlin
    combined_df.index = pd.to_datetime(combined_df.index, utc=True, errors='coerce')

# If still tz-naive (can happen if CSV lost tz info), localize to UTC then convert
if combined_df.index.tz is None:
    combined_df.index = combined_df.index.tz_localize('UTC')

# Finally convert to Europe/Berlin
combined_df.index = combined_df.index.tz_convert('Europe/Berlin')
combined_df.index.name = 'interval'

# -----------------------------
# Feature engineering
# -----------------------------
# Time-based features
combined_df["hour"] = combined_df.index.hour
combined_df["dow"] = combined_df.index.dayofweek  # 0=Mon, 6=Sun
combined_df["month"] = combined_df.index.month

# Seasons (meteorological): Winter(12,1,2), Spring(3,4,5), Summer(6,7,8), Autumn(9,10,11)
def season_from_month(m):
    if m in (12, 1, 2):
        return "winter"
    elif m in (3, 4, 5):
        return "spring"
    elif m in (6, 7, 8):
        return "summer"
    else:
        return "autumn"
combined_df["season"] = combined_df["month"].apply(season_from_month)

# One-hot encode season for correlation with numeric targets
season_dummies = pd.get_dummies(combined_df["season"], prefix="season")
combined_df = pd.concat([combined_df, season_dummies], axis=1)

# Morning/evening profiles (typical load/price/CO2 patterns)
combined_df["is_morning_peak"] = combined_df["hour"].between(7, 10).astype(int)
combined_df["is_evening_peak"] = combined_df["hour"].between(17, 21).astype(int)

# Weekday/weekend flags
combined_df["is_weekend"] = (combined_df["dow"] >= 5).astype(int)

# Lag features (previous hour/day) for price/intensity/emitted
combined_df["day_ahead_price_lag1h"] = combined_df["day_ahead_price"].shift(1)
combined_df["day_ahead_price_lag24h"] = combined_df["day_ahead_price"].shift(24)
combined_df["production_co2_intensity_lag1h"] = combined_df["production_co2_intensity"].shift(1)
combined_df["production_co2_intensity_lag24h"] = combined_df["production_co2_intensity"].shift(24)
combined_df["production_co2_emitted_lag1h"] = combined_df["production_co2_emitted"].shift(1)
combined_df["production_co2_emitted_lag24h"] = combined_df["production_co2_emitted"].shift(24)

# Inter-hour changes (momentum)
combined_df["price_delta_1h"] = combined_df["day_ahead_price"].diff(1)
combined_df["intensity_delta_1h"] = combined_df["production_co2_intensity"].diff(1)
combined_df["emitted_delta_1h"] = combined_df["production_co2_emitted"].diff(1)

# -----------------------------
# Correlation analysis
# -----------------------------
# Choose numeric columns for correlation
corr_cols = [
    "day_ahead_price",
    "production_co2_intensity",
    "production_co2_emitted",
    "hour", "dow", "is_weekend",
    "is_morning_peak", "is_evening_peak",
    "season_winter", "season_spring", "season_summer", "season_autumn",
    "day_ahead_price_lag1h", "day_ahead_price_lag24h",
    "production_co2_intensity_lag1h", "production_co2_intensity_lag24h",
    "production_co2_emitted_lag1h", "production_co2_emitted_lag24h",
    "price_delta_1h", "intensity_delta_1h", "emitted_delta_1h",
]

corr_df = combined_df[corr_cols].dropna(how="any")  # drop rows where lags create NaNs

# Correlation on actual values
corr_matrix_actual = corr_df.corr(method='pearson')
cols = ['day_ahead_price', 'production_co2_intensity', 'production_co2_emitted']
corr_matrix_thin = corr_matrix_actual.loc[:, cols]

fig, ax = plt.subplots(figsize=(6.5, 12))  # tall figure
sns.heatmap(
    corr_matrix_thin,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    cbar_kws={'label': 'Correlation Coefficient'},
    ax=ax
)
ax.set_title(f'Correlations against key variables {year}')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

corr_df_norm = combined_df.copy()
scaler = StandardScaler()
corr_df_norm[cols] = scaler.fit_transform(combined_df[cols])

# -----------------------------
# Visual checks that help explain correlations
# -----------------------------
# 1) Hour-of-day averages: price vs intensity/emitted
hourly_means = corr_df_norm.groupby("hour")[cols].mean()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hourly_means.index, hourly_means["day_ahead_price"], label="Day-Ahead Price", color="blue")
ax.plot(hourly_means.index, hourly_means["production_co2_intensity"], label="CO2 Intensity", color="red")
ax.plot(hourly_means.index, hourly_means["production_co2_emitted"], label="CO2 Emitted", color="green")
ax.set_title(f"Hour-of-Day Profiles (Normalized Averages) {year}")
ax.set_xlabel("Hour")
ax.set_ylabel("Normalized mean")
ax.grid(True)
ax.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 3: Weekday vs Weekend profiles
# -----------------------------
weekday_means = corr_df_norm.groupby("is_weekend")[cols].mean()

fig, ax = plt.subplots(figsize=(8, 5))
weekday_means.plot(kind="bar", ax=ax)
ax.set_title(f"Weekday vs Weekend Averages (Normalized) {year}")
ax.set_xlabel("is_weekend (0=weekday,1=weekend)")
ax.set_ylabel("Normalized mean")
ax.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 4: Seasonal averages
# -----------------------------
season_order = ["winter", "spring", "summer", "autumn"]
season_means = corr_df_norm.groupby("season")[cols].mean().reindex(season_order)

fig, ax = plt.subplots(figsize=(8, 5))
season_means.plot(kind="bar", ax=ax)
ax.set_title(f"Seasonal Averages (Normalized) {year}")
ax.set_ylabel("Normalized mean")
ax.grid(True)
plt.tight_layout()
plt.show()

# Create the plot
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot Day-Ahead Prices on the first y-axis
ax1.plot(combined_df.index, combined_df['day_ahead_price'], color='blue', label='Day-Ahead Price (€/MWh)', linewidth=0.7)
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Day-Ahead Price (€/MWh)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left') # Add legend for ax1

# Create a second y-axis for CO2 Intensity
ax2 = ax1.twinx()

# Plot CO2 Intensity on the second y-axis
ax2.plot(combined_df.index, combined_df['production_co2_intensity'], color='red', label='CO2 Intensity (gCO2eq/kWh)', linewidth=0.7)
ax2.set_ylabel('CO2 Intensity (gCO2eq/kWh)', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend(loc='upper right') # Add legend for ax2

# Add title and grid
plt.title(f'Day-Ahead Prices, CO2 Intensity (Actual) {year}')
fig.tight_layout() # Adjust layout to prevent overlapping elements
plt.grid(True)
plt.show()
