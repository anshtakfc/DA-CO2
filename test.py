import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from IPython.display import display

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

sns.set(style='whitegrid', context='talk')

import json
from entsoe import EntsoePandasClient
import requests
import os

year = 2023 # Set required year
start = pd.Timestamp(f'{year}-01-01 00:00', tz='Europe/Berlin')
end = pd.Timestamp(f'{year+1}-01-01 00:00', tz='Europe/Berlin')
country_code = "DE_LU"
ggc_key = "H4VkwhRFaI5FFxQf8uMSyOUG67pcaSUo9inpGiVEh9UN8K5t"
entsoe_key = 'fe5c8d5e-376c-47ad-8301-fa293edd6893'

os.makedirs("data", exist_ok=True)
combined_path = f"data/{country_code}_{year}_co2_price.csv"

# Function to retrieve ENTSO-E Data
def get_entsoe_data(year: int, country_code: str, entsoe_key: str) -> pd.DataFrame:
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
      print(f"Fetching {start_month} to {end_month} ...")
      queryParams = {
          "zone_code": country_code,
          "start": start_month,
          "end": end_month,
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
        ggc["interval"] = pd.to_datetime(ggc["interval"], errors="coerce")
        ggc = ggc.set_index("interval")
    if ggc.index.tz is None:
      ggc.index = ggc.index.tz_localize('Europe/Berlin', ambiguous=True, nonexistent='shift_forward')
    ggc = ggc.sort_index()
    return ggc

def apply_dst_shift(df_ggc: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Shift GGC data +1 hour inside DST period to attempt alignment
    (as per prior logic). Returns a new aligned DataFrame.
    """
    if year == 2021:
        dst_start = pd.Timestamp("2021-03-28 04:00:00", tz="Europe/Berlin")
        dst_end   = pd.Timestamp("2021-10-31 03:00:00", tz="Europe/Berlin")
    elif year == 2022:
        dst_start = pd.Timestamp("2022-03-27 04:00:00", tz="Europe/Berlin")
        dst_end   = pd.Timestamp("2022-10-30 03:00:00", tz="Europe/Berlin")
    elif year == 2023:
        dst_start = pd.Timestamp("2023-03-26 04:00:00", tz="Europe/Berlin")
        dst_end   = pd.Timestamp("2023-10-29 03:00:00", tz="Europe/Berlin")
    elif year == 2024:
        dst_start = pd.Timestamp("2024-03-31 04:00:00", tz="Europe/Berlin")
        dst_end   = pd.Timestamp("2024-10-27 03:00:00", tz="Europe/Berlin")
    elif year == 2025:
        dst_start = pd.Timestamp("2025-03-30 04:00:00", tz="Europe/Berlin")
        dst_end   = pd.Timestamp("2025-10-26 03:00:00", tz="Europe/Berlin")
    else:
        return df_ggc  # Fallback: no shift for other years

    is_dst = (df_ggc.index >= dst_start) & (df_ggc.index < dst_end)
    shifted = df_ggc[is_dst].copy()
    shifted.index = shifted.index + pd.Timedelta(hours=1)
    aligned = pd.concat([df_ggc[~is_dst], shifted])
    aligned = aligned[~aligned.index.duplicated(keep="first")].sort_index()
    aligned.index.name = "interval"
    return aligned

if os.path.exists(combined_path):
    print("Combined file already exists. Loading...")
    combined_df = pd.read_csv(combined_path, index_col=0, parse_dates=True)
    combined_df.index.name = "interval"
else:
    # Fetch GGC first
    df_ggc_raw = get_ggc_data(year, country_code, ggc_key)
    if df_ggc_raw.empty:
        raise RuntimeError("No GGC data retrieved; cannot build combined file.")

    # Apply DST alignment
    df_ggc_aligned = apply_dst_shift(df_ggc_raw, year)

    # Fetch ENTSO-E data
    df_entsoe = get_entsoe_data(year, country_code, entsoe_key)
    if df_entsoe.empty:
        print("Warning: ENTSO-E data empty; combined file will lack day_ahead_price values.")

    # Join ENTSO-E prices onto GGC intervals (left join keeps all GGC rows)
    combined_df = df_ggc_aligned.join(df_entsoe, how="left")

    # Optional sanity checks
    # Ensure index name
    combined_df.index.name = "interval"

    # Save single CSV
    combined_df.to_csv(combined_path)
    print(f"Combined data saved to {combined_path}")


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
plt.title('Day-Ahead Prices, CO2 Intensity')
fig.tight_layout() # Adjust layout to prevent overlapping elements
plt.grid(True)
plt.show()
