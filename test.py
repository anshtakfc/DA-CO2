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


# Function to retrieve ENTSO-E Data

def get_entsoe_data(year, start, end, country_code, entsoe_key):
  client = EntsoePandasClient(api_key=entsoe_key)
  monthly_data = []
  for month in range(1, 13):
    start = pd.Timestamp(f"{year}-{month:02d}-01", tz="Europe/Berlin")
    end = pd.Timestamp(f"{year}-{month%12+1:02d}-01", tz="Europe/Berlin") if month < 12 else pd.Timestamp(f"{year+1}-01-01", tz="Europe/Berlin")
    print(f"Fetching {start} to {end} ...")
    try:
      df = client.query_day_ahead_prices(
      country_code=country_code,
      start=start,
      end=end,
      )
      monthly_data.append(df)
    except Exception as e:
      print(f"Error fetching {start} to {end}: {e}")
  if monthly_data:
    entsoe_data = pd.concat(monthly_data)
    entsoe_data = entsoe_data[~entsoe_data.index.duplicated()]
    if entsoe_data.index.tz is None:
      entsoe_data.index = entsoe_data.index.tz_localize('Europe/Berlin', ambiguous='infer', nonexistent='shift_forward')
    entsoe_data = entsoe_data.sort_index()
    return entsoe_data
  else:
    print("No data fetched")
    return pd.DataFrame()

# Function to retrieve GGC Data

def get_ggc_data(year, start, end, country_code, ggc_key, limit=3139):
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
        intensity = entry.get("production_co2_intensity")
        emitted = entry.get("production_co2_emitted")
        rows.append({
            "interval": timestamp,
            "production_co2_intensity": intensity,
            "production_co2_emitted": emitted
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


# Load ENTSO-E data for the required year if not already available

os.makedirs("data", exist_ok=True)
local_path_entsoe = f"data/{country_code}_{year}_entsoe.csv"

if not os.path.exists(local_path_entsoe):
  df_entsoe = get_entsoe_data(year, start, end, country_code, entsoe_key)
  # Convert Series to DataFrame if necessary
  if isinstance(df_entsoe, pd.Series):
      df_entsoe = df_entsoe.to_frame()
  df_entsoe.to_csv(local_path_entsoe, index=True)
  print("ENTSO-E data saved")
else:
  print("ENTSO-E file already exists")
  df_entsoe = pd.read_csv(local_path_entsoe, index_col=0, parse_dates=True)
  df_entsoe = df_entsoe.sort_index()

# Load GGC data for the required year if not already available

os.makedirs("data", exist_ok=True)
local_path_ggc = f"data/{country_code}_{year}_ggc.csv"

if not os.path.exists(local_path_ggc):
  df_ggc = get_ggc_data(year, start, end, country_code, ggc_key)
  # Convert Series to DataFrame if necessary
  if isinstance(df_ggc, pd.Series):
      df_ggc = df_ggc.to_frame()
  df_ggc.to_csv(local_path_ggc, index=True)
  print("GGC data saved")
else:
  print("GGC file already exists")
  df_ggc = pd.read_csv(local_path_ggc, index_col=0, parse_dates=True)
  df_ggc = df_ggc.sort_index()

# DST alignment: Shift GGC by +1 hour during DST (summer time)

# Choosing DST period for the required year in Europe/Berlin
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

# Create mask for DST period
is_dst = (df_ggc.index >= dst_start) & (df_ggc.index < dst_end)
# Shift DST period by 1 hour forward
df_dst_shifted = df_ggc[is_dst].copy()
df_dst_shifted.index = df_dst_shifted.index + pd.Timedelta(hours=1)
# Create corrected DataFrame
df_ggc_aligned = pd.concat([df_ggc[~is_dst], df_dst_shifted])
df_ggc_aligned = df_ggc_aligned[~df_ggc_aligned.index.duplicated(keep='first')]
df_ggc_aligned = df_ggc_aligned.sort_index()


# Rename the column in df_entsoe for clarity
df_entsoe_renamed = df_entsoe.rename(columns={df_entsoe.columns[0]: 'DA_Price'})

# Merge the two dataframes on their index (timestamps)
# Use an outer join to keep all timestamps, then fill NaNs if necessary, or resample to common frequency
combined_df = pd.merge(df_entsoe_renamed, df_ggc_aligned[['production_co2_intensity', 'production_co2_emitted']], left_index=True, right_index=True, how='inner')

# Create the plot
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot Day-Ahead Prices on the first y-axis
ax1.lineplot(combined_df.index, y='DA_Price', data=combined_df, ax=ax1, color='blue', label='Day-Ahead Price (€/MWh)', linewidth=0.7)
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Day-Ahead Price (€/MWh)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left') # Add legend for ax1

# Create a second y-axis for CO2 Intensity
ax2 = ax1.twinx()

# Plot CO2 Intensity on the second y-axis
sns.lineplot(x=combined_df.index, y='production_co2_intensity', data=combined_df, ax=ax2, color='red', label='CO2 Intensity (gCO2eq/kWh)', linewidth=0.7)
ax2.set_ylabel('CO2 Intensity (gCO2eq/kWh)', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend(loc='upper right') # Add legend for ax2

# Add title and grid
plt.title('Day-Ahead Prices, CO2 Intensity')
fig.tight_layout() # Adjust layout to prevent overlapping elements
plt.grid(True)
plt.show()
