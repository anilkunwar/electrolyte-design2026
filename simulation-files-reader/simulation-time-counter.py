# Generated from: reading_time_battery.ipynb
# Converted to script style for Lily's solid mechanics diffusion model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def explore_model(file_path):
    # The MOOSE input file defines the output base as: 
    # file_base = results/solidmechdiff106/test1
    # MOOSE will automatically append _0001.csv, _0002.csv for multi-apps, 
    # or just create test1.csv if it's a single processor run.
    # We use glob to find the actual csv file(s) generated.
    import glob
    csv_files = sorted(glob.glob(os.path.join(file_path, '*.csv')))
    
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {file_path}")
        
    # If scalar postprocessors are output, MOOSE usually creates a single CSV 
    # named after the file_base (e.g., test1.csv)
    print(f"Loading: {csv_files[0]}")
    return pd.read_csv(csv_files[0])

# Define the path based on the MOOSE [Outputs] file_base
# Update the prefix to match your actual local directory structure
model_path = os.path.abspath('./results/solidmechdiff106/')
df = explore_model(model_path)

# Inspect the first few rows to see the columns (time, dt, eta_min, c_max, etc.)
print("DataFrame Head:")
print(df.head())

print("\nDataFrame Tail:")
print(df.tail())

# ---------------------------------------------------------
# 1. Inspecting the Adaptive Time Stepping
# ---------------------------------------------------------
# The MOOSE input uses IterationAdaptiveDT with dt=1e-3 and growth_factor=1.1
# Let's verify this behavior in the CSV data.

# Calculate the actual dt taken between steps
df['actual_dt'] = df['time'].diff()

# Check the first 10 steps (should be growing from 1e-3)
print("\nFirst 10 steps dt evolution:")
print(df[['time', 'dt', 'actual_dt']].head(10))

# Check where the time step hits the maximum cap (dtmax = 100)
dt_max_idx = df[df['actual_dt'] >= 100].first_valid_index()
if dt_max_idx is not None:
    print(f"\nTime step reaches maximum (dt=100) at row index: {dt_max_idx}")
    print(df[['time', 'dt', 'actual_dt']].iloc[dt_max_idx-2:dt_max_idx+3])

# ---------------------------------------------------------
# 2. Sampling the Time Data
# ---------------------------------------------------------
# Because the steps grow exponentially, array slicing like [::5] 
# is not physically meaningful for time. Instead, we sample by 
# actual time values or logarithmic intervals.

# Example 1: Skip the initial startup transient (e.g., ignore time < 0.1)
df_steady = df[df['time'] >= 0.1].reset_index(drop=True)
print(f"\nShape after skipping startup transient: {df_steady.shape}")

# Example 2: Sample the dataframe at regular physical time intervals
# Create a target time array (e.g., every 5000 time units up to 1E7)
target_times = np.arange(0, 1.01e7, 5000)
# Use reindex to find the closest MOOSE output time to our target times
df_sampled = df.iloc[(df['time'].values[:, None] - target_times).argmin(axis=0)]
print(f"Original rows: {len(df)}, Sampled rows: {len(df_sampled)}")

# ---------------------------------------------------------
# 3. Visualizing the Time Sampling
# ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Time step size vs Time (shows the exponential growth to dtmax=100)
ax1.plot(df['time'], df['actual_dt'], 'b-', label='Actual dt')
ax1.axhline(100, color='r', linestyle='--', label='MOOSE dtmax = 100')
ax1.set_xlabel('Simulation Time')
ax1.set_ylabel('Time Step Size (dt)')
ax1.set_title('Adaptive Time Stepping Behavior')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, which="both", ls="--", alpha=0.5)

# Plot 2: Postprocessor data over time (e.g., max concentration)
# We use the sampled dataframe to show how it captures the curve
if 'c_max' in df.columns:
    ax2.plot(df['time'], df['c_max'], 'lightgray', label='Raw MOOSE output', alpha=0.7)
    ax2.plot(df_sampled['time'], df_sampled['c_max'], 'ro', markersize=4, label='Sampled every 5000 time units')
    ax2.set_xlabel('Simulation Time')
    ax2.set_ylabel('Max Concentration (c_max)')
    ax2.set_title('Postprocessor Data vs Time')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.5)
else:
    ax2.text(0.5, 0.5, 'c_max not found in CSV', ha='center', va='center')

plt.tight_layout()
plt.show()
