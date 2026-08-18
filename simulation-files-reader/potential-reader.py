# Generated from: reading_exodus_pot.ipynb


import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay
from joblib import Parallel, delayed
from matplotlib.colors import ListedColormap
from scipy.io import loadmat

from skimage.segmentation import find_boundaries

# Load custom colormaps
colormap_data = loadmat('cold2hot.mat')
cold2hot = ListedColormap(colormap_data['colormap'])
hot2cold = ListedColormap(colormap_data['colormap'][::-1])

# Lily's solidmechdiff106 simulation
# file_base = results/solidmechdiff106/test1
power_level = 'solidmechdiff106'
path = os.path.abspath(f'./results/{power_level}/')
model_path = path+'/test1.e'
model = netCDF4.Dataset(model_path)

df = pd.read_csv(path+'/test1.csv')
df.head()

# model.variables.keys()

time_real = np.array(model.variables['time_whole'][:]).reshape((-1, ))
len(time_real)

# --- Automatically find the 'pot' variable index from the Exodus file ---
names = model.variables["name_nod_var"]
names.set_auto_mask(False)
var_names = [b"".join(c).decode("latin1") for c in names[:]]
print("Node variables available:", var_names)

var_name = 'pot'
variable_index = var_names.index(var_name) + 1  # Exodus variables are 1-based
print(f"Reading variable: {var_name} at index {variable_index}")

def make_array_parallel(nodes, X_all, Y_all, param, blockname="domain", n_jobs=4, ny=50, nx=75):
    start_time = time.time()
    
    # Convert to regular arrays in case they're masked
    x = np.asarray(X_all[nodes])
    y = np.asarray(Y_all[nodes])
    vals = np.asarray(param[:, nodes])  # shape (nt, nnodes_block)

    # Define structured target grid based on MOOSE mesh parameters (nx=75, ny=50)
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    Xi, Yi = np.meshgrid(xi, yi)

    nt = vals.shape[0]

    # Precompute Delaunay triangulation for linear interpolation
    tri = Delaunay(np.column_stack((x, y)))

    # Precompute nearest neighbor interpolator (used for NaNs)
    nearest_interp = NearestNDInterpolator(np.column_stack((x, y)), np.zeros_like(x))  # placeholder

    # Function to interpolate a single timestep
    def interp_timestep(t):
        field = vals[t, :]
        linear_interp = LinearNDInterpolator(tri, field)
        Zi = linear_interp(Xi, Yi)
        mask = np.isnan(Zi)
        if np.any(mask):
            Zi[mask] = NearestNDInterpolator(np.column_stack((x, y)), field)(Xi[mask], Yi[mask])
        return Zi

    # Parallel computation over time steps
    arr_list = Parallel(n_jobs=n_jobs)(delayed(interp_timestep)(t) for t in range(nt))
    arr = np.stack(arr_list, axis=2)

    # Flip once after the loop to match physical Y coordinates
    arr = np.flip(arr, axis=0)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total time for {blockname}: {total_time:.2f} seconds")

    return arr

# Extract coordinates and connectivity
X_all = model.variables['coordx'][:]
Y_all = model.variables['coordy'][:]

# Single block mesh (connect1)
connect1 = model.variables["connect1"][:] - 1
nodes_block = np.unique(connect1.ravel()).astype(int)

# Read the 'pot' variable data
param = model.variables[f'vals_nod_var{variable_index}'][:]

# Interpolate to structured grid and transpose to (time, y, x)
# Note: No *1e6 scaling is applied here because 'pot' is an electric overpotential (Volts), 
# not a velocity that requires micro-scale conversion.
POT = np.transpose(make_array_parallel(nodes_block, X_all, Y_all, param, blockname="domain", n_jobs=6, ny=50, nx=75), 
                   axes=(2, 0, 1))

print(f"POT array shape: {POT.shape}")

# Save to .npy file
numpy_path = os.path.abspath(f'./results/{power_level}/')
np.save(numpy_path+'/POT.npy', POT)
np.save(numpy_path+'/time.npy', time_real)

print("\n Job Done: Saved POT.npy")

# ---------------------------------------------------------
# Plotting a specific timestep
# ---------------------------------------------------------
t_step = 500  # Choose a timestep to visualize
fig, ax1 = plt.subplots(1,1, figsize=(8,5), frameon=True)
cmap = cold2hot

# Ensure the timestep exists
if t_step < POT.shape[0]:
    hmap1 = ax1.imshow(POT[t_step], cmap=cmap, extent=[0, 75, 0, 50], aspect=1.0, interpolation='quadric')
    ax1.set_title(f"Electric Overpotential (pot) at t = {time_real[t_step]:.2f}")
    plt.colorbar(hmap1, ax=ax1, label='Potential (V)')
else:
    print(f"Timestep {t_step} out of bounds (max {POT.shape[0]-1})")

plt.show()
