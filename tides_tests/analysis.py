import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

data_dir = 'run1'
diag1_file = 'diag1.glob.nc'
grid_file = 'grid.glob.nc'

grid = xr.open_dataset(os.path.join(data_dir, grid_file))
diag1 = xr.open_dataset(os.path.join(data_dir, diag1_file))

# %% bathymetry
fig, ax = plt.subplots(1, 1)
ax.plot(-grid.Depth)

# %% towyo section
it = 240
ysl = slice(500, 2600)
zsl = slice(80, 130)

fig, axs = plt.subplots(3, 1, figsize=(6.5, 5), sharex=True, sharey=True)
axs[0].pcolormesh(grid.Y[ysl], grid.Z[zsl], diag1.VVEL[it, zsl, ysl, 0],
                  vmin=-0.3, vmax=0.3, rasterized=True, cmap='coolwarm')
axs[1].pcolormesh(grid.Y[ysl], grid.Z[zsl], diag1.WVEL[it, zsl, ysl, 0],
                  vmin=-0.1, vmax=0.1, rasterized=True, cmap='coolwarm')
axs[2].pcolormesh(grid.Y[ysl], grid.Z[zsl], np.log10(diag1.KLeps[it, zsl, ysl, 0]),
                  vmin=-11, vmax=-4, rasterized=True)

axs[0].set_title('{} {}'.format(diag1['T'][it].values, diag1['T'][it].units))


# %% MP on sill top
iy = 861
zsl = slice(55, 105)

fig, ax = plt.subplots(1, 1)
ax.plot(grid.Y, -grid.Depth)
ax.plot(grid.Y[iy].values*np.ones(grid.Z[zsl].shape), grid.Z[zsl])

fig, axs = plt.subplots(2, 1, figsize=(6.5, 4), sharex=True, sharey=True)
axs[0].pcolormesh(diag1['T'], grid.Z[zsl], diag1.VVEL[:, zsl, iy, 0].T,
                  vmin=-0.3, vmax=0.3, rasterized=True, cmap='coolwarm')
axs[1].pcolormesh(diag1['T'], grid.Z[zsl], np.log10(diag1.KLeps[:, zsl, iy, 0]).T,
                  vmin=-11, vmax=-4, rasterized=True)

# %% MP just past sill top
iy = 1010
zsl = slice(64, 114)

fig, ax = plt.subplots(1, 1)
ax.plot(grid.Y, -grid.Depth)
ax.plot(grid.Y[iy].values*np.ones(grid.Z[zsl].shape), grid.Z[zsl])

fig, axs = plt.subplots(2, 1, figsize=(6.5, 4), sharex=True, sharey=True)
axs[0].pcolormesh(diag1['T'], grid.Z[zsl], diag1.VVEL[:, zsl, iy, 0].T,
                  vmin=-0.3, vmax=0.3, rasterized=True, cmap='coolwarm')
axs[1].pcolormesh(diag1['T'], grid.Z[zsl], np.log10(diag1.KLeps[:, zsl, iy, 0]).T,
                  vmin=-11, vmax=-4, rasterized=True)

# %% MP on down slope
iy = 2000
zsl = slice(74, 124)

fig, ax = plt.subplots(1, 1)
ax.plot(grid.Y, -grid.Depth)
ax.plot(grid.Y[iy].values*np.ones(grid.Z[zsl].shape), grid.Z[zsl])

fig, axs = plt.subplots(2, 1, figsize=(6.5, 4), sharex=True, sharey=True)
axs[0].pcolormesh(diag1['T'], grid.Z[zsl], diag1.VVEL[:, zsl, iy, 0].T,
                  vmin=-0.3, vmax=0.3, rasterized=True, cmap='coolwarm')
axs[1].pcolormesh(diag1['T'], grid.Z[zsl], np.log10(diag1.KLeps[:, zsl, iy, 0]).T,
                  vmin=-11, vmax=-4, rasterized=True)

# %% MP downstream
iy = 2500
zsl = slice(79, 129)

fig, ax = plt.subplots(1, 1)
ax.plot(grid.Y, -grid.Depth)
ax.plot(grid.Y[iy].values*np.ones(grid.Z[zsl].shape), grid.Z[zsl])

fig, axs = plt.subplots(2, 1, figsize=(6.5, 4), sharex=True, sharey=True)
axs[0].pcolormesh(diag1['T'], grid.Z[zsl], diag1.VVEL[:, zsl, iy, 0].T,
                  vmin=-0.3, vmax=0.3, rasterized=True, cmap='coolwarm')
axs[1].pcolormesh(diag1['T'], grid.Z[zsl], np.log10(diag1.KLeps[:, zsl, iy, 0]).T,
                  vmin=-11, vmax=-4, rasterized=True)
