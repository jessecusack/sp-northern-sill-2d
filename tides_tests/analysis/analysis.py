import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

data_dir = '../run1'
diag1_file = 'diag1.glob.nc'
grid_file = 'grid.glob.nc'
save_dir = 'figures'

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

# %% images for movie
plt.switch_backend("Agg")

ysl = slice(500, 2600)
zsl = slice(80, 130)

for it in range(diag1['T'].size):
    fig, axs = plt.subplots(3, 1, figsize=(6.5, 5), sharex=True, sharey=True)
    axs[0].pcolormesh(grid.Y[ysl], grid.Z[zsl], diag1.VVEL[it, zsl, ysl, 0],
                      vmin=-0.3, vmax=0.3, rasterized=True, cmap='coolwarm')
    axs[1].pcolormesh(grid.Y[ysl], grid.Z[zsl], diag1.WVEL[it, zsl, ysl, 0],
                      vmin=-0.1, vmax=0.1, rasterized=True, cmap='coolwarm')
    axs[2].pcolormesh(grid.Y[ysl], grid.Z[zsl], np.log10(diag1.KLeps[it, zsl, ysl, 0]),
                      vmin=-11, vmax=-4, rasterized=True)
    axs[0].set_title('{} {}'.format(diag1['T'][it].values, diag1['T'][it].units))
    name = '{:03d}.png'.format(it)
    fig.savefig(os.path.join(save_dir, name), bbox_inches='tight', pad_inches=0., dpi=200)

plt.switch_backend("Qt5Agg")

# %% MP on sill top
iy = 861
iz0 = 55
iz1 = 104
zsl = slice(iz0, iz1)
izs = np.arange(iz0, iz1, 9)

fig, ax = plt.subplots(1, 1)
ax.plot(grid.Y, -grid.Depth)
ax.plot(grid.Y[iy].values*np.ones(grid.Z[zsl].shape), grid.Z[zsl])
ax.plot(grid.Y[iy].values*np.ones(izs.shape), grid.Z[izs], 'o')

fig, axs = plt.subplots(2, 1, figsize=(6.5, 4), sharex=True, sharey=True)
axs[0].pcolormesh(diag1['T'], grid.Z[zsl], diag1.VVEL[:, zsl, iy, 0].T,
                  vmin=-0.3, vmax=0.3, rasterized=True, cmap='coolwarm')
axs[1].pcolormesh(diag1['T'], grid.Z[zsl], np.log10(diag1.KLeps[:, zsl, iy, 0]).T,
                  vmin=-11, vmax=-4, rasterized=True)


fig, ax = plt.subplots(1, 1, figsize=(6.5, 3))
for iz in izs:
    ax.semilogy(diag1['T'], diag1.KLeps[:, iz, iy, 0],
                label='{} {}'.format(grid.Z[iz].values, grid.Z.units))

ax.legend()

# %% MP just past sill top
iy = 1010
iz0 = 64
iz1 = 114
zsl = slice(iz0, iz1)
izs = np.arange(iz0, iz1, 9)

fig, ax = plt.subplots(1, 1)
ax.plot(grid.Y, -grid.Depth)
ax.plot(grid.Y[iy].values*np.ones(grid.Z[zsl].shape), grid.Z[zsl])
ax.plot(grid.Y[iy].values*np.ones(izs.shape), grid.Z[izs], 'o')

fig, axs = plt.subplots(2, 1, figsize=(6.5, 4), sharex=True, sharey=True)
axs[0].pcolormesh(diag1['T'], grid.Z[zsl], diag1.VVEL[:, zsl, iy, 0].T,
                  vmin=-0.3, vmax=0.3, rasterized=True, cmap='coolwarm')
axs[1].pcolormesh(diag1['T'], grid.Z[zsl], np.log10(diag1.KLeps[:, zsl, iy, 0]).T,
                  vmin=-11, vmax=-4, rasterized=True)


fig, ax = plt.subplots(1, 1, figsize=(6.5, 3))
for iz in izs:
    ax.plot(diag1['T'], diag1.KLeps[:, iz, iy, 0],
            label='{} {}'.format(grid.Z[iz].values, grid.Z.units))

ax.legend()

# %% MP on down slope
iy = 2000
iz0 = 74
iz1 = 124
zsl = slice(iz0, iz1)
izs = np.arange(iz0, iz1, 9)

fig, ax = plt.subplots(1, 1)
ax.plot(grid.Y, -grid.Depth)
ax.plot(grid.Y[iy].values*np.ones(grid.Z[zsl].shape), grid.Z[zsl])
ax.plot(grid.Y[iy].values*np.ones(izs.shape), grid.Z[izs], 'o')

fig, axs = plt.subplots(2, 1, figsize=(6.5, 4), sharex=True, sharey=True)
axs[0].pcolormesh(diag1['T'], grid.Z[zsl], diag1.VVEL[:, zsl, iy, 0].T,
                  vmin=-0.3, vmax=0.3, rasterized=True, cmap='coolwarm')
axs[1].pcolormesh(diag1['T'], grid.Z[zsl], np.log10(diag1.KLeps[:, zsl, iy, 0]).T,
                  vmin=-11, vmax=-4, rasterized=True)


fig, ax = plt.subplots(1, 1, figsize=(6.5, 3))
for iz in izs:
    ax.plot(diag1['T'], diag1.KLeps[:, iz, iy, 0],
            label='{} {}'.format(grid.Z[iz].values, grid.Z.units))

ax.legend()

# %% MP downstream
iy = 2500
iz0 = 79
iz1 = 129
zsl = slice(iz0, iz1)
izs = np.arange(iz0, iz1, 9)

fig, ax = plt.subplots(1, 1)
ax.plot(grid.Y, -grid.Depth)
ax.plot(grid.Y[iy].values*np.ones(grid.Z[zsl].shape), grid.Z[zsl])
ax.plot(grid.Y[iy].values*np.ones(izs.shape), grid.Z[izs], 'o')

fig, axs = plt.subplots(2, 1, figsize=(6.5, 4), sharex=True, sharey=True)
axs[0].pcolormesh(diag1['T'], grid.Z[zsl], diag1.VVEL[:, zsl, iy, 0].T,
                  vmin=-0.3, vmax=0.3, rasterized=True, cmap='coolwarm')
axs[1].pcolormesh(diag1['T'], grid.Z[zsl], np.log10(diag1.KLeps[:, zsl, iy, 0]).T,
                  vmin=-11, vmax=-4, rasterized=True)


fig, ax = plt.subplots(1, 1, figsize=(6.5, 3))
for iz in izs:
    ax.plot(diag1['T'], diag1.KLeps[:, iz, iy, 0],
            label='{} {}'.format(grid.Z[iz].values, grid.Z.units))

ax.legend()
