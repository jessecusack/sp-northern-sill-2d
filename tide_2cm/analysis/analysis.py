import os
import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as itr

matplotlib.rc('font', size=8)
matplotlib.rc('axes', titlepad=1)

data_dir = '../run'
diag1_file = 'diag1.glob.nc'
grid_file = 'grid.glob.nc'
save_dir = 'figures'

grid = xr.open_dataset(os.path.join(data_dir, grid_file))
diag1 = xr.open_dataset(os.path.join(data_dir, diag1_file))

# %% bathymetry
fig, ax = plt.subplots(1, 1)
ax.plot(grid.Y/1e3, -grid.Depth)

# %% towyo section
it = 500
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
    fig.savefig(os.path.join(save_dir, 'frames', name), bbox_inches='tight', pad_inches=0., dpi=200)
    plt.close(fig)

plt.switch_backend("Qt5Agg")

# %% movie of just velocity
plt.switch_backend("Agg")

ysl = slice(450, 2600)
zsl = slice(60, 130)

for it in range(diag1['T'].size):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3), sharex=True, sharey=True)
    C = ax.pcolormesh(grid.Y[ysl]/1000 - 270, -grid.Z[zsl], diag1.VVEL[it, zsl, ysl, 0],
                      vmin=-0.3, vmax=0.3, rasterized=True, cmap='coolwarm')
    cb = plt.colorbar(C, orientation='vertical')
    cb.set_label('Velocity (m s$^{-1}$)')
    ax.invert_yaxis()
    ax.set_title('{} {}'.format(diag1['T'][it].values, diag1['T'][it].units))
    name = 'vvel_{:03d}.png'.format(it)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Depth (m)')
    fig.savefig(os.path.join(save_dir, 'frames', name), bbox_inches='tight',
                pad_inches=0., dpi=200)
    plt.close(fig)

plt.switch_backend("Qt5Agg")

# %% movie of velocity and dissipation


def add_ext_cbar1(fig, ax, C, w=None, h=None, x0=None, y0=None):
    bbox = ax.get_position()
    if w is None:
        w = 0.05/fig.get_size_inches()[0]
    if h is None:
        h = (bbox.y1 - bbox.y0)
    if x0 is None:
        x0 = 1
    if y0 is None:
        y0 = bbox.y0
    cax = fig.add_axes((x0, y0, w, h))
    cb = plt.colorbar(C, cax, orientation='vertical')
    return cb


plt.switch_backend("Agg")

ysl = slice(450, 2600)
zsl = slice(60, 130)

for it in range(diag1['T'].size):
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True, sharey=True)
    C0 = axs[0].pcolormesh(grid.Y[ysl]/1000 - 270, -grid.Z[zsl], np.log10(diag1.KLeps[it, zsl, ysl, 0]),
                           vmin=-11, vmax=-4, rasterized=True)
    cb0 = add_ext_cbar1(fig, axs[0], C0, x0=0.92)
    cb0.set_label('$\log_{10} \epsilon$ (W kg$^{-1}$)')
    C1 = axs[1].pcolormesh(grid.Y[ysl]/1000 - 270, -grid.Z[zsl], diag1.VVEL[it, zsl, ysl, 0],
                           vmin=-0.3, vmax=0.3, rasterized=True, cmap='coolwarm')
    cb1 = add_ext_cbar1(fig, axs[1], C1, x0=0.92)
    cb1.set_label('Velocity (m s$^{-1}$)')
    axs[0].invert_yaxis()
    axs[0].set_title('{} {}'.format(diag1['T'][it].values, diag1['T'][it].units))
    name = 'vvel_diss_{:03d}.png'.format(it)
    axs[1].set_xlabel('Distance (km)')
    axs[0].set_ylabel('Depth (m)')
    axs[1].set_ylabel('Depth (m)')
    fig.savefig(os.path.join(save_dir, 'frames', name), bbox_inches='tight',
                pad_inches=0., dpi=200)
    plt.close(fig)

plt.switch_backend("Qt5Agg")

# %% MP type plots
plt.switch_backend("Agg")
iyl = np.arange(860, 2600, 20)
ibottom = np.squeeze(np.searchsorted(-grid.Z, grid.Depth))

for iy in iyl:
    iz1 = ibottom[iy]
    iz0 = iz1 - 40
    zsl = slice(iz0, iz1)
    izs = np.arange(iz0, iz1, 9)

    fig, ax = plt.subplots(1, 1)
    ax.plot(grid.Y, -grid.Depth)
    ax.plot(grid.Y[iy].values*np.ones(grid.Z[zsl].shape), grid.Z[zsl])
    ax.plot(grid.Y[iy].values*np.ones(izs.shape), grid.Z[izs], 'o')
    name = '{:04d}_MP_profile.png'.format(iy)
    fig.savefig(os.path.join(save_dir, name), bbox_inches='tight', pad_inches=0., dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(6.5, 4), sharex=True, sharey=True)
    axs[0].pcolormesh(diag1['T'], grid.Z[zsl], diag1.VVEL[:, zsl, iy, 0].T,
                      vmin=-0.3, vmax=0.3, rasterized=True, cmap='coolwarm')
    axs[1].pcolormesh(diag1['T'], grid.Z[zsl], np.log10(diag1.KLeps[:, zsl, iy, 0]).T,
                      vmin=-11, vmax=-4, rasterized=True)
    name = '{:04d}_MP_timeseries.png'.format(iy)
    fig.savefig(os.path.join(save_dir, name), bbox_inches='tight', pad_inches=0., dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3), sharex=True, sharey=True)
    ax.plot(diag1['T'], np.average(diag1.KLeps[:, zsl, iy, 0], axis=1))
    name = '{:04d}_MP_eps_int.png'.format(iy)
    fig.savefig(os.path.join(save_dir, name), bbox_inches='tight', pad_inches=0., dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3))
    for iz in izs:
        ax.plot(diag1['T'], diag1.KLeps[:, iz, iy, 0],
                label='{:1.0f} {}'.format(grid.Z[iz].values, grid.Z.units))

    ax.legend()
    name = '{:04d}_MP_eps.png'.format(iy)
    fig.savefig(os.path.join(save_dir, name), bbox_inches='tight', pad_inches=0., dpi=200)
    plt.close(fig)

plt.switch_backend("Qt5Agg")

# %% Compare with no tides
diag6 = xr.open_dataset('../../no_tide/run1/diag1.glob.nc')

iyl = [1010, 2000, 2500]
iz0l = [64, 74, 79]
iz1l = [114, 124, 129]

for iy, iz0, iz1 in zip(iyl, iz0l, iz1l):
    zsl = slice(iz0, iz1)
    izs = np.array([iz1-5])

    fig, ax = plt.subplots(1, 1)
    ax.plot(grid.Y, -grid.Depth)
    ax.plot(grid.Y[iy].values*np.ones(grid.Z[zsl].shape), grid.Z[zsl])
    ax.plot(grid.Y[iy].values*np.ones(izs.shape), grid.Z[izs], 'o')

    fig, axs = plt.subplots(2, 1, figsize=(6.5, 3), sharex=True,
                            gridspec_kw={'height_ratios': [1, 5]})

    for i, iz in enumerate(izs):
        axs[0].plot(diag1['T'], 2*np.sin(2*np.pi*diag1['T']/44640))
        axs[1].plot(diag1['T'], diag1.KLeps[:, iz, iy, 0], 'C{}-'.format(i),
                    label='tides')
        axs[1].plot(diag6['T'], diag6.KLeps[:, iz, iy, 0], 'C{}:'.format(i),
                    label='no tides')
        axs[0].set_title('{:1.0f} {}'.format(grid.Z[iz].values, grid.Z.units))

    axs[1].legend()

eps1m = diag1.KLeps.mean(axis=(1, 2, 3))
eps6m = diag6.KLeps.mean(axis=(1, 2, 3))
fig, axs = plt.subplots(3, 1, figsize=(6.5, 4), sharex=True,
                        gridspec_kw={'height_ratios': [1, 4, 4]})

axs[0].plot(diag1['T'], 2*np.sin(2*np.pi*diag1['T']/44640))
axs[1].plot(diag1['T'], eps1m, label='tides')
axs[1].plot(diag6['T'], eps6m, label='no tides')
axs[2].plot(diag1['T'], itr.cumtrapz(eps1m, diag1['T'], initial=0))
axs[2].plot(diag6['T'], itr.cumtrapz(eps6m, diag6['T'], initial=0))
axs[1].legend()

axs[0].set_ylabel('$v_{M2}$ (cm s$^{-1}$)')
axs[1].set_ylabel('Mean $\epsilon$\n(W kg$^{-1}$)')
axs[2].set_ylabel('Cumulative $\epsilon$\n(J kg$^{-1}$)')

name = 'tide_notide_comparison_mean_eps.png'
fig.savefig(os.path.join(save_dir, name), bbox_inches='tight', pad_inches=0.,
            dpi=200)
