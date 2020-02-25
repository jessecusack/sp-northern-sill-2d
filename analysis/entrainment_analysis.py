# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: spamex
#     language: python
#     name: spamex
# ---

# %% [markdown]
# # Entrainment and transformation of water masses in the model

# %%
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import scipy.interpolate as itpl
import ocean_tools.utils as utils


diag1_file = "diag1.glob.nc"
grid_file = "grid.glob.nc"

data_dir = "../no_tide/run1"
grid = xr.open_dataset(os.path.join(data_dir, grid_file))
data_dir = "../interface_height_down/run"
dat_down = xr.open_dataset(os.path.join(data_dir, diag1_file))
data_dir = "../interface_height_down50/run"
dat_down50 = xr.open_dataset(os.path.join(data_dir, diag1_file))
data_dir = "../interface_height_up/run"
dat_up = xr.open_dataset(os.path.join(data_dir, diag1_file))
data_dir = "../interface_height_up50/run"
dat_up50 = xr.open_dataset(os.path.join(data_dir, diag1_file))
data_dir = "../no_tide/run1"
dat = xr.open_dataset(os.path.join(data_dir, diag1_file))

# fix a problem a terrible way
dT = dat_down.THETA[0, -1, 0, 0] - dat.THETA[0, -1, 0, 0]
dat['THETA'] = dat.THETA + dT

# %% [markdown]
# Does out fix work? The lines should be the same at the bottom.

# %%
dats = [dat_down, dat_down50, dat_up, dat_up50, dat]
names = ['down 100 m', 'down 50 m', 'up 100 m', 'up 50 m', 'no change']

izs = 60

fig, ax = plt.subplots(1, 1)
for i, dat in enumerate(dats):
    dTdz = np.gradient(dat.THETA[0, izs:, 0, 0], grid.Z[izs:])
    ax.plot(dTdz, grid.Z[izs:], label=names[i])
    
ax.legend()

fig, ax = plt.subplots(1, 1)
for i, dat in enumerate(dats):
    ax.plot(dat.THETA[0, izs:, 0, 0], grid.Z[izs:], label=names[i])
    ax.plot(dat.THETA[0, izs:, -1, 0], grid.Z[izs:], label=names[i])
ax.legend()

# %% [markdown]
# ## Look at the no change case
#
# This bit of code take a little while to run.

# %%
Tc = 0.8
iy0 = 1000
iy1 = 2000
it0 = -200

Z = grid.Z.data
drF = grid.drF.data
# HFacC = grid.HFacC[:, :, 0].data
HFacS = grid.HFacS[:, :, 0].data
v_ = dat.VVEL[:, :, :, 0].data
# vc = 0.5*(v_[:, :, 1:]*HFacS[:, 1:] + v_[:, :, :-1]*HFacS[:, :-1])
v = v_[it0:, :, iy0:iy1].mean(axis=0)

T = dat.THETA[it0:, :, iy0:iy1, 0].data.mean(axis=0)
ino0 = T[:, 0] < Tc
ino1 = T[:, -1] < Tc

Vi = (drF*v[:, 0])[ino0].sum()
Vo = (drF*v[:, -1])[ino1].sum()

# %% [markdown]
# Area calculation. This bit of code is faster.

# %%
# Estimate area

y = grid.Y[iy0:iy1].data
z = np.zeros_like(y)
for iy in range(iy1-iy0):
    T_ = T[:, iy]
    use = (T_ > 0) & (T_ < Tc + 1)
    z[iy] = np.interp(Tc, np.flipud(T_[use]), np.flipud(Z[use]))
    
zs = utils.butter_filter(z, 1/1000, 1/20)

fig, ax = plt.subplots(1, 1)
C = ax.contourf((y-y[0])/1000, Z[70:], T[70:, :], np.arange(0.6, 1.15, 0.05), extend='both', cmap='inferno')
ax.fill_between((y-y[0])/1000, ax.get_ylim()[0], -grid.Depth[iy0:iy1, 0], color='grey')
# ax.plot(y, z, lw=3, zorder=2)
ax.plot((y-y[0])/1000, zs, lw=3, zorder=1)
cb = plt.colorbar(C)
cb.set_label('Temperature (deg C)')

A = np.sum(np.sqrt((y[1:] - y[:-1])**2 + (z[1:] - z[:-1])**2))
ax.set_xlabel('Distance (km)')
ax.set_ylabel('Height (m)')
fig.savefig('model_isotherms.png', dpi=300,
            bbox_inches='tight', pad_inches=0)

print("Length = {:1.0f} m".format(A))

# %%
# Entrainment velocity
w = (Vi - Vo)/A

print("Entrainment w* = {:1.1e} m/s".format(w))

# %% [markdown]
# Lets try this for a range of T.

# %%
Tcs = np.arange(0.66, 0.86, 0.01)

A = np.zeros_like(Tcs)
Vi = np.zeros_like(Tcs)
Vo = np.zeros_like(Tcs)

for i, Tc in enumerate(Tcs):
    ino0 = T[:, 0] < Tc
    ino1 = T[:, -1] < Tc

    Vi[i] = (drF*v[:, 0])[ino0].sum()
    Vo[i] = (drF*v[:, -1])[ino1].sum()
    
    z = np.zeros_like(y)
    for iy in range(iy1-iy0):
        T_ = T[:, iy]
        use = (T_ > 0) & (T_ < Tc + 1)
        z[iy] = np.interp(Tc, np.flipud(T_[use]), np.flipud(Z[use]))

    zs = utils.butter_filter(z, 1/1000, 1/20)
    
    A[i] = np.sum(np.sqrt((y[1:] - y[:-1])**2 + (z[1:] - z[:-1])**2))
    
Tr = Vi - Vo
w = Tr/A

fig, axs = plt.subplots(1, 4, sharey=True, figsize=(10, 10))
axs[0].plot(A, Tcs)
axs[1].plot(Vi, Tcs, label='Vi')
axs[1].plot(Vo, Tcs, label='Vo')
axs[2].plot(w, Tcs)
axs[3].plot(Tr, Tcs)

axs[0].set_xlabel('Length of contour (m)')
axs[0].set_ylabel('T (deg C)')
axs[1].set_xlabel('Area rate in/out (m^2/s)')
axs[2].set_xlabel('Entrainment velocity (m/s)')
axs[3].set_xlabel('Transformation rate (m^2/s)')

axs[1].legend()


# %% [markdown]
# Develop new coordinate system

# %%
idx = np.argmax(np.abs(np.gradient(dat_down.THETA[0, 60:100, 0, 0]))) + 60
ZT = itpl.interp1d(np.flipud(dat.THETA[0, :, 0, 0]), np.flipud(grid.Z - grid.Z[idx]))

plt.plot(np.arange(0.7, 1.1, 0.05), ZT(np.arange(0.7, 1.1, 0.05)))

# %% [markdown]
# ## Now lets do the same for all model runs and compare
#
# This takes a long time to run...

# %%
iy0 = 1000
iy1 = 2000
it0 = slice(250, 350)
Tcs = np.arange(0.7, 0.84, 0.005)
models = [dat_down, dat_down50, dat, dat_up50] #, dat_up]

A = np.zeros((len(Tcs), len(models)))
Vi = np.zeros((len(Tcs), len(models)))
Vo = np.zeros((len(Tcs), len(models)))

Z = grid.Z.data
drF = grid.drF.data
HFacS = grid.HFacS[:, :, 0].data
y = grid.Y[iy0:iy1].data

for j, model in enumerate(models):
    
    print("Model {}".format(j))

    v_ = model.VVEL[:, :, :, 0]
#     vc = 0.5*(v_[:, :, 1:]*HFacS[:, 1:] + v_[:, :, :-1]*HFacS[:, :-1])
    v = v_[it0, :, iy0:iy1].mean(axis=0).data
    T = model.THETA[it0, :, iy0:iy1, 0].mean(axis=0).data

    for i, Tc in enumerate(Tcs):
        ino0 = T[:, 0] < Tc
        ino1 = T[:, -1] < Tc

        Vi[i, j] = (drF*v[:, 0])[ino0].sum()
        Vo[i, j] = (drF*v[:, -1])[ino1].sum()

        z = np.zeros_like(y)
        for iy in range(iy1-iy0):
            T_ = T[:, iy]
            use = (T_ > 0) & (T_ < Tc + 1)
            z[iy] = np.interp(Tc, np.flipud(T_[use]), np.flipud(Z[use]))

        zs = utils.butter_filter(z, 1/1000, 1/20)

        A[i, j] = np.sum(np.sqrt((y[1:] - y[:-1])**2 + (z[1:] - z[:-1])**2))

Tr = Vi - Vo
w = Tr/A

# %% [markdown]
# Estimate transforms.

# %%
ZTs = []

for model in models:
    idx = np.argmax(np.abs(np.gradient(model.THETA[0, 60:100, 0, 0]))) + 60
    ZTs.append(itpl.interp1d(np.flipud(model.THETA[0, :, 0, 0]), np.flipud(grid.Z - grid.Z[idx])))

# %% [markdown]
# Plot up the variables.

# %%
mnames = ['- 100 m', '- 50 m', '0 m', '+ 50 m'] #, '+ 100 m']

fig, axs = plt.subplots(1, 4, sharey=True, figsize=(10, 10))

zcs = np.zeros((len(Tcs), len(models)))
for i in range(len(models)):
    zcs = ZTs[i](Tcs)

axs[0].plot(A, Tcs)
axs[1].plot(Vi, Tcs, label='Vi')
axs[1].plot(Vo, Tcs, label='Vo')
axs[2].plot(w, Tcs)
axs[3].plot(Tr, Tcs)

axs[0].set_xlabel('Length of contour (m)')
axs[0].set_ylabel('z')
axs[1].set_xlabel('Area rate in/out (m^2/s)')
axs[2].set_xlabel('Entrainment velocity (m/s)')
axs[3].set_xlabel('Transformation rate (m^2/s)')
axs[3].grid()

axs[0].legend(mnames)

# %%
fig, ax = plt.subplots(1, 1, figsize=(3, 4))
# axs[0].plot(Vi[:, (0, -1)], Tcs)
ax.plot(Tr[:, (0, -1)], Tcs)
ax.set_xlabel('Transformation rate (m^2/s)')
ax.set_ylabel('Temperature (deg C)')

ax.legend(['- 100 m',  '+ 50 m'])

fig.savefig('transformation.png', dpi=300,
            bbox_inches='tight', pad_inches=0)

# %%
fig, ax = plt.subplots(1, 1, figsize=(3, 4))
# axs[0].plot(Vi[:, (0, -1)], Tcs)
# ax.plot(np.diff(Tr[:, (0, -1)], axis=0), utils.mid(Tcs))
# ax.set_xlabel('Transformation rate (m^2/s)')
ax.set_ylabel('Temperature (deg C)')

ax.legend(['- 100 m',  '+ 50 m'])

Tbins = np.arange(0.7, 0.84, 0.02)
Tbinm = utils.mid(Tbins)
Tave, _, _ = utils.nan_binned_statistic(utils.mid(Tcs), np.diff(Tr[:, 0]), bins=Tbins)
ax.step(Tave, Tbinm, where='mid')
Tave, _, _ = utils.nan_binned_statistic(utils.mid(Tcs), np.diff(Tr[:, -1]), bins=Tbins)
ax.step(Tave, Tbinm, where='mid')

ax.legend(['- 100 m',  '+ 50 m'])

# fig.savefig('transformation.png', dpi=300,
#             bbox_inches='tight', pad_inches=0)

# %% [markdown]
# ## Volume transport comparison between models

# %%
mnames = ['- 100 m', '- 50 m', '0 m', '+ 50 m', '+ 100 m']

fig, ax = plt.subplots(1, 1, sharey=True, figsize=(10, 10))

zcs = np.zeros((len(Tcs), len(models)))
for i in range(len(models)):
    zcs = ZTs[i](Tcs)

ax.plot(Vi[:, :-1], zcs)

ax.set_ylabel('z (m)')
ax.set_xlabel('Area rate in/out (m^2/s)')

ax.legend(mnames[:-1])

# %%
dat_up50.VVEL[:, 50, 1500, 0]

# %%
