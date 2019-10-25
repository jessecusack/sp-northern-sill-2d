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
# # Generate mitgcm input data A6
# Note: Endianness of the binary files written here is not explicitly set and will depend on the machine you are working on. Use [numpy.dtype.newbyteorder](https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.newbyteorder.html) to explicitly set the endianness or compile mitgcm such that the machine default will be used.

# %%
# # %matplotlib notebook
# %matplotlib inline
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import gsw
import xarray as xr
import ocean_tools.utils as utils

data_dir = os.path.expanduser('~/data/SamoanPassage/Bathy/')
data_file = 'samoan_passage_bathymetry_200m_merged.mat'
B = utils.loadmat(os.path.join(data_dir, data_file))['bathy2']


# It seems like float64 is the 
# default data type in numpy, but let's be 
# clear about this and write a little 
# test/conversion function.
def CheckFloat64(x):
    if x.dtype == np.float64:
        print('its a float64')
    else:
        print('converting to float64')
        x = x.astype(np.float64)
    return x

if not os.path.exists('fig/'):
    os.mkdir('fig/')
    
# Basic geometry for this case is 100 vertical 
# levels and 4*750 horizontal levels, running 
# on 8 cores.
nx = 1
ny = 3000
nz = 130

# %% [markdown]
# ## Horizontal resolution

# %%
a = np.linspace(1000, 50, 500, dtype=np.float64)
b = np.squeeze(np.ones((1, 2000), dtype=np.float64))*20
c = np.linspace(50, 1000, 500, dtype=np.float64)

dy = np.hstack((a,b,c))
# unlike Matlab cumsum np.cumsum preserves input shape
y  = np.cumsum(dy)
y  = y - y[700]

# make sure dy is a float64
dy = CheckFloat64(dy)
# save to binary file
with open("delYvar", 'wb') as f:
    dy.tofile(f)

# %%
fig, ax = plt.subplots(2,1, sharex=True, figsize=(6,5))
ax[0].plot(dy, 'k.')
ax[0].set_title('cell size dy')
ax[0].set_ylabel('cell size [m]')

ax[1].plot(y/1000, 'k.')
ax[1].set_title('grid spacing y')
ax[1].set_ylabel('distance [km]')
ax[1].set_xlabel('grid index')

plt.tight_layout()
plt.savefig('fig/dy_and_y.pdf')

# %% [markdown]
# ## Vertical resolution

# %%
# from bottom up - start out with 20m resolution here
dz1 = np.ones(60)*20
rema = 5300 - np.sum(dz1)
xx = np.arange(1,71,1)+20
dz2 = xx*rema/np.sum(xx)
dz = np.hstack((dz1, dz2))
dz = np.flipud(dz)
z = np.cumsum(dz)
# make sure dz is in float64
dz = CheckFloat64(dz)
# save to binary file
with open("delZvar", 'wb') as f:
    dz.tofile(f)

# %%
fig, ax = plt.subplots(figsize=(4,4))
ax.plot(dz, -z, 'k.')
ax.set_ylabel('z [m]')
ax.set_xlabel('dz [m]')
plt.tight_layout()
plt.savefig('fig/dz.pdf')

# %% [markdown]
# ## Bottom topography

# %%
tmp = utils.loadmat('track_towyo104_long.mat')
track = tmp['track']
del tmp

# %%
from scipy.interpolate import interp1d
x = track['dist']
depth = -track['depth']
f = interp1d(x*1000, depth, bounds_error=False)
d2 = f(y)
d2[0:504] = -5082
d2[2524:] = d2[2524]

# make sure it's float64
d2 = CheckFloat64(d2)
# save to binary
with open("topogSamoa.bin", 'wb') as f:
    d2.tofile(f)

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharey=True)

ax[0].plot(y/1000,d2,'k')
ax[0].set_ylabel('z [m]')

ax[1].plot(y/1000, d2,'k');
ax[1].set(xlim=(-10,40))
for axi in ax:
    axi.set_xlabel('y [km]')
plt.savefig('fig/topo.pdf')

# %% [markdown]
# Adjust upstream botton depth to go to the same level as downstream.

# %%
y[504]/1e3

# %%
y[206]/1e3

# %%
from scipy.interpolate import interp1d
f = interp1d(y[[206, 504]], [d2[2524], d2[505]])

# %%
test = f(y[206:505])

# %%
plt.plot(y[206:505]/1000, test, 'k.')

# %%
d2old = d2.copy()
d2[206:505] = test
d2[0:206] = test[0]

# %%
# make sure it's float64
d2 = CheckFloat64(d2)
# save to binary
with open("topogSamoa.bin", 'wb') as f:
    d2.tofile(f)

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharey=True)

ax[0].plot(y/1000,d2,'k')
ax[0].set_ylabel('z [m]')

ax[1].plot(y/1000, d2,'k');
ax[1].set(xlim=(-10,40))
for axi in ax:
    axi.set_xlabel('y [km]')
plt.savefig('fig/topo.pdf')

# %% [markdown]
# Make sure the flat bottom connects nicely to the real bathymetry

# %%
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), sharey=True)
ax[0].plot(d2,'k');
ax[0].set(xlim=(300, 700))
ax[1].plot(d2,'k');
ax[1].set(xlim=(2400, 2600))


# %% [markdown]
# See how steppy the topography is at 20m vertical resolution (although I realize this doesn't make sense to look at since there are partially filled bottom cells).

# %%
xx = d2.copy()
bins = -z
inds = np.digitize(xx, bins)
d2d = -z[inds]

# %%
fig, ax = plt.subplots(figsize=(8,3))
[ax.axhline(y=yy, linewidth=0.2) for yy in np.arange(-5300,-4700,20)]
ax.plot(y/1000, d2d, 'k.')
ax.set_ylabel('z [m]')
ax.set_xlabel('y [km]')
ax.set_xlim([-110,110])
plt.tight_layout()
plt.savefig('fig/topo_resolution.pdf')

# %% [markdown]
# Find grid point of sill crest for stratification interpolation further below.

# %%
d2i = np.argmax(d2)
print(d2i)
print(d2[d2i])

# %% [markdown]
# ## Stratification
# Using a linear equation of state with $\alpha_T=2\times10^{-4}$.

# %%
gravity = 9.81;
talpha = 2.0e-4;

# %% [markdown]
# Upstream station 9-14, downstream 9-18

# %%
tmp = utils.loadmat('CTD0914.mat')
ctd1 = tmp['CTD']
del tmp

tmp = utils.loadmat('CTD0918.mat')
ctd2 = tmp['CTD']
del tmp

# %%
ctd1.keys()

# %% [markdown]
# Load a few more ctd profiles

# %%
c = xr.open_dataset('sp12_ctd.nc')

# %%
c

# %%
sec9 = c.where(c.sec==9, drop=True)

# %%
plt.contour(B['lon'], B['lat'], B['merged'], colors='k')
plt.plot(sec9.lon, sec9.lat, 'go')
tmp1 = sec9.th.where(sec9.station==12, drop=True)
tmp2 = sec9.th.where(sec9.station==19, drop=True)

plt.plot(tmp1.lon, tmp1.lat, 'ro')
plt.plot(tmp2.lon, tmp2.lat, 'yo')

# %%
sec9.station

# %% [markdown]
# ok, let's use stations 12 and 19 instead of 14 and 18 for a little more potential energy.

# %%
tmp = sec9.th.where(sec9.station==12, drop=True)
plt.plot(tmp, tmp.z, 'b')
tmp = sec9.th.where(sec9.station==19, drop=True)
plt.plot(tmp, tmp.z, 'r')

plt.plot(ctd1['theta1'], ctd1['z'],'b--')
plt.plot(ctd2['theta1'], ctd2['z'],'r--')
plt.gca().set(ylim=(5000, 3500), xlim=(0.67,1.25))

# %% [markdown]
# Calculate N$^2$ from the CTD data, smooth it and then translate it to the linear equation of state.
# $$
# N^2 = -\frac{g}{\rho} \frac{d \rho}{dz}
# $$
# $$
# \rho = \alpha_T \theta
# $$
# $$
# T_{z0} = \frac{N^2}{g \alpha_T}
# $$

# %%
zshift = +100 # shift peak by how many metres

tmp = sec9.where(sec9.station==12, drop=True)
SA = gsw.SA_from_SP(tmp.s.squeeze(), tmp.p.squeeze(), tmp.lon, tmp.lat)
CT = gsw.CT_from_t(SA, tmp.t.squeeze(), tmp.p.squeeze())
N2, pmid = gsw.Nsquared(SA, CT, tmp.p.squeeze(), tmp.lat)
zmid = utils.mid(tmp.z.data)
good = np.isfinite(N2)
N2smooth = utils.convolve_smooth(N2[good], 100)
zs = zmid[good]

imax, _ = sp.signal.find_peaks(N2smooth, height=(3e-6, 5e-6), prominence=2e-6)
hw = 220
ispeak = (zs > zs[imax-hw]) & (zs < zs[imax+hw])
N2filled = np.interp(zs, zs[~ispeak], N2smooth[~ispeak])
N2peak = (N2smooth - N2filled)
fpeak = sp.interpolate.interp1d(zs, N2peak, bounds_error=False, fill_value=(0, 0))
N2shifted = N2filled + fpeak(zs + zshift)

# %%
plt.plot(N2filled[3500:], zs[3500:])
plt.plot(N2smooth[3500:], zs[3500:], label='original')
plt.plot(N2shifted[3500:], zs[3500:], alpha=0.3, label='shifted')
plt.plot(N2smooth[imax], zs[imax], 'ro')
plt.plot(N2smooth[imax+hw], zs[imax+hw], 'go')
plt.plot(N2smooth[imax-hw], zs[imax-hw], 'go')
plt.gca().invert_yaxis()
plt.legend()
plt.xlabel('N squared')
plt.ylabel('Depth (m)')

# %%
f = sp.interpolate.interp1d(zs, N2shifted, bounds_error=False)
N2 = f(z)
# translate N2 to the linear equation of state
tz0 = N2/(gravity*talpha)
# integrate vertically
t = np.cumsum(-tz0*dz)
TrefS = t-t[0]+18.353
# replace bottom nan's with deepest value
ind = np.where(~np.isnan(TrefS))[0]
first, last = ind[0], ind[-1]
TrefS[last + 1:] = TrefS[last]
NrefS = N2

# %%
tmp = sec9.where(sec9.station==19, drop=True)
SA = gsw.SA_from_SP(tmp.s.squeeze(), tmp.p.squeeze(), tmp.lon, tmp.lat)
CT = gsw.CT_from_t(SA, tmp.t.squeeze(), tmp.p.squeeze())
N2, pmid = gsw.Nsquared(SA, CT, tmp.p.squeeze(), tmp.lat)
zmid = utils.mid(tmp.z.data)
good = np.isfinite(N2)
N2smooth = utils.convolve_smooth(N2[good], 100)
zs = zmid[good]

f = sp.interpolate.interp1d(zs, N2smooth, bounds_error=False)
N2 = f(z)
# translate N2 to the linear equation of state
tz0 = N2/(gravity*talpha)
# integrate vertically
t = np.cumsum(-tz0*dz)
TrefN = t-t[0]+19.294
# replace bottom nan's with deepest value
ind = np.where(~np.isnan(TrefN))[0]
first, last = ind[0], ind[-1]
TrefN[last + 1:] = TrefN[last]
NrefN = N2

# %%
fig, ax = plt.subplots(1,3, figsize=(9,4))
ax[0].plot(NrefS,z)
ax[0].plot(NrefN,z)
ax[0].set(ylim=(5500,2000), xlim=(-3e-7,7e-6))
ax[0].set_xlabel('Nsq')

ax[1].plot(TrefS,z)
ax[1].plot(TrefN,z)
ax[1].set(ylim=(5500,2000), xlim=(0.5, 2))
ax[1].set_xlabel('Temperature')

ax[2].plot(TrefS,z)
ax[2].plot(TrefN,z)
ax[2].set(ylim=(1000, 0), xlim=(2, 20))
ax[2].set_xlabel('Temperature')

# %%
tmp = sec9.where(sec9.station==12, drop=True)
print(tmp.th.min())
# print(np.nanmin(ctd1['theta1']))
print(np.nanmin(TrefS))

# %%
fig, ax = plt.subplots(1,4,sharey=True, figsize=(8,4))
ax[0].plot(N2,z)
ax[0].set_title('N$^2$')
ax[1].plot(tz0,z)
ax[1].set_title('T$_{Z0}$')
ax[2].plot(t,z)
ax[2].set_title('t')
ax[3].plot(TrefS,z)
ax[3].set_title('t$_{ref}$')
ax[0].invert_yaxis()
for axi in ax:
    axi.grid()
plt.tight_layout()
plt.savefig('fig/n2-tref.pdf')

# %%
fig, ax = plt.subplots(1,2)
ax[0].plot(TrefN,z)
ax[0].plot(TrefS,z)
ax[0].invert_yaxis()
ax[0].grid()

ax[1].plot(TrefN,z)
ax[1].plot(TrefS,z)
ax[1].invert_yaxis()
ax[1].grid()
ax[1].set_ylim(5300,4000)
ax[1].set_xlim(0.5,1.5)

# %% [markdown]
# Make the profiles the same in the upper layer

# %%
TrefN[z<4200-zshift] = TrefS[z<4200-zshift]
ijump = np.searchsorted(z, 4200-zshift)
DT = TrefN[ijump] - TrefN[ijump-1] - (TrefS[ijump-1] - TrefS[ijump-2])
TrefN[ijump:] -= DT

# %%
fig, ax = plt.subplots(1,2)
ax[0].plot(TrefN,z)
ax[0].plot(TrefS,z)
ax[0].invert_yaxis()
ax[0].grid()

ax[1].plot(TrefN,z)
ax[1].plot(TrefS,z)
ax[1].invert_yaxis()
ax[1].grid()
ax[1].set_ylim(5300,4000)
ax[1].set_xlim(0.5,1.5)

plt.tight_layout()
plt.savefig('fig/tref-profiles.pdf')

# %% [markdown]
# Save profiles for open boundary conditions

# %%
TrefN = CheckFloat64(TrefN)
# save to binary
with open("OB_North_T.bin", 'wb') as f:
    TrefN.tofile(f)
TrefS = CheckFloat64(TrefS)
# save to binary
with open("OB_South_T.bin", 'wb') as f:
    TrefS.tofile(f)

# %% [markdown]
# also save the southern profile as reference profile

# %%
with open("Tref", 'wb') as f:
    TrefS.tofile(f)

# %%
TrefN.shape

# %% [markdown]
# # Generate inital stratification

# %%
T = np.zeros((nz, ny))
for i, (ts, tn) in enumerate(zip(TrefS, TrefN)):
    # d2i is the index at the ridge crest
    f = sp.interpolate.interp1d(y[[0, d2i, 2600, 2999]], [ts, ts, tn, tn], bounds_error=False)
    T[i,:] = f(y)

# convert to 3D array (not sure if needed for 2D field, but nice to have for future cases)
# Tinit = np.zeros([nx,ny,nz])
# for k in np.arange(0,nx):
#     Tinit[k,:,:] = np.transpose(T[:,0:ny])

# this seems to work (not sure why it has to be nx, nz, ny)
Tinit2 = np.zeros([nx,nz,ny])
for k in np.arange(0,nx):
    Tinit2[k,:,:] = T[:,0:ny]

# Tinit = CheckFloat64(Tinit)
# # save to binary
# with open("T.init", 'wb') as f:
#     Tinit.tofile(f)
    
Tinit2 = CheckFloat64(Tinit2)
# save to binary
with open("T.init", 'wb') as f:
    Tinit2.tofile(f)

# %%
Tinit2.shape

# %%
fig, ax = plt.subplots()
h = ax.pcolormesh(Tinit2[0,:,:], vmin=0.6, vmax=1.1)
ax.contour(Tinit2[0,:,:], levels=np.arange(0.6,0.79,0.01), colors='k')
plt.colorbar(h)

# %%
fig, ax = plt.subplots()
cs = plt.contourf(y/1000, z, np.ma.masked_invalid(T), levels=[0.6, 0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,2,3,4,5,10,15,20], cmap='Spectral_r')
for c in cs.collections:
    c.set_edgecolor('face')
plt.contour(y/1000, z, np.ma.masked_invalid(T), levels=np.arange(0.6,0.8,0.01), colors='k', alpha=0.3)
plt.gca().invert_yaxis()
plt.colorbar(cs, label='T$_{init}$')
plt.plot(y/1000, -d2, 'k');
plt.xlabel('y [km]')
plt.ylabel('depth [m]')
plt.tight_layout()
plt.savefig('fig/Tinit.pdf')

# %%
fig, ax = plt.subplots(1, 1)
cs = plt.contourf(y/1000, z, np.ma.masked_invalid(T),
                  levels=np.arange(0.65,1.5,0.05),
                  cmap='Spectral_r')
for c in cs.collections:
    c.set_edgecolor('face')
ax.contour(y/1000, z, np.ma.masked_invalid(T), levels=np.arange(0.6,1.5,0.05), colors='k', alpha=0.3)
ax.invert_yaxis()
plt.colorbar(cs, label='T$_{init}$')
ax.fill_between(y/1000, np.abs(d2), np.ones_like(d2)*1e4, color='k');
ax.set(xlabel='y [km]', ylabel='depth [m]', xlim=(-25, 65), ylim=(5300, 3500))
ax.grid(False)
plt.tight_layout()
plt.savefig('fig/Tinit_zoom.pdf')
