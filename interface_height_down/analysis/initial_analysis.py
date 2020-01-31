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
# # Upstream interface height decreased by 100 m

# %%
import numpy as np
import matplotlib.pyplot as plt
import xmitgcm
import MITgcmutils

# %%
T = np.squeeze(MITgcmutils.mds.rdmds('../run/*/T', 90))
hFacC = np.squeeze(MITgcmutils.mds.rdmds('../run/*/hFacC'))
topo = (hFacC == 0)
T[topo] = np.nan

# %%
fig, ax = plt.subplots(1, 1)
ax.pcolormesh(hFacC)

# %%
ds = xmitgcm.open_mdsdataset('../run/000*/')

# %%
