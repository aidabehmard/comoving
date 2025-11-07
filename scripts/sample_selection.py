import pathlib
import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import h5py
import pandas as pd 

from astropy.table import Table, QTable
from astropy.io import fits

pd.options.mode.copy_on_write = True

# full Gaia dataset stored on FI servers
gaia_source_path = pathlib.Path("/mnt/ceph/users/gaia/dr3/hdf5/GaiaSource")
files = sorted(gaia_source_path.glob("*.hdf5"))

# load all Gaia data and store
dfs_gaia_source = []
for single_file in files[0:10]: # subset for now 
    gaia_source = Table()
    with h5py.File(single_file, "r") as f:
        for col in ["source_id", "ra", "ra_error", "dec", "dec_error", "parallax", "parallax_error",\
                    "pmra", "pmra_error", "pmdec", "pmdec_error", "phot_g_mean_mag", "phot_bp_mean_mag",\
                        "phot_rp_mean_mag", "radial_velocity", "radial_velocity_error"]:
            gaia_source[col] = np.array(f[col][:])
    df_gaia_source = gaia_source.to_pandas()
    dfs_gaia_source.append(df_gaia_source)

gaia_source = pd.concat(dfs_gaia_source,ignore_index=True)


# healpix level
# justified because pixel area radius is ~ ang sep. of two stars separated by 4pc at distance of 50 pc
level = 3
nside = hp.order2nside(level)
gaia_source["hp_pix_"+str(level)] = (gaia_source["source_id"] / (2**35 * 4**(12-level))).astype(int)


# El-Badry+2021 quality cuts - only apply to primaries
primaries = gaia_source[(gaia_source['parallax']>1) & (gaia_source['parallax_error']<2) & \
                          (gaia_source['phot_g_mean_mag']!=np.nan)]                     
primaries = primaries[primaries['parallax_error']/primaries['parallax']<0.2] 

primaries['Gmag'] = primaries['phot_g_mean_mag']-5*np.log10(1/(primaries['parallax']*1e-3))+5

primaries = primaries[np.isfinite(primaries['phot_bp_mean_mag']) & \
                          np.isfinite(primaries['phot_rp_mean_mag']) & np.isfinite(primaries['Gmag'])]


# get primary and potential companion indices
for i,row in primaries.iterrows():
    this_pix = int(row[f"hp_pix_{level}"]) # iterrows turns everything into float64, so have to recast
    other_pix = hp.get_all_neighbours(nside, this_pix, nest=True) # get nside=8 nearest pixels
    
    mask = np.isin(
        gaia_source[f"hp_pix_{level}"], np.concatenate(([this_pix], other_pix))
    )

    subset = gaia_source[mask]

    # ^^ how do we separate this into primaries and potential secondaries?