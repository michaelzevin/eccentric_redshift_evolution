"""
Function for calculating binary eccentricities at a set of reference frequencies
Requires dataframe with system masses and merger redshifts
"""

# --- Import packages --- #
import numpy as np
import pandas as pd

import time
import argparse
from tqdm import tqdm
import multiprocessing
from functools import partial

from astropy.cosmology import Planck18 as cosmo

from utilities import gw_calc


# --- Argument handling --- #
argp = argparse.ArgumentParser()
argp.add_argument("--data-path", type=str, required=True, help="Path to cluster data that contains binary masses, merger redshift, and semimajor axis/eccentricity at a point in the inspiral.")
argp.add_argument("--output-path", type=str, default='./eccentricities.hdf5', help="Path to write data. Default='./eccentricities.hdf5'")
argp.add_argument("--fLow", nargs="+", default=[10], help="Detector-frame frequencies at which to calculate eccentricities. Default='10'.")
argp.add_argument("--calc-with-no-redshifting", action='store_true', help="Determines whether to calculate eccentricities without redshifting. Default=False.")
argp.add_argument("--multiproc", type=int, help="Number of processors to parallelize on. By default, will parallelize on as many CPUs as available on machine.")
args = argp.parse_args()
fLow_vals = [int(f) for f in args.fLow]
mp = args.multiproc if args.multiproc else multiprocessing.cpu_count()


# --- Read in and process raw cluster data --- #
print("Reading and processing cluster data...\n")
data = pd.read_hdf(args.data_path, key='bbh')
# remove systems that don't merge within a Hubble time
data = data.loc[~data['z_mergers'].isna()]
# remove collisions where e_final=1
data = data.loc[data['e_final'] < 1.0]
# make the primary the more massive BH
data['primary_mass'] = np.where(data['m1'] >= data['m2'], data['m1'], data['m2'])
data['secondary_mass'] = np.where(data['m1'] >= data['m2'], data['m2'], data['m1'])
data['primary_spin'] = np.where(data['m1'] >= data['m2'], data['spin1'], data['spin2'])
data['secondary_spin'] = np.where(data['m1'] >= data['m2'], data['spin2'], data['spin1'])
# get luminosity distance of mergers for cosmological weights
data['luminosity_distance'] = np.asarray(cosmo.luminosity_distance(data['z_mergers']))
data['cosmo_weight'] = cosmo.differential_comoving_volume(data['z_mergers']) * (1+data['z_mergers'])**(-1)
# get rid of unnecesary columns
data = data[['primary_mass','secondary_mass','primary_spin','secondary_spin','z_mergers','a_final(AU)','e_final','channel','weights','cosmo_weight']]
data = data.rename(columns={'primary_mass':'m1', 'secondary_mass':'m2', 'primary_spin':'chi1', 'secondary_spin':'chi2', 'z_mergers':'z', 'a_final(AU)':'a0', 'e_final':'e0', 'weights':'cluster_weight'})


# --- Loop over reference frequencies --- #
for fLow in fLow_vals:
    print("Calculating eccentricities for fLow={:d}Hz...".format(fLow))
    print("  parallelizing over {:d} cores...".format(args.multiproc))
    start = time.time()

    if mp==1:
        ecc_vals = []
        for idx, samp in tqdm(data.iterrows(), total=len(data)):
            if args.calc_with_no_redshifting:
                ecc = gw_calc.eccentricity_at_eccentric_fLow(samp['m1'], samp['m2'], samp['a0'], samp['e0'], 0.0, fLow=fLow, eMax=0.999)
            else:
                ecc = gw_calc.eccentricity_at_eccentric_fLow(samp['m1'], samp['m2'], samp['a0'], samp['e0'], samp['z'], fLow=fLow, eMax=0.999)
            ecc_vals.append(ecc)
    else:
        # get systems in list format for parallelization
        if args.calc_with_no_redshifting:
            samps = [[x['m1'], x['m2'], x['a0'], x['e0'], 0.0] for _,x in data.iterrows()]
        else:
            samps = [[x['m1'], x['m2'], x['a0'], x['e0'], x['z']] for _,x in data.iterrows()]
        func = partial(gw_calc.eccentricity_at_eccentric_fLow_multiproc, fLow=fLow, eMax=0.999)
        pool = multiprocessing.Pool(mp)
        ecc_vals = pool.map(func, samps)

    ecc_vals = np.asarray(ecc_vals)
    key_string = 'e_{:s}Hz'.format(str(fLow))
    data[key_string] = ecc_vals

    end = time.time()
    print("  took {:0.1f}s for {:d} binaries!\n".format(end-start, len(data)))


# --- Save data --- #
print("Writing output file to {:s}".format(args.output_path))
data.to_hdf(args.output_path, key='bbh')
