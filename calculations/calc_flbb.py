#!/usr/bin/env python

"""
Author: Lori Garzio on 4/25/2024
Last modified: 7/12/2024
Re-calculate flbb variables (chl-a, cdom, backscatter) with corrected calibration coefficients and write over the
existing files
"""

import argparse
import sys
import os
import glob
import datetime as dt
import xarray as xr
import numpy as np
import pandas as pd
pd.set_option('display.width', 320, "display.max_columns", 20)  # for display in pycharm console


def calculate_flbb(scale_factor, dark_counts, output):
    # calculate chlorophyll ug/L; beta700nm m-1sr-1; CDOM ppb
    value = scale_factor * (output - dark_counts)

    return value


def main(args):
    filedir = args.filedir
    level = args.level

    now = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    add_comment = f'Incorrect optics FLBBCD calibration coefficients were corrected and data for this variable were re-calculated on {now}'
    add_comment_cc = f'Incorrect optics FLBBCD calibration ceofficients were corrected on {now}'

    files = sorted(glob.glob(os.path.join(filedir, '*.nc')))
    for f in files:
        with xr.open_dataset(f) as ds:
            ds = ds.load()

        # variables that need to be re-calculated
        if level == 'sci-profile':
            variables = ['chlorophyll_a', 'beta_700nm', 'cdom']
        elif level == 'raw-trajectory':
            variables = ['sci_flbbcd_chlor_units', 'sci_flbbcd_bb_units', 'sci_flbbcd_cdom_units']

        # correct calibration coefficients for ru40-20240215T1642 FLBBCD SN 8632
        u_flbbcd_chlor_cwo = 21  # clean water offset, nodim == counts
        u_flbbcd_bb_cwo = 48  # clean water offset, nodim == counts
        u_flbbcd_cdom_cwo = 50  # clean water offset, nodim == counts
        u_flbbcd_chlor_sf = 0.0072  # scale factor to get units
        u_flbbcd_bb_sf = .000001772  # (0.000003522) scale factor to get units
        u_flbbcd_cdom_sf = 0.0919  # scale factor to get units

        # make a copy of the variables, then re-calculate the original variable with the correct coefficients
        for v in variables:
            ds[f'{v}_bad'] = ds[v].copy()

            if v in ['chlorophyll_a', 'sci_flbbcd_chlor_units']:
                cal_values = dict(
                    sf=u_flbbcd_chlor_sf,
                    cwo=u_flbbcd_chlor_cwo
                )
                raw = 'sci_flbbcd_chlor_sig'
                rnd = True
            elif v in ['beta_700nm', 'sci_flbbcd_bb_units']:
                cal_values = dict(
                    sf=u_flbbcd_bb_sf,
                    cwo=u_flbbcd_bb_cwo
                )
                raw = 'sci_flbbcd_bb_sig'
                rnd = False
            elif v in ['cdom', 'sci_flbbcd_cdom_units']:
                cal_values = dict(
                    sf=u_flbbcd_cdom_sf,
                    cwo=u_flbbcd_cdom_cwo
                )
                raw = 'sci_flbbcd_cdom_sig'
                rnd = True

            corrected_output = calculate_flbb(cal_values['sf'], cal_values['cwo'], ds[raw].values)

            if rnd:
                corrected_output = np.round(corrected_output, 4)

            ds[v].values = corrected_output

            if not hasattr(ds[v], 'comment'):
                ds[v].attrs['comment'] = add_comment
            else:
                ds[v].attrs['comment'] = f'{ds[v].comment} {add_comment}'

            if level == 'raw-trajectory':
                # fix the values for the calibration coefficients
                for cn, cal_value in cal_values.items():
                    calname = raw.replace('sci', 'u')
                    calname = calname.replace('sig', cn)
                    ds[f'{calname}_bad'] = ds[calname].copy()

                    non_nan_ind = np.where(~np.isnan(ds[calname].values))[0]
                    if 'Mnodim' in ds[calname].units:
                        cal_value = cal_value * 10e5
                    ds[calname][non_nan_ind] = cal_value

                    if not hasattr(ds[calname], 'comment'):
                        ds[calname].attrs['comment'] = add_comment_cc
                    else:
                        ds[calname].attrs['comment'] = f'{ds[calname].comment} {add_comment_cc}'

        if not hasattr(ds, 'history'):
            ds.attrs['history'] = f'{now}: {os.path.basename(__file__)}'
        else:
            ds.attrs['history'] = f'{ds.attrs["history"]} {now}: {os.path.basename(__file__)}'

        # Save the netcdf file with QC variables over the original file
        ds.to_netcdf(f)
        ds.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('filedir',
                            help='Full filepath to directory containing .nc files. '
                                 'e.g. /home/coolgroup/slocum/deployments/2024/ru40-20240215T1642/data/out/nc/sci-profile/delayed')

    arg_parser.add_argument('--level',
                            choices=['sci-profile', 'raw-trajectory'],
                            help='Dataset type')

    parsed_args = arg_parser.parse_args()

    sys.exit(main(parsed_args))
