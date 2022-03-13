#!/usr/bin/env python

"""
Author: lnazzaro and lgarzio on 3/9/2022
Last modified: lgarzio on 3/12/2022
Calculate optimal time shifts by segment for variables defined in config files (e.g. DO and pH voltages)
"""

import os
import argparse
import sys
import glob
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
from ioos_qc.utils import load_config_as_dict as loadconfig
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import polygonize
from rugliderqc.common import find_glider_deployment_datapath, find_glider_deployments_rootdir
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname
np.set_printoptions(suppress=True)


def apply_qc(dataset, varname):
    """
    Make a copy of a data array and convert values with not_evaluated (2) suspect (3) and fail (4) QC flags to nans.
    Convert pH voltage values of 0.0 to nan
    :param dataset: xarray dataset
    :param varname: sensor variable name (e.g. dissolved_oxygen)
    """
    datacopy = dataset[varname].copy()
    for qv in [x for x in dataset.data_vars if f'{varname}_qartod' in x]:
        qv_vals = dataset[qv].values
        qv_idx = np.where(np.logical_or(np.logical_or(qv_vals == 2, qv_vals == 3), qv_vals == 4))[0]
        datacopy[qv_idx] = np.nan
    if 'ph_ref_voltage' in varname:
        zeros = np.where(datacopy == 0.0)[0]
        datacopy[zeros] = np.nan
    return datacopy


def apply_time_shift(df, varname, shift_seconds):
    """
    Apply a specified time shift to a variable.
    :param df: pandas dataframe containing the variable of interest (varname), pressure, and time as the index
    :param varname: sensor variable name (e.g. dissolved_oxygen)
    :param shift_seconds: desired time shift in seconds
    :returns: pandas dataframe containing the time-shifted variable, pressure, and time as the index
    """
    # split off the variable and profile direction identifiers into a separate dataframe
    sdf = pd.DataFrame(dict(shifted_var=df[varname],
                            downs=df['downs']))

    # calculate the shifted timestamps
    tm_shift = df.index - dt.timedelta(seconds=shift_seconds)

    # append the shifted timestamps to the new dataframe and drop the original time index
    sdf['time_shift'] = tm_shift
    sdf.reset_index(drop=True, inplace=True)

    # rename the new columns and set the shifted timestamps as the index
    sdf = sdf.rename(columns={'time_shift': 'time',
                              'downs': 'downs_shifted'})
    sdf = sdf.set_index('time')

    # merge back into the original dataframe and drop rows with nans
    df2 = df.merge(sdf, how='outer', left_index=True, right_index=True)

    # drop the original variable
    df2.drop(columns=[varname, 'downs'], inplace=True)
    df2 = df2.rename(columns={'shifted_var': f'{varname}_shifted',
                              'downs_shifted': 'downs'})

    return df2


def calculate_pressure_range(df):
    """
    Calculate pressure range for a dataframe
    :param df: pandas dataframe containing pressure
    :returns: pressure range
    """
    min_pressure = np.nanmin(df.pressure)
    max_pressure = np.nanmax(df.pressure)

    return max_pressure - min_pressure


def identify_nans(dataset, varname):
    # identify where not nan
    non_nan_ind = np.invert(np.isnan(dataset[varname].values))
    # get locations of non-nans
    non_nan_i = np.where(non_nan_ind)[0]

    # identify where pressure is not nan
    press_non_nan_ind = np.where(np.invert(np.isnan(dataset.pressure.values)))[0]

    return non_nan_i, press_non_nan_ind


def interp_pressure(df):
    """
    Linear interpolate pressure in a time-shifted dataframe.
    :param df: pandas dataframe containing pressure and the time-shifted data, and time as the index
    :returns: pandas dataframe containing the time-shifted variable, interpolated pressure, and time as the index
    """
    # drop the original time index
    df['pressure'] = df['pressure'].interpolate(method='linear', limit_direction='both')

    return df


def pressure_bins(df, interval=0.25):
    """
    Bin data according to a specified depth interval, calculate median values for each bin.
    :param df: pandas dataframe containing pressure and the time-shifted data, and time as the index
    :param interval: optional pressure interval for binning, default is 0.25
    :returns: pandas dataframe containing depth-binned median data
    """
    # specify the bin intervals
    max_pressure = np.nanmax(df.pressure)
    bins = np.arange(0, max_pressure, interval).tolist()
    bins.append(bins[-1] + interval)

    # calculate the bin for each row
    df['bin'] = pd.cut(df['pressure'], bins)

    # calculate depth-binned median
    # used median instead of mean to account for potential unreasonable values not removed by QC
    df = df.groupby('bin').median()

    return df


def main(args):
    status = 0

    loglevel = args.loglevel.upper()
    cdm_data_type = args.cdm_data_type
    mode = args.mode
    dataset_type = args.level
    test = args.test

    logFile_base = logfile_basename()
    logging_base = setup_logger('logging_base', loglevel, logFile_base)

    data_home, deployments_root = find_glider_deployments_rootdir(logging_base, test)
    if isinstance(deployments_root, str):

        # Set the default qc configuration path
        qc_config_root = os.path.join(data_home, 'qc', 'config')
        if not os.path.isdir(qc_config_root):
            logging_base.warning('Invalid QC config root: {:s}'.format(qc_config_root))
            return 1

        for deployment in args.deployments:

            data_path, deployment_location = find_glider_deployment_datapath(logging_base, deployment, deployments_root,
                                                                             dataset_type, cdm_data_type, mode)

            if not data_path:
                logging_base.error('{:s} data directory not found:'.format(deployment))
                continue

            if not os.path.isdir(os.path.join(deployment_location, 'proc-logs')):
                logging_base.error('{:s} deployment proc-logs directory not found:'.format(deployment))
                continue

            logfilename = logfile_deploymentname(deployment, dataset_type, cdm_data_type, mode)
            logFile = os.path.join(deployment_location, 'proc-logs', logfilename)
            logging = setup_logger('logging', loglevel, logFile)

            logging.info('Calculating optimal time shift: {:s}'.format(os.path.join(data_path, 'qc_queue')))

            # Set the deployment qc configuration path
            deployment_location = data_path.split('/data')[0]
            deployment_qc_config_root = os.path.join(deployment_location, 'config', 'qc')
            if not os.path.isdir(deployment_qc_config_root):
                logging.warning('Invalid deployment QC config root: {:s}'.format(deployment_qc_config_root))

            # Get the variable names for time shifting from the config file for the deployment. If not provided,
            # optimal time shifts aren't calculated
            config_file = os.path.join(deployment_qc_config_root, 'time_shift.yml')
            if not os.path.isfile(config_file):
                logging.warning(
                    'Deployment config file not specified: {:s}. Time shifts not calculated.'.format(config_file))
                status = 1
                continue

            config_dict = loadconfig(config_file)
            shift_dict = config_dict['time_shift']

            # List the netcdf files
            ncfiles = sorted(glob.glob(os.path.join(data_path, 'qc_queue', '*.nc')))

            if len(ncfiles) == 0:
                logging.error(' 0 files found to QC: {:s}'.format(os.path.join(data_path, 'qc_queue')))
                status = 1
                continue

            # define shifts in seconds to test
            seconds = 60
            shifts = np.arange(0, seconds, 1).tolist()
            shifts.append(seconds)

            logging.info('Finding unique source files')

            # group the files by trajectory using the source file attribute
            source_files = []
            for f in ncfiles:
                try:
                    ds = xr.open_dataset(f)
                    source_file = ds.source_file.source_file
                except OSError as e:
                    logging.error('Error reading file {:s} ({:})'.format(f, e))
                    source_file = None
                    status = 1

                source_files.append(source_file)

            unique_source_files, source_file_idx = np.unique(source_files, return_index=True)
            source_file_idx = np.append(source_file_idx, len(ncfiles))
            source_file_idx = np.sort(source_file_idx)

            files_tested = 0

            for idx, sf_idx in enumerate(source_file_idx):
                if idx == 0:
                    continue
                ii = source_file_idx[idx - 1]

                groupfiles = ncfiles[ii:sf_idx]

                for key, values in shift_dict.items():
                    shift_dict[key] = dict(shift=np.nan, t0=0, tf=0)

                # Iterate through the test variables
                for testvar in shift_dict:
                    times = np.array([], dtype='datetime64[ns]')

                    # Iterate through profile files in each trajectory, define profile direction and append to df
                    trajectory = pd.DataFrame()
                    for f in groupfiles:
                        try:
                            ds = xr.open_dataset(f)
                        except OSError as e:
                            logging.error('Error reading file {:s} ({:})'.format(f, e))
                            status = 1
                            continue

                        try:
                            ds[testvar]
                        except KeyError:
                            logging.error('{:s} not found in file {:s})'.format(testvar, f))
                            status = 1
                            continue

                        times = np.append(times, ds.time.values)

                        data_idx, pressure_idx = identify_nans(ds, testvar)

                        if len(data_idx) == 0:
                            logging.error('{:s} data not found in file {:s})'.format(testvar, f))
                            status = 1
                            continue

                        # make a copy of the data and apply QARTOD QC flags
                        data_copy = apply_qc(ds, testvar)

                        # convert to dataframe with pressure
                        df = data_copy.to_dataframe().merge(ds.pressure.to_dataframe(), on='time')
                        df = df.dropna(how='all')

                        # determine if profile is up or down, append to appropriate dataframe
                        if ds.pressure.values[pressure_idx][0] > ds.pressure.values[pressure_idx][-1]:
                            # up cast
                            df['downs'] = 0
                        else:
                            # down cast
                            df['downs'] = 1
                        trajectory = trajectory.append(df)

                    min_time = pd.to_datetime(np.nanmin(times)).strftime('%Y-%m-%dT%H:%M:%S')
                    max_time = pd.to_datetime(np.nanmax(times)).strftime('%Y-%m-%dT%H:%M:%S')

                    # check the pressure range
                    trajectory_pressure_range = calculate_pressure_range(trajectory)

                    # don't calculate area if the trajectory pressure range is <3 dbar
                    if trajectory_pressure_range < 3:
                        logging.info('Profile data spans <3 dbar, optimal time shift not calculated'
                                     ' for {} {} to {}'.format(testvar, min_time, max_time))

                        shift_dict[testvar]['shift'] = np.nan

                        # add start and end times
                        shift_dict[testvar]['t0'] = min_time
                        shift_dict[testvar]['tf'] = max_time
                    else:
                        # gets rid of duplicates and syncs the dataframes so they can be merged when time-shifted
                        trajectory_resample = trajectory.resample('1s').mean()

                        # For each shift, shift the master dataframes by x seconds, bin data by 0.25 dbar,
                        # calculate area between curves
                        areas = []
                        for shift in shifts:
                            trajectory_shift = apply_time_shift(trajectory_resample, testvar, shift)
                            trajectory_interp = interp_pressure(trajectory_shift)
                            trajectory_interp.dropna(subset=[f'{testvar}_shifted'], inplace=True)

                            # find down identifiers that were averaged in the resampling and reset
                            downs = np.array(trajectory_interp['downs'])
                            ind = np.argwhere(downs == 0.5).flatten()
                            downs[ind] = downs[ind - 1]
                            trajectory_interp['downs'] = downs

                            # after time shifting and interpolating pressure, divide df into down and up profiles
                            downs_df = trajectory_interp[trajectory_interp['downs'] == 1].copy()
                            ups_df = trajectory_interp[trajectory_interp['downs'] == 0].copy()

                            # check the pressure range
                            downs_pressure_range = calculate_pressure_range(downs_df)
                            ups_pressure_range = calculate_pressure_range(ups_df)

                            # don't calculate area if either profile grouping spans <3 dbar
                            if np.logical_or(downs_pressure_range < 3, ups_pressure_range < 3):
                                area = np.nan
                            else:
                                # bin data frames
                                downs_binned = pressure_bins(downs_df)
                                downs_binned.dropna(inplace=True)
                                ups_binned = pressure_bins(ups_df)
                                ups_binned.dropna(inplace=True)

                                downs_ups = downs_binned.append(ups_binned.iloc[::-1])

                                # calculate area between curves
                                polygon_points = downs_ups.values.tolist()
                                polygon_points.append(polygon_points[0])
                                polygon = Polygon(polygon_points)
                                polygon_lines = polygon.exterior
                                polygon_crossovers = polygon_lines.intersection(polygon_lines)
                                polygons = polygonize(polygon_crossovers)
                                valid_polygons = MultiPolygon(polygons)
                                area = valid_polygons.area

                            areas.append(area)

                        # add start and end times
                        shift_dict[testvar]['t0'] = min_time
                        shift_dict[testvar]['tf'] = max_time

                        # if >50% of the values are nan, return nan
                        fraction_nan = np.sum(np.isnan(areas)) / len(areas)
                        if fraction_nan > .5:
                            shift_dict[testvar]['shift'] = np.nan

                            logging.info('Optimal time shift for {} {} to {}: undetermined'.format(testvar,
                                                                                                   min_time,
                                                                                                   max_time))
                        else:
                            # find the shift that results in the minimum area between the curves
                            shift_dict[testvar]['shift'] = np.nanargmin(areas)

                            logging.info('Optimal time shift for {} {} to {}: {} sec'.format(testvar,
                                                                                             min_time,
                                                                                             max_time,
                                                                                             np.nanargmin(areas)))

                # add the optimal time shifts back into the .nc files
                for f in groupfiles:
                    try:
                        with xr.open_dataset(f) as ds:
                            ds = ds.load()
                    except OSError as e:
                        logging.error('Error reading file {:s} ({:})'.format(f, e))
                        status = 1
                        continue

                    for testvar, items in shift_dict.items():
                        try:
                            data = ds[testvar]
                        except KeyError:
                            continue
                        shift_varname = f'{testvar}_optimal_shift'

                        shift_vals = items['shift'] * np.ones(np.shape(data.values))

                        comment = 'Optimal time shift (seconds) determined by grouping down and up profiles for one' \
                                  'segment between {} and {}, then minimizing the area between the down/up profiles ' \
                                  'by testing time shifts between 0 and {} seconds'.format(items['t0'],
                                                                                           items['tf'],
                                                                                           seconds)

                        # set attributes
                        attrs = {
                            'comment': comment,
                            'units': 'sec',
                            'valid_min': 0,
                            'valid_max': seconds,
                            'qc_target': testvar
                        }

                        # add variable to the original dataset
                        da = xr.DataArray(shift_vals.astype('float32'), coords=data.coords, dims=data.dims,
                                          name=shift_varname, attrs=attrs)
                        ds[shift_varname] = da

                    ds.to_netcdf(f)
                    files_tested += 1

            logging.info('{}: {} of {} files tested.'.format(deployment, files_tested, len(ncfiles)))

    return status


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=main.__doc__,
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    arg_parser.add_argument('deployments',
                            nargs='+',
                            help='Glider deployment name(s) formatted as glider-YYYYmmddTHHMM')

    arg_parser.add_argument('-m', '--mode',
                            help='Deployment dataset status',
                            choices=['rt', 'delayed'],
                            default='rt')

    arg_parser.add_argument('--level',
                            choices=['sci', 'ngdac'],
                            default='sci',
                            help='Dataset type')

    arg_parser.add_argument('-d', '--cdm_data_type',
                            help='Dataset type',
                            choices=['profile'],
                            default='profile')

    arg_parser.add_argument('-l', '--loglevel',
                            help='Verbosity level',
                            type=str,
                            choices=['debug', 'info', 'warning', 'error', 'critical'],
                            default='info')

    arg_parser.add_argument('-test', '--test',
                            help='Point to the environment variable key GLIDER_DATA_HOME_TEST for testing.',
                            action='store_true')

    parsed_args = arg_parser.parse_args()

    sys.exit(main(parsed_args))
