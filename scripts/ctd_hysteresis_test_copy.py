#!/usr/bin/env python

"""
Author: lnazzaro and lgarzio on 12/7/2021
Last modified: lgarzio on 3/1/2022
Flag CTD profile pairs that are severely lagged, which can be an indication of CTD pump issues.
"""

import os
import argparse
import sys
import glob
import numpy as np
import xarray as xr
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import polygonize
from ioos_qc import qartod
from ioos_qc.utils import load_config_as_dict as loadconfig
from rugliderqc.common import find_glider_deployment_datapath, find_glider_deployments_rootdir
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname
np.set_printoptions(suppress=True)


def append_ancillary_variables(data_array, qc_variable_name):
    """
    Add the QC test to the associated ancillary variable attribute
    :param data_array: xarray data array  (e.g. ds.conductivity)
    :param qc_variable_name: name of the QC variable to be added to the dataset (e.g. conductivity_hysteresis_test)
    """
    if not hasattr(data_array, 'ancillary_variables'):
        data_array.attrs['ancillary_variables'] = qc_variable_name
    else:
        data_array.attrs['ancillary_variables'] = ' '.join((data_array.ancillary_variables, qc_variable_name))


def apply_qartod_qc(dataset, varname):
    """
    Make a copy of a data array and convert values with not_evaluated (2) suspect (3) and fail (4) QC flags to nans
    :param dataset: xarray dataset
    :param varname: sensor variable name (e.g. conductivity)
    """
    datacopy = dataset[varname].copy()
    for qv in [x for x in dataset.data_vars if f'{varname}_qartod' in x]:
        qv_vals = dataset[qv].values
        qv_idx = np.where(np.logical_or(np.logical_or(qv_vals == 2, qv_vals == 3), qv_vals == 4))[0]
        #qv_idx = np.where(np.logical_or(dataset[qv].values == 3, dataset[qv].values == 4))[0]
        datacopy[qv_idx] = np.nan
    return datacopy


def initialize_flags(dataset, varname):
    # start with flag values NOT_EVALUATED/UNKNOWN (2)
    flags = 2 * np.ones(np.shape(dataset[varname].values))

    # identify where not nan
    non_nan_ind = np.invert(np.isnan(dataset[varname].values))
    # get locations of non-nans
    non_nan_i = np.where(non_nan_ind)[0]

    # flag the missing values
    flags[np.invert(non_nan_ind)] = qartod.QartodFlags.MISSING

    # identify where pressure is not nan
    press_non_nan_ind = np.where(np.invert(np.isnan(dataset.pressure.values)))[0]

    return non_nan_i, press_non_nan_ind, flags


def add_da(dataset, flag_array, attributes, test_varname, qc_variable_name):
    """
    Add QC test results to the original dataset
    :param dataset: xarray dataset
    :param flag_array: numpy array containing QC results
    :param attributes: dictionary containing variable attributes
    :param test_varname: sensor variable name (e.g. conductivity)
    :param qc_variable_name: name of the QC variable to be added to the dataset (e.g. conductivity_hysteresis_test)
    """
    da = xr.DataArray(flag_array.astype('int32'), coords=dataset[test_varname].coords, dims=dataset[test_varname].dims,
                      name=qc_variable_name, attrs=attributes)
    dataset[qc_variable_name] = da


def set_hysteresis_attrs(test, sensor, thresholds=None):
    """
    Define the QC variable attributes for the CTD hysteresis test
    :param test: QC test
    :param sensor: sensor variable name (e.g. conductivity)
    :param thresholds: optional flag thresholds from QC configuration file
    """
    thresholds = thresholds or None

    flag_meanings = 'GOOD NOT_EVALUATED SUSPECT FAIL MISSING'
    flag_values = [1, 2, 3, 4, 9]
    standard_name = f'{test}_quality_flag'
    if 'ctd' in test:
        long_name = 'CTD Hysteresis Test Quality Flag'
    else:
        long_name = f'{sensor.capitalize()} Hysteresis Test Quality Flag'
    comment = 'Test for {} lag, determined by comparing the area between profile pairs normalized to pressure ' \
              'range against the data range multiplied by thresholds found in flag_configurations.'.format(sensor)

    # Defining QC variable attributes
    attrs = {
        'comment': comment,
        'standard_name': standard_name,
        'long_name': long_name,
        'flag_values': np.byte(flag_values),
        'flag_meanings': flag_meanings,
        'valid_min': np.byte(min(flag_values)),
        'valid_max': np.byte(max(flag_values)),
        'qc_target': sensor,
    }

    if thresholds:
        attrs['flag_configurations'] = str(thresholds)

    return attrs


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

            logging.info('Checking for CTD sensor lag: {:s}'.format(os.path.join(data_path, 'qc_queue')))

            # Set the deployment qc configuration path
            deployment_location = data_path.split('/data')[0]
            deployment_qc_config_root = os.path.join(deployment_location, 'config', 'qc')
            if not os.path.isdir(deployment_qc_config_root):
                logging.warning('Invalid deployment QC config root: {:s}'.format(deployment_qc_config_root))

            # Get the test thresholds from the config file for the deployment (if available) or the default
            config_file = os.path.join(deployment_qc_config_root, 'ctd_hysteresis.yml')
            if not os.path.isfile(config_file):
                logging.warning('Deployment config file not specified: {:s}. Using default config.'.format(config_file))
                config_file = os.path.join(qc_config_root, 'ctd_hysteresis.yml')
                if not os.path.isfile(config_file):
                    logging.error('Invalid default config file: {:s}.'.format(config_file))
                    status = 1
                    continue

            logging.info('Using config file: {:s}'.format(config_file))
            config_dict = loadconfig(config_file)

            # List the netcdf files
            ncfiles = sorted(glob.glob(os.path.join(data_path, 'qc_queue', '*.nc')))

            if len(ncfiles) == 0:
                logging.error(' 0 files found to QC: {:s}'.format(os.path.join(data_path, 'qc_queue')))
                status = 1
                continue

            test_varnames = ['conductivity', 'temperature']

            # build the summary
            summary = dict()
            for tv in test_varnames:
                summary[tv] = dict()
                summary[tv]['failed_profiles'] = 0
                summary[tv]['suspect_profiles'] = 0
                summary[tv]['not_evaluated_profiles'] = 0

            # Iterate through files
            skip = 0
            for i, f in enumerate(ncfiles):
                # skip the files that have already been QC'd
                try:
                    if f2skip > 0:
                        skip += 1
                except UnboundLocalError:
                    skip += 0

                i += skip

                # TODO end the code when all the files have been tested
                if i >= len(ncfiles):
                    continue

                try:
                    with xr.open_dataset(ncfiles[i]) as ds:
                        ds = ds.load()
                except OSError as e:
                    logging.error('Error reading file {:s} ({:})'.format(ncfiles[i], e))
                    status = 1
                    continue
                except IndexError:
                    continue

                f2skip = 0

                # Iterate through the test variables
                for testvar in test_varnames:
                    # get the configuration thresholds
                    hysteresis_thresholds = config_dict[f'{testvar}_hysteresis_test']

                    try:
                        ds[testvar]
                    except KeyError:
                        logging.error('{:s} not found in file {:s})'.format(testvar, ncfiles[i]))
                        status = 1
                        continue

                    qc_varname = f'{testvar}_hysteresis_test'
                    kwargs = dict()
                    kwargs['thresholds'] = hysteresis_thresholds
                    attrs = set_hysteresis_attrs(qc_varname, testvar, **kwargs)
                    data_idx, pressure_idx, flag_vals = initialize_flags(ds, testvar)

                    if len(data_idx) == 0:
                        logging.error('{:s} data not found in file {:s})'.format(testvar, ncfiles[i]))
                        status = 1
                        continue

                    # determine if first profile is up or down
                    if ds.pressure.values[pressure_idx][0] > ds.pressure.values[pressure_idx][-1]:
                        # if profile is up, test can't be run because you need a down profile paired with an up profile
                        # leave flag values as NOT_EVALUATED/UNKNOWN (2) and set the attributes
                        add_da(ds, flag_vals, attrs, testvar, qc_varname)
                        summary[testvar]['not_evaluated_profiles'] += 1
                    else:  # first profile is down, check the next file
                        try:
                            f2 = ncfiles[i + 1]
                        except IndexError:
                            # if there are no more files, leave flag values on the first file as
                            # NOT_EVALUATED/UNKNOWN (2) and set the attributes
                            add_da(ds, flag_vals, attrs, testvar, qc_varname)

                            # add the hysteresis test to ancillary variable attribute
                            append_ancillary_variables(ds[testvar], qc_varname)

                            # add the hysteresis test to the salinity and density ancillary variable attribute
                            for v in ['salinity', 'density']:
                                append_ancillary_variables(ds[v], qc_varname)

                            summary[testvar]['not_evaluated_profiles'] += 1
                            continue

                        try:
                            # see if the second file is already open
                            ds2
                        except NameError:
                            # if not, try to open the second file
                            try:
                                with xr.open_dataset(f2) as ds2:
                                    ds2 = ds2.load()
                            except OSError as e:
                                logging.error('Error reading file {:s} ({:})'.format(f2, e))
                                status = 1
                                f2skip += 1

                        try:
                            ds2[testvar]
                        except KeyError:
                            logging.error('{:s} not found in file {:s})'.format(testvar, f2))
                            status = 1
                            # TODO should we be checking the next file? example ru30_20210510T015902Z_sbd.nc
                            # leave flag values on the first file as NOT_EVALUATED/UNKNOWN (2) and set the attributes
                            add_da(ds, flag_vals, attrs, testvar, qc_varname)
                            summary[testvar]['not_evaluated_profiles'] += 1
                            pass  ## check this

                        data_idx2, pressure_idx2, flag_vals2 = initialize_flags(ds2, testvar)

                        # determine if second profile is up or down
                        if ds2.pressure.values[pressure_idx2][0] < ds2.pressure.values[pressure_idx2][-1]:
                            # if second profile is also down, test can't be run on the first file
                            # leave flag values on the first file as NOT_EVALUATED/UNKNOWN (2) and set the attributes
                            add_da(ds, flag_vals, attrs, testvar, qc_varname)
                            summary[testvar]['not_evaluated_profiles'] += 1
                        else:
                            # first profile is down and second profile is up
                            # determine if the end/start timestamps are < 5 minutes apart,
                            # indicating a paired yo (down-up profile pair)
                            if ds2.time.values[0] - ds.time.values[-1] < np.timedelta64(5, 'm'):

                                # make a copy of the data and apply QARTOD QC flags before testing for hysteresis
                                data_copy = apply_qartod_qc(ds, testvar)
                                data_copy2 = apply_qartod_qc(ds2, testvar)

                                # both yos must have data remaining after QARTOD flags are applied,
                                # otherwise, test can't be run and leave the flag values as NOT_EVALUATED/UNKNOWN (2)
                                if np.logical_and(np.sum(~np.isnan(data_copy)) > 0, np.sum(~np.isnan(data_copy2)) > 0):
                                    # calculate the area between the two profiles
                                    df = data_copy.to_dataframe().merge(ds.pressure.to_dataframe(), on='time')
                                    df2 = data_copy2.to_dataframe().merge(ds2.pressure.to_dataframe(), on='time')
                                    df = df.append(df2)
                                    df = df.dropna(subset=['pressure', testvar])

                                    # convert negative pressure values to 0
                                    pressure_copy = df.pressure.values.copy()
                                    pressure_copy[pressure_copy < 0] = 0
                                    pressure_range = (np.nanmax(pressure_copy) - np.nanmin(pressure_copy))
                                    data_range = (np.nanmax(df[testvar].values) - np.nanmin(df[testvar].values))

                                    # If the profile depth range is >5 dbar and the data range is >test_threshold,
                                    # run the test. If profile depth is <5 dbar leave flags NOT_EVALUATED/UNKNOWN (2)
                                    # since hysteresis can't be calculated with a profile that doesn't span a
                                    # substantial depth range (e.g. usually hovering at the surface or bottom)
                                    if pressure_range > 5:
                                        if data_range > hysteresis_thresholds['test_threshold']:
                                            polygon_points = df.values.tolist()
                                            polygon_points.append(polygon_points[0])
                                            polygon = Polygon(polygon_points)
                                            polygon_lines = polygon.exterior
                                            polygon_crossovers = polygon_lines.intersection(polygon_lines)
                                            polygons = polygonize(polygon_crossovers)
                                            valid_polygons = MultiPolygon(polygons)

                                            # normalize area between the profiles to the pressure range
                                            area = valid_polygons.area / pressure_range

                                            # Flag failed profiles
                                            if area > data_range * hysteresis_thresholds['fail_threshold']:
                                                flag = qartod.QartodFlags.FAIL
                                                summary[testvar]['failed_profiles'] += 2
                                            # Flag suspect profiles
                                            elif area > data_range * hysteresis_thresholds['suspect_threshold']:
                                                flag = qartod.QartodFlags.SUSPECT
                                                summary[testvar]['suspect_profiles'] += 2
                                            # Otherwise, both profiles are good
                                            else:
                                                flag = qartod.QartodFlags.GOOD
                                            flag_vals[data_idx] = flag
                                            flag_vals2[data_idx2] = flag
                                        else:
                                            # if data range is < test_threshold, set flags to 1 (GOOD) since
                                            # there will be no measureable hysteresis (usually in well-mixed water)
                                            flag = qartod.QartodFlags.GOOD
                                            flag_vals[data_idx] = flag
                                            flag_vals2[data_idx2] = flag

                                    # add data array with hysteresis flag applied
                                    # (or flag values = NOT_EVALUATED/UNKNOWN (2) if the profile depth range is <5 dbar)
                                    add_da(ds, flag_vals, attrs, testvar, qc_varname)
                                    add_da(ds2, flag_vals2, attrs, testvar, qc_varname)
                                    if 2. in flag_vals:
                                        summary[testvar]['not_evaluated_profiles'] += 2
                                    f2skip += 1

                                else:
                                    # if there is no data left after QARTOD tests are applied,
                                    # leave flag values NOT_EVALUATED/UNKNOWN (2)
                                    add_da(ds, flag_vals, attrs, testvar, qc_varname)
                                    add_da(ds2, flag_vals2, attrs, testvar, qc_varname)
                                    summary[testvar]['not_evaluated_profiles'] += 2
                                    f2skip += 1
                            else:
                                # if timestamps are too far apart they're likely not from the same profile pair
                                # leave flag values as NOT_EVALUATED/UNKNOWN (2) and set the attributes
                                add_da(ds, flag_vals, attrs, testvar, qc_varname)
                                add_da(ds2, flag_vals2, attrs, testvar, qc_varname)
                                summary[testvar]['not_evaluated_profiles'] += 2
                                f2skip += 1

                    # add the hysteresis test to ancillary variable attribute
                    append_ancillary_variables(ds[testvar], qc_varname)
                    try:
                        check = ds2[qc_varname]  # check that the qc variable is in the dataset
                        append_ancillary_variables(ds2[testvar], qc_varname)
                    except (KeyError, NameError):
                        pass

                    # add the hysteresis test to the salinity and density ancillary variable attribute
                    for v in ['salinity', 'density']:
                        append_ancillary_variables(ds[v], qc_varname)
                        try:
                            check = ds2[qc_varname]  # check that the qc variable is in the dataset
                            append_ancillary_variables(ds2[v], qc_varname)
                        except (KeyError, NameError):
                            pass

                # save the dataset(s)
                ds.to_netcdf(ncfiles[i])
                del ds
                try:
                    ds2.to_netcdf(f2)
                    del ds2
                except NameError:
                    pass

            for tv in test_varnames:
                tvs = summary[tv]
                logging.info('{:s}: {:} not evaluated profiles found (of {:} total profiles)'.format(tv,
                                                                                                     tvs['not_evaluated_profiles'],
                                                                                                     len(ncfiles)))
                logging.info('{:s}: {:} suspect profiles found (of {:} total profiles)'.format(tv,
                                                                                               tvs['suspect_profiles'],
                                                                                               len(ncfiles)))
                logging.info('{:s}: {:} failed profiles found (of {:} total profiles)'.format(tv,
                                                                                              tvs['failed_profiles'],
                                                                                              len(ncfiles)))
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