#!/usr/bin/env python

"""
Author: lnazzaro and lgarzio on 12/7/2021
Last modified: lgarzio on 2/18/2022
Run ioos_qc QARTOD tests on processed glider NetCDF files and append the results to the original file.
"""

import os
import argparse
import sys
from datetime import timedelta
import glob
import numpy as np
import pandas as pd
import xarray as xr
import gsw
from ioos_qc import qartod
from ioos_qc.config import Config
from ioos_qc.streams import XarrayStream
from ioos_qc.results import collect_results
from ioos_qc.utils import load_config_as_dict as loadconfig
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from rugliderqc.common import find_glider_deployment_datapath, find_glider_deployments_rootdir
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname


def build_global_regional_config(ds, qc_config_root):
    """
    Find the appropriate climatology, spike, and rate of change configuration for the dataset
    :param ds: glider data xarray dataset
    :param qc_config_root: root directory where QC configuration files are located
    """
    profile_time = pd.to_datetime(ds.profile_time.values)
    profile_lon = ds.profile_lon.values
    profile_lat = ds.profile_lat.values

    # Set the path for the global and regional configuration files
    qc_config_root = os.path.join(qc_config_root, 'global_regional')

    # Start with global config as default
    global_config_file = os.path.join(qc_config_root, 'global_configs.yml')
    c_global = Config(global_config_file)
    c = c_global.config['contexts'][0]

    # Find regional boundaries (if available)
    region_file = os.path.join(qc_config_root, 'regional_boundaries.yml')
    region_bounds = loadconfig(region_file)
    best_region = {'priority': 100, 'region': 'global'}

    # look for highest priority region containing profile lon and lat, with existing config
    for region in region_bounds['regions']:
        if region['priority'] >= best_region['priority']:
            continue
        if not os.path.exists(os.path.join(qc_config_root, region['region'] + '_configs.yml')):
            continue
        if Polygon(list(zip(region['boundaries']['longitude'], region['boundaries']['latitude']))).contains(
                Point(profile_lon, profile_lat)):
            best_region = region

    if best_region['region'] != 'global':
        # Pull in regional config
        config_file = os.path.join(qc_config_root, best_region['region'] + '_configs.yml')
        c_region = Config(config_file)

        # Loop through different time ranges and replace existing config 'c' if an appropriate time window with higher
        # priority is found. If not found, return the global config.
        for c0 in c_region.config['contexts']:
            if c0['priority'] >= c['priority']:
                continue
            t0 = c0['window']['starting'].replace(year=profile_time.year)
            t1 = c0['window']['ending'].replace(year=profile_time.year)
            if np.logical_and(profile_time >= t0, profile_time <= t1):
                c = c0

    c['window']['starting'] = c['window']['starting'].replace(year=profile_time.year)
    c['window']['ending'] = c['window']['ending'].replace(year=profile_time.year)

    return c


def define_gross_flatline_config(instrument_name, model_name):
    """
    Find the appropriate gross range/flatline configuration for an instrument
    :param instrument_name: instrument name (e.g. instrument_ctd, instrument_optode)
    :param model_name: instrument make-model
    """
    if instrument_name == 'instrument_ctd':
        config_filename = f'{model_name.split(" ")[0].lower()}_ctd_gross_flatline.yml'
    elif instrument_name == 'instrument_optode':
        config_filename = f'optode{model_name.split(" ")[-1].lower()}_gross_flatline.yml'
    else:
        config_filename = 'no_filename_specified'

    return config_filename


def set_qartod_attrs(test, sensor, thresholds):
    """
    Define the QARTOD QC variable attributes
    :param test: QARTOD QC test
    :param sensor: sensor variable name (e.g. conductivity)
    :param thresholds: flag thresholds from QC configuration files
    """

    flag_meanings = 'GOOD NOT_EVALUATED SUSPECT FAIL MISSING'
    flag_values = [1, 2, 3, 4, 9]
    standard_name = f'{test}_quality_flag'  # 'flat_line_test_quality_flag'
    long_name = f'{" ".join([x.capitalize() for x in test.split("_")])} Quality Flag'

    # Defining gross/flatline QC variable attributes
    attrs = {
        'standard_name': standard_name,
        'long_name': long_name,
        'flag_values': np.byte(flag_values),
        'flag_meanings': flag_meanings,
        'flag_configurations': str(thresholds),
        'valid_min': np.byte(min(flag_values)),
        'valid_max': np.byte(max(flag_values)),
        'ioos_qc_module': 'qartod',
        'ioos_qc_test': f'{test}',
        'ioos_qc_target': sensor,
    }

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

            logging.info('Running glider QARTOD QC: {:s}'.format(os.path.join(data_path, 'qc_queue')))

            # List the netcdf files in qc_queue
            ncfiles = sorted(glob.glob(os.path.join(data_path, 'qc_queue', '*.nc')))

            if len(ncfiles) == 0:
                logging.error(' 0 files found to QC: {:s}'.format(os.path.join(data_path, 'qc_queue')))
                status = 1
                continue

            # Iterate through files and apply QC
            for f in ncfiles:
                try:
                    with xr.open_dataset(f) as ds:
                        ds = ds.load()
                except OSError as e:
                    logging.error('Error reading file {:s} ({:})'.format(f, e))
                    status = 1
                    continue

                logging.info('Checking file: {:s}'.format(f))

                # Set the qc configuration path
                qc_config_root = os.path.join(data_home, 'qc', 'config')
                if not os.path.isdir(qc_config_root):
                    logging.warning('Invalid QC config root: {:s}'.format(qc_config_root))
                    return 1

                # run gross and flat line tests
                # Set the path for the gross range and flat line configuration files
                qc_config_gross_flatline = os.path.join(qc_config_root, 'gross_flatline')

                # List the instruments in the netcdf file
                instruments = [x for x in list(ds.data_vars) if 'instrument_' in x]

                for inst in instruments:
                    # Define the instrument make/model
                    try:
                        model = ds[inst].make_model
                    except AttributeError:
                        logging.error('Sensor make_model not specified {:s}'.format(inst))

                    # Build the configuration filename based on the instrument, make and model
                    qc_config_filename = define_gross_flatline_config(inst, model)

                    qc_config_file = os.path.join(qc_config_gross_flatline, qc_config_filename)

                    if not os.path.isfile(qc_config_file):
                        logging.warning('Missing QC configuration file: {:s} {:s}'.format(inst, qc_config_file))
                        continue
                    logging.info('QC configuration file: {:s}'.format(qc_config_file))

                    # Run ioos_qc gross/flatline tests based on the QC configuration file
                    c = Config(qc_config_file)
                    xs = XarrayStream(ds, time='time', lat='latitude', lon='longitude')
                    qc_results = xs.run(c)
                    collected_list = collect_results(qc_results, how='list')

                    # Parse each gross/flatline QC result
                    for cl in collected_list:
                        sensor = cl.stream_id
                        test = cl.test
                        qc_varname = f'{sensor}_{cl.package}_{test}'
                        # logging.info('Parsing QC results: {:s}'.format(qc_varname))
                        flag_results = cl.results.data

                        # Defining gross/flatline QC variable attributes
                        attrs = set_qartod_attrs(test, sensor, c.config[sensor]['qartod'][test])
                        if not hasattr(ds[sensor], 'ancillary_variables'):
                            ds[sensor].attrs['ancillary_variables'] = qc_varname
                        else:
                            ds[sensor].attrs['ancillary_variables'] = ' '.join((ds[sensor].ancillary_variables, qc_varname))

                        # Add gross/flatline QC variable to the original dataset
                        da = xr.DataArray(flag_results, coords=ds[sensor].coords, dims=ds[sensor].dims,
                                          name=qc_varname,
                                          attrs=attrs)
                        ds[qc_varname] = da

                # manually run gross range test for pressure based on depth_rating in file
                test = 'gross_range_test'
                sensor = 'pressure'

                # convert the depth_rating in the file (meters) to dbar before comparison with the pressure variable
                depth_rating = float("".join(filter(str.isdigit, ds.platform.depth_rating)))
                pressure_rating = gsw.p_from_z(-depth_rating, np.nanmean(ds.profile_lat.values))
                cinfo = {'fail_span': [0, pressure_rating]}
                qc_varname = f'{sensor}_qartod_gross_range_test'
                flag_vals = qartod.gross_range_test(inp=ds[sensor].values,
                                                    **cinfo)

                # Define QC variable attributes, add a comment about the conversion from depth_rating in meters to dbar
                cinfo = {'fail_span': [0, int(depth_rating)]}
                attrs = set_qartod_attrs(test, sensor, cinfo)
                attrs['comment'] = 'Glider depth rating (m) in flag_configurations converted to pressure (dbar) from ' \
                                   'pressure and profile_lat using gsw.p_from_z'
                if not hasattr(ds[sensor], 'ancillary_variables'):
                    ds[sensor].attrs['ancillary_variables'] = qc_varname
                else:
                    ds[sensor].attrs['ancillary_variables'] = ' '.join((ds[sensor].ancillary_variables, qc_varname))

                # Add QC variable to the original dataset
                da = xr.DataArray(flag_vals.astype('int32'), coords=ds[sensor].coords, dims=ds[sensor].dims,
                                  name=qc_varname, attrs=attrs)
                ds[qc_varname] = da

                # Find the configuration files for the climatology, spike, rate of change, and pressure tests
                c = build_global_regional_config(ds, qc_config_root)

                # run climatology, spike, rate of change, and pressure tests
                times = ds.time.values
                for sensor, config_info in c['streams'].items():
                    if sensor not in ds.data_vars:
                        continue
                    # grab data for sensor
                    data = ds[sensor].values
                    # identify where not nan
                    non_nan_ind = np.invert(np.isnan(data))
                    # get locations of non-nans
                    non_nan_i = np.where(non_nan_ind)[0]
                    # get time interval (s) between non-nan points
                    tdiff = np.diff(times[non_nan_ind]).astype('timedelta64[s]').astype(float)
                    # locate time intervals > 5 min
                    tdiff_long = np.where(tdiff > 60 * 5)[0]
                    # original locations of where time interval is long
                    tdiff_long_i = np.append(non_nan_i[tdiff_long], non_nan_i[tdiff_long + 1])

                    for test, cinfo in config_info['qartod'].items():
                        if test == 'pressure_test':  # check that the pressure values are continually increasing/decreasing
                            qc_varname = f'{sensor}_qartod_pressure_test'
                            flag_vals = 2 * np.ones(np.shape(data))
                            flag_vals[np.invert(non_nan_ind)] = qartod.QartodFlags.MISSING

                            # only run the test if the array has values
                            if len(non_nan_i) > 0:
                                flag_vals[non_nan_ind] = qartod.pressure_test(inp=data[non_nan_ind],
                                                                              tinp=times[non_nan_ind],
                                                                              **cinfo)

                        elif test == 'climatology_test':
                            qc_varname = f'{sensor}_qartod_climatology_test'
                            climatology_settings = {'tspan': [c['window']['starting'] - timedelta(days=2),
                                                              c['window']['ending'] + timedelta(days=2)],
                                                    'fspan': None, 'vspan': None, 'zspan': None}

                            # if no set depth range, apply thresholds to full profile depth
                            if 'depth_range' not in cinfo.keys():
                                climatology_settings['zspan'] = [0, np.nanmax(ds.depth.values)]

                                if 'suspect_span' in cinfo.keys():
                                    climatology_settings['vspan'] = cinfo['suspect_span']
                                if 'fail_span' in cinfo.keys():
                                    climatology_settings['fspan'] = cinfo['fail_span']

                                climatology_config = qartod.ClimatologyConfig()
                                climatology_config.add(**climatology_settings)
                                flag_vals = qartod.climatology_test(config=climatology_config,
                                                                    inp=data,
                                                                    tinp=times,
                                                                    zinp=ds.depth.values)
                            else:
                                # if one depth range provided, apply thresholds only to that depth range
                                if len(np.shape(cinfo['depth_range'])) == 1:
                                    climatology_settings = {'tspan': [c['window']['starting'] - timedelta(days=2),
                                                                      c['window']['ending'] + timedelta(days=2)],
                                                            'fspan': cinfo['depth_range'],
                                                            'vspan': None, 'zspan': None}

                                    if 'suspect_span' in cinfo.keys():
                                        climatology_settings['vspan'] = cinfo['suspect_span']
                                    if 'fail_span' in cinfo.keys():
                                        climatology_settings['fspan'] = cinfo['fail_span']

                                    climatology_config = qartod.ClimatologyConfig()
                                    climatology_config.add(**climatology_settings)
                                    flag_vals = qartod.climatology_test(config=climatology_config,
                                                                        inp=data,
                                                                        tinp=times,
                                                                        zinp=ds.depth.values)

                                else:  # if different thresholds for multiple depth ranges, loop through each
                                    flag_vals = 2 * np.ones(np.shape(data))
                                    for z_int in range(len(cinfo['depth_range'])):
                                        climatology_settings = {'tspan': [c['window']['starting'] - timedelta(days=2),
                                                                          c['window']['ending'] + timedelta(days=2)],
                                                                'fspan': cinfo['depth_range'][z_int],
                                                                'vspan': None, 'zspan': None}

                                        if 'suspect_span' in cinfo.keys():
                                            climatology_settings['vspan'] = cinfo['suspect_span'][z_int]
                                        if 'fail_span' in cinfo.keys():
                                            climatology_settings['fspan'] = cinfo['fail_span'][z_int]

                                        climatology_config = qartod.ClimatologyConfig()
                                        climatology_config.add(**climatology_settings)
                                        z_ind = np.logical_and(
                                            ds.depth.values > cinfo['depth_range'][z_int][0],
                                            ds.depth.values <= cinfo['depth_range'][z_int][1])

                                        flag_vals[z_ind] = qartod.climatology_test(config=climatology_config,
                                                                                   inp=data[z_ind],
                                                                                   tinp=times[z_ind],
                                                                                   zinp=ds.depth.values[z_ind])

                        elif test == 'spike_test':
                            qc_varname = f'{sensor}_qartod_spike_test'
                            spike_settings = {'suspect_threshold': None, 'fail_threshold': None}

                            # convert original threshold from units/s to units/average-timestep
                            if 'suspect_threshold' in cinfo.keys():
                                spike_settings['suspect_threshold'] = cinfo['suspect_threshold'] * np.nanmedian(tdiff)
                            if 'fail_threshold' in cinfo.keys():
                                spike_settings['fail_threshold'] = cinfo['fail_threshold'] * np.nanmedian(tdiff)

                            flag_vals = 2 * np.ones(np.shape(data))
                            flag_vals[np.invert(non_nan_ind)] = qartod.QartodFlags.MISSING

                            # only run the test if the array has values
                            if len(non_nan_i) > 0:
                                flag_vals[non_nan_ind] = qartod.spike_test(inp=data[non_nan_ind],
                                                                           method='differential',
                                                                           **spike_settings)
                                # flag as not evaluated/unknown on either end of long time gap
                                flag_vals[tdiff_long_i] = qartod.QartodFlags.UNKNOWN

                        elif test == 'rate_of_change_test':
                            qc_varname = f'{sensor}_qartod_rate_of_change_test'
                            flag_vals = 2 * np.ones(np.shape(data))
                            flag_vals[np.invert(non_nan_ind)] = qartod.QartodFlags.MISSING

                            # only run the test if the array has values
                            if len(non_nan_i) > 0:
                                flag_vals[non_nan_ind] = qartod.rate_of_change_test(inp=data[non_nan_ind],
                                                                                    tinp=times[non_nan_ind],
                                                                                    **cinfo)

                        # Define pressure/climatology/spike/rate of change QC variable attributes
                        attrs = set_qartod_attrs(test, sensor, cinfo)
                        if not hasattr(ds[sensor], 'ancillary_variables'):
                            ds[sensor].attrs['ancillary_variables'] = qc_varname
                        else:
                            ds[sensor].attrs['ancillary_variables'] = ' '.join((ds[sensor].ancillary_variables, qc_varname))

                        # Add QC variable to the original dataset
                        da = xr.DataArray(flag_vals.astype('int32'), coords=ds[sensor].coords, dims=ds[sensor].dims,
                                          name=qc_varname, attrs=attrs)
                        ds[qc_varname] = da

                # TODO add location test

                # Save the netcdf file with QC variables over the original file
                ds.to_netcdf(f)

    return status


if __name__ == '__main__':
    # deploy = 'ru34-20220113T1949'  # maracoos_02-20210716T1814 ru34-20200729T1430 ru33-20201014T1746 ru33-20200715T1558  ru32-20190102T1317 ru30-20210503T1929
    # mode = 'rt'
    # d = 'profile'
    # ll = 'info'
    # level = 'sci'
    # test = True
    # main(deploy, mode, d, ll, level, test)
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
