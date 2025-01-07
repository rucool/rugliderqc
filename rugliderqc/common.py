#!/usr/bin/env python

import os
import re
import pandas as pd
import pytz
from netCDF4 import num2date
import numpy as np
from dateutil import parser
from netCDF4 import default_fillvals
import xarray as xr
from ioos_qc import qartod
from ioos_qc.config import Config
from ioos_qc.streams import XarrayStream
from ioos_qc.results import collect_results
from ioos_qc.utils import load_config_as_dict as loadconfig


def convert_epoch_ts(data):
    if isinstance(data, xr.core.dataarray.DataArray):
        time = pd.to_datetime(num2date(data.values, data.units, only_use_cftime_datetimes=False))
    elif isinstance(data, pd.core.indexes.base.Index):
        time = pd.to_datetime(num2date(data, 'seconds since 1970-01-01T00:00:00Z', only_use_cftime_datetimes=False))
    elif isinstance(data, pd.core.indexes.datetimes.DatetimeIndex):
        time = pd.to_datetime(num2date(data, 'seconds since 1970-01-01T00:00:00Z', only_use_cftime_datetimes=False))

    return time


def find_glider_deployment_datapath(logger, deployment, deployments_root, dataset_type, cdm_data_type, mode):
    glider_regex = re.compile(r'^(.*)-(\d{8}T\d{4})')
    match = glider_regex.search(deployment)
    if match:
        try:
            (glider, trajectory) = match.groups()
            try:
                trajectory_dt = parser.parse(trajectory).replace(tzinfo=pytz.UTC)
            except ValueError as e:
                logger.error('Error parsing trajectory date {:s}: {:}'.format(trajectory, e))
                trajectory_dt = None
                data_path = None
                deployment_location = None

            if trajectory_dt:
                trajectory = '{:s}-{:s}'.format(glider, trajectory_dt.strftime('%Y%m%dT%H%M'))
                deployment_name = os.path.join('{:0.0f}'.format(trajectory_dt.year), trajectory)

                # Create fully-qualified path to the deployment location
                deployment_location = os.path.join(deployments_root, deployment_name)
                if os.path.isdir(deployment_location):
                    # Set the deployment netcdf data path
                    data_path = os.path.join(deployment_location, 'data', 'out', 'nc',
                                             '{:s}-{:s}/{:s}'.format(dataset_type, cdm_data_type, mode))
                    if not os.path.isdir(data_path):
                        logger.warning('{:s} data directory not found: {:s}'.format(trajectory, data_path))
                        data_path = None
                        deployment_location = None
                else:
                    logger.warning('Deployment location does not exist: {:s}'.format(deployment_location))
                    data_path = None
                    deployment_location = None

        except ValueError as e:
            logger.error('Error parsing invalid deployment name {:s}: {:}'.format(deployment, e))
            data_path = None
            deployment_location = None
    else:
        logger.error('Cannot pull glider name from {:}'.format(deployment))
        data_path = None
        deployment_location = None

    return data_path, deployment_location


def find_glider_deployments_rootdir(logger, test):
    # Find the glider deployments root directory
    if test:
        envvar = 'GLIDER_DATA_HOME_TEST'
    else:
        envvar = 'GLIDER_DATA_HOME'

    data_home = os.getenv(envvar)

    if not data_home:
        logger.error('{:s} not set'.format(envvar))
        return 1, 1
    elif not os.path.isdir(data_home):
        logger.error('Invalid {:s}: {:s}'.format(envvar, data_home))
        return 1, 1

    deployments_root = os.path.join(data_home, 'deployments')
    if not os.path.isdir(deployments_root):
        logger.warning('Invalid deployments root: {:s}'.format(deployments_root))
        return 1, 1

    return data_home, deployments_root


def return_season(ts):
    if ts.month in [12, 1, 2]:
        season = 'DJF'
    elif ts.month in [3, 4, 5]:
        season = 'MAM'
    elif ts.month in [6, 7, 8]:
        season = 'JJA'
    elif ts.month in [9, 10, 11]:
        season = 'SON'

    return season


def run_ioos_qc_gross_flatline(ds, qc_config_file):
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
            ds[sensor].attrs['ancillary_variables'] = ' '.join(
                (ds[sensor].ancillary_variables, qc_varname))

        # Create gross/flatline data array
        da = xr.DataArray(flag_results.astype('int32'), coords=ds[sensor].coords,
                          dims=ds[sensor].dims,
                          name=qc_varname,
                          attrs=attrs)

        # define variable encoding
        set_encoding(da)

        # Add gross/flatline QC variable to the original dataset
        ds[qc_varname] = da


def run_ioos_qc_spike(ds, qc_config_file):
    # Run ioos_qc spike test based on the QC configuration file
    spike_config = loadconfig(qc_config_file)
    for variable_name, values in spike_config.items():
        data = ds[variable_name]
        non_nan_ind = np.invert(np.isnan(data.values))  # identify where not nan
        non_nan_i = np.where(non_nan_ind)[0]  # get locations of non-nans
        tdiff = np.diff(data.time[non_nan_ind]).astype('timedelta64[s]').astype(
            float)  # get time interval (s) between non-nan points
        tdiff_long = np.where(tdiff > 60 * 5)[0]  # locate time intervals > 5 min
        tdiff_long_i = np.append(non_nan_i[tdiff_long],
                                 non_nan_i[tdiff_long + 1])  # original locations of where time interval is long

        spike_settings = values['qartod']['spike_test']
        # convert original threshold from units/s to units/average-timestep
        spike_settings['suspect_threshold'] = spike_settings['suspect_threshold'] * np.nanmedian(tdiff)
        spike_settings['fail_threshold'] = spike_settings['fail_threshold'] * np.nanmedian(tdiff)

        flag_vals = 2 * np.ones(np.shape(data))
        flag_vals[np.invert(non_nan_ind)] = qartod.QartodFlags.MISSING

        # only run the test if the array has values
        if len(non_nan_i) > 0:
            flag_vals[non_nan_ind] = qartod.spike_test(inp=data[non_nan_ind],
                                                       **spike_settings)

            # flag as not evaluated/unknown on either end of long time gap
            flag_vals[tdiff_long_i] = qartod.QartodFlags.UNKNOWN

            qc_varname = f'{variable_name}_qartod_spike_test'
            attrs = set_qartod_attrs('spike_test', variable_name, spike_settings)
            da = xr.DataArray(flag_vals.astype('int32'), coords=data.coords, dims=data.dims, attrs=attrs,
                              name=qc_varname)

            # define variable encoding
            set_encoding(da)

            # Add the QC variable to the dataset
            ds[qc_varname] = da


def set_encoding(data_array, original_encoding=None):
    """
    Define encoding for a data array, using the original encoding from another variable (if applicable)
    :param data_array: data array to which encoding is added
    :param original_encoding: optional encoding dictionary from the parent variable
    (e.g. use the encoding from "depth" for the new depth_interpolated variable)
    """
    if original_encoding:
        data_array.encoding = original_encoding

    try:
        encoding_dtype = data_array.encoding['dtype']
    except KeyError:
        data_array.encoding['dtype'] = data_array.dtype

    try:
        encoding_fillvalue = data_array.encoding['_FillValue']
    except KeyError:
        # set the fill value using netCDF4.default_fillvals
        data_type = f'{data_array.dtype.kind}{data_array.dtype.itemsize}'
        data_array.encoding['_FillValue'] = default_fillvals[data_type]


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
