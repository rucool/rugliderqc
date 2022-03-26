#!/usr/bin/env python

"""
Author: lgarzio on 3/25/2021
Last modified: lgarzio on 3/26/2021
Write an empty file at the beginning of each deployment that contains the attributes for all variables for proper
display in ERDDAP
"""

import os
import argparse
import sys
import glob
import xarray as xr
import pandas as pd
import numpy as np
from rugliderqc.common import find_glider_deployment_datapath, find_glider_deployments_rootdir
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname


def replace_data(ds, varname, timestamp, dim_name='ts'):
    """
    Replace the data arrays that contain values with one timestamp and fill values or 9 (MISSING) for QARTOD QC tests
    :param ds: xarray dataset
    :param varname: variable name (e.g. conductivity)
    :param timestamp: single timestamp for the placeholder data array
    :param dim_name: optional name for time dimension, default is 'ts'
    :return: data array containing a single timestamp placeholder dimension 'ts' and fill value/9 with all of the
    original attributes
    """
    data_type = ds[varname].encoding['dtype']
    try:
        value = ds[varname].encoding['_FillValue']
    except KeyError:
        if np.logical_or('_qartod_' in varname, '_hysteresis_test' in varname):
            value = 9
        else:
            value = np.nan
    placeholder_values = np.array([value], dtype=data_type)
    da = xr.DataArray(placeholder_values, coords=[timestamp], dims=[dim_name],
                      name=ds[varname].name, attrs=ds[varname].attrs)
    da.encoding = ds[varname].encoding

    return da


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

            attrs_filename = os.path.join(data_path, f'{deployment.replace("-", "_")}00Z.nc')

            # check if the attribute file has already been written
            if os.path.isfile(attrs_filename):
                logging.info('Attribute file already defined: {:s}'.format(attrs_filename))
                continue

            # define the timestamp for the attributes file
            (glider, trajectory) = deployment.split('-')
            ts = np.array([pd.to_datetime(trajectory).to_numpy()])

            # List the netcdf files in qc_queue
            ncfiles = sorted(glob.glob(os.path.join(data_path, 'qc_queue', '*.nc')))

            if len(ncfiles) == 0:
                logging.error(' 0 files found in {:s}'.format(os.path.join(data_path, 'qc_queue')))
                status = 1
                continue

            for f in ncfiles:
                try:
                    with xr.open_dataset(f) as ds:
                        ds = ds.load()
                except OSError as e:
                    logging.error('Error reading file {:s} ({:})'.format(f, e))
                    status = 1
                    continue

                if not os.path.isfile(attrs_filename):
                    # make a copy of the first file, specify the timestamp at the beginning of the deployment
                    # and create a file with all variables and their attributes
                    attrsds = ds.copy()
                    time_attrs = attrsds.time.attrs
                    time_encoding = attrsds.time.encoding
                    attrsds_data_vars = list(attrsds.data_vars)
                    for dv in attrsds_data_vars:
                        try:
                            length = len(attrsds[dv])
                        except TypeError:
                            # if the variable is informational (e.g. doesn't contain values) leave it as is
                            continue

                        # replace the data array with one timestamp and fill values or 9 (MISSING) for QARTOD QC tests
                        da = replace_data(attrsds, dv, ts)
                        attrsds[da.name] = da

                    # after setting all variables to fill values or missing (for QARTOD tests) at one timestamp,
                    # drop the original time dimension and rename the placeholder
                    attrsds = attrsds.drop_dims(['time'])
                    attrsds = attrsds.rename({'ts': 'time'})

                    # set time attrs and encoding
                    attrsds.time.attrs = time_attrs
                    attrsds.time.encoding = time_encoding

                    # save file
                    attrsds.to_netcdf(attrs_filename)
                    logging.info('Attribute file written: {:s}'.format(attrs_filename))

                else:
                    # if the file has already been generated, check the rest of the files to see if there are
                    # any additional variables that aren't in the first file and add them
                    add_ds = ds.copy()
                    data_vars = list(add_ds.data_vars)
                    additional_vars = [x for x in data_vars if x not in attrsds_data_vars]
                    if len(additional_vars) > 0:
                        for av in additional_vars:
                            try:
                                length = len(add_ds[av])
                            except TypeError:
                                # add to the attrs dataset if there's nothing to change
                                attrsds[add_ds[av].name] = add_ds[av]
                                continue

                            # replace data with one timestamp and fill values or 9 (MISSING for QARTOD QC tests)
                            da = replace_data(add_ds, av, ts, dim_name='time')
                            attrsds[da.name] = da

                        attrsds.to_netcdf(attrs_filename)

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
