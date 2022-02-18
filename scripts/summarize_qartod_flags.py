#!/usr/bin/env python

"""
Author: lgarzio on 1/18/2022
Last modified: lgarzio on 2/18/2022
Summarize the QARTOD QC flags for each variable.
"""

import os
import argparse
import sys
import glob
import numpy as np
import xarray as xr
from rugliderqc.common import find_glider_deployment_datapath, find_glider_deployments_rootdir
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname
np.set_printoptions(suppress=True)


def set_summary_qartod_attrs(sensor, ancillary_variables):
    """
    Define the QARTOD QC summary flag attributes
    :param sensor: sensor variable name (e.g. conductivity)
    :param ancillary_variables: variables included in the summary flag, format is a string attribute whose values
    are blank separated
    """

    flag_meanings = 'GOOD NOT_EVALUATED SUSPECT FAIL MISSING'
    flag_values = [1, 2, 3, 4, 9]
    standard_name = 'qartod_summary_quality_flag'
    long_name = 'QARTOD Summary Quality Flag'

    # Define variable attributes
    attrs = {
        'ancillary_variables': ancillary_variables,
        'standard_name': standard_name,
        'long_name': long_name,
        'flag_values': np.byte(flag_values),
        'flag_meanings': flag_meanings,
        'valid_min': np.byte(min(flag_values)),
        'valid_max': np.byte(max(flag_values)),
        'ioos_qc_module': 'qartod',
        'ioos_qc_target': sensor,
        'comment': f'Summary of the highest QARTOD flag value for all QARTOD tests for {sensor} (excluding 2/NOT_EVALUATED).'
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

            logging.info('Summarizing QARTOD flags: {:s}'.format(os.path.join(data_path, 'qc_queue')))

            # List the netcdf files in qc_queue
            ncfiles = sorted(glob.glob(os.path.join(data_path, 'qc_queue', '*.nc')))

            if len(ncfiles) == 0:
                logging.error(' 0 files found to QC: {:s}'.format(os.path.join(data_path, 'qc_queue')))
                status = 1
                continue

            # Iterate through files and summarize the QARTOD flags
            for f in ncfiles:
                try:
                    with xr.open_dataset(f) as ds:
                        ds = ds.load()
                except OSError as e:
                    logging.error('Error reading file {:s} ({:})'.format(f, e))
                    status = 1
                    continue

                # List the qartod flag variables
                qartod_vars = [x for x in list(ds.data_vars) if '_qartod_' in x]

                # List the sensors that were QC'd
                qc_vars = list(np.unique([x.split('_qartod_')[0] for x in qartod_vars]))

                # Iterate through each sensor that was QC'd and summarize the QARTOD flags
                for sensor in qc_vars:
                    summary_flag = np.empty(len(ds[sensor].values))
                    summary_flag[:] = 0
                    sensor_qartod_vars = [x for x in ds.data_vars if f'{sensor}_qartod_' in x]
                    for sqv in sensor_qartod_vars:
                        # make a copy of the flags so the original array isn't changed
                        flag = ds[sqv].values.copy()

                        # turn 2/NOT_EVALUATED/UNKNOWN to -1
                        flag[flag == 2] = 0

                        summary_flag = np.maximum(summary_flag, flag)

                    # check if any flags are zero and turn those back to 2 (NOT_EVALUATED/UNKNOWN)
                    summary_flag[summary_flag == 0] = 2

                    qc_varname = f'{sensor}_qartod_summary_flag'
                    attrs = set_summary_qartod_attrs(sensor, ' '.join(sensor_qartod_vars))

                    # add summary variable to the original dataset
                    da = xr.DataArray(summary_flag.astype('int32'), coords=ds[sensor].coords, dims=ds[sensor].dims,
                                      name=qc_varname, attrs=attrs)
                    ds[qc_varname] = da

                    # add the summary variable to the sensor ancillary_variables
                    if not hasattr(ds[sensor], 'ancillary_variables'):
                        ds[sensor].attrs['ancillary_variables'] = qc_varname
                    else:
                        ds[sensor].attrs['ancillary_variables'] = ' '.join((ds[sensor].ancillary_variables, qc_varname))

                ds.to_netcdf(f)
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
