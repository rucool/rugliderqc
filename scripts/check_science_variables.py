#!/usr/bin/env python

"""
Author: lgarzio on 12/22/2021
Last modified: lgarzio on 12/22/2021
Checks files for CTD science variables (pressure, conductivity and temperature). Renames files ".nosci" if the file
doesn't contain any of those variables, or only contains pressure.
"""

import os
import argparse
import sys
import glob
import xarray as xr
from rugliderqc.common import find_glider_deployment_datapath, find_glider_deployments_rootdir
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname


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

            logging.info('Checking for science variables: {:s}'.format(os.path.join(data_path, 'qc_queue')))

            # List the netcdf files in qc_queue
            ncfiles = sorted(glob.glob(os.path.join(data_path, 'qc_queue', '*.nc')))

            if len(ncfiles) == 0:
                logging.error(' 0 files found to check: {:s}'.format(os.path.join(data_path, 'qc_queue')))
                status = 1
                continue

            pressure_vars = ['pressure', 'pressure2', 'pressure_rbr', 'rbr_pressure', 'sci_water_pressure']
            sci_vars = ['conductivity', 'conductivity2', 'conductivity_rbr', 'rbr_conductivity',
                        'temperature', 'temperature2', 'temperature_rbr', 'rbr_temperature']

            # Iterate through files and find duplicated timestamps
            summary = 0
            for f in ncfiles:
                try:
                    ds = xr.open_dataset(f)
                except OSError as e:
                    logging.error('Error reading file {:s} ({:})'.format(f, e))
                    status = 1
                    continue

                # check for pressure
                ds_pressure_vars = list(set(ds.data_vars).intersection(set(pressure_vars)))

                if len(ds_pressure_vars) == 0:
                    os.rename(f, f'{f}.nosci')
                    logging.info('Pressure not found in file: {:s}'.format(f))
                    summary += 1
                else:
                    # check for conductivity or temperature
                    ds_sci_vars = list(set(ds.data_vars).intersection(set(sci_vars)))

                    if len(ds_sci_vars) == 0:
                        os.rename(f, f'{f}.nosci')
                        logging.info('Temperature and/or conductivity not found in file: {:s}'.format(f))
                        summary += 1

            logging.info('Found {:} files without CTD science variables (of {:} total files)'.format(summary, len(ncfiles)))
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
