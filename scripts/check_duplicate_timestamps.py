#!/usr/bin/env python

"""
Author: lgarzio on 12/7/2021
Last modified: lgarzio on 8/7/2023
Check two consecutive .nc files for duplicated timestamps and rename files that are full duplicates of all or part
of another file.
"""

import os
import argparse
import sys
import glob
import numpy as np
import xarray as xr
from rugliderqc.common import find_glider_deployment_datapath, find_glider_deployments_rootdir
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname
from ioos_qc.utils import load_config_as_dict as loadconfig


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

            # Set the deployment qc configuration path
            deployment_location = data_path.split('/data')[0]
            deployment_qc_config_root = os.path.join(deployment_location, 'config', 'qc')
            if not os.path.isdir(deployment_qc_config_root):
                logging.warning('Invalid deployment QC config root: {:s}'.format(deployment_qc_config_root))

            # Determine if the test should be run or not
            qctests_config_file = os.path.join(deployment_qc_config_root, 'qctests.yml')
            if os.path.isfile(qctests_config_file):
                qctests_config_dict = loadconfig(qctests_config_file)
                if not qctests_config_dict['check_duplicate_timestamps']:
                    logging.warning(
                        'Not checking files for duplicated timestamps because test is turned off, check: {:s}'.format(
                            qctests_config_file))
                    continue

            logging.info('Checking duplicated timestamps: {:s}'.format(os.path.join(data_path, 'qc_queue')))

            # List the netcdf files in qc_queue
            ncfiles = sorted(glob.glob(os.path.join(data_path, 'qc_queue', '*.nc')))

            if len(ncfiles) == 0:
                logging.error(' 0 files found to check: {:s}'.format(os.path.join(data_path, 'qc_queue')))
                status = 1
                continue

            # Iterate through files and find duplicated timestamps
            duplicates = 0
            for i, f in enumerate(ncfiles):
                try:
                    ds = xr.open_dataset(f)
                except OSError as e:
                    logging.error('Error reading file {:s} ({:})'.format(ncfiles[i], e))
                    status = 1
                    continue

                # find the next file and compare timestamps
                try:
                    f2 = ncfiles[i + 1]
                    ds2 = xr.open_dataset(f2)
                except OSError as e:
                    logging.error('Error reading file {:s} ({:})'.format(ncfiles[i + 1], e))
                    status = 1
                    continue
                except IndexError:
                    continue

                # find the unique timestamps between the two datasets
                unique_timestamps = list(set(ds.time.values).symmetric_difference(set(ds2.time.values)))

                # find the unique timestamps in each dataset
                check_ds = [t for t in ds.time.values if t in unique_timestamps]
                check_ds2 = [t for t in ds2.time.values if t in unique_timestamps]

                # if the unique timestamps aren't found in either dataset (i.e. timestamps are exactly the same)
                # rename the second dataset
                if np.logical_and(len(check_ds) == 0, len(check_ds2) == 0):
                    os.rename(f2, f'{f2}.duplicate')
                    logging.info('Duplicated timestamps found in file: {:s}'.format(f2))
                    duplicates += 1
                # if the unique timestamps aren't found in the second dataset, rename it
                elif np.logical_and(len(check_ds) > 0, len(check_ds2) == 0):
                    os.rename(f2, f'{f2}.duplicate')
                    logging.info('Duplicated timestamps found in file: {:s}'.format(f2))
                    duplicates += 1
                # if the unique timestamps aren't found in the first dataset, rename it
                elif np.logical_and(len(check_ds) == 0, len(check_ds2) > 0):
                    try:
                        os.rename(f, f'{f}.duplicate')
                        logging.info('Duplicated timestamps found in file: {:s}'.format(f))
                        duplicates += 1
                    except FileNotFoundError:  # file has already been identified as a duplicate
                        continue
                else:
                    continue

            logging.info('Found {:} duplicated files (of {:} total files)'.format(duplicates, len(ncfiles)))
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
