#!/usr/bin/env python

"""
Author: lgarzio on 12/7/2021
Last modified: lgarzio on 12/17/2021
Move quality controlled glider NetCDF files to the final data directory (out of qc_queue) to send to ERDDAP
"""

import os
import argparse
import sys
import glob
from pathlib import Path
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

            # List the netcdf files in qc_queue
            ncfiles = sorted(glob.glob(os.path.join(data_path, 'qc_queue', '*.nc')))

            if len(ncfiles) == 0:
                logging.error(' 0 files found to move: {:s}'.format(os.path.join(data_path, 'qc_queue')))
                status = 1
                continue

            # Iterate through files and move them to the parent directory
            moved = 0
            for f in ncfiles:
                p = Path(f).absolute()
                p.rename(os.path.join(data_path, p.name))
                moved += 1

            logging.info('Moved {:} of {:} valid netcdf files to: {:s}'.format(moved, len(ncfiles), data_path))

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
