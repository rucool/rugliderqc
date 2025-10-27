#!/usr/bin/env python

"""
Author: lgarzio on 10/27/2025
Last modified: lgarzio on 10/27/2025
Check that profiles have at least 5 depth data points
and 1m depth binned profiles have at least (depth range / 10) bins (minimum 5 bins).
"""

import os
import argparse
import sys
import glob
import math
import pandas as pd
import xarray as xr
import numpy as np
import rugliderqc.common as cf
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname
from ioos_qc.utils import load_config_as_dict as loadconfig


def main(args):
    loglevel = args.loglevel.upper()
    cdm_data_type = args.cdm_data_type
    mode = args.mode
    dataset_type = args.level
    test = args.test

    #logFile_base = os.path.join(os.path.expanduser('~'), 'glider_proc_log')  # for debugging
    logFile_base = logfile_basename()
    logging_base = setup_logger('logging_base', loglevel, logFile_base)

    data_home, deployments_root = cf.find_glider_deployments_rootdir(logging_base, test)
    if isinstance(deployments_root, str):

        # Set the default qc configuration path
        qc_config_root = os.path.join(data_home, 'qc', 'config')
        if not os.path.isdir(qc_config_root):
            logging_base.warning('Invalid QC config root: {:s}'.format(qc_config_root))
            return 1

        for deployment in args.deployments:

            data_path, deployment_location = cf.find_glider_deployment_datapath(logging_base, deployment, deployments_root,
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
            deployment_qc_config_root = os.path.join(deployment_location, 'config', 'qc')
            if not os.path.isdir(deployment_qc_config_root):
                logging.warning('Invalid deployment QC config root: {:s}'.format(deployment_qc_config_root))

            # Determine if the test should be run or not
            qctests_config_file = os.path.join(deployment_qc_config_root, 'qctests.yml')
            if os.path.isfile(qctests_config_file):
                qctests_config_dict = loadconfig(qctests_config_file)
                if not qctests_config_dict['check_profile_depth']:
                    logging.warning(
                        'Not checking files for profile depth because test is turned off, check: {:s}'.format(
                            qctests_config_file))
                    continue

            logging.info('Checking profile depths: {:s}'.format(os.path.join(data_path, 'qc_queue')))

            # List the netcdf files in qc_queue
            ncfiles = sorted(glob.glob(os.path.join(data_path, 'qc_queue', '*.nc')))

            if len(ncfiles) == 0:
                logging.error(' 0 files found to check: {:s}'.format(os.path.join(data_path, 'qc_queue')))
                continue

            summary = 0
            for f in ncfiles:
                try:
                    with xr.open_dataset(f, decode_times=False) as ds:
                        ds = ds.load()
                except OSError as e:
                    logging.error('Error reading file {:s} ({:})'.format(f, e))
                    os.rename(f, f'{f}.bad')
                    continue
                except ValueError as e:
                    logging.error('Error reading file {:s} ({:})'.format(f, e))
                    os.rename(f, f'{f}.bad')
                    continue

                # find the depth variable, either depth or pressure
                try:
                    testvar = ds.depth
                except AttributeError:
                    try:
                        testvar = ds.pressure
                        if testvar.units == 'bar':
                            testvar = testvar * 10  # convert bar to dbar
                    except AttributeError:
                        logging.warning(' No depth or pressure variable found in file: {:s}'.format(f))
                        continue
                
                # check that there are more than 5 depth data points, otherwise exclude the profile file
                if np.sum(~np.isnan(testvar)) < 5:
                    os.rename(f, f'{f}.exclude')
                    logging.debug('<5 depth data points found in file, exluding from dataset: {:s}'.format(f))
                    summary += 1
                # check that the number of 1m depth bins with data is > (depth range / 10) bins (minimum 5 bins)
                else:
                    df = testvar.to_dataframe()
                    
                    # convert values <0.1 to 0.1
                    df.loc[df[testvar.name] < 0.1] = 0.1
                    
                    # calculate number of bins allowed
                    depth_range = np.nanmax(df[testvar.name]) - np.nanmin(df[testvar.name])
                    bins_allowed = math.ceil(depth_range / 10)

                    if bins_allowed < 5:
                        bins_allowed = 5
                    
                    # bin the data in 1m depth bins
                    stride = 1
                    bins = np.arange(0, np.nanmax(df[testvar.name]) + stride, stride)
                    cut = pd.cut(df[testvar.name], bins)
                    binned_df = df.groupby(cut, observed=False).mean().dropna()  # depth bins, drop the nans

                    # check if number of 1m depth bins with data > than bins allowed, otherwise exclude the profile file
                    if len(binned_df) < bins_allowed:
                        os.rename(f, f'{f}.exclude')
                        logging.debug(f'Count of 1m depth bins ({len(binned_df)}) is less than bins allowed ({bins_allowed}) in file: {f}')
                        summary += 1

            logging.info('{:} of {:} files either had <5 depth data points or 1m depth bins was less than bins allowed'.format(summary, len(ncfiles)))


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
