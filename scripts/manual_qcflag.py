#!/usr/bin/env python

"""
Author: lgarzio on 5/1/2024
Last modified: lgarzio on 5/7/2024
Add a manual comment to a deployment and the option to convert values to nan using the manual_flag.yml config file
located in /home/coolgroup/slocum/deployments/YYYY/glider-YYYYMMDDTHHMM/config/qc
"""

import os
import argparse
import sys
import glob
import datetime as dt
import xarray as xr
import pandas as pd
import numpy as np
from rugliderqc.common import find_glider_deployment_datapath, find_glider_deployments_rootdir
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname
from ioos_qc.utils import load_config_as_dict as loadconfig
pd.set_option('display.width', 320, "display.max_columns", 10)


def main(args):
    status = 0

    loglevel = args.loglevel.upper()
    cdm_data_type = args.cdm_data_type
    mode = args.mode
    dataset_type = args.level
    test = args.test
    loglevel = loglevel.upper()

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

            # Set the deployment qc configuration path
            deployment_location = data_path.split('/data')[0]
            deployment_qc_config_root = os.path.join(deployment_location, 'config', 'qc')
            if not os.path.isdir(deployment_qc_config_root):
                logging.warning('Invalid deployment QC config root: {:s}'.format(deployment_qc_config_root))

            # Get the manual flag information from the deployment-specific manual_flag.yml file
            # If the file doesn't exist, no manual flags are added
            manflag_config_file = os.path.join(deployment_qc_config_root, 'manual_flag.yml')
            if not os.path.isfile(manflag_config_file):
                logging.error('manual_flag.yml file does not exist for this deployment: {:s}.'.format(manflag_config_file))
                status = 1
                continue

            manflags = loadconfig(manflag_config_file)

            # List the netcdf files
            ncfiles = sorted(glob.glob(os.path.join(data_path, '*.nc')))

            if len(ncfiles) == 0:
                logging.error(' 0 files found: {:s}'.format(data_path))
                status = 1
                continue

            logging.info('Adding manual QC flags')

            # Iterate through files, and add manual QC flags
            for f in ncfiles:
                logging.info(f)
                now = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
                file_modified = 0
                try:
                    with xr.open_dataset(f) as ds:
                        ds = ds.load()
                except OSError as e:
                    logging.error('Error reading file {:s} ({:})'.format(f, e))
                    os.rename(f, f'{f}.bad')
                    status = 1
                    continue
                except ValueError as e:
                    logging.error('Error reading file {:s} ({:})'.format(f, e))
                    os.rename(f, f'{f}.bad')
                    status = 1
                    continue

                for k, v in manflags.items():
                    if k == 'deployment_flag':
                        # if there is a deployment flag, add the comment to global attrs (comment and processing_level)
                        if not hasattr(ds, 'comment'):
                            ds.attrs['comment'] = f'{now}: {v["comment"]}'
                        else:
                            ds.attrs['comment'] = f'{ds.attrs["comment"]} {now}: {v["comment"]}'

                        if not hasattr(ds, 'processing_level'):
                            ds.attrs['processing_level'] = f'{now}: {v["comment"]}'
                        else:
                            ds.attrs['processing_level'] = f'{ds.attrs["processing_level"]} {now}: {v["comment"]}'

                        file_modified += 1

                    elif k == 'sensor_flag':
                        for kk, vv in v.items():
                            try:
                                ds[kk]
                                if vv['convert_to_nan']:
                                    # convert the data to NaNs between the timestamps specified in the config file
                                    starti = ds.time.to_index().get_loc(vv['start'], method='nearest')
                                    endi = ds.time.to_index().get_loc(vv['end'], method='nearest') + 1
                                    replace_shape = np.shape(ds[kk].values[starti:endi])
                                    ds[kk].values[starti:endi] = np.full(replace_shape, np.nan)

                                    file_modified += 1

                                # add to the variable attrs comment
                                if not hasattr(ds[kk], 'comment'):
                                    ds[kk].attrs['comment'] = f'{now}: {vv["comment"]}'
                                else:
                                    ds[kk].attrs['comment'] = f'{ds[kk].attrs["comment"]} {now}: {vv["comment"]}'

                                file_modified += 1

                            except KeyError:
                                continue

                # if the file was modified, add the script to the file history, change modified date and save the file
                if file_modified > 0:
                    if not hasattr(ds, 'history'):
                        ds.attrs['history'] = f'{now}: {os.path.realpath(__file__)}'
                    else:
                        ds.attrs['history'] = f'{ds.attrs["history"]} {now}: {os.path.realpath(__file__)}'

                    ds.attrs['date_modified'] = now

                    ds.to_netcdf(f)

            logging.info('Finished adding manual QC flags')

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
                            default='delayed')

    arg_parser.add_argument('--level',
                            choices=['sci', 'ngdac', 'raw'],
                            default='sci',
                            help='Dataset type')

    arg_parser.add_argument('-d', '--cdm_data_type',
                            help='Dataset type',
                            choices=['profile', 'trajectory'],
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
