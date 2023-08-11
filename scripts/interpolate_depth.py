#!/usr/bin/env python

"""
Author: lgarzio on 8/11/2023
Last modified: lgarzio on 8/11/2023
Add interpolated depth to files.
"""

import os
import argparse
import sys
import glob
import xarray as xr
import numpy as np
from rugliderqc.common import find_glider_deployment_datapath, find_glider_deployments_rootdir, set_encoding
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname
from ioos_qc.utils import load_config_as_dict as loadconfig


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
        datacopy[qv_idx] = np.nan
    return datacopy


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
                if not qctests_config_dict['interp_depth']:
                    logging.warning(
                        'Not interpolating depth because test is turned off, check: {:s}'.format(
                            qctests_config_file))
                    continue

            logging.info('Interpolating depth: {:s}'.format(os.path.join(data_path, 'qc_queue')))

            # List the netcdf files in qc_queue
            ncfiles = sorted(glob.glob(os.path.join(data_path, 'qc_queue', '*.nc')))

            if len(ncfiles) == 0:
                logging.error(' 0 files found: {:s}'.format(os.path.join(data_path, 'qc_queue')))
                status = 1
                continue

            # Iterate through files, apply pressure QARTOD QC to depth, interpolate depth and add to files
            for f in ncfiles:
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

                # apply pressure QARTOD QC to depth. convert fail (4) QC flags to nan
                depthcopy = ds.depth.copy()
                for qv in [x for x in ds.data_vars if 'pressure_qartod' in x]:
                    qv_vals = ds[qv].values
                    qv_idx = np.where(qv_vals == 4)[0]
                    depthcopy[qv_idx] = np.nan

                # interpolate depth
                df = depthcopy.to_dataframe()
                depth_interp = df['depth'].interpolate(method='linear', limit_direction='both', limit=2).values

                attrs = ds.depth.attrs.copy()
                attrs['ancillary_variables'] = f'{attrs["ancillary_variables"]} depth'
                attrs['comment'] = f'Linear interpolated depth using pandas.DataFrame.interpolate'
                attrs['long_name'] = 'Interpolated Depth'
                attrs['source_sensor'] = 'depth'

                da = xr.DataArray(depth_interp.astype(ds.depth.dtype), coords=ds.depth.coords, dims=ds.depth.dims,
                                  name='depth_interpolated', attrs=attrs)

                # use the encoding from the original depth variable
                set_encoding(da, original_encoding=ds.depth.encoding)

                ds['depth_interpolated'] = da

                # save the file
                ds.to_netcdf(f)

            logging.info('Added depth_interpolated to {:} files)'.format(len(ncfiles)))

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
