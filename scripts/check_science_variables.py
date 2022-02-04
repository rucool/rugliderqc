#!/usr/bin/env python

"""
Author: lgarzio on 12/22/2021
Last modified: lgarzio on 2/3/2022
Checks files for CTD science variables (pressure, conductivity and temperature). Renames files ".nosci" if the file
doesn't contain any of those variables, or only contains pressure. Also converts CTD science variables to fill values
if conductivity and temperature are both 0.000, and dissolved oxygen science variables to fill values if
oxygen_concentration and optode_water_temperature are both 0.000.
"""

import os
import argparse
import sys
import glob
import xarray as xr
import numpy as np
from rugliderqc.common import find_glider_deployment_datapath, find_glider_deployments_rootdir
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname
from ioos_qc.utils import load_config_as_dict as loadconfig


def check_zeros(varname_dict, dataset, ds_modified, var1, var2):
    """
    Find indices where values for 2 variables are 0.0000 and convert all defined variables to fill values
    :param varname_dict: dictionary containing variables to modify if condition is met
    :param dataset: xarray dataset
    :param ds_modified: int value indicating if the dataset was modified
    :param var1: first variable name to test for condition (e.g. 'conductivity' or 'oxygen_concentration')
    :param var2: second variable name to test for condition (e.g. 'temperature' or 'optode_water_temperature')
    returns int value indicating if the dataset was modified
    """
    for key, variables in varname_dict.items():
        try:
            check_var1 = dataset[variables[var1]]
            check_var2 = dataset[variables[var2]]
        except KeyError:
            continue

        var1_zero_idx = np.where(check_var1 == 0.0000)[0]
        if len(var1_zero_idx) > 0:
            var2_zero_idx = np.where(check_var2 == 0.0000)[0]
            dointersect_idx = np.intersect1d(var1_zero_idx, var2_zero_idx)
            if len(dointersect_idx) > 0:
                for cv, varname in variables.items():
                    dataset[varname][dointersect_idx] = dataset[varname].encoding['_FillValue']
                    ds_modified += 1

    return ds_modified


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

            logging.info('Checking for science variables: {:s}'.format(os.path.join(data_path, 'qc_queue')))

            # Get all of the possible CTD variable names from the config file
            ctd_config_file = os.path.join(qc_config_root, 'ctd_variables.yml')
            if not os.path.isfile(ctd_config_file):
                logging.error('Invalid CTD variable name config file: {:s}.'.format(ctd_config_file))
                status = 1
                continue

            ctd_vars = loadconfig(ctd_config_file)

            # Get dissolved oxygen variable names
            oxygen_config_file = os.path.join(qc_config_root, 'oxygen_variables.yml')
            if not os.path.isfile(oxygen_config_file):
                logging.error('Invalid DO variable name config file: {:s}.'.format(oxygen_config_file))
                status = 1
                continue

            oxygen_vars = loadconfig(oxygen_config_file)

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
            zeros_removed = 0
            for f in ncfiles:
                modified = 0
                try:
                    with xr.open_dataset(f) as ds:
                        ds = ds.load()
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

                    # Set CTD values to fill values where conductivity and temperature both = 0.00
                    # Try all versions of CTD variable names
                    modified = check_zeros(ctd_vars, ds, modified, 'conductivity', 'temperature')

                    # Set DO values to fill values where oxygen_concentration and oxygen_saturation both = 0.00
                    modified = check_zeros(oxygen_vars, ds, modified, 'oxygen_concentration', 'optode_water_temperature')

                # if the file was modified, save the file and add to the summary
                if modified > 0:
                    ds.to_netcdf(f)
                    zeros_removed += 1

            logging.info('Found {:} files without CTD science variables (of {:} total files)'.format(summary, len(ncfiles)))
            logging.info('Removed 0.00 values (Teledyne fill values) for CTD and/or DO variables in {:} files (of {:} '
                         'total files)'.format(zeros_removed, len(ncfiles)))
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
