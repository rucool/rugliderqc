#!/usr/bin/env python

"""
Author: lgarzio on 12/22/2023
Last modified: lgarzio on 1/2/2024
Calculate additional science variables, eg. pH and dissolved oxygen in mg/L
"""

import os
import argparse
import sys
import glob
import xarray as xr
import numpy as np
import pandas as pd
from ast import literal_eval
from rugliderqc.calc import oxygen_conversion_umol_to_mg, phcalc
from rugliderqc.common import find_glider_deployment_datapath, find_glider_deployments_rootdir, set_encoding
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname
from ioos_qc.utils import load_config_as_dict as loadconfig
pd.set_option('display.width', 320, "display.max_columns", 10)


def apply_qc(dataset, varname):
    """
    Make a copy of a data array and convert qartod_summary_flag and hysteresis test values (if applicable) of
    suspect (3) and fail (4) to nans
    :param dataset: xarray dataset
    :param varname: sensor variable name (e.g. conductivity)
    """
    datacopy = dataset[varname].copy()
    try:
        qv = f'{varname}_qartod_summary_flag'
        qv_vals = dataset[qv].values
        qv_idx = np.where(np.logical_or(qv_vals == 3, qv_vals == 4))[0]
        datacopy[qv_idx] = np.nan
    except KeyError:
        print(f'No QARTOD QC variables available for {varname}')

    # remove invalid pH reference voltages
    if 'ph_ref_voltage' in varname:
        zeros = np.where(datacopy == 0.0)[0]
        datacopy[zeros] = np.nan

    # apply CTD hysteresis test QC
    try:
        qv = f'{varname}_hysteresis_test'
        qv_vals = dataset[qv].values
        qv_idx = np.where(np.logical_or(qv_vals == 3, qv_vals == 4))[0]
        datacopy[qv_idx] = np.nan
    except KeyError:
        print(f'No CTD hysteresis test variables available for {varname}')

    if varname == 'salinity':
        qv_list = ['conductivity_hysteresis_test', 'temperature_hysteresis_test']
        for qv in qv_list:
            try:
                qv_vals = dataset[qv].values
                qv_idx = np.where(np.logical_or(qv_vals == 3, qv_vals == 4))[0]
                datacopy[qv_idx] = np.nan
            except KeyError:
                print(f'No CTD hysteresis test variable: {qv}')

    return datacopy


def calculate_ph(dataset, varname, log):
    data = apply_qc(dataset, varname)

    # get the calibration information
    try:
        cc = dataset.instrument_pH.attrs['calibration_coefficients']
    except AttributeError:
        log.error('instrument_pH variable not provided in dataset')
        cc = None
    except KeyError:
        log.error('instrument_pH attribute "calibration_coefficients" not provided')
        cc = None

    if not cc:
        log.error('Cannot calculate pH without calibration information')
    else:
        cc = literal_eval(cc)
        pressure = dataset.pressure  # pressure in dbar (as of 12/21/2023 the units in the files are incorrect (bar))
        temp = apply_qc(dataset, 'temperature')
        sal = apply_qc(dataset, 'salinity')

        data_dict = dict(time=dataset.time.values,
                         phvolt=data.values,
                         pressure=pressure.values,
                         temp=temp.values,
                         sal=sal.values)

        df = pd.DataFrame(data_dict)

        # interpolate CTD data in order to calculate pH
        df['pressure_interp'] = df['pressure'].interpolate(method='linear', limit_direction='both', limit=2)
        df['temp_interp'] = df['temp'].interpolate(method='linear', limit_direction='both', limit=2)
        df['sal_interp'] = df['sal'].interpolate(method='linear', limit_direction='both', limit=2)

        # calculate 6- or 12-order pressure polynomial
        try:
            # 12-order polynomial
            f_p = np.polyval(
                [cc['f12'], cc['f11'], cc['f10'], cc['f9'], cc['f8'], cc['f7'], cc['f6'], cc['f5'], cc['f4'],
                 cc['f3'], cc['f2'], cc['f1'], 0], df.pressure_interp)
            k2 = [cc['k2f3'], cc['k2f2'], cc['k2f1'], cc['k2f0']]
        except KeyError:
            # 6-order polynomial
            f_p = np.polyval([cc['f6'], cc['f5'], cc['f4'], cc['f3'], cc['f2'], cc['f1'], 0], df.pressure_interp)
            k2 = cc['k2']

        df['f_p'] = f_p
        phfree, phtot = phcalc(df.phvolt, df.pressure_interp, df.temp_interp, df.sal_interp, cc['k0'], k2, df.f_p)

        return np.array(phtot)


def convert_do_mgL(dataset, varname, log):
    data = apply_qc(dataset, varname)
    data_transformed = oxygen_conversion_umol_to_mg(data)
    return data_transformed


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

            # Set the deployment qc configuration path
            deployment_location = data_path.split('/data')[0]
            deployment_qc_config_root = os.path.join(deployment_location, 'config', 'qc')
            if not os.path.isdir(deployment_qc_config_root):
                logging.warning('Invalid deployment QC config root: {:s}'.format(deployment_qc_config_root))

            # Determine if the test should be run or not
            qctests_config_file = os.path.join(deployment_qc_config_root, 'qctests.yml')
            if os.path.isfile(qctests_config_file):
                qctests_config_dict = loadconfig(qctests_config_file)
                if not qctests_config_dict['additional_sci_calculations']:
                    logging.warning(
                        'Not calculating additional science vars because test is turned off, check: {:s}'.format(
                            qctests_config_file))
                    continue

            # Get variable names from the config file
            procvar_config_file = os.path.join(qc_config_root, 'sciencevar_processing.yml')
            if not os.path.isfile(procvar_config_file):
                logging.error('Invalid science variable config file: {:s}.'.format(procvar_config_file))
                status = 1
                continue

            proc_vars = loadconfig(procvar_config_file)

            # List the netcdf files in qc_queue
            ncfiles = sorted(glob.glob(os.path.join(data_path, 'qc_queue', '*.nc')))

            if len(ncfiles) == 0:
                logging.error(' 0 files found: {:s}'.format(os.path.join(data_path, 'qc_queue')))
                status = 1
                continue

            logging.info('Calculating additional science data: {:s}'.format(os.path.join(data_path, 'qc_queue')))

            calculated_vars = []

            # Iterate through files, apply QC to relevant variables
            for f in ncfiles:
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

                for pv in proc_vars.items():
                    variable_name = pv[0]
                    vardict = pv[1]
                    try:
                        # evaluate the function specified in the config file
                        data_calculated = eval(vardict['calculation'])(ds, variable_name, logging)

                        # grab the original attributes and update with any additional attributes from the config file
                        attrs = ds[variable_name].attrs.copy()
                        attrs.update(vardict['attrs'])

                        da = xr.DataArray(data_calculated, coords=ds[variable_name].coords, dims=ds[variable_name].dims,
                                          name=vardict['nc_var_name'], attrs=attrs)

                        # use the encoding from the original data variable
                        set_encoding(da, original_encoding=ds[variable_name].encoding)

                        ds[vardict['nc_var_name']] = da

                        file_modified += 1

                        if vardict['nc_var_name'] not in calculated_vars:
                            calculated_vars.append(vardict['nc_var_name'])
                    except KeyError:
                        continue

                # save the file if variables were added
                if file_modified > 0:
                    ds.to_netcdf(f)

            if len(calculated_vars) == 0:
                calculated_vars = ['no additional sci vars to calculate']
            logging.info(f'Finished calculating additional science variables: {",".join(calculated_vars)}')

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
