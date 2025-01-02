#!/usr/bin/env python

"""
Author: lgarzio on 12/22/2023
Last modified: lgarzio on 12/20/2024
Calculate additional science variables defined in the sciencevar_processing.yml config file,
eg. dissolved oxygen in mg/L, pH, TA, and omega
"""

import os
import argparse
import sys
import glob
import datetime as dt
import xarray as xr
import numpy as np
import pandas as pd
from itertools import chain
from ast import literal_eval
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from rugliderqc.calc import oxygen_conversion_umol_to_mg, phcalc
import rugliderqc.common as cf
from rugliderqc.loggers import logfile_basename, setup_logger, logfile_deploymentname
from ioos_qc.utils import load_config_as_dict as loadconfig
import PyCO2SYS as pyco2
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
        print(f'No QARTOD QC summary flag available for {varname}')
        # see if there are any other QC variables if a summary flag isn't available
        qv_vars = [x for x in list(dataset.data_vars) if f'{varname}_qartod_' in x]
        for qv in qv_vars:
            qv_vals = dataset[qv].values
            qv_idx = np.where(np.logical_or(qv_vals == 3, qv_vals == 4))[0]
            datacopy[qv_idx] = np.nan

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
        pressure = dataset.pressure  # pressure in dbar (the units in the files are incorrect (bar))
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
            try:
                # 6-order polynomial
                f_p = np.polyval([cc['f6'], cc['f5'], cc['f4'], cc['f3'], cc['f2'], cc['f1'], 0], df.pressure_interp)
                k2 = cc['k2']
            except KeyError:
                log.error('calibration_coefficients not formatted correctly')

        df['f_p'] = f_p
        phfree, phtot = phcalc(df.phvolt, df.pressure_interp, df.temp_interp, df.sal_interp, cc['k0'], k2, df.f_p)

        return np.array(phtot)


def calculate_omega(dataset, varname, log):
    try:
        data_dict = dict(
            salinity=apply_qc(dataset, 'salinity').values,
            pressure=apply_qc(dataset, 'pressure').values,
            temperature=apply_qc(dataset, 'temperature').values,
            ph=apply_qc(dataset, 'pH').values,
            ta=dataset.total_alkalinity.values
        )

        # interpolate CTD variables for omega calculation
        df = pd.DataFrame(data_dict)
        sal_interp = df['salinity'].interpolate(method='linear', limit_direction='both', limit=2)
        pressure_interp = df['pressure'].interpolate(method='linear', limit_direction='both', limit=2)
        temperature_interp = df['temperature'].interpolate(method='linear', limit_direction='both', limit=2)

        # run CO2sys
        par1 = df['ta']  # Total Alkalinity
        par1_type = 1  # parameter 1 type (TA)
        par2 = df['ph']
        par2_type = 3  # parameter 2 type (pH)

        kwargs = dict(salinity=sal_interp,
                      temperature=temperature_interp,
                      pressure=pressure_interp,
                      opt_pH_scale=1,
                      opt_k_carbonic=4,
                      opt_k_bisulfate=1,
                      opt_total_borate=1,
                      opt_k_fluoride=2)

        results = pyco2.sys(par1, par2, par1_type, par2_type, **kwargs)
        omega_arag = results['saturation_aragonite']  # aragonite saturation state

        return omega_arag

    except KeyError:
        log.error("One or more variables (salinity, temperature, pressure, pH, total_alkalinity) not available "
                  "in the files to calculate omega.")
        return []


def calculate_ta(dataset, varname, log, configfile):
    ta = []
    coeffs = []
    try:
        sal = apply_qc(dataset, 'salinity')
        ph = apply_qc(dataset, 'pH')

        # get the config file
        cc = os.path.join(configfile, 'TA-salinity-regressions.yml')
        configdict = loadconfig(cc)

        # determine if the glider is within any regions specified
        for region, values in configdict.items():
            if Polygon(list(zip(values['boundaries']['longitude'], values['boundaries']['latitude']))).contains(
                    Point(dataset.profile_lon, dataset.profile_lat)):
                log.debug(f'profile within the {region} region')

                # if the profile is within a defined boundary, determine profile season to calculate TA for that season
                coeffs = values['coefficients']
                # season = str(np.unique(dataset['profile_time.season'])[0])
                profile_ts = cf.convert_epoch_ts(dataset['profile_time'])
                season = cf.return_season(profile_ts)
                equation_coeffs = coeffs[season]

                # interpolate salinity for TA calculation (QC already done for salinity and pH)
                data_dict = dict(
                    salinity=sal.values,
                    ph=ph.values
                )
                df = pd.DataFrame(data_dict)
                sal_interp = df['salinity'].interpolate(method='linear', limit_direction='both', limit=2)
                ta = np.array(equation_coeffs['m'] * sal_interp + equation_coeffs['b'])

                # TA timestamps should line up with pH timestamps
                idx = np.isnan(df['ph'])
                ta[idx] = np.nan

        return ta, coeffs

            # TODO what to do if glider is in multiple regions? should these have priorities? or are the regions never going to overlap?

    except KeyError:
        log.error("salinity not available in the files, can't calculate TA")
        return ta, coeffs


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
    loglevel = loglevel.upper()

    logFile_base = logfile_basename()
    logging_base = setup_logger('logging_base', loglevel, logFile_base)

    data_home, deployments_root = cf.find_glider_deployments_rootdir(logging_base, test)
    if isinstance(deployments_root, str):

        # Set the default qc configuration path
        qc_config_root = os.path.join(data_home, 'qc', 'config')
        if not os.path.isdir(qc_config_root):
            logging_base.warning('Invalid QC config root: {:s}'.format(qc_config_root))
            return 1

        # Set the path for the derived variable QC configuration files
        qc_config_derived = os.path.join(qc_config_root, 'derived_variables')
        if not os.path.isdir(qc_config_derived):
            logging_base.warning('Invalid QC config path for derived variables: {:s}'.format(qc_config_derived))
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
            deployment_location = data_path.split('/data')[0]
            deployment_qc_config_root = os.path.join(deployment_location, 'config', 'qc')
            if not os.path.isdir(deployment_qc_config_root):
                logging.warning('Invalid deployment QC config root: {:s}'.format(deployment_qc_config_root))

            # Determine if the test should be run or not
            qctests_config_file = os.path.join(deployment_qc_config_root, 'qctests.yml')
            if os.path.isfile(qctests_config_file):
                qctests_config_dict = loadconfig(qctests_config_file)
                if not qctests_config_dict['add_derived_variables']:
                    logging.warning(
                        'Not calculating additional derived vars because test is turned off, check: {:s}'.format(
                            qctests_config_file))
                    continue

            # Get variable names from the config file
            procvar_config_file = os.path.join(qc_config_root, 'sciencevar_processing-test.yml')
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

            new_vars_final = []

            # Iterate through files, calculate additional science variables if the raw variables are found in the files
            for f in ncfiles:
                file_modified = 0
                try:
                    with xr.open_dataset(f, decode_times=False) as ds:
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

                original_vars = list(ds.data_vars)

                for variable_name, vardict in proc_vars.items():
                    # check if the variable required for additional processing is in the file
                    try:
                        ds[variable_name]
                    except KeyError:
                        continue

                    # evaluate the function specified in the config file
                    try:
                        data_calculated = eval(vardict['calculation'])(ds, variable_name, logging)
                    except TypeError:
                        # calculating TA (need the config file)
                        # Returns TA and the equations used to calculate TA (for metadata)
                        data_calculated, attr_coeffs = eval(vardict['calculation'])(ds, variable_name, logging, qc_config_derived)

                    if isinstance(data_calculated, np.ndarray):
                        # grab the attributes from the source variable (if use_sourcevar_attrs=True in the config file)
                        # update with any additional attributes from the config file
                        if vardict['use_sourcevar_attrs']:
                            attrs = ds[variable_name].attrs.copy()
                        else:
                            attrs = dict()
                        attrs.update(vardict['attrs'])

                        # for TA, add the equation coefficients to attributes
                        try:
                            attrs['comment'] = attrs['comment'].replace('-insert_coefficients-', str(attr_coeffs))
                            del attr_coeffs
                        except NameError:
                            print(f'no calculation coefficients to add to this variable attrs: {vardict["nc_var_name"]}')

                        da = xr.DataArray(data_calculated, coords=ds[variable_name].coords, dims=ds[variable_name].dims,
                                          name=vardict['nc_var_name'], attrs=attrs)

                        # use the encoding from the original data variable
                        cf.set_encoding(da, original_encoding=ds[variable_name].encoding)

                        ds[vardict['nc_var_name']] = da

                        file_modified += 1

                        # run QC if specified in the sciencevar_processing.yml config file
                        try:
                            qc_config_file_list = vardict['runqc']
                            for qcf in qc_config_file_list:
                                # QC configuration filename
                                qc_config_file = os.path.join(qc_config_derived, qcf)

                                if not os.path.isfile(qc_config_file):
                                    logging.error('Invalid QC config file: {:s}.'.format(qc_config_file))
                                    status = 1
                                    continue

                                if 'gross_flatline' in qcf:
                                    # run IOOS QC gross range/flatline tests according to the specs in the config file
                                    # this adds the QC variables to the dataset
                                    cf.run_ioos_qc_gross_flatline(ds, qc_config_file)
                                elif 'spike' in qcf:
                                    # run IOOS QC spike test according to the specs in the config file
                                    # this adds the QC variables to the dataset
                                    cf.run_ioos_qc_spike(ds, qc_config_file)

                        except KeyError:
                            continue

                new_vars = list(set(list(ds.data_vars)) - set(original_vars))
                if len(new_vars) > 0:
                    new_vars_final.append(new_vars)

                # save the file if variables were added
                if file_modified > 0:

                    # update the history attr
                    now = dt.datetime.now(dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
                    if not hasattr(ds, 'history'):
                        ds.attrs['history'] = f'{now}: {os.path.basename(__file__)}'
                    else:
                        ds.attrs['history'] = f'{ds.attrs["history"]} {now}: {os.path.basename(__file__)}'

                    ds.to_netcdf(f)

            if len(new_vars_final) == 0:
                new_vars_final = ['no additional sci vars to calculate']
            else:
                new_vars_final = list(set(chain(*new_vars_final)))
            logging.info(f'Finished calculating additional science variables: {",".join(new_vars_final)}')

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
