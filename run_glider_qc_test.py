#!/usr/bin/env python

"""
Author: lgarzio on 12/7/2021
Last modified: lgarzio 3/11/2021
This is a wrapper script that imports tools to quality control RUCOOL's glider data.
"""

import argparse
import sys
import scripts

arg_parser = argparse.ArgumentParser(description="QC RUCOOL's glider data",
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

# check for files that are missing CTD science variables
scripts.check_science_variables.main(parsed_args)

# check files that have duplicate timestamps
scripts.check_duplicate_timestamps.main(parsed_args)

# apply QARTOD QC
scripts.glider_qartod_qc.main(parsed_args)

# check for severely-lagged CTD profile pairs
scripts.ctd_hysteresis_test.main(parsed_args)

# summarize QARTOD flags
scripts.summarize_qartod_flags.main(parsed_args)

# calculate optimal time shift for each segment for variables defined in config files (e.g. DO and pH voltages)
# requires a deployment time_shift.yml config file in ./glider-deployment/config/qc to run
scripts.time_shift.main(parsed_args)

# move the files to the parent directory to be sent to ERDDAP
scripts.move_nc_files.main(parsed_args)

sys.exit()
