# rugliderqc
A collection of python tools to quality control real-time and delayed-mode profile-based glider NetCDF files. It uses a modified version of the [ioos_qc](https://ioos.github.io/ioos_qc/) package to add quality flag variables to the datasets that are outlined in the Required and Strongly Recommended tests in the [IOOS QC Manual of Glider Data](https://cdn.ioos.noaa.gov/media/2017/12/Manual-for-QC-of-Glider-Data_05_09_16.pdf). We have also developed additional tests to further quality control glider data, such as a [test](https://github.com/rucool/rugliderqc/blob/master/scripts/ctd_hysteresis_test.py) that flags CTD profile pairs that are severely lagged, which can be an indication of CTD pump issues.

This code is designed to run on profile-based glider NetCDF files with a specific file and directory structure. The files are processed to NetCDF using the [gncutils package](https://github.com/kerfoot/gncutils).

Full documentation of this repo can be found in the [wiki](https://github.com/rucool/rugliderqc/wiki).

## Note: this repository is under development

## Installation

`git clone https://github.com/rucool/rugliderqc.git`

`cd rugliderqc`

`conda env create -f environment.yml`

`conda activate rugliderqc`

`pip install .`

**Requires replacing scripts in the ioos_qc package (.../envs/rugliderqc/lib/python3.9/site-packages/ioos_qc) with all files of the same name in ./ioos_qc_mods (as of 12/8/2021, only qartod.py) after the environment is created.**

## Usage

`python run_glider_qc.py glider-YYYYmmddTHHMM`

This wrapper script runs:

1. [check_science_variables.py](https://github.com/rucool/rugliderqc/blob/master/scripts/check_science_variables.py)
2. [check_duplicate_timestamps.py](https://github.com/rucool/rugliderqc/blob/master/scripts/check_duplicate_timestamps.py)
3. [glider_qartod_qc.py](https://github.com/rucool/rugliderqc/blob/master/scripts/glider_qartod_qc.py)
4. [interpolate_depth.py](https://github.com/rucool/rugliderqc/blob/master/scripts/interpolate_depth.py)
5. [ctd_hysteresis_test.py](https://github.com/rucool/rugliderqc/blob/master/scripts/ctd_hysteresis_test.py)
6. [summarize_qartod_flags.py](https://github.com/rucool/rugliderqc/blob/master/scripts/summarize_qartod_flags.py)
7. [time_shift.py](https://github.com/rucool/rugliderqc/blob/master/scripts/time_shift.py)
8. [add_derived_variables.py](https://github.com/rucool/rugliderqc/blob/master/scripts/add_derived_variables.py)
9. [move_nc_files.py](https://github.com/rucool/rugliderqc/blob/master/scripts/move_nc_files.py)

## Acknowledgements

Development was supported in part by the [Mid-Atlantic Regional Association Coastal Ocean Observing System](https://maracoos.org/).
