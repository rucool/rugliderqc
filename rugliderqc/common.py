#!/usr/bin/env python

import os
import pytz
from dateutil import parser


def find_glider_deployment_datapath(logger, deployment, deployments_root, dataset_type, cdm_data_type, mode):
    try:
        (glider, trajectory) = deployment.split('-')
        try:
            trajectory_dt = parser.parse(trajectory).replace(tzinfo=pytz.UTC)
        except ValueError as e:
            logger.error('Error parsing trajectory date {:s}: {:}'.format(trajectory, e))
            trajectory_dt = None
            data_path = None
            deployment_location = None

        if trajectory_dt:
            trajectory = '{:s}-{:s}'.format(glider, trajectory_dt.strftime('%Y%m%dT%H%M'))
            deployment_name = os.path.join('{:0.0f}'.format(trajectory_dt.year), trajectory)

            # Create fully-qualified path to the deployment location
            deployment_location = os.path.join(deployments_root, deployment_name)
            if os.path.isdir(deployment_location):
                # Set the deployment netcdf data path
                data_path = os.path.join(deployment_location, 'data', 'out', 'nc',
                                         '{:s}-{:s}/{:s}'.format(dataset_type, cdm_data_type, mode))
                if not os.path.isdir(data_path):
                    logger.warning('{:s} data directory not found: {:s}'.format(trajectory, data_path))
                    data_path = None
                    deployment_location = None
            else:
                logger.warning('Deployment location does not exist: {:s}'.format(deployment_location))
                data_path = None
                deployment_location = None

    except ValueError as e:
        logger.error('Error parsing invalid deployment name {:s}: {:}'.format(deployment, e))
        data_path = None
        deployment_location = None

    return data_path, deployment_location


def find_glider_deployments_rootdir(logger):
    # Find the glider deployments root directory
    data_home = os.getenv('GLIDER_DATA_HOME')
    if not data_home:
        logger.error('GLIDER_DATA_HOME not set')
        return 1, 1
    elif not os.path.isdir(data_home):
        logger.error('Invalid GLIDER_DATA_HOME: {:s}'.format(data_home))
        return 1, 1

    deployments_root = os.path.join(data_home, 'deployments')
    if not os.path.isdir(deployments_root):
        logger.warning('Invalid deployments root: {:s}'.format(deployments_root))
        return 1, 1

    return data_home, deployments_root
