#!/usr/bin/env python

import os
import argparse
import numpy as np
from dsph_gnn import utils

FLAGS = None

def parse_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input', required=True,
        help='path to raw dataset in HDF5 format')
    parser.add_argument(
        '-o', '--output', required=True,
        help='path to processed dataset in HDF5 format')
    parser.add_argument(
        '--velocity-error', required=False, nargs=2, type=float,
        default=(0.1, 0.0), help='velocity error in [km / s]')
    parser.add_argument(
        '--num-samples', required=False, type=int, default=1,
        help='Number of samples for velocity')
    parser.add_argument(
        '--proper-motion', action='store_true', default=False,
        help='Enable to store proper motion')
    parser.add_argument(
        '-s', '--seed', required=False, type=int,
        help='Numpy seed for reproducibility')
    return parser.parse_args()


if __name__ == '__main__':
    ''' Process raw data '''

    # Read cmd args
    FLAGS = parse_cmd()

    # Set numpy seed
    np.random.seed(FLAGS.seed)

    # Read raw dataset
    print('Read raw dataset from {}'.format(FLAGS.input))
    features, labels, headers = utils.read_ds(FLAGS.input)
    num_samples = len(features)
    nx = 2  # number of position dimension

    print('Number of samples: {}'.format(num_samples))
    print('Velocity error: {} [km / s]'.format(FLAGS.velocity_error))
    print('Proper motion: {}'.format(FLAGS.proper_motion))
    print('Random seed: {}'.format(FLAGS.seed))

    # Iterate over all graphs
    new_features = []
    for i in range(num_samples):
        # get 6D Cartesian positions and velocities
        x, y, z, vx, vy, vz = features[i].T


        if not FLAGS.proper_motion:
            # assuming a plane x-y, we will only care about x, y, and vz
            # add velocity errors to vz
            verr = np.random.normal(
                loc=FLAGS.velocity_error[0], scale=FLAGS.velocity_error[1] ,
                size=len(vz))
            vz = vz + verr

            # add to new features
            feat = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1), vz.reshape(-1, 1)])
            new_features.append(feat)
            headers.update({'features': ['x', 'y', 'vz']})

        else:
            # assuming a plane x-y, we will only care about x, y, vx, vy, vz
            # add veocity errors to velocity
            vx_err = np.random.normal(
                loc=FLAGS.velocity_error[0], scale=FLAGS.velocity_error[1],
                size=len(vx))
            vy_err = np.random.normal(
                loc=FLAGS.velocity_error[0], scale=FLAGS.velocity_error[1],
                size=len(vy))
            vz_err = np.random.normal(
                loc=FLAGS.velocity_error[0], scale=FLAGS.velocity_error[1],
                size=len(vz))
            vx = vx + vx_err
            vy = vy + vy_err
            vz = vz + vz_err

            # add to new features
            feat = np.hstack([
                x.reshape(-1, 1),
                y.reshape(-1, 1),
                vx.reshape(-1, 1),
                vy.reshape(-1, 1),
                vz.reshape(-1, 1)
            ])
            new_features.append(feat)
            headers.update({'features': ['x', 'y', 'vx', 'vy', 'vz']})


    # Update attributes
    headers.update({
        'nx': nx,
        'verr': FLAGS.velocity_error,
        'seed': str(FLAGS.seed),
    })

    os.makedirs(os.path.dirname(FLAGS.output), exist_ok=True)

    # Write a processed dataset
    print('Write processed dataset to {}'.format(FLAGS.output))
    utils.write_ds(
        FLAGS.output, new_features,
        graph_properties={"labels": labels},
        headers=headers)


