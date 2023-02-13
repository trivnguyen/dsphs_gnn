#!/usr/bin/env python

import os
import sys
import json
import argparse
import logging
import h5py

import numpy as np
import torch
import pytorch_lightning as pl
from scipy.spatial.transform import Rotation
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import CSVLogger

from dsphs_gnn import data_module, utils
from dsphs_gnn.envs import DEFAULT_RUN_PATH, DEFAULT_DATASET_PATH

FLAGS = None

# Set logger
def set_logger():
    ''' Set up stdv out logger and file handler '''
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

# Parser cmd argument
def parse_cmd():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Input/output args
    parser.add_argument(
        '-i', '--input', required=True,
        help='Path to input particle data in HDF5 format')
    parser.add_argument(
        "--output", required=False, type=str,
        help="Name of output")

    # FIRE args
    parser.add_argument(
        '--num-parts', required=False, type=int,
        help='Maximum number of particles to consider')
    parser.add_argument(
        '--projection', required=False, type=str.lower, default='random',
        help='Projection to consider. Default to a random projection.')
    parser.add_argument(
        '--species', type=str, default='star', required=False,
        help='Species of particle to sample kinematics')

    # NN args
    parser.add_argument(
        "--run-prefix", required=False, type=str, default=DEFAULT_RUN_PATH,
        help="Prefix of run. Output is saved at \"run_prefix/run_name\".")
    parser.add_argument(
        "--run-name", required=False, type=str, default="default",
        help="Name of run. Output is saved at \"run_prefix/run_name\".")
    parser.add_argument(
        "--run-version", required=False, type=str, default="best",
        help="Name of version")

    # posteriors args
    parser.add_argument(
        "--num-runs", required=False, type=int, default=100,
        help="Number of realizations of one galaxy"
    )
    parser.add_argument(
        "--num-posteriors", required=False, type=int, default=5000,
        help="Number of posterior samples. For prediction, only")

    # misc args
    parser.add_argument(
        "--batch-size", required=False, type=int, default=64,
        help="Batch size")
    parser.add_argument(
        "--num-workers", required=False, type=int, default=1,
        help="Number of workers")
    return parser.parse_args()

# Read FIRE input
def read_fire(
    path, species='star', num_parts=None, num_runs=1,
    projection="random"):
    """ Read FIRE galaxy from path"""

    # read position and velocity of FIRE galaxy from path
    with h5py.File(path, 'r') as f:
        position = f[f'{species}/position'][:]
        velocity = f[f'{species}/velocity'][:]

    features = []
    fake_labels = np.zeros((num_runs, 5))   # fake labels
    for i in range(num_runs):
        # randomly select num_parts stars
        if num_parts is None:
            num_parts = len(position)
        else:
            select = np.random.permutation(len(position))[:num_parts]
            position = position[select]
            velocity = velocity[select]

        # project coordinates and calculate the projected radius
        if projection == "random":
            rot = Rotation.random().as_matrix()
            position = position @ rot.T
            velocity = velocity @ rot.T
            X = position[:, 0]
            Y = position[:, 1]
            v = velocity[:, 2]
        elif projection in ("xy", "yx"):
            X = position[:, 0]
            Y = position[:, 1]
            v = velocity[:, 2]
        elif projection in ("yz", "zy"):
            X = position[:, 1]
            Y = position[:, 2]
            v = velocity[:, 0]
        elif projection in ("zx", "xz"):
            X = position[:, 2]
            Y = position[:, 0]
            v = velocity[:, 1]
        else:
            raise ValueError(f"invalid projection {projection}")
        features.append(np.array([X, Y, v]).T)
    features = np.stack(features)

    return features, fake_labels


def main(FLAGS):
    """ Evaluate a processed datasetusing a pre-trained model """
    # Parse cmd args
    FLAGS = parse_cmd()
    LOGGER = set_logger()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get model from best checkpoint
    best_run_path = utils.get_run(
        FLAGS.run_name, version=FLAGS.run_version, prefix=FLAGS.run_prefix)
    best_checkpoint = utils.get_best_checkpoint(best_run_path)[0]
    model = data_module.DataModule.load_from_checkpoint(
        best_checkpoint, num_posteriors=FLAGS.num_posteriors,
        device=DEVICE).eval()

    LOGGER.info(f"Read checkpoint from {best_checkpoint}")

    # Get dataset and create data loader
    features, fake_labels = read_fire(
        FLAGS.input, species=FLAGS.species,
        num_parts=FLAGS.num_parts, num_runs=FLAGS.num_runs,
        projection=FLAGS.projection
    )
    loader = DataLoader(
        utils.create_ds(features, fake_labels, model.graph_transforms),
        batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Evaluate dataset
    LOGGER.info('Start inference')
    posteriors = []
    truths = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            context = model.model(
                batch.x, batch.edge_index, batch.batch,
                batch.edge_weight)
            samples = model.model.maf.sample(
                num_samples=FLAGS.num_posteriors, context=context)
            posteriors.append(samples.cpu().numpy())
            truths.append(batch.y.cpu().numpy())
    posteriors = np.vstack(posteriors)
    truths = np.concatenate(truths)

    # Write evaluate result to HDF5 file
    if os.path.isabs(FLAGS.output):
        output = FLAGS.output
    else:
        outdir = os.path.join(best_run_path, "evaluate_output")
        os.makedirs(outdir, exist_ok=True)
        output = os.path.join(outdir, FLAGS.output)

    LOGGER.info('Writing evaluation result to {}'.format(output))
    with h5py.File(output, 'w') as f:
        f.create_dataset('posteriors', data=posteriors)
        f.create_dataset('truths', data=truths)
