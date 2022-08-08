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
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import CSVLogger

from dsph_gnn import data_module, utils
from dsph_gnn.envs import DEFAULT_RUN_PATH, DEFAULT_DATASET_PATH

FLAGS = None

# Parser cmd argument
def parse_cmd():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--output", required=False, type=str,
        help="Name of output")
    parser.add_argument(
        "--dataset-prefix", required=False, type=str, default=DEFAULT_DATASET_PATH,
        help="Prefix of dataset. Dataset is read at \"dataset_prefix/dataset_name\".")
    parser.add_argument(
        "--dataset-name", required=False, type=str, default="",
        help="Name of dataset. Dataset is read at \"dataset_prefix/dataset_name\"")
    parser.add_argument(
        "--run-prefix", required=False, type=str, default=DEFAULT_RUN_PATH,
        help="Prefix of run. Output is saved at \"run_prefix/run_name\".")
    parser.add_argument(
        "--run-name", required=False, type=str, default="default",
        help="Name of run. Output is saved at \"run_prefix/run_name\".")
    parser.add_argument(
        "--run-version", required=False, type=str, default="best",
        help="Name of version")
    parser.add_argument(
        "--num-posteriors", required=False, type=int, default=5000,
        help="Number of posterior samples. For prediction, only")
    parser.add_argument(
        "--batch-size", required=False, type=int, default=64,
        help="Batch size")
    parser.add_argument(
        "--num-workers", required=False, type=int, default=1,
        help="Number of workers")
    return parser.parse_args()

# Set logger
def set_logger():
    ''' Set up stdv out logger and file handler '''
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

if __name__ == "__main__":
    """ Evaluate a processed datasetusing a pre-trained model """
    # Parse cmd args
    FLAGS = parse_cmd()
    logger = set_logger()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get model from best checkpoint
    best_run_path = utils.get_run(
        FLAGS.run_name, version=FLAGS.run_version, prefix=FLAGS.run_prefix)
    best_checkpoint, _ = utils.get_best_checkpoint(best_run_path)
    logger.info(f"Read checkpoint from {best_checkpoint}")
    model = data_module.DataModule.load_from_checkpoint(
        best_checkpoint, num_posteriors=FLAGS.num_posteriors)
    model = model.to(device)
    model = model.eval()

    # Get dataset
    test_path = utils.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_prefix, is_dir=False)
    test_features, test_labels, test_attrs = utils.read_ds(test_path)
    test_ds = utils.create_ds(
        test_features, test_labels, model.graph_transforms)
    test_loader = DataLoader(
        test_ds, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers,
        pin_memory=True if torch.cuda.is_available() else False)

    # Evaluate dataset
    logger.info('Start inference')
    posteriors = []
    truths = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            context = model.model(batch.x, batch.edge_index, batch.batch, batch.edge_weight)
            samples = model.model.maf.sample(
                num_samples=FLAGS.num_posteriors, context=context)
            posteriors.append(samples.cpu().numpy())
            truths.append(batch.y.cpu().numpy())
    posteriors = np.vstack(posteriors)
    truths = np.concatenate(truths)

    # Write evaluate result to HDF5 file
    if os.path.isabs(FLAGS.output):
        output = output
    else:
        outdir = os.path.join(best_run_path, "evaluate_output")
        os.makedirs(outdir, exist_ok=True)
        output = os.path.join(outdir, FLAGS.output)

    logger.info('Writing evaluation result to {}'.format(output))
    with h5py.File(output, 'w') as f:
        f.create_dataset('posteriors', data=posteriors)
        f.create_dataset('truths', data=truths)
