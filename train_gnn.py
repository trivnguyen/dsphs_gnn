#!/usr/bin/env python

import os
import sys
import json
import shutil
import argparse
import logging

import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import CSVLogger

from dsph_gnn import data_module, utils

FLAGS = None

# Parser cmd argument
def parse_cmd():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # input/output args
    parser.add_argument(
        "-i", "--input", required=True, nargs=2,
        help="Path to (train, validation) input files.")
    parser.add_argument(
        "-o", "--out-dir", required=True,
        help="Path to output directory.")

    # model and graph args
    parser.add_argument(
        "--in-channels", required=True, type=int,
        help='Number of input channels')
    parser.add_argument(
        "--out-channels", required=True, type=int,
        help='Number of output channels.')
    parser.add_argument(
        "--conv-name", required=False, type=str.upper, default='CHEB',
        help='Name of convolutional layers')
    parser.add_argument(
        "--num-layers", required=False, type=int, default=5,
        help="Number of convolutional layers")
    parser.add_argument(
        "--hidden-channels", required=False, type=int, default=128,
        help="Number of hidden channels in convolutional layers")
    parser.add_argument(
        "--num-layers-fc", required=False, type=int, default=3,
        help='Number of linear layers')
    parser.add_argument(
        "--hidden-channels-fc", required=False, type=int, default=128,
        help="Number of hidden channels in FC layer")
    parser.add_argument(
        "--num-transforms", required=False, type=int, default=4,
        help="Number of transforms for normalizing flows transformation")
    parser.add_argument(
        "--hidden-channels-flows", required=False, type=int, default=128,
        help="Number of hidden channels in normalizing flows layers")
    parser.add_argument(
        "--graph-name", required=False, type=str.upper, default='KNN',
        help="Graph name")
    parser.add_argument(
        "--edge-weight", required=False, action="store_true",
        help="If enable, compute edge attributes")
    parser.add_argument(
        "--edge-weight-norm", required=False, action="store_true",
        help="If enable, normalizing edge attributes")
    parser.add_argument(
        "--additional-hparams", required=False, type=str,
        help="Json file with additional  model and graph hyperparameters")

    # training args
    parser.add_argument(
        "--batch-size", required=False, type=int, default=64,
        help="Batch size")
    parser.add_argument(
        "--max-epochs", required=False, type=int, default=100,
        help="Maximum number of epochs. Stop training automatically if exceeds")
    parser.add_argument(
        "--num-workers", required=False, type=int, default=1,
        help="Number of workers")
    parser.add_argument(
        "--learning-rate", required=False, type=float, default=1e-3,
        help="Learning rate of optimizer")

    # misc args
    parser.add_argument(
        "--no-progress-bar", action="store_true",
        help="Enable to turn off progress bar")

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
    """ Train a GNN and NF """
    # Parse cmd args
    FLAGS = parse_cmd()
    logger = set_logger()

    # Create data module
    def create_model():
        if FLAGS.additional_hparams is not None:
            with open(FLAGS.additional_hparams, "r") as f:
                additional_hparams = json.load(f)
        else:
            additional_hparams = {
                "conv_hparams": {},
                "graph_hparams": {},
            }
        model_hparams = {
            "in_channels": FLAGS.in_channels,
            "out_channels": FLAGS.out_channels,
            "conv_name": FLAGS.conv_name,
            "conv_hparams": additional_hparams["conv_hparams"],
            "num_layers": FLAGS.num_layers,
            "hidden_channels": FLAGS.hidden_channels,
            "num_layers_fc": FLAGS.num_layers_fc,
            "hidden_channels_fc": FLAGS.hidden_channels_fc,
            "num_transforms": FLAGS.num_transforms,
            "hidden_channels_flows": FLAGS.hidden_channels_flows,
        }
        graph_hparams = {
            "graph_name" : FLAGS.graph_name,
            "graph_hparams": additional_hparams["graph_hparams"],
            "feature_hparams": {},
            "edge_weight": FLAGS.edge_weight,
            "edge_weight_norm": FLAGS.edge_weight_norm,
        }
        optimizer_hparams = {"lr": FLAGS.learning_rate}
        model = data_module.DataModule(
            model_hparams, graph_hparams, optimizer_hparams)
        return model
    model = create_model()
    logger.info(model)

    # Read in features and labels
    train_path, val_path = FLAGS.input
    train_features, train_labels, train_attrs = utils.read_ds(train_path)
    val_features, val_labels, val_attrs = utils.read_ds(val_path)

    # Create dataset from features and labels
    train_ds = utils.create_ds(
        train_features, train_labels, model.graph_transforms)
    val_ds = utils.create_ds(
        val_features, val_labels, model.graph_transforms)
    train_loader = DataLoader(
        train_ds, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers,
        pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(
        val_ds, batch_size=FLAGS.batch_size, num_workers=FLAGS.num_workers,
        pin_memory=True if torch.cuda.is_available() else False)

    # Create pytorch_lightning trainer
    trainer = pl.Trainer(
        default_root_dir=FLAGS.out_dir,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else 0,
        max_epochs=FLAGS.max_epochs,
        logger=CSVLogger(FLAGS.out_dir, name=None),
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                filename="{epoch}-{val_loss:.4f}", save_weights_only=True,
                mode="min", monitor="val_loss"),
            pl.callbacks.LearningRateMonitor("epoch"),
            pl.callbacks.early_stopping.EarlyStopping(
                monitor="val_loss", min_delta=0.00, patience=10,
                mode="min", verbose=True),
        ],
        enable_progress_bar=(not FLAGS.no_progress_bar)
    )

    # Start training
    trainer.fit(
        model=model, train_dataloaders=train_loader,
        val_dataloaders=val_loader)

    # Evaluate and return prediction for validation data




