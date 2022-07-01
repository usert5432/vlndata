#!/usr/bin/env python

"""
A simple example on how to train an RNN model using vlndata data primitives.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from vlndata.data_frame        import CSVFrame,  VarFrame
from vlndata.dataset           import VLDataset, DatasetTransform
from vlndata.dataset.transform import NoiseTransform
from vlndata.data_loader       import DataLoader

class KerasSequence(keras.utils.Sequence):
    """A wrapper to convert pytorch-like data loader into keras Sequence"""

    def __init__(self, dl, target_group = 'target', **kwargs):
        super().__init__(**kwargs)
        self._dl     = dl
        self._target = target_group

    def __len__(self):
        return len(self._dl)

    def __getitem__(self, index):
        result = self._dl[index]
        target = result.pop(self._target)

        return (result, target)

def calc_average_particle_energy(df):
    """A func that calculates average particle energy for each event"""
    return np.fromiter(
        (
            df.get_vlarr(column = 'particle_energy', index = index).mean()
                for index in range(len(df))
        ),
        dtype = np.float32,
        count = len(df)
    )

def load_dataframe(path):
    """Load frame from disc and add a new column of avg particle energies"""
    # Load basic data frame
    df = CSVFrame(path, dtype = np.float32)

    # Add a new column to data frame that holds average particle energy for
    # each event
    df = VarFrame(
        df, { 'avg_particle_energy' : calc_average_particle_energy }
    )

    return df

def construct_dataset(df, add_noise = True):
    """Construct a vl dataset from a data frame.

    Also this function adds Gaussian noise to the "particle_energy" column.
    """

    dset = VLDataset(
        df,
        scalar_groups = { 'target'          : [ 'avg_particle_energy', ] },
        vlarr_groups  = { 'particle_inputs' : [ 'particle_energy', ] },
    )

    if add_noise:
        # Add noise to the 'particle_inputs/particle_energy' values
        noise = { 'name' : 'normal', 'mu' : 0, 'sigma' : 0.1 }
        noise_transform = NoiseTransform(
            noise,
            vlarr_groups = { 'particle_inputs' : [ 'particle_energy', ] }
        )

        dset = DatasetTransform(dset, [ noise_transform, ])

    return dset

def construct_loader(dset, batch_size = 2, shuffle = True):
    """Construct a pytorch-like loader from a dataset"""
    return DataLoader(dset, batch_size, shuffle = shuffle)

def load_data(path, batch_size = 2):
    """Parse file `path` and construct a data loader from it"""
    df = load_dataframe(path)

    dset_train = construct_dataset(df, add_noise = True)
    dset_val   = construct_dataset(df, add_noise = False)

    dl_train = construct_loader(dset_train, batch_size,     shuffle = True)
    dl_val   = construct_loader(dset_val,   batch_size = 1, shuffle = False)

    return KerasSequence(dl_train), KerasSequence(dl_val)

def construct_keras_model():
    """Construct and compile a simple LSTM keras model"""
    input_layer = keras.layers.Input(
        shape = (None, 1), name = 'particle_inputs'
    )

    lstm_layer   = keras.layers.LSTM(units = 16)(input_layer)

    output_layer = keras.layers.Dense(8, activation = 'relu')(lstm_layer)
    output_layer = keras.layers.Dense(1, name = 'target')(output_layer)

    model = keras.Model(inputs = input_layer, outputs = output_layer)
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate = 1e-3),
        loss      = keras.losses.MSE,
    )

    return model

def eval_model(model, dl_val):

    for index, (inputs, target) in enumerate(dl_val):
        pred   = float(model(inputs))
        target = float(target)
        error  = (pred - target)

        print(
            f'Event: {index}. Prediction: {pred:.2f}, Target: {target:.2f}'
            f'. Error: {error:.3f}.'
        )

PATH       = './data.csv'
BATCH_SIZE = 2
EPOCHS     = 500

print("Loading data...")
dl_train, dl_val = load_data(PATH, BATCH_SIZE)
model = construct_keras_model()

print("Training model...")
model.fit(dl_train, epochs = EPOCHS)

print("Evaluating model...")
eval_model(model, dl_val)

