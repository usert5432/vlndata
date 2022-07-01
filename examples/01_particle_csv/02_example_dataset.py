#!/usr/bin/env python

"""This example constructs a pytorch-like dataset from a columnar data
stored in `./data.csv`.

It also illustrates how to insert new variables into the data frame.
"""

import numpy as np

from vlndata.data_frame import CSVFrame, VarFrame
from vlndata.dataset    import VLDataset

def calc_average_particle_energy(df):
    """A func that calculate average particle energy for each event in frame"""
    return np.fromiter(
        (
            df.get_vlarr(column = 'particle_energy', index = index).mean()
                for index in range(len(df))
        ),
        dtype = np.float32,
        count = len(df)
    )

def load_dataframe(path):
    """Load frame from the disc and add a new column of avg particle energies"""
    # Load basic data frame
    df = CSVFrame(path)

    # Add a new column to data frame that holds average particle energy for
    # each event
    df = VarFrame(
        df, { 'avg_particle_energy' : calc_average_particle_energy }
    )

    return df

def construct_dataset(df):
    """Construct a vl dataset from a data frame.

    This dataset will extract two groups of columns from the data frame.

    The first group will contain scalar values of `event_id` and
    `avg_particle_energy` columns.

    The second group will hold variable length array values corresponding
    to `particle_energy` and `particle_dir_z` columns.

    For each row in the original data frame, this dataset will return
    a dictionary of shape:
        { 'group1' : (2, ), 'group2' : (L, 2) }
    (where L is a length of a variable length array), according to the
    constructor below:
    """
    return VLDataset(
        df,
        scalar_groups = { 'group1' : [ 'event_id', 'avg_particle_energy', ] },
        vlarr_groups  = { 'group2' : [ 'particle_energy', 'particle_dir_z' ] },
    )


df   = load_dataframe('./data.csv')
dset = construct_dataset(df)

for idx in range(len(dset)):
    data = dset[idx]

    # data['group1'] : (2, )
    # c.f. `scalar_groups` above
    event_id            = data['group1'][0]

    # this energy was automatically calculated using
    #  `calc_average_particle_energy`
    avg_particle_energy = data['group1'][1]

    # data['group2'] : (L, 2)
    # c.f. `vlarr_groups` above
    particle_energies = data['group2'][:, 0]
    particle_dir_z    = data['group2'][:, 1]

    manual_avg_energy = particle_energies.mean()

    # Check that energies calculated by `calc_average_particle_energy`
    # match manually calculated average particle energies
    assert manual_avg_energy == avg_particle_energy

    print(
        f'Event: {event_id:g}.'
        f' Calculated average energy: {avg_particle_energy:.2f}.'
        f' Manual average energy: {manual_avg_energy:.2f}'
    )

