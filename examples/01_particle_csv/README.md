# Overview

This is a basic example on how to use the `vlndata` primitives to parse
a columnar data from `data.csv` and train an LSTM neural network.


## Data

The dataset `data.csv` has three columns and multiple rows. Each row
corresponds to a single event in some hypothetical HEP experiment observing
elementary particles.

In each event, there may be multiple particles produced. The number of
particles produced in each event can vary. For each particle we have an
estimate of its energy and direction w.r.t. the beam axis.

The first column 'event_id' of the dataset is a unique integer index associated
to each event. The second column 'particle_energy' is a variable length array
of estimated particle energies. The third column `particle_dir_z` is a vl array
of direction w.r.t. the beam axis for each particle.

The variable length arrays are stored in the 'csv' as strings of the form
"[x0,x1,x2,...]".


## Examples

1. `01_example_data_frame.py` -- A simple example on how to load and parse
    data from the `data.csv` file using vlndata frame.

2. `02_example_dataset.py` -- Another example on how to construct a
   pytorch-like dataset from the selected columns of `data.csv`.
   It also demonstrates how to embed additional columns to the `data.csv`
   file on the fly.

3. `03_example_rnn_train.py` -- Final example that demonstrates how to
   construct a `keras` Sequence using vlndata primitives and train a simple
   LSTM network to predict average particle energy for each event.

