# vlndata -- Data Primitives to Work with Columnar and Variable Length Data

This package provides pytorch-like Datasets and Data Loaders that simplify
working with columnar data formats and support handling variable length array
data.


## Overview

`vlndata` provides boilerplate functions to construct Datasets for the Deep
Learning applications from the columnar data. Additionally, it provides
routines to handle variable length arrays, which is a common data object
for many scientific application.

`vlndata` is a very lightweight package without intrusive dependencies
(currently the only dependencies are `numpy`, `padnas`, `h5py`).

`vlndata` currently can work with `CSV` and `HDF5` data formats, but the
support for other formats can be trivially implemented.


## Documentation

### 1. Example Usage

In this section, we will give a brief walk-through on how to use the basic
`vlndata` primitives to load and parse data from the
[example file](./examples/01_particle_csv/data.csv), and construct a
pytorch-like data loader. Please refer to the
[example README](./examples/01_particle_csv/README.md) file for further
details.

1. Loading data. The sample file is a csv file and can be loaded with a CSV
   Frame:

```python
from vlndata.data_frame import CSVFrame

df = CSVFrame(path = './examples/01_particle_csv/data.csv')
```

2. One can loop through the rows of the file and observe individual values:

```python
for idx in range(len(df)): # Loop through each row
    # extract 'event_id' value for event `idx`
    # 'event_id' -- is a unique numerical identifier of each event
    # in `data.csv`
    event_id  = df.get_scalar(column = 'event_id', index = idx)

    # extract 'particle_energy' values for event `idx`
    # 'particle_energy' -- in an array of particle energies in event `idx`
    particles = df.get_vlarr(column = 'particle_energy', index = idx)
```

3. If needed, the data frame can be separated into train/test parts:

```python
from vlndata.data_frame import train_test_split

df_train, df_val, df_test = train_test_split(
    df, val_size = 0.2, test_size = 0.2
)
```

4. Also, at this stage, new columns can be embedded into the data frame.
   For example, the snippet below will embed a new column into the data frame
   that will indicate if `event_id` is event or odd:

```python
from vlndata.data_frame import VarFrame

df = VarFrame(
    df, { 'event_id_odd' : lambda frame : (frame['event_id'] % 2) == 1 }
)
```

5. One can select the required columns from the DataFrame and construct a
   pytorch-like Dataset:

```python
from vlndata.dataset import VLDataset

dset = VLDataset(
    df,
    scalar_groups = { 'group_1' : [ 'event_id', ] },
    vlarr_groups  = { 'group_2' : [ 'particle_energy', 'particle_dir_z' ] },
)
```
(here 'group_1' and 'group_2' are just labels and can be arbitrary).

The resulting dataset will have the same length as the original DataFrame.
When `dset[index]` is called, the dataset will return a single dictionary of
the form:

```python
{
    'group_1' : np.array([ df.get_scalar('event_id', index), ])
    'group_2' : np.array([
        df.get_vlarr('particle_energy', index),
        df.get_vlarr('particle_dir_z', index)
    ]).T,
}
```

6. For training of the Deep Learning models one needs to create an object that
   combines data from multiple rows into a single batch.
   The `vlndata` package provides a `DataLoader` object to achieve that purpose

```python
from vlndata.data_loader import DataLoader

dl = DataLoader(dset, batch_size = 2)
```

  The vanilla `pytorch` `DataLoader` can also be used, but one needs to supply
  a collate function to its constructor that will batchify variable length
  array data:

```python
import torch
from vlndata.data_loader import vldata_dict_collate

dl = torch.utils.data.DataLoader(
    dset, batch_size = 2, collate_fn = vldata_dict_collate
)
```


### 2. Further documentation

Please refer to the docstrings and the source code of the `vlndata` for further
documentation.

The additional usage examples can be found in the `examples/` subdirectory.


### 3. Adding Support for New Data Formats

To support a new data format one simply needs to create a new object of type
`DataFrameBase` that implements three parser functions:

1. `get_scalar(column : str, index : int)` -- load a single scalar value from
   the dataset, using column name specified by `column` parameter, and the
   row number specified by `index` parameter.

2. `get_scalar(column : str, index : int)` -- load a variable length array
   value from the dataset.

3. `__getitem__(column : str)` -- return an array of raw values for the entire
   `column`.

Once such an object is implemented, it can be transparently used with other
parts of the `vlndata` package.


## Structure

The `vlndata` package has three main parts:

1. `data_frame` -- provides basic primitives to parse various data formats.
   The `vlndata` data frames loosely correspond to `pandas.DataFrame`, but
   support efficient handling of variable length arrays.

   Apart from parsing various data formats, this sub-package also provides
   wrappers to shuffle/slice/modify data frames.

2. `dataset` -- implements pytorch-like Dataset that can be constructed
   from select few columns of a `DataFrame`.

   Additionally, this sub-package provides wrappers to transform the dataset
   values on fly (e.g. shuffle variable length values, add noise, etc).

3. `data_loader` -- implements a simple pytorch-like DataLoader.
   Unlike the `pytroch.DataLoader` this loader supports constructing batches of
   variable length arrays.

