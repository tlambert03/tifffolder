# tifffolder

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Lazily read a subset of data from a folder of images using numpy slicing syntax.  Includes simplified but robust file pattern matching syntax and multithreaded file reading.  Note: this is not intended to promote a folder of tiffs as a useful way to store lots of information (things like hdf5/n5/klb are preferable).  But for data that begins as a folder of tiffs, tifffolder simplifies the process of parsing that folder into data along different axes (and could be used as an intermediate step in the coversion to a better format if desired).

### Install with pip

```bash
$ pip install tifffolder
```

### Install with conda

```bash
$ conda install -c talley tifffolder
```

### Usage

```python
>>> from tifffolder import TiffFolder
>>> tf = TiffFolder('/folder/of/tiffs', patterns={'t': '_stack{d4}', 'c': '_ch{d1}'})

# get dataset shape and order of axes
>>> tf.shape
(10, 2, 65, 184, 157)  # (nt, nc, nz, ny, nx)
>>> tf.axes
'tczyx'

# reorder data  (still experimental)
>>> tf.axes = 'tzcxy'
>>> tf.shape
(10, 65, 2, 157, 184)

# data is only read from disk when explicitly indexed
# get the last 10 Z planes from every other timepoint, 
# in the first channel cropping to the middle half in Y
>>> data = tf[::2, 0, -10:, tf.shape[-2] * 1 // 4 : tf.shape[-2] * 3 // 4 ]
>>> data.shape
(5, 10, 92, 157)   # (nt, nz, ny, nx)


# Can also be used as an iterator/generator for lazily reading data
>>> for timepoint in tf:
>>>     do_something(timepoint)
     
# or just load the whole thing
>>> alldata = tf.asarray()
>>> alldata.shape == tf.shape
True

# asarray() also accepts any axis kwargs
>>> somedata = tf.asarray(t=range(1,10), c=0)

# Or just to select filenames along certain axes:
>>> tf.select_filenames(t=range(1,10,2), c=0)
['./test_ch0_stack0001_488nm.tif',
 './test_ch0_stack0003_488nm.tif',
 './test_ch0_stack0005_488nm.tif',
 './test_ch0_stack0007_488nm.tif',
 './test_ch0_stack0009_488nm.tif']
```

### Specifying filename patterns:

tifffolder converts a simplified regex syntax into relatively robust lookahead regex that will match patterns in any order in the filename or fail elegantly.

The TiffFolder class accepts a `patterns` parameter (dict or list of two-tuples).  For each (key, value) in the `patterns` dict:
* key = the axis name (e.g. `'x', 'y', 'z', 'c', 't', 's'`)
* value = the simplified regex where:
    - things in brackets `{}` will be captured
    - things outside of brackest will be required to match, but not captured
    - `{d}` means match any number of digits
    - `{D}` means match any number of NON-digits
    - `{}` means match any alphanumeric character (excluding underscore)
    - `{d2}` means match exactly two digits (for example)


For example:

```python
>>> patterns = {
    'rel': '_{d7}msec',
    'w': '_{d3}nm',
    't': '_stack{d4}',
    'c': '_ch{d1}',
    'cam': 'Cam{D1}'
}
>>> tf = TiffFolder('/folder/of/tiffs', patterns)
>>> tf._parse_filename('cell1_ch0_stack0009_488nm_0034829msec.tif')
{'rel': 34829, 'w': 488, 't': 9, 'c': 0, 'cam': None}

>>> tf._parse_filename('cell1_CamA_ch2_stack0001_560nm_0034829msec.tif')
{'rel': 34829, 'w': 560, 't': 1, 'c': 2, 'cam': 'A'}

>>> tifffolder.build_regex('cam', 'Cam{}')
'(?=.*Cam(?P<cam>[a-zA-Z0-9]+))?'

>>> tifffolder.build_regex('c', '_ch{d1}')
'(?=.*_ch(?P<c>\\d{1}))?'
```


##### todo
* check axis reordering
* check z planes across files
