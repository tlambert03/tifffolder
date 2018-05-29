# tifffolder
Read a subset of data from a folder of images like a numpy array

Example:
```python
>>> from tifffolder import TiffFolder
>>> tf = TiffFolder('/folder/of/files', tpattern='_stack{:04d}_', cpattern='_ch{}_')

# extract data with numpy slicing with axes [t,c,z,y,x]
# for instance get timepoints 1-100, stepping by 10,
# in the first channel, last 10 z planes, cropping somewhere in the middle of y

>>> data = tf[0:100:10, 0, -10:, 200:400, :]
>>> data.shape
(10, 1, 10, 200, 512)   # (nt, nc, nz, ny, nx
    )
```

Can also be used as a generator for lazily reading data