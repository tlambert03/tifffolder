import os
from re import search
from numpy import stack
from fnmatch import fnmatch
from collections import Counter
from tifffile import imread, TiffFile


def make_iterable(idx):
    ''' Simple function to ensure that an argument is iterable'''
    try:
        iter(idx)
    except TypeError:
        idx = [idx]
    return idx


def slice_to_iterable(slc, maxidx):
    ''' Convert python slice to iterable sequence

    Args:
        slc (slice): python slice object
        maxidx (int): maximum index that the iterable should return

    Returns:
        `range` iterable
    '''
    return range(*slc.indices(maxidx))


def filter_flist(filelist, idx, pattern, exclude=False):
    ''' Filter

    Args:
        filelist (:obj:`list` of :obj:`str`): filenames to filter
        idx (int, Iterable[int]): number or iterable of numbers to substitute
            into `pattern` when filtering files
        pattern (str): formattable string, into which all items of idx will be
            inserted.

    Returns:
        list: filtered list of filenames

    Examples:
        >>> flist = ['/name_ch0_stack0000.tif', '/name_ch0_stack0001.tif',
                     '/name_ch1_stack0000.tif', '/name_ch1_stack0001.tif',
                     '/name_ch2_stack0000.tif', '/name_ch2_stack0001.tif']'
        >>> filter_flist(flist, (0,2), '_ch{}_')
        ['/name_ch0_stack0000.tif', '/name_ch0_stack0001.tif',
         '/name_ch2_stack0000.tif', '/name_ch2_stack0001.tif']
    '''
    idx = make_iterable(idx)
    return [f for f in filelist if
            any(pattern.format(i) in f for i in idx) is not exclude]


class TiffFolder(object):
    """Sequence of TIFF files that can be indexed like a numpy array

    The image data in all files must match shape, dtype, etc.  Currently
    returns all data in TCZYX order.

    Args:
        path (str): path to folder with files
        tpattern (str): formattable pattern that identifies timepoints
        cpattern (str): formattable pattern that identifies channels
        ext (str): fnmatch expression to filter files

    Examples:
        >>> L = TiffFolder('/path/to/files')
        >>> L[1:3, 1:100:10, :, 4].shape
        (2, 10, `nz`, 1, `nx`)

    """

    def __init__(self, path, tpattern='_stack{:04d}_', cpattern='_ch{}_',
                 ext='*.tif'):
        self.path = os.path.abspath(path)
        self.tpattern, self.cpattern = tpattern, cpattern
        self.flist = sorted([os.path.join(self.path, f)
                             for f in os.listdir(self.path)
                             if fnmatch(f, ext)])

    def getData(self, t=None, c=None, z=None, x=None, y=None):
        """ Actually open the files in flist.

        Args:
            t,c,z,x,y: int, slice, or sequence of indices to select data

        """
        files = self.flist
        nt, nc, nz, ny, nx = self.shape
        if t is not None:
            files = filter_flist(files, t, self.tpattern)
            nt = len(make_iterable(t))
        if c is not None:
            files = filter_flist(files, c, self.cpattern)
            nc = len(make_iterable(c))
        if z is not None:
            nz = len(make_iterable(z))
        stacks = [imread(f, key=z) for f in files]
        stacks = stack(stacks).reshape((nt, nc, nz, ny, nx))

        if y is not None:
            ny = len(make_iterable(y))
            stacks = stacks[:, :, :, y, :].reshape((nt, nc, nz, ny, nx))
        if x is not None:
            nx = len(make_iterable(x))
            stacks = stacks[:, :, :, :, x].reshape((nt, nc, nz, ny, nx))
        return stacks

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        # FIXME: ndims hardcoded
        t, c, z, y, x = (None,) * 5

        def handle_idx(idx, length):
            if isinstance(idx, slice):
                return slice_to_iterable(idx, length)
            if isinstance(idx, int):
                if idx < 0 : #Handle negative indices
                    idx += length
                if idx < 0 or idx >= length :
                    raise IndexError("The index (%d) is out of range." % idx)
                return idx

        if isinstance(key, tuple):
            keys = tuple(handle_idx(*i) for i in zip(key, self.shape[:len(key)]))
            t, c, z, y, x = keys + (None,) * (5 - len(key))
        elif isinstance(key, (slice, int)) :
            #Get the start, stop, and step from the slice
            t = handle_idx(key, self.shape[0])
        else:
            raise TypeError("Invalid argument type.")

        return self.getData(t, c, z, x, y)

    @property
    def shape(self):
        if not hasattr(self, '_shape'):
            ch = [search('(?<=' + self.cpattern.format(')(\\d+)(?=')
                            + ')', f) for f in self.flist]
            cnt = Counter([i.group() for i in ch if i])
            assert len(cnt), 'No matching tiff files found'
            if not len(set(cnt.values())) == 1:
                raise ValueError('Support for different number of timepoints '
                                 'per channel is not yet implemented')
            nc = len(cnt)
            nt = list(cnt.values())[0]
            with TiffFile(str(self.flist[0])) as first_tiff:
                nz = len(first_tiff.pages)
                nx = first_tiff.pages[0].imagewidth
                ny = first_tiff.pages[0].imagelength
            self._shape = (nt, nc, nz, ny, nx)
        return self._shape
