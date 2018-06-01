import os
import re
import numpy as np
from fnmatch import fnmatch
from collections import defaultdict
from tifffile import imread, TiffFile
import logging
logger = logging.getLogger(__name__)


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


def build_regex(label, easyreg):
    """ Convert a string with simplified syntax into lookahead regex.

    This allows one to search for the `easyreg` anywhere in the string and
    return a dict with matches.  brackets `{}` in the easy reg pattern
    are placeholders for digits (e.g. {d2}) or non digits ({D4}) and the
    number in the brackets determines how many characters/digits to match.
    '_stack{d4}' will convert to '(?=.*_stack(?P<t>\\d{4}))?'

    Args:
        label (str): label that will be associated when matching pattern
        easyreg (str): simplified regex pattern to search for

    Returns:
        regex (str): string that can be used to compile a regex

    Examples:
        This is intended to be used with a list of (label, easyreg) tuples,
        and a parser can be compiled as follows:
        >>> patterns = [('t','_stack{d4}'), ('c','_ch{d1}'), ('w','_{d3}nm')]
        >>> parser = re.compile("".join([build_regex(*p) for p in patterns]))
        >>> parser.match('cell2_CamA_ch3_stack0023_488nm.tif').groupdict()
        {'t': '0023', 'c': '3', 'w': '488'}

    """
    def wrap(label):
        def func(matchobj):
            t = matchobj.group('typ')
            n = matchobj.group('num')
            return r'(?P<' + label + '>\\' + t + '{' + n + '})'
        return func
    ss = re.sub(r'{(?P<typ>[dD])(?P<num>\d{1,3})}', wrap(label), easyreg)
    assert ss != easyreg, 'failed to parse pattern!: {}'.format(easyreg)
    return '(?=.*{})?'.format(ss)


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
    def converter(matchobj):
        """ converts a regex of the form {d4} to the form {:04d} """
        if matchobj.groups()[0] == 'd':
            return '{:0' + matchobj.groups()[1] + 'd}'
        else:
            return '{}'
    _pattern = re.sub(r'{([dD])(\d{1,3})}', converter, pattern)
    idx = make_iterable(idx)
    return [f for f in filelist if
            any(_pattern.format(i) in f for i in idx) is not exclude]


class TiffFolder(object):
    """Sequence of TIFF files that can be indexed like a numpy array

    The image data in all files must match shape, dtype, etc.  Currently
    returns all data in TCZYX order.

    Args:
        path (str): path to folder with files
        patterns (list of tuples): formattable pattern that identifies timepoints
        cpattern (str): formattable pattern that identifies channels
        ext (str): fnmatch expression to filter files

    Examples:
        >>> L = TiffFolder('/path/to/files')
        >>> L[1:3, 1:100:10, :, 4].shape
        (2, 10, `nz`, 1, `nx`)
    """
    _valid_axes = 'ptczyx'

    def __init__(self, path, patterns=None, ext='*.tif', axes='tczyx'):
        self.path = os.path.abspath(path)
        self.files = sorted([os.path.join(self.path, f)
                             for f in os.listdir(self.path)
                             if fnmatch(f, ext)])
        if not self.files:
            raise self.EmptyError('No %s files to parse in %s' % (ext, path))
        self.patterns = patterns or getattr(self, 'patterns', None)
        for i in axes.lower():
            if i not in self._valid_axes.lower():
                raise ValueError('Unrecognized axis: %s' % i)
        self.axes = axes.lower()
        if self.patterns:
            if not isinstance(self.patterns, (tuple, list)):
                raise TypeError('patterns argument must be a tuple or list')
            if not all([isinstance(p, tuple) and
                        len(p) == 2 and
                        all([isinstance(i, str) for i in p])
                        for p in self.patterns]):
                raise TypeError('patterns must be 2-tuples of strings')
        else:
            raise ValueError('No filename search patterns provided')
        self._parser = re.compile("".join([build_regex(*p)
                                  for p in self.patterns]))
        self.patterns = dict(self.patterns)
        self.decimated = False
        self._parse()

    def _filter_axis(self, axis, idx, files=None):
        files = files or self.files
        return filter_flist(files, idx, self.patterns.get(axis))

    def getData(self, t=None, c=None, z=None, x=None, y=None):
        """ Actually open the files in self.files.

        Args:
            t,c,z,x,y: int, slice, or sequence of indices to select data

        """
        if self.decimated:
            raise NotImplementedError('Cannot currently handle data with '
                                      'different number of timepoints per channel')
        files = self.files
        nt, nc, nz, ny, nx = self.shape
        if t is not None:
            files = self._filter_axis('t', t, files)
            nt = len(make_iterable(t))
        if c is not None:
            files = self._filter_axis('c', c, files)
            nc = len(make_iterable(c))
        if z is not None:
            nz = len(make_iterable(z))
        stacks = [imread(f, key=z) for f in files]
        stacks = np.stack(stacks).reshape((nt, nc, nz, ny, nx))

        if y is not None:
            ny = len(make_iterable(y))
            stacks = stacks[:, :, :, y, :].reshape((nt, nc, nz, ny, nx))
        if x is not None:
            nx = len(make_iterable(x))
            stacks = stacks[:, :, :, :, x].reshape((nt, nc, nz, ny, nx))
        return np.squeeze(stacks)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        # FIXME: ndims hardcoded
        t, c, z, y, x = (None,) * 5

        def handle_idx(idx, length):
            if isinstance(idx, slice):
                return slice_to_iterable(idx, length)
            if isinstance(idx, int):
                if idx < 0:  # Handle negative indices
                    idx += length
                if idx < 0 or idx >= length:
                    raise IndexError("The index (%d) is out of range." % idx)
                return idx

        if isinstance(key, tuple):
            keys = tuple(handle_idx(*i) for i in zip(key, self.shape[:len(key)]))
            t, c, z, y, x = keys + (None,) * (5 - len(key))
        elif isinstance(key, (slice, int)):
            # Get the start, stop, and step from the slice
            t = handle_idx(key, self.shape[0])
        else:
            raise TypeError("Invalid argument type.")

        return self.getData(t, c, z, x, y)

    @property
    def ndim(self):
        return len(self.shape)

    class ParseError(Exception):
        pass

    class EmptyError(Exception):
        pass

    def _parse_filename(self, filename):
        result = self._parser.match(filename)
        if result:
            D = result.groupdict()
            for k, v in D.items():
                if v is not None:
                    D[k] = int(v) if v.isdigit() else v
                else:
                    logger.info('Parser failed to match pattern: %s' % k)
            return D
        else:
            raise self.ParseError('Failed to parse filename: {}'
                                  .format(filename))

    def _parse(self):
        lod = [self._parse_filename(f) for f in self.files]
        dol = dict(zip(lod[0], zip(*[d.values() for d in lod])))
        nc = len(set(dol.get('c', ())))
        nt = len(set(dol.get('t', ())))

        self.channelinfo = defaultdict(lambda: defaultdict(list))
        for info in lod:
            for k, v in info.items():
                if not k == 'c':
                    self.channelinfo[info['c']][k].append(v)
        if not len(set([frozenset(v['t']) for v in self.channelinfo.values()])) == 1:
            self.decimated = True

        with TiffFile(str(self.files[0])) as first_tiff:
            nz = len(first_tiff.pages)
            ny = first_tiff.pages[0].imagelength
            nx = first_tiff.pages[0].imagewidth
            self.dtype = first_tiff.pages[0].dtype

        self.shape = (nt, nc, nz, ny, nx)


def mode1(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m]


class LLSFolder(TiffFolder):
    patterns = [
        ('rel', '_{d7}msec'),
        ('abs', '_{d10}msecAbs'),
        ('w', '_{d3}nm'),
        ('t', '_stack{d4}'),
        ('c', '_ch{d1}'),
        ('cam', 'Cam{D1}')
    ]

    def _parse(self):
        super()._parse()
        for chan, subdict in self.channelinfo.items():
            for k, v in subdict.items():
                if k in ('cam', 'w'):
                    # assume the same value across all timepoints
                    subdict[k] = v[0]
            if self.shape[0] > 1:
                if 'abs' in subdict and any(subdict['abs']):
                    subdict['interval'] = mode1(np.subtract(
                        subdict['abs'][1:], subdict['abs'][:-1]))

