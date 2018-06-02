import os
import re
import numpy as np
import logging
import itertools as it
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from collections import defaultdict, OrderedDict
from tifffile import imread, TiffFile

logger = logging.getLogger(__name__)


def make_iterable(idx):
    ''' Simple function to ensure that an argument is iterable'''
    try:
        iter(idx)
    except TypeError:
        if idx is None:
            return []
        else:
            idx = [idx]
    return idx


def mode1(x):
    """Returns the mode value of a one-dimensional array"""
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m]


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
            t = ('\\' + t) if t is not None else '[a-zA-Z0-9]'
            n = matchobj.group('num')
            n = '{' + n + '}' if n is not None else '+'
            return r'(?P<' + label + '>' + t + n + ')'
        return func
    ss = re.sub(r'{(?P<typ>[dD])?(?P<num>\d+)?}', wrap(label), easyreg)
    assert ss != easyreg, 'failed to parse pattern!: {}'.format(easyreg)
    return '(?=.*{})?'.format(ss)


class TiffFolder(object):
    """Sequence of TIFF files that can be indexed like a numpy array

    The image data in all files must match shape, dtype, etc.  Currently
    returns all data in TCZYX order.

    Args:
        path (str): path to folder with files
        patterns (list of tuples): formattable pattern that identifies timepoints
        ext (str): file extension to match

    Examples:
        >>> L = TiffFolder('/path/to/files')
        >>> L[1:3, 1:100:10, :, 4].shape
        (2, 10, `nz`, 1, `nx`)
    """
    _valid_axes = 'stczyx'

    def __init__(self, path, patterns=None, ext='.tif'):
        self.path = os.path.abspath(path)
        self.files = sorted([os.path.join(self.path, f)
                             for f in os.listdir(self.path)
                             if f.lower().endswith(ext.lower())])
        if not self.files:
            raise self.EmptyError('No %s files to parse in %s' % (ext, path))

        _patrns = patterns or getattr(self, 'patterns', None)
        if _patrns:
            if isinstance(_patrns, (tuple, list)):
                if not all([isinstance(p, tuple) and
                            len(p) == 2 and
                            all([isinstance(i, str) for i in p])
                            for p in _patrns]):
                    raise TypeError('patterns must be 2-tuples of strings')
            elif isinstance(_patrns, (dict)):
                if not all([isinstance(_patrns[k], str) for k in _patrns]):
                    raise TypeError('patterns must all be strings')
                _patrns = _patrns.items()
            else:
                raise TypeError('patterns argument must either be a dict, tuple, or list')
        else:
            raise ValueError('No filename search patterns provided')

        self._parser = re.compile("".join([build_regex(*p) for p in _patrns]))
        self.patterns = dict(_patrns)
        self._parse()

    class ParseError(Exception):
        pass

    class EmptyError(Exception):
        pass

    @property
    def shape(self):
        """ The shape of the data in the folder (in order of self.axes) """
        return tuple(i for i in self._shapedict.values() if i > 1)

    @property
    def axes(self):
        """ The order of axes returned by self.shape """
        return ''.join((k for k, v in self._shapedict.items() if v > 1))

    @property
    def ndim(self):
        """ Number of dimensions in the dataset """
        return len(self.shape)

    def select_filenames(self, **kwargs):
        """ Get a list of filenames matching axis selection criteria

        Args:
            **s (int, slice, iterable): subset of filenames along s axis
            **t (int, slice, iterable): subset of filenames along t axis
            **c (int, slice, iterable): subset of filenames along c axis
            **z (int, slice, iterable): subset of filenames along z axis
            **y (int, slice, iterable): subset of filenames along y axis
            **x (int, slice, iterable): subset of filenames along x axis

        Returns:
            list: list of string filepaths
        """
        axes_selections = self._get_axes_selections(**kwargs)
        stc_combos = it.product(*list(axes_selections.values())[:3])
        return sorted(self._stcz_farray[i][0].decode('utf-8') for i in stc_combos)

    def asarray(self, maxworkers=None, **kwargs):
        """ Read TIFF data as numpy array

        Args:
            maxworkers (int): Max number of threads to use. If None,
                will use half the availble of cpu cores.
            **s (int, slice, iterable): subset of data to retrieve in s axis
            **t (int, slice, iterable): subset of data to retrieve in t axis
            **c (int, slice, iterable): subset of data to retrieve in c axis
            **z (int, slice, iterable): subset of data to retrieve in z axis
            **y (int, slice, iterable): subset of data to retrieve in y axis
            **x (int, slice, iterable): subset of data to retrieve in x axis

        Returns:
            numpy array of image data

        Raises:
            NotImplementedError: if the number of timepoints, channels,
                positions, zplanes... is different across the dataset
        """
        if not self._symmetrical:
            raise NotImplementedError('Cannot currently handle data with '
                                      'different set of timepoints per channel')

        axes_selections = self._get_axes_selections(**kwargs)
        # find ultimate stack shape
        _sizes = [len(v) for v in axes_selections.values()]
        # don't crop XY until later
        _sizes[-2:] = list(self._shapedict.values())[-2:]

        stc_combos = it.product(*list(axes_selections.values())[:3])
        stacks = np.empty(tuple(_sizes))

        def reader(stc_index):
            stacks[stc_index] = imread(self._stcz_farray[stc_index],
                                       key=axes_selections['z'])

        # actually read the files, in parallel if requested
        maxworkers = maxworkers if maxworkers is not None else cpu_count() // 2
        if maxworkers < 2:
            for i in stc_combos:
                reader(i)
        else:
            with ThreadPoolExecutor(maxworkers) as executor:
                executor.map(reader, stc_combos)

        if len(axes_selections['y']) != self._shapedict['y']:
            stacks = stacks[:, :, :, :, axes_selections['y']]
        if len(axes_selections['x']) != self._shapedict['x']:
            stacks = stacks[:, :, :, :, :, axes_selections['x']]

        return np.squeeze(stacks)

    def __len__(self):
        return self.shape[0]

    def __missing__(self, key):
        raise IndexError("The index (%d) is out of range." % key)

    def __getitem__(self, key):
        axes_selections = OrderedDict.fromkeys(self._valid_axes)

        def handle_idx(idx, axis):
            length = self._shapedict[axis]
            if isinstance(idx, slice):
                return range(*idx.indices(length))
            if isinstance(idx, int):
                if idx < 0:  # Handle negative indices
                    idx += length
                if idx < 0 or idx >= length:
                    return self.__missing__(idx)
                return idx

        if isinstance(key, tuple):
            keys = tuple(handle_idx(*i) for i in zip(key, self.axes[:len(key)]))
            axes_selections.update(dict(zip(self.axes[:len(key)], keys)))
        elif isinstance(key, (slice, int)):
            # Get the start, stop, and step from the slice
            axes_selections[self.axes[0]] = handle_idx(key, self.axes[0])
        else:
            raise TypeError("Invalid argument type.")

        return self.asarray(**axes_selections)

    def _get_axes_selections(self, **kwargs):
        axes_selections = OrderedDict.fromkeys(self._valid_axes)
        for axis in axes_selections:
            if kwargs.get(axis, None) is not None:
                axes_selections[axis] = make_iterable(kwargs.get(axis))
            else:
                axes_selections[axis] = range(self._shapedict[axis])
        return axes_selections

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

        self.channelinfo = defaultdict(lambda: defaultdict(list))
        for info in lod:
            for k, v in info.items():
                if not k == 'c':
                    self.channelinfo[info['c']][k].append(v)

        _SD = OrderedDict(zip('stcz', (None,) * 4))
        _SD['c'] = len(self.channelinfo)
        self._axes_sets = OrderedDict(zip('stcz', (None,) * 4))
        self._axes_sets['c'] = sorted(self.channelinfo.keys())
        self._symmetrical = True
        for i in 'stz':
            i_set = {frozenset(v[i]) for v in self.channelinfo.values()}
            if len(i_set) > 1:
                self._symmetrical = False
            self._axes_sets[i] = sorted(i_set.pop()) or None
            _SD[i] = len(self._axes_sets[i]) if self._axes_sets[i] else 1

        string_padding = 10
        self._stcz_farray = np.empty(tuple(_SD.values()),
                                     dtype='S{}'.format(len(self.files[0])
                                                        + string_padding))
        for f in self.files:
            info = self._parse_filename(f)
            idx = [0, 0, 0, 0]
            for i, (k, v) in enumerate(self._axes_sets.items()):
                if v is not None:
                    idx[i] = v.index(info.get(k))
            self._stcz_farray[tuple(idx)] = f

        with TiffFile(str(self.files[0])) as first_tiff:
            if _SD['z'] == 1:
                _SD['z'] = len(first_tiff.pages)
            _SD['y'] = first_tiff.pages[0].imagelength
            _SD['x'] = first_tiff.pages[0].imagewidth
            self.dtype = first_tiff.pages[0].dtype

        self._shapedict = OrderedDict(zip(self._valid_axes, tuple(_SD.values())))


class LLSFolder(TiffFolder):
    """ Example class for handling lattice light sheet data """
    patterns = [
        ('rel', '_{d7}msec'),
        ('abs', '_{d10}msecAbs'),
        ('w', '_{d3}nm'),
        ('t', '_stack{d4}'),
        ('c', '_ch{d1}'),
        ('cam', 'Cam{D1}')
    ]

    def _parse(self):
        super(LLSFolder, self)._parse()
        for chan, subdict in self.channelinfo.items():
            for k, v in subdict.items():
                if k in ('cam', 'w'):
                    # assume the same value across all timepoints
                    subdict[k] = v[0]
            if self.shape[0] > 1:
                if 'abs' in subdict and any(subdict['abs']):
                    subdict['interval'] = mode1(np.subtract(
                        subdict['abs'][1:], subdict['abs'][:-1]))


class MetamorphFolder(TiffFolder):
    """ Example class for handling metamorph data """
    patterns = [
        ('t', '_t{d}'),
        ('c', '_w{}'),
        ('s', '_s{d}'),
    ]
