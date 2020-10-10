import os
import re
import numpy as np
import logging
import warnings
import itertools as it
import tifffile
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from collections import defaultdict, OrderedDict
from .axesarray import AxesArray, validate_newaxes

logger = logging.getLogger(__name__)


def imshow(data, *args, **kwargs):
    """ tifffile.imshow with my preferred settings """
    try:
        import matplotlib.pyplot as plt

        if "photometric" not in kwargs:
            kwargs["photometric"] = "MINISBLACK"
        if "cmap" not in kwargs:
            kwargs["cmap"] = "gray"
        if "vmin" not in kwargs:
            kwargs["vmin"] = data.min()
        if "vmax" not in kwargs:
            kwargs["vmax"] = data.max()
        tifffile.imshow(data, *args, **kwargs)
        plt.show()
    except ImportError:
        print("could not import matplotlib, cannot show image")


def imread(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tifffile.imread(*args, **kwargs)


class TiffFile(tifffile.TiffFile):
    def __init__(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            super(TiffFile, self).__init__(*args, **kwargs)


def make_iterable(idx):
    """ Simple function to ensure that an argument is iterable"""
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
    """Convert a string with simplified syntax into lookahead regex.

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
            t = matchobj.group("typ")
            t = ("\\" + t) if t is not None else "[a-zA-Z0-9]"
            n = matchobj.group("num")
            n = "{" + n + "}" if n is not None else "+"
            return r"(?P<" + label + ">" + t + n + ")"

        return func

    ss = re.sub(r"{(?P<typ>[dD])?(?P<num>\d+)?}", wrap(label), easyreg)
    assert ss != easyreg, "failed to parse pattern!: {}".format(easyreg)
    return "(?=.*{})?".format(ss)


class TiffFolder(object):
    """Sequence of TIFF files that can be indexed like a numpy array

    The image data in all files must match shape, dtype, etc.  Currently
    returns all data in TCZYX order.

    Args:
        path (str): path to folder with files
        patterns (list of tuples): formattable pattern that identifies timepoints
        axes (str): order of axes when getting data
        ext (str): file extension to match
        maxworkers (int): number of threads to use when reading data.
            When None, will use cpu_count() // 2

    Examples:
        >>> L = TiffFolder('/path/to/files')
        >>> L[1:3, 1:100:10, :, 4].shape
        (2, 10, `nz`, 1, `nx`)
    """

    _axes = "stczyx"

    def __init__(self, path, patterns=None, axes=None, ext=".tif", maxworkers=1):
        self.path = os.path.abspath(path)
        self.files = sorted(
            [
                os.path.join(self.path, f)
                for f in os.listdir(self.path)
                if f.lower().endswith(ext.lower())
            ]
        )
        if not self.files:
            raise self.EmptyError("No %s files to parse in %s" % (ext, path))

        _patrns = patterns or getattr(self, "patterns", None)
        if _patrns:
            if isinstance(_patrns, (tuple, list)):
                if not all(
                    [
                        isinstance(p, tuple)
                        and len(p) == 2
                        and all([isinstance(i, str) for i in p])
                        for p in _patrns
                    ]
                ):
                    raise TypeError("patterns must be 2-tuples of strings")
            elif isinstance(_patrns, (dict)):
                if not all([isinstance(_patrns[k], str) for k in _patrns]):
                    raise TypeError("patterns must all be strings")
                _patrns = _patrns.items()
            else:
                raise TypeError(
                    "patterns argument must either be a dict, tuple, or list"
                )
        else:
            raise ValueError("No filename search patterns provided")
        self._parser = re.compile("".join([build_regex(*p) for p in _patrns]))
        self.patterns = {k.lower(): v for k, v in _patrns}
        self._axes = (axes or self._axes).lower()
        for ax in set(self._axes):
            if ax not in list(self.patterns.keys()) + ["x", "y", "z"]:
                logger.info(
                    "Warning: No pattern regex provided for axis {}.".format(ax)
                    + "  Ignoring."
                )
                self._axes = self._axes.replace(ax, "")
        self.maxworkers = maxworkers
        self._shapedict = None  # will hold the shape of each axis in the data
        self._parse()

    class ParseError(Exception):
        pass

    class EmptyError(Exception):
        pass

    class ShapeError(Exception):
        pass

    @property
    def shape(self):
        """ The shape of the data in the folder (in order of self.axes) """
        return tuple(i for i in self._shapedict.values() if i > 1)

    @property
    def axes(self):
        """ The order of axes returned by self.shape """
        return "".join((k for k, v in self._shapedict.items() if v > 1))

    @axes.setter
    def axes(self, new_axes):
        validate_newaxes(self.axes, new_axes)
        self._shapedict = OrderedDict([(ax, self._shapedict[ax]) for ax in new_axes])

    @property
    def ndim(self):
        """ Number of dimensions in the dataset """
        return len(self.shape)

    def list_excluded(self):
        return sorted(set(self.files) - set(self._file_array.flatten()))

    def select_filenames(self, **kwargs):
        """Get a list of filenames matching axis selection criteria

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
        axsel = self._get_axes_selections(**kwargs)
        f_idx = it.product(*(v for k, v in axsel.items() if k in self._file_array_axes))
        return sorted(self._file_array[i] for i in f_idx)

    def asarray(self, maxworkers=None, **kwargs):
        """Read TIFF data as numpy array

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
            ShapeError: if the images are not the same size and a ValueError
                occurs when adding a new image to the array.
        """
        maxworkers = maxworkers or self.maxworkers
        maxworkers = maxworkers if maxworkers is not None else cpu_count() // 2

        axes_selections = self._get_axes_selections(**kwargs)

        if not self._symmetrical:
            if len(axes_selections.get("c")) > 1:
                raise NotImplementedError(
                    "Cannot currently handle data with "
                    "different set of timepoints per channel"
                )
            else:
                # FIX ME: channelinfo is NOT an ordered dict... this will break
                tset = tuple(self.channelinfo.values())[
                    axes_selections.get("c")[0]
                ].get("t")
                axes_selections["t"] = [
                    t for t in list(axes_selections["t"]) if t in tset
                ]

        # get requested stack shape
        out_shape = [len(v) for v in axes_selections.values()]
        # don't crop X or Y until later
        out_shape[-2:] = self.shape[-2:]

        # get indices in self._file_array for all files requested
        # and the corresponding location (slice) of the output stack
        file_idxs = OrderedDict.fromkeys(self._file_array_axes)
        stack_idxs = OrderedDict.fromkeys(self.axes)
        for ax, v in axes_selections.items():
            if ax in self._file_array_axes:
                file_idxs[ax] = v
                stack_idxs[ax] = range(len(v))
            else:
                stack_idxs[ax] = [np.s_[:]]
        file_idxs = tuple(it.product(*file_idxs.values()))
        stack_idxs = tuple(it.product(*stack_idxs.values()))
        assert len(file_idxs) == len(stack_idxs), "Unexpected tifffolder error!"
        zipped_idxs = zip(stack_idxs, file_idxs)

        # allocate empty stack
        stacks = np.empty(tuple(out_shape))

        def read_file(zipped_idx):
            stack_idx, f_array_idx = zipped_idx
            fpath = self._file_array[f_array_idx]
            if self._tiff3d:
                data = imread(fpath, key=axes_selections.get("z", None), is_ome=False)
            else:
                data = imread(fpath, is_ome=False)
            try:
                stacks[stack_idx] = data
            except ValueError:
                if data.shape != stacks[stack_idx].shape:
                    try:
                        # quick way to rotate the image if XY and
                        data = data.transpose(
                            [stacks[stack_idx].shape.index(x) for x in data.shape]
                        )
                        stacks[stack_idx] = data
                    except ValueError as e:
                        print(data.shape)
                        print(stacks[stack_idx].shape)
                        raise self.ShapeError(
                            "Error adding file {} to array: {}".format(
                                os.path.basename(fpath), e
                            )
                        )

        # actually read the files, in parallel if requested
        if maxworkers < 2:
            for i in zipped_idxs:
                read_file(i)
        else:
            with ThreadPoolExecutor(maxworkers) as executor:
                executor.map(read_file, zipped_idxs)

        requested_ny = len(axes_selections.get("y", []))
        requested_nx = len(axes_selections.get("x", []))
        if "y" in axes_selections and requested_ny != self._shapedict["y"]:
            stacks = stacks[:, :, :, :, axes_selections["y"]]
        if "x" in axes_selections and requested_nx != self._shapedict["x"]:
            stacks = stacks[:, :, :, :, :, axes_selections["x"]]

        axes = "".join([k for k, v in axes_selections.items() if len(v) > 1])
        return AxesArray(np.squeeze(stacks), axes=axes)

    def __len__(self):
        return self.shape[0]

    def __missing__(self, key):
        raise IndexError("The index (%d) is out of range." % key)

    def __getitem__(self, key):
        axes_selections = OrderedDict.fromkeys(self.axes)

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
            keys = tuple(handle_idx(*i) for i in zip(key, self.axes[: len(key)]))
            axes_selections.update(dict(zip(self.axes[: len(key)], keys)))
        elif isinstance(key, (slice, int)):
            # Get the start, stop, and step from the slice
            axes_selections[self.axes[0]] = handle_idx(key, self.axes[0])
        else:
            raise TypeError("Invalid argument type.")

        return self.asarray(**axes_selections)

    def _get_axes_selections(self, **kwargs):
        axes_selections = OrderedDict.fromkeys(self.axes)
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
                    logger.info("Parser failed to match pattern: %s" % k)
            return D
        else:
            raise self.ParseError("Failed to parse filename: {}".format(filename))

    def _parse(self):

        # get the axes specified by the user, but leave out x and y
        _axes = self._axes.replace("x", "").replace("y", "")
        # SHP will hold the size of each axis other than XY
        SHP = OrderedDict.fromkeys(self._axes)

        # the channelinfo dict will contain information about all axes
        # for each channel in the dataset (in case this information
        # differs across channels in the dataset)
        self.channelinfo = defaultdict(lambda: defaultdict(list))
        _parsed = {f: self._parse_filename(f) for f in self.files}
        for infodict in _parsed.values():
            for k, v in infodict.items():
                if k == "c":
                    continue
                self.channelinfo[infodict.get("c", 0)][k].append(v)
        SHP["c"] = len(self.channelinfo)

        # axes_sets represent all of the distinct items in each axis
        # represented in the dataset.  For instance, the actual list of
        # timepoints represented in the filenames, or the list of stage
        # positions, etc...
        self._axsets = OrderedDict.fromkeys(_axes)
        self._axsets["c"] = list(self.channelinfo.keys())
        self._symmetrical = True
        for ax in _axes.replace("c", ""):
            ax_sets = {frozenset(v[ax]) for v in self.channelinfo.values()}
            if len(ax_sets) > 1:
                # there is a different number of items in this axis
                # across channels.
                self._symmetrical = False
            self._axsets[ax] = sorted(
                set().union(*ax_sets), key=lambda x: (x is None, x)
            )
            SHP[ax] = len(self._axsets[ax]) or 1

        # the file_array will hold all of the filenames seperated and arranged
        # by axis.  for ease of later retrieval
        # NOTE: X and Y are still not part of SHP yet...

        self._file_array = np.empty(
            tuple(sz for ax, sz in SHP.items() if ax not in ("x", "y") and sz > 1),
            dtype=object,
        )
        self._file_array_axes = "".join(
            ax for ax, sz in SHP.items() if ax not in ("x", "y") and sz > 1
        )

        # place each file into the file_array, according to the
        # parsed information from the filename
        for f, info in _parsed.items():
            idx = tuple(
                axset.index(info.get(ax))
                for ax, axset in self._axsets.items()
                if len(axset) > 1
            )
            if len(idx) == 1:
                idx = idx[0]
            self._file_array[idx] = f

        if self.list_excluded():
            logger.warn(
                "WARNING: {} files were excluded during parsing. ".format(
                    len(self.list_excluded())
                )
                + " Provided patterns may not match files "
                "or axes parameter may not match patterns. "
                "Use TiffFolder.list_excluded() to show excluded "
                "files."
            )

        # peek into the header of the first file in the list to get file
        # dimensions
        try:
            with TiffFile(str(self.files[0])) as first_tiff:
                self._tiff3d = False
                if len(first_tiff.pages) > 1:
                    self._tiff3d = True
                    SHP["z"] = len(first_tiff.pages)
                SHP["y"] = first_tiff.pages[0].imagelength
                SHP["x"] = first_tiff.pages[0].imagewidth
                self.dtype = first_tiff.pages[0].dtype
        except Exception as e:
            raise self.ParseError("Could not get TIFF information: {}".format(e))

        if self._tiff3d and "z" in [k.lower() for k in self.patterns.keys()]:
            raise self.ShapeError(
                "Images are 3D (Z) stacks, but z was also "
                "specified in the filepatterns. Please remove "
                '"z" from the provided filename patterns'
            )

        self._shapedict = OrderedDict(SHP)


class LLSFolder(TiffFolder):
    """ Example class for handling lattice light sheet data """

    patterns = [
        ("rel", "_{d7}msec"),
        ("abs", "_{d10}msecAbs"),
        ("w", "_{d3}nm"),
        ("t", "_stack{d4}"),
        ("c", "_ch{d1}"),
        ("cam", "Cam{D1}"),
    ]

    def _parse(self):
        super(LLSFolder, self)._parse()
        for chan, subdict in self.channelinfo.items():
            for k, v in subdict.items():
                if k in ("cam", "w"):
                    # assume the same value across all timepoints
                    subdict[k] = v[0]
            if self._shapedict.get("t") > 1:
                if "abs" in subdict and any(subdict["abs"]):
                    subdict["interval"] = mode1(
                        np.subtract(subdict["abs"][1:], subdict["abs"][:-1])
                    )


class MetamorphFolder(TiffFolder):
    """ Example class for handling metamorph data """

    patterns = [
        ("t", "_t{d}"),
        ("c", "_w{}"),
        ("s", "_s{d}"),
    ]


if __name__ == "__main__":
    folder = "/Users/talley/Dropbox (HMS)/SharedData/lls_mitosis/"
    tf = LLSFolder(folder)
    print(tf.asarray(t=0))
