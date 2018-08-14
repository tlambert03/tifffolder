import numpy as np
import logging
logger = logging.getLogger(__name__)

def validate_newaxes(old_axes, new_axes):
    """ Validate axes string when changing the order/transposing axes

    This function confirms that:
        1. every character in the new_axes exists in the old_axes
        2. there is the same number of characters (ndims isn't changing)
        3. all of the old_axes are represented in the new_axes
    """
    for char in new_axes:
        if char not in old_axes:
            raise ValueError('Axis "{}" not in current axes: "{}"'
                             .format(char, old_axes))
    if len(new_axes) != len(old_axes):
        raise ValueError('Cannot change axes dimensions from "{}" to "{}".'
                         .format(len(old_axes), len(new_axes)))
    if not set(new_axes) == set(old_axes):
        raise ValueError('Error: unassigned axes remaining: {}'
                         .format(set(old_axes) - set(new_axes)))


class AxesArray(np.ndarray):
    """ Subclass of np.ndarray that can have attributes added to it

    Specifically, this class is aware of an "axes" attribute, which
    is added by TiffFolder.asarray(), and will reorder axes when np.transpose
    is called or __getitem__ is called
    """

    def __new__(cls, input_array, dtype=np.uint16, **kwargs):
        obj = np.asarray(input_array, dtype=dtype).view(cls)
        for k, v in kwargs.items():
            setattr(obj, k, v)
        if len(obj.axes):
            if not len(obj.axes) == obj.ndim:
                logger.warning('axes argument must have same length as array.ndim')
                obj.axes = ''
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.axes = getattr(obj, 'axes', '')

    def __repr__(self):
        return str(self) + '\n' + '{}'.format(self.__dict__)

    def __getattr__(self, key):
        """ slice a specific axis in the AxesArray by name

        a = AxesArray(np.random.rand(4,5,6,7), axes='czyx')
        a.z[::2].shape  # (4, 3, 6, 7)
        """
        if key in self.axes:
            return AxisIndexer(self, key, axes=self.axes)
        elif key.startswith('shape_') and key[-1] in self.axes:
            return self.shape[self.axes.find(key[-1])]
        else:
            raise AttributeError("'{}' object with axes '{}' has no attribute '{}'"
                                 .format(self.__class__.__name__, self.axes, key))

    def __getitem__(self, key):
        """ preserve axes information when doing standard numpy indexing """
        item = super(AxesArray, self).__getitem__(key)
        axes = self.axes
        if isinstance(key, tuple):
            for i, k in enumerate(key):
                if isinstance(k, int):
                    axes = axes.replace(self.axes[i], '')
        elif isinstance(key, int):
            axes = axes.replace(self.axes[0], '')
        return AxesArray(item, axes=axes)

    def transpose(self, *axes):
        """ If this array has an axes attribute, rearrange it and return """
        transposed = super(AxesArray, self).transpose(*axes)
        if self.axes:
            _ax = ''.join([self.axes[i] for i in tuple(axes)])
            return AxesArray(transposed, axes=_ax)
        return AxesArray(transposed)

    def transpose_axes(self, new_axes):
        """ If this array has an axes attribute, rearrange it and return """
        if not self.axes:
            raise AttributeError('Cannot transpose_axes because AxesArray '
                                 'does not have axes parameter')
        validate_newaxes(self.axes, new_axes)
        order = tuple(self.axes.find(a) for a in new_axes)
        return self.transpose(order)

    def updated_axes(self, key):
        # key is either int, or tuple of ints
        axes = self.axes
        if isinstance(key, tuple):
            for k in key:
                if isinstance(k, int):
                    axes = axes.replace(self.axes[k], '')
        elif isinstance(key, int):
            axes = axes.replace(self.axes[key], '')
        elif key is None:
            axes = ''
        return axes

    def max(self, axis=None, *args, **kwargs):
        out = super(AxesArray, self).max(axis, *args, **kwargs)
        return AxesArray(out, axes=self.updated_axes(axis))

    def std(self, axis=None, *args, **kwargs):
        out = super(AxesArray, self).std(axis, *args, **kwargs)
        return AxesArray(out, axes=self.updated_axes(axis))

    def mean(self, axis=None, *args, **kwargs):
        out = super(AxesArray, self).mean(axis, *args, **kwargs)
        return AxesArray(out, axes=self.updated_axes(axis))

    def sum(self, axis=None, *args, **kwargs):
        out = super(AxesArray, self).sum(axis, *args, **kwargs)
        return AxesArray(out, axes=self.updated_axes(axis))

    def min(self, axis=None, *args, **kwargs):
        out = super(AxesArray, self).min(axis, *args, **kwargs)
        return AxesArray(out, axes=self.updated_axes(axis))

    def prod(self, axis=None, *args, **kwargs):
        out = super(AxesArray, self).prod(axis, *args, **kwargs)
        return AxesArray(out, axes=self.updated_axes(axis))

    def squeeze(self, axis=None):
        axes = self.axes
        if axis is None:
            for i, s in enumerate(self.shape):
                if s == 1:
                    axes = axes.replace(self.axes[i], '')
        elif isinstance(axis, int):
            if self.shape[axis] == 1:
                axes = axes.replace(self.axes[axis], '')
        elif isinstance(axis, tuple):
            for i in axis:
                axes = axes.replace(self.axes[i], '')
        out = super(AxesArray, self).squeeze(axis)
        return AxesArray(out, axes=axes)


class AxisIndexer(AxesArray):
    """ subclass of AxesArray that is only intended to be used in the
    AxesArray.__getattr__ method.  allows syntax like this:

    a = AxesArray(np.random.rand(4,5,6,7), axes='czyx')
    a.shape  # (4, 5, 6, 7)
    first_channel = a.c[0]
    first_channel.shape  # (5, 6, 7)
    every_other_z = a.z[::2]
    every_other_z.shape  # (4, 3, 6, 7)
    """
    def __new__(cls, parent, axis, **kwargs):
        obj = super(AxisIndexer, cls).__new__(cls, parent, **kwargs)
        obj.axis = axis
        return obj

    def __getitem__(self, key):
        if not isinstance(key, (int, slice)):
            raise IndexError('{}-axis getter can only slice a single axis'
                             .format(self.axis))
        slc = [np.s_[:]] * self.ndim
        slc[self.axes.find(self.axis)] = key
        return super(AxisIndexer, self).__getitem__(tuple(slc))

    def __len__(self):
        return super(AxisIndexer, self).shape[self.axes.find(self.axis)]
