import numpy as np


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
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.axes = getattr(obj, 'axes', None)

    def __repr__(self):
        return str(self) + '\n' + '{}'.format(self.__dict__)

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
        if hasattr(self, 'axes'):
            _ax = ''.join([self.axes[i] for i in tuple(*axes)])
            return AxesArray(transposed, axes=_ax)
        return AxesArray(transposed)

    def transpose_axes(self, new_axes):
        """ If this array has an axes attribute, rearrange it and return """
        if not hasattr(self, 'axes'):
            raise AttributeError('Cannot transpose_axes because AxesArray '
                                 'does not have axes parameter')
        validate_newaxes(self.axes, new_axes)
        order = tuple(self.axes.find(a) for a in new_axes)
        return self.transpose(order)
