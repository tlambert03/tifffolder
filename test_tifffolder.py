import os
from itertools import product

import numpy as np
import pytest
import tifffile

from tifffolder import LLSFolder


ADD_REL = 423345


@pytest.fixture
def lls_folder(tmp_path):
    template = "cell1_ch{c}_stack{t:04}_{w}nm_{ab:07}msec_{rel:010}msecAbs.tif"
    wave = [488, 560]
    time = range(10)
    i = 100
    im = np.random.rand(16, 16, 16)
    for w, t in product(wave, time):
        fname = template.format(
            c=wave.index(w), w=w, t=t, ab=t * i, rel=t * i + ADD_REL
        )
        tifffile.imsave(tmp_path / fname, im)
    return tmp_path


def test_a(lls_folder):

    names = LLSFolder(lls_folder).select_filenames(t=0)
    assert [os.path.basename(i) for i in names] == [
        f"cell1_ch0_stack0000_488nm_0000000msec_{ADD_REL:010}msecAbs.tif",
        f"cell1_ch1_stack0000_560nm_0000000msec_{ADD_REL:010}msecAbs.tif",
    ]


def test_b(lls_folder):

    tf = LLSFolder(lls_folder)
    assert tf.asarray().shape == (10, 2, 16, 16, 16)
    assert tf.asarray(t=0).shape == (2, 16, 16, 16)
    assert tf.asarray(t=0, c=0).shape == (16, 16, 16)
    assert tf.asarray(t=0, c=0, z=range(0, 16, 2)).shape == (8, 16, 16)
    assert tf.asarray(t=0, c=0, z=range(0, 16, 2)).shape == (8, 16, 16)

    assert tf[0].shape == (2, 16, 16, 16)
    assert tf[0, 0].shape == (16, 16, 16)
    assert tf[0, 0, :16:2].shape == (8, 16, 16)

    np.testing.assert_allclose(tf.asarray(t=0), tf[0])
    np.testing.assert_allclose(tf.asarray(t=0, c=0), tf[0, 0])
    np.testing.assert_allclose(tf.asarray(t=0, c=0, z=range(0, 16, 2)), tf[0, 0, :16:2])
