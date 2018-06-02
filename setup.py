from setuptools import setup
import io
import re
from collections import OrderedDict

with io.open('README.md', 'rt', encoding='utf8') as f:
    readme = f.read()

with io.open('tifffolder/__init__.py', 'rt', encoding='utf8') as f:
    version = re.search(r'__version__ = \'(.*?)\'', f.read()).group(1)


setup(
    name='tifffolder',
    version=version,
    url='https://github.com/tlambert03/tifffolder',
    project_urls=OrderedDict((
        ('Code', 'https://github.com/tlambert03/tifffolder'),
        ('Issue tracker', 'https://github.com/tlambert03/tifffolder/issues'),
    )),
    description='Easily parse/access a subset of data from a <=6D folder of TIFFs',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Talley Lambert',
    author_email='talley.lambert@gmail.com',
    license='MIT',
    keywords=['image', 'analysis', 'tiff'],
    packages=['tifffolder'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    python_requires='>=2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*',
    install_requires=[
        'numpy',
        'tifffile',
    ],
)
