from setuptools import setup

with open('tifffolder/__init__.py') as f:
    exec(f.read())

setup(
    name='tifffolder',
    version=__version__,
    description='Lazily read/slice a folder of images like a numpy array',
    author='Talley Lambert',
    author_email='talley.lambert@gmail.com',
    url='https://github.com/tlambert03/tifffolder',
    license='MIT',
    keywords=['image', 'analysis', 'tiff'],
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
