from setuptools import setup


setup(
    name='tifffolder',
    version='0.1.0',
    description='Lazily read/slice a folder of images like a numpy array',
    author='Talley Lambert',
    author_email='talley.lambert@gmail.com',
    url='https://github.com/tlambert03/tifffolder',
    download_url='https://github.com/tlambert03/tifffolder/archive/0.1.0.tar.gz',
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
