"""setup script"""
from setuptools import setup


if __name__ == "__main__":

    setup(

        name='tqwt_tools',

        version='0.0.1',

        description='Computation of a Tunable-Q Wavelet Transform and a resonance-based signal decomposition',

        author='Jonas Hajek-Lellmann',

        author_email='jonas.hajek-lellmann@gmx.de',

        python_requires='>=3.7',

        classifiers=[
            "Programming Language :: Python :: 3.7",
            "Development Status :: 3 - Alpha",
            'Intended Audience :: Telecommunications Industry',
            'Intended Audience :: Science/Research',
            'Indended Audience :: Developers'
            'Environment :: Console',
            'Topic :: Scientific/Engineering :: Information Analysis',
        ],

        install_requires=[
            'scipy',
            'numpy',
            'sortedcontainers',
            'typeguard',
        ],
    )