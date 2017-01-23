#!/usr/bin/env python
# -*- coding: utf-8 -*-

# How to build source distribution
# python setup.py sdist --format bztar
# python setup.py sdist --format gztar
# python setup.py sdist --format zip

from setuptools import setup, find_packages


def setup_package():
    setup(
        name="grstools",
        version="0.1",
        description="Tools to manipulate genetic risk scores.",
        long_description="",
        author=u"Marc-André Legault",
        author_email="legaultmarc@gmail.com",
        url="https://github.com/legaultmarc/grstools",
        license="MIT",
        packages=find_packages(exclude=["tests", ]),
        entry_points={
            "console_scripts": [
                "grs-match-snps=grstools.scripts.match_snps:main",
                "grs-build=grstools.scripts.build_grs:main",
            ],
        },
        classifiers=["Development Status :: 4 - Beta",
                     "Intended Audience :: Developers",
                     "Intended Audience :: Science/Research",
                     "Operating System :: Unix",
                     "Operating System :: MacOS :: MacOS X",
                     "Operating System :: POSIX :: Linux",
                     "Programming Language :: Python",
                     "Programming Language :: Python :: 3",
                     "Topic :: Scientific/Engineering :: Bio-Informatics"],
        keywords="bioinformatics genomics grs genetic risk score",
        install_requires=["numpy >= 1.8.1", "pandas >= 0.15"],
        zip_safe=False,
    )


if __name__ == "__main__":
    setup_package()
