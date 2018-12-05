#!/usr/bin/env python
# -*- coding: utf-8 -*-

# How to build source distribution
# python setup.py sdist --format bztar
# python setup.py sdist --format gztar
# python setup.py sdist --format zip

import os

from setuptools import setup, find_packages


MAJOR = 0
MINOR = 4
MICRO = 0
VERSION = "{}.{}.{}".format(MAJOR, MINOR, MICRO)


def write_version_file(fn=None):
    if fn is None:
        fn = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join("grstools", "version.py"),
        )

    content = (
        "\n# THIS FILE WAS GENERATED AUTOMATICALLY BY GRSTOOLS SETUP.PY\n"
        'grstools_version = "{version}"\n'
    )

    a = open(fn, "w")
    try:
        a.write(content.format(version=VERSION))
    finally:
        a.close()


def setup_package():
    # Saving the version into a file
    write_version_file()

    setup(
        name="grstools",
        version=VERSION,
        description="Tools to manipulate genetic risk scores.",
        long_description="",
        author=u"Marc-André Legault",
        author_email="legaultmarc@gmail.com",
        url="https://github.com/legaultmarc/grstools",
        license="MIT",
        packages=find_packages(exclude=["tests", ]),
        package_data={"geneparse.tests": ["data/*"]},
        test_suite="grstools.tests.test_suite",
        entry_points={
            "console_scripts": [
                "grs-evaluate=grstools.scripts.evaluate:main",
                "grs-compute=grstools.scripts.build_grs:main",
                "grs-utils=grstools.scripts.utils:main",
                "grs-create=grstools.scripts.choose_snps:main",
                "grs-mr=grstools.scripts.mendelian_randomization:main",
            ],
        },
        install_requires=["geneparse >= 0.1.0", "genetest >= 0.1.0",
                          "matplotlib >= 2.0", "scipy >= 0.18"],
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
        zip_safe=False,
    )


if __name__ == "__main__":
    setup_package()
