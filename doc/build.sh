#!/bin/bash

# Create the script usage includes.
mkdir -p 'source/includes'

grs-create --help > 'source/includes/grs_create_help.txt'
grs-compute --help > 'source/includes/grs_compute_help.txt'
grs-utils --help > 'source/includes/grs_utils_help.txt'
grs-mr --help > 'source/includes/grs_mr_help.txt'
grs-evaluate --help > 'source/includes/grs_evaluate_help.txt'

# Run the regular sphinx-build
sphinx-build $@
