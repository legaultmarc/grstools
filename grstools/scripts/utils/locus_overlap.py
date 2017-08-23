#!/usr/bin/env python

import logging
import numpy as np

import geneparse

logger = logging.getLogger(__name__)

def locus_overlap(args):
    # Get variants from file
    variants = get_variants(args.variants)

    # Extract variants genotypes
    genotypes = get_genotypes(args.genotypes_filename,
                              args.genotypes_format,
                              args.genotypes_kwargs,
                              variants)

    # Create matrix n_sample x n_snps
    genotypes_mat = create_matrix(genotypes)
    # Compute LD matrix n_snps x n_snps
    ld_mat = compute_ld(genotypes_mat)

    # Cluster variants in LD
    get_overlaps(ld_mat, genotypes, args.ld_threshold)


def get_variants(variants_filename):
    variants = []

    with open(variants_filename, "r") as f:
        header = f.readline()
        header_to_pos = {title: pos for pos, title in enumerate(
            header.strip().split(","))}

        expected_headers = {"name", "chrom", "pos", "reference",
                            "risk", "p-value", "effect"}

        missing_headers = expected_headers - header_to_pos.keys()

        if len(missing_headers) != 0:
            raise ValueError(
                "Missing the columns {} in variants input file".format(
                    ",".join(missing_headers)
                )
            )

        for line in f:
            l = line.split(",")
            v = geneparse.Variant(
                    None,
                    l[header_to_pos["chrom"]],
                    l[header_to_pos["pos"]],
                    [l[header_to_pos["reference"]],
                        l[header_to_pos["risk"]]]
            )
            
            variants.append(v)

    return variants


def get_genotypes(genotypes_filename, genotypes_format, genotypes_kwargs,
                  variants):
    if genotypes_format not in geneparse.parsers.keys():
        raise ValueError(
            "Unknown reference format '{}'. Must be a genetest compatible "
            "format ({})."
            "".format(genotypes_format, list(geneparse.parsers.keys()))
        )

    genotypes_kwargs_info = {}
    if genotypes_kwargs:
        for argument in genotypes_kwargs.split(","):
            key, value = argument.strip().split("=")

            if value.startswith("int:"):
                value = int(value[4:])

            elif value.startswith("float:"):
                value = float(value[6:])

            genotypes_kwargs_info[key] = value

    reader = geneparse.parsers[genotypes_format]
    reader = reader(genotypes_filename, **genotypes_kwargs_info)

    extractor = geneparse.Extractor(reader,
                                    variants=variants)

    genotypes = []
    for g in extractor.iter_genotypes():
        genotypes.append(g)
    
    return genotypes


def create_matrix(genotypes):
    # Matrice n_samples x m_snps
    mat = np.vstack(tuple((i.genotypes for i in genotypes))).T

    # Normalize each column: mean(SNPi) = 0 and var(SNPi = 1)
    mat = mat - np.nanmean(mat, axis=0)[np.newaxis, :]
    mat = mat / np.nanstd(mat, axis=0)[np.newaxis, :]

    return mat


def compute_ld(mat):
    # Number samples for each pair of SNPs
    nans = np.isnan(mat)
    non_nans = 1 -  nans.astype(int)
    n = np.dot(non_nans.T, non_nans)

    # Replace nans (missing genotypes) by 0
    mat[nans] = 0

    r = np.dot(mat.T, mat) / n

    print(r**2)
    print((r**2).shape)

    return r**2


def get_overlaps(mat, genotypes, ld_threshold):
    # Sanitary check
    print(len(genotypes) == len(mat[0]))
    print(mat.shape[0] == mat.shape[1])

    # Ajouter check pour la diagonale


    # List of parents (variants in LD will have the same parent)
    parents = [None] * len(genotypes)
    parents[0] = 0

    p = 1

    # Add parent number to variants in LD
    for row_idx, row in enumerate(mat):
        for j in range(row_idx+1, mat.shape[0]):
            if row[j] >= ld_threshold:
                if (parents[row_idx] is not None) & (parents[j] is not None):
                    to_change = parents[j]
                    parents[j] = parents[row_idx]
                    
                    # Update parents
                    for k in range(len(parents)):
                        if parents[k] == to_change:
                            parents[k] = parents[row_idx]
                
                elif parents[row_idx] is not None:
                    parents[j] = parents[row_idx]

                elif parents[j] is not None:
                    parents[row_idx] = parents[j]

                else:
                    parents[row_idx] = p
                    parents[j] = p
                    p += 1


    # Add parent number to variants not in LD with any other variant
    for i in range(len(parents)):
        if parents[i] is None:
            parents[i] = p
            p += 1

    for i,g in enumerate(genotypes):
        print(i,g)
    print(parents)
    return parents
