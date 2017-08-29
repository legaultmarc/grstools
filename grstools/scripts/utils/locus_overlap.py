#!/usr/bin/env python

import logging
import numpy as np

import geneparse

logger = logging.getLogger(__name__)


class SelectedGroup(object):
    __slots__ = ("name", "variants", "references", "locus")

    def __init__(self, name, variants):
        self.name = name
        self.variants = variants
        self.references = [None]*len(variants)
        self.locus = [None]*len(variants)


def locus_overlap(args):
    selected_groups = init_selected_groups(args.variants)

    unique_variants = get_unique_variants(selected_groups)

    # Extract unique variants genotypes
    genotypes = get_genotypes(args.genotypes_filename,
                              args.genotypes_format,
                              args.genotypes_kwargs,
                              unique_variants)

    # Create matrix n_sample x n_snps
    genotypes_mat = create_matrix(genotypes)

    # Compute LD matrix n_snps x n_snps
    ld_mat = compute_ld(genotypes_mat)

    # Cluster variants in LD
    parents = get_overlaps(ld_mat, genotypes, args.ld_threshold)

    # Get locus and reference
    variant_to_locus, locus_to_reference = get_locus(parents, genotypes)

    fill_references_and_locus(selected_groups,
                              variant_to_locus,
                              locus_to_reference)

    # Print results to file
    print_results(selected_groups)


def init_selected_groups(filenames):
    selected_groups = []
    for i in filenames.split(","):
        name, filename = i.strip().split("=")
        variants = []

        with open(filename, "r") as f:
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

        selected_groups.append(SelectedGroup(name, variants))

    return selected_groups


def get_unique_variants(selected_groups):
    unique_variants = set()
    for sg in selected_groups:
        unique_variants = unique_variants | set(sg.variants)

    return list(unique_variants)


def get_genotypes(genotypes_filename, genotypes_format, genotypes_kwargs,
                  unqiue_variants):
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
                                    variants=unqiue_variants)

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
    non_nans = 1 - nans.astype(int)
    n = np.dot(non_nans.T, non_nans)

    # Replace nans (missing genotypes) by 0
    mat[nans] = 0

    r = np.dot(mat.T, mat) / n

    return r**2


def get_overlaps(mat, genotypes, ld_threshold):
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

    return parents


def get_locus(parents, genotypes):
    # Key: Variant, value: parent number
    variant_to_locus = {}

    # Key: parent number Value: variant chosen as the reference
    locus_to_reference = {}

    for idx, g in enumerate(genotypes):
        variant_to_locus[g.variant] = parents[idx]
        if parents[idx] not in locus_to_reference:
            locus_to_reference[parents[idx]] = g.variant

    return variant_to_locus, locus_to_reference


def fill_references_and_locus(selected_groups,
                              variant_to_locus,
                              locus_to_reference):
    for sg in selected_groups:
        for idx, variant in enumerate(sg.variants):
            if variant in variant_to_locus:
                locus = variant_to_locus[variant]
                sg.references[idx] = locus_to_reference[locus]
                sg.locus[idx] = locus
            else:
                logger.warning("No genotype found for {}".format(variant))


def print_results(selected_groups):
    for sg in selected_groups:
        output = sg.name + "_ref.grs"
        with open(output, "w") as f:
            f.write("name,chrom,pos,A1,A2,locus\n")
            for idx, v in enumerate(sg.variants):
                l = str(v.name)+","+str(v.chrom)+","+str(v.pos)+","
                l += v.alleles[0]+","+v.alleles[1]+","
                l += str(sg.locus[idx])+"\n"

                f.write(l)
