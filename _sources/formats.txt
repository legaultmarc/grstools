File formats
=============

When possible, we try to use standard file formats for `grstools`. For
instance, genotype data from various well-known file formats like VCF or plink
are understood. Because we are not aware of standards for the description
of genetic instruments, we define some expected file formats.

The GRS format
^^^^^^^^^^^^^^^

The GRS format is used to describe genetic variants with a known effect size.
In `grstools` it is used for two things:

- To select variants from  association summary statistics from large
  meta-analyses.
- To compute the GRS based on genotype data.

The GRS format is a **comma-delimited** file containing the following columns:

- ``name`` (mandatory unique identifier)
- ``chrom`` (e.g. "1", "22", "X", "MT")
- ``pos`` (the position e.g. 68184194)
- ``reference`` (the allele that is used as the reference in the effect
  estimate)
- ``risk`` (the risk, or effect allele)
- ``p-value`` (mandatory for the selection of variants but optional for the
  computation of the GRS. This is the association p-value)
- ``effect`` (a weighting term used in the computation of the GRS. Concretely,
  this is likely to be an estimated :math:`\beta` or odds ratio).

Optional columns are also useful in some cases:

- ``maf`` The minor allele frequency can be used in the selection of variants
  or to optimize LD computations by removing rare variants. It can also be used
  to distinguish ambiguous allele combinations (i.e. A/T or G/C).
