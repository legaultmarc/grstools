Command-line utilities
=======================

.. grs-compute
.. _grs-compute:

``grs-compute``
^^^^^^^^^^^^^^^

Computes the GRS using a '.grs' file and individual-level genotypes. 

The GRS are computed using the following formula:

.. math::
    GRS_i = \sum_{j=0}^{m} \beta_j \cdot X_{ij} 

Where :math:`GRS_i` is the genetic risk score for the `i`:sup:`th` sample, j
indexes the genetic variants included in the score, :math:`\beta_j` is the
coefficient estimated from the summary statistics and :math:`X_{ij}` is the
number of effect alleles carried by individual `i` at variant `j`.

.. note::
    If the coefficient (:math:`\beta`) is negative, it's additive inverse is
    used instead (:math:`-\beta`) and the effect and reference alleles are
    swapped accordingly. This ensures that contributions to the GRS are always
    positive.

If the IMPUTE2 format is used to hold the genotype information, the variants
are further weighted by the INFO score as a form of genotype confidence
weighting:

.. math::
    GRS_i = \sum_{j=0}^{m} \beta_j \cdot X_{ij}  \cdot info_j

.. todo::
    Eventually, we will weight variant as soon as there is some genotype
    quality metric (e.g. in a VCF file).

.. literalinclude:: includes/grs_compute_help.txt
    :language: none

.. grs-create
.. _grs-create:

``grs-create``
^^^^^^^^^^^^^^^

Creating a GRS consists of selecting variants that will be considered jointly
when the GRS will be computed using genotype data.

This script requires a ``.grs`` file and the selection is based on the p-value
sorting and thresholding algorithm.

This method consists of the following steps:

1. Rank the variants by increasing p-value.
2. Select the top variant, include it in the GRS and exclude all variants in LD
   at a pre-defined threshold (``--ld-threshold``).
3. Repeat 2. until the p-value threshold is reached (``--p-threshold``).

.. note::

    If a ``maf`` column is present in the summary statistics association file,
    all variants that are too rare will be automaticall excluded when reading
    the summary statistics file. If it is not available, the MAF will be
    computed from the genotype data to allow filtering, which is more
    computationally intensive.

For now, a plink binary file is required for the reference. A
population-specific reference panel like the 1000 genomes phase 3 should be
used.

This script generates a 'grs' file that is suitable for computing the GRS using
:ref:`grs-compute`.

.. literalinclude:: includes/grs_create_help.txt
    :language: none

.. grs-mr
.. _grs-mr:

``grs-mr``
^^^^^^^^^^^

Mendelian randomization is a technique where genetic variants (or a genetic
risk score (G)) is used to estimate the effect of an exposure (X) on an outcome
(Y). Here, the estimation is done using the ratio method:

.. math::
    \beta = \frac{\beta_{Y \sim G}}{\beta_{X \sim G}}

In the current implementation, bootstrap resampling is used to estimate the
standard error which can be used for further inference.

.. literalinclude:: includes/grs_mr_help.txt
    :language: none

.. grs-utils
.. _grs-utils:

``grs-utils``
^^^^^^^^^^^^^^

This tool contains multiple features to manipulate GRS after they have been
computed. Most of these tools either generate plots (as png files) or modified
versions of the computed GRS. The latter file format is simply a CSV with a
'sample' and a 'grs' column.

histogram
++++++++++

The histogram sub-command generates a histogram of all the GRS values. Here is
an example:

.. image:: _static/images/histogram.png


quantiles
++++++++++

This function is used to dichotomize the GRS. The parameters are:
- ``-k`` The number of quantiles to take to form a group.
- ``-q`` The total number of quantiles.

Here are various examples of dichotomizations: ::

    # Create two groups with respect to the median.
    grs-utils quantiles my_grs.grs -k 1 -q 2

    # 1st vs 5th quintiles.
    grs-utils quantiles my_grs.grs -k 1 -q 5

    # 1st vs 5th quintiles, but keeping all the samples (will be set to NA).
    grs-utils quantiles my_grs.grs -k 1 -q 5 --keep-unclassified

The output file will contain two columns and a header: ``sample`` and
``group``. The group '0' contains the individuals with lower values of the GRS
and the group '1' contains individuals with high values of the GRS.

standardize
++++++++++++

This function standardizes a GRS:

.. math::
    GRS' = \frac{GRS - \bar{GRS}}{s}

This function is useful when different GRS are to be compared. For example, is
linear regression is used to assess the effect of a GRS on a given trait (Y),
then the coefficient can be interpreted as the number of units increase in Y
for a 1 s.d. increase in the GRS.

correlation
++++++++++++

This utility takes two computed GRS files as input and plots the correlation
between the GRS. This can be useful to test the variability of changing
different hyperparameters (e.g. p-value threshold).

This script opens the plot in the interactive matplotlib viewer. From there, it
can manipulated and saved to a file.

Here is an example:

.. image:: _static/images/correlation.png

On the plot, the linear regression fit is displayed as well as the identity
line to quickly assess the agreement between GRS.

.. literalinclude:: includes/grs_utils_help.txt
    :language: none

.. grs-evaluate
.. _grs-evaluate:

``grs-evaluate``
^^^^^^^^^^^^^^^^^

After calculating a GRS, it is important to quantify and evaluate its
predictive performance. This tool provides multiple commands to easily run
common diagnostics and validation steps.

regress
++++++++

The regress subcommand assesses the effect of the GRS as a continuous trait on
a given binary or continuous phenotype. In the case of a continuous outcome, 
linear regression is used and a plot like the following is generated:

.. image:: _static/images/evaluate-regress-continuous.png

In this plot, it is possible to quickly validate that the GRS indeed has an
effect on the trait of interest. It is also possible to estimate the fraction
of the variance explained by the GRS (corresponding to the :math:`R^2` value or
to evaluate the expected change in the phenotype attributable to a 1 unit
change in the GRS (:math:`\beta` coefficient and its 95% CI). Note that
standardizing the GRS first is useful to get an interpretation in units of GRS
s.d. and to make comparisons across different GRS possible.

If the outcome is discrete, then logistic regression is better suited to
evaluate the fit with the GRS. In this case, the generated plot looks as
follows:

.. image:: _static/images/evaluate-regress-binary.png

In this plot, the perspective is somewhat reversed, when compared with the
linear case. The GRS is now in the y axis and its distribution is shown in
cases and controls. The more general term "level" is used to describe the
discrete numerical values of the phenotype as it could be used for factor
variables.

The results for the logistic regression are shown on the plot. The OR
corresponds to the expected increase in odds for a one unit increase in the
GRS.

dichotomize-plot
+++++++++++++++++

This plot can be used to select an optimal quantile for dichotmization. The
tradeoff between the number of individuals classified and the GRS effect size
is highlighted. Note that selecting a local maxima in this plot can be a bad
idea because it increases the risk of overfitting. Traditional approaches such
as using a test set, using cross-validation or bootstrapping can be used to
select a more robust threshold.

The dichotmization is done by comparing the extremums of the distribution. For
example, at the 0.25 quantile, the first quartile (1/4) is compared to the last
quartile (4/4). All of the individuals in between can be used to form an
"intermediate" group or can be excluded from downstream analyses. This is why
the right axis represents the number of individuals that are classified in
either the "high" or "low" group for a given dichotmization.

.. image:: _static/images/evaluate-dichotomize-plot.png


ROC
++++

To evaluate the predictive accuracy of a GRS, it can be useful to look at its
ROC curve.

.. image:: _static/images/evaluate-roc.png

.. literalinclude:: includes/grs_evaluate_help.txt
    :language: none
