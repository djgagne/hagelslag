.. title:: Forecast Evaluation

.. evaluation:

Forecast Evaluation
===================

Hagelslag's evaluation package contains many ways to evaluate probabilistic and deterministic forecasts and tools for
plotting them. Unlike other verification packages, the evaluation score objects in hagelslag are designed to be scaleable
to very large datasets that would not fit into memory on their own.

Standard vs. Distributed Metrics
--------------------------------

Hagelslag contains implementations of standard probability metrics, including scores related to the ROC curve and
reliability diagram. The scoring information is stored in two formats: standard and distributed. In the standard format,
all of the forecast and observation pairs are stored within the object, and scores can be calculated at any time.
For smaller sample sizes, these functions work well.

For larger verification tasks, such as calculating scores over multiple months of high resolution NWP ensemble runs,
all of the forecast and observation pairs may not fit into memory. There also may be some desire to easily examine
different subsets of the data and calculate scores on each of these subsets. The Distributed forms of the metrics are
useful for these tasks. Instead of storing every forecast-observation pair, the forecasts are discretized into a set of
probability thresholds, and then counts for each threshold are stored when the `update` method is called.
An array of DistributedROC or DistributedReliability objects can be summed together using numpy.sum
to aggregate statistics collected across multiple runs into one summary. Resampling this array of verification
objects can enable bootstrapping and permutation test statistics to be calculated on the data.

Distributed evaluation objects can be parsed into string format and saved into tabular data files for later analysis.
If you store the Distributed objects in a pandas dataframe and run the df.to_csv method, the contents of the objects
will automatically be converted to their string representations during the save process. After saving to disk, one
can create a new set of distributed objects and use their from_str method to convert the string back into an object.
One could also use pickle to serialize the objects, which may be more convenient in some cases.

Metric Plots
------------
Hagelslag has built-in support for common probability metric plots, including the ROC Curve,
Performance (or Roebber) diagram, reliability, and attributes diagrams in the
evaluation.MetricPlotter.py module. Each of these functions takes one or more DistributedROC
or DistributedReliability objects and plots them together. The code also provides support for
bootstrapping of the statistics to visualize confidence intervals around each score curve.

hseval Config Options
---------------------

:ensemble_name: Name of the Ensemble forecast system. Supports "SSEF", "NCAR", and others.
:ensemble_members: List of ensemble member names.
:start_date: Datetime object for the first model run date being evaluated.
:end_date: Datetime object for the last model run date being evaluated.
:start_hour: First forecast hour
:end_hour: Last forecast hour
