.. title:: Forecast Evaluation

.. evaluation:

Forecast Evaluation
===================

Hagelslag's evaluation package contains many ways to evaluate probabilistic and deterministic forecasts and tools for
plotting them.

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
probability thresholds, and then counts for each threshold are stored.

hseval Config Options
---------------------

:ensemble_name: Name of the Ensemble forecast system. Supports "SSEF", "NCAR", and others.
:ensemble_members: List of ensemble member names.
:start_date: Datetime object for the first model run date being evaluated.
:end_date: Datetime object for the last model run date being evaluated.
:start_hour: First forecast hour
:end_hour: Last forecast hour
