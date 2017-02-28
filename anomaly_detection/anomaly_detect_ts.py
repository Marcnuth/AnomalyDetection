'''
Description:

     A technique for detecting anomalies in seasonal univariate time
     series where the input is a series of <timestamp, count> pairs.


Usage:

     nomaly_detect_ts(x, max_anoms=0.1, direction="pos", alpha=0.05, only_last=None,
                      threshold="None", e_value=False, longterm=False, piecewise_median_period_weeks=2,
                      plot=False, y_log=False, xlabel="", ylabel="count", title=None, verbose=False)

Arguments:

       x: Time series as a two column data frame where the first column
          consists of the timestamps and the second column consists of
          the observations.

max_anoms: Maximum number of anomalies that S-H-ESD will detect as a
          percentage of the data.

direction: Directionality of the anomalies to be detected. Options are:
          ‘'pos' | 'neg' | 'both'’.

   alpha: The level of statistical significance with which to accept or
          reject anomalies.

only_last: Find and report anomalies only within the last day or hr in
          the time series. ‘NULL | 'day' | 'hr'’.

threshold: Only report positive going anoms above the threshold
          specified. Options are: ‘'None' | 'med_max' | 'p95' |
          'p99'’.

 e_value: Add an additional column to the anoms output containing the
          expected value.

longterm: Increase anom detection efficacy for time series that are
         greater than a month. See Details below.

piecewise_median_period_weeks: The piecewise median time window as
          described in Vallis, Hochenbaum, and Kejariwal (2014).
          Defaults to 2.

    plot: A flag indicating if a plot with both the time series and the
          estimated anoms, indicated by circles, should also be
          returned.

   y_log: Apply log scaling to the y-axis. This helps with viewing
          plots that have extremely large positive anomalies relative
          to the rest of the data.

  xlabel: X-axis label to be added to the output plot.

  ylabel: Y-axis label to be added to the output plot.

   title: Title for the output plot.

 verbose: Enable debug messages

Details:

     ‘longterm’ This option should be set when the input time series
     is longer than a month. The option enables the approach described
     in Vallis, Hochenbaum, and Kejariwal (2014).
     ‘threshold’ Filter all negative anomalies and those anomalies
     whose magnitude is smaller than one of the specified thresholds
     which include: the median of the daily max values (med_max), the
     95th percentile of the daily max values (p95), and the 99th
     percentile of the daily max values (p99).

Value:

     The returned value is a list with the following components.

   anoms: Data frame containing timestamps, values, and optionally
          expected values.

    plot: A graphical object if plotting was requested by the user. The
          plot contains the estimated anomalies annotated on the input
          time series.
     ‘threshold’ Filter all negative anomalies and those anomalies
     whose magnitude is smaller than one of the specified thresholds
     which include: the median of the daily max values (med_max), the
     95th percentile of the daily max values (p95), and the 99th
     percentile of the daily max values (p99).

Value:

     The returned value is a list with the following components.

   anoms: Data frame containing timestamps, values, and optionally
          expected values.

    plot: A graphical object if plotting was requested by the user. The
          plot contains the estimated anomalies annotated on the input
          time series.
     One can save ‘anoms’ to a file in the following fashion:
     ‘write.csv(<return list name>[["anoms"]], file=<filename>)’

     One can save ‘plot’ to a file in the following fashion:
     ‘ggsave(<filename>, plot=<return list name>[["plot"]])’

References:

     Vallis, O., Hochenbaum, J. and Kejariwal, A., (2014) "A Novel
     Technique for Long-Term Anomaly Detection in the Cloud", 6th
     USENIX, Philadelphia, PA.

     Rosner, B., (May 1983), "Percentage Points for a Generalized ESD
     Many-Outlier Procedure" , Technometrics, 25(2), pp. 165-172.

See Also:

     anomaly_detect_vec

Examples:

     anomaly_detect_ts(raw_data, max_anoms=0.02, direction='both', plot=TRUE)
     # To detect only the anomalies on the last day, run the following:
     anomaly_detect_ts(raw_data, max_anoms=0.02, direction='both', only_last="day", plot=TRUE)

'''

import numpy as np
import pandas as pd
import datetime
import statsmodels.api as sm


def anomaly_detect_ts(x, max_anoms=0.1, direction="pos", alpha=0.05, only_last=None,
                      threshold=None, e_value=False, longterm=False, piecewise_median_period_weeks=2,
                      plot=False, y_log=False, xlabel="", ylabel="count", title=None, verbose=False, dropna=False):

    # validation
    assert isinstance(x, pd.Series), 'Data must be a series(Pandas.Series)'
    assert x.values.dtype in [int, float], 'Values of the series must be number'
    assert x.index.dtype == datetime.datetime, 'Index of the series must be number or datetime'
    assert max_anoms <= 0.49 and max_anoms >= 0, 'max_anoms must be non-negative and less than 50% '
    assert direction in ['pos', 'neg', 'both'], 'direction options: pos | neg | both'
    assert only_last in [None, 'day', 'hr'], 'only_last options: None | day | hr'
    assert threshold in [None, 'med_max', 'p95', 'p99'], 'threshold options: None | med_max | p95 | p99'
    assert piecewise_median_period_weeks >= 2, 'piecewise_median_period_weeks must be greater than 2 weeks'

    # conversion
    title = '' if title is None else (title + ' : ')
    data = x.sort_index()
    # TODO...
    # Allow x.index to be number, here we can convert it to datetime

    # verbose
    if verbose:
        if max_anoms == 0:
            print('0 max_anoms results in max_outliers being 0.')
        if alpha < 0.01 or alpha > 0.1:
            print('Warning: alpha is the statistical signifigance, and is usually between 0.01 and 0.1')

    timediff = data.index[1] - data.index[0]
    if timediff.days > 0:
        num_days_per_line = 7
        only_last = 'day' if only_last == 'hr' else only_last
        period = 7
        granularity = 'day'
    elif timediff.seconds / 60 / 60 >= 1:
        granularity = 'hr'
        period = 24
    elif timediff.seconds / 60 >= 1:
        granularity = 'min'
        period = 1440
    elif timediff.seconds > 0:
        granularity = 'sec'
        # Aggregate data to minutely if secondly
        data = data.resample('60s', label='right').sum()
    else:
        granularity = 'ms'

    max_anoms = 1 / data.size if max_anoms < 1 / data.size else max_anoms

    # If longterm is enabled, break the data into subset data frames and store in all_data
    if longterm:
        # Pre-allocate list with size equal to the number of piecewise_median_period_weeks chunks in x + any left over chunk
        # handle edge cases for daily and single column data period lengths
        num_obs_in_period = period * piecewise_median_period_weeks + 1 if granularity == 'day' else period * 7 * piecewise_median_period_weeks
        num_days_in_period = (7 * piecewise_median_period_weeks) + 1 if granularity == 'day' else (7 * piecewise_median_period_weeks)

        all_data = []
        # Subset x into piecewise_median_period_weeks chunks
        for i in range(1, data.size + 1, num_obs_in_period):
            start_date = data.index[i]
            # if there is at least 14 days left, subset it, otherwise subset last_date - 14 days
            end_date = start_date + datetime.timedelta(days=num_days_in_period)
            if end_date < data.index[-1]:
                all_data.append(data.loc[lambda x: x.index >= start_date and x.index <= end_date])
            else:
                all_data.append(data.loc[lambda x: x.index >= data.index[-1] - datetime.timedelta(days=num_days_in_period)])

    else:
        all_data = data

    all_anoms = pd.Series()
    seasonal_plus_trend = pd.Series()
    # Detect anomalies on all data (either entire data in one-pass, or in 2 week blocks if longterm=TRUE)
    for series in all_data:
        pass
        
    raise Exception('Unfinied now!')

# Detects anomalies in a time series using S-H-ESD.
#
# Args:
#	 data: Time series to perform anomaly detection on.
#	 k: Maximum number of anomalies that S-H-ESD will detect as a percentage of the data.
#	 alpha: The level of statistical significance with which to accept or reject anomalies.
#	 num_obs_per_period: Defines the number of observations in a single period, and used during seasonal decomposition.
#	 use_decomp: Use seasonal decomposition during anomaly detection.
#	 use_esd: Uses regular ESD instead of hybrid-ESD. Note hybrid-ESD is more statistically robust.
#	 one_tail: If TRUE only positive or negative going anomalies are detected depending on if upper_tail is TRUE or FALSE.
#	 upper_tail: If TRUE and one_tail is also TRUE, detect only positive going (right-tailed) anomalies. If FALSE and one_tail is TRUE, only detect negative (left-tailed) anomalies.
#	 verbose: Additionally printing for debugging.
# Returns:
#   A list containing the anomalies (anoms) and decomposition components (stl).


def _detect_anoms(data, k=0.49, alpha=0.05, num_obs_per_period=None,
                  use_decomp=True, use_esd=False, direction="pos", verbose=False):

    # validation
    assert not num_obs_per_period, "must supply period length for time series decomposition"
    assert direction in ['pos', 'neg', 'both'], 'direction options: pos | neg | both'
    assert data.size >= num_obs_per_period * 2, 'Anomaly detection needs at least 2 periods worth of data'
    assert not data[data.isnull()], 'Data contains NA. We suggest replacing NA with interpolated values before detecting anomaly'

    # conversion
    if direction == 'pos':
        one_tail, upper_tail = True, True
    elif direction == 'neg':
        one_tail, upper_tail = True, False
    else:
        one_tail, upper_tail = False, True

    # -- Step 1: Decompose data. This returns a univarite remainder which will be used for anomaly detection. Optionally, we might NOT decompose.
    # Note: R use stl, but here we will use MA, the result may be different TODO.. Here need improvement
    decomposed = sm.tsa.seasonal_decompose(data)
    data = data - decomposed.seasonal - np.median(data)
    smoothed = decomposed.seasonal + decomposed.trend

    max_outliers = np.trunc(data.size * k)
    assert not max_outliers, 'With longterm=TRUE, AnomalyDetection splits the data into 2 week periods by default. You have {0} observations in a period, which is too few. Set a higher piecewise_median_period_weeks.'.format(data.size)

    R_idx = list(range(1, max_outliers))
    num_anoms = 0

    # Compute test statistic until r=max_outliers values have been
    # removed from the sample.

    for i in range(max_outliers):
        if verbose:
            print(i, '/', max_outliers, ' completed')

        if not data.mad:
            break

        if not one_tail:
            ares = abs(data - data.median())
        elif upper_tail:
            ares = data - data.median()
        else:
            ares = data.median() - data

        ares = ares / data.mad()

        tmp_anom_index = ares[ares.values == ares.max()].index
        R_idx[tmp_anom_index] = data.loc[tmp_anom_index]

        data.drop(tmp_anom_index, inplace=True)

        # Compute critical value.
        p = 1 - alpha / (n - i + 1) if one_tail else (1 - alpha / (2 * (n - i + 1)))

        # TODO..
    
    
    
