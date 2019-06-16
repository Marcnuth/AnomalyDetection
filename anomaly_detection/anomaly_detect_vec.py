import numpy as np
import pandas as pd
from anomaly_detection.anomaly_detect_ts import _detect_anoms

'''
Description:

     A technique for detecting anomalies in seasonal univariate time
     series where the input is a series of observations.

Usage:

     AnomalyDetectionVec(x, max_anoms = 0.1, direction = "pos", alpha = 0.05,
       period = NULL, only_last = F, threshold = "None", e_value = F,
       longterm_period = NULL, plot = F, y_log = F, xlabel = "",
       ylabel = "count", title = NULL, verbose = FALSE)
     
Arguments:

       x: Time series as a column data frame, list, or vector, where
          the column consists of the observations.

max_anoms: Maximum number of anomalies that S-H-ESD will detect as a
          percentage of the data.

direction: Directionality of the anomalies to be detected. Options are:
          "pos" | "neg" | "both".

   alpha: The level of statistical significance with which to accept or
          reject anomalies.

  period: Defines the number of observations in a single period, and
          used during seasonal decomposition.

only_last: Find and report anomalies only within the last period in the
          time series.

threshold: Only report positive going anoms above the threshold
          specified. Options are: None | "med_max" | "p95" |
          "p99".

 e_value: Add an additional column to the anoms output containing the
          expected value.
longterm_period: Defines the number of observations for which the trend
          can be considered flat. The value should be an integer
          multiple of the number of observations in a single period.
          This increases anom detection efficacy for time series that
          are greater than a month.

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

     "longterm_period" This option should be set when the input time
     series is longer than a month. The option enables the approach
     described in Vallis, Hochenbaum, and Kejariwal (2014).
     "threshold" Filter all negative anomalies and those anomalies
     whose magnitude is smaller than one of the specified thresholds
     which include: the median of the daily max values (med_max), the
     95th percentile of the daily max values (p95), and the 99th
     percentile of the daily max values (p99).

Value:

     The returned value is a list with the following components.

    anoms: Data frame containing index, values, and optionally expected
          values.

    plot: A graphical object if plotting was requested by the user. The
          plot contains the estimated anomalies annotated on the input
          time series.
     whose magnitude is smaller than one of the specified thresholds
     which include: the median of the daily max values (med_max), the
     95th percentile of the daily max values (p95), and the 99th
     percentile of the daily max values (p99).

Value:

     The returned value is a list with the following components.

   anoms: Data frame containing index, values, and optionally expected
          values.

    plot: A graphical object if plotting was requested by the user. The
          plot contains the estimated anomalies annotated on the input
          time series.
     One can save "anoms" to a file in the following fashion:
     write.csv(<return list name>[["anoms"]], file=<filename>)

     One can save plot to a file in the following fashion:
     ggsave(<filename>, plot=<return list name>[["plot"]])

References:

     Vallis, O., Hochenbaum, J. and Kejariwal, A., (2014) "A Novel
     Technique for Long-Term Anomaly Detection in the Cloud", 6th
     USENIX, Philadelphia, PA.

     Rosner, B., (May 1983), "Percentage Points for a Generalized ESD
     Many-Outlier Procedure" , Technometrics, 25(2), pp. 165-172.

See Also:

     "AnomalyDetectionTs"

Examples:

     data(raw_data)
     AnomalyDetectionVec(raw_data[,2], max_anoms=0.02, period=1440, direction="both", plot=TRUE)
     # To detect only the anomalies in the last period, run the following:
     AnomalyDetectionVec(raw_data[,2], max_anoms=0.02, period=1440, direction="both",
     only_last=TRUE, plot=TRUE)
     
'''


def __verbose_if(condition, args, kwargs=None):
    if condition:
        print(args, kwargs)


def anomaly_detect_vec(x, max_anoms=0.1, direction="pos", alpha=0.05,
                       period=None, only_last=False, threshold=None, e_value=False,
                       longterm_period=None, plot=False, y_log=False, xlabel="",
                       ylabel="count", title="", verbose=False):

    assert isinstance(x, pd.Series), 'x must be pandas series'
    assert max_anoms < 0.5, 'max_anoms must be < 0.5'
    assert direction in ['pos', 'neg', 'both'], 'direction should be one of "pos", "neg", "both"'
    assert period, "Period must be set to the number of data points in a single period"

    __verbose_if((alpha < 0.01 or alpha > 0.1) and verbose,
                 "Warning: alpha is the statistical significance, and is usually between 0.01 and 0.1")

    max_anoms = 1.0 / x.size if max_anoms < 1.0 / x.size else max_anoms

    step = int(np.ceil(x.size / longterm_period)
               ) if longterm_period else x.size
    all_data = [x.iloc[i:i + step] for i in range(0, x.size, step)]

    one_tail = True if direction in ['pos', 'neg'] else False
    upper_tail = True if direction in ['pos', 'both'] else False

    all_anoms = pd.Series()
    seasonal_plus_trend = pd.Series()
    for ts in all_data:
        tmp = _detect_anoms(
            ts, k=max_anoms, alpha=alpha, num_obs_per_period=period, use_decomp=True,
            use_esd=False, direction=direction, verbose=verbose)

        s_h_esd_timestamps = tmp['anoms'].keys()

        data_decomp = tmp['stl']

        anoms = ts.loc[s_h_esd_timestamps]
        if threshold:
            end = longterm_period - 1 if longterm_period else x.size - 1
            periodic_maxs = [ts.iloc[i: i + period].max()
                             for i in range(0, end, period)]

            if threshold == "med_max":
                thresh = periodic_maxs.median()
            elif threshold == "p95":
                thresh = periodic_maxs.quantile(0.95)
            elif threshold == "p99":
                thresh = periodic_maxs.quantile(0.99)

            anoms = anoms[anoms >= thresh]
            all_anoms.append(anoms)
            seasonal_plus_trend.append(data_decomp)

    all_anoms.drop_duplicates(inplace=True)
    seasonal_plus_trend.drop_duplicates(inplace=True)

    return anoms
