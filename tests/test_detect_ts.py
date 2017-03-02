import sys
sys.path.append("..")
sys.path.append("../anomaly_detection/")
from anomaly_detection import anomaly_detect_ts as detts


import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('test_data.csv', index_col='timestamp',
                   parse_dates=True, squeeze=True,
                   date_parser=lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

#sns.tsplot(data)
#plt.show()

#results = detts.anomaly_detect_ts(data, max_anoms=0.02, direction='both', only_last='day', plot=False)


#assert results['anoms'].size == 25


shesd = detts._detect_anoms(data, k=0.02, alpha=0.05, num_obs_per_period=1440, use_decomp=True, use_esd=False, direction='both', verbose=False)




'''

test_that("both directions, e_value, with longterm", {
  results <- AnomalyDetectionTs(raw_data, max_anoms=0.02, direction='both', longterm=TRUE, e_value=TRUE)
  expect_equal(length(results$anoms), 3)
  expect_equal(length(results$anoms[[2L]]), 131)
  expect_equal(results$plot, NULL)
})

test_that("both directions, e_value, threshold set to med_max", {
  results <- AnomalyDetectionTs(raw_data, max_anoms=0.02, direction='both', threshold="med_max", e_value=TRUE)
  expect_equal(length(results$anoms), 3)
  expect_equal(length(results$anoms[[2L]]), 4)
  expect_equal(results$plot, NULL)
})

'''
print('OK')
