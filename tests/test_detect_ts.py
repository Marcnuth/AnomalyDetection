import sys
sys.path.append("..")
sys.path.append("../anomaly_detection/")
from anomaly_detection.anomaly_detect_ts import anomaly_detect_ts, _detect_anoms

import pandas as pd
import unittest

class TestAnomalyDetection(unittest.TestCase):

    def setUp(self):
        self.data = pd.read_csv('test_data.csv', index_col='timestamp',
                       parse_dates=True, squeeze=True,
                       date_parser=self.dparserfunc)

    def dparserfunc(self, date):
        return pd.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')


    def test__detect_anoms(self):
        shesd = _detect_anoms(self.data, k=0.02, alpha=0.05,
                                num_obs_per_period=1440,
                                use_decomp=True, use_esd=False,
                                direction='both', verbose=False)
        self.assertEquals(133, len(shesd['anoms']))
        
    def test__detect_anoms_pos(self):
        shesd = _detect_anoms(self.data, k=0.02, alpha=0.05,
                                num_obs_per_period=1440,
                                use_decomp=True, use_esd=False,
                                direction='pos', verbose=False)
        self.assertEquals(50, len(shesd['anoms']))

    def test__detect_anoms_neg(self):
        shesd = _detect_anoms(self.data, k=0.02, alpha=0.05,
                                num_obs_per_period=1440,
                                use_decomp=True, use_esd=False,
                                direction='neg', verbose=False)
        self.assertEquals(85, len(shesd['anoms']))

    def test__detect_anoms_use_decomp_false(self):
        shesd = _detect_anoms(self.data, k=0.02, alpha=0.05,
                                num_obs_per_period=1440,
                                use_decomp=False, use_esd=False,
                                direction='both', verbose=False)
        self.assertEquals(133, len(shesd['anoms']))

    def test__detect_anoms_no_num_obs_per_period(self):
        with self.assertRaises(AssertionError): 
            _detect_anoms(self.data, k=0.02, alpha=0.05,
                            num_obs_per_period=None,
                            use_decomp=False, use_esd=False,
                            direction='both', verbose=False)

    def test__detect_anoms_use_esd_true(self):
        shesd = _detect_anoms(self.data, k=0.02, alpha=0.05,
                                num_obs_per_period=1440,
                                use_decomp=True, use_esd=True,
                                direction='both', verbose=False)
        self.assertEquals(133, len(shesd['anoms']))
                
    def test_anomaly_detect_ts_last_only_none(self):
        results = anomaly_detect_ts(self.data, max_anoms=0.02,
                                      direction='both',
                                      only_last=None, plot=False)
        self.assertEquals(132, len(results['anoms']))


    def test_anomaly_detect_ts_last_only_day(self):
        results = anomaly_detect_ts(self.data, max_anoms=0.02,
                                      direction='both',
                                      only_last='day', plot=False)
        self.assertEquals(23, len(results['anoms']))

    def test_anomaly_detect_ts_last_only_hr(self):
        results = anomaly_detect_ts(self.data, max_anoms=0.02,
                                      direction='both',
                                      only_last='hr', plot=False)
        self.assertEquals(3, len(results['anoms']))

    def test_anomaly_detect_ts_pos_only(self):
        results = anomaly_detect_ts(self.data, max_anoms=0.02,
                                      direction='pos', 
                                      only_last=None, plot=False)
        self.assertEquals(50, len(results['anoms']))
        
    def test_anomaly_detect_ts_neg_only(self):
        results = anomaly_detect_ts(self.data, max_anoms=0.02,
                                      direction='neg', 
                                      only_last=None, plot=False)
        self.assertEquals(84, len(results['anoms']))

    def test_anomaly_detect_ts_med_max_threshold(self):
        results = anomaly_detect_ts(self.data, max_anoms=0.02,
                                      direction='both', threshold='med_max',
                                      only_last=None, plot=False)
        self.assertEquals(4, len(results['anoms']))

    def test_anomaly_detect_ts_longterm(self):
        results = anomaly_detect_ts(self.data, max_anoms=0.02,
                                      direction='both', threshold=None,
                                      only_last=None, longterm=True)
        self.assertEquals(132, len(results['anoms']))

    def test_anomaly_detect_ts_piecewise_median_period_weeks(self):
        results = anomaly_detect_ts(self.data, max_anoms=0.02, piecewise_median_period_weeks=4,
                                      direction='both', threshold=None,
                                      only_last=None, longterm=False)
        self.assertEquals(132, len(results['anoms']))

    def test_invalid_data_parameter(self):
        with self.assertRaises(AssertionError): 
            anomaly_detect_ts(['invalid'], max_anoms=0.02,
                                      direction='both', threshold=None,
                                      only_last=None, longterm=False)

    def test_invalid_piecewise_median_period_weeks(self):
        with self.assertRaises(AssertionError): 
            anomaly_detect_ts(['invalid'], max_anoms=0.02, piecewise_median_period_weeks=1,
                                      direction='both', threshold=None,
                                      only_last=None, longterm=False, plot=False)