import sys
sys.path.append("..")
sys.path.append("../anomaly_detection/")

from anomaly_detection.anomaly_detect_ts import _detect_anoms, anomaly_detect_ts,\
    _get_data_tuple, _get_max_outliers, _get_max_anoms, _get_decomposed_data_tuple,\
    _perform_threshold_filter, _get_plot_breaks, _get_only_last_results, _get_period

import pandas as pd, numpy as np
from pandas.core.series import Series
import unittest

class TestAnomalyDetection(unittest.TestCase):

    def setUp(self):
        self.data1 = pd.read_csv('test_data_1.csv', index_col='timestamp',
                       parse_dates=True, squeeze=True,
                       date_parser=self.dparserfunc)
        
        self.data2 = pd.read_csv('test_data_2.csv', index_col='timestamp',
                       parse_dates=True, squeeze=True,
                       date_parser=self.dparserfunc)
        
        self.data3 = pd.read_csv('test_data_3.csv', index_col='timestamp',
                       parse_dates=True, squeeze=True,
                       date_parser=self.dparserfunc)
        
        self.data4 = pd.read_csv('test_data_4.csv', index_col='timestamp',
                       parse_dates=True, squeeze=True,
                       date_parser=self.dparserfunc)
        
        self.data5 = pd.read_csv('test_data_5.csv', index_col='timestamp',
                       parse_dates=True, squeeze=True,
                       date_parser=self.dparserfunc)        

    def get_test_value(self, raw_value):
        return np.float64(raw_value)

    def dparserfunc(self, date):
        return pd.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')

    def test_anomaly_detect_ts_1(self):
        results = anomaly_detect_ts(self.data1,
                                      direction='both', alpha=0.05,
                                      plot=False, longterm=True)
        values = results['anoms'].get_values()

        self.assertEquals(132, len(values))
        self.assertTrue(np.isin(self.get_test_value(40.0), values))
        self.assertTrue(np.isin(self.get_test_value(250.0), values))
        self.assertTrue(np.isin(self.get_test_value(210.0), values))
        self.assertTrue(np.isin(self.get_test_value(193.1036), values))
        self.assertTrue(np.isin(self.get_test_value(186.82299999999998), values))
        self.assertTrue(np.isin(self.get_test_value(181.514), values))
        self.assertTrue(np.isin(self.get_test_value(27.6501), values))
        self.assertTrue(np.isin(self.get_test_value(30.6972), values))
        self.assertTrue(np.isin(self.get_test_value(39.5523), values))
        self.assertTrue(np.isin(self.get_test_value(37.3052), values))
        self.assertTrue(np.isin(self.get_test_value(30.8174), values))
        self.assertTrue(np.isin(self.get_test_value(23.9362), values))
        self.assertTrue(np.isin(self.get_test_value(27.6677), values))
        self.assertTrue(np.isin(self.get_test_value(23.9362), values))
        self.assertTrue(np.isin(self.get_test_value(149.541), values))
        self.assertTrue(np.isin(self.get_test_value(52.0359), values))
        self.assertTrue(np.isin(self.get_test_value(52.7478), values))
        self.assertTrue(np.isin(self.get_test_value(151.549), values))
        self.assertTrue(np.isin(self.get_test_value(147.028), values))
        self.assertTrue(np.isin(self.get_test_value(31.2614), values))
               
    def test_anomaly_detect_ts_2(self):
        results = anomaly_detect_ts(self.data2,
                                      direction='both', alpha=0.02, max_anoms=0.02,
                                      plot=False, longterm=True)
        values = results['anoms'].get_values()   
        
        self.assertEquals(2, len(values))
        self.assertTrue(np.isin(self.get_test_value(-549.97419676451), values))
        self.assertTrue(np.isin(self.get_test_value(-3241.79887765979), values))
        
    def test_anomaly_detect_ts_3(self):
        results = anomaly_detect_ts(self.data3,
                                      direction='both', alpha=0.02, max_anoms=0.02,
                                      plot=False, longterm=True)
        values = results['anoms'].get_values()
        
        self.assertEquals(6, len(values))
        self.assertTrue(np.isin(self.get_test_value(677.306772096232), values))
        self.assertTrue(np.isin(self.get_test_value(3003.3770260296196), values))
        self.assertTrue(np.isin(self.get_test_value(375.68211544563), values)) 
        self.assertTrue(np.isin(self.get_test_value(4244.34731650009), values))
        self.assertTrue(np.isin(self.get_test_value(2030.44357652981), values))
        self.assertTrue(np.isin(self.get_test_value(4223.461867236129), values))
        
    def test_anomaly_detect_ts_4(self):
        results = anomaly_detect_ts(self.data4,
                                      direction='both', alpha=0.02, max_anoms=0.02,
                                      plot=False, longterm=True)
        values = results['anoms'].get_values()
        
        self.assertEquals(1, len(values))
        self.assertTrue(np.isin(self.get_test_value(-1449.62440286), values))
 
    def test_anomaly_detect_ts_5(self):
        results = anomaly_detect_ts(self.data5,
                                      direction='both', alpha=0.02, max_anoms=0.02,
                                      plot=False, longterm=True)
        values = results['anoms'].get_values()
        
        self.assertEquals(4, len(values))
        self.assertTrue(np.isin(self.get_test_value(-3355.47215640248), values))
        self.assertTrue(np.isin(self.get_test_value(941.905602754994), values))
        self.assertTrue(np.isin(self.get_test_value(-2428.98882200991), values)) 
        self.assertTrue(np.isin(self.get_test_value(-1263.4494013677302), values))
 
    def test_detect_anoms(self):
        shesd = _detect_anoms(self.data1, k=0.02, alpha=0.05,
                                num_obs_per_period=1440,
                                use_decomp=True, use_esd=False,
                                direction='both', verbose=False)
        self.assertEquals(133, len(shesd['anoms']))
        
    def test__detect_anoms_pos(self):
        shesd = _detect_anoms(self.data1, k=0.02, alpha=0.05,
                                num_obs_per_period=1440,
                                use_decomp=True, use_esd=False,
                                direction='pos', verbose=False)
        self.assertEquals(50, len(shesd['anoms']))

    def test__detect_anoms_neg(self):
        shesd = _detect_anoms(self.data1, k=0.02, alpha=0.05,
                                num_obs_per_period=1440,
                                use_decomp=True, use_esd=False,
                                direction='neg', verbose=False)
        self.assertEquals(85, len(shesd['anoms']))

    def test__detect_anoms_use_decomp_false(self):
        shesd = _detect_anoms(self.data1, k=0.02, alpha=0.05,
                                num_obs_per_period=1440,
                                use_decomp=False, use_esd=False,
                                direction='both', verbose=False)
        self.assertEquals(133, len(shesd['anoms']))

    def test__detect_anoms_no_num_obs_per_period(self):
        with self.assertRaises(AssertionError): 
            _detect_anoms(self.data1, k=0.02, alpha=0.05,
                            num_obs_per_period=None,
                            use_decomp=False, use_esd=False,
                            direction='both', verbose=False)

    def test__detect_anoms_use_esd_true(self):
        shesd = _detect_anoms(self.data1, k=0.02, alpha=0.05,
                                num_obs_per_period=1440,
                                use_decomp=True, use_esd=True,
                                direction='both', verbose=False)
        self.assertEquals(133, len(shesd['anoms']))
                  
    def test_anomaly_detect_ts_last_only_none(self):
        results = anomaly_detect_ts(self.data1, max_anoms=0.02,
                                      direction='both',
                                      only_last=None, plot=False)
        self.assertEquals(132, len(results['anoms']))


    def test_anomaly_detect_ts_last_only_day(self):
        results = anomaly_detect_ts(self.data1, max_anoms=0.02,
                                      direction='both',
                                      only_last='day', plot=False)
        self.assertEquals(23, len(results['anoms']))

    def test_anomaly_detect_ts_last_only_hr(self):
        results = anomaly_detect_ts(self.data1, max_anoms=0.02,
                                      direction='both',
                                      only_last='hr', plot=False)
        values = results['anoms'].get_values()
        
        self.assertEquals(3, len(values))
        self.assertTrue(np.isin(self.get_test_value(40.0), values))
        self.assertTrue(np.isin(self.get_test_value(250.0), values))
        self.assertTrue(np.isin(self.get_test_value(210.0), values)) 
        
    def test_anomaly_detect_ts_pos_only(self):
        results = anomaly_detect_ts(self.data1, max_anoms=0.02,
                                      direction='pos', 
                                      only_last=None, plot=False)
        self.assertEquals(50, len(results['anoms']))
        
    def test_anomaly_detect_ts_neg_only(self):
        results = anomaly_detect_ts(self.data1, max_anoms=0.02,
                                      direction='neg', 
                                      only_last=None, plot=False)
        self.assertEquals(84, len(results['anoms']))

    def test_anomaly_detect_ts_med_max_threshold(self):
        results = anomaly_detect_ts(self.data1, max_anoms=0.02,
                                      direction='both', threshold='med_max',
                                      only_last=None, plot=False)
        values = results['anoms'].get_values()

        self.assertEquals(4, len(values))
        self.assertTrue(np.isin(self.get_test_value(203.231), values))
        self.assertTrue(np.isin(self.get_test_value(203.90099999999998), values))
        self.assertTrue(np.isin(self.get_test_value(250.0), values)) 
        self.assertTrue(np.isin(self.get_test_value(210.0), values)) 

    def test_anomaly_detect_ts_longterm(self):
        results = anomaly_detect_ts(self.data1, max_anoms=0.02,
                                      direction='both', threshold=None,
                                      only_last=None, longterm=True)
        self.assertEquals(132, len(results['anoms']))

    def test_anomaly_detect_ts_piecewise_median_period_weeks(self):
        results = anomaly_detect_ts(self.data1, max_anoms=0.02, piecewise_median_period_weeks=4,
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
            
    def test_get_data_tuple(self):
        d_tuple = _get_data_tuple(self.data1, 24, None)
        raw_data = d_tuple[0]
        period = d_tuple[1]
        granularity = d_tuple[2]
        
        self.assertTrue(isinstance(raw_data, Series))  
        self.assertTrue(isinstance(period, int))
        self.assertTrue(isinstance(granularity, str))   
           
        self.assertEquals(24, period)
        self.assertEquals('min', granularity)
        self.assertEquals(14398, len(raw_data))
        
    def test_get_max_outliers(self):
        self.assertEquals(719, _get_max_outliers(self.data1, 0.05))
    
    def test_get_max_anoms(self):
        max_anoms = _get_max_anoms(self.data1, 0.1)
        self.assertEquals(0.1, max_anoms)
        
    def test_get_decomposed_data_tuple(self):
        data, smoothed_data = _get_decomposed_data_tuple(self.data1, 1440)
        self.assertTrue(isinstance(data, Series))
        self.assertTrue(isinstance(smoothed_data, Series))
        self.assertEquals(14398, len(data))
        self.assertEquals(14398, len(smoothed_data))
        
    def test_perform_threshold_filter(self):
        results = anomaly_detect_ts(self.data1, max_anoms=0.02, direction='both',
                                      only_last=None, plot=False)
        periodic_max = self.data1.resample('1D').max()
        filtered_results = _perform_threshold_filter(results['anoms'], periodic_max, 'med_max')
        self.assertTrue(isinstance(filtered_results, Series))
        self.assertEquals(4, len(filtered_results))
        
    def test_get_plot_breaks(self):
        self.assertEquals(36, _get_plot_breaks('day', 'day'))
        self.assertEquals(12, _get_plot_breaks('min', 'day'))
        self.assertEquals(3, _get_plot_breaks('min', 'min'))
        
    def test_get_only_last_results(self):
        results = anomaly_detect_ts(self.data1, max_anoms=0.02, direction='both', 
                                    only_last=None, plot=False)

        last_day = _get_only_last_results(self.data1, results['anoms'], 'min', 'day')
        last_hr = _get_only_last_results(self.data1, results['anoms'], 'min', 'hr')
        self.assertEquals(23, len(last_day))
        self.assertEquals(3, len(last_hr))
    
    def test_get_period(self):
        self.assertEquals(1440, _get_period(1440, None))
        
    def test_get_period_with_override(self):
        self.assertEquals(720, _get_period(1440, 720))
