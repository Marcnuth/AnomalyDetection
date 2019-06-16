# -*- coding: utf-8 -*-
"""
    anomaly_detection
    A Pure Python library of Twitter's AnomalyDetection R Package.
    :license: Apache-2.0, see LICENSE for more details.
"""

__version__ = '0.0.5'
__name__ = 'tad'
from anomaly_detection import settings
from anomaly_detection.anomaly_detect_ts import anomaly_detect_ts
from anomaly_detection.anomaly_detect_vec import anomaly_detect_vec
