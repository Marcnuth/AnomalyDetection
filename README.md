# Anomaly Detection for Python

![PyPI - Downloads](https://img.shields.io/pypi/dm/tad?color=lightgreen&label=PyPI)

## Introduction

Twitter's Anomaly Detection is easy to use, but it's a R library. 

Although there are some repos for python to run twitter's anomaly detection algorithm, but those libraies requires R installed.

This repo aims for rewriting twitter's Anomaly Detection algorithms in Python, and providing same functions for user.


## Install

```
pip3 install tad
```

## Requirement

1. The data should have the Index which is a datetime type. Single series is processed so only pass single numeric series at a time.
2. Plotting function is based on matplotlib, the plot is retured in the results if user wants to change any appearnaces etc.

## Usage

```
import tad

import pandas as pd 
import matplotlib.pyplot as plt

a = pd.DataFrame({'numeric_data_col1': [1,1, 1, 1, 1, 1, 10, 1, 1, 1, 1, 1, 1, 1]}, index=pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03','2020-01-04','2020-01-05','2020-01-06','2020-01-07','2020-01-08','2020-01-09','2020-01-10','2020-01-11','2020-01-12','2020-01-13','2020-01-14']))

results = anomaly_detect_ts(a['numeric_data_col1'],
                                              direction='both', alpha=0.02, max_anoms=0.20,
                                              plot=True, longterm=True)
if results['plot']: #some anoms were detected and plot was also True.
    plt.show()

```
results
{'anoms': 2020-01-14     1
 2020-01-07    10
 dtype: int64,
 'expected': None,
 'plot': <matplotlib.axes._subplots.AxesSubplot at 0x29b827b2748>}



Output shall be in the results dict

results['anoms'] : contains the anomalies detected
results['plot']: contains a matplotlib plot if anoms were detected and plot was True
results['expected'] : tries to return expected values for certain dates. TODO: inconsistent as provides different outputs compared to anoms

![Sample Script output](/resources/images/sample_execution.png)


## Other Sample Images

![Another sample of detecction using default parameters](/resources/images/sample_01.png)