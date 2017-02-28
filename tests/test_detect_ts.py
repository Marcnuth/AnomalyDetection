import sys
sys.path.append("..")
from anomaly_detection import anomaly_detect_ts as detts

detts([12, 3, 4])
print('OK')
