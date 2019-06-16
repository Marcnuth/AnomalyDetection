#!/usr/bin/python3
# coding=utf-8
from logging.config import dictConfig


dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "basic": {
            "format": "[%(asctime)s] [%(process)d:%(thread)d] [%(levelname)s] [%(name)s] %(filename)s:%(funcName)s:%(lineno)d %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "basic",
            "stream": "ext://sys.stdout"
        },
    },
    "loggers": {
        "anomaly_detection": {
            "handlers": ["console"],
            "propagate": "true",
            "level": "INFO"
        }
    }
})
