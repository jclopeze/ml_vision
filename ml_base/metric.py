#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import abc
from enum import Enum, auto
import json
from typing import Optional, Union, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from ml_base.dataset import Dataset
from ml_base.utils.logger import get_logger

logger = get_logger(__name__)

class MetricType(Enum):
    average = auto()
    per_class = auto()
    confusion_matrix = auto()


class Metric(abc.ABC):
    def __init__(self, name: str = None):
        self.name = name
        self._result = None

    @abc.abstractmethod
    def __call__(self, dataset_true: Dataset, dataset_pred: Dataset) -> Union[float, dict, list]:
        pass

    @abc.abstractmethod
    def get_type(self) -> MetricType:
        pass

    def _get_name(self) -> str:
        cls_name = self.__class__.__name__
        return f'{cls_name}-{self.get_type().name}'

    def get_name(self) -> str:
        if self.name is not None:
            return self.name
        return self._get_name()

    @abc.abstractmethod
    def __repr__(self):
        pass

    @property
    def result(self):
        assert self._result is not None, "The result has not been assigned yet."
        return self._result


class Precision(Metric):

    def __init__(self,
                 name: str = None,
                 labels: List[str] = None,
                 pos_label: Optional[str] = None,
                 average: Optional[str] = 'macro',
                 sample_weight: Union[List[float], Tuple[float]] = None,
                 zero_division: Union[float, int, str] = 'warn'):
        super().__init__(name=name)
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.zero_division = zero_division

    def __call__(self, dataset_true: Dataset, dataset_pred: Dataset) -> Union[float, dict]:
        y_true = list(dataset_true.df.sort_values(by='item')["label"].values)
        y_pred = list(dataset_pred.df.sort_values(by='item')["label"].values)

        if self.labels is None:
            self.labels = list(set(y_true) | set(y_pred))

        prec = precision_score(y_true,
                               y_pred,
                               labels=self.labels,
                               pos_label=self.pos_label,
                               average=self.average,
                               sample_weight=self.sample_weight,
                               zero_division=self.zero_division)

        if self.average is None:
            prec_per_class = {label: prec[i] for i, label in enumerate(self.labels)}
            self._result = prec_per_class
            return prec_per_class

        self._result = prec
        return prec

    def get_type(self) -> MetricType:
        if self.average is None:
            return MetricType.per_class
        return MetricType.average

    def __repr__(self):
        if self.get_type() == MetricType.per_class:
            return json.dumps(self.result, indent=2)
        return str(self.result)


class Recall(Metric):

    def __init__(self,
                 name: str = None,
                 labels: List[str] = None,
                 pos_label: Optional[str] = None,
                 average: Optional[str] = 'macro',
                 sample_weight: Union[List[float], Tuple[float]] = None,
                 zero_division: Union[float, int, str] = 'warn'):
        super().__init__(name=name)
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.zero_division = zero_division

    def __call__(self, dataset_true: Dataset, dataset_pred: Dataset) -> Union[float, dict]:
        y_true = list(dataset_true.df.sort_values(by='item')["label"].values)
        y_pred = list(dataset_pred.df.sort_values(by='item')["label"].values)

        if self.labels is None:
            self.labels = list(set(y_true) | set(y_pred))

        rec = recall_score(y_true,
                           y_pred,
                           labels=self.labels,
                           pos_label=self.pos_label,
                           average=self.average,
                           sample_weight=self.sample_weight,
                           zero_division=self.zero_division)

        if self.average is None:
            rec_per_class = {label: rec[i] for i, label in enumerate(self.labels)}
            self._result = rec_per_class
            return rec_per_class

        self._result = rec
        return rec

    def get_type(self) -> MetricType:
        if self.average is None:
            return MetricType.per_class
        return MetricType.average

    def __repr__(self):
        if self.get_type() == MetricType.per_class:
            return json.dumps(self.result, indent=2)
        return str(self.result)


class F1Score(Metric):

    def __init__(self,
                 name: str = None,
                 labels: List[str] = None,
                 pos_label: Optional[str] = None,
                 average: Optional[str] = 'macro',
                 sample_weight: Union[List[float], Tuple[float]] = None,
                 zero_division: Union[float, int, str] = 'warn'):
        super().__init__(name=name)
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.zero_division = zero_division

    def __call__(self, dataset_true: Dataset, dataset_pred: Dataset) -> Union[float, dict]:
        prec_eval = Precision(labels=self.labels,
                              pos_label=self.pos_label,
                              average=self.average,
                              sample_weight=self.sample_weight,
                              zero_division=self.zero_division)
        rec_eval = Recall(labels=self.labels,
                          pos_label=self.pos_label,
                          average=self.average,
                          sample_weight=self.sample_weight,
                          zero_division=self.zero_division)
        self.labels = prec_eval.labels

        prec = prec_eval(dataset_true=dataset_true, dataset_pred=dataset_pred)
        rec = rec_eval(dataset_true=dataset_true, dataset_pred=dataset_pred)

        if self.average is None:
            f1_score_per_class = {label: self.f1_score(prec[label], rec[label])
                                  for label in self.labels}
            self._result = f1_score_per_class
            return f1_score_per_class

        f1_score_avg = self.f1_score(prec, rec)
        self._result = f1_score_avg
        return f1_score_avg

    def get_type(self) -> MetricType:
        if self.average is None:
            return MetricType.per_class
        return MetricType.average

    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        return 2 * ((precision * recall) / (precision + recall))

    def __repr__(self):
        if self.get_type() == MetricType.per_class:
            return json.dumps(self.result, indent=2)
        return str(self.result)


class ConfusionMatrix(Metric):

    def __init__(self,
                 name: str = None,
                 labels: List[str] = None,
                 sample_weight: Union[List[float], Tuple[float]] = None,
                 normalize: Optional[str] = 'true'):
        super().__init__(name=name)
        self.labels = labels
        self.sample_weight = sample_weight
        self.normalize = normalize

    # TODO: Check the return type
    def __call__(self, dataset_true: Dataset, dataset_pred: Dataset) -> np.ndarray:
        y_true = list(dataset_true.df.sort_values(by='item')["label"].values)
        y_pred = list(dataset_pred.df.sort_values(by='item')["label"].values)

        if self.labels is None:
            self.labels = list(set(y_true) | set(y_pred))

        conf_mtx = confusion_matrix(y_true,
                                    y_pred,
                                    labels=self.labels,
                                    sample_weight=self.sample_weight,
                                    normalize=self.normalize)

        self._result = conf_mtx
        return conf_mtx

    def get_type(self) -> MetricType:
        return MetricType.confusion_matrix

    def _get_name(self) -> str:
        return self.get_type().name

    def __repr__(self):
        cmtx = pd.DataFrame(self._result, index=self.labels, columns=self.labels)
        cmtx.index.name = 'True labels'
        return str(cmtx)


