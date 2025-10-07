#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import abc
from enum import Enum, auto
import json
from typing import Optional, Union, List, Tuple, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, confusion_matrix, accuracy_score, matthews_corrcoef,
    roc_auc_score)

from ml_base.dataset import IDataset
from ml_base.utils.logger import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    average = auto()
    per_class = auto()
    confusion_matrix = auto()


class Metric(abc.ABC):
    def __init__(self, name: str = None):
        self._name = name
        self.set_result(None)

    @abc.abstractmethod
    def __call__(self, dataset_true: IDataset, dataset_pred: IDataset) -> Union[float, dict, list]:
        pass

    def _get_name(self) -> str:
        cls_name = self.__class__.__name__
        return f'{cls_name}-{self.type_name}'

    @abc.abstractmethod
    def __repr__(self):
        pass

    @staticmethod
    def _get_ds_values(dataset) -> list:
        return list(dataset.df.sort_values(by='item')["label"].values)

    @property
    def name(self) -> str:
        return self._name or self._get_name()

    @property
    @abc.abstractmethod
    def type(self) -> MetricType:
        pass

    @property
    def type_name(self) -> str:
        return self.type.name

    @property
    def result(self):
        assert not self._result is None, "The result has not been assigned yet."
        return self._result

    @property
    def attrs(self):
        return {name: value
                for name, value in self.__dict__.items() if not name.startswith('_')}

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)

    def to_dict(self):
        d = {
            'cls': self.__class__.__name__,
            'name': self.name,
            'result': self.result,
            'type': self.type_name,
            'attrs': self.attrs
        }
        return d

    def set_result(self, res):
        self._result = res


class Precision(Metric):

    def __init__(self,
                 name: str = None,
                 labels: List[str] = None,
                 pos_label: Optional[str] = None,
                 average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] = 'macro',
                 sample_weight: Union[List[float], Tuple[float]] = None,
                 zero_division: Union[float, int, str] = 'warn'):
        super().__init__(name=name)
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.zero_division = zero_division

    def __call__(self, dataset_true: IDataset, dataset_pred: IDataset) -> Union[float, dict]:
        y_true = self._get_ds_values(dataset_true)
        y_pred = self._get_ds_values(dataset_pred)

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
            prec_per_class = {label: float(prec[i]) for i, label in enumerate(self.labels)}
            self.set_result(prec_per_class)
            return prec_per_class

        prec = float(prec)
        self.set_result(prec)
        return prec

    @property
    def type(self) -> MetricType:
        if self.average is None:
            return MetricType.per_class
        return MetricType.average

    def __repr__(self):
        if self.type == MetricType.per_class:
            return json.dumps(self.result, indent=2)
        return str(self.result)


class Recall(Metric):

    def __init__(self,
                 name: str = None,
                 labels: List[str] = None,
                 pos_label: Optional[str] = None,
                 average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] = 'macro',
                 sample_weight: Union[List[float], Tuple[float]] = None,
                 zero_division: Union[float, int, str] = 'warn'):
        super().__init__(name=name)
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.zero_division = zero_division

    def __call__(self, dataset_true: IDataset, dataset_pred: IDataset) -> Union[float, dict]:
        y_true = self._get_ds_values(dataset_true)
        y_pred = self._get_ds_values(dataset_pred)

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
            rec_per_class = {label: float(rec[i]) for i, label in enumerate(self.labels)}
            self.set_result(rec_per_class)
            return rec_per_class

        rec = float(rec)
        self.set_result(rec)
        return rec

    @property
    def type(self) -> MetricType:
        if self.average is None:
            return MetricType.per_class
        return MetricType.average

    def __repr__(self):
        if self.type == MetricType.per_class:
            return json.dumps(self.result, indent=2)
        return str(self.result)


class F1Score(Metric):

    def __init__(self,
                 name: str = None,
                 labels: List[str] = None,
                 pos_label: Optional[str] = None,
                 average: Literal['micro', 'macro', 'samples', 'weighted', 'binary'] = 'macro',
                 sample_weight: Union[List[float], Tuple[float]] = None,
                 zero_division: Union[float, int, str] = 'warn'):
        super().__init__(name=name)
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight
        self.zero_division = zero_division

    def __call__(self, dataset_true: IDataset, dataset_pred: IDataset) -> Union[float, dict]:
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
            self.set_result(f1_score_per_class)
            return f1_score_per_class

        f1_score_avg = self.f1_score(prec, rec)
        self.set_result(f1_score_avg)
        return f1_score_avg

    @property
    def type(self) -> MetricType:
        if self.average is None:
            return MetricType.per_class
        return MetricType.average

    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        try:
            return float(2 * ((precision * recall) / (precision + recall)))
        except:
            return 0

    def __repr__(self):
        if self.type == MetricType.per_class:
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
    def __call__(self, dataset_true: IDataset, dataset_pred: IDataset) -> np.ndarray:
        y_true = self._get_ds_values(dataset_true)
        y_pred = self._get_ds_values(dataset_pred)

        if self.labels is None:
            self.labels = list(set(y_true) | set(y_pred))

        conf_mtx = confusion_matrix(y_true,
                                    y_pred,
                                    labels=self.labels,
                                    sample_weight=self.sample_weight,
                                    normalize=self.normalize)

        conf_mtx = conf_mtx.tolist()
        self.set_result(conf_mtx)
        return conf_mtx

    @property
    def type(self) -> MetricType:
        return MetricType.confusion_matrix

    def _get_name(self) -> str:
        return self.type_name

    def __repr__(self):
        cmtx = pd.DataFrame(self._result, index=self.labels, columns=self.labels)
        cmtx.index.name = 'True labels'
        return str(cmtx)


class Accuracy(Metric):

    def __init__(self,
                 name: str = None,
                 normalize: bool = True,
                 sample_weight: Union[List[float], Tuple[float]] = None):
        super().__init__(name=name)
        self.normalize = normalize
        self.sample_weight = sample_weight

    def __call__(self, dataset_true: IDataset, dataset_pred: IDataset) -> Union[float, dict]:
        y_true = self._get_ds_values(dataset_true)
        y_pred = self._get_ds_values(dataset_pred)

        acc = accuracy_score(y_true,
                             y_pred,
                             normalize=self.normalize,
                             sample_weight=self.sample_weight)

        acc = float(acc)
        self.set_result(acc)
        return acc

    @property
    def type(self) -> MetricType:
        return MetricType.average

    def __repr__(self):
        return str(self.result)

# TODO: Test


class MatthewsCorrcoef(Metric):

    def __init__(self,
                 name: str = None,
                 sample_weight: Union[List[float], Tuple[float]] = None):
        super().__init__(name=name)
        self.sample_weight = sample_weight

    def __call__(self, dataset_true: IDataset, dataset_pred: IDataset) -> Union[float, dict]:
        y_true = self._get_ds_values(dataset_true)
        y_pred = self._get_ds_values(dataset_pred)

        mtcc = matthews_corrcoef(y_true,
                                 y_pred,
                                 sample_weight=self.sample_weight)

        mtcc = float(mtcc)
        self.set_result(mtcc)
        return mtcc

    @property
    def type(self) -> MetricType:
        return MetricType.average

    def __repr__(self):
        return str(self.result)


class RocAuc(Metric):

    def __init__(self,
                 name: str = None,
                 labels: List[str] = None,
                 average: Literal['micro', 'macro', 'samples', 'weighted'] = 'macro',
                 sample_weight: Union[List[float], Tuple[float]] = None,
                 max_fpr: float = None,
                 multi_class: Literal['raise', 'ovr', 'ovo'] = 'raise'):
        super().__init__(name=name)
        self.labels = labels
        self.average = average
        self.sample_weight = sample_weight
        self.max_fpr = max_fpr
        self.multi_class = multi_class

    def __call__(self, dataset_true: IDataset, dataset_pred: IDataset) -> Union[float, dict]:
        y_true = dataset_true['label'].array.codes
        y_pred = dataset_pred['label'].array.codes

        if self.labels is None:
            self.labels = list(set(y_true) | set(y_pred))

        roc_auc = roc_auc_score(y_true,
                                y_pred,
                                average=self.average,
                                sample_weight=self.sample_weight,
                                max_fpr=self.max_fpr,
                                multi_class=self.multi_class,
                                labels=self.labels)

        if self.average is None:
            roc_auc_per_class = {label: float(roc_auc[i]) for i, label in enumerate(self.labels)}
            self.set_result(roc_auc_per_class)
            return roc_auc_per_class

        roc_auc = float(roc_auc)
        self.set_result(roc_auc)
        return roc_auc

    @property
    def type(self) -> MetricType:
        if self.average is None:
            return MetricType.per_class
        return MetricType.average

    def __repr__(self):
        if self.type == MetricType.per_class:
            return json.dumps(self.result, indent=2)
        return str(self.result)
