#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import numpy as np
from typing import List

from ml_base.dataset import IDataset
from ml_base.metric import Precision, Recall, F1Score, ConfusionMatrix, Accuracy
from ml_base.metric import Metric, MetricType
from ml_base.utils.logger import get_logger

logger = get_logger(__name__)


class Evaluator():

    @property
    def results(self):
        return self._results

    @property
    def metrics(self):
        return self._metrics

    def __getitem__(self, metric_name):
        for i, m in enumerate(self.metrics):
            if metric_name == m.name:
                return self.results[i]
        raise Exception(f'Invalid metric name: {metric_name}')

    def __init__(self, metrics: List[Metric]):
        self._metrics = metrics
        self._results: list = None

    def evaluate(self,
                 dataset_true: IDataset,
                 dataset_pred: IDataset,
                 verbose: bool = False) -> Evaluator:
        assert set(dataset_true.items) == set(dataset_pred.items), "Invalid items in evaluation"

        self._results = [metric(dataset_true, dataset_pred) for metric in self.metrics]

        if verbose:
            print(self)

        return self

    def __repr__(self):
        assert not self.results is None

        result_str = f'Results of the evaluation:\n\n'
        for metric in self.metrics:
            result_str += f'{metric.name}'
            metric_type = metric.type

            if metric_type is MetricType.per_class:
                result_str += ' (Per class): \n'
                result_str += str(metric.result)
            elif metric_type is MetricType.average:
                result_str += f' (Avg): {metric.result}'
            elif metric_type is MetricType.confusion_matrix:
                result_str += f'\n{np.array(metric.result)}'
            else:
                raise Exception

            result_str += '\n\n'

        return result_str

    def to_list(self) -> list[dict]:
        return [m.to_dict() for m in self.metrics]

    def to_json(self, dest_path):
        with open(dest_path, 'w') as f:
            json.dump(self.to_list(), f, indent=4)

    @classmethod
    def from_json(cls, source_path):
        data = json.load(open(source_path))
        metrics = []
        for l in data:
            m = eval(l['cls'])(**{**l['attrs'], 'name': l['name']})
            m.set_result(l['result'])
            metrics.append(m)

        return cls(metrics)