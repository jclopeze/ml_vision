#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Union, List

from ml_base.dataset import Dataset
from ml_base.metric import Metric, MetricType
from ml_base.utils.logger import get_logger

logger = get_logger(__name__)

class Evaluator():
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics
        self.results: list = None

    def evaluate(self,
                 dataset_true: Dataset,
                 dataset_pred: Dataset,
                 verbose: bool = False,
                 return_dict: bool = False) -> Union[list, dict]:
        assert set(dataset_true.items) == set(dataset_pred.items), "Invalid items in evaluation"

        self.results = [metric(dataset_true, dataset_pred) for metric in self.metrics]

        if verbose:
            print(self)

        if return_dict:
            return {metric.get_name(): res for metric, res in zip(self.metrics, self.results)}
        return self.results

    def __repr__(self):
        assert self.results is not None

        result_str = f'Results of the evaluation:\n\n'
        for metric in self.metrics:
            result_str += f'{metric.get_name()}'
            metric_type = metric.get_type()

            if metric_type is MetricType.per_class:
                result_str += ' (Per class): \n'
                result_str += str(metric.result)
            elif metric_type is MetricType.average:
                result_str += f' (Avg): {metric.result}'
            elif metric_type is MetricType.confusion_matrix:
                result_str += f'\n{metric.result}'
            else:
                raise Exception

            result_str += '\n\n'

        return result_str
