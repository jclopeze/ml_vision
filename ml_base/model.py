from __future__ import annotations
import abc
from typing import List, Union

from ml_base.dataset import Dataset
from ml_base.eval import Metric, Evaluator


class Model(abc.ABC):

    def __init__(self):
        """Constructor
        """

    @classmethod
    @abc.abstractmethod
    def load_model(cls, source_path: str) -> Model:
        """Loads a model
        """

    @abc.abstractmethod
    def predict(self,
                dataset: Dataset,
                **kwargs) -> Dataset:
        """Predicts given a `dataset`
        """
        kwargs.get('top_k')

    @abc.abstractmethod
    def classify(self, dataset: Dataset) -> Dataset:
        """Classifies a `dataset`
        """

    def evaluate(self,
                 dataset_true: Dataset,
                 metrics: List[Metric],
                 dataset_pred: Dataset = None,
                 verbose: bool = False,
                 return_dict: bool = False) -> Union[list, dict]:
        """Evaluates a `dataset`
        """
        if dataset_pred is None:
            dataset_pred = self.classify(dataset=dataset_true)

        evaluator = Evaluator(metrics)
        results_eval = evaluator.evaluate(dataset_true=dataset_true,
                                          dataset_pred=dataset_pred,
                                          verbose=verbose,
                                          return_dict=return_dict)
        return results_eval

    @abc.abstractmethod
    def train(self,
              dataset,
              epochs,
              batch_size,
              **kwargs):
        """Trains a model
        """
