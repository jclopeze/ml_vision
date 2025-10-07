from __future__ import annotations
import abc
from typing import List

from ml_base.dataset import IDataset
from ml_base.eval import Metric, Evaluator


class IModel(abc.ABC):

    def __init__(self):
        """Constructor
        """

    @classmethod
    @abc.abstractmethod
    def load_model(cls, source_path: str) -> IModel:
        """Loads a model
        """

    @abc.abstractmethod
    def predict(self,
                dataset: IDataset,
                **kwargs) -> IDataset:
        """Predicts given a `dataset`
        """
        kwargs.get('top_k')

    @abc.abstractmethod
    def classify(self, dataset: IDataset) -> IDataset:
        """Classifies a `dataset`
        """

    def evaluate(self,
                 dataset_true: IDataset,
                 metrics: List[Metric],
                 dataset_pred: IDataset = None,
                 verbose: bool = False) -> Evaluator:
        """Evaluates a `dataset`
        """
        if dataset_pred is None:
            dataset_pred = self.classify(dataset=dataset_true)

        evaluator = Evaluator(metrics)
        return evaluator.evaluate(dataset_true=dataset_true,
                                  dataset_pred=dataset_pred,
                                  verbose=verbose)

    @abc.abstractmethod
    def train(self,
              dataset,
              epochs,
              batch_size,
              **kwargs):
        """Trains a model
        """
