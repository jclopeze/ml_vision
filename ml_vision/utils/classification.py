from collections import defaultdict
from multiprocessing import Manager
from typing import Union
from itertools import groupby

import numpy as np
import pandas as pd

from ml_base.utils.logger import get_logger
from ml_base.utils.misc import parallel_exec
from ml_vision.datasets.image import ImageDataset
from ml_vision.datasets.video import VideoDataset

logger = get_logger(__name__)

# region Image level classification


# endregion

# region Sequence level classification
def _get_seq_level_label_and_score(preds_df: pd.DataFrame,
                                   seq_id: str,
                                   pred_method: str,
                                   seq_id_to_label_and_score: dict):
    preds_seq = preds_df[preds_df['seq_id'] == seq_id]
    if pred_method == 'highest_pred':
        highest_pred = (
            preds_seq.sort_values(by='score', ascending=False)
            .iloc[0]
        )
        label = highest_pred['label']
        max_score = highest_pred['score']
    elif pred_method == 'preds_mode':
        label_modes = preds_seq['label'].mode()
        if len(label_modes) > 1:
            highest_mode = (
                preds_seq[preds_seq["label"].isin(label_modes.values)]
                .sort_values(by='score', ascending=False)
                .iloc[0]
            )
            label = highest_mode['label']
            max_score = highest_mode['score']
        else:
            label = label_modes.iloc[0]
            max_score = preds_seq[preds_seq.label == label].score.max()
    else:
        raise ValueError(f"Invalid pred_method: {pred_method}")

    seq_id_to_label_and_score[seq_id] = {
        'label': label,
        'score': max_score
    }


def get_seq_level_pred_dataset(dataset_true: ImageDataset,
                               dataset_pred: ImageDataset,
                               pred_method: str = 'highest_pred',
                               **kwargs) -> ImageDataset:
    true_df = dataset_true.as_dataframe()
    preds_df = dataset_pred.as_dataframe()

    seq_id_to_label_and_score = Manager().dict()
    seqs_ids = true_df['seq_id'].unique()
    assert '' not in seqs_ids, "Empty sequences are not allowed to generate predictions"

    parallel_exec(
        func=_get_seq_level_label_and_score,
        elements=seqs_ids,
        preds_df=preds_df,
        seq_id=lambda seq_id: seq_id,
        pred_method=pred_method,
        seq_id_to_label_and_score=seq_id_to_label_and_score)

    preds_ds = dataset_true.copy()
    preds_ds['label'] = lambda rec: seq_id_to_label_and_score[rec['seq_id']]['label']
    preds_ds['score'] = lambda rec: seq_id_to_label_and_score[rec['seq_id']]['score']

    annotations = preds_ds.annotations
    metadata = preds_ds.metadata
    root_dir = preds_ds.root_dir

    preds_ds = ImageDataset(annotations, metadata, root_dir, **kwargs)
    return preds_ds
# endregion


def get_media_level_pred_dataset(dataset_true: Union[ImageDataset, VideoDataset],
                                 dataset_pred: Union[ImageDataset, VideoDataset],
                                 pred_method: str = 'highest_pred',
                                 empty_label: str = 'empty',
                                 det_labels_map: dict = None,
                                 score_empty_label: float = 0.,
                                 seq_level_for_images: bool = False,
                                 **kwargs) -> Union[ImageDataset, VideoDataset]:
    if isinstance(dataset_true, ImageDataset):
        if seq_level_for_images:
            media_level_dataset_pred = get_seq_level_pred_dataset(
                dataset_true=dataset_true, dataset_pred=dataset_pred, pred_method=pred_method,
                empty_label=empty_label, det_labels_map=det_labels_map,
                score_empty_label=score_empty_label, **kwargs)
        else:
            pass
    elif isinstance(dataset_true, VideoDataset):
        pass
    else:
        raise Exception('Invalid type of dataset_true')

    return media_level_dataset_pred


def set_label_and_score_for_item_in_ensemble(item: str,
                                             models_to_dfs: dict,
                                             item_to_lbls_scrs: dict,
                                             model_weights: dict):
    """Assigns for `item` the score that corresponds to each label according to the predictions of
    the models in `models_to_dfs` and the weights in `model_weights`, and stores them in
    `item_to_lbls_scrs`.

    Parameters
    ----------
    item : str
        Item to be analysed. It must be contained in the column "item" of each value (`DataFrame`)
        of `models_to_dfs`
    models_to_dfs : dict
        Dictionary containing the results of the predictions made by the models,
        in the form: {`model_name`: `model_preds`, ...}
    item_to_lbls_scrs : dict
        Dictionary in which the results will be inserted,
        in the form: {`item`: [{`label_i`: `score_i`}, ...]}
    model_weights : dict
        Dictionary containing the weights of each model, which will multiply the probabilities of
        each of its predictions
    """
    label_to_scores = defaultdict(list)
    for model, df in models_to_dfs.items():
        for _, crop in df[df["item"] == item].iterrows():
            score = crop['score']
            score *= model_weights[model]
            label_to_scores[crop["label"]].append(score)
    label_to_mean = {}
    for label, scores in label_to_scores.items():
        label_to_mean[label] = np.sum(np.array(scores))
    for lbl, scor in dict(sorted(label_to_mean.items(), key=lambda x: x[1], reverse=True)).items():
        item_to_lbls_scrs[item] = {'label': lbl,
                                   'score': scor}
        break
