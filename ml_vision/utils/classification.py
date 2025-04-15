from collections import defaultdict

import numpy as np
import pandas as pd
from typing import Final

from ml_base.utils.logger import get_logger
from ml_vision.utils.vision import Fields as VFields

logger = get_logger(__name__)


class MD_LABELS():
    ANIMAL: Final = 'animal'
    PERSON: Final = 'person'
    VEHICLE: Final = 'vehicle'
    EMPTY: Final = 'empty'


def wildlife_filtering_using_detections(dets_df: pd.DataFrame,
                                        item: str,
                                        threshold: float,
                                        results_per_item: dict):
    dets_item_thres = dets_df[(dets_df[VFields.ITEM] == item) &
                              (dets_df[VFields.SCORE] >= threshold)]

    if len(dets_item_thres) > 0:
        animal_dets = dets_item_thres[dets_item_thres[VFields.LABEL] == MD_LABELS.ANIMAL]
        if len(animal_dets) > 0:
            label = MD_LABELS.ANIMAL
            score = animal_dets[VFields.SCORE].max()
        else:
            label = MD_LABELS.PERSON
            score = dets_item_thres[VFields.SCORE].max()
    else:
        label = MD_LABELS.EMPTY
        dets_item_all = dets_df[dets_df[VFields.ITEM] == item]
        if len(dets_item_all) > 0:
            score = 1 - dets_item_all[VFields.SCORE].max()
        else:
            score = 1.
    results_per_item[item] = {
        'label': label,
        'score': score
    }


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
        Item to be analysed. It must be contained in the field "item" of each value (`DataFrame`)
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
