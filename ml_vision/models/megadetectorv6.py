#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from collections import defaultdict
from multiprocessing import Manager
import os
import pandas as pd
from typing import Union, Final, List, Tuple

from ml_base.model import Model
from ml_base.eval import Metric
from ml_base.utils.misc import parallel_exec, delete_dirs
from ml_base.utils.logger import get_logger

from ml_vision.datasets import ImageDataset
from ml_vision.datasets import VideoDataset
from ml_vision.datasets import VisionDataset
from ml_vision.utils.vision import VisionFields as VFields
from ml_vision.utils.coords import transform_coordinates, CoordinatesFormat, CoordinatesDataType

import torch
from PytorchWildlife.models import detection as pw_detection

logger = get_logger(__name__)


class MegadetectorV6(Model):

    ANIMAL: Final = 'animal'
    PERSON: Final = 'person'
    VEHICLE: Final = 'vehicle'
    EMPTY: Final = 'empty'

    CLASS_NAMES = {
        0: "animal",
        1: "person",
        2: "vehicle"
    }

    def __init__(self, version, detection_model):
        self.version = version
        self.detection_model = detection_model

    # TODO: Change returned type
    @classmethod
    def load_model(cls,
                   version="MDV6-yolov10-e") -> MegadetectorV6:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if DEVICE == "cuda":
            torch.cuda.set_device(0)

        detection_model = pw_detection.MegaDetectorV6(
            device=DEVICE, pretrained=True, version=version)

        instance = cls(version, detection_model)
        return instance

    def inference(self, dataset: ImageDataset, threshold: float = 0.01) -> pd.DataFrame:
        results_list = self.detection_model.batch_image_detection(
            data_path=dataset.root_dir, batch_size=16, det_conf_thres=threshold)

        data = defaultdict(list)
        for results in results_list:
            item = results['img_id']
            dets = results['detections']
            for bbox, score, class_id in zip(dets.xyxy, dets.confidence, dets.class_id):
                if score < threshold:
                    continue
                bbox_str = transform_coordinates(
                    bbox,
                    input_format=CoordinatesFormat.x1_y1_x2_y2,
                    output_format=CoordinatesFormat.x_y_width_height,
                    output_data_type=CoordinatesDataType.string)
                data[VFields.ITEM].append(item)
                data[VFields.BBOX].append(bbox_str)
                data[VFields.SCORE].append(float(score))
                data[VFields.LABEL].append(self.CLASS_NAMES[int(class_id)])

        return pd.DataFrame(data)

    def predict(self,
                dataset: VisionDataset,
                threshold: float = 0.01,
                freq_video_sampling: int = 5,
                frames_folder: str = None) -> VisionDataset:
        """Method that performs the prediction of the Megadetector on the images in `dataset`

        Parameters
        ----------
        dataset : ImageDataset
            ImagesDataset with the images on which the Megadetector will perform the prediction
        threshold : float
            Minimum score of detections to be considered in results

        Returns
        -------
        ImagePredictionDataset
            Image dataset with Megadetector predictions
        """

        if dataset.is_empty:
            logger.debug("No data to detect")
            return dataset

        dets_imgs_ds = MegadetectorV6Image.predict(model=self,
                                                   dataset=dataset.images_ds,
                                                   threshold=threshold)
        dets_vids_ds = MegadetectorV6Video.predict(model=self,
                                                   dataset=dataset.videos_ds,
                                                   threshold=threshold,
                                                   freq_video_sampling=freq_video_sampling,
                                                   frames_folder=frames_folder)

        return type(dataset).from_datasets(dets_imgs_ds, dets_vids_ds)

    def classify(self,
                 dataset: VisionDataset,
                 dets_threshold: float,
                 freq_video_sampling: int = 5,
                 frames_folder: str = None,
                 return_detections: bool = False,
                 ) -> Union[VisionDataset, Tuple[VisionDataset, VisionDataset]]:

        dets_ds = self.predict(dataset=dataset,
                               freq_video_sampling=freq_video_sampling,
                               frames_folder=frames_folder)

        classif_ds = self.classify_dataset_using_detections(
            dataset=dataset,
            detections=dets_ds,
            dets_threshold=dets_threshold)

        if return_detections:
            return classif_ds, dets_ds

        return classif_ds

    def evaluate(self,
                 dataset_true: VisionDataset,
                 metrics: List[Metric],
                 dets_threshold: float,
                 dataset_pred: VisionDataset = None,
                 verbose: bool = False,
                 return_dict: bool = False) -> Union[list, dict]:
        if dataset_pred is None:
            dataset_pred = self.classify(dataset=dataset_true, dets_threshold=dets_threshold)

        return super().evaluate(dataset_true=dataset_true,
                                metrics=metrics,
                                dataset_pred=dataset_pred,
                                verbose=verbose,
                                return_dict=return_dict)

    def train(self,
              dataset,
              epochs,
              batch_size,
              **kwargs):
        raise NotImplementedError

    @classmethod
    def _get_media_level_label_and_score(cls,
                                         dets_df: pd.DataFrame,
                                         item: str,
                                         threshold: float,
                                         item_to_label_score: dict):
        dets_item_thres = dets_df[(dets_df[VFields.ITEM] == item) &
                                  (dets_df[VFields.SCORE] >= threshold)]

        if len(dets_item_thres) > 0:
            person_dets = dets_item_thres[dets_item_thres[VFields.LABEL] == cls.PERSON]
            if len(person_dets) > 0:
                label = cls.PERSON
                score = person_dets[VFields.SCORE].max()
            else:
                label = cls.ANIMAL
                score = dets_item_thres[VFields.SCORE].max()
        else:
            label = cls.EMPTY
            dets_item_all = dets_df[dets_df[VFields.ITEM] == item]
            if len(dets_item_all) > 0:
                score = 1 - dets_item_all[VFields.SCORE].max()
            else:
                score = 1.
        item_to_label_score[item] = {
            'label': label,
            'score': score
        }

    @classmethod
    def classify_dataset_using_detections(cls,
                                          dataset: VisionDataset,
                                          detections: VisionDataset,
                                          dets_threshold: float) -> VisionDataset:
        dets_labels_mapping = {cls.VEHICLE: cls.PERSON, '*': '*'}
        detections.map_categories(category_mapping=dets_labels_mapping, inplace=True)

        item_to_label_score = Manager().dict()
        parallel_exec(
            func=cls._get_media_level_label_and_score,
            elements=dataset.items,
            dets_df=detections.df,
            item=lambda item: item,
            threshold=dets_threshold,
            item_to_label_score=item_to_label_score)

        # TODO: Check if it is needed to convert dataset to a media level dataset
        classif_ds = type(dataset)._copy_dataset(dataset)
        classif_ds[VFields.LABEL] = lambda x: item_to_label_score[x[VFields.ITEM]]['label']
        classif_ds[VFields.SCORE] = lambda x: item_to_label_score[x[VFields.ITEM]]['score']

        return classif_ds


class MegadetectorV6Image(MegadetectorV6):

    @staticmethod
    def predict(model: MegadetectorV6,
                dataset: ImageDataset,
                threshold: float = 0.01) -> ImageDataset:
        """Method that performs the prediction of the Megadetector on the images in `dataset`

        Parameters
        ----------
        dataset : ImageDataset
            ImagesDataset with the images on which the Megadetector will perform the prediction
        threshold : float
            Minimum score of detections to be considered in results

        Returns
        -------
        ImagePredictionDataset
            Image dataset with Megadetector predictions
        """
        if dataset.is_empty:
            return ImageDataset(annotations=None, metadata=None)

        anns_dets_df = model.inference(dataset, threshold)

        dets_ds = ImageDataset(annotations=anns_dets_df,
                               metadata=dataset.metadata.copy(),
                               root_dir=dataset.root_dir,
                               validate_filenames=False)
        return dets_ds


class MegadetectorV6Video(MegadetectorV6):

    @staticmethod
    def predict(model: MegadetectorV6,
                dataset: VideoDataset,
                threshold: float = 0.01,
                freq_video_sampling: int = 5,
                frames_folder: str = None,
                delete_frames_folder_on_finish: bool = True
                ) -> VideoDataset:
        frames_ds = VideoDataset.create_frames_dataset(
            dataset, frames_folder, freq_sampling=freq_video_sampling, add_new_file_id=False)

        dets_frames_ds = MegadetectorV6Image.predict(model=model,
                                                     dataset=frames_ds,
                                                     threshold=threshold)

        mapper = frames_ds.df.set_index(VFields.ITEM)[VFields.VID_FRAME_NUM]
        dets_frames_ds[VFields.VID_FRAME_NUM] = lambda record: mapper.loc[record[VFields.ITEM]]

        dets_vids_ds = dataset.create_object_level_dataset_using_detections(
            dets_frames_ds, use_detections_labels=True)

        if delete_frames_folder_on_finish:
            frames_dirs = list(set([os.path.dirname(it) for it in frames_ds.items]))
            delete_dirs(frames_dirs)
            delete_dirs(frames_ds.root_dir)

        return dets_vids_ds
