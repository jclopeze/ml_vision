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
from ml_base.utils.misc import parallel_exec, delete_dirs, get_temp_folder
from ml_base.utils.dataset import get_random_id
from ml_base.utils.logger import get_logger

from ml_vision.datasets import ImageDataset
from ml_vision.datasets import VideoDataset
from ml_vision.datasets import VisionDataset
from ml_vision.utils.vision import VisionFields as VFields
from ml_vision.utils.coords import transform_coordinates, CoordinatesFormat, CoordinatesDataType
from ml_vision.utils.classification import wildlife_filtering_using_detections, MD_LABELS

import torch
from PytorchWildlife.models import detection as pw_detection

logger = get_logger(__name__)


class MegadetectorV6(Model):

    CLASS_NAMES = {
        0: MD_LABELS.ANIMAL,
        1: MD_LABELS.PERSON,
        2: MD_LABELS.VEHICLE
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
            dataset.root_dir, batch_size=16, det_conf_thres=threshold)

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
                frames_folder: str = None,
                delete_frames_folder_on_finish: bool = True) -> VisionDataset:
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

        dets_imgs_ds = MegadetectorV6Image.predict(
            model=self,
            dataset=dataset.images_ds,
            threshold=threshold)
        dets_vids_ds = MegadetectorV6Video.predict(
            model=self,
            dataset=dataset.videos_ds,
            threshold=threshold,
            freq_video_sampling=freq_video_sampling,
            frames_folder=frames_folder,
            delete_frames_folder_on_finish=delete_frames_folder_on_finish)

        return type(dataset).from_datasets(dets_imgs_ds, dets_vids_ds)

    def classify(self,
                 dataset: VisionDataset,
                 dets_threshold: float,
                 freq_video_sampling: int = 5,
                 frames_folder: str = None,
                 return_detections: bool = False,
                 delete_frames_folder_on_finish: bool = True
                 ) -> Union[VisionDataset, Tuple[VisionDataset, VisionDataset]]:

        dets_ds = self.predict(
            dataset=dataset,
            freq_video_sampling=freq_video_sampling,
            frames_folder=frames_folder,
            delete_frames_folder_on_finish=delete_frames_folder_on_finish)

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
    def classify_dataset_using_detections(cls,
                                          dataset: VisionDataset,
                                          detections: VisionDataset,
                                          dets_threshold: float) -> VisionDataset:

        results_per_item = Manager().dict()
        parallel_exec(
            func=wildlife_filtering_using_detections,
            elements=dataset.items,
            dets_df=detections.df,
            item=lambda item: item,
            threshold=dets_threshold,
            results_per_item=results_per_item)

        # TODO: Check if it is needed to convert dataset to a media level dataset
        classif_ds = type(dataset)._copy_dataset(dataset)
        classif_ds[VFields.LABEL] = lambda x: results_per_item[x[VFields.ITEM]]['label']
        classif_ds[VFields.SCORE] = lambda x: results_per_item[x[VFields.ITEM]]['score']

        return classif_ds


class MegadetectorV6Image(MegadetectorV6):

    @staticmethod
    def predict(model: MegadetectorV6,
                dataset: ImageDataset,
                threshold: float = 0.01,
                move_files_to_temp_folder=True) -> ImageDataset:
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

        ds = dataset.copy()
        if move_files_to_temp_folder:
            # We need to be sure that only the files of the dataset are stored in root_dir
            temp_folder = os.path.join(get_temp_folder(), f"images-{get_random_id()}")
            ds.to_folder(
                temp_folder,
                use_labels=False,
                move_files=True,
                preserve_directory_hierarchy=True,
                update_dataset_filepaths=True)

        anns_dets_df = model.inference(ds, threshold)

        dets_ds = ImageDataset(annotations=anns_dets_df,
                               metadata=ds.metadata.copy(),
                               root_dir=ds.root_dir,
                               validate_filenames=False)

        if move_files_to_temp_folder:
            dets_ds.to_folder(
                dataset.root_dir,
                use_labels=False,
                move_files=True,
                preserve_directory_hierarchy=True,
                update_dataset_filepaths=True)

        return dets_ds


class MegadetectorV6Video(MegadetectorV6):

    @staticmethod
    def predict(model: MegadetectorV6,
                dataset: VideoDataset,
                threshold: float = 0.01,
                freq_video_sampling: int = 5,
                frames_folder: str = None,
                delete_frames_folder_on_finish: bool = True) -> VideoDataset:
        if dataset.is_empty:
            return VideoDataset(annotations=None, metadata=None)

        frames_ds = VideoDataset.create_frames_dataset(
            dataset, frames_folder, freq_sampling=freq_video_sampling, add_new_file_id=False)

        dets_frames_ds = MegadetectorV6Image.predict(model=model,
                                                     dataset=frames_ds,
                                                     threshold=threshold,
                                                     move_files_to_temp_folder=False)

        mapper = frames_ds.df.set_index(VFields.ITEM)[VFields.VID_FRAME_NUM]
        dets_frames_ds[VFields.VID_FRAME_NUM] = lambda record: mapper.loc[record[VFields.ITEM]]

        dets_vids_ds = dataset.create_object_level_dataset_using_detections(
            dets_frames_ds, use_detections_labels=True,
            fields_for_merging=[VFields.FILE_ID, VFields.VID_FRAME_NUM])

        if delete_frames_folder_on_finish:
            frames_dirs = list(set([os.path.dirname(it) for it in frames_ds.items]))
            delete_dirs(frames_dirs)
            delete_dirs(frames_ds.root_dir)

        return dets_vids_ds
