#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from datetime import datetime
import json
from multiprocessing import Manager
import os
import pandas as pd
from shutil import move
import subprocess
from typing import Union, Final, Literal, List

from ml_base.model import Model
from ml_base.eval import Evaluator, Metric
from ml_base.utils.misc import get_temp_folder
from ml_base.utils.misc import download_file
from ml_base.utils.misc import parallel_exec
from ml_base.utils.logger import get_logger
from ml_base.utils.dataset import STD_DATEFORMAT
from ml_base.utils.dataset import get_random_id

from ml_vision.datasets.image import ImageDataset
from ml_vision.datasets.video import VideoDataset
from ml_vision.datasets.vision import VisionDataset
from ml_vision.utils.image import get_list_of_detections
from ml_vision.utils.image import ImageFields
from ml_vision.utils.coords import CoordinatesType
from ml_vision.utils.vision import VisionFields as VFields

logger = get_logger(__name__)


class MegadetectorV5(Model):
    urls: Final = {
        'a': 'https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt',
        'b': 'https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5b.0.0.pt'
    }
    ANIMAL: Final = 'animal'
    PERSON: Final = 'person'
    VEHICLE: Final = 'vehicle'
    EMPTY: Final = 'empty'

    def __init__(self,
                 version,
                 model_path):
        self.version = version
        self.model_path = model_path

    @classmethod
    def load_model(cls,
                   source_path: str = None,
                   version: Literal['a', 'b'] = 'a') -> MegadetectorV5:
        if source_path is None or not os.path.isfile(source_path):
            model_url = cls.urls[version]
            model_path = download_file(model_url, get_temp_folder())
            if source_path is None:
                source_path = model_path
            else:
                move(model_path, source_path)

        instance = cls(version, model_path)
        return instance

    def _execution(self, dataset: ImageDataset, threshold: float = 0.01) -> str:

        dataset_items = dataset.items
        imgs_json = os.path.join(get_temp_folder(), f"{get_random_id()}-imgs.json")
        dets_json = os.path.join(get_temp_folder(), f"{get_random_id()}-dets.json")

        with open(imgs_json, 'w') as outfile:
            json.dump(dataset_items, outfile)

        try:
            MEGADETECTOR_EXEC = os.environ['MEGADETECTOR_EXEC']
        except Exception:
            raise Exception(
                f"You must assign the environment variable MEGADETECTOR_EXEC with the path of the "
                f"detection/run_detector_batch.py script from the CameraTraps repository.")

        PYTHON_EXEC = os.environ.get('PYTHON_EXEC', 'python')
        try:
            retcode = subprocess.call([PYTHON_EXEC, '--version'],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            assert retcode == 0, f'Invalid call to {PYTHON_EXEC}'
        except:
            raise Exception(f"You must export the PYTHON_EXEC environment variable with the "
                            f"python executable on your terminal. "
                            f"E.g. 'export PYTHON_EXEC=python'")

        assert os.path.isfile(self.model_path), (f"Invalid path for Megadetector model: "
                                                 f"{self.model_path}")

        cmd = [PYTHON_EXEC, MEGADETECTOR_EXEC,
               self.model_path,                     # --detector_file
               imgs_json,                           # --image_file
               dets_json,                           # --output_file
               '--quiet',
               '--threshold', f'{threshold}'
               ]
        logger.debug(f"Running Megadetector with the command: {' '.join(cmd)}")

        retcode = subprocess.call(cmd)

        if retcode != 0:
            raise Exception(f"Megadetector executed and returned status code: {retcode}")

        os.remove(imgs_json)

        return dets_json

    def predict(self,
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
        dets_json = self._execution(dataset, threshold)

        dets_ds = ImageDatasetMD.from_json(dets_json,
                                           metadata=dataset.metadata.copy(),
                                           root_dir=dataset.root_dir)
        return dets_ds

    def classify(self,
                 dataset: ImageDataset,
                 dets_threshold: float,
                 return_detections: bool = False) -> ImageDataset:

        dets_ds = self.predict(dataset=dataset, threshold=dets_threshold)

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
        dets_item_thres = dets_df[(dets_df[ImageFields.ITEM] == item) &
                                  (dets_df[ImageFields.SCORE] >= threshold)]

        if len(dets_item_thres) > 0:
            person_dets = dets_item_thres[dets_item_thres[ImageFields.LABEL] == cls.PERSON]
            if len(person_dets) > 0:
                label = cls.PERSON
                score = person_dets[ImageFields.SCORE].max()
            else:
                label = cls.ANIMAL
                score = dets_item_thres[ImageFields.SCORE].max()
        else:
            label = cls.EMPTY
            dets_item_all = dets_df[dets_df[ImageFields.ITEM] == item]
            if len(dets_item_all) > 0:
                score = 1 - dets_item_all[ImageFields.SCORE].max()
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
        detections.map_categories(mapping_classes=dets_labels_mapping, inplace=True)

        item_to_label_score = Manager().dict()
        parallel_exec(
            func=cls._get_media_level_label_and_score,
            elements=dataset.items,
            dets_df=detections.df,
            item=lambda item: item,
            threshold=dets_threshold,
            item_to_label_score=item_to_label_score)

        # TODO: Check if it is needed to convert dataset to a media level dataset
        classif_ds = dataset.copy()
        classif_ds[ImageFields.LABEL] = lambda x: item_to_label_score[x[ImageFields.ITEM]]['label']
        classif_ds[ImageFields.SCORE] = lambda x: item_to_label_score[x[ImageFields.ITEM]]['score']

        return classif_ds


class MegadetectorV5Video(MegadetectorV5):

    def predict(self,
                dataset: VideoDataset,
                threshold: float = 0.01,
                freq_video_sampling: int = 5,
                frames_folder: str = None,
                ) -> VideoDataset:
        frames_ds = dataset.create_frames_dataset(frames_folder, freq_video_sampling)

        dets_frames_ds = super().predict(dataset=frames_ds, threshold=threshold)

        dets_vids_ds = dataset.create_object_level_dataset_using_detections(
            dets_frames_ds, use_detections_labels=True)

        return dets_vids_ds

    def classify(self,
                 dataset: VideoDataset,
                 dets_threshold: float,
                 freq_video_sampling: int = 5,
                 frames_folder: str = None,
                 return_detections: bool = False) -> VideoDataset:

        dets_ds = self.predict(dataset, dets_threshold, freq_video_sampling, frames_folder)

        classif_ds = self.classify_dataset_using_detections(
            dataset=dataset,
            detections=dets_ds,
            dets_threshold=dets_threshold)

        if return_detections:
            return classif_ds, dets_ds

        return classif_ds


class ImageDatasetMD(ImageDataset):

    @staticmethod
    def _create_records(image_dets, labelmap, records):
        detections = image_dets.get('detections', [])
        for detection in detections:
            item = image_dets['file']
            label = labelmap[detection['category']]
            bbox = ",".join([str(x) for x in detection['bbox']])
            score = detection['conf']
            id = get_random_id()

            records[id] = {
                ImageFields.ITEM: item,
                ImageFields.LABEL: label,
                ImageFields.BBOX: bbox,
                ImageFields.SCORE: score
            }

    @classmethod
    def from_json(cls,
                  source_path: str,
                  metadata: pd.DataFrame,
                  root_dir: str,
                  **kwargs) -> ImageDataset:
        """Create an ImagePredictionDataset from a JSON file that contains predictions from an
        object detection model and are in the following format:

        ```
        {
            'images':[image],
            'detection_categories': {id_cat_str: cat_name, ...},
            'info': info
        }

        image{
            'id': str,
            'max_detection_conf': float,
            'detections':[detection]
        }

        detection{
            'bbox' : [x, y, width, height],
            'category': str,
            'conf': float
        }
        ```

        Parameters
        ----------
        source_path : str
            Path of a json file that contains the detections.
        **kwargs :
            Extra named arguments that may contains the following parameters:
            * categories : list of str, str or None
                List, string or path of a CSV or a text file with the categories to be included in
                the dataset. If None, registers of all categories will be included.
                If path to a CSV file, it should have the categories in the column `0` and should
                not have a header. If path to a text file, it must have the categories separated by
                a line break. If string, it must contain the categories separated by commas.
                If empty list, labeled images will not be included. (default is None)
            * exclude_categories : list of str, str or None
                List, string or path of a CSV file or a text file with categories to be excluded.
                If it is a path of a CSV file, it should have the categories in the column `0`
                and should not have a header. If it is a path of a text file, it must have the
                categories separated by a line break. If it is a string, it must contain the
                categories separated by commas. (default is None)
            * score_threshold : float
                Threshold for which detections will be considered or ignored, by default 0.3
            * set_filename_with_id_and_extension : str
                Extension to be added to the id of each item to form the file name
                (default is None)
            * root_dir : str
                The folder path where the images are already stored.
                (default is None)

        Returns
        -------
        Dataset
            Instance of the created Dataset
        """
        assert os.path.isfile(source_path), f"{source_path} is not a valid file"

        # TODO: Test this

        detections_data = json.load(open(source_path))
        images_dets = detections_data['images']
        labelmap = detections_data['detection_categories']
        records = Manager().dict()

        parallel_exec(
            cls._create_records,
            elements=images_dets,
            image_dets=lambda image_dets: image_dets,
            labelmap=labelmap,
            records=records)

        annotations = pd.DataFrame(data=records.values(),
                                   index=records.keys()).reset_index(names=ImageFields.ID)
        prediction_dataset = cls(annotations,
                                 metadata,
                                 root_dir,
                                 **kwargs)
        prediction_dataset.compute_and_set_media_dims()

        return prediction_dataset

    def to_json(self,
                dest_path: str | None,
                out_coordinates_type: CoordinatesType = CoordinatesType.relative,
                score_threshold: float = 0.3,
                form_id_with_filename_without_base_dir: bool = False,
                add_detection_id: bool = False,
                images_dims: dict = None) -> dict:
        """Creates a JSON representation of the detections contained in the dataset.
        The dataset must have the column 'image_id'.
        Detections are provided in the following format:
        ```
        {
            'images':[image],
            'detection_categories': {'1': 'animal', '2': 'person', 'vehicle'},
            'info': info
        }

        image{
            'id': str,
            'max_detection_conf': float,
            'detections':[detection]
        }

        detection{
            'bbox' : [x, y, width, height],
            'category': str,
            'conf': float,
            'id': str, optional
        }
        ```

        Parameters
        ----------
        dest_path : str or None
            Path of the resulting JSON file. In case it is None, the file information will only be
            returned and will not be saved
        out_coordinates_type : `CoordinatesType`, optional
            Determines if the value of the output coordinates in 'bbox' will be given in absolute
            pixel values, or normalized in relation to the image size and therefore in the range
            [0,1]. By default CoordinatesType.relative
        score_threshold : float, optional
            Threshold for which detections will be considered or ignored, by default 0.3
        form_id_with_filename_without_base_dir : bool, optional
            Whether or not the 'id' of each element will be formed with the path of each item,
            removing the base directory where the images were stored (generally, the `image_dir` or
            the `dest_path` parameters of some dataset constructors).
            If False, the 'id' will be the 'image_id' field of each item.
            By default False
        add_detection_id : bool, optional
            Whether to include the `id` in the `detections` field or not, by default False
        images_dims : dict, optional
            Dictionary containing 'width' and 'height' of each image, by default None

        Returns
        -------
        dict
            Dictionary with the resulting information
        """
        df = self[[self.ANNOTATIONS_FIELDS.ITEM,
                   self.ANNOTATIONS_FIELDS.LABEL,
                   self.ANNOTATIONS_FIELDS.BBOX,
                   self.METADATA_FIELDS.MEDIA_ID,
                   self.ANNOTATIONS_FIELDS.SCORE]]
        coords_type = self._get_coordinates_type()
        if images_dims is None and out_coordinates_type != coords_type:
            images_dims = self.get_media_dims().set_index(VFields.ITEM)
        if form_id_with_filename_without_base_dir:
            form_id_with_filename_without_prefix = self.root_dir
        else:
            form_id_with_filename_without_prefix = None
        results = get_list_of_detections(df,
                                         self._get_inverse_labelmap(),
                                         out_coordinates_type,
                                         score_threshold,
                                         form_id_with_filename_without_prefix,
                                         images_dims,
                                         add_detection_id)
        final_output = {
            'images': results,
            'detection_categories': {str(k): v for k, v in self.labelmap.items()},
            'info': {
                'detection_completion_time': datetime.utcnow().strftime(STD_DATEFORMAT),
                'format_version': '1.0'
            }
        }
        if dest_path is not None:
            with open(dest_path, 'w') as f:
                json.dump(final_output, f, indent=1)
        return final_output
