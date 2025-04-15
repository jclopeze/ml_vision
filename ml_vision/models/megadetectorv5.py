#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from datetime import datetime
from collections import defaultdict
import json
from multiprocessing import Manager
import os
import pandas as pd
from typing import Union, Final, List, Tuple

from ml_base.model import Model
from ml_base.eval import Metric
from ml_base.utils.misc import delete_dirs
from ml_base.utils.misc import parallel_exec
from ml_base.utils.logger import get_logger
from ml_base.utils.dataset import STD_DATEFORMAT
from ml_base.utils.dataset import get_random_id

from ml_vision.datasets import ImageDataset
from ml_vision.datasets import VideoDataset
from ml_vision.datasets import VisionDataset
from ml_vision.utils.classification import wildlife_filtering_using_detections, MD_LABELS
from ml_vision.utils.coords import CoordinatesType, CoordinatesFormat, CoordinatesDataType
from ml_vision.utils.coords import transform_coordinates
from ml_vision.utils.vision import VisionFields as VFields

import torch
from PytorchWildlife.models import detection as pw_detection

logger = get_logger(__name__)


class MegadetectorV5(Model):
    urls: Final = {
        'a': 'https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt',
        'b': 'https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5b.0.0.pt'
    }

    CLASS_NAMES = {
        0: MD_LABELS.ANIMAL,
        1: MD_LABELS.PERSON,
        2: MD_LABELS.VEHICLE
    }

    def __init__(self,
                 version,
                 detection_model):
        self.version = version
        self.detection_model = detection_model

    @classmethod
    def load_model(cls, version="a") -> MegadetectorV5:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if DEVICE == "cuda":
            torch.cuda.set_device(0)

        detection_model = pw_detection.MegaDetectorV5(
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

        dets_imgs_ds = MegadetectorV5Image.predict(model=self,
                                                   dataset=dataset.images_ds,
                                                   threshold=threshold)
        dets_vids_ds = MegadetectorV5Video.predict(model=self,
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
                 return_detections: bool = False
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


class MegadetectorV5Image(MegadetectorV5):

    @staticmethod
    def predict(model: MegadetectorV5,
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
            return ImageDatasetMD(annotations=None, metadata=None)

        anns_dets_df = model.inference(dataset, threshold)

        dets_ds = ImageDataset(annotations=anns_dets_df,
                               metadata=dataset.metadata.copy(),
                               root_dir=dataset.root_dir,
                               validate_filenames=False)
        return dets_ds


class MegadetectorV5Video(MegadetectorV5):

    @staticmethod
    def predict(model: MegadetectorV5,
                dataset: VideoDataset,
                threshold: float = 0.01,
                freq_video_sampling: int = 5,
                frames_folder: str = None,
                delete_frames_folder_on_finish: bool = True) -> VideoDataset:
        if dataset.is_empty:
            return VideoDataset(annotations=None, metadata=None)

        frames_ds = VideoDataset.create_frames_dataset(
            dataset, frames_folder, freq_sampling=freq_video_sampling, add_new_file_id=False)

        dets_frames_ds = MegadetectorV5Image.predict(model=model,
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
                VFields.ITEM: item,
                VFields.LABEL: label,
                VFields.BBOX: bbox,
                VFields.SCORE: score
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
                                   index=records.keys()).reset_index(names=VFields.ID)
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
        The dataset must have the field 'file_id'.
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
            If False, the 'id' will be the 'file_id' field of each item.
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
        coords_type = self._get_coordinates_type()
        if images_dims is None and out_coordinates_type != coords_type:
            images_dims = self.get_media_dims().set_index(VFields.ITEM)
        if form_id_with_filename_without_base_dir:
            form_id_with_filename_without_prefix = self.root_dir
        else:
            form_id_with_filename_without_prefix = None
        inv_labelmap = self._get_inverse_labelmap()
        results = self.get_list_of_detections(inv_labelmap,
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

    def get_list_of_detections(self,
                               inv_labelmap: dict,
                               out_coords_type: CoordinatesType = CoordinatesType.relative,
                               score_threshold: float = 0.3,
                               form_id_with_filename_without_prefix: str = None,
                               images_dims: pd.DataFrame = None,
                               add_detection_id: bool = False) -> list[dict]:
        """Gets a list with the dataset detections. Each item in the list is a dictionary as follows:

        ```
        {
            'detections': [detection],
            'id': str,   # Id of the image
            'max_detection_conf': float
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
        data : pd.DataFrame
            DataFrame with the data of the predictions dataset. The DataFrame must have the column
            'file_id'.
        inv_labelmap : dict
            Dictionary with class names to ids, in the form {'class_name': 'id'}
        out_coords_type : CoordinatesType, optional
            Determines the type of coordinates of the output, either relative or absolute,
            by default coords_utils.CoordinatesType.relative
        score_threshold : float, optional
            Threshold for which detections will be considered or ignored, by default 0.3
        form_id_with_filename_without_prefix : str, optional
                Prefix that must be removed from the items to form the id of each element.
                If None, id will be the 'file_id' field of each item.
                By default None
        images_dims : pd.DataFrame, optional
            Dataframe containing 'width' and 'height' of each image, having the item as the index,
            by default None
        add_detection_id : bool, optional
            Whether to include the `id` in the `detections` field or not, by default False

        Returns
        -------
        list of dict
            List with the detections for each image in the dataset
        """
        results = Manager().list()
        parallel_exec(
            func=ImageDatasetMD.append_detection_to_list,
            elements=self.items,
            item=lambda item: item,
            df=self.df,
            results=results,
            inv_labelmap=inv_labelmap,
            out_coords_type=out_coords_type,
            score_threshold=score_threshold,
            form_id_with_filename_without_prefix=form_id_with_filename_without_prefix,
            images_dims=images_dims,
            add_detection_id=add_detection_id)
        return [x for x in results]

    @staticmethod
    def append_detection_to_list(item,
                                 df,
                                 results,
                                 inv_labelmap,
                                 out_coords_type,
                                 score_threshold,
                                 form_id_with_filename_without_prefix,
                                 images_dims,
                                 add_detection_id=False):
        """Appends the detections of the `item` to the `results` list, performing a coordinate
        conversion if necessary

        Parameters
        ----------
        item : str
            Item for which detections will be added
        data : pd.DataFrame
            DataFrame containing the data and in which the search for detections will be carried out
        results : list
            List to which the element's detections are appended
        inv_labelmap : dict
            Reverse dictionary containing a mapping of the category names to their corresponding id
        out_coords_type : CoordinatesType
            Determines the type of coordinates of the output, either relative or absolute
        score_threshold : float
            Threshold for which detections will be considered or ignored, by default 0.3
        form_id_with_filename_without_prefix : str
            Prefix that must be removed from the items to form the id of each element.
            If None, id will be the 'file_id' field of each item.
            By default None
        images_dims : pd.DataFrame
            Dataframe containing 'width' and 'height' of each image, having the item as the index
        add_detection_id : bool, optional
            Whether to include the `id` in the `detections` field or not, by default False
        """
        rows_item = df[(df[VFields.ITEM] == item) & (df[VFields.SCORE] >= score_threshold)]
        if len(rows_item) == 0:
            return
        detections = []
        for _, record in rows_item.iterrows():
            img_width = images_dims.loc[item][VFields.WIDTH] if images_dims is not None else None
            img_height = images_dims.loc[item][VFields.HEIGHT] if images_dims is not None else None
            bbox = transform_coordinates(
                record[VFields.BBOX],
                input_format=CoordinatesFormat.x_y_width_height,
                output_format=CoordinatesFormat.x_y_width_height,
                output_coords_type=out_coords_type,
                output_data_type=CoordinatesDataType.array,
                media_width=img_width,
                media_height=img_height)
            det = {
                'category': str(inv_labelmap[record[VFields.LABEL]]),
                'bbox': bbox,
                'conf': record[VFields.SCORE]
            }
            if add_detection_id:
                det['id'] = record[VFields.ID]
            detections.append(det)
        id = (
            os.path.relpath(rows_item.iloc[0][VFields.ITEM], form_id_with_filename_without_prefix)
            if form_id_with_filename_without_prefix is not None
            else rows_item.iloc[0][VFields.FILE_ID]
        )
        results.append({
            'detections': detections,
            'id': str(id),
            'max_detection_conf': rows_item.iloc[0][VFields.SCORE],
            'file': os.path.basename(item)
        })
