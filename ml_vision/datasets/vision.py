#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
from multiprocessing import Manager
import math
import numpy as np
from pathlib import Path
import pandas as pd
import shutil
from typing import List, Optional, Final, Literal, Union
import time
from functools import partial
from collections import defaultdict
import uuid

import cv2

from ml_base.dataset import Dataset
from ml_base.utils.misc import parallel_exec
from ml_base.utils.misc import is_array_like
from ml_base.utils.dataset import get_random_id, Fields
from ml_base.utils.logger import get_logger
from ml_base.utils.dataset import map_category
from ml_base.utils.dataset import Fields
from ml_base.utils.misc import get_temp_folder
from ml_base.utils.dataset import get_media_name_with_prefix

from ml_vision.utils.vision import VisionFields as VFields
from ml_vision.utils.coords import CoordinatesType
from ml_vision.utils.coords import CoordinatesFormat
from ml_vision.utils.coords import transform_coordinates_to_absolute_str
from ml_vision.utils.coords import transform_coordinates
from ml_vision.utils.coords import get_coordinates_type_from_coords
from ml_vision.utils.image import set_image_dims
from ml_vision.utils.image import draw_detections_of_image
from ml_vision.utils.image import crop_bboxes_on_image
from ml_vision.utils.image import get_bbox_from_json_record
from ml_vision.utils.video import frames_to_video, get_image_id_for_frame


logger = get_logger(__name__)


class VisionDataset(Dataset):
    class MetadataFields(Dataset.MetadataFields):
        """Field names allowed in the creation of image datasets."""
        MEDIA_ID: Final = VFields.MEDIA_ID
        SEQ_ID: Final = VFields.SEQ_ID
        WIDTH: Final = VFields.WIDTH
        HEIGHT: Final = VFields.HEIGHT
        FILE_TYPE: Final = VFields.FILE_TYPE
        PARENT_MEDIA_ID: Final = VFields.PARENT_MEDIA_ID

        TYPES = {
            **Dataset.MetadataFields.TYPES,
            MEDIA_ID: str,
            SEQ_ID: str,
            WIDTH: float,
            HEIGHT: float,
            PARENT_MEDIA_ID: str
        }

    class AnnotationFields(Dataset.AnnotationFields):
        BBOX: Final = VFields.BBOX
        VID_FRAME_NUM: Final = VFields.VID_FRAME_NUM

        TYPES = {
            **Dataset.AnnotationFields.TYPES,
            VID_FRAME_NUM: int
        }

    EMPTY_LABEL: Final = 'empty'
    FIELDS_TO_REMOVE_IN_MEDIA_LEVEL_DS: Final = [
        VFields.BBOX, VFields.SCORE, VFields.VID_FRAME_NUM]

    FILES_EXTS: Final = [".avi", ".mp4", ".jpg", ".png", ".jpeg"]

    DETS_FIELDS_TO_USE_IN_OBJ_LEVEL_DS: Final = {VFields.BBOX,
                                                 VFields.MEDIA_ID,
                                                 VFields.VID_FRAME_NUM,
                                                 VFields.PARENT_MEDIA_ID}

    # region PROPERTIES

    @property
    def images_ds(self) -> VisionDataset:
        return self.filter_by_column(self.MetadataFields.FILE_TYPE, ImageDataset.FILE_TYPE,
                                     mode='include', inplace=False)

    @property
    def videos_ds(self) -> VisionDataset:
        return self.filter_by_column(self.MetadataFields.FILE_TYPE, VideoDataset.FILE_TYPE,
                                     mode='include', inplace=False)

    # endregion

    # region CONSTRUCTOR

    def __init__(self: Dataset,
                 annotations: pd.DataFrame,
                 metadata: pd.DataFrame,
                 root_dir: str = None,
                 use_partitions: bool = True,
                 **kwargs) -> Dataset:
        super().__init__(annotations, metadata, root_dir, use_partitions, **kwargs)

        if not self.is_empty and not VFields.FILE_TYPE in self.fields:
            def _get_filetype(record):
                if Path(record[Fields.ITEM]).suffix.lower() in ImageDataset.FILES_EXTS:
                    return ImageDataset.FILE_TYPE
                return VideoDataset.FILE_TYPE
            self[VFields.FILE_TYPE] = lambda record: _get_filetype(record)

    # endregion

    def to_json(self,
                dest_path: str,
                include_annotations_info: bool = True):
        # TODO: Test and document
        # FIXME: Add MEDIA_ID field to anns
        anns = self.annotations[[VFields.ID]]
        categories = [{'id': int(k), 'name': v} for k, v in self.labelmap.items()]
        inv_labelmap = self._get_inverse_labelmap()
        anns['category_id'] = anns[VFields.LABEL].apply(lambda x: int(inv_labelmap[x]))
        if VFields.BBOX in self.annotations.columns:
            anns[VFields.BBOX] = self.annotations[VFields.BBOX].apply(
                lambda x: [float(c) for c in x.split(',')] if x else np.NaN)

        metad = self.metadata
        metad[VFields.ITEM] = metad[VFields.ITEM].apply(self._format_item_for_storage)
        output = {
            'images': metad.to_dict('records')
        }
        if include_annotations_info:
            output['annotations'] = anns.to_dict('records')
            output['categories'] = categories
        if os.path.isdir(dest_path):
            dest_path = os.path.join(dest_path, "dataset.json")
        with open(dest_path, 'w') as f:
            json.dump(output, f, indent=1)

    #   region SET_* METHODS

    def compute_and_set_media_dims(self, dims_correction: bool = False):
        media_dims = self.get_media_dims(dims_correction=dims_correction)
        self[[VFields.WIDTH, VFields.HEIGHT]] = media_dims
    #   endregion

    def get_media_dims(self, dims_correction: bool = False) -> pd.DataFrame:
        """Get the media dimensions of the elements in the dataset, in the form of a DataFrame having
        the item as the index.
        In case there are elements without assigned dimensions, this function will try to read the dimensions
        of the stored files.

        Parameters
        ----------
        dims_correction :  bool, optional
            Whether to correct the dimensions of the media or not, i.e., calculate the dimensions of all the
            media by obtaining it from the files. By default False

        Returns
        -------
        pd.DataFrame
            Dataframe containing columns `item`, `width` and `height`
        """
        metadata = self.metadata
        if (VFields.WIDTH not in self.metadata or VFields.HEIGHT not in self.metadata
                or dims_correction):
            items_wo_dims = self.items
        else:
            media_wo_dims = metadata[
                (metadata[VFields.WIDTH].isna()) | (metadata[VFields.WIDTH] == '')
                | (metadata[VFields.HEIGHT].isna()) | (metadata[VFields.HEIGHT] == '')]
            items_wo_dims = list(media_wo_dims[VFields.ITEM].values)

        if len(items_wo_dims) > 0:
            dims_of_media_wo_dims_df = self._get_media_dims_of_items(items_wo_dims)
            if len(dims_of_media_wo_dims_df) != len(metadata):
                media_wo_dims = dims_of_media_wo_dims_df[VFields.ITEM].values
                media_w_dims_info = metadata[~metadata[VFields.ITEM].isin(media_wo_dims)]
                dims_of_media_w_dims_info = media_w_dims_info[[VFields.ITEM,
                                                               VFields.HEIGHT,
                                                               VFields.WIDTH]]
                media_dims = pd.concat([dims_of_media_wo_dims_df, dims_of_media_w_dims_info],
                                       ignore_index=True)
            else:
                media_dims = dims_of_media_wo_dims_df
        else:
            media_dims = metadata[[VFields.ITEM, VFields.HEIGHT, VFields.WIDTH]]

        return media_dims

    # region Factory methods

    def create_media_level_ds(self: VisionDataset) -> VisionDataset:
        """Method that create an media-level dataset from the current
        object-level dataset, assigning only one annotation label to each media,
        accordingly to the parameter `keep`, that indicates which label should be taken

        Parameters
        ----------
        how : str one of {'first', 'last'}, optional
            Specify which of the annotations should be considered to obtain the label of each media
            - `first` : Drop duplicates except for the first occurrence.
            - `last` : Drop duplicates except for the last occurrence.
            labels.
            By default 'first'

        Returns
        -------
        VisionDataset
            VisionDataset of type `classification`
        """
        if self.is_empty:
            logger.debug("No data to create a media level dataset")
            return type(self)(annotations=None, metadata=None)

        anns = self.get_annotations(remove_fields=self.FIELDS_TO_REMOVE_IN_MEDIA_LEVEL_DS)
        anns = anns.drop_duplicates(self._key_field_metadata, keep='first', ignore_index=True)
        instance = type(self)(annotations=anns,
                              metadata=self.metadata.copy(),
                              root_dir=self.root_dir,
                              avoid_initialization=True)
        return instance

    def create_object_level_dataset_using_detections(self,
                                                     detections: VisionDataset,
                                                     use_partitions: bool = False,
                                                     use_detections_labels: bool = False,
                                                     use_detections_scores: bool = True,
                                                     fields_for_merging: str = None
                                                     ) -> VisionDataset:
        """Function that creates a dataset with object-level annotations from two instances:
        one with the predictions of an object detector (e.g. Megadetector) with object-level
        'annotations' and generic classes (e.g. Animal, Person), and other with
        media-level annotations and specific classes (e.g. Canis latrans).
        For each media the bounding boxes of the object detector are taken and each one is assigned the
        given media-level class, to form object-level annotations with specific classes.

        Parameters
        ----------
        detections : VisionDataset
            Instance of a VisionDataset. It must contain the media id column
        use_partitions : bool, optional
            Whether to inherit the partitions from the original dataset or not, by default False
        min_score_detections : float, optional
            Minimum score that the detections in `detections` must have. The rest will be
            discarded, by default 0.1
        **kwargs
            Extra named arguments passed to the `VisionDataset` constructor

        Returns
        -------
        VisionDataset
            Resulting object detection dataset
        """

        if self.is_empty:
            logger.debug("No data to create an object level dataset")
            return type(self)(annotations=None, metadata=None)

        fields_for_merging = fields_for_merging or [self.MetadataFields.MEDIA_ID]
        for fld in fields_for_merging:
            assert set(self[fld].values) & set(detections[fld].values)

        dets_fields_to_use = detections.DETS_FIELDS_TO_USE_IN_OBJ_LEVEL_DS
        if use_detections_labels:
            dets_fields_to_use |= {VFields.LABEL}
        if use_detections_scores:
            dets_fields_to_use |= {VFields.SCORE}
        dets_fields_to_use &= set(detections.fields)
        dets_df = detections[list(dets_fields_to_use)]

        anns_fields_to_use = set(self.fields) - dets_fields_to_use | {*fields_for_merging}
        anns_df = self[list(anns_fields_to_use)]

        obj_level_df = pd.merge(left=dets_df, right=anns_df, how='inner', on=fields_for_merging)
        obj_level_df[VFields.ID] = obj_level_df[VFields.ID].apply(lambda _: get_random_id())

        obj_level_ds = type(self).from_dataframe(obj_level_df,
                                                 root_dir=self.root_dir,
                                                 validate_filenames=False,
                                                 use_partitions=use_partitions)
        return obj_level_ds

    def create_crops_dataset_using_detections(self,
                                              detections: VisionDataset,
                                              use_partitions: bool = False,
                                              use_detections_labels: bool = False,
                                              use_detections_scores: bool = False,
                                              dest_path: str = None,
                                              **kwargs) -> VisionDataset:
        if self.is_empty:
            logger.debug("No data to create a cropped dataset")
            return type(self)(annotations=None, metadata=None)

        obj_level_ds = self.create_object_level_dataset_using_detections(
            detections=detections,
            use_partitions=use_partitions,
            use_detections_labels=use_detections_labels,
            use_detections_scores=use_detections_scores,
            **kwargs)
        crops_ds = obj_level_ds.create_crops_dataset(
            dest_path=dest_path,
            use_partitions=use_partitions,
            **kwargs)

        return crops_ds

    def create_crops_dataset(self,
                             dest_path: str = None,
                             use_partitions: bool = False,
                             allow_label_empty: bool = False,
                             force_crops_creation: bool = False,
                             dims_correction: bool = False,
                             bottom_offset: Union[int, float] = 0,
                             prefix_field: str = None,
                             **kwargs) -> VisionDataset:
        if self.is_empty:
            logger.debug("No data to create a cropped dataset")
            return type(self)(annotations=None, metadata=None)

        imgs_ds = self.images_ds
        vids_ds = self.videos_ds

        if len(imgs_ds) > 0:
            imgs_crops_ds = ImageDataset.create_crops_dataset(
                dataset=imgs_ds,
                dest_path=dest_path,
                use_partitions=use_partitions,
                allow_label_empty=allow_label_empty,
                force_crops_creation=force_crops_creation,
                dims_correction=dims_correction,
                prefix_field=prefix_field,
                bottom_offset=bottom_offset)
            if len(vids_ds) == 0:
                return imgs_crops_ds

        if len(vids_ds) > 0:
            frames_folder = kwargs.get('frames_folder')
            force_frames_creation = kwargs.get('force_frames_creation', False)
            delete_frames_folder_on_finish = kwargs.get('delete_frames_folder_on_finish', True)

            vids_crops_ds = VideoDataset.create_crops_dataset(
                dataset=vids_ds,
                frames_folder=frames_folder,
                dest_path=dest_path,
                use_partitions=use_partitions,
                allow_label_empty=allow_label_empty,
                force_crops_creation=force_crops_creation,
                force_frames_creation=force_frames_creation,
                bottom_offset=bottom_offset,
                prefix_field=prefix_field,
                delete_frames_folder_on_finish=delete_frames_folder_on_finish)
            if len(imgs_ds) == 0:
                return vids_crops_ds

        df = pd.concat([imgs_crops_ds.df, vids_crops_ds.df], ignore_index=True)
        return type(self).from_dataframe(df, root_dir=dest_path)
        return imgs_crops_ds + vids_crops_ds

    # endregion

    # TODO: include fields: use_detection_labels, use_detections_scores
    def draw_bounding_boxes(self,
                            include_labels: bool = False,
                            include_scores: bool = False,
                            blur_people: bool = False,
                            thickness: int = None,
                            **kwargs):
        if self.is_empty:
            logger.debug("No data to draw bounding boxes")
            return

        imgs_ds = self.images_ds
        if len(imgs_ds) > 0:
            ImageDataset.draw_bounding_boxes(
                dataset=imgs_ds,
                include_labels=include_labels,
                include_scores=include_scores,
                blur_people=blur_people,
                thickness=thickness)

        vids_ds = self.videos_ds
        if len(vids_ds) > 0:
            freq_sampling = kwargs.get('freq_sampling')
            frames_folder = kwargs.get('frames_folder')
            assert not freq_sampling is None

            VideoDataset.draw_bounding_boxes(
                dataset=vids_ds,
                freq_sampling=freq_sampling,
                frames_folder=frames_folder,
                include_labels=include_labels,
                include_scores=include_scores,
                blur_people=blur_people,
                thickness=thickness)

    # region Auxiliar methods for images and video creation

    # endregion

    #       region OTHER

    def _get_media_dims_of_items(self, items: List[str]) -> pd.DataFrame:
        """Determines the dimensions of `items`

        Parameters
        ----------
        items : list of str
            List of items paths

        Returns
        -------
        pd.DataFrame
            Dataframe containing 'width' and 'height' columns and the item path as index
        """
        logger.debug("Getting dimensions from stored files...")

        images_ds = self.images_ds.filter_by_column(
            self.MetadataFields.ITEM, items, mode='include', inplace=False)
        if len(images_ds) > 0:
            images_dict = Manager().dict()
            parallel_exec(
                func=set_image_dims,
                elements=items,
                image=lambda item: item,
                images_dict=images_dict)

            imgs_df = pd.DataFrame(data=images_dict.values(),
                                   index=images_dict.keys()).reset_index(names=VFields.ITEM)
        else:
            imgs_df = pd.DataFrame()

        videos_ds = self.videos_ds.filter_by_column(
            self.MetadataFields.ITEM, items, mode='include', inplace=False)
        if len(videos_ds) > 0:
            raise NotImplementedError
        else:
            vids_df = pd.DataFrame()

        df = pd.concat([imgs_df, vids_df], ignore_index=True)

        return df

    #       endregion

    #   region METHODS FOR COORDINATES

    def _get_coordinates_type(self) -> Optional[CoordinatesType]:
        """Determines if the dataset has absolute or relative coordinates

        Returns
        -------
        str
            Either `CoordinatesType.relative` ('relative'), `CoordinatesType.absolute`
            ('absolute'), or None in case the type cannot be determined from the coordinates

        Raises
        ------
        ValueError
            In case the data type of the coordinates is invalid
        """
        if len(self) == 0:
            return None
        bbox = self.df.iloc[0][VFields.BBOX]
        if type(bbox) == str:
            [coord1, coord2, coord3, coord4] = [float(x) for x in bbox.split(',')]
        elif is_array_like(bbox):
            [coord1, coord2, coord3, coord4] = [x for x in bbox]
        else:
            raise ValueError(f"'{type(bbox)}' is not a valid type for a bounding box.")
        return get_coordinates_type_from_coords(coord1, coord2, coord3, coord4)

    #   endregion

    def batch_iter(self, batch_size):
        items = self.items
        n_items = len(items)
        n_batches = math.ceil(n_items / batch_size)
        for i in range(n_batches):
            _items = items[i * batch_size: (i+1) * batch_size]
            new_ds = self.filter_by_column(Fields.ITEM, _items, inplace=False)
            yield new_ds

# region ImageDataset


class ImageDataset(VisionDataset):
    class MetadataFields(VisionDataset.MetadataFields):
        """Field names allowed in the creation of image datasets."""
        pass

    class AnnotationFields(VisionDataset.AnnotationFields):
        pass

    FILES_EXTS: Final = [".jpg", ".png", ".jpeg"]
    DEFAULT_EXT: Final = ".jpg"
    FILE_TYPE: Final = "image"
    DETS_FIELDS_TO_USE_IN_OBJ_LEVEL_DS = {*VisionDataset.DETS_FIELDS_TO_USE_IN_OBJ_LEVEL_DS,
                                          VFields.MEDIA_ID}

    # region CONSTRUCTOR

    def __init__(self: Dataset,
                 annotations: pd.DataFrame,
                 metadata: pd.DataFrame,
                 root_dir: str = None,
                 use_partitions: bool = True,
                 **kwargs) -> Dataset:
        metadata[VFields.FILE_TYPE] = ImageDataset.FILE_TYPE
        super().__init__(annotations, metadata, root_dir, use_partitions, **kwargs)

    # endregion

    # region PUBLIC API METHODS

    # TODO: add the parameter use_bboxes
    @staticmethod
    def create_crops_dataset(dataset: VisionDataset,
                             dest_path: str = None,
                             use_partitions: bool = False,
                             allow_label_empty: bool = False,
                             force_crops_creation: bool = False,
                             dims_correction: bool = False,
                             bottom_offset: Union[int, float] = 0,
                             prefix_field: str = None,
                             **_) -> ImageDataset:
        """Method that generates crops with the coordinates of the bounding boxes from the
        annotations of a dataset of type `object detection`, and assigns the labels to that
        new images in order to create a dataset of type `classification`

        Parameters
        ----------
        dest_path : str, optional
            Folder in which the images created from the crops of the bouding boxes are saved.
            If None, the images will be saved in the folder `./crops_images`.
            By default None
        use_partitions : bool, optional
            Whether to use the partitions from the original dataset or not, by default False
        **kwargs
            Extra named arguments passed to the `ImageDataset` constructor and also may include the
            parameters:
            * allow_label_empty : bool
                Whether to allow annotations with label 'empty' or not, by default False
            * force_crops_creation : bool
                Whether to force the creation of the crops or not, by default False

        Returns
        -------
        ImageDataset
            Instance of the created `classification` dataset

        Raises
        ------
        Exception
            in case the original dataset is not of type `object detection`
        """
        assert VFields.BBOX in dataset.fields, "The dataset must be of detection type"
        assert prefix_field is None or prefix_field in dataset.fields

        logger.debug("Creating classification dataset from detection bounding boxes")

        if dest_path is None:
            dest_path = os.path.join(get_temp_folder(), f"{get_random_id()}")
        os.makedirs(dest_path, exist_ok=True)

        dataset.compute_and_set_media_dims(dims_correction=dims_correction)

        crops_exist = None
        crops_paths = defaultdict(list)
        bboxes = defaultdict(list)
        id_to_new_item = {}
        id_to_parent_file_id = {}

        for record in dataset.records:
            if record[VFields.LABEL] == dataset.EMPTY_LABEL and not allow_label_empty:
                continue

            crop_item = f"{record[VFields.ID]}{ImageDataset.DEFAULT_EXT}"
            crop_path = os.path.join(dest_path, crop_item)
            if prefix_field is not None:
                crop_path = get_media_name_with_prefix(record, prefix_field, '_', crop_path)
            if crops_exist is None:
                crops_exist = os.path.isfile(crop_path)

            x1, y1, x2, y2 = transform_coordinates(
                bbox=record[VFields.BBOX],
                output_format=CoordinatesFormat.x1_y1_x2_y2,
                media_width=record[VFields.WIDTH],
                media_height=record[VFields.HEIGHT])

            crops_paths[record[VFields.ITEM]].append(crop_path)
            bboxes[record[VFields.ITEM]].append((x1, y1, x2, y2))
            id_to_new_item[record[VFields.ID]] = crop_path
            # In case it is a dataset of frames, use the media_id from the original video
            if VFields.PARENT_MEDIA_ID in record:
                id_to_parent_file_id[record[VFields.ID]
                                     ] = record[VFields.PARENT_MEDIA_ID]
            else:
                id_to_parent_file_id[record[VFields.ID]] = record[VFields.MEDIA_ID]

        if not crops_exist or force_crops_creation:
            _destpath = os.path.abspath(dest_path)
            logger.info(f"Generating {len(id_to_new_item)} crops in folder {_destpath}")
        else:
            logger.info(f"Using already created crops in {dest_path}")
        bboxes_coords_inside_crops = Manager().dict()
        crops_dims = Manager().dict()
        parallel_exec(
            crop_bboxes_on_image,
            elements=dataset.items,
            source_path=lambda item: item,
            dest_paths=lambda item: crops_paths[item],
            bboxes=lambda item: bboxes[item],
            bboxes_coords_inside_crops=bboxes_coords_inside_crops,
            crops_dims=crops_dims,
            bottom_offset=bottom_offset,
            force_creation=force_crops_creation)

        bboxes_coords_inside_crops = dict(bboxes_coords_inside_crops)
        crops_dims_df = pd.DataFrame(data=crops_dims.values(),
                                     index=crops_dims.keys()).reset_index(names=VFields.ITEM)
        crops_ds = dataset.copy()
        crops_ds[VFields.ITEM] = lambda rec: id_to_new_item[rec[VFields.ID]]
        media_id_mapper = dataset._add_media_id_field_to_dataframe(crops_ds[[VFields.ITEM]])
        crops_ds[VFields.PARENT_MEDIA_ID] = lambda rec: id_to_parent_file_id[rec[VFields.ID]]
        crops_ds[VFields.BBOX] = lambda rec: bboxes_coords_inside_crops[rec[VFields.ITEM]]
        crops_ds[[VFields.WIDTH, VFields.HEIGHT]] = crops_dims_df
        crops_ds[VFields.ID] = lambda _: get_random_id()
        crops_ds[VFields.MEDIA_ID] = media_id_mapper
        crops_ds[VFields.FILE_TYPE] = ImageDataset.FILE_TYPE
        crops_ds._split(use_partitions=use_partitions)
        crops_ds._set_root_dir(dest_path)

        return crops_ds

    # TODO: include fields: use_detection_labels, use_detections_scores
    @staticmethod
    def draw_bounding_boxes(dataset: VisionDataset,
                            include_labels: bool = False,
                            include_scores: bool = False,
                            blur_people: bool = False,
                            thickness: int = None):
        assert VFields.BBOX in dataset.fields, "Invalid dataset for drawing bounding boxes"

        dims = dataset.get_media_dims().set_index(VFields.ITEM)
        dataset[VFields.BBOX] = (
            lambda rec: transform_coordinates_to_absolute_str(
                bbox=rec[VFields.BBOX],
                media_width=dims.loc[rec[VFields.ITEM]][VFields.WIDTH],
                media_height=dims.loc[rec[VFields.ITEM]][VFields.HEIGHT]))

        dets_df = dataset.df

        if not include_labels and VFields.LABEL in dets_df.columns:
            dets_df = dets_df.drop(VFields.LABEL, axis=1)
        if not include_scores and VFields.SCORE in dets_df.columns:
            dets_df = dets_df.drop(VFields.SCORE, axis=1)

        parallel_exec(
            func=draw_detections_of_image,
            elements=dataset.items,
            item=lambda item: item,
            detections=dets_df,
            blur_people=blur_people,
            thickness=thickness)

    # endregion

    # region PRIVATE API METHODS

    #   region DATASET CALLBACKS

    @classmethod
    def _get_dataframe_from_json(cls,
                                 source_path: str,
                                 **kwargs) -> pd.DataFrame:
        """Build an ImageDataset from a json file.

        Parameters
        ----------
        source_path: str
            Path to a JSON file in COCO format that contains the information of the dataset.
        categories : list of str, str or None, optional
            List, string or path of a CSV or a text file with the categories to be included in the
            dataset.
            If None, registers of all categories will be included.
            If path to a CSV file, it should have the categories in the column `0` and should not
            have a header.
            If path to a text file, it must have the categories separated by a line break.
            If string, it must contain the categories separated by commas.
            If empty list, labeled images will not be included.
        **kwargs
            Extra named arguments that may contains the following parameters:
            * mapping_classes : str or dict
                Dictionary or path to a CSV file containing a mapping that will rename the
                categories present, either to group them into super-categories or to match those in
                other datasets.
                In the case of a dictionary, the `key` of each entry will be the current name
                of the category, and the `value` will be the new name that will be given to that
                category.
                In the case of the path of a CSV file, the file must contain two columns and
                have no header. The column `0` will be the current name of the category and the
                column `1` will be the new name (or the `id` of the category) that will be given to
                that category.
                In both cases you can use the wildcard `*` (as the `key` or in the column `0`)
                to indicate 'all other current categories in the data set'.
                E.g., {'Homo sapiens': 'Person', '*': 'Animal'} will designate the Homo
                sapiens category as 'Person' and the rest of the categories as 'Animal'.
                categories.
            * exclude_categories : list of str, str or None
                List, string or path of a CSV file or a text file with categories to be excluded.
                If it is a path of a CSV file, it should have the categories in the column `0`
                and should not have a header. If it is a path of a text file, it must have the
                categories separated by a line break. If it is a string, it must contain the
                categories separated by commas. (default is None)
            * mapping_fields : dict
                Dictionary with a mapping of the specific field names contained in a dataset,
                to the standard names for the specific type of `ImageDataset`
                (e.g., from a JSON file in COCO format).
                E.g., {'file_id': 'id', 'datetime': 'date_captured', 'image_width': 'width'}
                for a JSON file with the field name 'file_id' for the id of the images, the
                field name 'datetime' for the date and time the photo was captured and the
                field name 'image_width' for the width of images (default is {})
            * set_filename_with_id_and_extension : str
                Extension to be added to the id of each item to form the file name
                (default is None)
            * include_bboxes_with_label_empty : bool
                Whether to allow annotations with label 'empty' or not.
                (default is False)
            * mapping_classes_from_col : str or int
                Name or position (0-based) of the column to be used as 'from' in the mapping,
                in case of `mapping_classes` is a CSV. By default None
            * mapping_classes_to_col : str or int
                Name or position (0-based) of the column to be used as 'to' in the mapping,
                in case of `mapping_classes` is a CSV. By default None
            * mapping_classes_filter_expr : Callable, optional
                A Callable that will be used to filter the CSV records in which the mapping is found,
                in case of `mapping_classes` is a CSV. By default None
            * media_base_url : str
                URL where the images of the collection are located. If it is not set, an attempt
                will be made to obtain this URL with the collection name.
            * dest_path : str
                Folder in which images will be downloaded
            * collection : str
                Name of the collection of the dataset
            * collection_year : int
                Year of collection of the dataset
            * collection_version : str
                Version of collection of the dataset

        Returns
        -------
        (pd.DataFrame, dict)
            Tuple of DataFrame object and info dict
        """
        include_bboxes_with_label_empty = kwargs.get("include_bboxes_with_label_empty", False)
        fname_w_id_and_ext = kwargs.get("set_filename_with_id_and_extension")

        json_handler = ImagesJsonHandler(source_path)

        # TODO: refactor this. This is to ensure that items does not repeat among differents images
        imgs_ids = json_handler.imgs.keys()
        img_id_to_item = {
            img_id: cls._get_filename(json_handler.loadImgs(img_id), fname_w_id_and_ext)
            for img_id in imgs_ids
        }

        # region Annotations data
        if len(json_handler.imgToAnns) > 0:
            annotations = (
                pd.DataFrame([{VFields.MEDIA_ID: img_id, **ann}
                             for img_id, anns in json_handler.imgToAnns.items() for ann in anns]))
            if not VFields.LABEL in annotations.columns:
                annotations[VFields.LABEL] = (
                    annotations['category_id']
                    .apply(lambda x: json_handler.cats[x]['name']))
            if VFields.BBOX in annotations.columns:
                _get_bbox_from_json_rec = partial(
                    get_bbox_from_json_record,
                    include_bboxes_with_label_empty=include_bboxes_with_label_empty)
                annotations[VFields.BBOX] = annotations.apply(_get_bbox_from_json_rec, axis=1)

            annotations[VFields.ITEM] = (
                annotations[VFields.MEDIA_ID].apply(lambda x: img_id_to_item[x]))

        elif len(img_id_to_item) > 0:
            # Dataset with images but no annotations (e.g., a test dataset)
            annotations = pd.DataFrame([{VFields.ITEM: item,
                                         VFields.MEDIA_ID: img_id,
                                         VFields.ID: get_random_id()}
                                        for img_id, item in img_id_to_item.items()])
        else:
            raise Exception("No images or annotations were found in the dataset.")
        # endregion

        # region Metadata
        metadata = pd.DataFrame([{**img} for img in json_handler.imgs.values()])
        if len(metadata) > 0:
            logger.debug(f"{len(metadata)} images found with {len(annotations)} annotations")
        else:
            logger.debug(f"No images found for the dataset")
        # endregion

        df = annotations.merge(metadata, how='left', on=VFields.MEDIA_ID)

        return df

    #   endregion

    # endregion


class ImagesJsonHandler():
    """ JSON File Handler.
    """

    def __init__(self, json_path, cat_mappings=None):
        """Constructor of JSON handler class for reading annotations.

        Parameters
        ----------
        json_path : str
            Location of annotation file
        cat_mappings : dict, optional
            Dictionary containing category mappings, by default an empty dictionary
        """
        cat_mappings = {} if cat_mappings is None else cat_mappings
        dataset = json.load(open(json_path, 'r'))
        self.cats = {}
        self.catsIds = []
        self.catsNames = []
        self.imgs = {}
        self.imgToAnns = defaultdict(list)
        self.dataset = dataset
        self.cat_mappings = cat_mappings or {}
        self.createIndex()

    def createIndex(self):
        """Create index
        """
        for ann in self.dataset.get('annotations', []):
            # Convert VFields.MEDIA_ID to str
            self.imgToAnns[str(ann['image_id'])].append(
                {**ann, VFields.MEDIA_ID: str(ann['image_id'])})
        for image in self.dataset.get('images', []):
            # Convert 'id' of images to str
            self.imgs[str(image['id'])] = {**image, VFields.MEDIA_ID: str(image['id'])}
        for cat in self.dataset.get('categories', []):
            catNm = cat['name']
            cat['name'] = map_category(cat_name=catNm, cat_mappings=self.cat_mappings)
            self.cats[cat['id']] = cat
        self.catsIds = list(self.cats.keys())
        self.catsNames = [cat["name"] for cat in self.cats.values()]

    def getCatIds(self, catNms=[]) -> Optional[List[str]]:
        """Get category ids from category names.

        Parameters
        ----------
        catNms : list of str or None, optional
            List of category names to get the ids (default is [])

        Returns
        -------
        list of int
            List of category ids.
        """
        if catNms is None:
            return None
        else:
            catNms = [cat_nm for cat_nm in catNms]
            return [cat_id for cat_id, cat in self.cats.items() if cat['name'] in catNms]

    def loadImgs(self, ids=[]):
        """Load images with the specified ids.

        Parameters
        ----------
        ids : str or list of str, optional
            Image id or list of image ids to filter (default is [])

        Returns
        -------
        list of dict or dict
            List of image dicts, if `ids` is an array-like object, a single image dict otherwise
        """
        if is_array_like(ids):
            return [self.imgs[id] for id in ids]
        else:
            return self.imgs[ids]

# endregion


# region VideoDataset
class VideoDataset(VisionDataset):
    """Represent a VideoDataset specification."""

    class MetadataFields(VisionDataset.MetadataFields):
        pass

    class AnnotationFields(VisionDataset.AnnotationFields):
        pass

    """
    MEDIA FIELD NAMES FOR VIDEOS
    """
    FILES_EXTS: Final = [".avi", ".mp4"]
    DEFAULT_EXT: Final = ".mp4"
    FILE_TYPE = "video"

    # region CONSTRUCTOR

    def __init__(self: Dataset,
                 annotations: pd.DataFrame,
                 metadata: pd.DataFrame,
                 root_dir: str = None,
                 use_partitions: bool = True,
                 **kwargs) -> Dataset:
        metadata[VFields.FILE_TYPE] = VideoDataset.FILE_TYPE
        super().__init__(annotations, metadata, root_dir, use_partitions, **kwargs)

    # endregion
    @staticmethod
    def create_videos_from_frames_ds(frames_ds,
                                     dest_folder: str,
                                     freq_sampling: int = 1,
                                     original_vids_ds=None,
                                     force_videos_creation: bool = False,
                                     default_videos_ext: str = '.mp4'):
        """Create videos from the ImageDataset `frames_ds`, assuming that the videos was previously
        sampled with a sampling frequency `freq_sampling`. The created videos will be stored in
        `dest_folder`

        Parameters
        ----------
        frames_ds : ImageDataset
            Image dataset with the information of the video frames to be created. Ideally it was
            previously created with the `create_frames_dataset` function
        dest_folder : str
            Path to the directory where the created videos will be saved
        freq_sampling : int, optional
            Number of frames per second taken from the videos when divided into frames,
            by default 1
        split_in_labels : bool, optional
            Whether or not to split the created videos into folders according to the label in the
            'label' column, by default True

        """
        assert_cond = (VFields.VID_FRAME_NUM in frames_ds.fields
                       and VFields.PARENT_MEDIA_ID in frames_ds.fields)
        assert_str = (
            f"The dataset must contain the {VFields.VID_FRAME_NUM} and "
            f"{VFields.PARENT_MEDIA_ID} columns")
        assert assert_cond, assert_str
        os.makedirs(dest_folder, exist_ok=True)

        logger.info(f"Creating videos from frames and storing them in the path: {dest_folder}")

        if original_vids_ds is not None:
            original_vids_df = original_vids_ds.df
            original_vids_dir = original_vids_ds.root_dir

        df = frames_ds.df

        def get_vid_file(vid_id):
            if original_vids_ds is not None:
                vid_rec = original_vids_df[original_vids_df[VFields.MEDIA_ID]
                                           == vid_id].iloc[0]
                if original_vids_dir is not None:
                    fname = os.path.relpath(vid_rec[VFields.ITEM], original_vids_dir)
                else:
                    fname = vid_rec[VFields.ITEM]
            elif not str(vid_id).lower().endswith(default_videos_ext):
                fname = f"{vid_id}{default_videos_ext}"
            else:
                fname = f"{get_random_id()}{default_videos_ext}"
            return os.path.join(dest_folder, fname)

        def _get_imgs(vid_id):
            return (df[df[VFields.PARENT_MEDIA_ID] == vid_id]
                    .sort_values(VFields.VID_FRAME_NUM)[VFields.ITEM].values)

        vids_ids = df[VFields.PARENT_MEDIA_ID].unique()
        parallel_exec(
            func=frames_to_video,
            elements=vids_ids,
            images=_get_imgs,
            freq_sampling=freq_sampling,
            output_file_name=get_vid_file,
            force=force_videos_creation)

    @staticmethod
    def create_frames_dataset(videos_ds: VisionDataset,
                              frames_folder: str = None,
                              freq_sampling: int = None,
                              frame_numbers: dict = None,
                              time_positions: dict = None,
                              **kwargs) -> ImageDataset:
        """Create an image dataset from the current video dataset, taking either `freq_sampling`
        frames every second, or the frames whose positions (1-based) are given in the list
        `frame_numbers`, and then store the resulting frame images in the path `frames_folder`

        Parameters
        ----------
        frames_folder : str, optional
            Path to the directory where the resulting frames will be stored.
            For each video a folder will be created inside `frames_folder` with the base name of
            the video without extension, and there the frames will be saved in the form
            `frame0000i.jpg`, where `i` is the number of the frame inside the video,
            i.e. `[1, cv2.CAP_PROP_FRAME_COUNT]`
            If `None` a temporary folder will be used. By default None
        freq_sampling : int, optional
            If provided, it is the number of frames per second to be taken from each video.
            This parameter is mutually exclusive with `time_positions` and `frame_numbers`.
        frame_numbers : dict, optional
            If provided, it is a dict in which each key is an item of the current video dataset,
            and its value is a list containing the 1-based positions of the frames to be taken from
            that video.
            For example, let's assume a dataset of videos that have 30 frames per second and a
            duration of 60 seconds (1,800 frames in total).
            ```
            frame_numbers = {
                'path/to/videos/video01.mp4': [13, 39, 119, 235, ..., 1642, 1701, 1800],
                'path/to/videos/video02.mp4': [1, 9, 93, 192, 345, 355, 432, 546, 899, ..., 1789],
                ...
            }
            ````
            This parameter is mutually exclusive with `time_positions` and `freq_sampling`.
            By default None
        time_positions : dict, optional
            If provided, it is a dict in which each key is an item of the current video dataset,
            and its value is a list containing the time positions in seconds to be taken from that
            video.
            For example, let's assume we have a dataset of videos that are of varying length but
            less than 60 seconds.
            ```
            time_positions = {
                'path/to/videos/video01.mp4': [0.04, 6.56, 10.12, 21.88, 32.52, 48.98, 56.2],
                'path/to/videos/video02.mp4': [1.1, 9.98, 13.0, 19.2, 35.5, 45.3],
                ...
            }
            ````
            This parameter is mutually exclusive with `frame_numbers` and `freq_sampling`.
            By default None

        Returns
        -------
        ImageDataset
            Instance of the created image dataset
        """
        if frames_folder is None:
            frames_folder = os.path.join(
                get_temp_folder(), f'frames_from_videos-{get_random_id()}')
        zero_based_indexing = kwargs.get('zero_based_indexing', False)
        verify_1st_frame_to_skip = kwargs.get('verify_if_the_first_frame_exists_to_skip', True)

        videos_data = Manager().dict()
        items = videos_ds.items

        frame_numbers_fn = None
        time_positions_fn = None
        if frame_numbers is not None:
            msg = f'Items in frame_numbers are not present in the dataset'
            assert len(set(frame_numbers.keys()) & set(items)) == len(frame_numbers), msg

            def frame_numbers_fn(record):
                return frame_numbers[record[VFields.ITEM]]
            logger.info(f"Converting {len(videos_ds.items)} videos to frames, "
                        f"given frame numbers for each video.")
        elif time_positions is not None:
            msg = f'Items in time_positions are not present in the dataset'
            assert len(set(time_positions.keys()) & set(items)) == len(time_positions), msg

            def time_positions_fn(record):
                return time_positions[record[VFields.ITEM]]
            logger.info(f"Converting {len(videos_ds.items)} videos to frames, "
                        f"given time positions for each video.")
        else:
            logger.info(f"Converting {len(videos_ds.items)} videos to frames, "
                        f"sampling {freq_sampling} frames per second.")

        tic = time.time()
        # To get only records from unique videos
        video_level_ds = videos_ds.create_media_level_ds()
        n_stems = len(set([Path(elem).stem for elem in items]))
        method = 'stems' if n_stems == len(items) else VFields.MEDIA_ID
        get_frams_dir = partial(VideoDataset.get_directory_for_frames,
                                base_folder=frames_folder, method=method)

        records = video_level_ds.records
        parallel_exec(
            func=VideoDataset.video_to_frames,
            elements=records,
            input_video_file=lambda rec: rec[VFields.ITEM],
            output_folder=lambda rec: get_frams_dir(rec),
            freq_sampling=freq_sampling,
            frame_numbers=frame_numbers_fn,
            time_positions=time_positions_fn,
            videos_data=videos_data,
            overwrite=False,
            zero_based_indexing=zero_based_indexing,
            verify_if_the_first_frame_exists_to_skip=verify_1st_frame_to_skip)

        logger.debug(f'Conversion of videos into frames took {time.time()- tic:.2f} seconds.')

        df = video_level_ds.df.drop_duplicates(
            VFields.ITEM, inplace=False).set_index(VFields.ITEM)
        imgs_data = defaultdict(list)

        for video_item, video_data in videos_data.items():
            vid_rec = df.loc[video_item]
            frames_list = video_data['frames_filenames']
            frames_num_video = video_data['frames_num_video']
            width = video_data['width']
            height = video_data['height']
            n_seq_frames = len(frames_list)

            ids = [get_random_id() for _ in range(n_seq_frames)]
            image_ids = [
                get_image_id_for_frame(vid_rec[VFields.MEDIA_ID], frames_list[i])
                for i in range(n_seq_frames)]
            video_ids = [vid_rec[VFields.MEDIA_ID]] * n_seq_frames
            widths = [width] * n_seq_frames
            heights = [height] * n_seq_frames

            if VFields.LABEL in vid_rec:
                labels = [vid_rec[VFields.LABEL]] * n_seq_frames
                imgs_data[VFields.LABEL].extend(labels)
            imgs_data[VFields.ITEM].extend(frames_list)
            imgs_data[VFields.ID].extend(ids)
            imgs_data[VFields.MEDIA_ID].extend(image_ids)
            imgs_data[VFields.PARENT_MEDIA_ID].extend(video_ids)
            imgs_data[VFields.VID_FRAME_NUM].extend(frames_num_video)
            imgs_data[VFields.WIDTH].extend(widths)
            imgs_data[VFields.HEIGHT].extend(heights)

        data = pd.DataFrame(imgs_data)
        imgs_ds = ImageDataset.from_dataframe(data, root_dir=frames_folder)
        return imgs_ds

    @staticmethod
    def create_crops_dataset(dataset: VisionDataset,
                             frames_folder: str = None,
                             dest_path: str = None,
                             use_partitions=False,
                             allow_label_empty: bool = False,
                             force_crops_creation: bool = False,
                             force_frames_creation: bool = True,  # TODO: remove
                             bottom_offset: Union[int, float] = 0,
                             prefix_field: str = None,
                             delete_frames_folder_on_finish: bool = True,
                             batch_size: int = 300) -> ImageDataset:
        """Method that generates crops with the coordinates of the bounding boxes from the
        annotations of a dataset of type `object detection`, and assigns the labels to that
        new images in order to create a dataset of type `classification`

        Parameters
        ----------
        dest_path : str, optional
            Folder in which the images created from the crops of the bouding boxes are saved.
            If None, the images will be saved in the folder `./crops_images`.
            By default None
        use_partitions : bool, optional
            Whether to use the partitions from the original dataset or not, by default False
        info : dict, optional
            Information to be stored in the new dataset, by default {}
        **kwargs
            Extra named arguments passed to the `ImageDataset` constructor and also may include the
            parameters:
            * allow_label_empty : bool
                Whether to allow annotations with label 'empty' or not, by default False
            * force_crops_creation : bool
                Whether to force the creation of the crops or not, by default False

        Returns
        -------
        ImageDataset
            Instance of the created `classification` dataset

        Raises
        ------
        Exception
            in case the original dataset is not of type `object detection`
        """
        assert_cond = VFields.VID_FRAME_NUM in dataset.fields
        assert_msg = f"The dataset must contain the field {VFields.VID_FRAME_NUM}"
        assert assert_cond, assert_msg

        if frames_folder is None:
            frames_folder = os.path.join(get_temp_folder(), f'frames_from_videos-{uuid.uuid4()}')

        crops_dss = []
        for _dataset in dataset.batch_iter(batch_size):

            df = _dataset.df

            frame_numbers = (
                df[[VFields.ITEM, VFields.VID_FRAME_NUM]]
                .groupby(VFields.ITEM)
                .apply(lambda x: sorted(set(x[VFields.VID_FRAME_NUM].tolist())))
                .to_dict())
            VideoDataset.create_frames_dataset(_dataset, frames_folder,
                                                frame_numbers=frame_numbers)
            df[VFields.ITEM] = df.apply(VideoDataset.get_frame_item,
                                        axis=1, frames_folder=frames_folder)

            frames_ds = ImageDataset.from_dataframe(
                df, root_dir=frames_folder, validate_filenames=False, accept_all_fields=True)
            _crops_ds = ImageDataset.create_crops_dataset(
                dataset=frames_ds,
                dest_path=dest_path,
                use_partitions=use_partitions,
                allow_label_empty=allow_label_empty,
                force_crops_creation=force_crops_creation,
                dims_correction=False,
                bottom_offset=bottom_offset,
                prefix_field=prefix_field)
            crops_dss.append(_crops_ds)

            if delete_frames_folder_on_finish:
                dirs_to_del = list(set([os.path.dirname(it) for it in frames_ds.items]))
                parallel_exec(
                    func=shutil.rmtree,
                    elements=dirs_to_del,
                    path=lambda dir_to_del: dir_to_del)

        crops_ds = ImageDataset.from_datasets(*crops_dss)
        return crops_ds

    # TODO: include fields: use_detection_labels, use_detections_scores
    @staticmethod
    def draw_bounding_boxes(dataset: VisionDataset,
                            freq_sampling: int,
                            frames_folder: str = None,
                            include_labels: bool = False,
                            include_scores: bool = False,
                            blur_people: bool = False,
                            thickness: int = None):
        frames_folder = frames_folder or get_temp_folder()

        frames_ds = VideoDataset.create_frames_dataset(
            dataset, frames_folder=frames_folder, freq_sampling=freq_sampling)

        # TODO: We need both datasets having the same fields in order to be joined
        frames_ds['video_id'] = lambda record: record[frames_ds.MetadataFields.PARENT_MEDIA_ID]
        dataset['video_id'] = lambda record: record[dataset.MetadataFields.MEDIA_ID]

        frames_w_boxes_ds = frames_ds.create_object_level_dataset_using_detections(
            dataset, fields_for_merging=['video_id', VFields.VID_FRAME_NUM])

        frames_w_boxes_ds.draw_bounding_boxes(include_labels=include_labels,
                                              include_scores=include_scores,
                                              blur_people=blur_people,
                                              thickness=thickness)

        VideoDataset.create_videos_from_frames_ds(
            frames_ds=frames_ds,
            dest_folder=dataset.root_dir,
            freq_sampling=freq_sampling,
            original_vids_ds=dataset,
            force_videos_creation=True,
            default_videos_ext=dataset.DEFAULT_EXT)

    @staticmethod
    def get_directory_for_frames(record: dict,
                                 base_folder: str,
                                 method: Literal['stems', 'uuid', 'file_id'] = 'stems'):
        if method == 'stems':
            return os.path.join(base_folder, Path(record[VFields.ITEM]).stem)
        elif method == 'uuid':
            return os.path.join(base_folder, f'{uuid.uuid4()}')
        elif method == VFields.MEDIA_ID:
            return os.path.join(base_folder, record[VFields.MEDIA_ID])

    @staticmethod
    def get_frame_item(record: dict, frames_folder: str):
        directory_for_frames = VideoDataset.get_directory_for_frames(
            record, frames_folder, method='stems')
        frame_path = VideoDataset.get_frame_path(
            record[VFields.VID_FRAME_NUM], directory_for_frames)
        return frame_path

    @staticmethod
    def get_frame_path(frame_number: int, frames_folder: str = None):
        frame_fname = 'frame{:05d}.jpg'.format(frame_number)
        if frames_folder is not None:
            return os.path.join(frames_folder, frame_fname)
        return frame_fname

    @staticmethod
    def video_to_frames(input_video_file: str,
                        output_folder: str,
                        freq_sampling: int = None,
                        frame_numbers: List[int] = None,
                        time_positions: List[float] = None,
                        videos_data: dict = None,
                        overwrite: bool = False,
                        zero_based_indexing: bool = False,
                        verify_if_the_first_frame_exists_to_skip: bool = True):
        """Create the image files by taking `freq_sampling` frames every second from the video file
        `input_video_file` and stores them in `output_folder`, saving in the dictionary `videos_data`
        the information related to the conversion

        Parameters
        ----------
        input_video_file : str
            Path of the video to be used for conversion
        output_folder : str
            Path to the directory where the created frames will be stored.
            Inside the folder the frames will be saved in the form `frame0000i.jpg`, where `i` is the
            number of the frame inside the video, i.e. `[1, cv2.CAP_PROP_FRAME_COUNT]`
        freq_sampling : int, optional
            If provided, it is the number of frames per second to be taken from each video.
            This parameter is mutually exclusive with `time_positions` and `frame_numbers`.
            By default None
        frame_numbers : list of int, optional
            If provided, it is a list containing the 1-based positions of the frames to be taken from
            the video.
            This parameter is mutually exclusive with `time_positions` and `freq_sampling`.
            By default None
        time_positions : list of float, optional
            If provided, it is a list containing the time positions in seconds to be taken from the
            video.
            This parameter is mutually exclusive with `frame_numbers` and `freq_sampling`.
            By default None
        videos_data : dict, optional
            Dictionary that will contain the information resulting from the conversion, in the form:
            `{input_video_file: {'frames_filenames': [frame_filename_0, ...],
            'frames_num_video': [frame_num_video_0, ...], 'fps': fps, 'frame_count': n_frames}}`,
            by default None
        overwrite : bool, optional
            Whether or not to overwrite the frames in case they have been previously created,
            by default False

        Returns
        -------
        tuple of ([str], int, int)
            A tuple with the values (frame_filenames, `cv2.CAP_PROP_FPS`, `cv2.CAP_PROP_FRAME_COUNT`),
            where the first element is the list with the paths of the created images and the other two
            are video properties
        """
        assert_cond = (bool(freq_sampling is not None)
                       + bool(frame_numbers is not None)
                       + bool(time_positions is not None)) == 1
        assert_msg = (f"You must specify ONLY one of the parameters (freq_sampling, "
                      f"frame_numbers, time_positions)")
        assert assert_cond, assert_msg
        assert os.path.isfile(input_video_file), f'File {input_video_file} not found'

        avoid_reading = False
        if not overwrite and verify_if_the_first_frame_exists_to_skip:
            # TODO: Check the case when frame_numbers are given
            first_frame_number = 0 if zero_based_indexing else 1
            first_frame_filename = VideoDataset.get_frame_path(
                first_frame_number, frames_folder=output_folder)
            if os.path.isfile(first_frame_filename):
                avoid_reading = True

        os.makedirs(output_folder, exist_ok=True)

        vidcap = cv2.VideoCapture(input_video_file)
        n_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if freq_sampling is not None:
            if freq_sampling > fps:
                freq_sampling = fps
            every_n_frames = round(fps / freq_sampling)  # TODO: think about removing round()

        frame_filenames = []
        frame_nums_video = []

        if time_positions is not None:
            frame_numbers = {round(time_pos * fps) + 1: time_pos for time_pos in time_positions}

        # frame_idx is always 0-base
        for frame_idx in range(0, n_frames):

            if not avoid_reading:
                success, image = vidcap.read()
                if not success:
                    assert image is None
                    break

            # frame_number can be 1-base so that the name of the first frame is frame00001.jpg
            frame_number = frame_idx if zero_based_indexing else frame_idx + 1

            if freq_sampling is not None:
                if frame_idx % every_n_frames != 0:
                    continue
            else:
                if frame_number not in frame_numbers:
                    continue

            frame_filename = VideoDataset.get_frame_path(frame_number, frames_folder=output_folder)
            frame_filenames.append(frame_filename)
            frame_nums_video.append(frame_number)

            if avoid_reading or (not overwrite and os.path.isfile(frame_filename)):
                continue

            try:
                cv2.imwrite(os.path.normpath(frame_filename), image)
                assert os.path.isfile(frame_filename), f'Output frame {frame_filename} unavailable'
            except Exception as e:
                print(f'Error on frame {frame_number} of {n_frames}: {str(e)}')

        vidcap.release()

        if videos_data is not None:
            videos_data[input_video_file] = {
                'frames_filenames': frame_filenames,
                'frames_num_video': frame_nums_video,
                'fps': int(fps),
                'width': int(width),
                'height': int(height),
                'frame_count': n_frames
            }

        return frame_filenames, fps, n_frames


# endregion
