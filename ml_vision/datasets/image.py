#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
from multiprocessing import Manager
from collections import defaultdict
import pandas as pd
from typing import List, Optional, Final
from functools import partial

from ml_base.utils.misc import is_array_like
from ml_base.utils.misc import get_temp_folder
from ml_base.utils.misc import parallel_exec
from ml_base.utils.dataset import get_random_id
from ml_base.utils.dataset import map_category
from ml_base.utils.logger import get_logger

from ml_vision.utils.image import crop_bboxes_on_image
from ml_vision.utils.image import set_image_dims
from ml_vision.utils.image import draw_detections_of_image
from ml_vision.utils.image import ImageFields
from ml_vision.utils.vision import get_bbox_from_json_record
from ml_vision.datasets.vision import VisionDataset
from ml_vision.utils.coords import transform_coordinates, transform_coordinates_to_absolute_str
from ml_vision.utils.coords import CoordinatesFormat
# from utils.lila import get_azcopy_exec

logger = get_logger(__name__)


class ImageDataset(VisionDataset):
    class METADATA_FIELDS(VisionDataset.METADATA_FIELDS):
        """Field names allowed in the creation of image datasets."""
        MEDIA_ID: Final = ImageFields.MEDIA_ID
        # In case of frames
        PARENT_VID_FRAME_NUM: Final = ImageFields.PARENT_VID_FRAME_NUM
        PARENT_VID_NUM_FRAMES: Final = ImageFields.PARENT_VID_NUM_FRAMES
        PARENT_VID_ID: Final = ImageFields.PARENT_VID_ID
        # In case of crops
        PARENT_IMG_ID: Final = ImageFields.PARENT_IMG_ID

        TYPES = {
            **VisionDataset.METADATA_FIELDS.TYPES,
            MEDIA_ID: str,
            PARENT_IMG_ID: str,
            PARENT_VID_ID: str
        }

    class ANNOTATIONS_FIELDS(VisionDataset.ANNOTATIONS_FIELDS):
        pass

    FILES_EXTS: Final = [".jpg", ".png", ".jpeg"]
    DEFAULT_EXT: Final = ".jpg"
    FIELDS_TO_REMOVE_IN_MEDIA_LEVEL_DS = VisionDataset.FIELDS_TO_REMOVE_IN_MEDIA_LEVEL_DS
    DETS_FIELDS_TO_USE_IN_OBJ_LEVEL_DS = {*VisionDataset.DETS_FIELDS_TO_USE_IN_OBJ_LEVEL_DS,
                                          ImageFields.MEDIA_ID,
                                          ImageFields.PARENT_VID_FRAME_NUM,
                                          ImageFields.PARENT_VID_ID}
    # region PUBLIC API METHODS

    #   region FACTORY METHODS

    def create_crops_dataset(self,
                             dest_path: str = None,
                             use_partitions: bool = False,
                             allow_label_empty: bool = False,
                             force_crops_creation: bool = False,
                             dims_correction: bool = True,
                             bottom_offset: int = 0,
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
        assert ImageFields.BBOX in self.fields, "The dataset must be of detection type"
        logger.debug("Creating classification dataset from detection bounding boxes")

        if dest_path is None:
            dest_path = os.path.join(get_temp_folder(), f"{get_random_id()}")
        os.makedirs(dest_path, exist_ok=True)

        self.compute_and_set_media_dims(dims_correction=dims_correction)

        crops_exist = None
        crops_paths = defaultdict(list)
        bboxes = defaultdict(list)
        id_to_new_item = {}
        id_to_parent_img_id = {}

        for record in self.records:
            if record[ImageFields.LABEL] == self.EMPTY_LABEL and not allow_label_empty:
                continue

            crop_item = f"{record[ImageFields.ID]}{self.DEFAULT_EXT}"
            crop_path = os.path.join(dest_path, crop_item)
            if crops_exist is None:
                crops_exist = os.path.isfile(crop_path)

            x1, y1, x2, y2 = transform_coordinates(
                bbox=record[ImageFields.BBOX],
                output_format=CoordinatesFormat.x1_y1_x2_y2,
                media_width=record[ImageFields.WIDTH],
                media_height=record[ImageFields.HEIGHT])

            crops_paths[record[ImageFields.ITEM]].append(crop_path)
            bboxes[record[ImageFields.ITEM]].append((x1, y1, x2, y2))
            id_to_new_item[record[ImageFields.ID]] = crop_path
            id_to_parent_img_id[record[ImageFields.ID]] = record[ImageFields.MEDIA_ID]

        if not crops_exist or force_crops_creation:
            _destpath = os.path.abspath(dest_path)
            logger.info(f"Generating {len(id_to_new_item)} crops in folder {_destpath}")
        else:
            logger.info(f"Using already created crops in {dest_path}")
        bboxes_coords_inside_crops = Manager().dict()
        crops_dims = Manager().dict()
        parallel_exec(
            crop_bboxes_on_image,
            elements=self.items,
            source_path=lambda item: item,
            dest_paths=lambda item: crops_paths[item],
            bboxes=lambda item: bboxes[item],
            bboxes_coords_inside_crops=bboxes_coords_inside_crops,
            crops_dims=crops_dims,
            bottom_offset=bottom_offset,
            force_creation=force_crops_creation)

        bboxes_coords_inside_crops = dict(bboxes_coords_inside_crops)
        crops_dims_df = pd.DataFrame(data=crops_dims.values(),
                                     index=crops_dims.keys()).reset_index(names=ImageFields.ITEM)
        crops_ds = self.copy()
        crops_ds[ImageFields.ITEM] = lambda rec: id_to_new_item[rec[ImageFields.ID]]
        media_id_mapper = self._add_media_id_field_to_dataframe(crops_ds[[ImageFields.ITEM]])
        crops_ds[ImageFields.PARENT_IMG_ID] = lambda rec: id_to_parent_img_id[rec[ImageFields.ID]]
        crops_ds[ImageFields.BBOX] = lambda rec: bboxes_coords_inside_crops[rec[ImageFields.ITEM]]
        crops_ds[[ImageFields.WIDTH, ImageFields.HEIGHT]] = crops_dims_df
        crops_ds[ImageFields.ID] = lambda _: get_random_id()
        crops_ds[ImageFields.MEDIA_ID] = media_id_mapper
        crops_ds._split(use_partitions=use_partitions)
        crops_ds._set_root_dir(dest_path)

        return crops_ds

    #   endregion

    #   region DRAW BBOXES
    def draw_bounding_boxes(self,
                            include_labels: bool = False,
                            include_scores: bool = False,
                            blur_people: bool = False,
                            thickness: int = None):
        assert ImageFields.BBOX in self.fields, "Invalid dataset for drawing bounding boxes"

        dims = self.get_media_dims().set_index(ImageFields.ITEM)
        self[ImageFields.BBOX] = (
            lambda rec: transform_coordinates_to_absolute_str(
                bbox=rec[ImageFields.BBOX],
                media_width=dims.loc[rec[ImageFields.ITEM]][ImageFields.WIDTH],
                media_height=dims.loc[rec[ImageFields.ITEM]][ImageFields.HEIGHT]))

        dets_df = self.df

        if not include_labels and ImageFields.LABEL in dets_df.columns:
            dets_df = dets_df.drop(ImageFields.LABEL, axis=1)
        if not include_scores and ImageFields.SCORE in dets_df.columns:
            dets_df = dets_df.drop(ImageFields.SCORE, axis=1)

        parallel_exec(
            func=draw_detections_of_image,
            elements=self.items,
            item=lambda item: item,
            detections=dets_df,
            blur_people=blur_people,
            thickness=thickness)

    #   endregion

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
                E.g., {'image_id': 'id', 'datetime': 'date_captured', 'image_width': 'width'}
                for a JSON file with the field name 'image_id' for the id of the images, the
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
                pd.DataFrame([{ImageFields.MEDIA_ID: img_id, **ann}
                             for img_id, anns in json_handler.imgToAnns.items() for ann in anns]))
            if not ImageFields.LABEL in annotations.columns:
                annotations[ImageFields.LABEL] = (
                    annotations['category_id']
                    .apply(lambda x: json_handler.cats[x]['name']))
            if ImageFields.BBOX in annotations.columns:
                _get_bbox_from_json_rec = partial(
                    get_bbox_from_json_record,
                    include_bboxes_with_label_empty=include_bboxes_with_label_empty)
                annotations[ImageFields.BBOX] = annotations.apply(_get_bbox_from_json_rec, axis=1)

            annotations[ImageFields.ITEM] = (
                annotations[ImageFields.MEDIA_ID].apply(lambda x: img_id_to_item[x]))

        elif len(img_id_to_item) > 0:
            # Dataset with images but no annotations (e.g., a test dataset)
            annotations = pd.DataFrame([{ImageFields.ITEM: item,
                                         ImageFields.MEDIA_ID: img_id,
                                         ImageFields.ID: get_random_id()}
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

        df = annotations.merge(metadata, how='left', on=ImageFields.MEDIA_ID)

        return df

    #   endregion

    #   region OVERLOADED METHODS

    def _get_media_dims_of_items(self, items: List[str]) -> pd.DataFrame:
        """Determines the dimensions of `items`

        Parameters
        ----------
        items : list of str
            List of image paths

        Returns
        -------
        pd.DataFrame
            Dataframe containing 'width' and 'height' columns and the image path as index
        """
        images_dict = Manager().dict()
        logger.debug("Getting images dims from stored files...")

        parallel_exec(
            func=set_image_dims,
            elements=items,
            image=lambda item: item,
            images_dict=images_dict)

        df = pd.DataFrame(data=images_dict.values(),
                          index=images_dict.keys()).reset_index(names=ImageFields.ITEM)
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
            # Convert 'image_id' to str
            self.imgToAnns[str(ann['image_id'])].append(
                {**ann, ImageFields.MEDIA_ID: str(ann['image_id'])})
        for image in self.dataset.get('images', []):
            # Convert 'id' of images to str
            self.imgs[str(image['id'])] = {**image, ImageFields.MEDIA_ID: str(image['id'])}
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
