#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from typing import List, Optional, Final

from ml_base.dataset import Dataset
from ml_base.utils.misc import is_array_like
from ml_base.utils.dataset import get_random_id
from ml_base.utils.logger import get_logger

from ml_vision.utils.vision import VisionFields as VFields
from ml_vision.utils.coords import get_coordinates_type_from_coords
from ml_vision.utils.coords import CoordinatesType

logger = get_logger(__name__)


class VisionDataset(Dataset):
    class METADATA_FIELDS(Dataset.METADATA_FIELDS):
        """Field names allowed in the creation of image datasets."""
        MEDIA_ID: Final = VFields.MEDIA_ID
        SEQ_ID: Final = VFields.SEQ_ID
        WIDTH: Final = VFields.WIDTH
        HEIGHT: Final = VFields.HEIGHT

        TYPES = {
            **Dataset.METADATA_FIELDS.TYPES,
            MEDIA_ID: str,
            SEQ_ID: str,
            WIDTH: float,
            HEIGHT: float
        }

    class ANNOTATIONS_FIELDS(Dataset.ANNOTATIONS_FIELDS):
        BBOX: Final = VFields.BBOX

        TYPES = {
            **Dataset.ANNOTATIONS_FIELDS.TYPES,
        }

    EMPTY_LABEL: Final = 'empty'
    FIELDS_TO_REMOVE_IN_MEDIA_LEVEL_DS: Final = [VFields.BBOX, VFields.SCORE]
    FILES_EXTS: Final = [".jpg", ".png", ".jpeg", ".avi", ".mp4"]
    DETS_FIELDS_TO_USE_IN_OBJ_LEVEL_DS: Final = {VFields.BBOX}

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
        fields_for_merging = fields_for_merging or [self.METADATA_FIELDS.MEDIA_ID]
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

        obj_level_df = pd.merge(left=dets_df, right=anns_df, how='left', on=fields_for_merging)
        obj_level_df[VFields.ID] = obj_level_df[VFields.ID].apply(lambda _: get_random_id())

        obj_level_ds = type(self).from_dataframe(obj_level_df,
                                                 root_dir=self.root_dir,
                                                 validate_filenames=False,
                                                 use_partitions=use_partitions)
        return obj_level_ds

    # FIXME: Fix this call
    def create_crops_dataset_using_detections(self,
                                              detections: VisionDataset,
                                              use_partitions: bool = False,
                                              use_detections_labels: bool = False,
                                              use_detections_scores: bool = False,
                                              dest_path: str = None,
                                              **kwargs) -> VisionDataset:
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
                             **kwargs) -> VisionDataset:
        raise NotImplementedError

    def draw_bounding_boxes(self,
                            include_labels: bool = False,
                            include_scores: bool = False,
                            blur_people: bool = False,
                            thickness: int = None):
        raise NotImplementedError

    #       region OTHER

    def _get_media_dims_of_items(self, items: List[str]) -> pd.DataFrame:
        raise NotImplementedError

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
