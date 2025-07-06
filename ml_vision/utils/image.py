#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import pandas as pd
from typing import Iterable, Tuple, List, Union
import numpy as np

from PIL import Image
from PIL import ImageFile
import cv2

from ml_base.utils.logger import get_logger

from ml_vision.utils.vision import VisionFields
from ml_vision.utils.coords import CoordinatesType, CoordinatesFormat, CoordinatesDataType
from ml_vision.utils.coords import transform_coordinates

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = get_logger(__name__)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (51, 51, 255)
ORANGE = (51, 153, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 128, 0)
PURPLE = (255, 102, 178)
COLORS_MAP = {
    'GREEN': GREEN,
    'ORANGE': ORANGE,
    'BLUE': BLUE,
    'YELLOW': YELLOW,
    'PURPLE': PURPLE,
    'WHITE': WHITE,
    'RED': RED,
}

__all__ = ['COLORS_MAP', 'set_image_dims', 'draw_detections_of_image', 'draw_bboxes_in_image',
           'anonymize_image', 'crop_bboxes_on_image', 'get_bbox_from_json_record']

def set_image_dims(image: str, images_dict: dict):
    """Determines the dims of the image passed in `image` and store it in the `images_dict`
    dictionary in the form {`image`: {'width': `width`, 'height': `height`}}

    Parameters
    ----------
    image : str
        Path of an image
    images_dict : dict
        Dictionary where the image dimensions will be stored
    """
    try:
        with Image.open(image) as img:
            width, height = img.size
    except:
        logger.exception(f"Exception occurred while opening image {image}")
        width, height = None, None

    images_dict[image] = {
        'width': width,
        'height': height
    }


def draw_detections_of_image(item: str,
                             detections: pd.DataFrame,
                             blur_people: bool = False,
                             thickness: int = None,
                             color: str = RED):
    """Draws bounding boxes of the detections in the image `item`.
    Optionally applies a Gaussian filter to blur the region of person
    detections and preserve anonymity; in such cases, neither the bounding boxes nor the labels
    will be drawn on the image.

    Parameters
    ----------
    item : str
        Element to which the detection bounding boxes will be drawn
    detections : pd.DataFrame
        DataFrame containing the detections for `item`
    dest_dir : str
        Folder in which the resulting image will be stored
    score_thres_colors : float
        If provided, detections with `score ≥ score_thres_colors` will be drawn in green,
        and the others in orange. By default None
    copy_img_if_has_no_dets : bool, optional
        Whether or not to make a copy of those images without detections in `detections`,
        by default True
    include_score : bool, optional
        Whether or not to include the score in % in the label of the detection, by default False
    blur_people : bool, optional
        Whether or not to blur the detections of persons in order to preserve anonymity,
        by default False
    """
    dets_item = detections[detections[VisionFields.ITEM] == item]

    if len(dets_item) == 0:
        return

    bboxes, labels, scores, colors = [], [], [], []
    for record in dets_item.to_dict('records'):
        bboxes.append([int(x) for x in record[VisionFields.BBOX].split(',')])
        labels.append(record.get(VisionFields.LABEL))
        scores.append(record.get(VisionFields.SCORE))
        if 'color' in record:
            colors.append(record['color'])
    if len(colors) == 0:
        colors = color

    draw_bboxes_in_image(item, bboxes, labels, scores, colors, thickness, blur_people)


def draw_bboxes_in_image(img_path: str,
                         bboxes: List[Tuple],
                         labels: List[str],
                         scores: List[float],
                         colors: Union[List[str], str] = RED,
                         thickness: int = None,
                         blur_people: bool = False,
                         blur_people_factor: float = 7.0,
                         person_label: str = 'person'):
    """Function that draws bounding boxes in the image `img_orig_path` and saves the resulting
    image in `img_dest_path`.

    Parameters
    ----------
    img_orig_path : str
        Path of the original image
    img_dest_path : str
        Path of the resulting image
    bboxes : list
        List with the coordinates of the bounding boxes in format [x, y, width, height]
    labels : list
        List with the labels for the bounding boxes
    scores : list
        List with the scores of the detections. It must have the same lenght of bboxes
        If empty list, scores won't be showed.
    colors : list of tuples or tuple, optional
        Colors of the bounding boxes lines, by default GREEN
    thickness : int, optional
        Width of the bounding box lines. If None, it will be determined by the dimensions of the image.
        By default None
    blur_people : bool, optional
        Whether or not to blur the detections of persons in order to preserve anonymity,
        by default False
    blur_people_factor : float, optional
        The 'blurriness' applied to each human. Lower values are more obscured.
        Recommended values are between 3 and 7, by default 7.0
    person_label : str, optional
        Label that people detections have, by default 'person'
    """
    assert len(bboxes) == len(labels) == len(scores), "Invalid parameters sizes"
    colors = colors if type(colors) is list else [colors] * len(bboxes)

    image = cv2.imread(img_path)

    height = image.shape[0]
    width = image.shape[1]

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    LINE_TYPE = 2

    if thickness is None:
        thickness = math.ceil(width / 400)
    THICKNESS = math.ceil(width / 400)

    if width <= 800:
        FONT_SCALE = 0.5
    elif width <= 1200:
        FONT_SCALE = 0.8
    else:
        FONT_SCALE = 1.

    for bbox, label, score, color in zip(bboxes, labels, scores, colors):
        [x, y, width, height] = bbox

        if label == person_label and blur_people:
            person_data = image[y:y+height, x:x+width]
            person_data = anonymize_image(person_data, factor=blur_people_factor)
            image[y:y+height, x:x+width] = person_data
            continue

        offset_thick = int(thickness / 2)
        upper_left = (x - offset_thick, y - offset_thick)
        upper_right = (x + offset_thick + width, y - offset_thick)
        lower_right = (x + offset_thick + width, y + offset_thick + height)
        lower_left = (x - offset_thick, y + offset_thick + height)

        cv2.line(image, upper_left, upper_right, color=color, thickness=thickness)
        cv2.line(image, upper_right, lower_right, color=color, thickness=thickness)
        cv2.line(image, lower_right, lower_left, color=color, thickness=thickness)
        cv2.line(image, lower_left, upper_left, color=color, thickness=thickness)

        if not label:
            continue
        label = label.capitalize()

        if score:
            label += f' {int(score * 100)} %'

        (lbl_width, lbl_height), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
        x = 1 if x-thickness < 1 else x-thickness
        y = lbl_height if y-thickness < 0 else y-thickness
        cv2.rectangle(image, (x, y),  (x + lbl_width, y - lbl_height), color, -1)
        cv2.putText(image, label, (x, y), FONT,  FONT_SCALE, BLACK, LINE_TYPE)

    cv2.imwrite(img_path, image)


def anonymize_image(image, factor=7.0):
    """Blurs an image using a Gaussian filter.

    Parameters
    ----------
    image : np.array
        Input image; the image can have any number of channels, which are processed independently.
    factor : float, optional
        The 'blurriness' applied to each human. Lower values are more obscured.
        Recommended values are between 3 and 7, by default 7.0

    Returns
    -------
    np.array
        Output blurred image of n-dimensional array.
    """
    # automatically determine the size of the blurring kernel based
    # on the spatial dimensions of the input image
    (h, w) = image.shape[:2]
    kW = int(w / factor)
    kH = int(h / factor)
    # ensure the width of the kernel is odd
    if kW % 2 == 0:
        kW += 1
    # ensure the height of the kernel is odd
    if kH % 2 == 0:
        kH += 1

    # apply a Gaussian blur to the input image using our computed
    # kernel size
    return cv2.GaussianBlur(image, (kW, kH), 0)


def crop_bboxes_on_image(source_path: str,
                         dest_paths: list[str],
                         bboxes: list[Iterable[int]],
                         bboxes_coords_inside_crops: dict,
                         crops_dims: dict,
                         bottom_offset: Union[int, float] = 0,
                         force_creation: bool = True):
    assert len(dest_paths) == len(bboxes)
    assert isinstance(bottom_offset, int) or (
        isinstance(bottom_offset, float) and bottom_offset < 1)

    LEFT = 0
    UPPER = 1
    RIGHT = 2
    LOWER = 3
    try:
        img_orig = Image.open(source_path).convert('RGB')
        im_width, im_height = img_orig.size

        y2_max = im_height-1
        if isinstance(bottom_offset, int) and bottom_offset < y2_max:
            y2_max -= bottom_offset

        for dest_path, bbox in zip(dest_paths, bboxes):
            bbox_width = bbox[RIGHT] - bbox[LEFT]
            bbox_height = bbox[LOWER] - bbox[UPPER]

            extra = int(abs(bbox_width - bbox_height) / 2)
            if bbox_width >= bbox_height:
                x1, x2 = bbox[LEFT], bbox[RIGHT]
                y1 = bbox[UPPER] - extra
                if isinstance(bottom_offset, float) and bbox[LOWER] >= bottom_offset * im_height:
                    y2 = bbox[LOWER]
                    y1 -= extra
                else:
                    y2 = bbox[LOWER] + extra

                if (bbox_width - bbox_height) % 2 != 0:
                    y1 -= 1
                if y1 < 0:
                    y2 += abs(y1)
                    y2 = min(y2, im_height-1)
                    y1 = 0
                if y2 > y2_max:
                    y1 -= (y2 - y2_max)
                    y1 = max(y1, 0)
                    y2 = y2_max
            else:
                x1, x2 = bbox[LEFT] - extra, bbox[RIGHT] + extra
                y1, y2 = bbox[UPPER], bbox[LOWER]
                if (bbox_height - bbox_width) % 2 != 0:
                    x1 -= 1
                if x1 < 0:
                    x2 += abs(x1)
                    x2 = min(x2, im_width-1)
                    x1 = 0
                if x2 > im_width-1:
                    x1 -= (x2 - (im_width-1))
                    x1 = max(x1, 0)
                    x2 = im_width-1

            box = (x1, y1, x2, y2)

            img_crop = img_orig.crop(box)

            x_in_crop = bbox[LEFT]-x1
            y_in_crop = bbox[UPPER]-y1
            w_in_crop = bbox[RIGHT]-bbox[LEFT]
            h_in_crop = bbox[LOWER]-bbox[UPPER]
            bbox_inside_crop_coords = [x_in_crop, y_in_crop, w_in_crop, h_in_crop]
            crop_width = x2-x1
            crop_height = y2-y1

            crops_dims[dest_path] = {'width': crop_width, 'height': crop_height}
            bboxes_coords_inside_crops[dest_path] = transform_coordinates(
                bbox_inside_crop_coords,
                input_format=CoordinatesFormat.x_y_width_height,
                output_format=CoordinatesFormat.x_y_width_height,
                output_coords_type=CoordinatesType.relative,
                output_data_type=CoordinatesDataType.string,
                media_width=crop_width,
                media_height=crop_height,
                round_digits=4)

            if force_creation or not os.path.isfile(dest_path):
                img_crop.save(dest_path)
    except Exception as e:
        logger.exception(f"Exception while cropping the image {source_path}: {e}")


def get_bbox_from_json_record(record: dict, include_bboxes_with_label_empty: bool) -> Union[str, object]:
    """Get the bbox field of a record that comes from a JSON file of type COCO

    Parameters
    ----------
    record : dict
        Register of a JSON file of type COCO
    include_bboxes_with_label_empty : bool
        Whether to allow annotations with label 'empty' or not.

    Returns
    -------
    str or np.NaN
        String containing the values of the coordinates of the bbox field,
        in the same format as in `record`.
    """
    if ((VisionFields.BBOX in record and record[VisionFields.BBOX] is not np.NaN
         and record[VisionFields.LABEL]) and (record[VisionFields.LABEL] != 'empty'
                                              or include_bboxes_with_label_empty)):
        return ','.join([str(x) for x in record[VisionFields.BBOX]])
    return np.NaN
