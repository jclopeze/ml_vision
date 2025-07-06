from enum import Enum, auto
from math import floor, ceil
from typing import Optional, Union, Iterable, Tuple
from functools import partial

from ml_base.utils.misc import is_array_like
from ml_base.utils.logger import get_logger

from ml_vision.utils.vision import VisionFields as VFields

logger = get_logger(__name__)

__all__ = ['CoordinatesFormat', 'CoordinatesType', 'CoordinatesDataType', 'transform_coordinates',
           'get_coordinates_type_from_coords', 'rescale_bboxes', 'non_max_supression',
           'is_insideof', 'bboxes_overlap', 'bb_intersection_over_union', 'get_bbox_dims',
           'transform_coordinates_to_absolute_str']


class CoordinatesFormat(Enum):
    """Valid coordinate formats, either [x, y, width, height], [x1, y1, x2, y2] and
    [y1, x1, y2, x2]
    """
    x_y_width_height = auto()
    x1_y1_x2_y2 = auto()
    y1_x1_y2_x2 = auto()


class CoordinatesType(Enum):
    """Valid coordinate types, either `absolute`, `relative` or `normalized` (the same as
    `relative`)
    """
    # Values given in pixels; does not depend on media size
    absolute = auto()
    # Values given in a floating number in [0,1] and depends on the size of the media
    relative = auto()
    # All values of the coordinates are zero or any of them are NaN
    invalid = auto()


class CoordinatesDataType(Enum):
    """Valid coordinate data types, either `string`, `tuple`, `array` or `list` (the same as
    `array`)
    """
    string = auto()
    tuple = auto()
    array = auto()


def get_coordinates_type_from_coords(coord1, coord2, coord3, coord4) -> CoordinatesType:
    """Determines if the set of coordinates are absolute or relative, according to their value

    Parameters
    ----------
    coord1 : int or float
        Value of the first coordinate
    coord2 : int or float
        Value of the second coordinate
    coord3 : int or float
        Value of the third coordinate
    coord4 : int or float
        Value of the fourth coordinate

    Returns
    -------
    `CoordinatesType`
        Either `CoordinatesType.relative` or `CoordinatesType.absolute`
    """
    if all([i == 0. for i in [coord1, coord2, coord3, coord4]]):
        return CoordinatesType.invalid
    if all([0. <= i <= 1. for i in [coord1, coord2, coord3, coord4]]):
        return CoordinatesType.relative
    return CoordinatesType.absolute


def transform_coordinates(bbox: Union[str, Iterable],
                          *,
                          input_format: CoordinatesFormat = CoordinatesFormat.x_y_width_height,
                          output_format: CoordinatesFormat = CoordinatesFormat.x_y_width_height,
                          output_coords_type: CoordinatesType = CoordinatesType.absolute,
                          output_data_type: CoordinatesDataType = CoordinatesDataType.tuple,
                          media_width: Optional[int] = None,
                          media_height: Optional[int] = None,
                          round_digits: Optional[int] = None) -> Union[list, tuple, str]:
    """Function to convert between coordinate formats, specifying the input and output formats,
    the output coordinate types and the output data type

    Parameters
    ----------
    bbox : str or array-like
        Coordinates to be converted. It must contain four elements that have the meaning according
        to `input_format`. If str, elements must be separated by commas
    input_format : `CoordinatesFormat`, optional
        Determine the order and meaning of the elements of `bbox`, by default
        `CoordinatesFormat.x_y_width_height`
    output_format : `CoordinatesFormat`, optional
        Determine the order and meaning of the elements of the output, by default
        `CoordinatesFormat.x_y_width_height`
    output_coords_type : `CoordinatesType`, optional
        Determines if the value of the output coordinates will be given in absolute pixel values,
        or normalized in relation to the media size and therefore in the range [0,1].
        By default CoordinatesType.absolute
    output_data_type : `CoordinatesDataType`, optional
        Determines the data type of the output, by default CoordinatesDataType.tuple
    media_width : int, optional
        Media width value, given in pixels, by default None
    media_height : int, optional
        Media height value, given in pixels, by default None

    Returns
    -------
    list, tuple or str
        Coordinates resulting from the conversion, given in the format and type of coordinates and
        in the type of data requested
    """
    if media_width is not None and media_height is not None:
        media_width = float(media_width)
        media_height = float(media_height)
    if type(bbox) == str:
        [coord1, coord2, coord3, coord4] = [float(x) for x in bbox.split(',')]
    elif is_array_like(bbox):
        [coord1, coord2, coord3, coord4] = [x for x in bbox]
    else:
        raise ValueError(f"'{type(bbox)}' is not a valid type for a bounding box.")

    if input_format == CoordinatesFormat.x_y_width_height:
        [x, y, width, height] = [coord1, coord2, coord3, coord4]
        [x1, y1, x2, y2] = [x, y, x+width, y+height]
    elif input_format == CoordinatesFormat.x1_y1_x2_y2:
        [x1, y1, x2, y2] = [coord1, coord2, coord3, coord4]
    elif input_format == CoordinatesFormat.y1_x1_y2_x2:
        [y1, x1, y2, x2] = [coord1, coord2, coord3, coord4]
    else:
        raise ValueError("Invalid input coordinates format.")

    input_coords_type = get_coordinates_type_from_coords(x1, y1, x2, y2)
    if input_coords_type == CoordinatesType.invalid:
        raise ValueError(f"Coordinates are given in and invalid type: {x1, y1, x2, y2}")
    if (input_coords_type == CoordinatesType.absolute and
            output_coords_type == CoordinatesType.relative):
        if media_width is None or media_height is None:
            raise ValueError("You must supply the media_width and media_height of media")
        x1, y1, x2, y2 = (x1 / media_width,
                          y1 / media_height,
                          x2 / media_width,
                          y2 / media_height)
    elif (input_coords_type == CoordinatesType.relative and
            output_coords_type == CoordinatesType.absolute):
        if media_width is None or media_height is None:
            raise ValueError("You must supply the media_width and media_height of media")
        x1, y1, x2, y2 = (x1 * media_width,
                          y1 * media_height,
                          x2 * media_width,
                          y2 * media_height)

    if output_coords_type == CoordinatesType.absolute:
        x1, y1, x2, y2 = (int(floor(x1)),
                          int(floor(y1)),
                          int(ceil(x2)),
                          int(ceil(y2)))

    if output_format == CoordinatesFormat.x_y_width_height:
        coord1, coord2, coord3, coord4 = x1, y1, x2-x1, y2-y1
    elif output_format == CoordinatesFormat.y1_x1_y2_x2:
        coord1, coord2, coord3, coord4 = y1, x1, y2, x2
    elif output_format == CoordinatesFormat.x1_y1_x2_y2:
        coord1, coord2, coord3, coord4 = x1, y1, x2, y2
    else:
        raise ValueError("Invalid input output coordinates format.")

    if round_digits is not None:
        coord1, coord2, coord3, coord4 = [round(c, round_digits)
                                          for c in [coord1, coord2, coord3, coord4]]

    if output_data_type == CoordinatesDataType.array:
        return [coord1, coord2, coord3, coord4]
    elif output_data_type == CoordinatesDataType.tuple:
        return (coord1, coord2, coord3, coord4)
    elif output_data_type == CoordinatesDataType.string:
        return ",".join([str(x) for x in [coord1, coord2, coord3, coord4]])
    else:
        raise ValueError("Invalid output data type for coordinates transform")


def rescale_bboxes(bbox, actual_dims, new_dims):
    """Rescale bounding boxes when media are resized

    Parameters
    ----------
    bbox : str
        Actual value of the bounding box. It must be a str in the form 'x1,y1,width,height'
    actual_dims : dict or 2-tuple of integers
        Size of the media before the resizing, in the form {'width': width, 'height': height} or
        (width, height)
    new_dims : dict or 2-tuple of integers
        Size of the media after the resizing, in the form {'width': width, 'height': height} or
        (width, height)

    Returns
    -------
    str
        The bounding box resized according to the new media size, in the form 'x1,y1,width,height'
    """
    if bbox == "":
        return ""

    [x1, y1, width, height] = [float(x) for x in bbox.split(',')]

    if type(actual_dims) is dict:
        act_width = actual_dims['width']
        act_height = actual_dims['height']
    else:
        act_width = actual_dims[0]
        act_height = actual_dims[1]

    if type(new_dims) is dict:
        new_width = new_dims['width']
        new_height = new_dims['height']
    else:
        new_width = new_dims[0]
        new_height = new_dims[1]

    x1 = int(x1 * new_width/act_width)
    y1 = int(y1 * new_height/act_height)
    width = int(width * new_width/act_width)
    height = int(height * new_height/act_height)

    return ','.join([str(int(x)) for x in [x1, y1, width, height]])


def non_max_supression(boxA, boxB, iou_threshold=0.5):
    """Determine if one box is inside the other or if the intersection on the union (IoU) of both
    is < `iou_threshold`

    Parameters
    ----------
    boxA : list of float
        Represents the coordinates of a bounding box, given in the format [x1, y1, x2, y2]
    boxB : list of float
        Represents the coordinates of a bounding box, given in the format [x1, y1, x2, y2]
    iou_threshold : float, optional
       Minimum value of the intersection on the union (IoU) to know if the two bounding boxes
       overlap , by default 0.5

    Returns
    -------
    bool
        `False` in case one box is inside the other or the two overlap
    """
    # bbox: x1, y1, x2, y2
    if is_insideof(boxA, boxB) or is_insideof(boxB, boxA):
        return False
    if bb_intersection_over_union(boxA, boxB) >= iou_threshold:
        return False
    return True


def is_insideof(boxA, boxB):
    """Determine if one box is inside the other from the coordinates of both

    Parameters
    ----------
    boxA : list of float
        Represents the coordinates of a bounding box, given in the format [x1, y1, x2, y2]
    boxB : list of float
        Represents the coordinates of a bounding box, given in the format [x1, y1, x2, y2]

    Returns
    -------
    bool
        `True` in case one bounding box is inside the other
    """
    # bbox: x1, y1, x2, y2
    x1A, y1A, x2A, y2A = boxA[0], boxA[1], boxA[2], boxA[3]
    x1B, y1B, x2B, y2B = boxB[0], boxB[1], boxB[2], boxB[3]

    return (
        x1A >= x1B and
        x2A <= x2B and
        y1A >= y1B and
        y2A <= y2B)


def bboxes_overlap(boxA, boxB):
    """Determine if two boxes overlap from the coordinates of both

    Parameters
    ----------
    boxA : list of float
        Represents the coordinates of a bounding box, given in the format [x1, y1, x2, y2]
    boxB : list of float
        Represents the coordinates of a bounding box, given in the format [x1, y1, x2, y2]

    Returns
    -------
    bool
        `True` in case the two boxes overlap
    """
    # bbox: x1, y1, x2, y2
    x1A, y1A, x2A, y2A = boxA[0], boxA[1], boxA[2], boxA[3]
    x1B, y1B, x2B, y2B = boxB[0], boxB[1], boxB[2], boxB[3]

    # If one rectangle is on the left side of other
    if x1A > x2B or x1B > x2A:
        return False
    # If one rectangle is above other
    if y1A > y2B or y1B > y2A:
        return False

    return True


def bb_intersection_over_union(boxA, boxB):
    """Compute the intersection over union (IoU) of two bounding boxes from the coordinates
    of both

    Parameters
    ----------
    boxA : list of float
        Represents the coordinates of a bounding box, given in the format [x1, y1, x2, y2]
    boxB : list of float
        Represents the coordinates of a bounding box, given in the format [x1, y1, x2, y2]

    Returns
    -------
    float
        The value of the intersection over union IoU of the two bounding boxes, and 0. if the boxes
        do not overlap
    """
    # bbox: x1, y1, x2, y2
    if not boxA or not boxB:
        return 0.

    if not bboxes_overlap(boxA, boxB):
        return 0.

    x1_A, y1_A, x2_A, y2_A = boxA[0], boxA[1], boxA[2], boxA[3]
    x1_B, y1_B, x2_B, y2_B = boxB[0], boxB[1], boxB[2], boxB[3]

    xA = max(x1_A, x1_B)
    yA = max(y1_A, y1_B)
    xB = min(x2_A, x2_B)
    yB = min(y2_A, y2_B)

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (x2_A - x1_A) * (y2_A - y1_A)
    boxBArea = (x2_B - x1_B) * (y2_B - y1_B)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


transform_coordinates_to_absolute_str = partial(
    transform_coordinates,
    output_coords_type=CoordinatesType.absolute,
    output_data_type=CoordinatesDataType.string)


def get_bbox_dims(record) -> Tuple[int, int]:
    x1, y1, x2, y2 = transform_coordinates(
        bbox=record[VFields.BBOX],
        output_format=CoordinatesFormat.x1_y1_x2_y2,
        media_width=record[VFields.WIDTH],
        media_height=record[VFields.HEIGHT])

    return x2-x1, y2-y1
