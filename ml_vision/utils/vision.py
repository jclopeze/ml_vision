from typing import Union
import numpy as np

from ml_base.utils.dataset import Fields


class VisionFields(Fields):
    MEDIA_ID = "media_id"
    BBOX = 'bbox'
    WIDTH = "width"
    HEIGHT = "height"
    SEQ_ID = "seq_id"


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
