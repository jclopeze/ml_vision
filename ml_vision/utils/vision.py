
from ml_base.utils.dataset import Fields

class VisionFields(Fields):
    BBOX = 'bbox'
    WIDTH = "width"
    HEIGHT = "height"
    SEQ_ID = "seq_id"
    FILE_TYPE = "file_type"

    VID_FRAME_NUM = 'video_frame_num'
    PARENT_FILE_ID = "parent_file_id"
    SCORE_DET = "score_det"


