#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd
from multiprocessing import Manager
from pathlib import Path
from typing import Tuple, Iterable

import cv2

from ml_base.utils.misc import parallel_exec
from ml_base.utils.logger import get_logger

from ml_vision.utils.vision import VisionFields as VFields

logger = get_logger(__name__)

__all__ = ['get_videos_dims', 'set_video_dims', 'frames_to_video', 'get_file_id_for_frame',
           'get_video_dims', 'get_frame_numbers_from_vids']


def get_videos_dims(videos: Iterable[str]) -> pd.DataFrame:
    """Determine the dimensions of `videos`

    Parameters
    ----------
    videos : list of str
        List of video paths

    Returns
    -------
    pd.DataFrame
        Dataframe containing 'width' and 'height' columns and the item as index
    """
    videos_dict = Manager().dict()
    logger.debug("Getting videos dimensionss from stored files...")

    parallel_exec(
        func=set_video_dims,
        elements=videos,
        video=lambda video: video,
        videos_dict=videos_dict)

    videos_sizes = {
        video: {
            'width': size['width'],
            'height': size['height'],
        }
        for video, size in videos_dict.items()
    }

    return pd.DataFrame(data=videos_sizes.values(), index=videos_sizes.keys())


def set_video_dims(video, videos_dict):
    """Determine the dimensions of the passed video and store it in the `videos_dict` dictionary in the
    form {`video`: {'width': `width`, 'height': `height`}}

    Parameters
    ----------
    video : str
        Path of an video
    videos_dict : dict
        Dictionary where the video size will be stored
    """
    width, height = get_video_dims(video)
    videos_dict[video] = {
        'width': width,
        'height': height
    }


def frames_to_video(frames_paths: Iterable[str],
                    freq_sampling: int,
                    output_file_name: str,
                    force=False):
    """Create a video from the set of frames specified in `frames_paths`, assuming that the video was
    previously sampled with a sampling frequency `freq_sampling`

    Parameters
    ----------
    frames_paths : List[str]
        List with the paths of the frames from which the video will be created
    freq_sampling : int
        Number of frames per second that were taken from the video when the video was split into
        frames
    output_file_name : str
        Path of the video file to be created
    codec_spec : str, optional
        Code of the codec to be used in the constructor `cv2.VideoWriter_fourcc`, by default "mp4v"
    """
    if len(frames_paths) == 0 or (os.path.isfile(output_file_name) and not force):
        return

    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

    frame = cv2.imread(frames_paths[0])
    height, width, _ = frame.shape

    # Define the codec and create VideoWriter object
    codec_spec = "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*codec_spec)
    out = cv2.VideoWriter(output_file_name, fourcc, freq_sampling, (width, height))

    for image in frames_paths:
        frame = cv2.imread(image)
        out.write(frame)

    out.release()


def get_file_id_for_frame(file_id, frame_filename) -> str:
    return f"{file_id}-{Path(frame_filename).stem}"


def get_video_dims(video_path: str) -> Tuple[int, int]:
    """Open an video, get its dimensions and return it in format (width, height).
    In case an exception occurs when trying to open the video, None will be returned for both width
    and height

    Parameters
    ----------
    video_path : str or a file object
        Path where the video is stored, or file object containing the video data

    Returns
    -------
    tuple of int
        The dimensions of the video in format (width, height)
    """
    try:
        vid = cv2.VideoCapture(video_path)
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    except:
        logger.exception(f"Exception occurred while opening video {video_path}")
        width, height = None, None

    return width, height


def get_frame_numbers_from_vids(dataframe: pd.DataFrame) -> dict:
    return (dataframe[[VFields.ITEM, VFields.VID_FRAME_NUM]]
            .groupby(VFields.ITEM)
            .apply(lambda x: sorted(set(x[VFields.VID_FRAME_NUM].tolist())))
            .to_dict())
