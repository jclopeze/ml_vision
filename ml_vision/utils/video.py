#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pandas as pd

from multiprocessing import Manager
import uuid

from ml_base.utils.misc import parallel_exec

import os
import cv2
from multiprocessing import Manager
from pathlib import Path
from typing import List
import uuid

from .vision import VisionFields
from ml_base.utils.misc import parallel_exec
from ml_base.utils.logger import get_logger

logger = get_logger(__name__)


def get_videos_dims(videos):
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
        first_frame_filename = get_frame_path(first_frame_number, frames_folder=output_folder)
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

        frame_filename = get_frame_path(frame_number, frames_folder=output_folder)
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


def frames_to_video(images: List[str],
                    freq_sampling: int,
                    output_file_name: str,
                    force=False):
    """Create a video from the set of frames specified in `images`, assuming that the video was
    previously sampled with a sampling frequency `freq_sampling`

    Parameters
    ----------
    images : List[str]
        List with the paths of the frames from which the video will be created
    freq_sampling : int
        Number of frames per second that were taken from the video when the video was split into
        frames
    output_file_name : str
        Path of the video file to be created
    codec_spec : str, optional
        Code of the codec to be used in the constructor `cv2.VideoWriter_fourcc`, by default "mp4v"
    """
    if len(images) == 0 or (os.path.isfile(output_file_name) and not force):
        return

    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    # Define the codec and create VideoWriter object
    codec_spec = "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*codec_spec)
    out = cv2.VideoWriter(output_file_name, fourcc, freq_sampling, (width, height))

    for image in images:
        frame = cv2.imread(image)
        out.write(frame)

    out.release()


def get_directory_for_frames(record, base_folder, method='stems'):
    if method == 'stems':
        return os.path.join(base_folder, Path(record[VideoFields.ITEM]).stem)
    elif method == 'uuid':
        return os.path.join(base_folder, f'{uuid.uuid4()}')
    elif method == 'video_id':
        return os.path.join(base_folder, record[VideoFields.MEDIA_ID])


def get_frame_path(frame_number, frames_folder=None):
    frame_fname = 'frame{:05d}.jpg'.format(frame_number)
    if frames_folder is not None:
        return os.path.join(frames_folder, frame_fname)
    return frame_fname


def get_frame_item(record, frames_folder):
    directory_for_frames = get_directory_for_frames(record, frames_folder, method='stems')
    frame_path = get_frame_path(record[VideoFields.VID_FRAME_NUM], directory_for_frames)
    return frame_path


def get_image_id_for_frame(video_id, frame_filename):
    return f"{video_id}-{Path(frame_filename).stem}"


def get_video_dims(video):
    """Open an video, get its dimensions and return it in format (width, height).
    In case an exception occurs when trying to open the video, None will be returned for both width
    and height

    Parameters
    ----------
    video : str or a file object
        Path where the video is stored, or file object containing the video data

    Returns
    -------
    tuple of int
        The dimensions of the video in format (width, height)
    """
    try:
        vid = cv2.VideoCapture(video)
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    except:
        logger.exception(f"Exception occurred while opening video {video}")
        width, height = None, None

    return width, height

class VideoFields(VisionFields):
    VID_NUM_FRAMES = 'video_num_frames'
    MEDIA_ID = "video_id"
    VID_FRAME_NUM = 'video_frame_num'
