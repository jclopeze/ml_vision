#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from pathlib import Path
from multiprocessing import Manager
from collections import defaultdict
import time
import pandas as pd
from functools import partial
from typing import Final

from ml_base.utils.logger import get_logger
from ml_base.utils.misc import parallel_exec
from ml_base.utils.dataset import get_random_id
from ml_base.utils.misc import get_temp_folder

from ml_vision.utils.video import get_directory_for_frames
from ml_vision.utils.video import video_to_frames
from ml_vision.utils.video import frames_to_video
from ml_vision.utils.video import get_image_id_for_frame
from ml_vision.utils.video import get_frame_item
from ml_vision.utils.video import VideoFields
from ml_vision.utils.image import ImageFields

from ml_vision.datasets.vision import VisionDataset
from ml_vision.datasets.image import ImageDataset

logger = get_logger(__name__)


class VideoDataset(VisionDataset):
    """Represent a VideoDataset specification."""

    class METADATA_FIELDS(VisionDataset.METADATA_FIELDS):
        MEDIA_ID: Final = VideoFields.MEDIA_ID
        VID_NUM_FRAMES: Final = VideoFields.VID_NUM_FRAMES

        TYPES = {
            **VisionDataset.METADATA_FIELDS.TYPES,
            MEDIA_ID: str,
            VID_NUM_FRAMES: int
        }

    class ANNOTATIONS_FIELDS(VisionDataset.ANNOTATIONS_FIELDS):
        VID_FRAME_NUM: Final = VideoFields.VID_FRAME_NUM

        TYPES = {
            **VisionDataset.ANNOTATIONS_FIELDS.TYPES,
            VID_FRAME_NUM: int
        }

    """
    MEDIA FIELD NAMES FOR VIDEOS
    """
    FILES_EXTS: Final = [".avi", ".mp4"]
    DEFAULT_EXT: Final = ".mp4"
    FIELDS_TO_REMOVE_IN_MEDIA_LEVEL_DS: Final = (
        VisionDataset.FIELDS_TO_REMOVE_IN_MEDIA_LEVEL_DS + [ANNOTATIONS_FIELDS.VID_FRAME_NUM])
    DETS_FIELDS_TO_USE_IN_OBJ_LEVEL_DS = {*VisionDataset.DETS_FIELDS_TO_USE_IN_OBJ_LEVEL_DS,
                                          ImageFields.PARENT_VID_ID,
                                          VideoFields.VID_FRAME_NUM}

    def create_frames_dataset(self,
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
        items = self.items

        frame_numbers_fn = None
        time_positions_fn = None
        if frame_numbers is not None:
            msg = f'Items in frame_numbers are not present in the dataset'
            assert len(set(frame_numbers.keys()) & set(items)) == len(frame_numbers), msg

            def frame_numbers_fn(record):
                return frame_numbers[record[VideoFields.ITEM]]
            logger.info(f"Converting {len(self.items)} videos to frames, "
                        f"given frame numbers for each video.")
        elif time_positions is not None:
            msg = f'Items in time_positions are not present in the dataset'
            assert len(set(time_positions.keys()) & set(items)) == len(time_positions), msg

            def time_positions_fn(record):
                return time_positions[record[VideoFields.ITEM]]
            logger.info(f"Converting {len(self.items)} videos to frames, "
                        f"given time positions for each video.")
        else:
            logger.info(f"Converting {len(self.items)} videos to frames, "
                        f"sampling {freq_sampling} frames per second.")

        tic = time.time()
        # To get only records from unique videos
        media_level_ds = self.create_media_level_ds()
        n_stems = len(set([Path(elem).stem for elem in items]))
        method = 'stems' if n_stems == len(items) else VideoFields.MEDIA_ID
        get_frams_dir = partial(get_directory_for_frames, base_folder=frames_folder, method=method)

        parallel_exec(
            func=video_to_frames,
            elements=media_level_ds.records,
            input_video_file=lambda rec: rec[VideoFields.ITEM],
            output_folder=lambda rec: get_frams_dir(rec),
            freq_sampling=freq_sampling,
            frame_numbers=frame_numbers_fn,
            time_positions=time_positions_fn,
            videos_data=videos_data,
            overwrite=False,
            zero_based_indexing=zero_based_indexing,
            verify_if_the_first_frame_exists_to_skip=verify_1st_frame_to_skip)

        logger.debug(f'Conversion of videos into frames took {time.time()- tic:.2f} seconds.')

        df = media_level_ds.df.drop_duplicates(
            VideoFields.ITEM, inplace=False).set_index(VideoFields.ITEM)
        imgs_data = defaultdict(list)

        for video_item, video_data in videos_data.items():
            vid_rec = df.loc[video_item]
            frames_list = video_data['frames_filenames']
            frames_num_video = video_data['frames_num_video']
            n_vid_frames = video_data['frame_count']
            width = video_data['width']
            height = video_data['height']
            n_seq_frames = len(frames_list)

            ids = [get_random_id() for _ in range(n_seq_frames)]
            image_ids = [
                get_image_id_for_frame(vid_rec[VideoFields.MEDIA_ID], frames_list[i])
                for i in range(n_seq_frames)]
            video_ids = [vid_rec[VideoFields.MEDIA_ID]] * n_seq_frames
            vid_num_frames = [n_vid_frames] * n_seq_frames
            widths = [width] * n_seq_frames
            heights = [height] * n_seq_frames

            if VideoFields.LABEL in vid_rec:
                labels = [vid_rec[VideoFields.LABEL]] * n_seq_frames
                imgs_data[ImageFields.LABEL].extend(labels)
            imgs_data[ImageFields.ITEM].extend(frames_list)
            imgs_data[ImageFields.ID].extend(ids)
            imgs_data[ImageFields.MEDIA_ID].extend(image_ids)
            imgs_data[ImageFields.PARENT_VID_ID].extend(video_ids)
            imgs_data[ImageFields.PARENT_VID_FRAME_NUM].extend(frames_num_video)
            imgs_data[ImageFields.PARENT_VID_NUM_FRAMES].extend(vid_num_frames)
            imgs_data[ImageFields.WIDTH].extend(widths)
            imgs_data[ImageFields.HEIGHT].extend(heights)

        data = pd.DataFrame(imgs_data)
        imgs_ds = ImageDataset.from_dataframe(data, root_dir=frames_folder)
        return imgs_ds

    @classmethod
    def create_videos_from_frames_ds(cls,
                                     image_ds: ImageDataset,
                                     dest_folder: str,
                                     freq_sampling: int = 1,
                                     original_vids_ds=None,
                                     force_videos_creation: bool = False):
        """Create videos from the ImageDataset `image_ds`, assuming that the videos was previously
        sampled with a sampling frequency `freq_sampling`. The created videos will be stored in
        `dest_folder`

        Parameters
        ----------
        image_ds : ImageDataset
            Image dataset with the information of the video frames to be created. Ideally it was
            previously created with the `create_frames_dataset` method
        dest_folder : str
            Path to the directory where the created videos will be saved
        freq_sampling : int, optional
            Number of frames per second taken from the videos when divided into frames,
            by default 1
        split_in_labels : bool, optional
            Whether or not to split the created videos into folders according to the label in the
            'label' column, by default True

        """
        assert_cond = (VideoFields.VID_FRAME_NUM in image_ds.fields
                       and VideoFields.MEDIA_ID in image_ds.fields)
        assert_str = (
            f"The dataset must contain the {VideoFields.VID_FRAME_NUM} and {VideoFields.MEDIA_ID} "
            f"columns")
        assert assert_cond, assert_str
        os.makedirs(dest_folder, exist_ok=True)

        logger.info(f"Creating videos from frames and storing them in the path: {dest_folder}")

        if original_vids_ds is not None:
            original_vids_df = original_vids_ds.df
            original_vids_dir = original_vids_ds.root_dir

        df = image_ds.df

        def get_vid_file(vid_id):
            if original_vids_ds is not None:
                vid_rec = original_vids_df[original_vids_df.video_id == vid_id].iloc[0]
                if original_vids_dir is not None:
                    fname = os.path.relpath(vid_rec[VideoFields.ITEM], original_vids_dir)
                else:
                    fname = vid_rec[VideoFields.ITEM]
            elif not str(vid_id).lower().endswith(cls.DEFAULT_EXT):
                fname = f"{vid_id}{cls.DEFAULT_EXT}"
            else:
                fname = f"{get_random_id()}{cls.DEFAULT_EXT}"
            return os.path.join(dest_folder, fname)

        def _get_imgs(vid_id):
            return (df[df.video_id == vid_id]
                    .sort_values(VideoFields.VID_FRAME_NUM)[VideoFields.ITEM]
                    .values)
        vids_ids = df[VideoFields.MEDIA_ID].unique()
        parallel_exec(
            func=frames_to_video,
            elements=vids_ids,
            images=_get_imgs,
            freq_sampling=freq_sampling,
            output_file_name=get_vid_file,
            force=force_videos_creation)

    def create_crops_dataset(self: VideoDataset,
                             frames_folder: str,
                             dest_path: str = None,
                             use_partitions=False,
                             allow_label_empty: bool = False,
                             force_crops_creation: bool = False,
                             force_frames_creation: bool = False,
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
        assert_cond = VideoFields.VID_FRAME_NUM in self.fields
        assert_msg = f"The dataset must contain the field {VideoFields.VID_FRAME_NUM}"
        assert assert_cond, assert_msg

        df = self.df

        if force_frames_creation:
            frame_numbers = (
                df[[VideoFields.ITEM, VideoFields.VID_FRAME_NUM]]
                .groupby(VideoFields.ITEM)
                .apply(lambda x: sorted(set(x[VideoFields.VID_FRAME_NUM].tolist())))
                .to_dict())
            self.create_frames_dataset(frames_folder, frame_numbers=frame_numbers)

        df[VideoFields.ITEM] = df.apply(get_frame_item, axis=1, frames_folder=frames_folder)

        frames_ds = ImageDataset.from_dataframe(df, root_dir=frames_folder,
                                                validate_filenames=False)
        crops_ds = frames_ds.create_crops_dataset(
            dest_path=dest_path,
            use_partitions=use_partitions,
            allow_label_empty=allow_label_empty,
            force_crops_creation=force_crops_creation,
            dims_correction=dims_correction,
            bottom_offset=bottom_offset)
        return crops_ds

    def draw_bounding_boxes(self,
                            freq_sampling: int,
                            frames_folder: str = None,
                            include_labels: bool = False,
                            include_scores: bool = False,
                            blur_people: bool = False,
                            thickness: int = None):
        frames_folder = frames_folder or get_temp_folder()

        frames_ds = self.create_frames_dataset(
            frames_folder=frames_folder, freq_sampling=freq_sampling)

        frames_w_boxes_ds = frames_ds.create_object_level_dataset_using_detections(
            self, fields_for_merging=[self.METADATA_FIELDS.MEDIA_ID, VideoFields.VID_FRAME_NUM])
        frames_w_boxes_ds.draw_bounding_boxes(include_labels=include_labels,
                                              include_scores=include_scores,
                                              blur_people=blur_people,
                                              thickness=thickness)

        VideoDataset.create_videos_from_frames_ds(
            image_ds=frames_ds,
            dest_folder=self.root_dir,
            freq_sampling=freq_sampling,
            original_vids_ds=self,
            force_videos_creation=True)

    @classmethod
    def _download(cls, dest_path, media_base_url, media_dict, set_filename_with_id_and_ext):
        raise ValueError('Not implemented method')

    def _get_media_dims_of_items(self, items):
        raise Exception('Not implemented method')
