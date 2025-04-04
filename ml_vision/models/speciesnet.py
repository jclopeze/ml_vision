#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
from collections import defaultdict
import json
from multiprocessing import Manager
import os
import pandas as pd
import subprocess

from ml_base.model import Model
from ml_base.utils.misc import get_temp_folder
from ml_base.utils.misc import parallel_exec, delete_dirs
from ml_base.utils.logger import get_logger
from ml_base.utils.dataset import get_random_id

from ml_vision.datasets import ImageDataset
from ml_vision.datasets import VideoDataset
from ml_vision.datasets import VisionDataset
from ml_vision.utils.coords import transform_coordinates
from ml_vision.utils.coords import CoordinatesFormat, CoordinatesDataType, CoordinatesType
from ml_vision.utils.video import get_frame_numbers_from_vids
from ml_vision.utils.vision import VisionFields as VFields

logger = get_logger(__name__)


class SpeciesNet(Model):
    def __init__(self,
                 script_path,
                 version,
                 ):
        self.version = version
        self.script_path = script_path

    @classmethod
    def load_model(cls,
                   script_path,
                   version=None,) -> SpeciesNet:
        return SpeciesNet(script_path=script_path, version=version)

    def inference(self,
                  dataset: ImageDataset,
                  detections: ImageDataset,
                  country: str = None) -> str:

        dets_json = os.path.join(get_temp_folder(), f"{get_random_id()}-dets.json")
        classifs_json = os.path.join(get_temp_folder(), f"{get_random_id()}-classifs.json")
        ensemble_json = os.path.join(get_temp_folder(), f"{get_random_id()}-ensemble.json")
        detections = ImageDatasetSpeciesNet.cast(detections)
        detections.to_json(dets_json)

        PYTHON_EXEC = os.environ.get('PYTHON_EXEC', 'python')

        # Execute the classifier
        cmd = [PYTHON_EXEC, self.script_path,
               '--classifier_only',
               '--folders', dataset.root_dir,
               '--predictions_json', classifs_json,
               '--detections_json',  dets_json,
               ]
        if country is not None:
            cmd += ['--country', f'{country}']
        logger.debug(f"Running SpeciesNet classifier with the command: {' '.join(cmd)}")

        retcode = subprocess.call(cmd)

        if retcode != 0:
            raise Exception(f"SpeciesNet executed and returned status code: {retcode}")

        # Execute the ensemble
        cmd = [PYTHON_EXEC, self.script_path,
               '--ensemble_only',
               '--folders', dataset.root_dir,
               '--predictions_json', ensemble_json,
               '--detections_json',  dets_json,
               '--classifications_json', classifs_json
               ]
        if country is not None:
            cmd += ['--country', f'{country}']
        logger.debug(f"Running SpeciesNet ensemble with the command: {' '.join(cmd)}")

        retcode = subprocess.call(cmd)

        if retcode != 0:
            raise Exception(f"SpeciesNet executed and returned status code: {retcode}")

        return ensemble_json

    def predict(self,
                dataset: VisionDataset,
                detections: VisionDataset,
                country: str = None,
                threshold: float = 0.01,
                frames_folder: str = None) -> VisionDataset:
        """Method that performs the prediction of the Megadetector on the images in `dataset`

        Parameters
        ----------
        dataset : ImageDataset
            ImagesDataset with the images on which the Megadetector will perform the prediction
        threshold : float
            Minimum score of detections to be considered in results

        Returns
        -------
        ImagePredictionDataset
            Image dataset with Megadetector predictions
        """

        if dataset.is_empty:
            logger.debug("No data to detect")
            return dataset

        classifs_imgs_ds = SpeciesNetImage.predict(model=self,
                                                   dataset=dataset.images_ds,
                                                   detections=detections.images_ds,
                                                   country=country,
                                                   threshold=threshold)
        classifs_vids_ds = SpeciesNetVideo.predict(model=self,
                                                   dataset=dataset.videos_ds,
                                                   detections=detections.videos_ds,
                                                   country=country,
                                                   threshold=threshold,
                                                   frames_folder=frames_folder)

        return type(dataset).from_datasets(classifs_imgs_ds, classifs_vids_ds)

    def classify(self,
                 dataset: VisionDataset,
                 detections: VisionDataset = None,
                 country: str = None,
                 threshold: float = 0.01,
                 frames_folder: str = None):

        classifs_ds = self.predict(dataset=dataset,
                                   detections=detections,
                                   country=country,
                                   threshold=threshold,
                                   frames_folder=frames_folder)

        classif_ds = self.classify_dataset_using_seqs(dataset=dataset, classifications=classifs_ds)

        return classif_ds

    @classmethod
    def _get_seq_level_classif(cls,
                               classifs_df: pd.DataFrame,
                               seq_id: str,
                               item_to_label_score: dict):
        items_seq = classifs_df[classifs_df[VFields.SEQ_ID] == seq_id]
        taxa_levels = ['species', 'genus', 'family', 'order', 'class', 'kingdom', 'empty']

        label, taxonomy_level, score = 'empty', 'empty', 1.
        for taxa_level in taxa_levels:
            items_taxa_lvl = items_seq[items_seq['taxonomy_level'] == taxa_level]
            if len(items_taxa_lvl) == 0:
                continue

            taxonomy_level = taxa_level
            label_modes = items_taxa_lvl['label'].mode()
            if len(label_modes) > 1:
                highest_mode = (
                    items_taxa_lvl[items_taxa_lvl["label"].isin(label_modes.values)]
                    .sort_values(by='score', ascending=False)
                    .iloc[0]
                )
                label = highest_mode['label']
                score = highest_mode['score']
            else:
                label = label_modes.iloc[0]
                score = items_taxa_lvl[items_taxa_lvl.label == label].score.max()
            break

        item_to_label_score[seq_id] = {
            'label': label,
            'taxonomy_level': taxonomy_level,
            'score': score
        }

    @classmethod
    def classify_dataset_using_seqs(cls,
                                    dataset: VisionDataset,
                                    classifications: VisionDataset) -> VisionDataset:

        item_to_label_score = Manager().dict()
        parallel_exec(
            func=cls._get_seq_level_classif,
            elements=dataset['seq_id'].unique(),
            classifs_df=classifications.df,
            seq_id=lambda seq_id: seq_id,
            item_to_label_score=item_to_label_score)

        classif_ds = type(dataset).cast(dataset)
        classif_ds[VFields.LABEL] = lambda x: item_to_label_score[x[VFields.SEQ_ID]]['label']
        classif_ds['taxonomy_level'] = lambda x: item_to_label_score[x[VFields.SEQ_ID]]['taxonomy_level']
        classif_ds[VFields.SCORE] = lambda x: item_to_label_score[x[VFields.SEQ_ID]]['score']

        return classif_ds

    def train(self,
              dataset,
              epochs,
              batch_size,
              **kwargs):
        raise NotImplementedError


class SpeciesNetImage(SpeciesNet):

    @staticmethod
    def predict(model: SpeciesNet,
                dataset: ImageDataset,
                detections: VisionDataset,
                country: str = None,
                threshold: float = 0.01,
                move_files_to_temp_folder=True) -> ImageDataset:
        """Method that performs the prediction of the Megadetector on the images in `dataset`

        Parameters
        ----------
        dataset : ImageDataset
            ImagesDataset with the images on which the Megadetector will perform the prediction
        threshold : float
            Minimum score of detections to be considered in results

        Returns
        -------
        ImagePredictionDataset
            Image dataset with Megadetector predictions
        """
        if dataset.is_empty:
            return ImageDatasetSpeciesNet(annotations=None, metadata=None)

        _dataset = dataset.copy()
        _detections = detections.copy()
        if move_files_to_temp_folder:
            # We need to be sure that only the files of the dataset are stored in root_dir
            temp_folder = os.path.join(get_temp_folder(), f"images-{get_random_id()}")
            _dataset.to_folder(
                temp_folder,
                use_labels=False,
                move_files=False,                                                                   #######
                preserve_directory_hierarchy=True,
                update_dataset_filepaths=True)
            _detections._set_abspaths_and_validate_filenames(temp_folder, validate_filenames=True)

        classifs_json = model.inference(_dataset, detections=_detections, country=country)

        classifs_ds = ImageDatasetSpeciesNet.from_json(classifs_json,
                                                       metadata=_dataset.metadata.copy(),
                                                       root_dir=_dataset.root_dir,
                                                       min_score=threshold)

        if move_files_to_temp_folder:
            classifs_ds.to_folder(
                dataset.root_dir,
                use_labels=False,
                move_files=True,
                preserve_directory_hierarchy=True,
                update_dataset_filepaths=True)

        return classifs_ds


class SpeciesNetVideo(SpeciesNet):

    @staticmethod
    def predict(model,
                dataset: VideoDataset,
                detections: VisionDataset,
                country: str = None,
                threshold: float = 0.01,
                frames_folder: str = None,
                delete_frames_folder_on_finish: bool = True,
                batch_size: int = 500) -> VideoDataset:
        if dataset.is_empty:
            return ImageDatasetSpeciesNet(annotations=None, metadata=None)

        frame_numbers = get_frame_numbers_from_vids(detections.df)

        dets_vids_dss = []
        for ds in dataset.batch_gen(batch_size):
            frames_ds = VideoDataset.create_frames_dataset(
                ds, frames_folder, frame_numbers=frame_numbers, add_new_file_id=False)
            frames_ds[VFields.LABEL] = 'animal'
            frames_with_bboxes_ds = frames_ds.create_object_level_dataset_using_detections(
                detections, fields_for_merging=[VFields.FILE_ID, VFields.VID_FRAME_NUM])

            dets_frames_ds = SpeciesNetImage.predict(model=model,
                                                     dataset=frames_ds,
                                                     detections=frames_with_bboxes_ds,
                                                     country=country,
                                                     threshold=threshold,
                                                     move_files_to_temp_folder=False)

            mapper = frames_ds.df.set_index(VFields.ITEM)[VFields.VID_FRAME_NUM]
            dets_frames_ds[VFields.VID_FRAME_NUM] = lambda record: mapper.loc[record[VFields.ITEM]]

            _dets_vids_ds = ds.create_object_level_dataset_using_detections(
                dets_frames_ds,
                use_detections_labels=True,
                additional_fields_from_detections=['taxonomy_level'])
            dets_vids_dss.append(_dets_vids_ds)

            if delete_frames_folder_on_finish:
                frames_dirs = list(set([os.path.dirname(it) for it in frames_ds.items]))
                delete_dirs(frames_dirs)
                delete_dirs(frames_ds.root_dir)

        dets_vids_ds = type(dataset).from_datasets(*dets_vids_dss)

        return dets_vids_ds


class ImageDatasetSpeciesNet(ImageDataset):
    taxa_levels = {
        -2: 'species',
        -3: 'genus',
        -4: 'family',
        -5: 'order',
        -6: 'class'
    }
    labelmap_detector = {
        "1": "animal",
        "2": "person",  # "human"
        "3": "vehicle",
    }

    class AnnotationFields(VisionDataset.AnnotationFields):
        TAXA_LEVEL = 'taxonomy_level'

    TAXA_LEVEL = 'taxonomy_level'

    def to_json(self, dest_path: str) -> dict:

        images_dims = self.get_media_dims().set_index(VFields.ITEM)
        inv_labelmap = {v: k for k, v in ImageDatasetSpeciesNet.labelmap_detector.items()}

        item_to_dets = Manager().dict()
        parallel_exec(
            func=ImageDatasetSpeciesNet.append_detections,
            elements=self.items,
            item=lambda item: item,
            df=self.df,
            item_to_dets=item_to_dets,
            inv_labelmap=inv_labelmap,
            images_dims=images_dims)
        predictions = [
            {
                "filepath": item,
                "detections": dets
            } for item, dets in item_to_dets.items()
        ]
        predictions_dict = {"predictions": predictions}
        with open(dest_path, 'w') as f:
            json.dump(predictions_dict, f, ensure_ascii=False, indent=4)

        return predictions_dict

    @staticmethod
    def append_detections(item: str,
                          df: pd.DataFrame,
                          item_to_dets: dict,
                          inv_labelmap: dict,
                          images_dims: pd.DataFrame):

        rows_item = df[df[VFields.ITEM] == item].sort_values(VFields.SCORE, ascending=False)

        if len(rows_item) == 0:
            return

        detections = []
        for _, record in rows_item.iterrows():
            bbox = transform_coordinates(
                record[VFields.BBOX],
                input_format=CoordinatesFormat.x_y_width_height,
                output_format=CoordinatesFormat.x_y_width_height,
                output_coords_type=CoordinatesType.relative,
                output_data_type=CoordinatesDataType.array,
                media_width=images_dims.loc[item][VFields.WIDTH],
                media_height=images_dims.loc[item][VFields.HEIGHT])
            label = record[VFields.LABEL]
            det = {
                "category": str(inv_labelmap[label]),
                "label": label,
                "conf": record[VFields.SCORE],
                "bbox": bbox
            }
            detections.append(det)

        item_to_dets[item] = detections

    @classmethod
    def from_json(cls,
                  source_path: str,
                  metadata: pd.DataFrame,
                  root_dir: str,
                  **kwargs) -> ImageDataset:
        json_data = json.load(open(source_path, 'r'))

        data = defaultdict(list)
        for prediction in json_data['predictions']:
            item = prediction["filepath"]
            if not "DETECTOR" in prediction.get('failures', ''):
                label_pred = prediction["prediction"]
                label, taxa_level = ImageDatasetSpeciesNet._get_label_and_taxa_level(label_pred)
                score = prediction["prediction_score"]
            else:
                label, taxa_level = 'animalia', 'kingdom'
                score = 1.
            id = get_random_id()

            data[VFields.ITEM].append(item)
            data[VFields.LABEL].append(label)
            data[cls.AnnotationFields.TAXA_LEVEL].append(taxa_level)
            data[VFields.SCORE].append(score)
            data[VFields.ID].append(id)

        annotations = pd.DataFrame(data)
        prediction_dataset = cls(annotations,
                                 metadata,
                                 root_dir,
                                 **kwargs)
        return prediction_dataset

    @staticmethod
    def _get_label_and_taxa_level(prediction_str):
        prediction_list = prediction_str.split(';')
        if prediction_list[-1] == 'blank':
            return 'empty', 'empty'
        if prediction_list[-1] ==  'no cv result':
            return 'animalia', 'kingdom'
        if prediction_list[-2] != '':
            return (f'{prediction_list[-3]} {prediction_list[-2]}',
                    ImageDatasetSpeciesNet.taxa_levels[-2])
        for i in range(-3, -7, -1):
            if prediction_list[i] != '':
                return prediction_list[i], ImageDatasetSpeciesNet.taxa_levels[i]
        return 'animalia', 'kingdom'
