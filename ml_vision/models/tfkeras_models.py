#!/usr/bin/env python
# -*- coding: utf-8 -*-
import collections
from multiprocessing import Manager
import os
import pandas as pd
from collections import defaultdict
import uuid

from ml_base.model import Model as BaseModel
from ml_base.dataset import Partitions
from ml_base.utils.dataset import get_sorted_df
from ml_base.utils.dataset import read_labelmap_file
from ml_base.utils.dataset import write_labelmap_file
from ml_base.utils.logger import get_logger, debugger
from ml_base.utils.misc import parallel_exec

from ml_vision.datasets import ImageDataset, VisionDataset
from ml_vision.utils.classification import set_label_and_score_for_item_in_ensemble
from ml_vision.utils.vision import VisionFields as Fields


import tensorflow as tf
import keras
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB4, EfficientNetB5
from tensorflow.keras.applications import EfficientNetB6, EfficientNetB7
from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2
from tensorflow.keras.applications import EfficientNetV2B3

logger = get_logger(__name__)
debug = debugger.debug


class TFKerasBasicModel(BaseModel):

    ModelSpecs = collections.namedtuple(
        "ModelSpecs", ['name', 'input_size', 'model_base_instance'])

    efficientnet_b0 = ModelSpecs(
        name='efficientnet-b0',
        input_size=224,
        model_base_instance=EfficientNetB0)

    efficientnet_b1 = ModelSpecs(
        name='efficientnet-b1',
        input_size=240,
        model_base_instance=EfficientNetB1)

    efficientnet_b2 = ModelSpecs(
        name='efficientnet-b2',
        input_size=260,
        model_base_instance=EfficientNetB2)

    efficientnet_b3 = ModelSpecs(
        name='efficientnet-b3',
        input_size=300,
        model_base_instance=EfficientNetB3)

    efficientnet_b4 = ModelSpecs(
        name='efficientnet-b4',
        input_size=380,
        model_base_instance=EfficientNetB4)

    efficientnet_b5 = ModelSpecs(
        name='efficientnet-b5',
        input_size=456,
        model_base_instance=EfficientNetB5)

    efficientnet_b6 = ModelSpecs(
        name='efficientnet-b6',
        input_size=528,
        model_base_instance=EfficientNetB6)

    efficientnet_b7 = ModelSpecs(
        name='efficientnet-b7',
        input_size=600,
        model_base_instance=EfficientNetB7)

    efficientnetv2_b0 = ModelSpecs(
        name='efficientnetv2-b0',
        input_size=224,
        model_base_instance=EfficientNetV2B0)

    efficientnetv2_b1 = ModelSpecs(
        name='efficientnetv2-b1',
        input_size=240,
        model_base_instance=EfficientNetV2B1)

    efficientnetv2_b2 = ModelSpecs(
        name='efficientnetv2-b2',
        input_size=260,
        model_base_instance=EfficientNetV2B2)

    efficientnetv2_b3 = ModelSpecs(
        name='efficientnetv2-b3',
        input_size=300,
        model_base_instance=EfficientNetV2B3)

    MODELS_SPECS = {
        'efficientnet-b0': efficientnet_b0,
        'efficientnet-b1': efficientnet_b1,
        'efficientnet-b2': efficientnet_b2,
        'efficientnet-b3': efficientnet_b3,
        'efficientnet-b4': efficientnet_b4,
        'efficientnet-b5': efficientnet_b5,
        'efficientnet-b6': efficientnet_b6,
        'efficientnet-b7': efficientnet_b7,
        'efficientnetv2-b0': efficientnetv2_b0,
        'efficientnetv2-b1': efficientnetv2_b1,
        'efficientnetv2-b2': efficientnetv2_b2,
        'efficientnetv2-b3': efficientnetv2_b3
    }

    def __init__(self,
                 name,
                 labelmap_path=None,
                 checkpoint=None,
                 num_classes=None):
        assert name.lower() in self.MODELS_SPECS, f"Invalid model name: {name}"
        model_specs = self.MODELS_SPECS[name.lower()]

        self.name = name
        self.target_size = (model_specs.input_size, model_specs.input_size)
        self.model_base_instance = model_specs.model_base_instance
        self._model = None
        self.checkpoint = checkpoint
        if checkpoint is not None and labelmap_path is None:
            labelmap_path = os.path.join(os.path.dirname(checkpoint), 'labels.txt')
            logger.info(f'labelmap_path was not provided and {labelmap_path} is going to be used')
        self.labelmap_path = labelmap_path
        self.num_classes = num_classes

    @classmethod
    def load_model(cls, source_path: str) -> Model:
        pass

    def _get_callbacks(self, **kwargs):
        tensorboard_dir = kwargs.get('tensorboard_dir')
        early_stopping_pat = kwargs.get('early_stopping_patience', None)
        early_stopping_mon = kwargs.get('early_stopping_monitor', 'val_accuracy')
        model_ckpt_save_best_only = kwargs.get('model_checkpoint_save_best_only', False)
        model_ckpt_monitor = kwargs.get('model_checkpoint_monitor', 'val_loss')
        reducelr_factor = kwargs.get('reducelr_factor', None)
        reducelr_patience = kwargs.get('reducelr_patience', 5)

        callbacks = [ModelCheckpoint(self.checkpoint,
                                     monitor=model_ckpt_monitor,
                                     save_best_only=model_ckpt_save_best_only)]
        if tensorboard_dir is not None:
            callbacks.append(TensorBoard(log_dir=tensorboard_dir))
        if early_stopping_pat is not None:
            callbacks.append(EarlyStopping(monitor=early_stopping_mon,
                                           patience=early_stopping_pat))
        if reducelr_factor is not None:
            callbacks.append(ReduceLROnPlateau(monitor=model_ckpt_monitor,
                                               factor=reducelr_factor,
                                               patience=reducelr_patience))

        return callbacks

    def train(self,
              dataset,
              epochs,
              batch_size,
              **kwargs):
        force_training = kwargs.get('force_training', False)
        if not os.path.isfile(self.checkpoint) or force_training == True:
            callbacks = self._get_callbacks(**kwargs)

            train_batches, val_batches = self._get_train_val_data(dataset, batch_size, **kwargs)

            logger.info(f"Training model {self.name}")

            with tf.distribute.MirroredStrategy().scope():
                self._build_model(**kwargs)

            self._model.fit(train_batches,
                            epochs=epochs,
                            verbose=1,
                            validation_data=val_batches,
                            callbacks=callbacks)

        return self

    def _get_train_val_data(self, dataset, batch_size, **kwargs):
        val_partition = kwargs.get('val_partition')
        val_perc = kwargs.get('val_perc')
        val_groupby = kwargs.get('val_groupby', 'location')
        val_stratify = kwargs.get('val_stratify', True)
        val_rand_state = kwargs.get('val_rand_state')

        if val_partition is not None:
            train_df = dataset.filter_by_partition(Partitions.TRAIN, inplace=False).as_dataframe()
            val_df = dataset.filter_by_partition(val_partition, inplace=False).as_dataframe()
        elif val_perc is not None:
            dataset_cpy = dataset.copy()
            dataset_cpy.filter_by_partition(Partitions.TRAIN, mode='include', inplace=True)
            dataset_cpy.split(
                train_perc=1-val_perc, test_perc=0, val_perc=val_perc,
                groupby=val_groupby, stratify=val_stratify, random_state=val_rand_state)
            train_df = dataset_cpy.filter_by_partition(
                Partitions.TRAIN, inplace=False).as_dataframe()
            val_df = dataset_cpy.filter_by_partition(
                Partitions.VALIDATION, inplace=False).as_dataframe()
        else:
            train_df = dataset.filter_by_partition(Partitions.TRAIN, inplace=False).as_dataframe()

        classes = dataset.get_categories()
        train_batches = ImageDataGenerator().flow_from_dataframe(
            train_df,
            x_col="item",
            y_col="label",
            classes=classes,
            target_size=self.target_size,
            batch_size=batch_size,
            validate_filenames=False)

        if val_partition is not None or val_perc is not None:
            logger.info(f"Using validation data")
            validation_batches = ImageDataGenerator().flow_from_dataframe(
                val_df,
                x_col="item",
                y_col="label",
                classes=classes,
                target_size=self.target_size,
                batch_size=batch_size,
                validate_filenames=False,
                shuffle=False)
        else:
            validation_batches = None

        labelmap = {v: k for k, v in train_batches.class_indices.items()}
        write_labelmap_file(labelmap=labelmap, dest_path=self.labelmap_path)

        return train_batches, validation_batches

    def _build_model(self, **kwargs):
        assert self.model_base_instance is not None, "model_base_instance not assigned"
        assert self.num_classes is not None, "For model training you must assign num_classes"

        img_augmentation = self._get_augmentations(**kwargs)

        inputs = Input(shape=(self.target_size[0], self.target_size[1], 3))

        x = img_augmentation(inputs)
        model = self.model_base_instance(include_top=False, input_tensor=x, weights="imagenet")

        # Freeze the pretrained weights
        model.trainable = False

        # Rebuild top
        x = GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = BatchNormalization()(x)

        top_dropout_rate = 0.2
        x = Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = Dense(self.num_classes, activation="softmax", name="pred")(x)

        self._model = Model(inputs, outputs, name=self.name)

        # Compile
        optimizer = self._get_optimizer(**kwargs)
        loss_fn = self._get_loss_fn(**kwargs)
        metrics = self._get_metrics(**kwargs)
        self._model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics)

    def _get_loss_fn(self, **kwargs):
        return 'categorical_crossentropy'

    def _get_optimizer(self, **kwargs):
        lr = kwargs.get('learning_rate', 1e-2)
        optimizer = Adam(learning_rate=lr)
        return optimizer

    def _get_metrics(self, **kwargs):
        other_metrics = kwargs.get('metrics', 'precision,recall')
        metrics = ["accuracy"]

        if 'precision' in other_metrics:
            metrics.append(keras.metrics.Precision(name='precision'))
        if 'recall' in other_metrics:
            metrics.append(keras.metrics.Recall(name='recall'))

        return metrics

    def _get_augmentations(self, **kwargs):
        rot_factor = kwargs.get('augm_rotation_factor', 0.15)
        transl_h = kwargs.get('augm_translation_height_factor', 0.2)
        transl_w = kwargs.get('augm_translation_width_factor', 0.2)
        flip_mode = kwargs.get('augm_flip_mode', "horizontal")
        contr_factor = kwargs.get('augm_contrasct_factor', 0.2)
        zoom_factor = kwargs.get('augm_zoom_factor', 0.2)
        bright_l_factor = kwargs.get('augm_brightness_lower_factor', -0.2)
        bright_u_factor = kwargs.get('augm_brightness_upper_factor', 0.2)

        augms = Sequential(
            [
                tf.keras.layers.RandomRotation(factor=rot_factor),
                tf.keras.layers.RandomTranslation(
                    height_factor=transl_h, width_factor=transl_w),
                tf.keras.layers.RandomFlip(mode=flip_mode),
                tf.keras.layers.RandomContrast(factor=contr_factor),
                tf.keras.layers.RandomZoom(zoom_factor),
                tf.keras.layers.RandomBrightness([bright_l_factor, bright_u_factor])
            ],
            name="img_augmentation",
        )

        return augms

    def predict(self, dataset: VisionDataset):
        assert self.labelmap_path is not None

        batch_size = 32

        batches = ImageDataGenerator().flow_from_dataframe(
            dataset.df,
            x_col=Fields.ITEM,
            class_mode=None,
            target_size=self.target_size,
            batch_size=batch_size,
            shuffle=False,
            validate_filenames=False)

        with tf.distribute.MirroredStrategy().scope():
            self._model = load_model(self.checkpoint)

        preds = self._model.predict(batches, batch_size=batch_size)
        labelmap = read_labelmap_file(self.labelmap_path)

        labels_mapper = {}
        scores_mapper = {}
        for i, record in enumerate(dataset):
            sorted_idx = [y[0]
                          for y in sorted(enumerate(preds[i]), key=lambda x: x[1], reverse=True)]
            ind = sorted_idx[0]
            label = labelmap[ind]
            score = preds[i][ind]

            labels_mapper[record[Fields.ID]] = label
            scores_mapper[record[Fields.ID]] = score

        preds_ds = dataset.copy()
        preds_ds[Fields.LABEL] = lambda record: labels_mapper[record[Fields.ID]]
        preds_ds[Fields.SCORE_DET] = lambda record: record[Fields.SCORE]
        preds_ds[Fields.SCORE] = lambda record: scores_mapper[record[Fields.ID]]

        return preds_ds

    def classify(self, dataset: VisionDataset) -> VisionDataset:
        predictions = self.predict(dataset)

        def _get_label_score(preds_df: pd.DataFrame, seq_id: str, seq_id_to_label_score: dict):
            preds_seq = preds_df[preds_df[Fields.SEQ_ID] == seq_id]
            preds_seq[Fields.SCORE] = preds_seq.apply(
                lambda rec: rec[Fields.SCORE] * rec[Fields.SCORE_DET], axis=1)

            highest_pred = preds_seq.sort_values(by=Fields.SCORE, ascending=False).iloc[0]

            seq_id_to_label_score[seq_id] = {
                Fields.LABEL: highest_pred[Fields.LABEL],
                Fields.SCORE: highest_pred[Fields.SCORE]
            }

        seq_id_mapper = Manager().dict()
        seqs_ids = predictions[Fields.SEQ_ID].unique()

        parallel_exec(
            func=_get_label_score,
            elements=seqs_ids,
            preds_df=predictions.df,
            seq_id=lambda seq_id: seq_id,
            seq_id_to_label_score=seq_id_mapper)

        predictions[Fields.LABEL] = lambda rec: seq_id_mapper[rec[Fields.SEQ_ID]][Fields.LABEL]
        predictions[Fields.SCORE] = lambda rec: seq_id_mapper[rec[Fields.SEQ_ID]][Fields.SCORE]

        return predictions

    @classmethod
    def ensemble_predictions_of_models(cls,
                                       models_names,
                                       models_preds=None,
                                       models_preds_csvs=None,
                                       models_weights=None,
                                       dest_classifs_csv=None,
                                       images_dir=None):
        if dest_classifs_csv is not None and os.path.isfile(dest_classifs_csv):
            return ImageDataset.from_csv(dest_classifs_csv, images_dir=images_dir)

        assert_msg = "You must specify either models_preds or models_preds_csvs"
        assert bool(models_preds) + bool(models_preds_csvs) == 1, assert_msg

        models_to_dfs = {}
        if models_preds_csvs is not None:
            for model_name, model_preds_csv in zip(models_names, models_preds_csvs):
                ds = ImageDataset.from_csv(model_preds_csv)
                models_to_dfs[model_name] = ds.as_dataframe()
        else:
            for model_name, model_preds in zip(models_names, models_preds):
                ds = model_preds
                models_to_dfs[model_name] = ds.as_dataframe()

        logger.info(f"Doing ensemble of models {', '.join(models_names)}")
        items = ds.items
        # TODO: Check this
        base_df = get_sorted_df(ds.df, only_highest_score=True)

        item_to_lbls_scores = Manager().dict()

        parallel_exec(func=set_label_and_score_for_item_in_ensemble,
                      elements=items,
                      item=lambda elem: elem,
                      models_to_dfs=models_to_dfs,
                      item_to_lbls_scrs=item_to_lbls_scores,
                      model_weights=models_weights)

        data = defaultdict(list)
        rest_cols = list(set(base_df.columns) - {'label', 'score'})

        for record in base_df.to_dict('records'):
            lbl_score = item_to_lbls_scores[record["item"]]
            data["label"].append(lbl_score['label'])
            data["score"].append(lbl_score['score'])
            for col in rest_cols:
                data[col].append(record[col])

        data_df = pd.DataFrame(data)
        res_ds = ImageDataset(data_df, images_dir=images_dir)
        if dest_classifs_csv is not None:
            res_ds.to_csv(dest_path=dest_classifs_csv)

        return res_ds
