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

from ml_vision.datasets.image import ImageDataset
from ml_vision.utils.classification import set_label_and_score_for_item_in_ensemble


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

    def classify_images_dataset(self,
                                dataset,
                                batch_size,
                                out_preds_csv=None,
                                partition=None,
                                max_classifs=1,
                                mappings_labels={},
                                **kwargs):
        if out_preds_csv is not None and os.path.isfile(out_preds_csv):
            images_dir = dataset.get_images_dir()
            return ImageDataset.from_csv(out_preds_csv, images_dir=images_dir)
        assert self.labelmap_path is not None, "labelmap_path has not been assigned for the model"
        labelmap = read_labelmap_file(self.labelmap_path)
        dataset_part = dataset.filter_by_partition(partition, inplace=False)
        df = (
            dataset_part.as_dataframe()
            .reset_index(drop=True))
        img_gen = ImageDataGenerator()
        batches = img_gen.flow_from_dataframe(
            df,
            x_col="item",
            class_mode=None,
            target_size=self.target_size,
            batch_size=batch_size,
            shuffle=False,
            validate_filenames=False)

        with tf.distribute.MirroredStrategy().scope():
            self._model = load_model(self.checkpoint)

        preds = self._model.predict(batches, batch_size=batch_size)

        fields_of_result = {"item", "label", "score", "image_id", "id", "partition"}
        anns_extra_flds = set()
        # FIXME
        anns_extra_flds |= (
            (set(dataset._get_default_cols_of_anns_data()) & set(df.columns)) - fields_of_result)
        anns_extra_flds = list(anns_extra_flds)

        # TODO: Create a copy of dataset and modify the following fields instead. Delete BBOX, etc.
        ann_info = defaultdict(list)
        for i, record in enumerate(dataset_part):
            srtd_idx = [y[0] for y in sorted(enumerate(preds[i]), key=lambda x:x[1], reverse=True)]
            for k in range(max_classifs):
                ind = srtd_idx[k]
                label = mappings_labels.get(labelmap[ind], mappings_labels.get('*', labelmap[ind]))
                # ann_info["item"].append(record["item"])
                ann_info["label"].append(label)
                ann_info["score"].append(preds[i][ind])
                # ann_info["image_id"].append(record["image_id"])
                ann_info["id"].append(str(uuid.uuid4()))
                for anns_extra_field in anns_extra_flds:
                    ann_info[anns_extra_field].append(record[anns_extra_field])
        anns_data_df = pd.DataFrame(ann_info)
        metadata = dataset.metadata.copy()
        root_dir = dataset.root_dir

        classifs_ds = ImageDataset(
            annotations=anns_data_df, metadata=metadata, root_dir=root_dir, **kwargs)

        if out_preds_csv is not None:
            classifs_ds.to_csv(out_preds_csv)

        return classifs_ds

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


class TFKerasModel(TFKerasBasicModel):

    def _get_train_val_data(self, dataset, batch_size, **kwargs):
        val_partition = kwargs.get('val_partition')
        val_perc = kwargs.get('val_perc')
        val_groupby = kwargs.get('val_groupby', 'location')
        val_stratify = kwargs.get('val_stratify', True)
        val_rand_state = kwargs.get('val_rand_state')

        inverse_labelmap = dataset._get_inverse_labelmap()
        write_labelmap_file(labelmap=dataset.get_labelmap(), dest_path=self.labelmap_path)
        classes = dataset.get_categories()
        num_classes = len(classes)

        def parse_image(filename):
            image = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, [self.target_size[0], self.target_size[1]])
            image = tf.round(image)
            return image

        def parse_labels(label):
            return tf.one_hot(label, num_classes)

        rot_factor = kwargs.get('augm_rotation_factor', 0.15)
        flip_mode = kwargs.get('augm_flip_mode', "horizontal")
        contr_factor = kwargs.get('augm_contrast_factor', 0.1)
        zoom_factor = kwargs.get('augm_zoom_factor', 0.2)
        bright_l_factor = kwargs.get('augm_brightness_lower_factor', -0.2)
        bright_u_factor = kwargs.get('augm_brightness_upper_factor', 0.2)
        trans_factor_h = kwargs.get('augm_translation_height_factor', 0.1)
        trans_factor_w = kwargs.get('augm_translation_width_factor', 0.1)
        cutout_w = kwargs.get('augm_cutout_width_factor', 0)
        cutout_h = kwargs.get('augm_cutout_height_factor', 0)

        augm_ops = []
        if cutout_h > 0 or cutout_w > 0:
            # pip install keras_cv
            import keras_cv
            augm_ops.append(keras_cv.layers.RandomCutout(cutout_h, cutout_w))
        if rot_factor > 0:
            augm_ops.append(tf.keras.layers.RandomRotation(factor=rot_factor))
        if trans_factor_h > 0 or trans_factor_w > 0:
            augm_ops.append(tf.keras.layers.RandomTranslation(height_factor=trans_factor_h,
                                                              width_factor=trans_factor_w))
        if flip_mode != 'none':
            augm_ops.append(tf.keras.layers.RandomFlip(mode=flip_mode))
        if bright_l_factor != 0 or bright_u_factor != 0:
            augm_ops.append(tf.keras.layers.RandomBrightness([bright_l_factor, bright_u_factor]))
        if contr_factor > 0:
            augm_ops.append(tf.keras.layers.RandomContrast(factor=contr_factor))
        if zoom_factor > 0:
            augm_ops.append(tf.keras.layers.RandomZoom(zoom_factor))
        rand_augment = Sequential(augm_ops, name="img_augmentation")

        def apply_rand_augment(images, labels):
            images = rand_augment(images)
            return images, labels

        def configure_for_performance(ds, train=True):
            if train:
                ds = ds.shuffle(buffer_size=5*batch_size)
            ds = ds.batch(batch_size)
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return ds

        def get_tfds(df, train=True):
            df['label_int'] = df['label'].apply(lambda label: inverse_labelmap[label])

            filenames_ds = tf.data.Dataset.from_tensor_slices(df['item'].values)
            images_ds = filenames_ds.map(parse_image)
            labels_ds = tf.data.Dataset.from_tensor_slices(df['label_int'].values)
            labels_ds = labels_ds.map(parse_labels)
            ds = tf.data.Dataset.zip((images_ds, labels_ds))
            ds = configure_for_performance(ds, train=train)

            if train and len(augm_ops):
                ds = ds.map(apply_rand_augment, num_parallel_calls=tf.data.AUTOTUNE)

            return ds

        if val_partition is not None:
            train_df = dataset.filter_by_partition(Partitions.TRAIN, inplace=False).as_dataframe()
            val_df = dataset.filter_by_partition(val_partition, inplace=False).as_dataframe()
        elif val_perc is not None:
            dataset_cpy = dataset.copy()
            dataset_cpy.filter_by_partition(Partitions.TRAIN, inplace=True)
            dataset_cpy.split(
                train_perc=1-val_perc, test_perc=0, val_perc=val_perc,
                groupby=val_groupby, stratify=val_stratify, random_state=val_rand_state)
            train_df = dataset_cpy.filter_by_partition(
                Partitions.TRAIN, inplace=False).as_dataframe()
            val_df = dataset_cpy.filter_by_partition(
                Partitions.VALIDATION, inplace=False).as_dataframe()
        else:
            if dataset.is_partitioned():
                train_df = dataset.filter_by_partition(
                    Partitions.TRAIN, inplace=False).as_dataframe()
            else:
                train_df = dataset.as_dataframe()
            val_df = None

        # Shuffle the train_df itself
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        train_batches = get_tfds(train_df, train=True)
        if val_df is not None:
            validation_batches = get_tfds(val_df, train=False)
        else:
            validation_batches = None

        return train_batches, validation_batches

    def _get_loss_fn(self, **kwargs):
        loss_fn = kwargs.get('loss_fn', 'categorical_crossentropy')

        if loss_fn == 'focal':
            alpha = kwargs.get('focal_alpha', 0.25)
            gamma = kwargs.get('focal_gamma', 2.0)
            return keras.losses.CategoricalFocalCrossentropy(alpha=alpha, gamma=gamma)
        else:
            return 'categorical_crossentropy'

    def _get_optimizer(self, **kwargs):
        lr = kwargs.get('learning_rate', 1e-2)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        return optimizer

    def _get_augmentations(self, **kwargs):
        augms = Sequential(
            [
            ],
            name="img_augmentation",
        )

        return augms

    def unfreeze_layers(self, finetune_layer):
        self._model.trainable = True
        set_trainable = False

        for layer in self._model.layers:
            if layer.name == finetune_layer:
                set_trainable = True
            if set_trainable:
                if not isinstance(layer, BatchNormalization):
                    layer.trainable = True
                else:
                    layer.trainable = False
            else:
                layer.trainable = False

        return self

    def _build_model(self, **kwargs):
        assert self.model_base_instance is not None, "model_base_instance not assigned"
        assert self.num_classes is not None, "For model training you must assign num_classes"

        from_base_model = kwargs.get('from_base_model', True)

        if from_base_model:
            img_augmentation = self._get_augmentations(**kwargs)
            inputs = Input(shape=(self.target_size[0], self.target_size[1], 3))
            x = img_augmentation(inputs)
            base_model = self.model_base_instance(
                include_top=False, input_tensor=x, weights="imagenet")

            # Freeze the pretrained weights
            base_model.trainable = False  # training

            # Rebuild top
            x = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
            x = BatchNormalization()(x)
            top_dropout_rate = 0.2
            x = Dropout(top_dropout_rate, name="top_dropout")(x)
            outputs = Dense(self.num_classes, activation="softmax", name="pred")(x)

            self._model = Model(inputs, outputs, name=self.name)
        else:
            assert self._model is not None, "No model has been created yet"

        optimizer = self._get_optimizer(**kwargs)
        loss_fn = self._get_loss_fn(**kwargs)
        metrics = self._get_metrics(**kwargs)
        self._model.compile(optimizer=optimizer,
                            loss=loss_fn,
                            metrics=metrics)

    def train(self,
              dataset,
              epochs=10,
              batch_size=32,
              **kwargs):
        train_batches, val_batches = self._get_train_val_data(dataset, batch_size, **kwargs)

        if not os.path.isfile(self.checkpoint):
            self._stage(1, epochs, train_batches, val_batches, **kwargs)
        else:
            logger.info(f"Loading model {self.name} from previuos checkpoint {self.checkpoint}")
            with tf.distribute.MirroredStrategy().scope():
                self._model = load_model(self.checkpoint)

        if 'finetune_layer' in kwargs:
            self._stage(2, epochs, train_batches, val_batches, **kwargs)

        return self._model

    def _stage(self, n_stage, epochs, train_batches, val_batches, **kwargs):
        epochs_last_stage = kwargs.get(f'epochs_stage{n_stage-1}', 0)
        _epochs = kwargs.get(f'epochs_stage{n_stage}', epochs)
        _epochs += epochs_last_stage
        if f'lr_stage{n_stage}' in kwargs:
            kwargs['learning_rate'] = kwargs.get(f'lr_stage{n_stage}')
        if f'tensorboard_dir_stage{n_stage}' in kwargs:
            kwargs['tensorboard_dir'] = kwargs.get(f'tensorboard_dir_stage{n_stage}')
        if kwargs.get('initial_epoch') is not None:
            initial_epoch = kwargs.get('initial_epoch')
        else:
            initial_epoch = epochs_last_stage

        callbacks = self._get_callbacks(**kwargs)

        if n_stage > 1:
            assert 'finetune_layer' in kwargs
            self.unfreeze_layers(kwargs['finetune_layer'])

        with tf.distribute.MirroredStrategy().scope():
            from_base_model = n_stage == 1
            self._build_model(**kwargs, from_base_model=from_base_model)

        logger.info(f"Training model {self.name} for {_epochs} epochs in stage {n_stage}")

        self._model.fit(train_batches,
                        epochs=_epochs,
                        initial_epoch=initial_epoch,
                        verbose=1,
                        validation_data=val_batches,
                        callbacks=callbacks)
