#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Module with the specification of the Dataset interface, which defines the common properties and
basic functionality of datasets of different modalities, allowing to extend it for particular
cases, such as datasets of image, video, audio, etc.

Datasets are structured data collections that contain information related to the elements used in
the different stages of ML pipelines (e.g., training, evaluation, and inference), which includes
both the metadata and annotations associated with them. However, different producers of such
datasets often use different names and data formats for fields with the same semantic meaning, so
if we intend to create ML pipelines that are agnostic to the datasets they use, it becomes
necessary to have a common interface to all of them, thus allowing us to apply the same operations
on them and interpret the information they contain in the same way, regardless of their source of
origin.
"""
from __future__ import annotations
from pathlib import Path
from functools import partial
import shutil
from typing import List, Union, Iterable, Callable, Any, Optional, Tuple, Literal, Final
from multiprocessing import Manager
import os
from shutil import copy2

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from .utils.dataset import (
    seek_files, fix_partitioning_by_priorities, append_to_partition, get_file_id_str_from_item,
    get_cleaned_label, fix_category_mapping, map_category, set_field_types_in_data, get_random_id,
    get_abspath_and_validate_item, get_cats_from_source, sample_data, get_media_name_with_prefix,
    get_default_fields, Fields)
from .utils.stratified_group_split import gradient_group_stratify
from .utils.misc import is_array_like, parallel_exec, download_file, get_chunk_func
from .utils.logger import get_logger

logger = get_logger(__name__)
pd.options.mode.chained_assignment = None


class Dataset():
    """Class that defines the common properties and implements basic functionality of the datasets

    Datasets can be created from very diverse data sources (e.g., files on a local device, public
    data collections, private databases, etc.), and their metadata and annotation information can
    be available in different formats (e.g. the names of the folders in which the files are stored,
    in CSV or JSON files, in databases, etc.). However, datasets from different sources have common
    properties and behaviors, so the Dataset API provides methods that allow to interact with them
    in a standard way, even allowing to join them to create datasets with data from different
    sources to be used in ML pipelines and data analysis.

"""

    # region FIELDS DEFINITIONS
    class MetadataFields():
        # The ITEM field serves to join both the metadata and the annotations
        ITEM: Final = Fields.ITEM
        FILE_ID: Final = Fields.FILE_ID
        FILENAME: Final = Fields.FILE_NAME
        DATE_CAPTURED: Final = Fields.DATE_CAPTURED
        LOCATION: Final = Fields.LOCATION

        TYPES = {
            FILE_ID: str,
            LOCATION: str
        }

    class AnnotationFields():
        ITEM: Final = Fields.ITEM
        LABEL: Final = Fields.LABEL
        ID: Final = Fields.ID
        PARTITION: Final = Fields.PARTITION
        SCORE: Final = Fields.SCORE

        TYPES = {
            ID: str,
            SCORE: float
        }
    # endregion

    # region CONSTANT PROPERTIES
    FILES_EXTS: Final = []
    DEFAULT_EXT: Final = ""
    # endregion

    # region SPECIAL METHODS
    def __repr__(self):
        return repr(self.df)

    def __len__(self):
        if self.is_empty:
            return 0
        return len(self.annotations)

    def __iter__(self):
        # TODO: improve
        for record in self.records:
            yield record

    def __getitem__(self, fields):
        if self.is_empty:
            if isinstance(fields, str):
                return pd.Series()
            else:
                return pd.DataFrame()
        return self.df[fields]

    def __setitem__(self, field, value):
        self.set_field_values(field, values=value, inplace=True)

    # TODO: Fix
    def __add__(self, other):
        return type(self).from_datasets(self, other)

    # endregion

    # region PROPERTIES

    @property
    def annotations(self) -> pd.DataFrame:
        return self._annotations

    @property
    def metadata(self) -> pd.DataFrame:
        return self._metadata

    @property
    def df(self) -> pd.DataFrame:
        return self.as_dataframe()

    @property
    def root_dir(self) -> str:
        try:
            return self._root_dir
        except Exception:
            return ''

    @property
    def classes(self) -> list:
        return self._get_categories()

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def fields(self) -> list:
        return list(set(self.metadata.columns) | set(self.annotations.columns))

    @property
    def labelmap(self) -> dict:
        return self._generate_labelmap_from_categories()

    @property
    def items(self) -> list:
        return self._get_items()

    @property
    def records(self):
        return self.df.to_dict('records')

    @property
    def partitions(self) -> list:
        if Fields.PARTITION in self.annotations.columns:
            return list(self.annotations[Fields.PARTITION].unique())
        return []

    @property
    def is_empty(self) -> bool:
        return (self.annotations is None and self.metadata is None) or len(self.annotations) == 0

    @property
    def _key_field_annotations(self) -> str:
        return Fields.ID

    @property
    def _key_field_metadata(self) -> str:
        return Fields.ITEM

    @property
    def metadata_default_fields(self) -> set:
        return set(get_default_fields(self.MetadataFields))

    @property
    def annotations_default_fields(self) -> set:
        return set(get_default_fields(self.AnnotationFields))

    @property
    def lbls(self) -> pd.Series:
        assert Fields.LABEL in self.fields
        return self[Fields.LABEL].value_counts()

    @property
    def item0(self) -> str:
        if not self.is_empty:
            return self.annotations.iloc[0][Fields.ITEM]
        else:
            return None

    @property
    def valid0(self) -> bool:
        return os.path.isfile(self.item0())

    # endregion

    # region CONSTRUCTOR

    def __init__(self: Dataset,
                 annotations: pd.DataFrame,
                 metadata: pd.DataFrame,
                 root_dir: str = None,
                 use_partitions: bool = True,
                 validate_filenames: bool = True,
                 not_exist_ok: bool = False,
                 avoid_initialization: bool = False,
                 **kwargs) -> Dataset:
        is_empty = annotations is None or len(annotations) == 0
        if avoid_initialization or is_empty:
            self._set_annotations(annotations)
            self._set_metadata(metadata)
            self._set_root_dir(root_dir)
            return

        self._fit_and_set_annotations(annotations, verify_integrity=True)
        self._fit_and_set_metadata(metadata, verify_integrity=True)

        self._verify_annotations_and_metadata_consistency()
        self._apply_filters_and_mappings(**kwargs)
        self.set_root_dir(
            root_dir, not_exist_ok=not_exist_ok, validate_filenames=validate_filenames)
        self._split(use_partitions)

    # endregion

    # region PUBLIC API METHODS

    #   region STATIC FACTORY METHODS

    @classmethod
    def from_dataframe(cls,
                       dataframe: pd.DataFrame,
                       root_dir: str,
                       accept_all_fields: bool = False,
                       mapping_fields: dict = None,
                       use_partitions: bool = True,
                       validate_filenames: bool = True,
                       not_exist_ok: bool = False,
                       avoid_initialization: bool = False,
                       **kwargs) -> Dataset:
        """Create a Dataset instance from the contents of `dataframe`.

        Parameters
        ----------
        dataframe
            Dataframe that contains the information with which the dataset instance will be
            created, both the metadata and the possible annotations of each element.
            The `dataframe` columns must match in name with those defined in
            `Dataset.MetadataFields` and `Dataset.AnnotationFields`, otherwise they will not be
            taken into account, unless the mapping is done using the `mapping_fields` parameter
        root_dir
            Path of the root directory in which the dataset elements are stored
        accept_all_fields, optional
            Whether or not to include in the dataset the columns of `dataframe` that are not
            contained in `Dataset.MetadataFields` or `Dataset.AnnotationFields`, by default False
        mapping_fields, optional
            _description_, by default None
        use_partitions, optional
            _description_, by default True
        validate_filenames, optional
            _description_, by default True
        not_exist_ok, optional
            _description_, by default False
        avoid_initialization, optional
            _description_, by default False

        Returns
        -------
            _description_
        """
        assert isinstance(dataframe, pd.DataFrame), "A pd.DataFrame must be provided for data"
        if len(dataframe) == 0:
            logger.debug(f"The dataset does not contain any elements.")
            return cls(annotations=None, metadata=None)

        dataframe = cls._fit_dataframe(dataframe, mapping_fields)
        annotations = cls._extract_annotations_from_dataframe(dataframe, accept_all_fields)
        metadata = cls._extract_metadata_from_dataframe(dataframe)

        instance = cls(annotations,
                       metadata,
                       root_dir,
                       use_partitions=use_partitions,
                       validate_filenames=validate_filenames,
                       not_exist_ok=not_exist_ok,
                       avoid_initialization=avoid_initialization,
                       **kwargs)
        return instance

    @classmethod
    def from_json(cls,
                  source_path: str,
                  root_dir: str,
                  accept_all_fields: bool = False,
                  mapping_fields: dict = None,
                  use_partitions: bool = True,
                  validate_filenames: bool = True,
                  not_exist_ok: bool = False,
                  include_bboxes_with_label_empty: bool = False,
                  set_filename_with_id_and_extension: str = None,
                  **kwargs) -> Dataset:
        """Create a Dataset from a json file in COCO format.

        Parameters
        ----------
        source_path : str, optional
            Path of a json file that will be converted into a Dataset.
            (default is None)
        **kwargs :
            Extra named arguments

        Returns
        -------
        Dataset
            Instance of the created Dataset
        """
        abspath = os.path.abspath(source_path)
        if source_path is not None:
            logger.info(f"Creating dataset from JSON file {abspath}")

        df = cls._get_dataframe_from_json(source_path,
                                          include_bboxes_with_label_empty,
                                          set_filename_with_id_and_extension)

        return cls.from_dataframe(df,
                                  root_dir,
                                  accept_all_fields=accept_all_fields,
                                  mapping_fields=mapping_fields,
                                  use_partitions=use_partitions,
                                  validate_filenames=validate_filenames,
                                  not_exist_ok=not_exist_ok,
                                  **kwargs)

    @classmethod
    def from_csv(cls,
                 source_path: str,
                 root_dir: str,
                 accept_all_fields: bool = False,
                 mapping_fields: dict = None,
                 use_partitions: bool = True,
                 validate_filenames: bool = True,
                 not_exist_ok: bool = False,
                 **kwargs) -> Dataset:
        """Create a Dataset from a csv file.

        Parameters
        ----------
        source_path : str
            Path of a csv file or a folder containing csv files that will be
            converted into a Dataset.
            The csv file(s) must have at least two columns, which represents
            `item` and `label` data.
        mapping_fields : dict, optional
            Dictionary to map field names in dataset.
            If header=True mapping must be done by the field names,
            otherwise must be done by column positions.
            E.g.::
                header=True: mapping_fields={"c1": "item", "c2": "label"}
                header=False: mapping_fields={0: "item", 1: "label"}
            (default is None)
        **kwargs :
            Extra named arguments

        Returns
        -------
        Dataset
            Instance of the created `Dataset`
        """
        abspath = os.path.abspath(source_path)
        if os.path.isdir(source_path):
            logger.info(f"Creating dataset from CSV files in folder {abspath}")
        else:
            assert os.path.isfile(abspath)
            logger.info(f"Creating dataset from CSV file {abspath}")

        df = cls._get_dataframe_from_csv(source_path=source_path)

        return cls.from_dataframe(df,
                                  root_dir,
                                  accept_all_fields=accept_all_fields,
                                  mapping_fields=mapping_fields,
                                  use_partitions=use_partitions,
                                  validate_filenames=validate_filenames,
                                  not_exist_ok=not_exist_ok,
                                  **kwargs)

    # TODO: allow to configure the way in which the labels are taken; e.g., last folder, first folder, join folder, etc.
    # TODO: add argument use_partitions
    @classmethod
    def from_folder(cls,
                    source_path: str,
                    extensions: List[str] = None,
                    use_labels: bool = True,
                    **kwargs) -> Dataset:
        """Create a `Dataset` from a folder structure

        Parameters
        ----------
        source_path : str
            Path of a folder to be loaded and converted into a Dataset object.
        extensions : list of str, optional
            List of extensions to seek files in folders.
            [""] to seek all files in folders (default is [""])
        use_labels : bool
            Whether or not you want to label each item of the dataset according
            to the name of the folder that contains it.
            (default is True)
        **kwargs :
            Extra named arguments that may contains the following parameters:
            * lower_case_exts: bool
                Whether to convert `extensions` to lower case or not, by default True

        Returns
        -------
        Dataset
            Instance of the created `Dataset`
        """
        extensions = extensions or cls.FILES_EXTS
        assert extensions, "You must specify extensions"
        assert os.path.isdir(source_path), f"{source_path} is not a valid folder name"
        folder_path = os.path.abspath(source_path)
        logger.info(f"Creating dataset from folder {folder_path}")

        df = cls._get_dataframe_from_folder(source_path, extensions, use_labels=use_labels)

        return cls.from_dataframe(df,
                                  root_dir=source_path,
                                  validate_filenames=False,
                                  **kwargs)

    @classmethod
    def from_datasets(cls,
                      *args: Dataset,
                      **kwargs) -> Dataset:
        """Create a Dataset from several dataset instances.

        Parameters
        ----------
        *args :
            Dataset instances to build a new dataset instance
        **kwargs :
            Arguments to be passed to the Dataset constructor for each Dataset in `*args`

        Returns
        -------
        Dataset
            Instance of the created Dataset
        """
        datasets = []
        for dataset in args:
            assert isinstance(dataset, Dataset), f"{type(dataset)} is not a Dataset instance."
            if not dataset.is_empty:
                datasets.append(cls.cast(dataset))

        if not datasets:
            return cls(annotations=None, metadata=None)

        annotations, metadata, root_dir = cls._get_concat_data_from_datasets(datasets)

        return cls(annotations, metadata, root_dir, validate_filenames=False, **kwargs)

    #   endregion

    #   region STORAGE METHODS

    def to_folder(self: Dataset,
                  dest_path: str,
                  *,
                  use_labels: bool = True,
                  use_partitions: bool = False,
                  move_files: bool = False,
                  preserve_directory_hierarchy: bool = False,
                  prefix_field: str = None,
                  prefix_separator: str = '_',
                  update_dataset_filepaths: bool = False) -> str:
        """Create a filesystem representation of the dataset.

        Parameters
        ----------
        dest_path : str
            Path where the dataset files will be stored
        use_partitions : bool, optional
            Whether or not to split items in folders with partition names.
            Only works if the dataset has been previously split.
            (default is True)
        use_labels : bool, optional
            Whether or not to split items in folders with label names.
            These folders will be below the partition folders (default is True)
        move_files : bool, optional
            Whether to move (`True`) or copy (`False`) the files during the operation,
            default is False
        preserve_directory_hierarchy : bool
            Whether or not to preserve the original structure of the directories.
            If False, all the media will be stored directly under the directory `dest_path`.
            By default False
        prefix_field : str
            Field to be used to add as a prefix to the filename. By default None
        prefix_separator : str
            String to be used to separate the prefix from the filename. By default '_'

        Returns
        -------
        str
            Path of folder created
        """
        folder = os.path.abspath(dest_path)
        if self.is_empty:
            logger.info(f"No data to write in folder {folder}")
            return dest_path

        logger.info(f"Writing the dataset elements in folder {folder}")

        use_partitions = use_partitions and Fields.PARTITION in self.fields
        update_dataset_filepaths = update_dataset_filepaths or move_files

        def _get_dest_path(record):
            dest_folder = dest_path
            if use_partitions:
                dest_folder = os.path.join(dest_folder, record[Fields.PARTITION])
            if use_labels:
                dest_folder = os.path.join(dest_folder, record[Fields.LABEL])
            return dest_folder

        fields = [Fields.ITEM]
        items_mapper = None
        if update_dataset_filepaths:
            items_mapper = Manager().dict()
            fields.append(Fields.ID)
        if prefix_field is not None:
            fields.append(prefix_field)
        if use_labels:
            fields.append(Fields.LABEL)
        if use_partitions and Fields.PARTITION in self.fields:
            fields.append(Fields.PARTITION)

        records = self[fields].drop_duplicates(Fields.ITEM).to_dict('records')

        parallel_exec(
            func=self._filesystem_writer,
            elements=records,
            record=lambda record: record,
            dest_path=_get_dest_path,
            move_files=move_files,
            preserve_directory_hierarchy=preserve_directory_hierarchy,
            prefix_field=prefix_field,
            prefix_separator=prefix_separator,
            items_mapper=items_mapper)

        if update_dataset_filepaths:
            self[Fields.ITEM] = lambda record: items_mapper[record[Fields.ITEM]]
            self._set_root_dir(dest_path)

        return dest_path

    def to_csv(self: Dataset,
               dest_path: Union[str, Path],
               columns: List[str] = None) -> str:
        """Create csv file(s) representing the dataset.

        Parameters
        ----------
        dest_path : str or Path
            Path where the csv file(s) will be stored
        columns : list of str, optional
            List of columns to add in the csv (default is None)

        Returns
        -------
        str or list of str
            Path or list of paths of csv file(s) created
        """
        if self.is_empty:
            logger.info(f"No data to write in the CSV file {dest_path}")
            return dest_path

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        df = self.df
        if columns is not None:
            df = self[columns]
        df[Fields.ITEM] = df[Fields.ITEM].apply(self._format_item_for_storage)

        df.to_csv(dest_path, index=False, header=True)

        logger.info(f"CSV file {os.path.abspath(dest_path)} created")
        return dest_path

    #   endregion

    #   region ACCESSORS

    def as_dataframe(self: Dataset, columns: List[str] = None) -> pd.DataFrame:
        """Gets a DataFrame representation of the Dataset.

        Parameters
        ----------
        columns : list of str, optional
            List of columns in the DataFrame (default is None)

        Returns
        -------
        DataFrame
            DataFrame representing the Dataset.
        """
        if self.is_empty:
            return pd.DataFrame()

        df = self._merge_annotations_and_metadata()

        if not columns is None:
            columns = [c for c in columns if c in df.columns]
        if not columns:
            columns = self._get_ordered_fields_in_dataset()

        df = df[columns]

        return df

    def get_chunk(self, num_chunks: int, chunk_num: int) -> Dataset:
        """Function that divides the elements of a dataset into `num_chunks` chunks and
        returns a dataset with the elements of chunk number `chunk_num`

        Parameters
        ----------
        num_chunks : int
            Number of chunks into which the dataset is divided
        chunk_num : int
            Number of the chunk that will be taken to form the generated dataset.
            It must be in the range `[1, num_chunks]`

        Returns
        -------
        Dataset
            Data set with the elements of chunk number `chunk_num` of the `num_chunks` chunks into
            which the original dataset was divided
        """
        if self.is_empty:
            return self

        items = get_chunk_func(self.items, num_chunks, chunk_num, sort_elements=True)
        new_ds = self.filter_by_field(Fields.ITEM, items, inplace=False)
        return new_ds

    #   endregion

    #   region MUTATORS

    #     region VALUES OF FIELDS

    def set_field_values(self,
                         field: str,
                         values: Union[Callable, pd.DataFrame, object],
                         inplace: bool = True) -> Dataset:
        """Method that assigns the information of the `field` from the information of the
        annotations fields by applying the expression `expr`

        Parameters
        ----------
        field : str
            Name of the field in `data`
        values : Callable
            Expression to be applied in order to change the value of `field` in `data`
        inplace : bool, optional
            If True, perform operation in-place, by default True

        Returns
        -------
        Dataset
            Instance of the Dataset
        """
        if self.is_empty:
            return self

        instance = self if inplace else self.copy()

        if isinstance(values, pd.DataFrame):
            meta_flds = instance.metadata_default_fields
            anns_flds = instance.annotations_default_fields
            key_meta = instance._key_field_metadata
            key_anns = instance._key_field_annotations
            assert not field in (key_anns, key_meta), f"Invalid field to set: {field}"
            assert key_meta in values.columns or key_anns in values.columns, "Error in values fields"

            fields = [field] if not is_array_like(field) else field
            for _field in fields:
                if _field in meta_flds and key_meta in values.columns:
                    instance._set_metadata_field_using_dataframe(_field, dataframe=values)
                if (_field in anns_flds or not _field in meta_flds) and key_anns in values.columns:
                    instance._set_annotations_field_using_dataframe(_field, dataframe=values)
        else:
            df = instance.df
            if callable(values):
                df[field] = df.apply(values, axis=1)
            else:
                df[field] = values
            instance._update_modified_dataset(df, field_modified=field)

        # TODO: In case field == 'item', validate root_dir

        return instance

    def map_categories(self,
                       category_mapping: dict,
                       inplace: bool = True) -> Dataset:
        """Function that performs a mapping of the categories that a dataset contains, changing the
        value of the field `label` of the rows of each category by its corresponding target value

        Parameters
        ----------
        category_mapping : Union[dict, str]
            Dictionary containing the mappings
        inplace : bool, optional
            If True, perform operation in-place, by default True

        Returns
        -------
        Dataset
            Instance of the Dataset
        """
        if self.is_empty:
            return self
        assert_msg = "The dataset doesn't contain the 'label' field"
        assert Fields.LABEL in self.annotations.columns, assert_msg

        instance = self if inplace else self.copy()
        category_mapping = fix_category_mapping(category_mapping)
        anns = instance.annotations
        for from_cat, to_cat in category_mapping.items():
            if from_cat == '*' and to_cat != '*':
                remaining_cats = set(category_mapping.values()) - {to_cat}
                anns.loc[~anns[Fields.LABEL].isin(remaining_cats), Fields.LABEL] = to_cat
            else:
                anns.loc[anns[Fields.LABEL] == from_cat, Fields.LABEL] = to_cat

        instance._fit_and_set_annotations(
            anns, verify_integrity=False, field_modified=Fields.LABEL)

        return instance

    def split(self: Dataset,
              train_perc: float = 0.8,
              test_perc: float = 0.2,
              val_perc: float = 0.,
              group_by_field: str = None,
              stratify: bool = True,
              random_state: int = 42) -> Dataset:
        """Split the dataset between `train`, `test` and `validation`
        partitions. `train_perc`, `test_perc` and `val_perc` must sum to 1.

        Parameters
        ----------
        train_perc : float, optional
            Train set percentage. Should be between 0 and 1 (default is 0.8)
        test_perc : float, optional
            Test set percentage. Should be between 0 and 1 (default is 0.2)
        val_perc : float, optional
            Validation set percentage. Should be between 0 and 1 (default is 0.)
        group_by_field : str, optional
            Field of the media by which to group the elements of the dataset when partitioning,
            and ensures that there are no elements of the same group in more than one partition
            (default is None)
        stratify : bool, optional
            Whether or not to split in a stratified fashion, using the class labels
            (default is True)
        random_state : int, optional
            Seed for random number generator, by default None

        Returns
        -------
        Dataset
            Instance of the Dataset
        """
        if self.is_empty:
            return self
        assert_cond = np.isclose(train_perc + test_perc + val_perc, 1.)
        assert assert_cond, 'The proportions of the partitions must add up to 1'
        logger.debug(f"Partitioning the dataset with proportions (train/test/val): "
                     f"{train_perc}/{test_perc}/{val_perc}")
        if stratify:
            logger.debug("Partitioning will be performed in stratified fashion")
        if group_by_field is not None:
            logger.debug(
                f"During partitioning, grouping by the field {group_by_field} "
                f"will be done to avoid elements of the same groups being in different partitions")

        parts_to_idxs = {}

        parts_perc = {
            Partitions.TRAIN: train_perc,
            Partitions.TEST: test_perc,
            Partitions.VALIDATION: val_perc
        }
        reverse = group_by_field is None
        ordered_partitions = dict(sorted(parts_perc.items(), key=lambda k: k[1], reverse=reverse))

        df = self.df
        label_counts = df[Fields.LABEL].value_counts().rename_axis(Fields.LABEL).to_frame('count')
        one_sample_labels = label_counts[label_counts['count'] == 1].index.values
        one_sample_ordinals = df[df[Fields.LABEL].isin(one_sample_labels)].index.values
        ordinals = df[~df[Fields.LABEL].isin(one_sample_labels)].index.values

        if group_by_field is not None and group_by_field in df:
            total = 0.
            for part, perc in ordered_partitions.items():
                if np.isclose(perc, 0.):
                    continue
                if np.isclose(total + perc, 1.):  # Train partition (the biggest one last)
                    ordinals = np.concatenate([ordinals, one_sample_ordinals])
                    append_to_partition(parts_to_idxs, part, ordinals)
                else:
                    part_size = perc / (1. - total)

                    if stratify:
                        rest, ordinals_part = gradient_group_stratify(
                            df.loc[ordinals],
                            test_size=part_size,
                            group_field=group_by_field,
                            random_state=random_state)

                        append_to_partition(parts_to_idxs, part, ordinals_part)
                        ordinals = rest
                    else:
                        grp_kfold = GroupShuffleSplit(
                            n_splits=1,
                            test_size=part_size,
                            random_state=random_state)

                        groups = df.loc[ordinals][group_by_field].values
                        for rest_idx, ords_idx_part in grp_kfold.split(X=ordinals, groups=groups):
                            append_to_partition(parts_to_idxs, part, ordinals[ords_idx_part])
                            ordinals = ordinals[rest_idx]
                    total += perc
        else:
            total = 0.
            for part, perc in ordered_partitions.items():
                if np.isclose(perc, 0.):
                    break
                if np.isclose(total + perc, 1.):
                    append_to_partition(parts_to_idxs, part, ordinals)
                else:
                    strat_labels = df.loc[ordinals].label.values if stratify else None
                    part_size = perc / (1. - total)
                    rest, ordinals_part = train_test_split(
                        ordinals, test_size=part_size, random_state=random_state,
                        stratify=strat_labels)
                    if np.isclose(total, 0.):  # Training partition (the biggest one first)
                        ordinals_part = np.concatenate([ordinals_part, one_sample_ordinals])
                    append_to_partition(parts_to_idxs, part, ordinals_part)
                    ordinals = rest
                    total += perc

        self._add_partitions_to_dataset(parts_to_idxs)
        self._fix_partitioning_by_priorities(parts_to_idxs)

        return self

    #     endregion

    #     region FILTERING METHODS

    def filter_by_categories(self,
                             categories: Union[List[str], str],
                             mode: Literal['include', 'exclude'] = 'include',
                             inplace: bool = True) -> Optional[Dataset]:
        """Method that filters the dataset by `label`

        Parameters
        ----------
        categories : list of str or str
            List of categories to filter media, or path of a CSV file or a text file.
            If empty list or None, no filtering will be performed.
            If it is a CSV file, it should have the categories in the first column and should not
            have a header. If it is a text file, it must have the categories separated by a line
            break
        mode : str, optional
            Whether to 'include' or 'remove' registers with `label` in `categories` in the dataset,
            by default 'include'
        inplace : bool, optional
            If True, perform operation in-place, by default True

        Returns
        -------
        Dataset
            Instance of the Dataset
        """
        cats = get_cats_from_source(categories)
        return self.filter_by_field(Fields.LABEL, cats, mode=mode, inplace=inplace)


    def filter_by_field(self,
                        field: str,
                        values: Union[Iterable, Any],
                        mode: Literal['include', 'exclude'] = 'include',
                        inplace: bool = True) -> Dataset:
        """Method that filters the dataset by field `field`

        Parameters
        ----------
        field : str
            Name of the field to filter by
        values : list or str
            Values to filter by
        mode : str, optional
            Whether to 'include' or 'remove' registers with `values` in `field` in the dataset,
            by default 'include'
        inplace : bool, optional
            If True, perform operation in-place, by default True

        Returns
        -------
        Dataset
            Instance of the Dataset
        """
        if self.is_empty:
            return self
        assert mode in ('include', 'exclude'), f"Invalid value of mode: {mode}"
        instance = self if inplace else self.copy()

        if not is_array_like(values):
            values = [values]

        df = instance.df

        if mode == 'include':
            df = df[df[field].isin(values)]
        else:
            df = df[~df[field].isin(values)]

        instance._update_modified_dataset(df)

        logger.debug(f"The elements of the dataset were filtered by the field '{field}' "
                     f"and {len(instance)} elements were obtained.")

        return instance

    def filter_by_partition(self,
                            partition: str,
                            mode: Literal['include', 'exclude'] = 'include',
                            inplace: bool = True) -> Optional[Dataset]:
        """Method that filters the dataset by field 'partition'

        Parameters
        ----------
        partition : str
            The value of the partition by which the dataset is going to be filtered
        mode : str, optional
            Whether to `include` or `exclude` the elements with the partition field equals to
            `partition`, by default 'include'
        inplace : bool, optional
            If True, perform operation in-place, by default True

        Returns
        -------
        Dataset
            Instance of the filtered Dataset
        """
        if partition is not None:
            Partitions.check_partition(partition)
            return self.filter_by_field(Fields.PARTITION, partition, mode=mode, inplace=inplace)
        return self

    def filter_by_score(self,
                        max_score: float = None,
                        min_score: float = None,
                        score: float = None,
                        column_name: str = Fields.SCORE,
                        inplace: bool = True) -> Dataset:
        """Method that filters the predictions by the `score` field

        Parameters
        ----------
        max_score : float, optional
            Float number in [0.,1.] that indicates the maximum value of the `score` field,
            by default None
        min_score : float, optional
            Float number in [0.,1.] that indicates the minimum value of the `score` field,
            by default None
        score : float, optional
            Float number in [0.,1.] that indicates the exact value of the `score` field,
            by default None
        column_name : str, optional
            Column name of the score field in `self._anns_data`, by default "score"
        inplace : bool, optional
            If True, perform operation in-place, by default True

        Returns
        -------
        Dataset
            Instance of the resulting dataset
        """
        if self.is_empty:
            return self

        instance = self if inplace else self.copy()
        df = instance.df

        if max_score is not None:
            df = df.loc[df[column_name] <= float(max_score)]
        if min_score is not None:
            df = df.loc[df[column_name] >= float(min_score)]
        if score is not None:
            df = df.loc[df[column_name] == float(score)]

        instance._update_modified_dataset(df)

        return instance

    def filter(self,
               expr: Callable,
               mode: Literal['include', 'exclude'] = 'include',
               inplace: bool = True) -> Dataset:
        """Method that filters the dataset by applying `expr` to every element

        Parameters
        ----------
        expr : function
            Function to apply to each record of the dataset.
        mode : str, optional
            Whether to 'include' or 'remove' registers selected in `expr`, by default 'include'
        inplace : bool, optional
            If True, perform operation in-place, by default True

        Returns
        -------
        Dataset
            Instance of the filtered Dataset
        """
        if self.is_empty:
            return self

        instance = self if inplace else self.copy()
        df = instance.df

        if mode == 'include':
            df = df[df.apply(expr, axis=1)]
        else:
            df = df[~df.apply(expr, axis=1)]

        instance._update_modified_dataset(df)

        return instance

    #     endregion

    #     region SAMPLING METHODS

    def sample(self,
               n: str | int | float | dict,
               use_labels: bool = False,
               use_partitions: bool = False,
               random_state: int = None,
               inplace: bool = True) -> Dataset:
        """Method that samples the elements of a Dataset by grouping them by their `label` and
        taking a random number determined by the value of `n`.

        Parameters
        ----------
        n : str, int, float or dict
            If int, indicates the maximum number of samples to be taken from among the elements
            of each label.
            If float (0, 1), it refers to the percentage of elements to be taken from each
            label.
            If str, indicates the name of the method to be used in the sampling operation.
            The possible method names are:
            * fewer: will take as the sample number the smallest value of the elements grouped
            by label in the set.
            * mean: calculates the average of the element counts for each category and takes
            this value as the maximum number of elements for each category.
            If dict, it must contain an integer for each category that you want to configure,
            in the form {'label_name': max_elements} (default is None)
        use_labels : bool, optional
            Whether to permorm the sampling taking the elements of each label or not,
            by default False
        use_partitions : bool, optional
            Whether to permorm the sampling taking the elements of each partition or not,
            by default False
        random_state : int, optional
            Seed for random number generator, by default None
        inplace : bool, optional
            If True, perform operation in-place, by default True

        Returns
        -------
        Dataset
            Instance of the sampled Dataset
        """
        if self.is_empty:
            return self
        instance = self if inplace else self.copy()
        groupby = Fields.LABEL if use_labels else None

        if use_partitions and len(instance.partitions) > 0:
            partitions_dfs = []
            for partition in instance.partitions:
                df_part = instance.filter_by_partition(partition, inplace=False).df
                elems = sample_data(
                    data=df_part,
                    n=n,
                    random_state=random_state,
                    groupby=groupby)
                partitions_dfs.append(elems)
            df = pd.concat(partitions_dfs).reset_index(drop=True, inplace=False)
        else:
            df = sample_data(
                data=instance.df,
                n=n,
                random_state=random_state,
                groupby=groupby)

        instance._update_modified_dataset(df)

        return instance

    def take(self, n: int, random_state: int = None) -> Dataset:
        return self.sample(
            n=n, use_labels=False, use_partitions=False, random_state=random_state, inplace=False)

    #     endregion

    #   endregion

    #   region DUPLICATION METHODS

    def copy(self) -> Dataset:
        """Create a copy of the dataset

        Returns
        -------
        Dataset
            Copy of the original dataset
        """
        return self._copy_dataset(self)

    @classmethod
    def cast(cls, dataset: Dataset) -> Dataset:
        instance = cls(dataset.annotations,
                       dataset.metadata,
                       dataset.root_dir,
                       avoid_initialization=True)
        return instance

    #   endregion

    # endregion

    # region PRIVATE API METHODS

    #   region CALLBACKS

    @classmethod
    def _get_dataframe_from_json(cls, source_path, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def _get_dataframe_from_csv(cls,
                                source_path: str,
                                na_values='nan',
                                header=0,
                                usecols=None,
                                keep_default_na: bool = False,
                                regex_filename=None) -> pd.DataFrame:

        na_values = na_values.split(',')

        read_csv = partial(
            pd.read_csv,
            usecols=usecols,
            header=header,
            na_values=na_values,
            keep_default_na=keep_default_na)

        if os.path.isdir(source_path):
            csvs = seek_files(source_path, seek_name=regex_filename, seek_extension=[".csv"])
            assert len(csvs) > 0, f"Folder {source_path} does not contain valid csv files."

            data = pd.DataFrame()
            for csv_file in csvs:
                csv_path = os.path.join(csv_file["path"], csv_file["title"])
                logger.debug(f"Reading data from CSV file {csv_path}")
                temp = read_csv(filepath_or_buffer=csv_path)
                data = pd.concat([data, temp], ignore_index=True)
        else:
            data = read_csv(filepath_or_buffer=source_path)

        return data

    @classmethod
    def _get_dataframe_from_folder(cls,
                                   source_path: str,
                                   extensions: List,
                                   use_labels: bool = True,
                                   lower_case_exts: bool = True) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        source_path : str
            _description_
        extensions : List, optional
            _description_, by default None
        use_labels : bool, optional
            _description_, by default True
        lower_case_exts : bool, optional
            _description_, by default True

        Returns
        -------
        pd.DataFrame
            _description_
        """
        extensions = list(set([x.lower() if lower_case_exts else x for x in extensions]))
        _files = seek_files(source_path, seek_extension=extensions)

        exts_str = ', '.join(extensions)
        logger.debug(f"{len(_files)} files found for extensions {exts_str} in {source_path}")

        if use_labels:
            n_valid_labels_names = 0
            for _, label_dirs, _ in os.walk(source_path):
                n_valid_labels_names += len(label_dirs)
            if n_valid_labels_names == 0:
                use_labels = False

        res = []
        for f in _files:
            path = str(Path(f["path"]).relative_to(source_path))
            temp_item = os.path.join(path, f["title"])
            temp = {
                Fields.ITEM: temp_item
            }
            if use_labels:
                temp[Fields.LABEL] = path
            temp[Fields.ID] = get_random_id()
            res.append(temp)

        data = pd.DataFrame(res)

        return data

    @classmethod
    def _get_concat_data_from_datasets(cls,
                                       datasets: List[Dataset]
                                       ) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """Method that is executed in the calls to the `from_datasets` constructor, and that
        assigns the content of the `metadata` field with the information contained
        in the dataset instances.

        Parameters
        ----------
        datasets : List[Dataset]
            _description_

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, str]
            _description_
        """
        dataframe = pd.DataFrame()
        root_dirs = set()
        for ds in datasets:
            if ds.is_empty:
                continue
            dataframe = pd.concat([dataframe, ds.df], ignore_index=True)
            root_dirs |= {ds.root_dir}

        annotations = cls._extract_annotations_from_dataframe(dataframe, accept_all_fields=True)
        metadata = cls._extract_metadata_from_dataframe(dataframe)

        root_dirs = list(root_dirs)
        if len(root_dirs) == 1 and root_dirs[0] != '':
            # The root dirs are all the same, so we can pick it for the new dataset
            root_dir = root_dirs[0]
        else:
            # The root dirs are different, so we can't pick any for the new dataset
            root_dir = None

        return annotations, metadata, root_dir

    #   endregion

    #   region FILESYSTEM METHODS
    def _filesystem_writer(self,
                           record: dict,
                           dest_path: str,
                           move_files: bool,
                           preserve_directory_hierarchy:  bool,
                           prefix_field: str,
                           prefix_separator: str,
                           items_mapper: dict = None):
        """Auxiliar function to move items in a `to_folder` call

         Parameters
        ----------
        record : dict
            Dictionary containing information of a record.
        dest_path : str
            Path where media will be copied
        move_files: bool
            Whether to move (`True`) or copy (`False`) the files during the operation
        preserve_directory_hierarchy : bool
            Whether or not to preserve the original structure of the directories.
            If False, all the media will be stored directly under the directory `dest_path`
        prefix_field : str
            Field to be used to add as a prefix to the filename
        prefix_separator : str
            String to be used to separate the prefix from the filename
        """

        root_dir = self.root_dir
        item = record[Fields.ITEM]
        if preserve_directory_hierarchy and root_dir is not None:
            media_name = os.path.relpath(item, root_dir)
        else:
            media_name = os.path.basename(item)

        if prefix_field is not None:
            assert_msg = f"The field {prefix_field} is not present"
            assert prefix_field in record, assert_msg

            media_name = get_media_name_with_prefix(
                record, prefix_field, prefix_separator, media_name)

        os.makedirs(dest_path, exist_ok=True)

        dest_media = os.path.join(dest_path, media_name)

        try:
            os.makedirs(os.path.dirname(dest_media), exist_ok=True)
            if move_files:
                shutil.move(item, dest_media)
            else:
                copy2(item, dest_media)
        except Exception:
            logger.exception(
                f"An error occurred while copying the media {item} to {dest_media}")

        if items_mapper is not None:
            items_mapper[record[Fields.ITEM]] = dest_media
    #   endregion

    #   region SET_* METHODS

    def _fit_and_set_metadata(self,
                              metadata: pd.DataFrame,
                              verify_integrity: bool = True,
                              field_modified: Optional[str] = None):
        """Assigns the information of the media to the property `_metadata`

        Parameters
        ----------
        metadata : pd.DataFrame
            Dataframe that contains the metadata information of the items in the dataset
        """
        metadata = self._fit_metadata(metadata, verify_integrity, field_modified)
        self._set_metadata(metadata)

    def _fit_and_set_annotations(self,
                                 annotations: pd.DataFrame,
                                 verify_integrity: bool = True,
                                 field_modified: Optional[str] = None):
        """Assigns the information of the annotations to the property `_annotations`

        Parameters
        ----------
        annotations : pd.DataFrame
            Dataframe that contains the annotations information of the items in the dataset
        """
        annotations = self._fit_annotations(annotations, verify_integrity, field_modified)
        self._set_annotations(annotations)

    def _set_metadata(self, metadata: pd.DataFrame):
        self._metadata = metadata

    def _set_annotations(self, annotations: pd.DataFrame):
        self._annotations = annotations

    def _set_root_dir(self, root_dir):
        self._root_dir = root_dir

    def set_root_dir(self,
                     root_dir: Path,
                     not_exist_ok: bool = False,
                     validate_filenames: bool = True):
        """Set the absolute paths and validate items

        Parameters
        ----------
        root_dir : str
            Path of the base folder where the media are saved
        not_exist_ok : bool
            Whether to include media that exist and silently pass exceptions for those media
            that are not present (default is False)
        validate_filenames : bool
            Whether or not to validate that the items exist.
            Set to False to speed up execution time. By default True
        """
        assert root_dir is not None or validate_filenames is False, "root_dir must be assigned"

        if root_dir is not None or validate_filenames != False:
            items_to_abspaths = Manager().dict()
            invalid_items = Manager().list()
            parallel_exec(
                func=get_abspath_and_validate_item,
                elements=self.items,
                item=lambda item: item,
                old_root_dir=self.root_dir,
                new_root_dir=root_dir,
                validate_filenames=validate_filenames,
                not_exist_ok=not_exist_ok,
                items_to_abspaths=items_to_abspaths,
                invalid_items=invalid_items)
            self[Fields.ITEM] = lambda record: items_to_abspaths[record[Fields.ITEM]]

            n_invalid = len(invalid_items)
            if n_invalid > 0:
                logger.info(f'{n_invalid} invalid items found that were ignored')
                self.filter_by_field(
                    Fields.ITEM, values=list(invalid_items), mode='exclude', inplace=True)

        self._set_root_dir(root_dir)

        n_items = len(self.metadata)
        logger.debug(f'{n_items} {"" if validate_filenames else "not"} validated items found')

    @classmethod
    def _add_file_id_field_to_dataframe(cls, dataframe: pd.DataFrame):
        """Assigns the field `Fields.FILE_ID` in `dataframe` from the information contained
        in the field `Fields.ITEM`

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe with media file information from the dataset
        """
        stems = dataframe[Fields.ITEM].apply(lambda x: Path(x).stem)
        if stems.nunique() == dataframe[Fields.ITEM].nunique():
            file_ids = stems
        else:
            file_ids = dataframe[Fields.ITEM].apply(get_file_id_str_from_item)

        dataframe[cls.MetadataFields.FILE_ID] = file_ids

        return dataframe

    def _set_metadata_field_using_dataframe(self, field: str, dataframe: pd.DataFrame):
        dataframe = (dataframe
                     .drop_duplicates(self._key_field_metadata, keep='first')
                     .set_index(self._key_field_metadata))
        metadata = self.metadata.set_index(self._key_field_metadata)
        metadata[field] = dataframe[field]
        metadata = metadata.reset_index(drop=False)
        self._fit_and_set_metadata(metadata, verify_integrity=False, field_modified=field)

    def _set_annotations_field_using_dataframe(self, field: str, dataframe: pd.DataFrame):
        dataframe = dataframe.set_index(self._key_field_annotations)
        annotations = self.annotations.set_index(self._key_field_annotations)
        annotations[field] = dataframe[field]
        annotations = annotations.reset_index(drop=False)
        self._fit_and_set_annotations(annotations, verify_integrity=False, field_modified=field)

    #   endregion

    #   region GET_* METHODS

    #       region COLUMNS INFORMATION
    def _get_categories(self, sort_by_name: bool = True) -> list:
        """Get the list of distinct values in the 'label' field of the dataset.
        This method could be used to construct the labelmap of the dataset in case it did not be
        provided in the constructor.
        The list of categories could be ordered and/or lowered.

        Parameters
        ----------
        sort_by_name : bool, optional
            Whether to sort categories by name or not. (default True)

        Returns
        -------
        list of str
            List of sorted categories of the dataset
        """

        if Fields.LABEL in self.annotations.columns:
            cats = self.annotations[Fields.LABEL].unique().tolist()
            if sort_by_name:
                cats = sorted(cats)
        else:
            cats = []

        return cats

    def _get_items(self) -> list:
        """Get a list of unique items in the dataset, optionally filtered by partition name.

        Returns
        -------
        list or set
            List of unique items in the dataset
        """
        if len(self) > 0:
            items = self[Fields.ITEM].unique()
        else:
            items = []

        return list(items)

    def _get_default_fields_of_dataset(self) -> List:
        default_fields_anns = get_default_fields(self.AnnotationFields)
        default_fields_meta = get_default_fields(self.MetadataFields)
        default_fields = default_fields_anns + default_fields_meta
        # Drop duplicated fields
        return list(dict.fromkeys(default_fields))

    def _get_ordered_fields_in_dataset(self) -> List:
        default_fields = self._get_default_fields_of_dataset()
        current_dataset_fields = set(self.fields)
        extra_fields = list(current_dataset_fields - set(default_fields))
        return [c for c in default_fields + extra_fields if c in current_dataset_fields]

    @classmethod
    def _get_common_field_anns_meta(cls) -> str:
        metadata_default_fields = set(get_default_fields(cls.MetadataFields))
        annotations_default_fields = set(get_default_fields(cls.AnnotationFields))
        common_field = list(metadata_default_fields & annotations_default_fields)

        assert len(common_field) == 1

        return common_field[0]

    def _merge_annotations_and_metadata(self, how='left') -> pd.DataFrame:
        annotations = self.annotations
        metadata = self.metadata
        common_field = self._get_common_field_anns_meta()

        return annotations.merge(metadata, how=how, on=common_field)

    #       endregion

    #   endregion

    #   region CREATION AND CONFIGURATION OF INTERNAL DATA

    #     region VERIFICATION

    def _verify_annotations_and_metadata_consistency(self):
        common_field = self._get_common_field_anns_meta()
        elems_meta = self.metadata[common_field].values
        elems_anns = self.annotations[common_field].values

        # TODO: Check if this can be omitted
        if set(elems_anns) != set(elems_meta):
            metadata = self.metadata
            metadata = metadata[metadata[common_field].isin(elems_anns)].reset_index(drop=True)
            self._set_metadata(metadata)
            elems_meta = self.metadata[common_field].values
        assert set(elems_anns) == set(elems_meta) and len(self.metadata) > 0

    def _verify_integrity_annotations(self, annotations: pd.DataFrame):
        assert isinstance(annotations, pd.DataFrame), f"Annotations data must be a pd.DataFrame"

        assert_msg = f"annotations data must contain the column {Fields.ITEM}"
        assert Fields.ITEM in annotations.columns, assert_msg

        if self._key_field_annotations in annotations.columns:
            assert_cond = annotations[self._key_field_annotations].nunique() == len(annotations)
            assert assert_cond, f"The field {self._key_field_annotations} must have unique values"
        else:
            annotations[self._key_field_annotations] = [get_random_id()
                                                        for _ in range(len(annotations))]

    def _verify_integrity_metadata(self, metadata: pd.DataFrame):
        assert isinstance(metadata, pd.DataFrame), "Media data must be a pd.DataFrame"

        cnd = self.MetadataFields.FILE_ID in metadata.columns and Fields.ITEM in metadata.columns
        msg = f"Metadata must contain the fields {self.MetadataFields.FILE_ID} and {Fields.ITEM}"
        assert cnd, msg

    #     endregion

    #     region AUXILIAR METHODS
    @classmethod
    def _download(cls,
                  dest_path: str,
                  metadata: pd.DataFrame,
                  set_filename_with_id_and_ext: str,
                  media_base_url: str = None):
        """Function that downloads media from the collection

        Parameters
        ----------
        dest_path : str
            Folder in which media will be downloaded
        metadata : pd.DataFrame
            DataFrame containing information about the media in the collection.
            Each record must contain at least the fields {`id`, `file_name`}.
        set_filename_with_id_and_ext : str, optional
            Extension to be added to the id of each item to set the file name (default is None)
        **kwargs :
            Extra named arguments that may contains the following parameters:
            * media_base_url : str
                URL where the media are located
        """
        cond = media_base_url is not None
        assert cond, f"In order to download media you must assign media_base_url parameter"
        os.makedirs(dest_path, exist_ok=True)

        def get_dest_filename(record) -> str:
            fname = cls._get_filename(record, set_filename_with_id_and_ext)
            return os.path.join(dest_path, fname)

        records = metadata[[cls.MetadataFields.FILE_ID, Fields.FILE_NAME]].to_dict('records')

        logger.info(f"Downloading {len(records)} media...")

        parallel_exec(
            func=download_file,
            elements=records,
            url=lambda rec: f"{media_base_url}/{rec[Fields.FILE_NAME]}",
            dest_filename=get_dest_filename,
            verbose=False)

    def _apply_filters_and_mappings(self,
                                    categories: Union[List[str], str] = None,
                                    exclude_categories: Union[List[str], str] = None,
                                    min_score: float = None,
                                    max_score: float = None,
                                    category_mapping: dict = None,
                                    round_score_digits: int = None):
        if category_mapping is not None:
            cat_mappings = fix_category_mapping(category_mapping)
            self[Fields.LABEL] = lambda record: map_category(record[Fields.LABEL], cat_mappings)
        if categories is not None:
            self.filter_by_categories(categories, mode='include', inplace=True)
        if exclude_categories is not None:
            self.filter_by_categories(exclude_categories, mode='exclude', inplace=True)
        if min_score is not None or max_score is not None:
            self.filter_by_score(min_score=min_score, max_score=max_score, inplace=True)
        if round_score_digits is not None and Fields.SCORE in self.fields:
            annotations[Fields.SCORE] = (
                annotations[Fields.SCORE].apply(lambda x: round(x, round_score_digits)))
            self[Fields.SCORE] = lambda record: round(record[Fields.SCORE], round_score_digits)

    @classmethod
    def _get_filename(cls,
                      record: dict,
                      set_filename_with_id_and_extension: str = None) -> str:
        """Get the file name of the media based on its own id.

        Parameters
        ----------
        record : dict
            Dict that contains media information with the keys 'file_name' and 'id'
        set_filename_with_id_and_extension : str, optional
            Extension to be added to the id of each item to set the file name (default is None)

        Returns
        -------
        str
            File name of the media.
        """
        if set_filename_with_id_and_extension is not None:
            ext = set_filename_with_id_and_extension
            if ext.startswith('.'):
                ext = ext[1:]
            fname = f'{record[cls.MetadataFields.FILE_ID]}.{ext}'
        elif 'file_name' in record:
            fname = record[Fields.FILE_NAME]
        else:
            fname = f'{record[cls.MetadataFields.FILE_ID]}{cls.DEFAULT_EXT}'
        return fname.replace('\\', '/')

    def _format_item_for_storage(self, item):
        if self.root_dir is not None:
            return os.path.relpath(item, self.root_dir)
        return item

    #     endregion

    #     region BUILD AND UPDATE

    @classmethod
    def _fit_dataframe(cls, dataframe: pd.DataFrame, mapping_fields: dict = None) -> pd.DataFrame:
        dataframe = dataframe.dropna(how='all', axis=1).fillna('')
        if mapping_fields is not None:
            dataframe = dataframe.rename(columns=mapping_fields)
        if not cls.MetadataFields.FILE_ID in dataframe.columns:
            dataframe = cls._add_file_id_field_to_dataframe(dataframe)

        return dataframe

    def _fit_annotations(self,
                         annotations: pd.DataFrame,
                         verify_integrity: bool = True,
                         field_modified: Optional[str] = None) -> pd.DataFrame:
        if len(annotations) == 0:
            return annotations

        if verify_integrity:
            self._verify_integrity_annotations(annotations)
        if (Fields.LABEL in annotations.columns
                and (field_modified is None or field_modified == Fields.LABEL)):
            annotations[Fields.LABEL] = annotations[Fields.LABEL].apply(get_cleaned_label)
        if field_modified is not None:
            types = {k: v for k, v in self.AnnotationFields.TYPES.items() if k == field_modified}
        else:
            types = self.AnnotationFields.TYPES
        annotations = set_field_types_in_data(annotations, field_types=types)

        return annotations

    def _fit_metadata(self,
                      metadata: pd.DataFrame,
                      verify_integrity: bool = True,
                      field_modified: Optional[str] = None) -> pd.DataFrame:
        if len(metadata) == 0:
            return metadata

        if verify_integrity:
            self._verify_integrity_metadata(metadata)
        metadata = metadata.drop_duplicates(subset=self._key_field_metadata).reset_index(drop=True)
        if field_modified is not None:
            types = {k: v for k, v in self.MetadataFields.TYPES.items() if k == field_modified}
        else:
            types = self.MetadataFields.TYPES
        metadata = set_field_types_in_data(metadata, field_types=types)

        return metadata

    @classmethod
    def _extract_annotations_from_dataframe(cls,
                                            dataframe: pd.DataFrame,
                                            accept_all_fields: bool = False) -> pd.DataFrame:
        if len(dataframe) == 0:
            return dataframe

        dataframe_fields = set(dataframe.columns.values)
        if accept_all_fields:
            common_fld = cls._get_common_field_anns_meta()
            metadata_default_fields = set(get_default_fields(cls.MetadataFields))
            annotations_fields = (dataframe_fields - metadata_default_fields) | {common_fld}
        else:
            annotations_fields = set(get_default_fields(cls.AnnotationFields))

        annotations_fields &= dataframe_fields
        return dataframe[list(annotations_fields)]

    @classmethod
    def _extract_metadata_from_dataframe(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        if len(dataframe) == 0:
            return dataframe

        dataframe_fields = set(dataframe.columns.values)
        metadata_default_fields = set(get_default_fields(cls.MetadataFields))
        metadata_fields = metadata_default_fields & dataframe_fields
        return dataframe[list(metadata_fields)]

    def _update_modified_dataset(self,
                                 modified_df: pd.DataFrame,
                                 field_modified: Optional[str] = None):
        annotations = self._extract_annotations_from_dataframe(modified_df, accept_all_fields=True)
        metadata = self._extract_metadata_from_dataframe(modified_df)
        self._fit_and_set_annotations(
            annotations, verify_integrity=False, field_modified=field_modified)
        self._fit_and_set_metadata(metadata, verify_integrity=False, field_modified=field_modified)

    #     endregion

    #   endregion

    #   region AUXILIAR FACTORY METHODS

    @classmethod
    def _copy_dataset(cls, dataset: Dataset) -> Dataset:
        annotations = dataset.annotations.copy()
        metadata = dataset.metadata.copy()
        root_dir = dataset.root_dir
        instance = cls(annotations, metadata, root_dir, avoid_initialization=True)

        return instance

    #   endregion

    #   region SPLIT & PARTITION MANAGEMENT

    def _split(self, use_partitions: bool):
        """Set the `partitions` property in `params` with the `index` of the `pd.DataFrame`
        and the partition of each item in the dataset.
        """
        if not Fields.PARTITION in self.fields:
            return

        anns = self.annotations
        if not use_partitions:
            anns = anns.drop(Fields.PARTITION, axis=1)
            self._set_annotations(anns)
        else:
            part_values = anns[Fields.PARTITION].unique()
            inval = list(set([x for x in part_values if x.lower() not in Partitions.NAMES]))
            assert not inval, f"Invalid values in {Fields.PARTITION}. Allowed values: {Partitions}"

    def _add_partitions_to_dataset(self, partitions_to_idx: dict):
        """Adds the generated partitions to the dataset
        """
        anns_ids_and_parts_df = self.annotations[[Fields.ID]]
        anns_ids_and_parts_df[Fields.PARTITION] = ""
        for partition, idxs in partitions_to_idx.items():
            anns_ids_and_parts_df.loc[idxs, Fields.PARTITION] = partition
        self[Fields.PARTITION] = anns_ids_and_parts_df

    def _fix_partitioning_by_priorities(self, partitions_to_idx: dict):
        fix_partitioning_by_priorities(self.annotations, partitions_to_idx)
        self._add_partitions_to_dataset(partitions_to_idx)
    #   endregion

    #   region LABELMAP
    def _generate_labelmap_from_list(self, labelmap_list: list) -> dict:
        """Generates the labelmap from a list of class names, enumerating the elements that the
        list contains.

        Parameters
        ----------
        labelmap_list: list of str
            List that contains the class names of the labelmap

        Returns
        -------
        dict
            Dictionary of the form {index: class_name} with the elements of the labelmap
        """
        return {i: lbl for i, lbl in enumerate(labelmap_list)}

    def _generate_labelmap_from_categories(self) -> dict:
        """Generates the labelmap from the different categories that the dataset contains.
        If the dataset does not contain categories, `None` will be returned.

        Returns
        -------
        dict
            Dictionary of the form {index: class_name} with the elements of the labelmap
        """
        return dict(zip(range(len(self.classes)), self.classes))

    def _get_inverse_labelmap(self) -> dict:
        """Get dictionary with class names to ids

        Returns
        -------
        dict
            Dictionary with class names to ids in the form {'class_name': 'id'}
        """
        return {v: k for k, v in self.labelmap.items()}

    #   endregion

    # endregion


class Partitions():
    """Allowed types of dataset partitions
    """
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"
    NAMES = [TRAIN, TEST, VALIDATION]

    @classmethod
    def check_partition(cls, partition):
        assert partition in cls.NAMES, f"Partition must be one of: {', '.join(cls.NAMES)}"

    def __repr__(self):
        return ", ".join(self.NAMES)
