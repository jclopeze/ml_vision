#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

from .utils.dataset import seek_files
from .utils.dataset import fix_partitioning_by_priorities
from .utils.dataset import append_to_partition
from .utils.dataset import get_media_id_str_from_item
from .utils.dataset import get_cleaned_label
from .utils.dataset import get_mapping_classes
from .utils.dataset import map_category
from .utils.dataset import set_field_types_in_data
from .utils.dataset import get_random_id
from .utils.dataset import get_abspath_and_validate_item
from .utils.dataset import get_cats_from_source
from .utils.dataset import sample_data
from .utils.dataset import get_media_name_with_prefix
from .utils.dataset import get_default_fields
from .utils.dataset import Fields
from .utils.stratified_group_split import gradient_group_stratify
from .utils.misc import is_array_like
from .utils.misc import parallel_exec
from .utils.misc import download_file
from .utils.misc import get_chunk as get_chunk_func
from .utils.logger import get_logger, debugger

logger = get_logger(__name__)
val_filenames_debug = debugger.get_validate_filenames_env()
pd.options.mode.chained_assignment = None


class Dataset():

    class METADATA_FIELDS():
        ITEM: Final = Fields.ITEM
        MEDIA_ID: Final = Fields.MEDIA_ID
        FILENAME: Final = Fields.FILE_NAME
        DATE_CAPTURED: Final = Fields.DATE_CAPTURED
        LOCATION: Final = Fields.LOCATION

        TYPES = {
            MEDIA_ID: str,
            LOCATION: str
        }

    class ANNOTATIONS_FIELDS():
        ITEM: Final = Fields.ITEM
        LABEL: Final = Fields.LABEL
        ID: Final = Fields.ID
        PARTITION: Final = Fields.PARTITION
        SCORE: Final = Fields.SCORE

        TYPES = {
            ID: str,
            SCORE: float
        }

    FILES_EXTS = []
    DEFAULT_EXT = ""
    # region SPECIAL METHODS

    def __repr__(self):
        return repr(self.df)

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        # TODO: improve
        for record in self.records:
            yield record

    def __getitem__(self, fields):
        return self.df[fields]

    def __setitem__(self, field, value):
        self.set_field_values(field, values=value, inplace=True)

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
    def _key_field_annotations(self) -> str:
        return Fields.ID

    @property
    def _key_field_metadata(self) -> str:
        return Fields.ITEM

    @property
    def metadata_default_fields(self) -> set:
        return set(get_default_fields(self.METADATA_FIELDS))

    @property
    def annotations_default_fields(self) -> set:
        return set(get_default_fields(self.ANNOTATIONS_FIELDS))

    # endregion

    # region CONSTRUCTOR

    def __init__(self: Dataset,
                 annotations: pd.DataFrame,
                 metadata: pd.DataFrame,
                 root_dir: str = None,
                 use_partitions: bool = True,
                 **kwargs) -> Dataset:
        if kwargs.get('avoid_initialization', False):
            self._set_annotations(annotations)
            self._set_metadata(metadata)
            self._set_root_dir(root_dir)
            return

        verify_integrity = kwargs.get('verify_integrity', True)
        self._fit_and_set_annotations(annotations, verify_integrity=verify_integrity)
        self._fit_and_set_metadata(metadata, verify_integrity=verify_integrity)

        if len(annotations) == 0 or len(metadata) == 0:
            # TODO: return empty dataset
            return

        self._verify_annotations_and_metadata_consistency()
        self._apply_filters_and_mappings(**kwargs)
        self._set_abspaths_and_validate_filenames(root_dir, **kwargs)
        self._split(use_partitions)

    # endregion

    # region PUBLIC API METHODS

    #   region STATIC FACTORY METHODS

    @classmethod
    def from_dataframe(cls,
                       dataframe: pd.DataFrame,
                       root_dir: str,
                       **kwargs) -> Dataset:
        """
        Parameters
        ----------
        data : pd.DataFrame
            DataFrame object that make up the dataset.
        **kwargs
            Extra named arguments
        """
        assert isinstance(dataframe, pd.DataFrame), "A pd.DataFrame must be provided for data"
        if len(dataframe) == 0:
            logger.debug(f"The dataset does not contain any elements.")
            # TODO: return empty dataset

        dataframe = cls._fit_dataframe(dataframe, **kwargs)
        annotations = cls._extract_annotations_from_dataframe(dataframe, **kwargs)
        metadata = cls._extract_metadata_from_dataframe(dataframe)

        instance = cls(annotations, metadata, root_dir, **kwargs)

        return instance

    @classmethod
    def from_json(cls,
                  source_path: str,
                  root_dir: str,
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

        df = cls._get_dataframe_from_json(source_path, **kwargs)

        return cls.from_dataframe(df, root_dir, **kwargs)

    @classmethod
    def from_csv(cls,
                 source_path: str,
                 root_dir: str,
                 column_mapping: dict = None,
                 **kwargs) -> Dataset:
        """Create a Dataset from a csv file.

        Parameters
        ----------
        source_path : str
            Path of a csv file or a folder containing csv files that will be
            converted into a Dataset.
            The csv file(s) must have at least two columns, which represents
            `item` and `label` data.
        column_mapping : dict, optional
            Dictionary to map column names in dataset.
            If header=True mapping must be done by the column names,
            otherwise must be done by column positions.
            E.g.::
                header=True: column_mapping={"c1": "item", "c2": "label"}
                header=False: column_mapping={0: "item", 1: "label"}
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

        df = cls._get_dataframe_from_csv(source_path=source_path, **kwargs)

        kwargs['mapping_fields'] = column_mapping
        return cls.from_dataframe(df, root_dir, **kwargs)

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
        assert os.path.isdir(source_path), f"{source_path} is not a valid folder name"
        folder_path = os.path.abspath(source_path)
        logger.info(f"Creating dataset from folder {folder_path}")

        df = cls._get_dataframe_from_folder(source_path, use_labels, extensions, **kwargs)

        kwargs['validate_filenames'] = False
        return cls.from_dataframe(df, root_dir=source_path, **kwargs)

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
            datasets.append(dataset)

        annotations, metadata, root_dir = cls._get_concat_data_from_datasets(datasets)

        kwargs['validate_filenames'] = kwargs.get('validate_filenames', False)
        return cls(annotations, metadata, root_dir, **kwargs)

    #   endregion

    #   region TO_* METHODS

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
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        df = self.df
        if columns is not None:
            df = self[columns]
        df[Fields.ITEM] = df[Fields.ITEM].apply(self._format_item_for_storage)

        df.to_csv(dest_path, index=False, header=True)

        logger.info(f"CSV file {os.path.abspath(dest_path)} created")
        return dest_path

    #   endregion

    #   region AS_* METHODS

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
        df = self._merge_annotations_and_metadata()

        if len(df) == 0:
            # TODO: check empty datasets
            return df

        if columns is not None:
            columns = [c for c in columns if c in df.columns]
        if not columns:
            columns = self._get_ordered_fields_in_dataset()

        df = df[columns]

        return df

    #   endregion

    #   region GET_* METHODS

    def get_chunk(self,
                  num_chunks: int,
                  chunk_num: int,
                  partition: str = None,
                  verbose: bool = True) -> Dataset:
        """Function that divides the elements of a dataset into `num_chunks` chunks and
        returns a dataset with the elements of chunk number `chunk_num`

        Parameters
        ----------
        num_chunks : int
            Number of chunks into which the dataset is divided
        chunk_num : int
            Number of the chunk that will be taken to form the generated dataset.
            It must be in the range `[1, num_chunks]`
        partition : str, optional
            Partition from which the elements of the dataset will be taken, by default None
        verbose : bool, optional
            Whether or not to print the elements to be taken in the current process,
            by default True

        Returns
        -------
        Dataset
            Data set with the elements of chunk number `chunk_num` of the `num_chunks` chunks into
            which the original dataset was divided
        """
        items = self._get_items(partition=partition)
        items = get_chunk_func(items, num_chunks, chunk_num, verbose=verbose, sort_elements=True)
        new_ds = self.filter_by_column('item', items, inplace=False)
        return new_ds

    def get_annotations(self,
                        columns: Union[str, Iterable] = None,
                        remove_fields: Union[str, Iterable] = None) -> pd.DataFrame:
        if columns is not None and not is_array_like(columns):
            columns = [columns]
        if remove_fields is not None and not is_array_like(remove_fields):
            remove_fields = [remove_fields]

        annotations_fields = set(self.annotations.columns.values)
        if remove_fields is not None:
            annotations_fields -= set(remove_fields)
        if columns is not None:
            annotations_fields = annotations_fields & set(columns)
        return self[list(annotations_fields)]

    #   endregion

    #   region MUTATORS

    #     region VALUES OF FIELDS

    def set_field_values(self,
                         field: str,
                         values: Union[Callable, pd.DataFrame, object],
                         inplace: bool = True):
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
                       mapping_classes: Union[dict, str],
                       inplace: bool = True) -> Union[None, Dataset]:
        """Function that performs a mapping of the categories that a dataset contains, changing the
        value of the column `label` of the rows of each category by its corresponding target value

        Parameters
        ----------
        mapping_classes : Union[dict, str]
            Dictionary or path to a CSV file containing the mappings
        inplace : bool, optional
            If True, perform operation in-place, by default True

        Returns
        -------
        Dataset
            Instance of the Dataset
        """
        assert_msg = "The dataset doesn't contain the 'label' field"
        assert Fields.LABEL in self.annotations.columns, assert_msg

        instance = self if inplace else self.copy()
        mapping_classes_dict = get_mapping_classes(mapping_classes)
        anns = instance.annotations
        for from_cat, to_cat in mapping_classes_dict.items():
            if from_cat == '*' and to_cat != '*':
                remaining_cats = set(mapping_classes_dict.values()) - {to_cat}
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

    # TODO: Add verification of len(ds) == 0
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
        return self.filter_by_column(Fields.LABEL, cats, mode=mode, inplace=inplace)

    def filter_by_column(self,
                         column: str,
                         values: Union[Iterable, Any],
                         mode: Literal['include', 'exclude'] = 'include',
                         inplace: bool = True) -> Optional[Dataset]:
        """Method that filters the dataset by field `column`

        Parameters
        ----------
        column : str
            Name of the column to filter
        values : list or str
            Values to filter
        mode : str, optional
            Whether to 'include' or 'remove' registers with `values` in `column` in the dataset,
            by default 'include'
        inplace : bool, optional
            If True, perform operation in-place, by default True

        Returns
        -------
        Dataset
            Instance of the Dataset
        """
        assert mode in ('include', 'exclude'), f"Invalid value of mode: {mode}"
        instance = self if inplace else self.copy()

        if not is_array_like(values):
            values = [values]

        df = instance.df

        if mode == 'include':
            df = df[df[column].isin(values)]
        else:
            df = df[~df[column].isin(values)]

        instance._update_modified_dataset(df)

        logger.debug(f"The elements of the dataset were filtered by the field '{column}' "
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
            return self.filter_by_column(Fields.PARTITION, partition, mode=mode, inplace=inplace)
        return self

    def filter_by_label_counts(self,
                               min_label_counts: int = None,
                               max_label_counts: int = None,
                               label_counts: int = None,
                               inplace: bool = True) -> Optional[Dataset]:
        """Filter the dataset with respect to the number of samples per label it contains

        Parameters
        ----------
        min_label_counts : int, optional
            Minimum number of samples per label, by default None
        max_label_counts : int, optional
            Maximum number of samples per label, by default None
        label_counts : int, optional
            Exact number of samples per label, by default None
        inplace : bool, optional
            If True, perform operation in-place, by default True

        Returns
        -------
        Dataset
            Instance of the resulting dataset
        """
        assert_cond = bool(min_label_counts) + bool(max_label_counts) + bool(label_counts) == 1
        msg = "You must configure only one of: min_label_counts, max_label_counts and label_counts"
        assert assert_cond, msg

        instance = self if inplace else self.copy()
        counts = self.label_counts()
        if min_label_counts is not None:
            labels_to_search = counts[counts >= min_label_counts].index.values
        elif max_label_counts is not None:
            labels_to_search = counts[counts <= min_label_counts].index.values
        else:
            labels_to_search = counts[counts == min_label_counts].index.values

        self.filter_by_categories(labels_to_search, mode='include', inplace=True)

        return instance

    def filter_by_score(self,
                        max_score: float = None,
                        min_score: float = None,
                        score: float = None,
                        column_name: str = Fields.SCORE,
                        inplace: bool = True) -> Optional[Dataset]:
        """Method that filters the predictions by the `score` column

        Parameters
        ----------
        max_score : float, optional
            Float number in [0.,1.] that indicates the maximum value of the `score` column,
            by default None
        min_score : float, optional
            Float number in [0.,1.] that indicates the minimum value of the `score` column,
            by default None
        score : float, optional
            Float number in [0.,1.] that indicates the exact value of the `score` column,
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
               inplace: bool = True) -> Optional[Dataset]:
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
               inplace: bool = True) -> Optional[Dataset]:
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

    def take(self, n: int) -> Dataset:
        return self.sample(n=n, use_labels=False, use_partitions=False, inplace=False)

    #     endregion

    #   endregion

    #   region FACTORY METHODS

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
        return cls._copy_dataset(dataset)

    #   endregion

    #   region DEBUGGING METHODS

    def labels(self) -> pd.Series:
        return self.label_counts()

    def item0(self) -> str:
        return self.annotations.iloc[0][Fields.ITEM]

    def valid0(self) -> bool:
        return os.path.isfile(self.item0())

    def label_counts(self) -> pd.Series:
        """Get the `value_counts` of the label field in the dataset

        Returns
        -------
        pd.Series
            Series with the `value_counts` of the label field
        """
        assert Fields.LABEL in self.fields
        return self[Fields.LABEL].value_counts()

    #   endregion

    # endregion

    # region PRIVATE API METHODS

    #   region CALLBACKS

    @classmethod
    def _get_dataframe_from_json(cls, source_path, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def _get_dataframe_from_csv(cls, source_path: str, **kwargs) -> pd.DataFrame:
        usecols = kwargs.get('usecols')
        na_values = kwargs.get('na_values', 'nan').split(',')
        header = kwargs.get('header', 0)
        keep_default_na = kwargs.get('keep_default_na', False)

        read_csv = partial(
            pd.read_csv,
            usecols=usecols,
            header=header,
            na_values=na_values,
            keep_default_na=keep_default_na)

        if os.path.isdir(source_path):
            regex_fname = kwargs.get('regex_filename')
            csvs = seek_files(source_path, seek_name=regex_fname, seek_extension=[".csv"])
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
                                   use_labels: bool = True,
                                   extensions: List = None,
                                   lower_case_exts: bool = True,
                                   **_) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        source_path : str
            _description_
        use_labels : bool, optional
            _description_, by default True
        extensions : List, optional
            _description_, by default None
        lower_case_exts : bool, optional
            _description_, by default True

        Returns
        -------
        pd.DataFrame
            _description_
        """
        extensions = extensions or cls.FILES_EXTS
        assert extensions, "You must specify extensions"

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
        annotations = pd.DataFrame()
        metadata = pd.DataFrame()
        root_dirs = set()
        for ds in datasets:
            if len(ds) == 0:
                continue
            _metadata = ds.metadata
            _annotations = ds.get_annotations()
            annotations = pd.concat([annotations, _annotations], ignore_index=True)
            metadata = pd.concat([metadata, _metadata], ignore_index=True)
            root_dirs |= {ds.root_dir}

        root_dirs = list(root_dirs)
        if len(root_dirs) == 1 and root_dirs[0] != '':
            # The media dirs are all the same, so we can pick it for the new dataset
            root_dir = root_dirs[0]
        else:
            # The media dirs are different, so we can't pick any for the new dataset
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

    def _set_abspaths_and_validate_filenames(self,
                                             root_dir: Path,
                                             not_exist_ok: bool = False,
                                             validate_filenames: bool = True,
                                             **_):
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

        validate_filenames = validate_filenames and val_filenames_debug

        self._set_root_dir(root_dir)

        if len(self) == 0:
            # TODO: Check empty dataset
            return

        items_to_abspaths = Manager().dict()
        invalid_items = Manager().list()

        parallel_exec(
            func=get_abspath_and_validate_item,
            elements=self.items,
            item=lambda item: item,
            root_dir=root_dir,
            validate_filenames=validate_filenames,
            not_exist_ok=not_exist_ok,
            items_to_abspaths=items_to_abspaths,
            invalid_items=invalid_items)
        self[Fields.ITEM] = lambda record: items_to_abspaths[record[Fields.ITEM]]

        n_invalid = len(invalid_items)
        if n_invalid > 0:
            logger.info(f'{n_invalid} invalid items found that were ignored')
            self.filter_by_column(
                Fields.ITEM, values=list(invalid_items), mode='exclude', inplace=True)

        n_items = len(self.metadata)
        logger.debug(f'{n_items} {"" if validate_filenames else "not"} validated items found')

    @classmethod
    def _add_media_id_field_to_dataframe(cls, dataframe: pd.DataFrame):
        """Assigns the field `Fields.MEDIA_ID` in `dataframe` from the information contained
        in the field `Fields.ITEM`

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe with media file information from the dataset
        """
        stems = dataframe[Fields.ITEM].apply(lambda x: Path(x).stem)
        if stems.nunique() == dataframe[Fields.ITEM].nunique():
            dataframe[cls.METADATA_FIELDS.MEDIA_ID] = stems
        else:
            dataframe[cls.METADATA_FIELDS.MEDIA_ID] = (
                dataframe[Fields.ITEM].apply(get_media_id_str_from_item))

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
        """Get the list of distinct values in the 'label' column of the dataset.
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

    def _get_items(self, partition: str = None) -> list:
        """Get a list of unique items in the dataset, optionally filtered by
        partition name.

        Parameters
        ----------
        partition : str or None
            Name of the partition from which the records will be obtained.
            If None, dataset will not be filtered.

        Returns
        -------
        list or set
            List of unique items in the dataset
        """
        if partition is None:
            if len(self.metadata) > 0:
                items = self.metadata[Fields.ITEM].unique()
            else:
                items = []
        else:
            Partitions.check_partition(partition)
            df = self.df
            items = df.loc[df[Fields.PARTITION] == partition][Fields.ITEM].unique()

        return list(items)

    def _get_default_fields_of_dataset(self) -> List:
        default_fields_anns = get_default_fields(self.ANNOTATIONS_FIELDS)
        default_fields_meta = get_default_fields(self.METADATA_FIELDS)
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
        metadata_default_fields = set(get_default_fields(cls.METADATA_FIELDS))
        annotations_default_fields = set(get_default_fields(cls.ANNOTATIONS_FIELDS))
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

        assert_msg = f"Annotations data must contain the column {Fields.ITEM}"
        assert Fields.ITEM in annotations.columns, assert_msg

        if self._key_field_annotations in annotations.columns:
            assert_cond = annotations[self._key_field_annotations].nunique() == len(annotations)
            assert assert_cond, f"The field {self._key_field_annotations} must have unique values"
        else:
            annotations[self._key_field_annotations] = [get_random_id()
                                                        for _ in range(len(annotations))]

    def _verify_integrity_metadata(self, metadata: pd.DataFrame):
        assert isinstance(metadata, pd.DataFrame), "Media data must be a pd.DataFrame"

        cnd = self.METADATA_FIELDS.MEDIA_ID in metadata.columns and Fields.ITEM in metadata.columns
        msg = f"Metadata must contain the fields {self.METADATA_FIELDS.MEDIA_ID} and {Fields.ITEM}"
        assert cnd, msg

    #     endregion

    #     region AUXILIAR METHODS
    @classmethod
    def _download(cls,
                  dest_path: str,
                  metadata: pd.DataFrame,
                  set_filename_with_id_and_ext: str,
                  **kwargs):
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
        media_base_url = kwargs.get('media_base_url')
        cond = media_base_url is not None
        assert cond, f"In order to download media you must assign media_base_url parameter"
        os.makedirs(dest_path, exist_ok=True)

        def get_dest_filename(media_dict) -> str:
            fname = cls._get_filename(media_dict, set_filename_with_id_and_ext)
            return os.path.join(dest_path, fname)

        media_dict = metadata[[cls.METADATA_FIELDS.MEDIA_ID, Fields.FILE_NAME]].to_dict('records')

        logger.info(f"Downloading {len(media_dict)} media...")

        parallel_exec(
            func=download_file,
            elements=media_dict,
            url=lambda img_dict: f"{media_base_url}/{img_dict[Fields.FILE_NAME]}",
            dest_filename=get_dest_filename,
            verbose=False)

    def _apply_filters_and_mappings(self,
                                    categories: Union[List[str], str] = None,
                                    exclude_cats: Union[List[str], str] = None,
                                    min_score: float = None,
                                    max_score: float = None,
                                    mapping_classes: dict = None,
                                    mapping_classes_from_col: str = None,
                                    mapping_classes_to_col: str = None,
                                    mapping_classes_filter_expr: Callable = None,
                                    round_score_digits: int = None,
                                    **_):
        if mapping_classes is not None:
            cat_mappings = get_mapping_classes(
                mapping_classes,
                from_col=mapping_classes_from_col,
                to_col=mapping_classes_to_col,
                filter_expr=mapping_classes_filter_expr)
            self[Fields.LABEL] = lambda record: map_category(record[Fields.LABEL], cat_mappings)
        if round_score_digits is not None and Fields.SCORE in self.fields:
            annotations[Fields.SCORE] = (
                annotations[Fields.SCORE].apply(lambda x: round(x, round_score_digits)))
            self[Fields.SCORE] = lambda record: round(record[Fields.SCORE], round_score_digits)
        if categories is not None:
            self.filter_by_categories(categories, mode='include', inplace=True)
        if exclude_cats is not None:
            self.filter_by_categories(exclude_cats, mode='exclude', inplace=True)
        if min_score is not None or max_score is not None:
            self.filter_by_score(min_score=min_score, max_score=max_score, inplace=True)

    @classmethod
    def _get_filename(cls,
                      media_dict: dict,
                      set_filename_with_id_and_extension: str = None) -> str:
        """Get the file name of the media based on its own id.

        Parameters
        ----------
        media_dict : dict
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
            fname = f'{media_dict[cls.METADATA_FIELDS.MEDIA_ID]}.{ext}'
        elif 'file_name' in media_dict:
            fname = media_dict[Fields.FILE_NAME]
        else:
            fname = f'{media_dict[cls.METADATA_FIELDS.MEDIA_ID]}{cls.DEFAULT_EXT}'
        return fname.replace('\\', '/')

    def _format_item_for_storage(self, item):
        if self.root_dir is not None:
            return os.path.relpath(item, self.root_dir)
        return item

    #     endregion

    #     region BUILD AND UPDATE

    @classmethod
    def _fit_dataframe(cls, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        dataframe = dataframe.dropna(how='all', axis=1).fillna('')
        if kwargs.get('mapping_fields') is not None:
            dataframe = dataframe.rename(columns=kwargs['mapping_fields'])
        if not cls.METADATA_FIELDS.MEDIA_ID in dataframe.columns:
            dataframe = cls._add_media_id_field_to_dataframe(dataframe)

        return dataframe

    def _fit_annotations(self,
                         annotations: pd.DataFrame,
                         verify_integrity: bool = True,
                         field_modified: Optional[str] = None) -> pd.DataFrame:
        if verify_integrity:
            self._verify_integrity_annotations(annotations)
        if (Fields.LABEL in annotations.columns
                and (field_modified is None or field_modified == Fields.LABEL)):
            annotations[Fields.LABEL] = annotations[Fields.LABEL].apply(get_cleaned_label)
        if field_modified is not None:
            types = {k: v for k, v in self.ANNOTATIONS_FIELDS.TYPES.items() if k == field_modified}
        else:
            types = self.ANNOTATIONS_FIELDS.TYPES
        annotations = set_field_types_in_data(annotations, field_types=types)

        return annotations

    def _fit_metadata(self,
                      metadata: pd.DataFrame,
                      verify_integrity: bool = True,
                      field_modified: Optional[str] = None) -> pd.DataFrame:
        if verify_integrity:
            self._verify_integrity_metadata(metadata)
        metadata = metadata.drop_duplicates(subset=self._key_field_metadata).reset_index(drop=True)
        if field_modified is not None:
            types = {k: v for k, v in self.METADATA_FIELDS.TYPES.items() if k == field_modified}
        else:
            types = self.METADATA_FIELDS.TYPES
        metadata = set_field_types_in_data(metadata, field_types=types)

        return metadata

    @classmethod
    def _extract_annotations_from_dataframe(cls,
                                            dataframe: pd.DataFrame,
                                            accept_all_fields: bool = False,
                                            **_) -> pd.DataFrame:
        dataframe_fields = set(dataframe.columns.values)
        if accept_all_fields:
            common_fld = cls._get_common_field_anns_meta()
            metadata_default_fields = set(get_default_fields(cls.METADATA_FIELDS))
            annotations_fields = (dataframe_fields - metadata_default_fields) | {common_fld}
        else:
            annotations_fields = set(get_default_fields(cls.ANNOTATIONS_FIELDS))

        annotations_fields &= dataframe_fields
        return dataframe[list(annotations_fields)]

    @classmethod
    def _extract_metadata_from_dataframe(cls, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe_fields = set(dataframe.columns.values)
        metadata_default_fields = set(get_default_fields(cls.METADATA_FIELDS))
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