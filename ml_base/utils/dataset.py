#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
from typing import Optional

import math
import numpy as np
from typing import Iterable, Union, Callable, List
import uuid

from .misc import is_array_like
from .logger import get_logger

logger = get_logger(__name__)

LABELMAP_FILENAME = 'labels.txt'
STD_DATEFORMAT = '%Y-%m-%d %H:%M:%S'
pd.options.mode.chained_assignment = None


def seek_files(path: str,
               seek_name: str = None,
               seek_extension: str = None,
               recursive: bool = True,
               order_by_filename: bool = False) -> list:
    """
    Retrieve all files in a folder on format
        path, relative path (from path), file name

    Parameters
    ----------
    path : str
        Path to find all files
    seek_name : str, optional
        Name of file needed (it could be a regex) (default is None)
    seek_extension : str, optional
        Extension of file needed (it could be a regex) (default is None)
    recursive : bool
        Whether or not to seek files also in subdirectories (default is True)
    order_by_filename : bool
        Whether or not to order results by filename.

    Returns
    -------
    List
        List of found files
    """
    paths = []

    assert_cond = seek_name is not None or seek_extension is not None
    assert assert_cond, "You must provide at least one of seek_name and seek_extension"

    if seek_extension is not None:
        seek_extension = seek_extension if isinstance(seek_extension, list) else [seek_extension]
        seek_extension = [*map(lambda x: re.sub("[.]*", "", x, count=1), seek_extension)]
    else:
        seek_extension = [r'\w*']

    if seek_name is not None:
        seek_name = seek_name if isinstance(seek_name, list) else [seek_name]
    else:
        seek_name = ['.*']

    for _file in [name for name in os.listdir(path)]:
        if _file.startswith("."):
            continue

        fname = os.path.basename(_file).lower()
        matches = [
            f'{name}.{ext}'
            for name in seek_name for ext in seek_extension if re.match(f"^{name}[.]{ext}$", fname)
        ]

        if recursive and os.path.isdir(os.path.join(path, _file)):
            temps = seek_files(os.path.join(path, _file),
                               seek_name=seek_name,
                               seek_extension=seek_extension,
                               recursive=True)
            paths += temps
        elif len(matches) > 0:
            if os.path.isfile(os.path.join(path, _file)):
                paths += [{"path": path,
                           "title": _file}]

    if order_by_filename:
        paths = sorted(paths, key=lambda i: i['title'])

    return paths


def read_labelmap_file(labelmap_path: str) ->  Optional[dict]:
    """Reads the labels file and returns a mapping from ID to class name.

    Parameters
    ----------
    labelmap_path : str
        The filename where the class names are read.

    Returns
    -------
    dict
        A map from a label (integer) to class name.
    """
    if os.path.isdir(labelmap_path):
        labelmap_path = os.path.join(labelmap_path, LABELMAP_FILENAME)
        if not os.path.isfile(labelmap_path):
            return None
    with open(labelmap_path, 'r') as f:
        lines = f.read()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index+1:]
    return labels_to_class_names


def write_labelmap_file(labelmap: dict, dest_path: str) -> str:
    """Writes a file with the map of labels to class names.

    Parameters
    ----------
    labelmap: dict
        A map of (integer) labels to class names.
    dest_path: str
        The path of the file (or directory) in which the labelmap file should be written.

    Returns
    -------
    str
        Path of the file where the labelmap file was written
    """
    if not dest_path.lower().endswith('.txt'):
        dest_path = os.path.join(dest_path, LABELMAP_FILENAME)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, 'w') as f:
        for label in labelmap:
            class_name = labelmap[label]
            f.write('%d:%s\n' % (label, class_name))
    logger.debug(f"File {dest_path} with labelmap was created")
    return dest_path


def get_mapping_classes(mapping_classes: Union[dict, str],
                        from_col: Union[str, int] = None,
                        to_col: Union[str, int] = None,
                        filter_expr: Callable = None) -> Optional[dict]:
    """Function that gets a mapping of categories, either to group them into super-categories or to
    match them to those in other datasets.
    The mapping can be done either with a dictionary or with a CSV file.
    In both cases you can use the wildcard `*` (as the `key` of the dict or the column `0` of the
    CSV) to indicate 'all other current categories in the data set'.
    E.g., `{'Homo sapiens': 'Person', '*': 'Animal'}` will designate the Homo sapiens category as
    'Person' and the rest of the categories as 'Animal'.
    By default, the resulting mappings will be in the form `{orig_cat_id: dest_cat_name}`

    Parameters
    ----------
    mapping_classes : dict or str or None
        Dictionary or path to a CSV file containing the mappings.
        In the case of a dictionary, the `key` of each element is the current name of the category,
        and `value` is the name to be given to that category.
        In the case of a CSV file, the file must contain two columns and have no header.
        The column `0` is the current name of the category, and the column `1` is the name to be
        given to that category.
        If None, None is returned
    from_col : str or int, optional
        Name or position (0-based) of the column to be used as 'from' in the mapping, in case of
        `mapping_classes` is a CSV. By default None
    to_col : str or int, optional
        Name or position (0-based) of the column to be used as 'to' in the mapping, in case of
        `mapping_classes` is a CSV. By default None
    filter_expr : Callable, optional
        A Callable that will be used to filter the CSV records in which the mapping is found,
        in case of `mapping_classes` is a CSV. By default None

    Returns
    -------
    dict
        Dictionary containing the category mappings

    Raises
    ------
    ValueError
        In case `mapping_classes` is neither dictionary nor file path
    """
    if mapping_classes is None:
        return None
    if type(mapping_classes) == dict:
        return get_mapping_classes_from_dict(mapping_classes_dict=mapping_classes)
    elif os.path.isfile(mapping_classes):
        return get_mapping_classes_from_csv(
            mapping_classes_csv=mapping_classes,
            from_col=from_col, to_col=to_col, filter_expr=filter_expr)
    else:
        raise ValueError(f"Invalid value for mapping_classes")


def get_mapping_classes_from_csv(mapping_classes_csv: str,
                                 from_col: Union[str, int] = None,
                                 to_col: Union[str, int] = None,
                                 filter_expr: Callable = None) -> Optional[dict]:
    """Function that gets a mapping of categories, either to group them into super-categories or to
    match them to those in other datasets, from the definitions contained in `mapping_classes_csv`.
    You can use the wildcard `*` in the column `0` to indicate 'all other current categories in the
    dataset'. E.g.,
       `Homo sapiens  |   Person`
        `*          |   Animal`
    will designate the 'Homo sapiens' category as 'Person' and the rest of the categories as
    'Animal'.
    By default, the resulting mappings will be in the form {`orig_cat_id`: `dest_cat_name`}

    Parameters
    ----------
    mapping_classes_csv : str
        Path to a CSV file containing the mappings. The file must contain two columns and have no
        header. The column `0` is the current name of the category and the column `1` is the name
        to be given to that category.
    from_col : str or int, optional
        Name or position (0-based) of the column to be used as 'from' in the mapping,
        by default None
    to_col : str or int, optional
        Name or position (0-based) of the column to be used as 'to' in the mapping,
        by default None
    filter_expr : Callable, optional
        A Callable that will be used to filter the CSV records in which the mapping is found.
        By default None

    Returns
    -------
    dict
        Dictionary containing the category mappings

    """
    if not os.path.isfile(mapping_classes_csv):
        return None
    mapping_classes_dict = {}
    if from_col is not None and to_col is not None:
        df = pd.read_csv(mapping_classes_csv, header=0, na_values=['nan'], keep_default_na=False)
        if callable(filter_expr):
            df = df[df.apply(filter_expr, axis=1)].reset_index(drop=True)
        df = df.rename(columns={from_col: 'from', to_col: 'to'})[['from', 'to']]
    else:
        df = pd.read_csv(mapping_classes_csv, header=None, names=["from", "to"],
                         na_values=['nan'], keep_default_na=False)
    for _, x in df.iterrows():
        if type(x["from"]) is str:
            key = get_cleaned_label(x["from"])
        else:
            key = x["from"]
        value = x["to"]
        mapping_classes_dict[key] = get_cleaned_label(value)
    return mapping_classes_dict


def get_mapping_classes_from_dict(mapping_classes_dict: dict) -> dict:
    """Function that gets a mapping of categories, either to group them into super-categories or to
    match them to those in other datasets, from the definitions contained in
    `mapping_classes_dict`.
    You can use the wildcard `*` as the `key` to indicate 'all other current categories in the
    dataset'. E.g.,
    E.g., `{'Homo sapiens': 'Person', '*': 'Animal'}` will designate the Homo sapiens category as
    'Person' and the rest of the categories as 'Animal'.
    By default, the resulting mappings will be in the form `{orig_cat_id: dest_cat_name}`

    Parameters
    ----------
    mapping_classes_dict : dict
        Dictionary containing the mappings. The `key` of each element is the current name of the
        category, and `value` is the name to be given to that category.

    Returns
    -------
    dict
        Dictionary containing the category mappings

    """
    mapping_classes_dict_new = {}
    for key, value in mapping_classes_dict.items():
        if type(key) is str:
            key = get_cleaned_label(key)
        mapping_classes_dict_new[key] = get_cleaned_label(value)
    return mapping_classes_dict_new


def get_cleaned_label(value: str) -> str:
    return " ".join(value.lower().strip().split())


class Fields():
    ITEM = 'item'
    LABEL = 'label'
    PARTITION = 'partition'
    ID = 'id'
    FILE_NAME = "file_name"
    DATE_CAPTURED = "date_captured"
    LOCATION = "location"
    MEDIA_ID = "media_id"
    SCORE = "score"


def get_cats_from_source(categories: Union[Iterable, str, None],
                         clean_names: bool = True) -> Optional[list[str]]:
    """Obtains the categories from one of the following sources:
    - List of strings containing the categories.
    - Path to a CSV file that has no header and contains the categories in `column 0`.
    - Path to a text file containing the categories separated by line breaks.
    - String containing the categories separated by commas.

    Parameters
    ----------
    categories : array-like, str or None
        Source from which to obtain the categories. If None, None is returned
    clean_names : bool, optional
        Whether to clean or not the category names, converting to lower case and removing spaces at
        the beginning and at the end. By default True

    Returns
    -------
    list
        List of the categories

    Raises
    ------
    ValueError
        In case `categories` is invalid
    """
    if categories is None:
        return None
    elif is_array_like(categories):
        results = categories
    elif os.path.isfile(categories):
        if categories.lower().endswith('.csv'):
            df = pd.read_csv(categories, header=None, names=["cat"])
            results = [row["cat"] for _, row in df.iterrows()]
        else:   # text file
            with open(categories, mode="r") as f:
                results = [cat.replace("\n", "") for cat in f.readlines() if cat.replace("\n", "")]
    elif type(categories) is str:
        results = categories.split(",")
    else:
        raise ValueError(f"Invalid value for categories")

    return [get_cleaned_label(x) if clean_names else x for x in results]


def sample_data(data: pd.DataFrame,
                n: Union[str, int, float, dict],
                random_state: int = None,
                groupby: str = 'label'):
    """Function that samples the elements of a DataFrame by grouping them by field(s) `groupby` and
    taking a random number determined by the value of `n`.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the elements to be sampled
    n : str, int, float or dict
        If int, indicates the maximum number of samples to be taken from among the elements of each
        `groupby`.
        If float (0, 1), it refers to the percentage of elements to be taken from each `groupby`.
        If str, indicates the name of the method to be used in the sampling operation. The possible
        method names are:
        * fewer: will take as the sample number the smallest value of the elements grouped by label
        in the set.
        * mean: calculates the average of the element counts for each category and takes this value
        as the maximum number of elements for each category.

    random_state : int, optional
        Seed for random number generator, by default None
    groupby : list of str, str or None, optional
        Field(s) of `data` by which the data will be grouped for sampling.
        If None, no grouping will be done and sampling will be performed for all `data`.
        By default 'label'
    Returns
    -------
    pd.DataFrame
        Instance of the modified DataFrame after sampling
    """
    if n is None:
        return data

    def sample_num(x, n, rand_state):
        return x.sample(n=n, random_state=rand_state) if len(x) > n else x

    def sample_perc(x, frac, rand_state):
        return x.sample(frac=frac, random_state=rand_state)

    def sample_dict(x, n, rand_state):
        if type(x.name) is tuple:
            n_val = n.get(x.name[0])
        else:
            n_val = n.get(x.name)
        if type(n_val) is float and n_val < 1.:
            return sample_perc(x, frac=n_val, rand_state=rand_state)
        elif type(n_val) in (int, float) and n_val >= 1.:
            return sample_num(x, n=n_val, rand_state=rand_state)
        return x

    # Rudimentary way to avoid grouping by any column
    if groupby is None or (groupby == 'label' and Fields.LABEL not in data):
        data['dummy_col'] = 'dummy_val'
        groupby = 'dummy_col'

    if random_state is not None:
        data = data.sort_values(Fields.ID)

    if type(n) is float and n < 1.:
        data = (
            data
            .reset_index(drop=True)
            .groupby(groupby, as_index=False)
            .apply(sample_perc, frac=n, rand_state=random_state)
            .reset_index(drop=True)
        )
    if type(n) is str:
        if n == 'fewer':
            counts_res = (
                data
                .groupby(groupby).apply(lambda x: len(x))
                .reset_index(name='counts')
                .sort_values(by="counts", axis=0, ascending=True, inplace=False)
                .iloc[0]["counts"]
            )
        if n == 'mean':
            counts_res = (
                math.ceil(data.groupby(groupby).apply(lambda x: len(x))
                          .reset_index(name='counts')
                          .counts.mean())
            )
        n = int(counts_res)
    if type(n) in (int, float) and n >= 1:
        data = (
            data
            .reset_index(drop=True)
            .groupby(groupby, as_index=False)
            .apply(sample_num, n=n, rand_state=random_state)
            .reset_index(drop=True)
        )
    elif type(n) is dict:
        data = (
            data
            .reset_index(drop=True)
            .groupby(groupby, as_index=False)
            .apply(sample_dict, n=n, rand_state=random_state)
            .reset_index(drop=True)
        )
    if 'dummy_col' in data.columns:
        data.drop('dummy_col', axis=1, inplace=True)
    return data


def set_partition_idxs(partitions_to_idx: dict, partition: str, idxs: Iterable):
    """Set the elements passed in `idxs` in the `partition` entry of `partitions_to_idx`.

    Parameters
    ----------
    partitions_to_idx : dict
        _description_
    partition : str
        Name of the partition to set
    idxs : Iterable
        Indices of the elements in the dataset that will be part of the partition
    """
    if len(idxs) == 0 and partition in partitions_to_idx:
        del partitions_to_idx[partition]
    else:
        partitions_to_idx[partition] = idxs


def append_to_partition(partitions_to_idx: dict, partition: str, idxs: Iterable):
    if not isinstance(idxs, (np.ndarray,)):
        idxs = np.array(idxs)
    old_value = partitions_to_idx.get(partition, [])
    new_value = np.append(old_value, idxs)
    set_partition_idxs(partitions_to_idx, partition, new_value)


def fix_partitioning_by_priorities(df, partitions_to_idx, priority_order=['train', 'test', 'validation']):
    """Repairs the partitioning of the dataset based on the number of samples for each tag, moving
    elements to higher priority partitions from lower priority partitions

    Parameters
    ----------
    dataset : Dataset
        Dataset to be processed
    priority_order : list, optional
        List of partitions, given in the desired order of priority,
        by default ['train', 'test', 'validation']
    """
    parts = [part for part in priority_order if part in partitions_to_idx.keys()]
    data = df[Fields.LABEL].value_counts().to_frame()
    for partition in parts:
        data[partition] = df[df[Fields.PARTITION] == partition][Fields.LABEL].value_counts()
    data.fillna(0, inplace=True)
    # prior_1 goes from highest to lowest + 1 priorities
    for prior_1 in range(len(parts) - 1):
        prior_1_part = parts[prior_1]
        # prior_2 goes from lowest to prior_1 - 1 priorities
        for prior_2 in reversed(range(prior_1 + 1, len(parts))):
            prior_2_part = parts[prior_2]

            # Get labels without samples for prior_1_part and at least one sample in prior_2_part
            lbls = data[(data[prior_1_part] == 0) & (data[prior_2_part] > 0)].index.values
            idxs = set(df[(df[Fields.LABEL].isin(lbls)) & (
                df[Fields.PARTITION] == prior_2_part)].index.values)
            prior_1_idx = set(partitions_to_idx.get(prior_1_part, []))
            prior_2_idx = set(partitions_to_idx.get(prior_2_part, []))
            # Move elements from prior_2_part to prior_1_part
            set_partition_idxs(partitions_to_idx, prior_1_part, list(prior_1_idx | idxs))
            set_partition_idxs(partitions_to_idx, prior_2_part, list(prior_2_idx - idxs))

            # When there are categories with more elements in prior_2_part than elements in
            # prior_1_part, then swap them
            counts_per_parts = (
                df[df.partition.isin([prior_1_part, prior_2_part])]
                .groupby([Fields.LABEL, Fields.PARTITION])[Fields.ITEM]
                .count()
                .unstack()
                .fillna(0))
            lbls = (
                counts_per_parts[counts_per_parts.apply(
                    lambda rec: rec[prior_2_part] > rec[prior_1_part], axis=1)].index.values)
            if len(lbls) > 0:
                lbls_in_prior_1_idxs = set(df[(df[Fields.LABEL].isin(lbls)) & (
                    df[Fields.PARTITION] == prior_1_part)].index.values)
                lbls_in_prior_2_idxs = set(df[(df[Fields.LABEL].isin(lbls)) & (
                    df[Fields.PARTITION] == prior_2_part)].index.values)
                prior_1_idx = set(partitions_to_idx.get(prior_1_part, []))
                prior_2_idx = set(partitions_to_idx.get(prior_2_part, []))
                idxs = list(prior_1_idx - lbls_in_prior_1_idxs | lbls_in_prior_2_idxs)
                set_partition_idxs(partitions_to_idx, prior_1_part, idxs)
                idxs = list(prior_2_idx - lbls_in_prior_2_idxs | lbls_in_prior_1_idxs)
                set_partition_idxs(partitions_to_idx, prior_2_part, idxs)


def get_abspath_and_validate_item(item,
                                  root_dir,
                                  validate_filenames=True,
                                  not_exist_ok=False,
                                  items_to_abspaths=None,
                                  invalid_items=None):
    """Creates the path that `item` will have once it is assigned the directory
    `root_dir`

    Parameters
    ----------
    item : str
        Item in a dataset
    items_to_abspaths : dict
        Dictionary containing the mapping of each `item` to its corresponding new file path
    current_root_dir : str
        The current media directory for the item.
    root_dir : str
        The media directory which will be assigned to the item.
    validate_filenames : bool
        Whether or not to validate that the items exist. Set to False to speed up execution time
    not_exist_ok : bool
        Whether to include media that exist and silently pass exceptions for those media that are
        not present

    """
    if root_dir is not None and not item.startswith(root_dir):
        new_item = os.path.join(root_dir, item)
    else:
        new_item = item

    if validate_filenames and not os.path.isfile(new_item):
        if not_exist_ok:
            if invalid_items is not None:
                invalid_items.append(new_item)
        else:
            raise ValueError(f"Item not found in file system: {new_item}")

    if items_to_abspaths is not None:
        items_to_abspaths[item] = new_item

    return new_item


def map_category(cat_name: str, cat_mappings: dict) -> str:
    if cat_name in cat_mappings:
        return cat_mappings[cat_name]
    elif '*' in cat_mappings:
        return cat_mappings['*']
    return cat_name


def get_media_id_str_from_item(item):
    item = (item
            .replace(' ', '_')
            .replace('..', '_')
            .replace('.', '_')
            .replace('/', '-'))
    return item


def set_field_types_in_data(data: pd.DataFrame, field_types: dict) -> pd.DataFrame:
    """Casts the data types in the fields contained in `field_types` in the DataFrame `data`

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with data from a dataset

    Returns
    -------
    pd.DataFrame
        The modified dataframe
    """
    for field_name, field_type in field_types.items():
        if field_name in data.columns:
            data[field_name] = data[field_name].astype(field_type, errors='ignore')

    return data


def get_media_name_with_prefix(record, field, separator, media_name):
    prefix = f'{record[field]}{separator}' if record[field] else ''
    dirname, fname = os.path.split(media_name)
    media_name = os.path.join(dirname, f'{prefix}{fname}')

    return media_name


def get_random_id() -> str:
    return str(uuid.uuid4())

# TODO: Delete
def get_sorted_df(df: pd.DataFrame,
                  sort_by: Union[str, List[str]] = None,
                  sort_asc: bool = True,
                  only_highest_score: bool = False) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    sort_by : Union[str, List[str]], optional
        Column or list of columns to sort by (default is None)
    sort_asc : bool, optional
        Whether to sort ascending the elements in the dataset, by field(s) specified in
        `sort_by`. Specify a list of bools for multiple sort orders, in that case this must
        match the length of the `sort_by`.
    only_highest_score : bool, optional
        _description_, by default False

    Returns
    -------
    pd.DataFrame
        _description_
    """
    if only_highest_score:
        if sort_by is None:
            sort_by = Fields.SCORE
            sort_asc = False
        elif type(sort_by) not in (list, tuple):
            sort_by = [sort_by, Fields.SCORE]
            sort_asc = [sort_asc, False]
        else:
            sort_by = sort_by + [Fields.SCORE]
            sort_asc = sort_asc + [False]
    if sort_by is not None:
        if not is_array_like(sort_by):
            sort_by = [sort_by]
        if all(x in df.columns for x in sort_by):
            df = df.sort_values(by=sort_by, axis=0, ascending=sort_asc, inplace=False)
        else:
            logger.warning("Not all fields in sort_by exist in the dataset.")
    if only_highest_score:
        df = df.drop_duplicates(subset='item', keep='first', inplace=False)

    return df


def get_default_fields(class_) -> list:
    """Gets the fields from `class_`

    Returns
    -------
    list or set
        Fields found in `class_`
    """
    fields = [getattr(class_, x) for x in dir(class_)
              if not x.startswith('_') and x not in ('TYPES', 'NAMES')]
    return fields
