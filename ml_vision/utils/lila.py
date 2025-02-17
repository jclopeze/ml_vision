import os
import pandas as pd
from typing import Union, List
from urllib.parse import urlparse
import zipfile

from ml_base.utils.misc import unzip_file, download_file, is_array_like, get_temp_folder


lila_metadata_url = 'http://lila.science/wp-content/uploads/2020/03/lila_sas_urls.txt'
max_path_len = 255


def read_lila_metadata(metadata_dir: str, verbose: bool = False) -> dict:
    """Reads LILA metadata (URLs to each dataset), downloading the txt file if necessary.

    Returns

    Parameters
    ----------
    metadata_dir : str
        Base directory where all the metadata will be stored
    verbose : bool, optional
        Whether to be verbose or not, by default False

    Returns
    -------
    dict
        A dict mapping dataset names (e.g. "Caltech Camera Traps") to dicts
        with keys "sas_url" (pointing to the image base) and "json_url" (pointing to the metadata
        file).
    """
    # Put the master metadata file in the same folder where we're putting images
    p = urlparse(lila_metadata_url)
    metadata_filename = os.path.join(metadata_dir, os.path.basename(p.path))
    download_file(lila_metadata_url, metadata_filename)

    # Read lines from the master metadata file
    with open(metadata_filename, 'r') as f:
        metadata_lines = f.readlines()
    metadata_lines = [s.strip() for s in metadata_lines]

    # Parse those lines into a table
    metadata_table = {}

    for s in metadata_lines:

        if len(s) == 0 or s[0] == '#':
            continue

        # Each line in this file is name/sas_url/json_url/[bbox_json_url]
        tokens = s.split(',')
        assert len(tokens) == 4
        ds_name = tokens[0].strip()
        url_mapping = {'sas_url': tokens[1], 'json_url': tokens[2]}
        metadata_table[ds_name] = url_mapping

        # Create a separate entry for bounding boxes if they exist
        if len(tokens[3].strip()) > 0:
            if verbose:
                print('Adding bounding box dataset for {}'.format(ds_name))
            bbox_url_mapping = {'sas_url': tokens[1], 'json_url': tokens[3]}
            metadata_table[tokens[0]+'_bbox'] = bbox_url_mapping
            assert 'https' in bbox_url_mapping['json_url']

        assert 'https' not in tokens[0]
        assert 'https' in url_mapping['sas_url']
        assert 'https' in url_mapping['json_url']

    return metadata_table


def get_all_json_files(metadata_dir: str, collections: Union[List, str] = None,
                       delete_zip_files: bool = False) -> dict:
    """Downloads if necessary - then unzips if necessary - the .json files for all LILA datasets.

    Parameters
    ----------
    metadata_dir : str
        Base directory where the metadata of the datasets will be stored
    collections : Union[List, str], optional
        Collection(s) to be returned. If None, all the collections will be returned.
        By default None

    Returns
    -------
    dict
        Dictionary with the JSON path on the local disk for each collection
    """
    metadata_table = read_lila_metadata(metadata_dir)
    all_ds_names = list(metadata_table.keys())

    if collections is not None:
        if not is_array_like(collections):
            collections = [collections]
        assert all([x in all_ds_names for x in collections]), f"Invalid collections: {collections}"
        ds_names = collections
    else:
        ds_names = all_ds_names

    all_json_filenames = {}
    for ds_name in ds_names:
        all_json_filenames[ds_name] = get_json_file_for_dataset(
            ds_name=ds_name,
            metadata_dir=metadata_dir,
            metadata_table=metadata_table,
            delete_zip_files=delete_zip_files)

    return all_json_filenames


def get_json_file_for_dataset(ds_name: str, metadata_dir: str, metadata_table: dict = None,
                              delete_zip_files: bool = False, verbose: bool = False) -> str:
    """Downloads if necessary - then unzips if necessary - the .json file for a specific LILA
    dataset. Returns the .json filename on the local disk.

    Parameters
    ----------
    ds_name : str
        Dataset name
    metadata_dir : str
        Base directory where the metadata of the dataset will be stored
    metadata_table : dict, optional
        A dict mapping dataset names (e.g. "Caltech Camera Traps") to dicts with keys "sas_url"
        (pointing to the image base) and "json_url" (pointing to the metadata file).
        By default None
    delete_zip_files : bool, optional
        Whether to delete the zip files after uncompress it or not, by default True

    Returns
    -------
    str
        JSON path for the collection `ds_name`
    """
    if metadata_table is None:
        metadata_table = read_lila_metadata(metadata_dir)

    json_url = metadata_table[ds_name]['json_url']

    p = urlparse(json_url)
    json_filename = os.path.join(metadata_dir, os.path.basename(p.path))
    download_file(json_url, json_filename)

    # Unzip if necessary
    if json_filename.endswith('.zip'):

        with zipfile.ZipFile(json_filename, 'r') as z:
            files = z.namelist()
        assert len(files) == 1
        unzipped_json_filename = os.path.join(metadata_dir, files[0])
        if not os.path.isfile(unzipped_json_filename):
            unzip_file(json_filename, metadata_dir)
            if delete_zip_files:
                os.remove(json_filename)
        elif verbose:
            print('{} already unzipped'.format(unzipped_json_filename))
        json_filename = unzipped_json_filename

    return json_filename

