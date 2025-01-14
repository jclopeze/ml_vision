import os
import math
import pandas as pd
from shutil import move
import subprocess
from typing import Union, List
from urllib.parse import urlparse
import uuid
import zipfile

from ml_base.utils.misc import unzip_file, download_file, is_array_like, get_temp_folder
# from ml_base.utils.misc import untar_file


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


def get_all_md_result_jsons(csv_path: str, metadata_dir: str, collections: Union[List, str] = None,
                            md_version: str = 'MDv5a', delete_zip_files: bool = False) -> dict:
    """Downloads if necessary - then unzips if necessary - the .json files for all the Megadetector
    results of LILA datasets.

    Parameters
    ----------
    csv_path : str
        Path to a CSV file containing to URLs to download the Megadetector results for the LILA
        collections. It must contain a column named 'Collection' (i.e. the collection name) and one
        for the Megadetector version named in the `md_version` parameter (e.g. MDv5).
    metadata_dir : str
        Base directory where all the metadata will be stored
    collections : Union[List, str], optional
        Collection(s) to be returned. If None, all the collections will be returned.
        By default None
    md_version : str, optional
        Version of the Megadetector. The CSV in `csv_path` must contain a column with this name,
        by default 'MDv5a'

    Returns
    -------
    dict
        Dictionary with the JSON path on the local disk for each collection
    """
    df = pd.read_csv(csv_path)
    assert md_version in df.columns, f'Invalid md_version: {md_version}'
    all_col_to_json = (
        df[['Collection', md_version]]
        .set_index('Collection')
        .squeeze('columns').to_dict()
    )
    if collections is not None:
        if not is_array_like(collections):
            collections = [collections]
        assert_cond = all([x in all_col_to_json.keys() for x in collections])
        assert assert_cond, f"Invalid collections: {collections}"
        col_to_json = {col: all_col_to_json[col] for col in collections}
    else:
        col_to_json = all_col_to_json

    result_jsons = {
        ds_name: get_md_dets_json_file_for_dataset(json_path, metadata_dir, delete_zip_files)
        for ds_name, json_path in col_to_json.items()
    }
    return result_jsons


def get_md_dets_json_file_for_dataset(json_url: str, metadata_dir: str,
                                      delete_zip_files: bool = False, verbose: bool = False) -> str:
    """Downloads if necessary - then unzips if necessary - the .json file for a specific
    Megadetector results of a LILA dataset.
    Returns the .json filename on the local disk.

    Parameters
    ----------
    json_url : str
        Remote URL of Megadetector results of a LILA dataset
    metadata_dir : str
        Base directory where the metadata of the Megadetector results will be stored
    delete_zip_files : bool, optional
        Whether to delete the zip files after uncompress it or not, by default True

    Returns
    -------
    str
        JSON path of Megadetector results for a LILA dataset
    """
    p = urlparse(json_url)
    json_filename = os.path.join(metadata_dir, os.path.basename(p.path))
    download_file(json_url, json_filename)

    # Unzip if necessary
    if json_filename.endswith('.zip'):

        with zipfile.ZipFile(json_filename, 'r') as z:
            files = z.namelist()
        if len(files) == 1:
            unzipped_json_filename = os.path.join(metadata_dir, files[0])
            if not os.path.isfile(unzipped_json_filename):
                unzip_file(json_filename, metadata_dir)
                if delete_zip_files:
                    os.remove(json_filename)
            elif verbose:
                print('{} already unzipped'.format(unzipped_json_filename))
            return unzipped_json_filename
        else:
            unzip_file(json_filename, metadata_dir)
            unzipped_json_filenames = [os.path.join(metadata_dir, f) for f in files]
            return unzipped_json_filenames

    return json_filename


# def get_azcopy_exec(azcopy_exec, azcopy_url):
#     if azcopy_exec is None or not os.path.isfile(azcopy_exec):
#         temp_folder = get_temp_folder()
#         downld_path = os.path.join(temp_folder, 'azcopy.tar.gz')
#         azcp_path = download_file(azcopy_url, downld_path)
#         untar_folder = os.path.join(temp_folder, 'azcopy_untar')
#         untar_file(azcp_path, untar_folder)
#         _azcopy_exec = os.path.join(untar_folder, os.path.basename(azcopy_exec))
#         if not os.path.isfile(_azcopy_exec):
#             raise Exception("No azcopy executable file was found")

#         if azcopy_exec is None:
#             azcopy_exec = _azcopy_exec
#         else:
#             os.makedirs(os.path.dirname(azcopy_exec), exist_ok=True)
#             move(_azcopy_exec, azcopy_exec)

def download_with_azcopy(filenames,
                         azcopy_exec,
                         batch_size,
                         aux_files_folder,
                         output_imgs_dir,
                         sas_url):
    """Download images with the command-line utility Azcopy

    Parameters
    ----------
    filenames : list of str
        List of the filenames to download
    azcopy_exec : str
        Path of the Azcopy executable
    batch_size : int
        Number of images to be downloaded simultaneously in each request, through auxiliary files
    aux_files_folder : str
        Folder in which auxiliary files will be stored
    output_imgs_dir : str
        Path of the folder in which downloaded images will be stored
    sas_url : str
        SAS (Shared Access Signature) URL of the source collection (E.g., given by LILA BC datasets
        in http://lila.science/wp-content/uploads/2020/03/lila_sas_urls.txt)
    """
    num_aux_files = math.ceil(len(filenames) / batch_size)
    os.makedirs(aux_files_folder, exist_ok=True)
    os.makedirs(output_imgs_dir, exist_ok=True)

    for i in range(num_aux_files):
        az_filename = os.path.join(aux_files_folder, f'{str(uuid.uuid4())}.txt')
        with open(az_filename, 'w') as f:
            for filename in filenames[batch_size*i: batch_size*(i+1)]:
                f.write(filename.replace('\\', '/') + '\n')
        cmd = [
            azcopy_exec, "cp", sas_url, output_imgs_dir,
            "--list-of-files", az_filename,
            "--overwrite=false",
            "--log-level=ERROR",
            "--check-length=false",
            "--check-md5=NoCheck"
        ]
        _ = subprocess.call(cmd)

        os.remove(az_filename)
