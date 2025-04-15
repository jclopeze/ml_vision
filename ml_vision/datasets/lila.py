from __future__ import annotations
import os
import pandas as pd
from shutil import move
from typing import Callable, Union, Optional

from ml_base.utils.dataset import get_cleaned_label
from ml_base.utils.misc import get_temp_folder
from ml_base.utils.misc import parallel_exec
from ml_base.utils.misc import download_file
from ml_base.utils.logger import get_logger

from ml_vision.datasets import ImageDataset
from ml_vision.utils.lila import read_lila_metadata
from ml_vision.utils.vision import VisionFields

logger = get_logger(__name__)


class LILADataset(ImageDataset):

    DEFAULTS = {
        'lila-taxonomy-mapping-url': 'https://lila.science/wp-content/uploads/2022/07/lila-taxonomy-mapping_release.csv',
        'azcopy-download-linux': 'https://aka.ms/downloadazcopy-v10-linux'
    }

    class MetadataFields(ImageDataset.MetadataFields):
        COLLECTION = 'collection'
        LICENSE = 'license'

    class AnnotationFields(ImageDataset.AnnotationFields):
        KINGDOM = 'kingdom'
        PHYLUM = 'phylum'
        CLASS = 'class'
        ORDER = 'order'
        FAMILY = 'family'
        GENUS = 'genus'
        SPECIES = 'species'
        SUBSPECIES = 'subspecies'
        VARIETY = 'variety'

        TAXA_LEVEL = 'taxonomy_level'
        SCIENTIFIC_NAME = 'scientific_name'

        _TAXA_RANKS_NAMES = [
            KINGDOM, PHYLUM, CLASS, ORDER, FAMILY, GENUS, SPECIES, SUBSPECIES, VARIETY]
        _TAXA_FIELD_NAMES = _TAXA_RANKS_NAMES + [TAXA_LEVEL, SCIENTIFIC_NAME]

        TYPES = ImageDataset.AnnotationFields.TYPES

    class Collections():
        IDAHO = 'Idaho Camera Traps'
        SNAPSHOT_CAMDEBOO = 'Snapshot Camdeboo'
        SNAPSHOT_ENONKISHU = 'Snapshot Enonkishu'
        SNAPSHOT_KAROO = 'Snapshot Karoo'
        SNAPSHOT_KGALAGADI = 'Snapshot Kgalagadi'
        SNAPSHOT_KRUGER = 'Snapshot Kruger'
        SNAPSHOT_MOUNTAIN_ZEBRA = 'Snapshot Mountain Zebra'
        ORINOQUIA = 'Orinoquia Camera Traps'
        CALTECH = 'Caltech Camera Traps'
        CALTECH_BBOX = 'Caltech Camera Traps_bbox'
        WELLINGTON = 'Wellington Camera Traps'
        MISSOURI = 'Missouri Camera Traps'
        NACTI = 'NACTI'
        NACTI_BBOX = 'NACTI_bbox'
        ENA24 = 'ENA24'
        SWG = 'SWG Camera Traps'
        SWG_BBOX = 'SWG Camera Traps_bbox'
        SNAPSHOT_SERENGETI = 'Snapshot Serengeti'
        SNAPSHOT_SERENGETI_BBOX = 'Snapshot Serengeti_bbox'
        WCS = 'WCS Camera Traps'
        WCS_BBOX = 'WCS Camera Traps_bbox'
        ISLAND_CONSERVATION = 'Island Conservation Camera Traps'
        CHANNEL_ISLANDS = 'Channel Islands Camera Traps'

        NAMES = [IDAHO, SNAPSHOT_CAMDEBOO, SNAPSHOT_ENONKISHU, SNAPSHOT_KAROO, SNAPSHOT_KGALAGADI,
                 SNAPSHOT_KRUGER, SNAPSHOT_MOUNTAIN_ZEBRA, ORINOQUIA, CALTECH, CALTECH_BBOX,
                 WELLINGTON, MISSOURI, NACTI, NACTI_BBOX, ENA24, SWG, SWG_BBOX, SNAPSHOT_SERENGETI,
                 SNAPSHOT_SERENGETI_BBOX, WCS, WCS_BBOX, ISLAND_CONSERVATION, CHANNEL_ISLANDS]

    mapping_fields = {
        Collections.IDAHO: {'datetime': 'date_captured', 'frame_num': 'seq_frame_num'},
        Collections.SNAPSHOT_CAMDEBOO: {'datetime': 'date_captured', 'frame_num': 'seq_frame_num'},
        Collections.SNAPSHOT_ENONKISHU: {'datetime': 'date_captured', 'frame_num': 'seq_frame_num'},
        Collections.SNAPSHOT_KAROO: {'datetime': 'date_captured', 'frame_num': 'seq_frame_num'},
        Collections.SNAPSHOT_KGALAGADI: {'datetime': 'date_captured', 'frame_num': 'seq_frame_num'},
        Collections.SNAPSHOT_KRUGER: {'datetime': 'date_captured', 'frame_num': 'seq_frame_num'},
        Collections.SNAPSHOT_MOUNTAIN_ZEBRA: {'datetime': 'date_captured', 'frame_num': 'seq_frame_num'},
        Collections.ORINOQUIA: {'datetime': 'date_captured', 'frame_num': 'seq_frame_num'},
        Collections.CALTECH: {'frame_num': 'seq_frame_num'},
        Collections.CALTECH_BBOX: {'frame_num': 'seq_frame_num'},
        Collections.WELLINGTON: {'datetime': 'date_captured', 'frame_num': 'seq_frame_num', 'site': 'location'},
        Collections.MISSOURI: {'frame_num': 'seq_frame_num'},
        Collections.NACTI: {},
        Collections.NACTI_BBOX: {},
        Collections.ENA24: {},
        Collections.SWG: {'datetime': 'date_captured', 'frame_num': 'seq_frame_num'},
        Collections.SWG_BBOX: {'datetime': 'date_captured', 'frame_num': 'seq_frame_num'},
        Collections.SNAPSHOT_SERENGETI: {'datetime': 'date_captured', 'frame_num': 'seq_frame_num'},
        Collections.SNAPSHOT_SERENGETI_BBOX: {'datetime': 'date_captured', 'frame_num': 'seq_frame_num'},
        Collections.WCS: {'datetime': 'date_captured', 'frame_num': 'seq_frame_num'},
        Collections.WCS_BBOX: {'datetime': 'date_captured', 'frame_num': 'seq_frame_num'},
        Collections.ISLAND_CONSERVATION: {},
        Collections.CHANNEL_ISLANDS: {'frame_num': 'seq_frame_num'}
    }

    mapping_image_id_rules = {
        Collections.IDAHO: lambda x: f"public/{x['file_name']}",
        Collections.SNAPSHOT_CAMDEBOO: lambda x: f"CDB_public/{x['file_name']}",
        Collections.SNAPSHOT_ENONKISHU: lambda x: f"ENO_public/{x['file_name']}",
        Collections.SNAPSHOT_KAROO: lambda x: f"KAR_public/{x['file_name']}",
        Collections.SNAPSHOT_KGALAGADI: lambda x: f"KGA_public/{x['file_name']}",
        Collections.SNAPSHOT_KRUGER: lambda x: f"KRU_public/{x['file_name']}",
        Collections.SNAPSHOT_MOUNTAIN_ZEBRA: lambda x: f"MTZ_public/{x['file_name']}",
        Collections.ORINOQUIA: lambda x: x['file_name'],
        Collections.CALTECH: lambda x: f"cct_images/{x['file_name']}",
        Collections.CALTECH_BBOX: lambda x: f"cct_images/{x['file_name']}",
        Collections.WELLINGTON: lambda x: x['file_name'],
        Collections.MISSOURI: lambda x: "images/" + x['file_name'].replace('\\', '/'),
        Collections.NACTI: lambda x: x['file_name'],
        Collections.NACTI_BBOX: lambda x: x['file_name'],
        Collections.ENA24: lambda x: f"images/{x['file_name']}",
        Collections.SWG: lambda x: f"{x['file_name'].split('public/')[1]}" if x['file_name'].startswith('public/') else f"{x['file_name'].split('private/')[1]}",
        Collections.SWG_BBOX: lambda x: f"{x['file_name'].split('public/')[1]}" if x['file_name'].startswith('public/') else f"{x['file_name'].split('private/')[1]}",
        Collections.SNAPSHOT_SERENGETI: lambda x: '/'.join(x['file_name'].split('/')[1:]),
        Collections.SNAPSHOT_SERENGETI_BBOX: lambda x: '/'.join(x['file_name'].split('/')[1:]),
        Collections.WCS: lambda x: f"{x['file_name'].split('/', maxsplit=1)[1]}",
        Collections.WCS_BBOX: lambda x: f"{x['file_name'].split('animals/')[1]}",
        Collections.ISLAND_CONSERVATION: lambda x: f"public/{x['file_name']}",
        Collections.CHANNEL_ISLANDS: lambda x: f"images/{x['file_name']}"
    }

    mapping_downloaded_imgs_azcopy = {
        Collections.IDAHO: lambda x: f"public/{x['item']}",
        Collections.SNAPSHOT_CAMDEBOO: lambda x: f"CDB_public/{x['item']}",
        Collections.SNAPSHOT_ENONKISHU: lambda x: f"ENO_public/{x['item']}",
        Collections.SNAPSHOT_KAROO: lambda x: f"KAR_public/{x['item']}",
        Collections.SNAPSHOT_KGALAGADI: lambda x: f"KGA_public/{x['item']}",
        Collections.SNAPSHOT_KRUGER: lambda x: f"KRU_public/{x['item']}",
        Collections.SNAPSHOT_MOUNTAIN_ZEBRA: lambda x: f"MTZ_public/{x['item']}",
        Collections.ORINOQUIA: lambda x: f"public/{x['item']}",
        Collections.CALTECH: lambda x: f"cct_images/{x['item']}",
        Collections.CALTECH_BBOX: lambda x: f"cct_images/{x['item']}",
        Collections.WELLINGTON: lambda x: f"images/{x['item']}",
        Collections.MISSOURI: lambda x: f"images/{x['item']}",
        Collections.NACTI: lambda x: f"nacti-unzipped/{x['item']}",
        Collections.NACTI_BBOX: lambda x: f"nacti-unzipped/{x['item']}",  # duda
        Collections.ENA24: lambda x: f"images/{x['item']}",
        Collections.SWG: lambda x: f"swg-camera-traps/{x['item']}",
        Collections.SWG_BBOX: lambda x: f"swg-camera-traps/{x['item']}",
        Collections.SNAPSHOT_SERENGETI: lambda x: f"snapshotserengeti-unzipped/{x['item']}",
        Collections.SNAPSHOT_SERENGETI_BBOX: lambda x: f"snapshotserengeti-unzipped/{x['item']}",
        Collections.WCS: lambda x: f"wcs-unzipped/{x['item']}",
        Collections.WCS_BBOX: lambda x: f"wcs-unzipped/{x['item']}",  # duda
        Collections.ISLAND_CONSERVATION: lambda x: f"public/{x['item']}",
        Collections.CHANNEL_ISLANDS: lambda x: f"images/{x['item']}"
    }

    mapping_license = {
        Collections.IDAHO: "no license",
        Collections.SNAPSHOT_CAMDEBOO: "Community Data License Agreement (permissive variant)",
        Collections.SNAPSHOT_ENONKISHU: "Community Data License Agreement (permissive variant)",
        Collections.SNAPSHOT_KAROO: "Community Data License Agreement (permissive variant)",
        Collections.SNAPSHOT_KGALAGADI: "Community Data License Agreement (permissive variant)",
        Collections.SNAPSHOT_KRUGER: "Community Data License Agreement (permissive variant)",
        Collections.SNAPSHOT_MOUNTAIN_ZEBRA: "Community Data License Agreement (permissive variant)",
        Collections.ORINOQUIA: "Community Data License Agreement (permissive variant)",
        Collections.CALTECH: "Community Data License Agreement (permissive variant)",
        Collections.CALTECH_BBOX: "Community Data License Agreement (permissive variant)",
        Collections.WELLINGTON: "Community Data License Agreement (permissive variant)",
        Collections.MISSOURI: "Community Data License Agreement (permissive variant)",
        Collections.NACTI: "Community Data License Agreement (permissive variant)",
        Collections.NACTI_BBOX: "Community Data License Agreement (permissive variant)",
        Collections.ENA24: "Community Data License Agreement (permissive variant)",
        Collections.SWG: "Community Data License Agreement (permissive variant)",
        Collections.SWG_BBOX: "Community Data License Agreement (permissive variant)",
        Collections.SNAPSHOT_SERENGETI: "Community Data License Agreement (permissive variant)",
        Collections.SNAPSHOT_SERENGETI_BBOX: "Community Data License Agreement (permissive variant)",
        Collections.WCS: "Community Data License Agreement (permissive variant)",
        Collections.WCS_BBOX: "Community Data License Agreement (permissive variant)",
        Collections.ISLAND_CONSERVATION: "Community Data License Agreement (permissive variant)",
        Collections.CHANNEL_ISLANDS: "Community Data License Agreement (permissive variant)"
    }

    @classmethod
    def _download(cls,
                  dest_path,
                  metadata,
                  **kwargs):
        metadata_dir = kwargs.get('metadata_dir')

        metadata_dir = metadata_dir or os.path.join(get_temp_folder(), 'metadata_dir')
        metadata_table = read_lila_metadata(metadata_dir)

        os.makedirs(dest_path, exist_ok=True)

        logger.info(f"Downloading {len(metadata)} images...")

        collections = metadata[cls.MetadataFields.COLLECTION].unique()

        for collection in collections:
            coll_df = metadata[metadata[cls.MetadataFields.COLLECTION] == collection]
            filenames = coll_df[cls.MetadataFields.ITEM].values
            output_imgs_dir = os.path.join(dest_path, collection)
            sas_url = metadata_table[collection]['sas_url']

            def get_dest_filename(fname):
                filename = cls.mapping_downloaded_imgs_azcopy[collection]({'item': fname})
                return os.path.join(output_imgs_dir, filename)
            parallel_exec(
                func=download_file,
                elements=filenames,
                url=lambda fname: f'{sas_url}/{fname}',
                dest_filename=get_dest_filename,
                verbose=False)

    def download(self, dest_path, **kwargs):
        num_tasks = kwargs.get('num_tasks')
        task_num = kwargs.get('task_num')

        if num_tasks is not None and task_num is not None:
            split_dataset = self.get_chunk(num_tasks, task_num)
            metadata = split_dataset.metadata
            split_dataset._download(dest_path, metadata=metadata, **kwargs)
        else:
            metadata = self.metadata
            self._download(dest_path, metadata=metadata, **kwargs)

    def set_items_after_downloading(self):
        fn = self.mapping_downloaded_imgs_azcopy
        self[self.AnnotationFields.ITEM] = lambda rec: fn[rec[self.MetadataFields.COLLECTION]](rec)
        self[self.AnnotationFields.ITEM] = lambda rec: os.path.join(
            rec[self.MetadataFields.COLLECTION], rec[self.AnnotationFields.ITEM])

    @classmethod
    def from_json(cls,
                  source_path: str = None,
                  **kwargs) -> LILADataset:
        """Create a LILADataset from a JSON file in COCO format.
        This method allows you to create a dataset by simply providing the name of a LILA
        collection or the local `source_path` of the JSON.

        Parameters
        ----------
        source_path : str, optional
            Path of a json file that will be converted into a LILADataset.
            If None, it will be downloaded by using the `collection` name.
            By default None
        **kwargs :
            Extra named arguments that may contains the following parameters:
            * collection : str
                A valid collection name of a LILA dataset
            * map_to_scientific_names : bool
                Whether to map or not the common names of species given by LILA to scientific
                names. The mapping will be performed using the taxonomy mapping released by LILA,
                and the URL used to download that CSV file could be set in the env variable
                `LILA_TAXONOMY_MAPPING_URL`.
                By default True
            * exclude_invalid_scientific_names : bool
                Whether to delete or not the elements of the dataset with no valid scientific names
                (e.g. vehicle, empty, unknown, etc).
                By default True
            * mapping_classes_csv : str
                The path to the CSV taxonomy mapping released by LILA, in case you've aready
                downloaded it or if you want to store it in this specific path

        Returns
        -------
        LILADataset
            The instance of the created dataset
        """
        collection = kwargs.get('collection')
        map_to_scientific_names = kwargs.get('map_to_scientific_names', True)
        exclude_invalid_scinames = kwargs.get('exclude_invalid_scientific_names', True)
        map_image_id_field = kwargs.get('map_image_id_field', True)
        taxonomy_level = kwargs.get('taxonomy_level', None)

        if source_path is None:
            assert collection is not None, "You must specify the collection name"
            all_json_files_dict = lila_utils.get_all_json_files(
                get_temp_folder(), collections=[collection])
            source_path = all_json_files_dict[collection]

        if collection is not None and 'mapping_fields' not in kwargs:
            kwargs['mapping_fields'] = LILADataset.mapping_fields[collection]

        category_mapping = None
        if map_to_scientific_names:
            mapping_classes_csv = kwargs.get('mapping_classes_csv')
            if mapping_classes_csv is None or not os.path.isfile(mapping_classes_csv):
                mapping_url = os.environ.get(
                    'LILA_TAXONOMY_MAPPING_URL', LILADataset.DEFAULTS['lila-taxonomy-mapping-url'])
                file_path = download_file(mapping_url, get_temp_folder())
                if mapping_classes_csv is None:
                    mapping_classes_csv = file_path
                else:
                    move(file_path, mapping_classes_csv)
            coll_name = (
                collection.split('_bbox')[0] if collection.endswith('_bbox') else collection)

            category_mapping = get_mapping_classes_from_csv(
                mapping_classes_csv=mapping_classes_csv,
                from_col='query',
                to_col='scientific_name',
                filter_expr=lambda rec: rec['dataset_name'] == coll_name)

        instance = super().from_json(
            source_path=source_path, category_mapping=category_mapping, **kwargs)

        if map_to_scientific_names and exclude_invalid_scinames:
            instance.filter_by_categories('', mode='exclude', inplace=True)

        if map_image_id_field:
            if collection in LILADataset.mapping_image_id_rules:
                mapping_img_id_rule = LILADataset.mapping_image_id_rules[collection]
                instance[VisionFields.FILE_ID] = mapping_img_id_rule

        instance[LILADataset.MetadataFields.COLLECTION] = collection

        if map_to_scientific_names:
            taxa_fields = cls.AnnotationFields._TAXA_FIELD_NAMES
            _coll = collection.replace('_bbox', '') if collection.endswith('_bbox') else collection

            tax = pd.read_csv(mapping_classes_csv).fillna('')
            taxa = (
                tax[(~tax['scientific_name'].isna()) & (tax['dataset_name'] == _coll)][taxa_fields]
                .drop_duplicates(subset='scientific_name')
            )
            _data = instance.df
            _data = pd.merge(left=_data, right=taxa, left_on='label', right_on='scientific_name')
            _data = _data.drop(['scientific_name'], axis=1)
            # TODO: Check this, maybe instance._set_annotations(_data)
            instance.annotations = _data

        if taxonomy_level is not None:
            instance.mapping_to_taxonomy_level(taxonomy_level)

        _license = LILADataset.mapping_license[collection]
        instance[LILADataset.MetadataFields.LICENSE] = _license

        return instance

    def mapping_to_taxonomy_level(self, taxonomy_level_to):
        assert_cond = taxonomy_level_to in self.AnnotationFields._TAXA_RANKS_NAMES
        assert assert_cond, "Invalid taxonomy_level_to"

        self.filter_by_field(taxonomy_level_to, '', mode='exclude', inplace=True)
        self['label'] = lambda rec: rec[taxonomy_level_to]

    @classmethod
    def from_csv(cls, source_path: str, **kwargs) -> LILADataset:
        taxonomy_level = kwargs.get('taxonomy_level', None)
        taxons = kwargs.get('taxons', None)

        root_dir_prov = None
        if 'root_dir' in kwargs:
            root_dir_prov = kwargs.get('root_dir')
            if 'root_dir' in kwargs:
                del kwargs['root_dir']

        instance = ImageDataset.from_csv(
            source_path=source_path, root_dir=None, validate_filenames=False, **kwargs)

        if taxonomy_level is not None:
            instance.mapping_to_taxonomy_level(taxonomy_level)

        if taxons is not None:
            instance.filter_by_categories(taxons, inplace=True)

        if root_dir_prov is not None:
            instance.set_items_after_downloading()
            instance.set_root_dir(root_dir_prov, not_exist_ok=True)

        return instance


def get_mapping_classes_from_csv(mapping_classes_csv: str,
                                 from_col: Union[str, int] = None,
                                 to_col: Union[str, int] = None,
                                 filter_expr: Callable = None) -> dict:
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
