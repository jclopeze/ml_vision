from __future__ import annotations
from multiprocessing import Manager
import os
import numpy as np
import pandas as pd

from ml_base.utils.dataset import get_cats_from_source, get_cleaned_label, get_random_id
from ml_base.utils.logger import get_logger
from ml_base.utils.misc import parallel_exec, download_file, get_temp_folder

from ml_vision.utils.inat import read_csv_in_burst, get_corrupted_images
from ml_vision.datasets.image import ImageDataset

logger = get_logger(__name__)


class INatMetadata():

    BURST_SIZE = int(os.environ.get('BURST_SIZE', 10000000))
    s3_url_str = 's3://inaturalist-open-data/photos/{photo_id}/{images_size}.{ext}'
    inat_url_str = (
        'https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/{images_size}.{ext}')

    class Taxa():
        TAXON_ID = 'taxon_id'
        ANCESTRY = 'ancestry'
        RANK_LEVEL = 'rank_level'
        RANK = 'rank'
        NAME = 'name'
        ACTIVE = 'active'

        # extra
        SPECIES = 'species'
        TAXON_NAME = 'taxon_name'

        COL_TYPES = {TAXON_ID: str, ANCESTRY: str,
                     RANK_LEVEL: str, RANK: str, NAME: str, ACTIVE: bool}
        COL_NAMES = COL_TYPES.keys()

        class Ranks:
            KINGDOM = 'kingdom'
            PHYLUM = 'phylum'
            CLASS = 'class'
            ORDER = 'order'
            FAMILY = 'family'
            GENUS = 'genus'
            SPECIES = 'species'
            SUBSPECIES = 'subspecies'
            VARIETY = 'variety'

            NAMES = [KINGDOM, PHYLUM, CLASS, ORDER, FAMILY, GENUS, SPECIES, SUBSPECIES, VARIETY]

        @classmethod
        def get_col_types(cls, fields):
            return {fld: cls.COL_TYPES[fld] for fld in fields}

    class Observation():
        OBSERVATION_UUID = 'observation_uuid'
        OBSERVER_ID = 'observer_id'
        LATITUDE = 'latitude'
        LONGITUDE = 'longitude'
        POS_ACCURACY = 'positional_accuracy'
        TAXON_ID = 'taxon_id'
        QUAL_GRADE = 'quality_grade'
        OBSERVED_ON = 'observed_on'

        COL_TYPES = {OBSERVATION_UUID: str, OBSERVER_ID: int, LATITUDE: float, LONGITUDE: float,
                     POS_ACCURACY: str, TAXON_ID: str, QUAL_GRADE: str, OBSERVED_ON: str}
        COL_NAMES = COL_TYPES.keys()

        @classmethod
        def get_col_types(cls, fields):
            return {fld: cls.COL_TYPES[fld] for fld in fields}

    class Photo():
        PHOTO_UUID = 'photo_uuid'
        PHOTO_ID = 'photo_id'
        OBSERVATION_UUID = 'observation_uuid'
        OBSERVER_ID = 'observer_id'
        EXTENSION = 'extension'
        LICENSE = 'license'
        WIDTH = 'width'
        HEIGHT = 'height'
        POSITION = 'position'

        # extra
        USER_ID = 'user_id'
        INAT_URL = 'inat_url'
        S3_URL = 's3_url'

        COL_TYPES = {PHOTO_UUID: str, PHOTO_ID: str, OBSERVATION_UUID: str,
                     OBSERVER_ID: str, EXTENSION: str, LICENSE: str, WIDTH: 'Int64',
                     HEIGHT: 'Int64'}  # , POSITION: 'Int64'}
        COL_NAMES = COL_TYPES.keys()

        @classmethod
        def get_col_types(cls, fields):
            return {fld: cls.COL_TYPES[fld] for fld in fields}

    photo_flds = [Photo.PHOTO_UUID, Photo.PHOTO_ID, Photo.OBSERVATION_UUID, Photo.LICENSE,
                  Photo.S3_URL, Photo.INAT_URL, Photo.WIDTH, Photo.HEIGHT, Photo.USER_ID]
    obs_flds = [Observation.QUAL_GRADE, Observation.OBSERVED_ON, Observation.LATITUDE,
                Observation.LONGITUDE, Observation.POS_ACCURACY]
    taxa_flds = [Taxa.TAXON_NAME, Taxa.TAXON_ID, Taxa.RANK] + Taxa.Ranks.NAMES
    all_flds = photo_flds + obs_flds + taxa_flds

    def __init__(self, metadata):
        self.metadata = metadata

    @classmethod
    def from_metadata_csv_files(cls,
                                source_dir,
                                taxons,
                                taxonomy_level,
                                metadata_aux_csv_files_dir=None,
                                images_size='original',
                                obs_quality_grades=[],
                                delete_aux_files_on_finish=True,
                                **kwargs):
        use_aux_mammals_and_birds_file = kwargs.get('use_aux_mammals_and_birds_file', True)
        valid_imgs_sizes = ('original', 'large', 'medium', 'small', 'thumb', 'square')
        assert_cond = images_size in valid_imgs_sizes
        assert assert_cond, f"Invalid images_size. It must be in {valid_imgs_sizes}"

        burst_size = cls.BURST_SIZE
        if type(obs_quality_grades) is str:
            obs_quality_grades = [x.strip() for x in obs_quality_grades.split(',')]
        if metadata_aux_csv_files_dir is None:
            metadata_aux_csv_files_dir = os.path.join(get_temp_folder(), f'{get_random_id()}')

        obs_csv = os.path.join(source_dir, 'observations.csv')
        photos_csv = os.path.join(source_dir, 'photos.csv')
        taxa_csv = os.path.join(source_dir, 'taxa.csv')
        mammls_birds_csv = os.path.join(source_dir, 'mammls_birds.csv')
        taxa_aux_csv = os.path.join(metadata_aux_csv_files_dir, 'taxa_aux.csv')
        obs_aux_csv = os.path.join(metadata_aux_csv_files_dir, 'observations_aux.csv')
        photos_aux_csv = os.path.join(metadata_aux_csv_files_dir, 'photos_aux.csv')

        if not os.path.isfile(taxa_aux_csv):
            logger.info("Building the taxa information CSV file...")

            if not os.path.isfile(mammls_birds_csv) or not use_aux_mammals_and_birds_file:
                taxa_df = pd.read_csv(taxa_csv, delimiter='\t', dtype=cls.Taxa.COL_TYPES)
                taxon_id_to_name_and_rank = (
                    taxa_df[['taxon_id', 'name', 'rank']]
                    .set_index('taxon_id')
                    .to_dict('index'))
                taxa_df[cls.Taxa.Ranks.KINGDOM] = taxa_df.apply(
                    INatTaxonomyMapper.get_taxa_name, axis=1, ancestry_pos=1,
                    taxonomy_level=cls.Taxa.Ranks.KINGDOM,
                    taxon_id_to_name_and_rank=taxon_id_to_name_and_rank)
                taxa_df[cls.Taxa.Ranks.PHYLUM] = taxa_df.apply(
                    INatTaxonomyMapper.get_taxa_name, axis=1, ancestry_pos=1,
                    taxonomy_level=cls.Taxa.Ranks.PHYLUM,
                    taxon_id_to_name_and_rank=taxon_id_to_name_and_rank)
                taxa_df[cls.Taxa.Ranks.CLASS] = taxa_df.apply(
                    INatTaxonomyMapper.get_taxa_name, axis=1, ancestry_pos=4,
                    taxonomy_level=cls.Taxa.Ranks.CLASS,
                    taxon_id_to_name_and_rank=taxon_id_to_name_and_rank)
                mammals_birds = (
                    taxa_df[taxa_df[cls.Taxa.Ranks.CLASS].isin(['mammalia', 'aves'])]
                )
                mammals_birds['ancestry_dict'] = mammals_birds.apply(
                    INatTaxonomyMapper.get_ancestry_dict, axis=1,
                    taxon_id_to_name_and_rank=taxon_id_to_name_and_rank
                )
                mammals_birds[cls.Taxa.Ranks.ORDER] = mammals_birds.apply(
                    lambda rec: rec['ancestry_dict'].get(cls.Taxa.Ranks.ORDER, ''), axis=1)
                mammals_birds[cls.Taxa.Ranks.FAMILY] = mammals_birds.apply(
                    lambda rec: rec['ancestry_dict'].get(cls.Taxa.Ranks.FAMILY, ''), axis=1)
                mammals_birds[cls.Taxa.Ranks.GENUS] = mammals_birds.apply(
                    lambda rec: rec['ancestry_dict'].get(cls.Taxa.Ranks.GENUS, ''), axis=1)
                mammals_birds[cls.Taxa.Ranks.SPECIES] = mammals_birds.apply(
                    lambda rec: rec['ancestry_dict'].get(cls.Taxa.Ranks.SPECIES, ''), axis=1)
                mammals_birds[cls.Taxa.Ranks.SUBSPECIES] = mammals_birds.apply(
                    lambda rec: rec['ancestry_dict'].get(cls.Taxa.Ranks.SUBSPECIES, ''),
                    axis=1)
                mammals_birds[cls.Taxa.Ranks.VARIETY] = mammals_birds.apply(
                    lambda rec: rec['ancestry_dict'].get(cls.Taxa.Ranks.VARIETY, ''), axis=1)
                mammals_birds.drop('ancestry_dict', axis=1, inplace=True)
                mammals_birds.to_csv(mammls_birds_csv, index=False)
            else:
                mammals_birds = (
                    pd.read_csv(mammls_birds_csv, dtype=cls.Taxa.get_col_types(cls.Taxa.COL_NAMES))
                    .fillna(''))
            taxa_df = mammals_birds

            taxons = get_cats_from_source(taxons, clean_names=True)
            taxa_df = taxa_df[taxa_df[taxonomy_level].apply(get_cleaned_label).isin(taxons)]

            taxa_df.to_csv(taxa_aux_csv, index=False)
        else:
            taxa_df = pd.read_csv(taxa_aux_csv, dtype=cls.Taxa.get_col_types(cls.Taxa.COL_NAMES))

        logger.info(f"{len(taxa_df)} registers found for taxa information")
        taxa_ids = taxa_df[cls.Taxa.TAXON_ID].values
        if not os.path.isfile(obs_aux_csv):
            logger.info("Building the observation information CSV file...")
            read_csv_in_burst(
                origin_csv=obs_csv,
                dest_csv=obs_aux_csv,
                cols_names=cls.Observation.COL_NAMES,
                filter_col=cls.Observation.TAXON_ID,
                filter_vals=taxa_ids,
                dtypes=cls.Observation.COL_TYPES,
                cols_to_write=cls.Observation.COL_NAMES,
                burst_size=burst_size,
                delimiter='\t')

        obs_df = pd.read_csv(
            obs_aux_csv, dtype=cls.Observation.get_col_types(cls.Observation.COL_NAMES))
        logger.info(f"{len(obs_df)} registers found for observation information")
        obs_ids = obs_df[cls.Observation.OBSERVATION_UUID].values
        if not os.path.isfile(photos_aux_csv):
            logger.info("Building the photos information CSV file...")
            read_csv_in_burst(
                origin_csv=photos_csv,
                dest_csv=photos_aux_csv,
                cols_names=cls.Photo.COL_NAMES,
                filter_col=cls.Photo.OBSERVATION_UUID,
                filter_vals=obs_ids,
                dtypes=cls.Photo.COL_TYPES,
                cols_to_write=cls.Photo.COL_NAMES,
                usecols=cls.Photo.COL_NAMES,
                burst_size=burst_size,
                delimiter='\t')

        photos_df = pd.read_csv(photos_aux_csv, dtype=cls.Photo.get_col_types(cls.Photo.COL_NAMES))
        photos_df = photos_df.rename(columns={cls.Photo.OBSERVER_ID: cls.Photo.USER_ID})
        logger.info(f"{len(photos_df)} registers found for photos information")

        meta_df = pd.merge(left=photos_df, right=obs_df,
                           on=cls.Observation.OBSERVATION_UUID, how='left')
        meta_df = pd.merge(left=meta_df, right=taxa_df, on=cls.Taxa.TAXON_ID, how='left')

        meta_df = meta_df.rename(columns={cls.Taxa.NAME: cls.Taxa.TAXON_NAME})
        meta_df[cls.Photo.S3_URL] = meta_df.apply(
            lambda rec: cls.s3_url_str.format(photo_id=rec[cls.Photo.PHOTO_ID],
                                              images_size=images_size,
                                              ext=rec[cls.Photo.EXTENSION]), axis=1)
        meta_df[cls.Photo.INAT_URL] = meta_df.apply(
            lambda rec: cls.inat_url_str.format(photo_id=rec[cls.Photo.PHOTO_ID],
                                                images_size=images_size,
                                                ext=rec[cls.Photo.EXTENSION]), axis=1)

        metadata_flds = [fld for fld in cls.all_flds if fld in meta_df.columns]
        meta_df = meta_df[metadata_flds]

        if obs_quality_grades:
            meta_df = meta_df[meta_df[cls.Observation.QUAL_GRADE].isin(obs_quality_grades)]
        if delete_aux_files_on_finish:
            os.remove(taxa_aux_csv)
            os.remove(obs_aux_csv)
            os.remove(photos_aux_csv)

        instance = cls(meta_df)

        return instance

    def to_csv(self, dest_path):
        self.metadata.to_csv(dest_path, index=False)

    @classmethod
    def from_csv(cls, source_path):
        meta_df = pd.read_csv(source_path).replace(np.nan, '')
        instance = cls(meta_df)

        return instance


class INatDataset(ImageDataset):

    class METADATA_FIELDS(ImageDataset.METADATA_FIELDS):
        POS_ACCURACY = INatMetadata.Observation.POS_ACCURACY
        LATITUDE = INatMetadata.Observation.LATITUDE
        LONGITUDE = INatMetadata.Observation.LONGITUDE
        USER_ID = INatMetadata.Photo.USER_ID
        LICENSE = INatMetadata.Photo.LICENSE

    class ANNOTATIONS_FIELDS(ImageDataset.ANNOTATIONS_FIELDS):
        QUAL_GRADE = INatMetadata.Observation.QUAL_GRADE

        KINGDOM = INatMetadata.Taxa.Ranks.KINGDOM
        PHYLUM = INatMetadata.Taxa.Ranks.PHYLUM
        CLASS = INatMetadata.Taxa.Ranks.CLASS
        ORDER = INatMetadata.Taxa.Ranks.ORDER
        FAMILY = INatMetadata.Taxa.Ranks.FAMILY
        GENUS = INatMetadata.Taxa.Ranks.GENUS
        SPECIES = INatMetadata.Taxa.Ranks.SPECIES
        SUBSPECIES = INatMetadata.Taxa.Ranks.SUBSPECIES
        VARIETY = INatMetadata.Taxa.Ranks.VARIETY

        TAXA_LEVEL = 'taxonomy_level'

        _TAXA_RANKS_NAMES = [
            KINGDOM, PHYLUM, CLASS, ORDER, FAMILY, GENUS, SPECIES, SUBSPECIES, VARIETY]

        TYPES = ImageDataset.ANNOTATIONS_FIELDS.TYPES

    @classmethod
    def from_metadata(cls, metadata: INatMetadata, **kwargs) -> INatDataset:
        taxonomy_level = kwargs.get('taxonomy_level', None)
        remove_gifs = kwargs.get('remove_gifs', True)

        photo_id_fld = INatMetadata.Photo.PHOTO_ID
        inat_url_fld = INatMetadata.Photo.INAT_URL

        df = metadata.metadata.apply(
            lambda x: x.fillna(np.nan) if x.dtype.kind in 'biufc' else x.fillna(''))
        if remove_gifs:
            df = df[~df[inat_url_fld].str.lower().str.endswith('.gif')].reset_index()

        logger.info(f"Creating dataset from metadata instance with {len(df)} records")

        data = pd.DataFrame()

        data[cls.ANNOTATIONS_FIELDS.ITEM] = df.apply(
            lambda rec: f"{rec[photo_id_fld]}{os.path.splitext(rec[inat_url_fld])[1]}", axis=1)
        obs_uuid_fld = INatMetadata.Observation.OBSERVATION_UUID
        photo_id_fld = INatMetadata.Photo.PHOTO_ID
        data[cls.ANNOTATIONS_FIELDS.ID] = (
            df.apply(lambda rec: f"{rec[obs_uuid_fld]}-{rec[photo_id_fld]}", axis=1))
        data[cls.ANNOTATIONS_FIELDS.MEDIA_ID] = df[INatMetadata.Photo.PHOTO_ID].copy()
        data[cls.METADATA_FIELDS.FILENAME] = data[cls.ANNOTATIONS_FIELDS.ITEM].copy()
        data[cls.METADATA_FIELDS.SEQ_ID] = df[INatMetadata.Observation.OBSERVATION_UUID].copy()

        if taxonomy_level is not None:
            assert_msg = f"Taxon rank {taxonomy_level} not present in Metadata"
            assert taxonomy_level in df.columns, assert_msg
            data[cls.ANNOTATIONS_FIELDS.LABEL] = df[taxonomy_level]
        else:
            data[cls.ANNOTATIONS_FIELDS.LABEL] = df[INatMetadata.Taxa.TAXON_NAME]
        if INatMetadata.Taxa.RANK in df.columns:
            data[cls.ANNOTATIONS_FIELDS.TAXA_LEVEL] = df[INatMetadata.Taxa.RANK].copy()
        if INatMetadata.Observation.OBSERVED_ON in df.columns:
            data[cls.METADATA_FIELDS.DATE_CAPTURED] = df[INatMetadata.Observation.OBSERVED_ON].copy()
        if INatMetadata.Observation.LATITUDE in df.columns:
            data[cls.METADATA_FIELDS.LATITUDE] = df[INatMetadata.Observation.LATITUDE].copy()
        if INatMetadata.Observation.LONGITUDE in df.columns:
            data[cls.METADATA_FIELDS.LONGITUDE] = df[INatMetadata.Observation.LONGITUDE].copy()
        if INatMetadata.Observation.QUAL_GRADE in df.columns:
            data[cls.ANNOTATIONS_FIELDS.QUAL_GRADE] = (
                df[INatMetadata.Observation.QUAL_GRADE].copy())
        if INatMetadata.Photo.WIDTH in df.columns:
            data[cls.METADATA_FIELDS.WIDTH] = df[INatMetadata.Photo.WIDTH].copy()
        if INatMetadata.Photo.HEIGHT in df.columns:
            data[cls.METADATA_FIELDS.HEIGHT] = df[INatMetadata.Photo.HEIGHT].copy()
        # license
        if INatMetadata.Photo.LICENSE in df.columns:
            data[cls.METADATA_FIELDS.LICENSE] = df[INatMetadata.Photo.LICENSE].copy()
        if INatMetadata.Observation.POS_ACCURACY in df.columns:
            data[cls.METADATA_FIELDS.POS_ACCURACY] = df[INatMetadata.Observation.POS_ACCURACY].copy()
        if INatMetadata.Photo.USER_ID in df.columns:
            data[cls.METADATA_FIELDS.USER_ID] = df[INatMetadata.Photo.USER_ID].copy()
        if INatMetadata.Taxa.Ranks.KINGDOM in df.columns:
            data[cls.ANNOTATIONS_FIELDS.KINGDOM] = (
                df[INatMetadata.Taxa.Ranks.KINGDOM].apply(lambda x: get_cleaned_label(x)))
        if INatMetadata.Taxa.Ranks.CLASS in df.columns:
            data[cls.ANNOTATIONS_FIELDS.CLASS] = (
                df[INatMetadata.Taxa.Ranks.CLASS].apply(lambda x: get_cleaned_label(x)))
        if INatMetadata.Taxa.Ranks.ORDER in df.columns:
            data[cls.ANNOTATIONS_FIELDS.ORDER] = (
                df[INatMetadata.Taxa.Ranks.ORDER].apply(lambda x: get_cleaned_label(x)))
        if INatMetadata.Taxa.Ranks.FAMILY in df.columns:
            data[cls.ANNOTATIONS_FIELDS.FAMILY] = (
                df[INatMetadata.Taxa.Ranks.FAMILY].apply(lambda x: get_cleaned_label(x)))
        if INatMetadata.Taxa.Ranks.GENUS in df.columns:
            data[cls.ANNOTATIONS_FIELDS.GENUS] = (
                df[INatMetadata.Taxa.Ranks.GENUS].apply(lambda x: get_cleaned_label(x)))
        if INatMetadata.Taxa.Ranks.SPECIES in df.columns:
            data[cls.ANNOTATIONS_FIELDS.SPECIES] = (
                df[INatMetadata.Taxa.Ranks.SPECIES].apply(lambda x: get_cleaned_label(x)))
        if INatMetadata.Taxa.Ranks.SUBSPECIES in df.columns:
            data[cls.ANNOTATIONS_FIELDS.SUBSPECIES] = (
                df[INatMetadata.Taxa.Ranks.SUBSPECIES].apply(lambda x: get_cleaned_label(x)))
        if INatMetadata.Taxa.Ranks.VARIETY in df.columns:
            data[cls.ANNOTATIONS_FIELDS.VARIETY] = (
                df[INatMetadata.Taxa.Ranks.VARIETY].apply(lambda x: get_cleaned_label(x)))

        instance = cls.from_dataframe(data, **kwargs)

        if taxonomy_level is not None:
            instance.filter_by_categories('', mode='exclude', inplace=True)

        images_size = kwargs.get('images_size', 'original')
        if images_size != 'original':
            instance.set_the_size_of_the_images(size_correction=True)

        return instance

    @classmethod
    def from_metadata_csv_files(cls,
                                source_dir,
                                taxons,
                                taxonomy_level=None,
                                out_metadata_csv=None,
                                metadata_aux_csv_files_dir=None,
                                images_size='original',
                                obs_quality_grades=[],
                                delete_aux_files_on_finish=False,
                                **kwargs):
        force = kwargs.get('force', False)
        store_metadata_csv = True
        if out_metadata_csv is None:
            out_metadata_csv = os.path.join(get_temp_folder(), f'{get_random_id()}')
            delete_aux_files_on_finish = True
            store_metadata_csv = False
        if os.path.isfile(out_metadata_csv) and not force:
            inat_metadata = INatMetadata.from_csv(out_metadata_csv)
        else:
            inat_metadata = INatMetadata.from_metadata_csv_files(
                source_dir=source_dir,
                taxons=taxons,
                taxonomy_level=taxonomy_level,
                metadata_aux_csv_files_dir=metadata_aux_csv_files_dir,
                images_size=images_size,
                obs_quality_grades=obs_quality_grades,
                delete_aux_files_on_finish=delete_aux_files_on_finish,
                **kwargs)
            if store_metadata_csv:
                inat_metadata.to_csv(out_metadata_csv)

        return cls.from_metadata(inat_metadata, taxonomy_level=taxonomy_level, **kwargs)

    def mapping_to_taxonomy_level(self, taxonomy_level_to):
        assert_cond = taxonomy_level_to in self.ANNOTATIONS_FIELDS._TAXA_RANKS_NAMES
        assert assert_cond, "Invalid taxonomy_level_to"

        self.filter_by_column(taxonomy_level_to, '', mode='exclude', inplace=True)
        self['label'] = lambda record: record[taxonomy_level_to]

    @classmethod
    def from_csv(cls, source_path: str, **kwargs) -> INatDataset:
        taxonomy_level = kwargs.get('taxonomy_level', None)
        taxons = kwargs.get('taxons', None)

        instance = super().from_csv(source_path=source_path, **kwargs)

        if taxonomy_level is not None:
            assert_cond = taxonomy_level in cls.ANNOTATIONS_FIELDS._TAXA_RANKS_NAMES
            assert assert_cond, "Invalid taxonomy_level"
            instance.mapping_to_taxonomy_level(taxonomy_level)

        if taxons is not None:
            instance.filter_by_categories(taxons, inplace=True)

        return instance

    def download(self, dest_path, **kwargs):
        num_tasks = kwargs.get('num_tasks')
        task_num = kwargs.get('task_num')

        if num_tasks is not None and task_num is not None:
            split_dataset = self.get_chunk(num_tasks, task_num)
            media_data = split_dataset.get_media_data()
        else:
            media_data = self.get_media_data()

        self._download(dest_path, media_data, **kwargs)

    @classmethod
    def _download(cls,
                  dest_path,
                  media_data,
                  **kwargs):
        images_size = kwargs.get('images_size', 'original')

        os.makedirs(dest_path, exist_ok=True)

        def get_dest_filename(record):
            return os.path.join(dest_path, cls.get_filename(record))

        def get_inat_url(record):
            return INatMetadata.inat_url_str.format(
                photo_id=record[cls.METADATA_FIELDS.ID],
                images_size=images_size,
                ext=os.path.splitext(record[cls.METADATA_FIELDS.FILENAME])[1].replace('.', ''))

        media_dict = media_data[[cls.METADATA_FIELDS.ID,
                                 cls.METADATA_FIELDS.FILENAME]].to_dict('records')
        logger.info(f"Downloading {len(media_dict)} images...")

        parallel_exec(
            func=download_file,
            elements=media_dict,
            url=get_inat_url,
            dest_filename=get_dest_filename,
            verbose=False)

    def check_for_corrupted(self):
        items = self.items
        bad_items_list = Manager().list()

        logger.info('Checking for corrupted images...')

        parallel_exec(
            func=get_corrupted_images,
            elements=items,
            images=lambda item: item,
            bads=bad_items_list)

        if len(bad_items_list) > 0:
            logger.info(f'Removing {len(bad_items_list)} corrupted images')
            self.filter_by_column(
                self.ANNOTATIONS_FIELDS.ITEM, list(bad_items_list), mode='exclude', inplace=True)

    @classmethod
    def get_filename(cls, record):
        return record[cls.METADATA_FIELDS.FILENAME]


class INatTaxonomyMapper():
    life = 'Life'
    kingdom = 'kingdom'
    phylum = 'phylum'
    subphylum = 'subphylum'
    class_taxon = 'class'
    subclass = 'subclass'
    infraclass = 'infraclass'
    superorder = 'superorder'
    order = 'order'
    family = 'family'
    genus = 'genus'
    species = 'species'

    subspecies = 'subspecies'
    variety = 'variety'

    @classmethod
    def get_taxa_name(cls, row, ancestry_pos, taxonomy_level, taxon_id_to_name_and_rank):
        if type(row['ancestry']) is str:
            try:
                ancestr_split = row['ancestry'].split('/')
                if len(ancestr_split) > ancestry_pos:
                    taxa = taxon_id_to_name_and_rank[ancestr_split[ancestry_pos]]
                    if taxa['rank'] == taxonomy_level:
                        return get_cleaned_label(taxa['name'])
                    else:
                        return ''
            except Exception as _:
                print(f'Error en: {row["ancestry"]}')
        return ''

    @classmethod
    def get_ancestry_dict(cls, row, taxon_id_to_name_and_rank):
        dict_1 = {
            taxon_id_to_name_and_rank[y]['rank']: taxon_id_to_name_and_rank[y]['name']
            for y in row['ancestry'].split('/')
        }
        dict_2 = {row['rank']: row['name']}
        return {**dict_1, **dict_2}
