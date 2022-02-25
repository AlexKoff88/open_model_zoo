from pathlib import Path
import yaml
from .data_readers import REQUIRES_ANNOTATIONS, serialize_identifier, deserialize_identifier
from .utils import read_yaml, set_image_metadata
from .annotation_provider import AnnotationProvider
from .input_provider import InputProvider

class DatasetIterator:
    def __init__(
            self, input_provider, annotation_provider=None, tag='', dataset_config=None, data_list=None, subset=None,
            batch=None
    ):
        self.tag = tag
        self.input_provider = input_provider
        self.annotation_provider = annotation_provider
        self.dataset_config = dataset_config or {}
        self.batch = batch if batch is not None else self.dataset_config.get('batch')
        self.subset = subset
        self.create_data_list(data_list)
        if self.store_subset:
            self.sava_subset()
    @classmethod
    def from_config(cls, config, load_annotation=False):
        if load_annotation:
            annotation_provider = AnnotationProvider.from_config(config)
        else:
            annotation_provider = None
        input_provider = InputProvider.from_config(config, annotation_provider)
    
        return cls(input_provider, annotation_provider, dataset_config=config
        )

    def create_data_list(self, data_list=None):
        if data_list is not None:
            self._data_list = data_list
            return
        self.store_subset = self.dataset_config.get('store_subset', False)
        if 'subset' in self.dataset_config:
            self._create_data_list(self.dataset_config['subset'])
            return

        if 'subset_file' in self.dataset_config:
            subset_file = Path(self.dataset_config['subset_file'])
            if subset_file.exists() and not self.store_subset:
                self.read_subset(subset_file)
                return

            self.store_subset = True

        if self.annotation_provider:
            self._data_list = self.annotation_provider.identifiers
            return
        self._data_list = [file.name for file in self.input_provider.data_source.glob('*')]

    def read_subset(self, subset_file):
        self._create_data_list(read_yaml(subset_file))
        print("loaded {} data items from {}".format(len(self._data_list), subset_file))

    def _create_data_list(self, subset):
        identifiers = [deserialize_identifier(idx) for idx in subset]
        self._data_list = identifiers

    def sava_subset(self):
        identifiers = [serialize_identifier(idx) for idx in self._data_list]
        subset_file = Path(self.dataset_config.get(
            'subset_file', '{}_subset_{}.yml'.format(self.dataset_config['name'], len(identifiers))))
        print("Data subset will be saved to {} file".format(subset_file))
        with subset_file.open(mode="w") as sf:
            yaml.safe_dump(identifiers, sf)

    def __getitem__(self, item):
        if self.batch is None:
            self.batch = 1
        if not isinstance(item, slice) and self.size <= item * self.batch:
            raise IndexError
        batch_annotation = []
        batch_start = item * self.batch
        batch_end = min(self.size, batch_start + self.batch)
        batch_input_ids = self.subset[batch_start:batch_end] if self.subset else range(batch_start, batch_end)
        batch_identifiers = [self._data_list[idx] for idx in batch_input_ids]
        batch_input = [self.input_provider.read(identifier=identifier) for identifier in batch_identifiers]
        if self.annotation_provider:
            batch_annotation = [self.annotation_provider[idx] for idx in batch_identifiers]

            for annotation, input_data in zip(batch_annotation, batch_input):
                self.set_annotation_metadata(
                    annotation, input_data, self.input_provider.data_reader.data_source)
        batch_input, batch_meta = self.input_provider.preprocess_batch(batch_input)

        return list(zip(batch_input_ids, batch_annotation)), batch_input, batch_meta

    def __len__(self):
        if self.subset is None:
            return len(self._data_list)
        return len(self.subset)

    @property
    def identifiers(self):
        return self._data_list

    def make_subset(self, ids=None, start=0, step=1, end=None, accept_pairs=False):
        if self.annotation_provider:
            ids = self.annotation_provider.make_subset(ids, start, step, end, accept_pairs)
        if ids:
            self.subset = ids
            return
        if not end:
            end = self.size
        self.subset = range(start, end, step)
        if self.data_reader.name in REQUIRES_ANNOTATIONS:
            self.data_reader.subset = self.subset

    @property
    def batch(self):
        return self._batch

    @batch.setter
    def batch(self, batch):
        self._batch = batch

    @property
    def labels(self):
        meta = self.annotation_provider.metadata or {}
        return meta.get('label_map', {})

    @property
    def metadata(self):
        if self.annotation_provider:
            return self.annotation_provider.metadata
        return {}

    def reset(self, reload_annotation=False):
        if self.subset:
            self.subset = None
        if self.annotation_provider and reload_annotation:
            self.annotation_provider.reload()
            self.create_data_list()
        self.input_provider.reset()

    def set_transforms(self, transforms):
        self.input_provider.set_transforms(transforms)

    @property
    def full_size(self):
        if not self.annotation_provider:
            return len(self._data_list)
        return len(self.annotation_provider)

    @property
    def size(self):
        return self.__len__()

    @property
    def multi_infer(self):
        return getattr(self.data_reader, 'multi_infer', False)

    def set_annotation_metadata(self, annotation, image, data_source):
        set_image_metadata(annotation, image)
        annotation.set_data_source(data_source if not isinstance(data_source, (list, AnnotationProvider)) else [])
        segmentation_mask_source = self.dataset_config.get('segmentation_masks_source')
        annotation.set_segmentation_mask_source(segmentation_mask_source)
        annotation.set_additional_data_source(self.dataset_config.get('additional_data_source'))
        annotation.set_dataset_metadata(self.annotation_provider.metadata)

