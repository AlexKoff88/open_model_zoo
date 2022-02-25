from copy import deepcopy
import pickle
from pathlib import Path
import warnings
import numpy as np
from .config import ConfigError
from .data_readers import create_ann_identifier_key
from .representation import ReIdentificationAnnotation, ReIdentificationClassificationAnnotation, PlaceRecognitionAnnotation, SentenceSimilarityAnnotation, BaseRepresentation
from .annotation_converters import  BaseFormatConverter, DatasetConversionInfo, save_annotation, make_subset, analyze_dataset
from .utils import OrderedSet, contains_all, get_path, RenameUnpickler
MODULES_RENAMING = {
                    'accuracy_checker': 'openvino.tools.accuracy_checker',
                    'libs.open_model_zoo.tools.accuracy_checker.accuracy_checker':
                        [
                            'libs.open_model_zoo.tools.accuracy_checker.openvino.tools.accuracy_checker',
                            'thirdparty.open_model_zoo.tools.accuracy_checker.openvino.tools.accuracy_checker',
                            'openvino.tools.accuracy_checker',
                        ],
                    'thirdparty.open_model_zoo.tools.accuracy_checker.openvino.tools.accuracy_checker':
                        'openvino.tools.accuracy_checker',
                }


class AnnotationProvider:
    def __init__(self, annotations, meta, name='', config=None):
        self.name = name
        self.config = config
        self._data_buffer = {}
        self._meta = meta
        for ann in annotations:
            idx = create_ann_identifier_key(ann.identifier)
            self._data_buffer[idx] = ann

    @classmethod
    def from_config(cls, config, log=True):
        def _convert_annotation():
            if log:
                print("Annotation conversion for {dataset_name} dataset has been started".format(
                    dataset_name=config['name']))
                print("Parameters to be used for conversion:")
                for key, value in config['annotation_conversion'].items():
                    print('{key}: {value}'.format(key=key, value=value))
            annotation, meta = cls.convert_annotation(config)
            if annotation is not None:
                if log:
                    print("Annotation conversion for {dataset_name} dataset has been finished".format(
                        dataset_name=config['name']))
            return annotation, meta

        def _run_dataset_analysis(meta):
            if config.get('segmentation_masks_source'):
                meta['segmentation_masks_source'] = config.get('segmentation_masks_source')
            meta = analyze_dataset(annotation, meta)
            if meta.get('segmentation_masks_source'):
                del meta['segmentation_masks_source']
            return meta

        def _save_annotation():
            annotation_name = config['annotation']
            meta_name = config.get('dataset_meta')
            if meta_name:
                meta_name = Path(meta_name)
                if log:
                    print("{dataset_name} dataset metadata will be saved to {file}".format(
                        dataset_name=config['name'], file=meta_name))
            if log:
                print('Converted annotation for {dataset_name} dataset will be saved to {file}'.format(
                    dataset_name=config['name'], file=Path(annotation_name)))
            save_annotation(annotation, meta, Path(annotation_name), meta_name, config)

        annotation, meta = None, None
        use_converted_annotation = True
        if 'annotation' in config:
            annotation_file = Path(config['annotation'])
            if annotation_file.exists():
                if log:
                    print('Annotation for {dataset_name} dataset will be loaded from {file}'.format(
                        dataset_name=config['name'], file=annotation_file))
                annotation = read_annotation(get_path(annotation_file), log)
                meta = cls.load_meta(config)
                use_converted_annotation = False

        if not annotation and 'annotation_conversion' in config:
            annotation, meta = _convert_annotation()

        if not annotation:
            raise ConfigError('path to converted annotation or data for conversion should be specified')
        no_recursion = (meta or {}).get('no_recursion', False)
        annotation = _create_subset(annotation, config, no_recursion)
        dataset_analysis = config.get('analyze_dataset', False)

        if dataset_analysis:
            meta = _run_dataset_analysis(meta)

        if use_converted_annotation and contains_all(config, ['annotation', 'annotation_conversion']):
            _save_annotation()

        return cls(annotation, meta)


    def __getitem__(self, item):
        return self._data_buffer[create_ann_identifier_key(item)]

    @staticmethod
    def convert_annotation(config):
        conversion_params = config.get('annotation_conversion')
        converter = conversion_params['converter']
        annotation_converter = BaseFormatConverter.provide(converter, conversion_params)
        results = annotation_converter.convert()
        annotation = results.annotations
        meta = results.meta
        errors = results.content_check_errors
        if errors:
            warnings.warn('Following problems were found during conversion:\n{}'.format('\n'.join(errors)))

        return annotation, meta

    @property
    def identifiers(self):
        return list(map(lambda ann: ann.identifier, self._data_buffer.values()))

    def __len__(self):
        return len(self._data_buffer)

    def make_subset(self, ids=None, start=0, step=1, end=None, accept_pairs=False):
        pairwise_subset = isinstance(
            next(iter(self._data_buffer.values())), (
                ReIdentificationAnnotation,
                ReIdentificationClassificationAnnotation,
                PlaceRecognitionAnnotation,
                SentenceSimilarityAnnotation
            )
        )
        if ids:
            return ids if not pairwise_subset else self._make_subset_pairwise(ids, accept_pairs)
        if not end:
            end = self.__len__()
        ids = range(start, end, step)
        return ids if not pairwise_subset else self._make_subset_pairwise(ids, accept_pairs)

    def _make_subset_pairwise(self, ids, add_pairs=False):
        def reid_pairwise_subset(pairs_set, subsample_set, ids):
            identifier_to_index = {
                idx: index for index, idx in enumerate(self._data_buffer)
            }
            index_to_identifier = dict(enumerate(self._data_buffer))
            for idx in ids:
                subsample_set.add(idx)
                current_annotation = self._data_buffer[index_to_identifier[idx]]
                positive_pairs = [
                    identifier_to_index[pair_identifier] for pair_identifier in current_annotation.positive_pairs
                ]
                pairs_set |= positive_pairs
                negative_pairs = [
                    identifier_to_index[pair_identifier] for pair_identifier in current_annotation.positive_pairs
                ]
                pairs_set |= negative_pairs
            return pairs_set, subsample_set

        def reid_subset(pairs_set, subsample_set, ids):
            index_to_identifier = dict(enumerate(self._data_buffer))
            for idx in ids:
                subsample_set.add(idx)
                selected_annotation = self._data_buffer[index_to_identifier[idx]]
                if not selected_annotation.query:
                    query_for_person = [
                        idx for idx, (_, annotation) in enumerate(self._data_buffer.items())
                        if annotation.person_id == selected_annotation.person_id and annotation.query
                    ]
                    pairs_set |= OrderedSet(query_for_person)
                else:
                    gallery_for_person = [
                        idx for idx, (_, annotation) in enumerate(self._data_buffer.items())
                        if annotation.person_id == selected_annotation.person_id and not annotation.query
                    ]
                    pairs_set |= OrderedSet(gallery_for_person)
            return pairs_set, subsample_set

        def sentence_sim_subset(pairs_set, subsample_set, ids):
            index_to_info = {
                idx: (identifier, ann.id, ann.pair_id)
                for idx, (identifier, ann) in enumerate(self._data_buffer.items())
            }
            pair_id_to_idx = {pair_id: idx for idx, (_, _, pair_id) in index_to_info.items() if pair_id is not None}
            id_to_idx = {inst_id: idx for idx, (_, inst_id, _) in index_to_info.items()}
            for idx in ids:
                subsample_set.add(idx)
                current_annotation = self._data_buffer[index_to_info[idx][0]]
                if current_annotation.pair_id is not None and current_annotation.pair_id in id_to_idx:
                    pairs_set.add(id_to_idx[current_annotation.pair_id])
                if current_annotation.id in pair_id_to_idx:
                    pairs_set.add(pair_id_to_idx[current_annotation.id])
            return pairs_set, subsample_set

        def ibl_subset(pairs_set, subsample_set, ids):
            queries_ids = [idx for idx, (_, ann) in enumerate(self._data_buffer.items()) if ann.query]
            gallery_ids = [idx for idx, (_, ann) in enumerate(self._data_buffer.items()) if not ann.query]
            subset_id_to_q_id = {s_id: idx for idx, s_id in enumerate(queries_ids)}
            subset_id_to_g_id = {s_id: idx for idx, s_id in enumerate(gallery_ids)}
            queries_loc = [ann.coords for ann in self._data_buffer.values() if ann.query]
            gallery_loc = [ann.coords for ann in self._data_buffer.values() if not ann.query]
            dist_mat = np.zeros((len(queries_ids), len(gallery_ids)))
            for idx, query_loc in enumerate(queries_loc):
                dist_mat[idx] = np.linalg.norm(np.array(query_loc) - np.array(gallery_loc), axis=1)
            for idx in ids:
                if idx in subset_id_to_q_id:
                    pair = gallery_ids[np.argmin(dist_mat[subset_id_to_q_id[idx]])]
                else:
                    pair = queries_ids[np.argmin(dist_mat[:, subset_id_to_g_id[idx]])]
                subsample_set.add(idx)
                pairs_set.add(pair)
            return pairs_set, subsample_set

        realisation = [
            (SentenceSimilarityAnnotation, sentence_sim_subset),
            (PlaceRecognitionAnnotation, ibl_subset),
            (ReIdentificationClassificationAnnotation, reid_pairwise_subset),
            (ReIdentificationAnnotation, reid_subset),
        ]
        subsample_set = OrderedSet()
        pairs_set = OrderedSet()
        for (dtype, func) in realisation:
            if isinstance(next(iter(self._data_buffer.values())), dtype):
                pairs_set, subsample_set = func(pairs_set, subsample_set, ids)
                break
        if add_pairs:
            subsample_set |= pairs_set

        return list(subsample_set)

    @property
    def metadata(self):
        return deepcopy(self._meta)  # read-only

    @property
    def labels(self):
        return self._meta.get('label_map', {})

def read_annotation(annotation_file: Path, log=True):
    annotation_file = Path(annotation_file)

    result = []
    loader_cls = pickle.Unpickler # nosec - disable B301:pickle check
    with annotation_file.open('rb') as file:
        loader = loader_cls(file)
        try:
            first_obj = loader.load()
            if isinstance(first_obj, DatasetConversionInfo):
                if log:
                    describe_cached_dataset(first_obj)
            else:
                result.append(first_obj)
        except ModuleNotFoundError:
            loader_cls = RenameUnpickler
            loader = loader_cls(file, MODULES_RENAMING)
            try:
                first_obj = loader.load()
                if isinstance(first_obj, DatasetConversionInfo):
                    if log:
                        describe_cached_dataset(first_obj)
                else:
                    result.append(first_obj)
            except EOFError:
                return result
        except EOFError:
            return result
        while True:
            try:
                result.append(
                    BaseRepresentation.load(file, loader_cls(file) if loader_cls != RenameUnpickler
                    else loader_cls(file, MODULES_RENAMING))
                )
            except EOFError:
                break

    return result

def ignore_subset_settings(config):
    subset_file = config.get('subset_file')
    store_subset = config.get('store_subset')
    if subset_file is None:
        return False
    if Path(subset_file).exists() and not store_subset:
        return True
    return False

def _create_subset(annotation, config, no_recursion=False):
    subsample_size = config.get('subsample_size')
    if not ignore_subset_settings(config):

        if subsample_size is not None:
            subsample_seed = config.get('subsample_seed', 666)
            shuffle = config.get('shuffle', True)
            annotation = create_subset(annotation, subsample_size, subsample_seed, shuffle, no_recursion)

    elif subsample_size is not None:
        warnings.warn("Subset selection parameters will be ignored")
        config.pop('subsample_size', None)
        config.pop('subsample_seed', None)
        config.pop('shuffle', None)

    return annotation

def describe_cached_dataset(dataset_info):
    print('Loaded dataset info:')
    if dataset_info.dataset_name:
        print('\tDataset name: {}'.format(dataset_info.dataset_name))
    print('\tAccuracy Checker version {}'.format(dataset_info.ac_version))
    print('\tDataset size {}'.format(dataset_info.dataset_size))
    print('\tConversion parameters:')
    for key, value in dataset_info.conversion_parameters.items():
        print('\t\t{key}: {value}'.format(key=key, value=value))
    if dataset_info.subset_parameters:
        print('\nSubset selection parameters:')
        for key, value in dataset_info.subset_parameters.items():
            print('\t\t{key}: {value}'.format(key=key, value=value))

def create_subset(annotation, subsample_size, subsample_seed, shuffle=True, no_recursion=False):
    if isinstance(subsample_size, str):
        if subsample_size.endswith('%'):
            try:
                subsample_size = float(subsample_size[:-1])
            except ValueError as value_err:
                raise ConfigError('invalid value for subsample_size: {}'.format(subsample_size)) from value_err
            if subsample_size <= 0:
                raise ConfigError('subsample_size should be > 0')
            subsample_size *= len(annotation) / 100
            subsample_size = int(subsample_size) or 1
    try:
        subsample_size = int(subsample_size)
    except ValueError as value_err:
        raise ConfigError('invalid value for subsample_size: {}'.format(subsample_size)) from value_err
    if subsample_size < 1:
        raise ConfigError('subsample_size should be > 0')
    return make_subset(annotation, subsample_size, subsample_seed, shuffle, no_recursion)
