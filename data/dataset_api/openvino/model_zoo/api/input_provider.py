from .config import ConfigError
from .data_readers import BaseReader, REQUIRES_ANNOTATIONS
from .preprocessor import PreprocessingExecutor
from .utils import extract_image_representations


class InputProvider:
    def __init__(self, data_reader, preprocessor):
        self.data_reader = data_reader
        self.preprocessor = preprocessor

    @classmethod
    def from_config(cls, config, annotation_provider=None):
        data_reader_config = config.get('reader', 'opencv_imread')
        data_source = config.get('data_source')
        if isinstance(data_reader_config, str):
            data_reader_type = data_reader_config
            data_reader_config = None
        elif isinstance(data_reader_config, dict):
            data_reader_type = data_reader_config['type']
        else:
            raise ConfigError('reader should be dict or string')
        if data_reader_type in REQUIRES_ANNOTATIONS:
            data_source = annotation_provider
        data_reader = BaseReader.provide(data_reader_type, data_source, data_reader_config)
        transform = None
        if 'preprocessing' in config:
            transform = PreprocessingExecutor(preprocessing_configuration=config.get('preprocessing'))
        return cls(data_reader, transform)

    def read_batch(self, batch_identifiers):
        return [self.read(identifier) for identifier in batch_identifiers]

    def read(self, identifier):
        return self.data_reader(identifier)

    def preprocess_batch(self, batch_input):
        if self.preprocessor:
            batch_input = self.preprocessor.process(batch_input)
        return extract_image_representations(batch_input)

    def preprocess(self, input_data):
        if self.preprocessor:
            input_data = self.preprocessor.preprocess(input_data)
        return extract_image_representations(input_data)

    def __call__(self, identifier):
        input_data = self.read(identifier)
        return self.preprocess(input_data)

    def set_transforms(self, transforms):
        self.preprocessor = PreprocessingExecutor(transforms)

    def reset(self):
        self.data_reader.reset()