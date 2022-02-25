from ..api.dataset_iterator import DatasetIterator
from ..api.annotation_provider import AnnotationProvider
from ..api.input_provider import InputProvider


class ImageNetDataset(DatasetIterator):
    def __init__(annotation_file, labels_file, data_dir, has_background, transforms=None):
        annotation_provider = AnnotationProvider.from_config({
            'annotation_conversion': {
                'converter': 'imagenet',
                'annotation_file': annotation_file,
                'labels_file': labels_file,
                'has_background': True,
                'images_dir': data_dir
            }
        })
        input_provider = InputProvider.from_config({'data_source': data_dir})
        input_provider.set_transforms(transforms)
        super().__init__(input_provider, annotation_provider)
