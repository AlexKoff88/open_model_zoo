from pathlib import Path
from ..api.dataset_iterator import DatasetIterator
from ..api.annotation_provider import AnnotationProvider
from ..api.input_provider import InputProvider

# AC configuration based
class CIFAR10Dataset(DatasetIterator):
    def __init__(self, data_dir, split='test', transforms=None, reader='opencv_imread'):
        data_dir = Path(data_dir)
        data_source = data_dir / split
        if not data_source.exists():
            data_source = data_dir / split
        annotation_provider = AnnotationProvider.from_config({
            'name': 'cifar10',
            'annotation_conversion': {
            'converter': 'cifar',
            'data_batch_file': data_dir / f'{split}_batch' if (data_dir / f'{split}_batch').exists() else data_dir / 'cifar-10-batches-py' / f'{split}_batch',
            'convert_images': True,
            'converted_images_dir': data_source,
            'num_classes': 10}})
        input_provider = InputProvider.from_config({
            'data_source': data_source, 'reader': reader
            })
        input_provider.set_transforms(transforms)
        super().__init__(input_provider, annotation_provider)