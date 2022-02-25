from ..api.dataset_iterator import DatasetIterator
from ..api.annotation_provider import AnnotationProvider
from ..api.input_provider import InputProvider

class COCO2017Dataset(DatasetIterator):
    tasks = {
        'detection': 'mscoco_detection',
        'instance_segmentation': 'mscoco_maskrcnn',
        'keypoints': 'mscoc_keypoints'
    }
    def __init__(self, annotation_file, data_dir, task_type='detection', transforms=None):
        if task_type not in self.tasks:
            raise ValueError(f'Unsupported task type: {task_type}')

        annotation_provider = AnnotationProvider.from_config({
            'name': 'coco2017',
            'annotation_conversion': {
                'converter': self.tasks[task_type],
                'annotation_file': annotation_file
            }
        })
        input_provider = InputProvider.from_config({'data_source': data_dir})
        if transforms:
            input_provider.set_transforms(transforms)
        super().__init__(input_provider, annotation_provider)