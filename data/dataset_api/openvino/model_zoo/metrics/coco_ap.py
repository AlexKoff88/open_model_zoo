from ..api.metrics import Metric
metric_by_type = {
    'detection': 'coco_precision',
    'segmentation': 'coco_segm_precision',
    'keypoints': 'coco_keypoints_precision'
}

def create_coco_ap(max_detections=None, threshold='.50:.05:.95', task_type='detection', dataset=None, name=None):
    assert task_type in metric_by_type
    metric_identifier = metric_by_type[task_type]
    return Metric.provide(
        metric_identifier, {
            'type': metric_identifier, 'max_detections': max_detections, 'threshold': threshold
            }, dataset, name or metric_identifier)

