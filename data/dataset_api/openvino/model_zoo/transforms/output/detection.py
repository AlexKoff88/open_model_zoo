from ...api.adapters import create_adapter
from ...api.postprocessor import PostprocessingExecutor

class YoloV5:
    def __init__(self,
    output_blobs, 
    anchors='yolo_v3',
    num_classes=80,
    conf_threshold=0.001,
    nms_threshold=0.65,
    additional_transforms=None
    ):
        self.outputs = output_blobs
        self.anchors = anchors
        self.classes = num_classes
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self._adapter = create_adapter({
            "type": "yolo_v5",
            "anchors": self.anchors,
            "num": 3,
            "coords": 4,
            "classes": self.classes,
            "threshold": 0.001,
            "anchor_masks": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            "raw_output": True,
            "transpose": [0, 3, 1, 2],
            "output_format": "BHW",
            "cells": [80, 40, 20],
            "outputs": [out.get_node().friendly_name for out in self.outputs]
        })
        self._postprocessor = PostprocessingExecutor(postprocessing_configuration=[
                    {
                        "type": "resize_prediction_boxes"
                    },
                    {
                        "type": "filter",
                        "apply_to": "prediction",
                        "min_confidence": self.conf_threshold,
                        "remove_filtered": True
                    },
                    {
                        "type": "nms",
                        "overlap": self.nms_threshold
                    },
                    {
                        "type": "clip_boxes",
                        "apply_to": "prediction"
                    }
                ]
        )

        self.additional_transforms = additional_transforms

    def __call__(self, predictions, meta, identifiers=None):
        raw_predictions = {out.get_node().friendly_name: predictions[out] for out in self.outputs}
        if identifiers is None:
            identifiers = [None]
        base_results = self._adapter.process(identifiers, raw_predictions, meta)
        results = self._postprocessor.process_batch(None, base_results, meta, allow_empty_annotation=True)
        if self.additional_transforms:
            results = self.additional_transforms(results)
        return results

    
    


