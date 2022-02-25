from ...api.representation import ClassificationPrediction

class ClassificationPostprocessor:
    def __init__(self, output_tensor=None, additional_transforms=None):
        self.output_tensor = output_tensor
        self._output_tensor_names = [output_tensor]
        self._output_tensor_name = output_tensor
        self._output_node_name = output_tensor
        if output_tensor is not None and not isinstance(output_tensor, str):
            self._output_tensor_names = output_tensor.names
            self._output_tensor_name = output_tensor.any_name
            self._output_node_name = output_tensor.get_node().friendly_name

        self.additional_transforms = additional_transforms

    def _get_preds(self, predictions):
        if self.output_tensor:
            preds = predictions.get(self.output_tensor)
            if preds is not None:
                return preds
            if self._output_node_name in predictions:
                return predictions[self._output_node_name]
            for out_tensor_name in self._output_tensor_names:
                if out_tensor_name in predictions:
                    return predictions[out_tensor_name]
            raise ValueError('Output tensor is not found')
        
        return next(iter(predictions.values()))

    def __call__(self, predictions, meta=None, identifiers=None):
        raw_predictions = self._get_preds(predictions)
        results = []
        if identifiers is None:
            identifiers = [None]
        if meta is None:
            meta = [None] * len(identifiers)
        for identifier, out, single_meta in zip(identifiers, raw_predictions, meta):
            pred = ClassificationPrediction(identifier, out)
            if self.additional_transforms:
                pred = self.additional_transforms(pred, single_meta)
            results.append(pred)
        return results