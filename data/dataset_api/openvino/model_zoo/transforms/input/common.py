from ...api.preprocessor import PreprocessingExecutor
from ...api.data_readers import DataRepresentation
from ...api.utils import extract_image_representations

class TransformComposition(PreprocessingExecutor):
    def __init__(self, transforms):
        super().__init__(transforms)

    def __call__(self, batch_data, batch_meta, *args, **kwargs):
        data_rep = [DataRepresentation(data, meta) for data, meta in zip(batch_data, batch_meta)]
        data_rep_batch = super().__call__(data_rep, *args, **kwargs)
        return extract_image_representations(data_rep_batch)