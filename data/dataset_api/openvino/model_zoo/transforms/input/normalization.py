from ...api.preprocessor import Preprocessor

def normalization(mean=None, std=None):
    cfg = {
        'type': 'normalization',
        'mean': mean,
        'std': std
    }
    return Preprocessor.provide('normalization', cfg, name='normalization')