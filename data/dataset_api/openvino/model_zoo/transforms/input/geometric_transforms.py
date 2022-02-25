from unicodedata import name
from ...api.preprocessor import Preprocessor

def transpose(axes):
    cfg = {
        'type': 'transpose',
        'axes': axes
    }
    return Preprocessor.provide('transpose', cfg, name='transpose')
