from ...api.preprocessor import Preprocessor

def _provide_size(cfg, size=None, scale=None, dst_height=None, dst_width=None):
    if size:
        cfg['size'] = size
    if dst_height and dst_width:
        cfg['dst_height'] = dst_height
        cfg['dst_width'] = dst_width
    if scale:
        cfg['scale'] = scale
    return cfg

def pillow_resize(
    size=None, scale=None, dst_height=None, dst_width=None, 
    aspect_ratio_scale=None, interpolation='LINEAR'
    ):
    cfg = {
        'type': 'resize',
        'resize_realization': 'pillow',
        'aspect_ratio_scale': aspect_ratio_scale,
        'interpolation': interpolation
        }
    _provide_size(cfg, size, scale, dst_height, dst_width)
    return Preprocessor.provide('resize', cfg, name='pillow_resize')

def opencv_resize(
    size=None, scale=None, dst_height=None, dst_width=None, 
    aspect_ratio_scale=None, interpolation='LINEAR'
    ):
    cfg = {
        'type': 'resize',
        'resize_realization': 'opencv',
        'aspect_ratio_scale': aspect_ratio_scale,
        'interpolation': interpolation
        }
    _provide_size(cfg)
    return Preprocessor.provide('resize', cfg, name='opencv_resize')
