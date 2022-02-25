import numpy as np
from PIL import Image
import torch

def from_torch(torch_transforms, input_data_type='pil'):
    assert input_data_type in ['pil', 'numpy', 'tensor']
    return TorchPreprocessingWrapper(torch_transforms, input_data_type)


class TorchPreprocessingWrapper:
    def __init__(self, transforms, input_type):
        self.torch_transforms = transforms
        self.input_type = input_type
    
    def __call__(self, image, *args, **kwrg):
        data_tensor = self.torch_transforms(self._prepare_input(image.data))
        image.data = data_tensor.cpu().numpy()
        return image
    
    def _prepare_input(self, data):
        if self.input_type == 'numpy':
            return data
        if self.input_type == 'pil':
            return Image.fromarray(data)
        return torch.from_numpy(data)

