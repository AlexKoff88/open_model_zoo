# Copyright (c) 2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import yaml

from pathlib import Path

from  openvino.model_zoo import (
    _configuration, _common, omz_downloader, omz_converter
)
from openvino.model_zoo.download_engine import validation


class Model:
    def __init__(
        self, name=None, model_path=None, description=None, task_type=None,
        subdirectory=None, architecture_type=None, ie=None
    ):
        self.model_path = model_path
        self.description = description
        self.task_type = task_type
        self.name = name
        self.subdirectory = subdirectory
        self.architecture_type = architecture_type

        self.model = None
        self.compiled_model = None
        self.ie = ie
        self.device = None
        self._set_config_paths()

    def _set_config_paths(self):
        accuracy_config_path = _common.MODEL_ROOT / self.subdirectory / 'accuracy-check.yml'
        self.accuracy_config_path = accuracy_config_path if accuracy_config_path.exists() else None

        model_config_path = _common.MODEL_ROOT / self.subdirectory / 'model.yml'
        self.model_config_path = model_config_path if model_config_path.exists() else None

    @classmethod
    def download(cls, model_name, *, precision='FP32', download_dir='models', cache_dir=None, ie=None):
        '''
        Downloads target model. If the model has already been downloaded,
        retrieves the model from the cache instead of downloading it again.
        Then creates OpenVINO Intermediate Representation (IR) network model.

        Creates an object which stores information about the downloaded and IR model.

        Parameters
        ----------
            model_name
                Target model name.
            precision
                Target model precisions.
            download_dir
                Model download directory. By default creates a folder 'models'
                in current directory and downloads to it.
            cache_dir
                Cache directory.
                The script will place a copy of each downloaded file in the cache, or,
                if it is already there, retrieve it from the cache.
                By default creates a folder '.cache' in current download directory.
            ie
                Inference Engine instance
        '''
        download_dir = Path(download_dir)
        cache_dir = cache_dir or download_dir / '.cache'

        model = cls._load_model(cls, model_name)

        model_dir = download_dir / model.subdirectory

        flags = ['--name=' + model_name,
                 '--output_dir=' + str(download_dir),
                 '--cache_dir=' + str(cache_dir),
                 '--precisions=' + str(precision)]

        omz_downloader.download(flags)

        description = '{}\n\n    License: {}'.format(model.description, model.license_url)
        task_type = model.task_type

        if precision not in model.precisions:
            raise ValueError('Incorrect precision value!\n    Current precision value is: \''
                                        + str(precision) + '\'.\n    Allowed precision values for model are: '
                                        + str(model.precisions))

        prefix = Path(model_dir) / precision / model_name

        model_path = str(prefix) + '.xml'
        bin_path = str(prefix) + '.bin'
        name = model.name
        subdirectory = model.subdirectory

        if not os.path.exists(model_path) or not os.path.exists(bin_path):
            omz_converter.converter(['--name=' + model_name, '--precisions=' + precision,
                        '--download_dir=' + str(download_dir)])
        return cls(name, model_path, description, task_type, subdirectory, model.architecture_type, ie)

    @classmethod
    def from_pretrained(cls, model_path, *, task_type=None, ie=None):
        '''
        Loads model from existing .xml, .onnx files.
        Parameters
        ----------
            model_path
                Path to .xml or .onnx file.
            task_type
                Type of task that the model performs.
            ie
                Inference Engine instance
        '''
        model_path = Path(model_path).resolve()

        if not model_path.exists():
            raise ValueError('Path {} to model file does not exist.'.format(model_path))

        if model_path.suffix not in ['.xml', '.onnx']:
            raise ValueError('Unsupported model format {}. Only .xml or .onnx supported.'.format(model_path.suffix))

        description = 'Pretrained model {}'.format(model_path.name)
        name = model_path.stem
        model = cls._load_model(cls, name)
        task_type = model.task_type if model else task_type
        subdirectory = model.subdirectory if model else model_path.parent
        architecture_type = model.architecture_type if model else None
        model_path = str(model_path)

        return cls(name, model_path, description, task_type, subdirectory, architecture_type, ie)

    def _load_model(self, model_name):
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(all=False, list=None, name=model_name, print_all=False)
        try:
            model = _configuration.load_models_from_args(parser, args, _common.MODEL_ROOT,
                        mode=_configuration.ModelLoadingMode.ignore_composite)[0]
        except SystemExit:
            model = None

        return model

    def _load(self, device):
        self.model = self.ie.read_model(self.model_path)
        self.compiled_model = self.ie.compile_model(self.model, device)
        self.device = device

    def __call__(self, inputs, device='CPU'):
        if self.ie is None:
            raise TypeError('ie is not specified or is of the wrong type.'
                            'Please check ie is of openvino.runtime.Core type.')

        if self.compiled_model is None or device != self.device:
            self._load(device)

        input_names = [input.get_any_name() for input in self.model.inputs]

        for input_name in inputs:
            if input_name not in input_names:
                raise ValueError('Unknown input name {}'.format(input_name))

        return self.compiled_model.infer_new_request(inputs)

    def accuracy_checker_config(self):
        accuracy_config = None
        if self.accuracy_config_path is not None:
            with self.accuracy_config_path.open('rb') as config_file, \
                    validation.deserialization_context('Loading config "{}"'.format(self.accuracy_config_path)):
                accuracy_config = yaml.safe_load(config_file)

        return accuracy_config

    def model_config(self):
        model_config = None
        if self.model_config_path is not None:
            with self.model_config_path.open('rb') as config_file, \
                    validation.deserialization_context('Loading config "{}"'.format(self.model_config_path)):
                model_config = yaml.safe_load(config_file)

        return model_config

    def inputs(self):
        if self.model is None:
            try:
                self.model = self.ie.read_model(self.model_path)
            except AttributeError:
                raise TypeError('ie is not specified or is of the wrong type.'
                                'Please check ie is of openvino.runtime.Core type.')

        return self.model.inputs

    def outputs(self):
        if self.model is None:
            try:
                self.model = self.ie.read_model(self.model_path)
            except AttributeError:
                raise TypeError('ie is not specified or is of the wrong type.'
                                'Please check ie is of openvino.runtime.Core type.')

        return self.model.outputs

    def preferable_input_shape(self, name):
        input_info = self.model_config().get('input_info', [])
        for input in input_info:
            if input['name'] == name:
                return input['shape']

        return None

    def layout(self, name):
        input_info = self.model_config().get('input_info', [])
        for input in input_info:
            if input['name'] == name:
                return input['layout']

        return None

    def input_info(self):
        return self.model_config().get('input_info', [])

    def _create_model_api_instance(
        self, model_creator, num_streams='', num_threads=None, max_num_requests=1, configuration=None
    ):
        plugin_config = _common.get_user_config(self.device, num_streams, num_threads)
        model_adapter = _common.OpenvinoAdapter(_common.create_core(), self.model_path, device=self.device,
                            plugin_config=plugin_config, max_num_requests=max_num_requests)
        self.model_api = model_creator(model_adapter, configuration)
        self.model_api.load()

    def model_api_inference(self, inputs, device='CPU', model_creator=None, asynk_mode=False, **kwargs):
        self.device = device
        if model_creator is None:
            model_creator = self.model_creator

        self._create_model_api_instance(model_creator, **kwargs)

        # if not asynk_mode:
        preprocessed_inputs, meta = self.model_api.preprocess(inputs)
        raw_results = self.model_api.infer_sync(preprocessed_inputs)
        results = self.model_api.postprocess(raw_results, meta)

        return results

    def model_creator(self, model_adapter, configuration):
        try:
            if self.task_type == 'classification' and self.architecture_type is None:
                return _common.Classification(model_adapter, configuration)
            elif self.architecture_type is None:
                raise TypeError('architecture_type is not set or model is usupported by Model API.')
            else:
                return _common.Model.create_model(self.architecture_type, model_adapter, configuration)
        except Exception as exc:
            raise RuntimeError('Unable to create model class, please check configuration parameters.'
                               'Errors occured: ' + str(exc))
