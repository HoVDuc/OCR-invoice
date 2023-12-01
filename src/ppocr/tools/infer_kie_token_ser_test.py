# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys
import threading

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append('./src/')

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'
import cv2
import json
import paddle

from src.ocr.tools.predictor import Predictor
from src.ocr.tools.config import Cfg

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import load_model
from ppocr.utils.visual import draw_ser_results
from ppocr.utils.utility import get_image_file_list, load_vqa_bio_label_maps

from PIL import Image
from tqdm import tqdm
import tools.program as program


def to_tensor(data):
    import numbers
    from collections import defaultdict
    data_dict = defaultdict(list)
    to_tensor_idxs = []

    for idx, v in enumerate(data):
        if isinstance(v, (np.ndarray, paddle.Tensor, numbers.Number)):
            if idx not in to_tensor_idxs:
                to_tensor_idxs.append(idx)
        data_dict[idx].append(v)
    for idx in to_tensor_idxs:
        data_dict[idx] = paddle.to_tensor(data_dict[idx])
    return list(data_dict.values())


class SerPredictor(object):
    def __init__(self, config):
        global_config = config['Global']
        self.algorithm = config['Architecture']["algorithm"]

        # build post process
        self.post_process_class = build_post_process(config['PostProcess'],
                                                     global_config)
        # build model
        self.model = build_model(config['Architecture'])
        self.config = Cfg.load_config_from_file(global_config['rec_config_path'])
        self.config['predictor']['import'] = global_config['rec_weight']
        self.config['predictor']['beamsearch'] = True
        self.config['cnn']['pretrained'] = False
        self.config['device'] = 'cuda' if global_config['use_gpu'] else 'cpu'
        self.detector = Predictor(self.config)
        

        load_model(
            config, self.model, model_type=config['Architecture']["model_type"])

        from paddleocr import PaddleOCR

        self.ocr_engine = PaddleOCR(
            use_gpu=global_config['use_gpu'],
            det_db_box_thresh=0.3,
            det_db_unclip_ratio=2.0,
            det_limit_type=global_config['det_limit_type'],
            use_angle_cls=False,
            show_log=False,
            det_model_dir=global_config.get("kie_det_model_dir", None),
        )
        
        # create data ops
        transforms = []
        for op in config['Eval']['dataset']['transforms']:
            op_name = list(op)[0]
            if 'Label' in op_name:
                op[op_name]['ocr_engine'] = self.ocr_engine
            elif op_name == 'KeepKeys':
                op[op_name]['keep_keys'] = [
                    'input_ids', 'bbox', 'attention_mask', 'token_type_ids',
                    'image', 'labels', 'segment_offset_id', 'ocr_info',
                    'entities'
                ]

            transforms.append(op)
        
        if config["Global"].get("infer_mode", None) is None:
            global_config['infer_mode'] = True
        self.ops = create_operators(config['Eval']['dataset']['transforms'],
                                    global_config)
        self.model.eval()

    def recog(self, image, transcripts):
        for trans in tqdm(transcripts[0]):
            if trans['pred_id'] != 0:
                x, y, w, h = trans['bbox']
                roi = Image.fromarray(image[y:h, x:w])
                trans['transcription'] = self.detector.predict(roi)
        return transcripts 

    def __call__(self, data):
        try:
            with open(data["img_path"], 'rb') as f:
                img = f.read()
        except:
            _, encoded_image = cv2.imencode('.png', data["img_path"])
            img = encoded_image.tobytes()
            
        data["image"] = img
        batch = transform(data, self.ops)
        batch = to_tensor(batch)
        preds = self.model(batch)
        post_result = self.post_process_class(
            preds, segment_offset_ids=batch[6], ocr_infos=batch[7])
        self.recog(data['image'], batch[7])
        return post_result, batch


def main(otp):
    config, device, logger, vdl_writer = program.preprocess(otp)
    os.makedirs(config['Global']['save_res_path'], exist_ok=True)
    return config

if __name__ == "__main__":
    main()
