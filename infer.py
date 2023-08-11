import os
import sys
__dir__ = os.path.join('src', 'ppocr')
sys.path.append(__dir__)
sys.path.append('./src/')

from PIL import Image
from src.ppocr.tools.infer_kie_token_ser_test import *
from src.ppocr.ppocr.utils.visual import draw_ser_results
from src.ocr.tools.predictor import Predictor
from src.ocr.tools.config import Cfg
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

def recog(image, transcripts, weight_path):
    config = Cfg.load_config_from_name('vgg_transformer')
    config['predictor']['import'] = weight_path
    config['predictor']['beamsearch'] = True
    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    detector = Predictor(config)
    for trans in tqdm(transcripts):
        x, y, w, h = trans['bbox']
        roi = Image.fromarray(image[y:h, x:w])
        trans['transcription'] = detector.predict(roi)
    return transcripts 

def infer(image_path):
    opt = {
        'config': './src/config/kie/vi_layoutxlm/ser_mcocr.yml',
        'otp': {
            'Architecture.Backbone.checkpoints': './src/weights/mcocr/best_accuracy',
            'Global.infer_img': image_path
        }
    }

    weight_path = './src/weights/vgg_transformerocr_1M_500k.pth'
    output = main(opt)
    result = recog(image_path, output, weight_path)
    img_res = draw_ser_results(image_path, result)
    return img_res, result
        
def GUI():
    demo = gr.Interface(
        fn=infer,
        inputs=[gr.Image()],
        outputs=["image", "text"]
    )

    demo.launch()

if __name__ == "__main__":
    GUI()