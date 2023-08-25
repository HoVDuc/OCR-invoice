import os
import sys
import json
__dir__ = os.path.join('src', 'ppocr')
sys.path.append(__dir__)
sys.path.append('./src/')

from src.ppocr.tools.infer_kie_token_ser_test import *
from src.ppocr.ppocr.utils.visual import draw_ser_results
import gradio as gr


class Inference:
    
    def __init__(self, otp) -> None:
        self.config = main(otp)
        self.ser_engine = SerPredictor(self.config)
    
    
    
    def process_info(self, results):
        info = {
            'SELLER': '',
            'ADDRESS': '',
            'STAFF': '',
            'TIMESTAMP': '',
            'CODE': '',
            'PRODUCTS': [],
            'TOTAL_COST': 0
        }
        
        current_product = {
            'PRODUCT': '',
            'NUMBER': 0,
            'PRICE': 0
        }
        
        products = []
        for result in results:
            label = result['pred']
            transcription = result['transcription']
            
            if label != 'O':
                if label in ['PRODUCT', 'NUMBER', 'PRICE']:
                    if label == 'PRODUCT':
                        current_product['PRODUCT'] = transcription
                        products.append(current_product.copy())
                    else:
                        products[-1][label] = transcription
                else: 
                    if label == 'TIMESTAMP':
                        text = transcription
                        code, time = text.split('Ngày')
                        info['CODE'] = code.strip()
                        info['TIMESTAMP'] = time[1:]
                    else:
                        info[label] = transcription
                        
        info['PRODUCTS'] = products 
        return json.dumps(info, indent=1, ensure_ascii=False)
            
    def __call__(self, image_path):
        self.config['Global']['infer_img'] = image_path
        if self.config["Global"].get("infer_mode", None) is False:
            data_dir = self.config['Eval']['dataset']['data_dir']
            with open(self.config['Global']['infer_img'], "rb") as f:
                infer_imgs = f.readlines()
        else:
            try:
                infer_imgs = get_image_file_list(self.config['Global']['infer_img'])
            except:
                infer_imgs = [self.config['Global']['infer_img']]

        for idx, info in enumerate(infer_imgs):
            if self.config["Global"].get("infer_mode", None) is False:
                data_line = info.decode('utf-8')
                substr = data_line.strip("\n").split("\t")
                img_path = os.path.join(data_dir, substr[0])
                data = {'img_path': img_path, 'label': substr[1]}
            else:
                img_path = info
                data = {'img_path': img_path}

            result, _ = self.ser_engine(data)
            result = result[0]      
        
        img_res = draw_ser_results(image_path, result)
        info = self.process_info(result)
        return img_res, info
        
def GUI():
    otp = {
        'config': './src/config/kie/vi_layoutxlm/ser_mcocr.yml',
        'otp': {
            'Architecture.Backbone.checkpoints': './src/weights/mcocr/best_accuracy',
        }
    }
    infer = Inference(otp)
    demo = gr.Interface(
        fn=infer,
        inputs=[gr.Image()],
        outputs=["image", "text"]
    )

    demo.launch(share=False)

if __name__ == "__main__":
    GUI()