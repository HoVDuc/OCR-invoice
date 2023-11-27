import os
import sys
import json
__dir__ = os.path.join('src', 'ppocr')
sys.path.append(__dir__)
sys.path.append('./src/')

from src.ppocr.tools.infer_kie_token_ser_test import *
from src.ppocr.ppocr.utils.visual import draw_ser_results
import gradio as gr
import pandas as pd


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
                    info[label] = transcription
                        
        info['PRODUCTS'] = products 
        return json.dumps(info, indent=1, ensure_ascii=False)
            
    def __call__(self, image_path):
        self.config['Global']['infer_img'] = image_path
        try:
            infer_imgs = get_image_file_list(self.config['Global']['infer_img'])
        except:
            infer_imgs = self.config['Global']['infer_img']

        img_path = infer_imgs
        data = {'img_path': img_path}

        result, _ = self.ser_engine(data)
        result = result[0]      
        img_res = draw_ser_results(image_path, result)
        info = self.process_info(result)
        
        info_ = json.loads(info)
        product_rows = []
        for product in info_["PRODUCTS"]:
            product_row = [info_["SELLER"], info_["ADDRESS"], info_["STAFF"], info_["TIMESTAMP"], info_["CODE"],
                        product["PRODUCT"], product["NUMBER"], product["PRICE"], info_["TOTAL_COST"]]
            product_rows.append(product_row)

        # Create a Pandas DataFrame
        columns = ["SELLER", "ADDRESS", "STAFF", "TIMESTAMP", "CODE", "PRODUCT", "NUMBER", "PRICE", "TOTAL_COST"]
        df = pd.DataFrame(product_rows, columns=columns)

        # Create an Excel file
        path_save = './invoice.xlsx'
        try:
            excel = pd.read_excel(path_save)
            dataframe = pd.concat([excel, df])
        except:
            print('Created invoice.xlsx')
            dataframe = df
        dataframe.to_excel(path_save, index=False, engine="openpyxl")
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