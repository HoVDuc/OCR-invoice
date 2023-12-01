from src.ppocr.ppocr.utils.visual import draw_ser_results
from src.ppocr.tools.infer_kie_token_ser_test import *
from argparse import ArgumentParser
import os
import sys
import json
import pandas as pd
import gradio as gr
__dir__ = os.path.join('src', 'ppocr')
sys.path.append(__dir__)
sys.path.append('./src/')


class Inference:

    def __init__(self, otp) -> None:
        self.config = main(otp)
        self.ser_engine = SerPredictor(self.config)

    def process_info(self, results):
        info = {
            'SELLER': '',
            'ADDRESS': '',
            'TIMESTAMP': '',
            'PRODUCTS': [],
            'TOTAL_COST': 0
        }

        current_product = {
            'PRODUCT': '',
            'NUM': 0,
            'VALUE': 0
        }

        products = []
        print(results)
        for result in results:
            label = result['pred']
            transcription = result['transcription']

            if label != 'O':
                if label in ['PRODUCT', 'NUM', 'VALUE']:
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
            infer_imgs = get_image_file_list(
                self.config['Global']['infer_img'])
        except:
            infer_imgs = self.config['Global']['infer_img']

        img_path = infer_imgs
        data = {'img_path': img_path}

        result, _ = self.ser_engine(data)
        result = result[0]
        img_res = draw_ser_results(image_path, result)
        info = self.process_info(result)

        # info_ = json.loads(info)
        # product_rows = []
        # for product in info_["PRODUCTS"]:
        #     product_row = [info_["SELLER"], info_["ADDRESS"], info_["STAFF"], info_["TIMESTAMP"],
        #                    product["PRODUCT"], product["NUMBER"], product["PRICE"], info_["TOTAL_COST"]]
        #     product_rows.append(product_row)

        # # Create a Pandas DataFrame
        # columns = ["SELLER", "ADDRESS", "STAFF", "TIMESTAMP", 
        #            "PRODUCT", "NUMBER", "PRICE", "TOTAL_COST"]
        # df = pd.DataFrame(product_rows, columns=columns)

        # # Create an Excel file
        # path_save = './invoice.xlsx'
        # try:
        #     excel = pd.read_excel(path_save)
        #     dataframe = pd.concat([excel, df])
        # except:
        #     print('Created invoice.xlsx')
        #     dataframe = df
        # dataframe.to_excel(path_save, index=False, engine="openpyxl")
        return img_res, info


def GUI():
    otp = {
        'config': args.config,
        'otp': {
            'Architecture.Backbone.checkpoints': args.checkpoint,
            'Global.use_gpu': args.use_gpu,
            'Global.det_limit_type': args.det_limit_type
        }
    }
    infer = Inference(otp)
    demo = gr.Interface(
        fn=infer,
        inputs=[gr.Image()],
        outputs=["image", "text"]
    )

    demo.launch(share=args.share_link)


if __name__ == "__main__":
    parse = ArgumentParser()
    parse.add_argument('--config', type=str, default='')
    parse.add_argument('--checkpoint', type=str, default='')
    parse.add_argument('--det_limit_type', type=str, default='max')
    parse.add_argument('--share_link', action='store_true', default=False)
    parse.add_argument('--use_gpu', action='store_true', default=False)
    args = parse.parse_args()
    GUI()