import torch
import numpy as np
import math
from PIL import Image
from torch.nn.functional import log_softmax, softmax

from ocr.model.transformerocr import VietOCR
from ocr.model.vocab import Vocab
from ocr.model.beam import Beam

def batch_translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: NxCxHxW
    model.eval()
    device = img.device
    sents = []

    with torch.no_grad():
        src = model.cnn(img)
        memories = model.transformer.forward_encoder(src)
        for i in range(src.size(0)):
#            memory = memories[:,i,:].repeat(1, beam_size, 1) # TxNxE
            memory = model.transformer.get_memory(memories, i)
            sent = beamsearch(memory, model, device, beam_size, candidates, max_seq_length, sos_token, eos_token)
            sents.append(sent)

    sents = np.asarray(sents)

    return sents

def beamsearch(memory, model, device, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):    
    # memory: Tx1xE
    model.eval()

    beam = Beam(beam_size=beam_size, min_length=0, n_top=candidates, ranker=None, start_token_id=sos_token, end_token_id=eos_token)

    with torch.no_grad():
#        memory = memory.repeat(1, beam_size, 1) # TxNxE
        memory = model.transformer.expand_memory(memory, beam_size)

        for _ in range(max_seq_length):
            
            tgt_inp = beam.get_current_state().transpose(0,1).to(device)  # TxN
            decoder_outputs, memory = model.transformer.forward_decoder(tgt_inp, memory)

            log_prob = log_softmax(decoder_outputs[:,-1, :].squeeze(0), dim=-1)
            beam.advance(log_prob.cpu())
            
            if beam.done():
                break
                
        scores, ks = beam.sort_finished(minimum=1)

        hypothesises = []
        for i, (times, k) in enumerate(ks[:candidates]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)
    
    return [1] + [int(i) for i in hypothesises[0][:-1]]

def translate_beam_search(img, model, beam_size=4, candidates=1, max_seq_length=128, sos_token=1, eos_token=2):
    # img: 1xCxHxW
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src) #TxNxE
        sent = beamsearch(memory, model, device, beam_size, candidates, max_seq_length, sos_token, eos_token)

    return sent

def translate(img, model, max_seq_length=128, sos_token=1, eos_token=2):
    """_summary_

    Args:
        img (torch): Ảnh đã được xử lý
        model (_type_): Mô hình dùng để đánh giá
        max_seq_length (int, optional): _description_. Defaults to 128.
        sos_token (int, optional): _description_. Defaults to 1.
        eos_token (int, optional): _description_. Defaults to 2.
    """
    "data: BxCxHxW"
    model.eval()
    device = img.device

    with torch.no_grad():
        src = model.cnn(img)
        memory = model.transformer.forward_encoder(src)

        translated_sentence = [[sos_token]*len(img)]
        char_probs = [[1]*len(img)]

        max_length = 0

        while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):

            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            
#            output = model(img, tgt_inp, tgt_key_padding_mask=None)
#            output = model.transformer(src, tgt_inp, tgt_key_padding_mask=None)
            output, memory = model.transformer.forward_decoder(tgt_inp, memory)
            output = softmax(output, dim=-1)
            output = output.to('cpu')

            values, indices  = torch.topk(output, 5)
            
            indices = indices[:, -1, 0]
            indices = indices.tolist()
            
            values = values[:, -1, 0]
            values = values.tolist()
            char_probs.append(values)

            translated_sentence.append(indices)   
            max_length += 1

            del output

        translated_sentence = np.asarray(translated_sentence).T
        
        char_probs = np.asarray(char_probs).T
        char_probs = np.multiply(char_probs, translated_sentence>3)
        char_probs = np.sum(char_probs, axis=-1)/(char_probs>0).sum(-1)
    
    return translated_sentence, char_probs


def build_model(config):
    vocab = Vocab(config['vocab']) #Load bộ vocab từ config
    device = config['device'] #Load cuda hoặc cpu
    
    model = VietOCR(len(vocab),
            config['backbone'],
            config['cnn'], 
            config['transformer'],
            config['seq_modeling']) #Load model VietOCR
    
    model = model.to(device) 

    return model, vocab 

def resize(w, h, expected_height, image_min_width, image_max_width):
    # Tính new width dựa vào expected height
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    # Làm tròn lên và nhân với round_to
    new_w = math.ceil(new_w/round_to)*round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height


def process_image(image, image_height, image_min_width, image_max_width):
    """_summary_

    Args:
        image (Image): ảnh đầu cần predict
        image_height (int): Chiều cao của ảnh
        image_min_width (int): Chiều rộng tối thiểu của ảnh
        image_max_width (_type_): Chiều rộng tối đa của ảnh

    Returns:
        np.array: Ảnh được xử lý
    """
    img = image.convert('RGB')  # Đưa về màu RGB

    w, h = img.size
    # Resize lại chiều rộng ảnh
    new_w, image_height = resize(
        w, h, image_height, image_min_width, image_max_width)

    # Resize lại ảnh dựa vào chiều rộng mới và image cố định, với kernel ANTIAIAS
    img = img.resize((new_w, image_height), Image.ANTIALIAS)

    # Đưa về chiều của pytorch (,c, h, w)
    img = np.asarray(img).transpose(2, 0, 1)
    img = img/255  # scale lại ảnh
    return img

def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image(image, image_height, image_min_width, image_max_width)
    img = img[np.newaxis, ...] # Thêm một chiều mới vào ảnh
    img = torch.FloatTensor(img) # Convert ảnh từ numpy sang torch float
    return img

def predict(filename, config):
    """_summary_

    Args:
        filename (str): Đường dẫn của ảnh
        config (_type_):

    Returns:
        str: Kết quả dự đoán
    """
    img = Image.open(filename) # Load ảnh
    img = process_input(img) #Xử lý đầu vào

    img = img.to(config['device'])

    model, vocab = build_model(config) # Load model và bộ vocab
    s = translate(img, model)[0].tolist()
    s = vocab.decode(s)
    
    return s
