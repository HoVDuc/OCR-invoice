# Load libraries
from ocr.tools.create_dataset import createDataset
from ocr.tools.translate import resize
from ocr.tools.translate import process_image
import torch
# classes are used to specify the sequence of indices/keys used in data loading.
# This module implements specialized container datatypes providing alternatives to Python’s general purpose built-in containers, dict, list, set, and tuple.
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from tqdm import tqdm
# dict subclass that calls a factory function to supply missing values
from collections import defaultdict
import sys
import os
import random
import lmdb
import six
import numpy as np

from PIL import Image
from PIL import ImageFile
# Whether or not to load truncated image files.
ImageFile.LOAD_TRUNCATED_IMAGES = True


class OCRDataset(Dataset):
    
    def __init__(self, lmdb_path, root_dir, annotation_path, vocab, image_height=32, image_min_width=32, image_max_width=512, transform=None) -> None:
        self.root_dir = root_dir
        self.annotation_path = annotation_path
        self.vocab = vocab
        self.transform = transform

        self.image_height = image_height
        self.image_min_width = image_min_width
        self.image_max_width = image_max_width

        self.lmdb_path = lmdb_path

        # Check file is exist or not
        if os.path.isdir(self.lmdb_path):
            print('{} exists. Remove folder if you want to create new dataset'.format(
                self.lmdb_path))
            sys.stdout.flush() # Used to flush the output buffer associated with the standard output stream.
        else:
            createDataset(self.lmdb_path, root_dir, annotation_path)

        # Open envrioment
        self.env = lmdb.open(self.lmdb_path, max_readers=8,
                             readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

        # Get num samples
        n_samples = int(self.txn.get('num-samples'.encode()))
        self.n_samples = n_samples

        self.build_cluster_indices()

    def build_cluster_indices(self):
        # Create cluster indices
        self.cluster_indices = defaultdict(list)

        pbar = tqdm(range(self.__len__()), desc='{} build cluster'.format(
            self.lmdb_path), ncols=100, position=0, leave=True)
        for i in pbar:
            bucket = self.get_bucket(i)
            self.cluster_indices[bucket].append(i)

    def get_bucket(self, idx):
        # Create a key with used index nine digits and prefix is dim 
        key = 'dim-%09d' % idx

        # Retrival info encoded in database based key
        dim_img = self.txn.get(key.encode())
        # Convert info type from string to int32
        dim_img = np.fromstring(dim_img, dtype=np.int32)
        
        # Get Height and width of image
        imgH, imgW = dim_img

        # Get new width
        new_w, image_height = resize(
            imgW, imgH, self.image_height, self.image_min_width, self.image_max_width)
        return new_w

    def read_buffer(self, idx):
        # Get key correspoding with data fileds in database
        image_file = 'image-%09d' % idx
        label_file = 'label-%09d' % idx
        path_file = 'path-%09d' % idx

        # Get image buffed
        imgbuf = self.txn.get(image_file.encode())

        # Get label
        label = self.txn.get(label_file.encode()).decode()
        
        # Get image path
        img_path = self.txn.get(path_file.encode()).decode()

        # Write image to buf, buf is Byte
        buf = six.BytesIO()
        buf.write(imgbuf)
        # Reset the read/write position to the beginning of the buffer.
        buf.seek(0)

        return buf, label, img_path

    def read_data(self, idx):
        # Read data to Database
        buf, label, img_path = self.read_buffer(idx)

        # Load image to buf
        img = Image.open(buf).convert('RGB')
        
        # Using transform
        if self.transform:
            img = self.transform(img)

        # Processing image 
        img_bw = process_image(img, self.image_height,
                               self.image_min_width, self.image_max_width)

        # Encode label
        word = self.vocab.encode(label)
        return img_bw, word, img_path

    def __getitem__(self, index):
        """
        In Python, __getitem__ is a special method that is used to define how an object behaves when an item is
        accessed using square brackets []. It allows objects to be accessed like sequences, dictionaries, or other
        containers.
        """
        img, word, img_path = self.read_data(index)

        img_path = os.path.join(self.root_dir, img_path)

        sample = {
            'img': img,
            'word': word,
            'img_path': img_path
        }
        return sample

    def __len__(self):
        return self.n_samples

class ClusterRandomSampler(Sampler):
    
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle        

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):
        batch_lists = []
        for cluster, cluster_indices in self.data_source.cluster_indices.items():
            if self.shuffle:
                random.shuffle(cluster_indices)

            batches = [cluster_indices[i:i + self.batch_size] for i in range(0, len(cluster_indices), self.batch_size)]
            batches = [_ for _ in batches if len(_) == self.batch_size]
            if self.shuffle:
                random.shuffle(batches)

            batch_lists.append(batches)

        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)

        lst = self.flatten_list(lst)

        return iter(lst)

    def __len__(self):
        return len(self.data_source)
    
class Collator(object):
    def __init__(self, masked_language_model=True):
        self.masked_language_model = masked_language_model

    def __call__(self, batch):
        filenames = []
        img = []
        target_weights = []
        tgt_input = []
        max_label_len = max(len(sample['word']) for sample in batch)
        for sample in batch:
            img.append(sample['img'])
            filenames.append(sample['img_path'])
            label = sample['word']
            label_len = len(label)
            
            
            tgt = np.concatenate((
                label,
                np.zeros(max_label_len - label_len, dtype=np.int32)))
            tgt_input.append(tgt)

            one_mask_len = label_len - 1

            target_weights.append(np.concatenate((
                np.ones(one_mask_len, dtype=np.float32),
                np.zeros(max_label_len - one_mask_len,dtype=np.float32))))
            
        img = np.array(img, dtype=np.float32)


        tgt_input = np.array(tgt_input, dtype=np.int64).T
        tgt_output = np.roll(tgt_input, -1, 0).T
        tgt_output[:, -1]=0
        
        # random mask token
        if self.masked_language_model:
            mask = np.random.random(size=tgt_input.shape) < 0.05
            mask = mask & (tgt_input != 0) & (tgt_input != 1) & (tgt_input != 2)
            tgt_input[mask] = 3

        tgt_padding_mask = np.array(target_weights)==0

        rs = {
            'img': torch.FloatTensor(img),
            'tgt_input': torch.LongTensor(tgt_input),
            'tgt_output': torch.LongTensor(tgt_output),
            'tgt_padding_mask': torch.BoolTensor(tgt_padding_mask),
            'filenames': filenames
        }   
        
        return rs
