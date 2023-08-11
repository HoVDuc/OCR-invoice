import sys
import os
import lmdb
import cv2
import numpy as np
from tqdm import tqdm

def checkImageIsValid(image_bin):
    """_summary_
    Kiểm tra ảnh có hợp lệ hay không
    Args:
        image_bin (_type_): Image

    Returns:
        _type_: _description_
    """
    
    # Initianizer
    isvalid = True
    imgH = None
    imgW = None
    
    # Convert image from string binary to uint8
    imageBuf = np.fromstring(image_bin, dtype=np.uint8)
    try:
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE) # Read and decode image to grayscale
        
        imgH, imgW = img.shape # Get height and width of Image
        # Check shape of image is valid 
        if imgH * imgW == 0: 
            isvalid = False
    except Exception as e:
        isvalid = False
        
    return isvalid, imgH, imgW
        
def writeCache(env, cache):
    """_summary_
    Writes content of a dictionary to a database
    Args:
        env (_type_): Enviroment object
        cache (_type_): dictionary
    """
    
    with env.begin(write=True) as txn: #inited database, write equal True enable writes operators
        for k, v in cache.items(): # Get key and value
            txn.put(k.encode(), v) # Put key and value to database

def createDataset(output_path, root_dir, annotation_path):
    """
    Create LMDB dataset for CNN training
    Args:
        output_path (_type_): LMDB output path
        root_dir (_type_): root directory path
        annotation_path (_type_): _description_
    """
    # Read file 
    annotation_path = os.path.join(root_dir, annotation_path)
    with open(annotation_path, 'r') as ann_file:
        lines = ann_file.readlines()
        annotations = [l.strip().split('\t') for l in lines]
    
    # Số lượng data
    n_samples = len(annotations)
    env = lmdb.open(output_path, map_size=1099511627776) # Create a enviroment
    cache = {}
    cnt = 0
    error = 0
    
    process_bar = tqdm(range(n_samples), ncols=100, desc="Create {}".format(output_path))
    for i in process_bar:
        #Get name and label of image file
        try:
            image_file, label = annotations[i]
        except ValueError as e:
            print(annotations[i]) 
        # Get image path
        image_path = os.path.join(root_dir, image_file)
        
        # Check image exist and get num error
        if not os.path.exists(image_path):
            error += 1
            continue
        
        # Open and check image is valid or not
        with open(image_path, 'rb') as f:
            image_bin = f.read()
        isvalid, imgH, imgW = checkImageIsValid(image_bin)
        
        if not isvalid:
            error += 1
            continue
        
        image_key = 'image-%09d' % cnt
        label_key = 'label-%09d' % cnt
        path_key = 'path-%09d' % cnt
        dim_key = 'dim-%09d' % cnt
        
        # Set data for dictionary
        cache[image_key] = image_bin
        cache[label_key] = label.encode()
        cache[path_key] = image_file.encode()
        cache[dim_key] = np.array([imgH, imgW], dtype=np.int32).tobytes()
        
        cnt += 1
        
        # Each cnt equal 1000 put cache to database
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
    
    n_samples = cnt - 1 
    cache['num-samples'] = str(n_samples).encode()
    writeCache(env, cache)
    
    if error > 0:
        print('Remove {} invalid images'.format(error))
    print('Created dataset with %d samples' % n_samples)
    sys.stdout.flush()