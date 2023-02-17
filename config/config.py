import os

import torch

class Config(object):

    WEIGHTS_PATH = os.path.normpath('model/weights')
    MODEL_NAME = 'vggface2'
    
    # TODO() разобраться с памятью
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"