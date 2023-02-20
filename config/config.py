import os

import torch


class Config(object):
    WEIGHTS_PATH = os.path.normpath("model/weights")
    RES_NET_MODEL = os.path.normpath("model/weights/vggface2-features.pt")
    ONET_MODEL = os.path.normpath("model/weights/onet.pt")
    PNET_MODEL = os.path.normpath("model/weights/pnet.pt")
    RNET_MODEL = os.path.normpath("model/weights/rnet.pt")
    DATA_PATH = os.path.normpath("data/vgg_face2_batch")
    FROAD_PROBA = 0.36

    # TODO() разобраться с памятью
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
