import torchvision

from config.config import Config
from model import res_net

if __name__ == '__main__':
    
    model = res_net.InceptionResnetV1(
        pretrained = True,
        dropout_prob=0.6, 
        device=None
    )
    model.eval()
    
    photo_path = "data/dima/photo_2023-02-12_16-10-37.jpg"

    photo = torchvision.io.read_image(photo_path)
    embanding = model.forward(photo)

    print(embanding)
