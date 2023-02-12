import torchvision

from config.config import Config
from model import res_net

from config.config import Config

if __name__ == '__main__':

    config = Config()
    
    model = res_net.InceptionResnetV1(
        pretrained = True,
        dropout_prob=0.6, 
        device=config.DEVICE
    )
    model.eval().to(config.DEVICE)
    
    photo_path = "data/dima/"
    photo1 = photo_path+"photo3.jpg"
    photo2 = photo_path+"photo4.jpg"


    photo1 = torchvision.io.read_image(photo1)
    photo1 = photo1.float().unsqueeze(0).to(config.DEVICE)

    photo2 = torchvision.io.read_image(photo2)
    photo2 = photo2.float().unsqueeze(0).to(config.DEVICE)

    embanding1 = model.forward(photo1)
    embanding2 = model.forward(photo2)

    dists = [[(e1 - e2).norm().item() for e2 in embanding1] for e1 in embanding2]

    print(dists)
