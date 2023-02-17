import torchvision
import torch
import cv2

from config.config import Config
from model import res_net, mtcnn

config = Config()

res_net = res_net.InceptionResnetV1(
        pretrained = True,
        dropout_prob=0.6, 
        device=config.DEVICE
    )
res_net.eval().to(config.DEVICE)

mtcnn = mtcnn.MTCNN(
    keep_all=True, 
    device=config.DEVICE
)

if __name__ == '__main__':

    photo_path = "data/dima/"
    photo1 = cv2.imread(photo_path+"photo1.jpg")
    photo2 = cv2.imread(photo_path+"photo2.jpg", cv2.IMREAD_GRAYSCALE) 

    photo1 = cv2.cvtColor(photo1, cv2.COLOR_BGR2RGB)
    photo2 = cv2.cvtColor(photo2, cv2.COLOR_BGR2RGB)

    # Draw the face detection annotations on the image.
    photo1 = cv2.cvtColor(photo1, cv2.COLOR_RGB2BGR)
    boxes1, _ = mtcnn.detect(photo1)
    
    # represents the top left corner of rectangle
    start_point = (int(boxes1[0][0]), int(boxes1[0][1]))
    # represents the bottom right corner of rectangle
    end_point = (int(boxes1[0][2]), int(boxes1[0][3]))

    cropped_photo1 = torch.from_numpy(photo1[
        start_point[1]:end_point[1],    #y1:y2
        start_point[0]:end_point[0]     #x1:x2
    ])

     # Draw the face detection annotations on the image.
    photo2 = cv2.cvtColor(photo2, cv2.COLOR_RGB2BGR)
    
    boxes2, _ = mtcnn.detect(photo2)
    
    # represents the top left corner of rectangle
    start_point = (int(boxes2[0][0]), int(boxes2[0][1]))
    # represents the bottom right corner of rectangle
    end_point = (int(boxes2[0][2]), int(boxes2[0][3]))
    
    cropped_photo2 = torch.from_numpy(photo2[
        start_point[1]:end_point[1],    #y1:y2
        start_point[0]:end_point[0]     #x1:x2
    ])

    with torch.no_grad():
        embanding1 = res_net.forward(cropped_photo1)
        embanding2 = res_net.forward(cropped_photo2)                                              

    print(torch.cdist(embanding1, embanding2))
