from typing import List
import os

import cv2
import numpy as np
import torch

from config import config

config = config.Config()


def read_photo(photo_path: str):
    photo = cv2.imread(photo_path)
    # photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)

    return photo


# TODO() описать переменные функций
def cropp_photo(photo: np.ndarray, face_detect_model) -> torch.tensor:
    boxes, _ = face_detect_model.detect(photo)

    start_point = (int(boxes[0][0]), int(boxes[0][1]))
    end_point = (int(boxes[0][2]), int(boxes[0][3]))

    cropped_photo = (
        torch.from_numpy(
            photo[
                start_point[1] : end_point[1],  # y1:y2
                start_point[0] : end_point[0],  # x1:x2
            ]
        )
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(config.DEVICE)
    )

    return cropped_photo

class ChoosePersons():

    def __init__(
        self, 
        photo_path: str,
        persons_size: int = 25,
        photo_size: int = 15,

    ):
        self.photo_path = photo_path    
        # numpy array could't has more 32 dimentions.
        # So if we need to specify probability below /
        # we need to choose not more than 32 persons (size)
        self.persons = np.random.choice(
            os.listdir(self.photo_path),
            size=persons_size,
            replace=False           # to choose without /
        )                           # replacement

        self.person_photos = {
            person: np.random.choice(
                os.listdir(
                    os.path.join(
                        self.photo_path,
                        person
                    )
                ),
                size=photo_size
            )
            for person in self.persons
        }

    def random_choose_person_photos(self):

        # choose persons for test
        person1 = np.random.choice(self.persons)
        
        # rise the opportunity choosing person1
        # for other persons probability will calculate /
        # as froad probabilyty divide on len persons
        
        proba_persons = [
            (1 - config.FROAD_PROBA) if x == person1
            else config.FROAD_PROBA / (len(self.persons)-1)
            for x in self.persons
        ]
        
        person2 = np.random.choice(
            a=self.persons, 
            p=proba_persons
        )

        if person1 == person2:
            froad = 0
        else:
            froad = 1

        # get the pathes for each person
        person1_path = os.path.join(
            self.photo_path,
            person1
        )
        person2_path = os.path.join(
            self.photo_path,
            person2
        )

        # random choose the photo of each person
        photo1 = np.random.choice(
            self.person_photos[person1]      # choose random
        )
        photo2 = np.random.choice(
            self.person_photos[person2]      # choose random
        )

        person1_photo_path = os.path.join(
            person1_path,
            photo1
        )
        person2_photo_path = os.path.join(
            person2_path,
            photo2
        )

        return person1_photo_path, person2_photo_path
