import os
import pickle

import numpy as np
import torch
from tqdm import tqdm
import face_recognition

from config.config import Config
import utils.utils

config = Config()

# extract photos for each person in data folders
photo_path = config.DATA_PATH
choose_photos = utils.ChoosePersons(photo_path, persons_size=25, photo_size=15)

if __name__ == "__main__":
    results = {}
    for epoche in tqdm(range(10000)):
        person1, person2, photo1, photo2, person1_photo_path, person2_photo_path, froad = (
            choose_photos.random_choose_person_photos()
        )

        # it doesn't make sense to compare the same photos
        if person1_photo_path == person2_photo_path:
            continue

        # read and crop photos
        try:
            cropped_photo1 = face_recognition.load_image_file(person1_photo_path)
        except TypeError:
            cropped_photo1 = "face not recognized"

        try:
            cropped_photo2 = face_recognition.load_image_file(person2_photo_path)
        except TypeError:
            cropped_photo2 = "face not recognized"

        if (
            cropped_photo1 == "face not recognized"
            or cropped_photo2 == "face not recognized"
        ):
            results[epoche] = {
                "person_1": person1,
                "person_1_photo": photo1,
                "person_2": person2,
                "person_2_photo": photo2,
                "results": "face not recognized",
                "froad": froad,
            }

        # compute embandings
        # with torch.no_grad():
        try:
            embanding1 = face_recognition.face_encodings(cropped_photo1)
            embanding2 = face_recognition.face_encodings(cropped_photo2)
        except RuntimeError:
            results[epoche] = {
                "person_1": person1,
                "person_1_photo": photo1,
                "person_2": person2,
                "person_2_photo": photo2,
                "results": "RuntimeError",
                "froad": froad,
            }
            continue
        except TypeError:
            results[epoche] = {
                "person_1": person1,
                "person_1_photo": photo1,
                "person_2": person2,
                "person_2_photo": photo2,
                "results": "TypeError",
                "froad": froad,
            }
            continue
        except IndexError:
            print(embanding1, embanding2)
        results[epoche] = {
            "person_1": person1,
            "person_1_photo": photo1,
            "person_2": person2,
            "person_2_photo": photo2,
            "results": {
                "euclidean": (
                    np.sum([(x - y)**2 for x,y in zip(embanding1, embanding2)])
                )
            },
            "froad": froad,
        }

    # save the experiment results to file
    with open("data/experiments/face_recognition__exp1.pkl", "wb") as f:
        pickle.dump(results, f)
