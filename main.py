import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from config.config import Config
from model import mtcnn, res_net
from utils import utils

config = Config()

res_net = (
    res_net.InceptionResnetV1(pretrained=True, dropout_prob=0.6)
    .eval()
    .to(config.DEVICE)
)

mtcnn = mtcnn.MTCNN(keep_all=True).eval().to(config.DEVICE)

# extract photos for each person in data folders
photo_path = config.DATA_PATH
choose_photos = utils.ChoosePersons(photo_path, persons_size=25, photo_size=15)

if __name__ == "__main__":
    results = {}
    for epoche in tqdm(range(10000)):
        (
            person1_photo_path,
            person2_photo_path,
        ) = choose_photos.random_choose_person_photos()

        # it doesn't make sense to compare the same photos
        if person1_photo_path == person2_photo_path:
            continue

        # read and crop photos
        person1_photo = utils.read_photo(person1_photo_path)
        try:
            cropped_photo1 = utils.cropp_photo(person1_photo, mtcnn)
        except TypeError:
            cropped_photo1 = "face not recognized"

        person2_photo = utils.read_photo(person2_photo_path)
        try:
            cropped_photo2 = utils.cropp_photo(person2_photo, mtcnn)
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
        with torch.no_grad():
            try:
                embanding1 = res_net.forward(cropped_photo1)
                embanding2 = res_net.forward(cropped_photo2)
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
        results[epoche] = {
            "person_1": person1,
            "person_1_photo": photo1,
            "person_2": person2,
            "person_2_photo": photo2,
            "results": {
                "euclidean": (
                    torch.cdist(embanding1, embanding2).numpy(  # compute distances
                        force=True
                    )[0][0]
                )
            },
            "froad": froad,
        }

    # save the experiment results to file
    with open("data/experiments/res_net_mtcnn_vgg__exp1.pkl", "wb") as f:
        pickle.dump(results, f)
