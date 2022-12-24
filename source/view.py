import glob
import os

import cv2
import numpy as np
import pandas as pd
from utils import read_data, write_data


class View:
    def __init__(self, dataset_path: "str", image_path: "str", image_name: "str") -> None:
        self.name = image_name[0:-4]  # remove extension
        self.image_path = image_path
        self.dataset_path = dataset_path

        self.image = cv2.imread(os.path.join(image_path, image_name))

        self.keypoints = []
        self.descriptors = []

        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

        self.K = np.zeros((3, 3))
        self.load_camera_parameters()

        # self.scale = 840/max(self.image.shape[0], self.image.shape[1])
        # self.scaled_height = int(self.image.shape[0] * self.scale)
        # self.scaled_width = int(self.image.shape[1] * self.scale)
        # self.scaled_image = cv2.resize(self.image, (self.scaled_width, self.scaled_height))

        self.R = np.zeros((3, 3), dtype=float)  # rotation matrix for the view
        self.t = np.zeros((3, 1), dtype=float)  # translation vector for the view

        # self.extract_features()
        if not (features_path := self.dataset_path / "features").exists():
            features_path.mkdir()

        if not (path := features_path / f"{self.name}.pkl").exists():
            self.extract_features()
            _data = [
                (point.pt, point.size, point.angle, point.response, point.octave, point.class_id, self.descriptors[idx])
                for idx, point in enumerate(self.keypoints)
            ]
            write_data(path, _data)
        else:
            keypoints = []
            descriptors = []

            for point in read_data(path):
                keypoint = cv2.KeyPoint(
                    x=point[0][0], y=point[0][1], size=point[1], angle=point[2], response=point[3], octave=point[4], class_id=point[5]
                )
                descriptor = point[6]
                keypoints.append(keypoint)
                descriptors.append(descriptor)

            self.keypoints = keypoints
            self.descriptors = np.array(descriptors)

    def extract_features(self):
        sift = cv2.SIFT_create()
        self.keypoints, self.descriptors = sift.detectAndCompute(self.image, None)

    def load_camera_parameters(self):
        data = pd.read_csv(os.path.join(self.dataset_path, "calibration.csv"))
        image_name = self.name.split("_", 1)

        if "baseline" in self.name:
            image_name = image_name[1]
        else:
            image_name = self.name

        df = data.loc[data["image_id"] == image_name, "camera_intrinsics"]
        self.K = np.array([i for i in df.values[0].split()], dtype=np.float32).reshape(3, 3)


def create_views(dataset_path: "str") -> "list[View]":
    views = []
    image_dir = os.path.join(dataset_path, "images")
    image_names1 = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))

    for image_name in image_names1:
        imn = image_name.split("/")[-1]
        view = View(dataset_path, image_dir, imn)
        views.append(view)

    return np.array(views)
