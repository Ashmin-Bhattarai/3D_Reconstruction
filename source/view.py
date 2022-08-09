import numpy as np
import cv2
import os


class View:
    def __init__(self, dataset_path:'str', image_path:'str', image_name:'str') -> None:
        self.name = image_name[0:-4]    # remove extension
        self.image_path = image_path
        self.dataset_path = dataset_path
        self.image = cv2.imread(os.path.join(image_path, image_name))
        self.keypoints = []
        self.descriptors = []

        

def create_views(dataset_path:'str') -> 'list[View]':
    views = []
    image_dir = os.path.join(dataset_path, 'images')
    print(image_dir)
    image_names = os.listdir(image_dir)
    for image_name in image_names:
        view = View(dataset_path, image_dir, image_name)
        views.append(view)
    return np.array(views)