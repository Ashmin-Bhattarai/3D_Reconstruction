import numpy as np
import pandas as pd
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
        
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

        self.K = np.zeros((3, 3))
        self.load_camera_parameters()

        self.scale = 840/max(self.image.shape[0], self.image.shape[1])
        self.scaled_height = int(self.image.shape[0] * self.scale)
        self.scaled_width = int(self.image.shape[1] * self.scale)
        self.scaled_image = cv2.resize(self.image, (self.scaled_width, self.scaled_height))


    def load_camera_parameters(self):
        data = pd.read_csv(os.path.join(self.dataset_path, "calibration.csv"))
        df=data.loc[data['image_id']==self.name,'camera_intrinsics']
        self.K = np.array([i for i in df.values[0].split()], dtype=np.float32).reshape(3,3)




    
        

def create_views(dataset_path:'str') -> 'list[View]':
    views = []
    image_dir = os.path.join(dataset_path, 'images')
    print(image_dir)
    image_names = os.listdir(image_dir)
    for image_name in image_names:
        view = View(dataset_path, image_dir, image_name)
        views.append(view)
    return np.array(views)