import os
import cv2
import random
import pickle
import numpy as np
import torch
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *
from PIL import Image
from view import View


class Match:
    def __init__(self, view1:'View', view2:'View') -> None:
        self.image_name1 = view1.name
        self.image_name2 = view2.name

        self.image1 = view1.scaled_image
        self.image2 = view2.scaled_image

        self.dataset_path = view1.dataset_path

        self.view1 = view1
        self.view2 = view2

        self.indices1 = []
        self.indices2 = []

        self.scaled_indices1 = []
        self.scaled_indices2 = []

        self.F = np.zeros((3, 3))
        self.E = np.zeros((3, 3))
        self.mask = []


        self.device = torch.device('cpu')
        self.matcher = KF.LoFTR(pretrained='outdoor')
        self.matcher.to(self.device).eval()



        if not os.path.exists(os.path.join(self.dataset_path, "matches")):
            os.makedirs(os.path.join(self.dataset_path, "matches"))
            os.makedirs(os.path.join(self.dataset_path, "features"))

        if not os.path.exists(os.path.join(self.dataset_path, "features", f"{self.image_name1}-{self.image_name2}.pkl")):
            print(f"\n=========Matching {self.image_name1} and {self.image_name2}=========")
            self.get_matches()
            print(f"=========Done matching {self.image_name1} and {self.image_name2}=========")
            self.store_data()
            print(f"=========Done Storing {self.image_name1}-{self.image_name2}.pkl==========")
            self.draw_matches()
            print(f"=========Done drawing matches for {self.image_name1} and {self.image_name2}=========")

        else:
            self.load_data()
            print(f"\n=========Loaded {self.image_name1}-{self.image_name2}.pkl==========")


    def get_matches(self) -> None:
        img1 = self.load_torch_images(self.image1)
        img2 = self.load_torch_images(self.image2)

        input_dict = {"image0": K.color.rgb_to_grayscale(img1),
                      "image1": K.color.rgb_to_grayscale(img2)}

        with torch.no_grad():
            output_dict = self.matcher(input_dict)

        self.scaled_indices1 = output_dict["keypoints0"].cpu().numpy()
        self.scaled_indices2 = output_dict["keypoints1"].cpu().numpy()

        self.indices1 = self.scaled_indices1 // self.view1.scale
        self.indices2 = self.scaled_indices2 // self.view2.scale


        if len(self.indices1) > 7:
            self.F, self.mask = cv2.findFundamentalMat(self.scaled_indices1, self.scaled_indices2, cv2.USAC_MAGSAC, 0.1845, 0.999999, 220000)
            self.mask = self.mask > 0
            self.E = self.view2.K.T @ self.F @ self.view1.K
            print(">>>>>>>>>Number of inliers: ", self.number_of_inliers())
        else:
            self.K = np.zeros((3, 3))
            self.E = np.zeros((3, 3))
            self.mask = np.zeros(len(self.indices1))
       
    
    def load_data(self) -> None:
        PIK = os.path.join(self.dataset_path, "features", f"{self.image_name1}-{self.image_name2}.pkl")
        with open(PIK, 'rb') as f:
            data = pickle.load(f)
        self.F = data[0]
        self.E = data[1]
        self.mask = data[2]
        self.indices1 = data[3]
        self.indices2 = data[4]
        self.scaled_indices1 = data[5]
        self.scaled_indices2 = data[6]

    def store_data(self) -> None:
        PIK = os.path.join(self.dataset_path, "features", f"{self.image_name1}-{self.image_name2}.pkl")
        data = [self.F, self.E, self.mask, self.indices1, self.indices2, self.scaled_indices1, self.scaled_indices2]
        with open(PIK, 'wb') as f:
            pickle.dump(data, f)
        
        
    def number_of_inliers(self) -> int:
        return np.sum(self.mask)
            
    
    # def rescale_image(self, image:'np.array') -> 'np.array':
    #     scale = 840/max(image.shape[0], image.shape[1])
    #     w = int(image.shape[1] * scale)
    #     h = int(image.shape[0] * scale)
    #     image = cv2.resize(image, (self.w, self.h))
    #     return image

    def load_torch_images(self, image:'np.array') -> 'K.Tensor':
        image = K.image_to_tensor(image, False).float() /255
        image = K.color.bgr_to_rgb(image)
        image = image.to(self.device)
        return image

    def draw_matches(self)->None:
        # img1 = self.rescale_image(self.image1)
        # img2 = self.rescale_image(self.image2)

        concatImg = np.zeros((max(self.image1.shape[0], self.image2.shape[0]), self.image1.shape[1] + self.image2.shape[1], 3), dtype=np.uint8) 
        concatImg[:, :] = (255, 255, 255)
        concatImg[:self.image1.shape[0], :self.image1.shape[1], :3] = self.image1
        concatImg[:self.image2.shape[0], self.image1.shape[1]:, :3] = self.image2

        # concatImg = np.concatenate((self.image1, self.image2), axis=1)

        for (p1, p2) in random.sample(list(zip(self.scaled_indices1, self.scaled_indices2)), 50):
            starting_point = (int(p1[0]), int(p1[1]))
            ending_point = (int(p2[0] + self.image1.shape[1]), int(p2[1]))
            cv2.circle(concatImg, starting_point, 5, (0, 255, 0), -1)
            cv2.circle(concatImg, ending_point, 5, (0, 255, 0), -1)
            cv2.line(concatImg, starting_point, ending_point, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(self.dataset_path, "matches", f"{self.image_name1}-{self.image_name2}.png"), concatImg)




def create_matches(views:'list[View]') -> 'dict[Match]':
    matches = {}
    for i in range(len(views)-1):
        for j in range(i+1, len(views)):
            matches[(views[i].name, views[j].name)] = Match(views[i], views[j])
    return matches
        