import os
import cv2
import random
import numpy as np
import torch
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *
from PIL import Image


class Match:
    def __init__(self, view1:'View', view2:'View') -> None:
        self.image_name1 = view1.name
        self.image_name2 = view2.name

        self.image1 = view1.image
        self.image2 = view2.image

        self.dataset_path = view1.dataset_path

        self.view1 = view1
        self.view2 = view2

        self.indices1 = []
        self.indices2 = []

        self.K = []
        self.inliers = []

        self.device = torch.device('cpu')
        self.matcher = KF.LoFTR(pretrained='outdoor')
        self.matcher.to(self.device).eval()

        print(f"=========Matching {self.image_name1} and {self.image_name2}=========")
        self.get_matches()
        print(f"=========Done matching {self.image_name1} and {self.image_name2}=========")
        self.draw_matches()
        print(f"=========Done drawing matches for {self.image_name1} and {self.image_name2}=========")



    def get_matches(self) -> None:
        img1 = self.load_torch_images(self.image1)
        img2 = self.load_torch_images(self.image2)

        input_dict = {"image0": K.color.rgb_to_grayscale(img1),
                      "image1": K.color.rgb_to_grayscale(img2)}

        with torch.no_grad():
            output_dict = self.matcher(input_dict)

        self.indices1 = output_dict["keypoints0"].cpu().numpy()
        self.indices2 = output_dict["keypoints1"].cpu().numpy()

        
        if len(self.indices1) > 7:
            self.K, self.inliers = F, inliers = cv2.findFundamentalMat(self.indices1, self.indices2, cv2.USAC_MAGSAC, 0.1845, 0.999999, 220000)
            self.inliers = self.inliers > 0
        else:
            self.K = np.zeros((3, 3))
            self.inliers = np.zeros(len(self.indices1))
            
    
    def rescale_image(self, image:'np.array') -> 'np.array':
        self.scale = 840/max(image.shape[0], image.shape[1])
        self.w = int(image.shape[1] * self.scale)
        self.h = int(image.shape[0] * self.scale)
        image = cv2.resize(image, (self.w, self.h))
        return image

    def load_torch_images(self, image:'np.array') -> 'K.Tensor':
        image = self.rescale_image(image)
        image = K.image_to_tensor(image, False).float() /255
        image = K.color.bgr_to_rgb(image)
        image = image.to(self.device)
        return image

    def draw_matches(self)->None:
        img1 = self.rescale_image(self.image1)
        img2 = self.rescale_image(self.image2)
        concatImg = np.concatenate((img1, img2), axis=1)


        for (p1, p2) in random.sample(list(zip(self.indices1, self.indices2)), 50):
            starting_point = (int(p1[0]), int(p1[1]))
            ending_point = (int(p2[0] + img1.shape[1]), int(p2[1]))
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
        