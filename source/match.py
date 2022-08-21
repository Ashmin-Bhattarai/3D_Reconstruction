import os
import cv2
import random
import pickle
import numpy as np
import torch
import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *
# from PIL import Image
from view import View

# match_technique = 'SIFT'
match_technique = 'LoFTR'

class Match_info:
    def __init__(self,queryIdx,trainIdx,confidence,pixel_points1,pixel_points2) -> None:
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.confidence = confidence
        self.pixel_points1= pixel_points1
        self.pixel_points2= pixel_points2


class Match:
    def __init__(self, view1:'View', view2:'View') -> None:
        self.image_name1 = view1.name
        self.image_name2 = view2.name

        self.dataset_path = view1.dataset_path

        self.view1 = view1
        self.view2 = view2

        self.pixel_points1 = []
        self.pixel_points2 = []

        self.indices1 = []
        self.indices2 = []
        
        self.inliers1 = []
        self.inliers2 = []

        self.scaled_pixel_points1 = []
        self.scaled_pixel_points2 = []

        self.F = np.zeros((3, 3))
        self.E = np.zeros((3, 3))
        self.mask = []

        self.matcher_SIFT = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        self.device = torch.device('cpu')
        self.matcher = KF.LoFTR(pretrained='outdoor')
        self.matcher.to(self.device).eval()
        self.matches=[]



        if not os.path.exists(os.path.join(self.dataset_path, "matches")):
            os.makedirs(os.path.join(self.dataset_path, "matches"))
            os.makedirs(os.path.join(self.dataset_path, "features"))

        if not os.path.exists(os.path.join(self.dataset_path, "features", f"{self.image_name1}-{self.image_name2}.pkl")):
            print(f"\n=========Matching {self.image_name1} and {self.image_name2}=========")
            if match_technique == 'SIFT':
                self.get_matches_SIFT()
            elif match_technique == 'LoFTR':
                self.get_matches()
            
            print(f"=========Done matching {self.image_name1} and {self.image_name2}=========")
            # self.store_data()
            print(f"=========Done Storing {self.image_name1}-{self.image_name2}.pkl==========")
            if match_technique == 'LoFTR':
                self.draw_matches()
            # self.draw_matches()
            print(f"=========Done drawing matches for {self.image_name1} and {self.image_name2}=========")

        else:
            self.load_data()
            print(f"\n=========Loaded {self.image_name1}-{self.image_name2}.pkl==========")

    def get_matches_SIFT(self):
        
        self.matches = self.matcher_SIFT.match(trainDescriptors=self.view1.descriptors, queryDescriptors=self.view2.descriptors)
        # matches = sorted(matches, key=lambda x: x.distance)
        trainidx= [m.trainIdx for m in self.matches]
        queryidx= [m.queryIdx for m in self.matches]
        print('max trainIdx:', max(trainidx))
        print('max queryIdx:', max(queryidx))
        print('len of keypoints 1:', len(self.view1.keypoints))
        print('len of keypoints 2:', len(self.view2.keypoints))
        print('len of matches:', len(self.matches))

        
        self.indices1 = [m.trainIdx for m in self.matches]
        self.indices2 = [m.queryIdx for m in self.matches]

        self.pixel_points1=np.array([key.pt for key in self.view1.keypoints])
        self.pixel_points2=np.array([key.pt for key in self.view2.keypoints])

        print(self.pixel_points1[0])

        if len(self.pixel_points1) > 7:
            self.F, self.mask = cv2.findFundamentalMat(np.array(self.pixel_points1)[self.indices1], np.array(self.pixel_points2)[self.indices2], method=cv2.FM_RANSAC,ransacReprojThreshold=0.9, confidence=0.99)
            self.mask = self.mask.astype(bool).flatten()
            self.inliers1 = np.array(self.indices1)[self.mask]
            self.inliers2 = np.array(self.indices2)[self.mask]
            self.E = self.view2.K.T @ self.F @ self.view1.K
            print(">>>>>>>>>Number of mask: ", self.number_of_inliers())
            print(">>>>>>>>>Number of inliers1: ", len(self.inliers1))
        else:
            self.K = np.zeros((3, 3))
            self.E = np.zeros((3, 3))
            self.mask = np.zeros(len(self.pixel_points1))




    def get_matches(self) -> None:
        img1 = self.load_torch_images(self.view1.scaled_image)
        img2 = self.load_torch_images(self.view2.scaled_image)

        input_dict = {"image0": K.color.rgb_to_grayscale(img1),
                      "image1": K.color.rgb_to_grayscale(img2)}

        with torch.no_grad():
            self.output_dict = self.matcher(input_dict)

        self.confidence = self.output_dict['confidence'].cpu().numpy()
        
        self.scaled_pixel_points1 = self.output_dict["keypoints0"].cpu().numpy()
        self.scaled_pixel_points2 = self.output_dict["keypoints1"].cpu().numpy()

        self.pixel_points1 = self.scaled_pixel_points1 // self.view1.scale
        self.pixel_points2 = self.scaled_pixel_points2 // self.view2.scale


        self.indices1=[i for i in range(len(self.pixel_points1))]
        self.indices2=[i for i in range(len(self.pixel_points2))]
        if len(self.pixel_points1) > 7:
            self.F, self.mask = cv2.findFundamentalMat(self.pixel_points1, self.pixel_points2, cv2.USAC_MAGSAC, 0.1845, 0.999999, 220000)
            self.mask = self.mask.astype(bool).flatten()
            self.inliers1 = np.array(self.indices1)[self.mask]
            self.inliers2 = np.array(self.indices2)[self.mask]
            self.E = self.view2.K.T @ self.F @ self.view1.K
            print(">>>>>>>>>Number of inliers: ", self.number_of_inliers())
        else:
            self.K = np.zeros((3, 3))
            self.E = np.zeros((3, 3))
            self.mask = np.zeros(len(self.pixel_points1))

        for i in range(len(self.pixel_points1)):
            self.matches.append(Match_info(self.indices1[i], self.indices2[i], self.confidence[i],self.pixel_points1[i], self.pixel_points2[i]))

            
    
    def load_data(self) -> None:
        PIK = os.path.join(self.dataset_path, "features", f"{self.image_name1}-{self.image_name2}.pkl")
        with open(PIK, 'rb') as f:
            data = pickle.load(f)
        self.F = data[0]
        self.E = data[1]
        self.mask = data[2]
        self.pixel_points1 = data[3]
        self.pixel_points2 = data[4]
        self.inliers1 = data[5]
        self.inliers2 = data[6]
        self.indices1 = data[7]
        self.indices2 = data[8]
        self.matches = data[9]

    def store_data(self) -> None:
        PIK = os.path.join(self.dataset_path, "features", f"{self.image_name1}-{self.image_name2}.pkl")
        data = [self.F, self.E, self.mask, self.pixel_points1, self.pixel_points2, self.inliers1, self.inliers2,self.indices1,self.indices2,self.matches]
        with open(PIK, 'wb') as f:
            pickle.dump(data, f)
        
        
    def number_of_inliers(self) -> int:
        return np.sum(self.mask)
            

    def load_torch_images(self, image:'np.array') -> 'K.Tensor':
        image = K.image_to_tensor(image, False).float() /255
        image = K.color.bgr_to_rgb(image)
        image = image.to(self.device)
        return image

    def draw_matches(self)->None:

        concatImg = np.zeros((max(self.view1.scaled_image.shape[0], self.view2.scaled_image.shape[0]), self.view1.scaled_image.shape[1] + self.view2.scaled_image.shape[1], 3), dtype=np.uint8) 
        concatImg[:, :] = (255, 255, 255)
        concatImg[:self.view1.scaled_image.shape[0], :self.view1.scaled_image.shape[1], :3] = self.view1.scaled_image
        concatImg[:self.view2.scaled_image.shape[0], self.view1.scaled_image.shape[1]:, :3] = self.view2.scaled_image

        for (p1, p2) in random.sample(list(zip(self.scaled_pixel_points1, self.scaled_pixel_points2)), 50):
            starting_point = (int(p1[0]), int(p1[1]))
            ending_point = (int(p2[0] + self.view1.scaled_image.shape[1]), int(p2[1]))
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
        