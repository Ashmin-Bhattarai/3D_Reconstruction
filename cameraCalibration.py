""" 
    Latest Update: 2022:12:23
    Updated By: @Ashmin-Bhattarai
    Description: This program is used to calibrate the camera using the images captured from the camera.
"""

# importing libraries
import numpy as np
import cv2 as cv
import os
from captureCalibrationImages import detect_checker_board

# defining global variables
imgDirName:str = "images" # folder name to store images, give only the name of the folder not the path
patternSize:tuple = (10, 7) # number of inner corners in the checker board pattern
squareSize:float = 25 # size of the square in the checker board pattern

# main function
def main():
    """
        This is the main function of the script.
        
        Args:
            None
        
        Returns:
            None
    """

    # checking if the folder exists
    if not os.path.exists(imgDirName):
        print("Folder does not exist")
        return
    # checking if the folder is empty
    elif len(imagesName := os.listdir(imgDirName)) == 0:
        print("No images found")
        return
    
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # defining arrays to store object points and image points from all the images
    objPoints3D = [] # 3D points in real world space
    imgPoints2D = [] # 2D points in image plane

    # preparing object points
    objP = np.zeros((patternSize[0] * patternSize[1], 3), np.float32)
    objP[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)

    # setting checker board size
    objP *= squareSize

    # looping through all the images
    for imageName in imagesName:
        # reading image
        image = cv.imread(os.path.join(imgDirName, imageName))
        # converting image to gray scale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # finding checker board corners
        ret, corners = detect_checker_board(gray, criteria, patternSize)

        # if corners are found, add object points, image points
        if ret:
            objPoints3D.append(objP)
            imgPoints2D.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objPoints3D, imgPoints2D, gray.shape[::-1], None, None)

    # saving the camera matrix and distortion coefficients
    # np.savez("calibration.npz", mtx=mtx, dist=dist)
    with open("calibration.txt", "a") as f:
        f.write(f"\n\nCamera Matrix:\n{mtx}\n\nDistortion Coefficients:\n{dist}")

    print("Calibration Complete")
    print("Camera Matrix:\n", mtx)
    

if __name__ == "__main__":
    main()