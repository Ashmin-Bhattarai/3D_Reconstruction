""" 
    Latest Update: 2022:12:23
    Updated By: @Ashmin-Bhattarai
    Description: This program is used to capture images from a webcam or URL and 
    save them in a folder. It also detects the chessboard corners. 
"""

# defining global variables
imgDirName:str = "images" # folder name to store images, give only the name of the folder not the path
# videoURL = "http://192.168.1.13:8080/video" # 0 for webcam and URL for IP camera or video file path for video file 
videoURL = 0 # 0 for webcam and URL for IP camera or video file path for video file 
patternSize:tuple = (10, 7) # number of inner corners in the checker board pattern


# Importing Libraries
import numpy as np
import cv2 as cv
import os


def detect_checker_board(image, criteria, patternSize):
    """
        This function detects the chessboard corners in the image.
        
        Args:
            image: image in gray scale to detect corners
            criteria: termination criteria
            patternSize: number of inner corners in the chessboard pattern
        
        Returns:
            ret: boolean value
            corners: detected corners
    """

    # finding chessboard corners
    ret, corners = cv.findChessboardCorners(image, patternSize, None)

    # refining corners
    if ret:
        corners = cv.cornerSubPix(image, corners, (3, 3), (-1, -1), criteria)

    return ret, corners


# main function
def main():
    """
        This is the main function of the script.
        
        Args:
            None
        
        Returns:
            None
    """
    # creating a folder to store images
    if not os.path.exists(imgDirName):
        os.makedirs(imgDirName)
    else:
        # deleting all the images in the folder
        for imageName in os.listdir(imgDirName):
            os.remove(os.path.join(imgDirName, imageName))
    
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # reading video from camera
    cap = cv.VideoCapture(videoURL)

    # counter for number of saved images
    numberOfSavedImages = 0

    # reading frames
    while True:
        _, frame = cap.read()
        frameCopy = frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # finding chessboard corners
        ret, corners = detect_checker_board(gray, criteria, patternSize)

        # drawing corners
        if ret:
            cv.drawChessboardCorners(frameCopy, patternSize, corners, ret)
        
        # displaying text on the image to guide the user on how to use the program 
        cv.putText(frameCopy, "Press 's' to save the image", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.putText(frameCopy, "Press 'q' to quit", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv.putText(frameCopy, f"No of saved images: {numberOfSavedImages}", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # displaying the image
        cv.imshow("frame", frameCopy)

        # waiting for user input
        key = cv.waitKey(1)

        # checking user input
        # if user presses 's' then save the image in the folder and increment the counter by 1
        if key == ord("s") and ret:
            cv.imwrite(f"{imgDirName}/image{numberOfSavedImages}.jpg", frame)
            numberOfSavedImages += 1
        # if user presses 'q' then quit the program
        elif key == ord("q"):
            break
    
    # releasing the camera and destroying all windows
    cap.release()
    cv.destroyAllWindows()


# main function call
if __name__ == "__main__":
    main()