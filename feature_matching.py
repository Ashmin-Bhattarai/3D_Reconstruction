from turtle import width
import numpy as np
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("./images/japan1.jpg", 0)
img2 = cv2.imread("./images/japan3.jpg", 0)

width = img1.shape[1]
height = img1.shape[0]

img1 = cv2.resize(img1, (width-100, height))
img2 = cv2.resize(img2, (width-100, height))

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k = 2)

matchesMAsk = [[0, 0] for i in range(len(matches))]
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMAsk[i] = [1, 0]

draw_params = dict(matchColor = (0, 255, 0),
                     singlePointColor = (255, 0, 0),
                        matchesMask = matchesMAsk,
                            flags = cv2.DrawMatchesFlags_DEFAULT)

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches[:], None)
img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches[:50], None)
img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params) 


# plt.subplot(1, 3, 1)
# plt.imshow(img3)
# plt.title("All matches")
# plt.subplot(1, 3, 2)
# plt.imshow(img4)
# plt.title("50 matches")
# plt.subplot(1, 3, 3)
# plt.imshow(img5)
# plt.title("50 matches with mask")
# plt.show()

cv2.imshow("All matches", img3)
cv2.imshow("50 matches", img4)
cv2.imshow("50 matches with mask", img5)
cv2.waitKey(0)
cv2.destroyAllWindows()