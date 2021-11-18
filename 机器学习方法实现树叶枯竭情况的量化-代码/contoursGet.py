import cv2
import numpy as np

img1 = cv2.imread('../dataset/1.jpg')
img2 = cv2.imread('../dataset/1.jpg')

img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
areas = []

for c in range(len(contours)):
    areas.append(cv2.contourArea(contours[c]))

max_id = areas.index(max(areas))

max_rect = cv2.minAreaRect(contours[max_id])
max_box = cv2.boxPoints(max_rect)
max_box = np.int0(max_box)
img2 = cv2.drawContours(img2, [max_box], 0, (0, 255, 0), 2)

pts1 = np.float32(max_box)
pts2 = np.float32([[max_rect[0][0] + max_rect[1][1] / 2, max_rect[0][1] + max_rect[1][0] / 2],
                   [max_rect[0][0] - max_rect[1][1] / 2, max_rect[0][1] + max_rect[1][0] / 2],
                   [max_rect[0][0] - max_rect[1][1] / 2, max_rect[0][1] - max_rect[1][0] / 2],
                   [max_rect[0][0] + max_rect[1][1] / 2, max_rect[0][1] - max_rect[1][0] / 2]])
M = cv2.getPerspectiveTransform(pts1, pts2)
dst = cv2.warpPerspective(img2, M, (img2.shape[1], img2.shape[0]))

# 此处可以验证 max_box点的顺序
color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255)]
i = 0
for point in pts2:
    cv2.circle(dst, tuple(point), 2, color[i], 4)
    i += 1

target = dst[int(pts2[2][1]):int(pts2[1][1]), int(pts2[2][0]):int(pts2[3][0]), :]

cv2.imshow('img2', img2)
cv2.imshow('dst', dst)
cv2.imshow('target', target)
cv2.waitKey()
cv2.destroyAllWindows()
