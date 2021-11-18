import cv2
import numpy as np


# 由图像提取特征的函数
def featureGet(filename):
    image = cv2.imread('../dataset/' + filename, cv2.IMREAD_COLOR)
    # 类型转换，防止数据溢出
    img1 = np.array(image, dtype='int')
    # 超绿灰度图
    b, g, r = cv2.split(img1)
    ExG = 2 * g - r - b
    [m, n] = ExG.shape
    sum_color = 0
    num = 0
    # 标准化
    for i in range(m):
        for j in range(n):
            if ExG[i, j] < 0:
                ExG[i, j] = 0
            elif ExG[i, j] > 255:
                ExG[i, j] = 255
    # 转换回opencv可读类型
    ExG = np.array(ExG, dtype='uint8')
    ret2, th2 = cv2.threshold(ExG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 求树叶均色调
    for i in range(m):
        for j in range(n):
            if th2[i, j] != 0:
                sum_color += img1[i, j]
                num += 1
    color_avg = sum_color / num
    # 识别图像轮廓
    contours, heridency = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = 1
    for i in contours:
        if count == 1:
            points = i
            count += 1
        else:
            points = np.concatenate((points, i))
    # 求最大轮廓外接矩形
    rect = cv2.minAreaRect(points)
    area = cv2.contourArea(points)
    width, height = rect[1]
    ratio = area / (width * height)
    b, g, r = color_avg
    return b, g, r, ratio
