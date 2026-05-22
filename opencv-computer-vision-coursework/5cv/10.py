import cv2
import numpy as np


def has_large_gap(contour, max_distance):
    points = contour.reshape(-1, 2)
    for i in range(len(points)):
        point1 = points[i]
        point2 = points[(i + 1) % len(points)]
        distance = np.linalg.norm(point1 - point2)
        if distance > max_distance:
            return True
    return False


#img = cv2.VideoCapture(0)
#img = cv2.IMREAD_GRAYSCALE(img)
img = cv2.imread('3612773_original.jpg')
img_gray = cv2.imread('3612773_original.jpg', 0)
# Загружаем изображение в оттенках серого.
# Флаг '0' (или cv2.IMREAD_GRAYSCALE) указывает OpenCV,
# что изображение нужно прочитать как одноканальное (серое).
# Это ускоряет вычисления и упрощает работу с фильтрами.

blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)

edges = cv2.Canny(blurred, 20, 50)
good_contours = []

contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    area = cv2.contourArea(contour)
    if area < 500:
        continue
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if has_large_gap(approx, 190):
        continue
    good_contours.append(approx)

# Создаем копию изображения, чтобы не портить оригинал
output = img.copy()
# Рисуем все найденные контуры красным цветом
output = cv2.drawContours(output, good_contours, -1, (0, 0, 255), 2)

cv2.imshow('canny', output)
cv2.waitKey(0)
