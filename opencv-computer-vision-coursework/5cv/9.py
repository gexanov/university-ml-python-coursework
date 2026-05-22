import cv2
import numpy as np


#img = cv2.VideoCapture(0)
#img = cv2.IMREAD_GRAYSCALE(img)
img = cv2.imread('image.jpg', 0)

# Загружаем изображение в оттенках серого.
# Флаг '0' (или cv2.IMREAD_GRAYSCALE) указывает OpenCV,
# что изображение нужно прочитать как одноканальное (серое).
# Это ускоряет вычисления и упрощает работу с фильтрами.

sobel_x = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=3)
# Применяем фильтр Собеля для выделения вертикальных граней (градиента по оси Y).
# Параметры функции:
#   img: исходное изображение.
#   cv2.CV_64F: глубина выходного изображения (64-битный float для точности).
#   dx=0: производная по горизонтали (икс) не берется.
#   dy=1: производная по вертикали (игрек) берется (первого порядка).
#   ksize=5: размер ядра фильтра (матрицы свертки). Чем больше размер, тем сильнее размытие.

#sobel_x = cv2.convertScaleAbs(sobel_x)
#sobel_y = cv2.convertScaleAbs(sobel_y) # из ща формата np.uint8 отр знач плохо обрабатывает


combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
# Отображаем итоговое изображение.
# Важно: результат работы Sobel имеет тип float64, а imshow требует uint8.
# Поэтому мы приводим тип данных к uint8 перед показом.
inverted = cv2.bitwise_not(combined)

cv2.imshow('Sobel Edges', inverted.astype(np.uint8))
cv2.waitKey(0)
