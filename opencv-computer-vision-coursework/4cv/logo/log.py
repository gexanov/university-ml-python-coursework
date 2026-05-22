import cv2
import numpy as np

# Загружаем большое изображение и логотип
large_image = cv2.imread('large_image.png')
logo = cv2.imread('Python_logo_(icon_only).svg.png')

# Метод сопоставления (TM_CCOEFF_NORMED дает хорошие результаты)
method = cv2.TM_CCOEFF_NORMED

# Проводим сопоставление шаблона
result = cv2.matchTemplate(large_image, logo, method)

# Находим максимальное значение сопоставления
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Верхний левый угол найденного шаблона
top_left = max_loc

# Размеры логотипа
logo_height, logo_width = logo.shape[:2]

# Нижний правый угол найденного шаблона
bottom_right = (top_left[0] + logo_width, top_left[1] + logo_height)

# Рисуем рамку вокруг найденного шаблона
cv2.rectangle(large_image, top_left, bottom_right, (0, 255, 0), 2)

# Показываем результат
cv2.imshow('Logo Found', large_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
