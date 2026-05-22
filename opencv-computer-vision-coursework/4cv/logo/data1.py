import cv2
import numpy as np

background = cv2.imread('pngtree-d-render-and-illustration-of-office-desk-from-top-view-computer-picture-image_13280381.png')
logo = cv2.imread('Python_logo_(icon_only).svg.png')

if background is None:
    raise ValueError("Не удалось загрузить background.png")
if logo is None:
    raise ValueError("Не удалось загрузить логотип")

# уменьшаем логотип до разумного размера
logo = cv2.resize(logo, (120, 120))

h, w = logo.shape[:2]
H, W = background.shape[:2]

x, y = 200, 150

if y + h > H or x + w > W:
    raise ValueError("Логотип не помещается в фон")

background[y:y+h, x:x+w] = logo

cv2.imwrite('large_image.png', background)
print("large_image.png создан")
print("Логотип вставлен в:", (x, y))