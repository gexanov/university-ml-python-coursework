import cv2
import numpy as np

# читаем фон и логотип
bg = cv2.imread('pngtree-d-render-and-illustration-of-office-desk-from-top-view-computer-picture-image_13280381.png')
logo = cv2.imread('Python_logo_(icon_only).svg.png')

# уменьшаем логотип (если нужно)
scale = 0.3
logo = cv2.resize(logo, None, fx=scale, fy=scale)

h, w = logo.shape[:2]
H, W = bg.shape[:2]

# случайная позиция
np.random.seed(42)
x = np.random.randint(0, W - w)
y = np.random.randint(0, H - h)

# вставка
bg[y:y+h, x:x+w] = logo

cv2.imwrite('large_image.png', bg)

print(f"Логотип вставлен в: x={x}, y={y}")