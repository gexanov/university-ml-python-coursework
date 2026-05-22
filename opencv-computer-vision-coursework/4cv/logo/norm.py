import cv2
import numpy as np

# ---------- загрузка ----------
img = cv2.imread('large_image.png')
template = cv2.imread('Python_logo_(icon_only).svg.png')

if img is None:
    raise ValueError("Не удалось загрузить большое изображение")
if template is None:
    raise ValueError("Не удалось загрузить шаблон")

# ---------- grayscale ----------
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# ---------- небольшое сглаживание ----------
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
tpl_gray = cv2.GaussianBlur(tpl_gray, (5, 5), 0)

# ---------- функция поворота ----------
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated

# ---------- параметры ----------
scales = np.linspace(0.5, 1.5, 11)
angles = [-30, -15, 0, 15, 30]
threshold = 0.72

all_boxes = []
all_scores = []

# ---------- поиск ----------
for angle in angles:
    rotated_tpl = rotate_image(tpl_gray, angle)

    for scale in scales:
        scaled_tpl = cv2.resize(rotated_tpl, None, fx=scale, fy=scale)

        th, tw = scaled_tpl.shape[:2]

        # шаблон не должен быть больше изображения
        if th > img_gray.shape[0] or tw > img_gray.shape[1]:
            continue

        res = cv2.matchTemplate(img_gray, scaled_tpl, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            x, y = pt
            all_boxes.append([x, y, tw, th])
            all_scores.append(res[y, x])

# ---------- если ничего не найдено ----------
if len(all_boxes) == 0:
    print("Ничего не найдено")
else:
    # groupRectangles лучше работает, когда прямоугольники дублируются
    rects_for_group = []
    for box in all_boxes:
        rects_for_group.append(box)
        rects_for_group.append(box)

    grouped_boxes, weights = cv2.groupRectangles(rects_for_group, groupThreshold=1, eps=0.4)

    # если groupRectangles ничего не вернул, рисуем сырые боксы
    if len(grouped_boxes) == 0:
        print("После groupRectangles ничего не осталось, показываю сырые совпадения")
        grouped_boxes = all_boxes

    for (x, y, w, h) in grouped_boxes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print(f"Найдено прямоугольников: {len(grouped_boxes)}")

cv2.imshow("detected", img)
cv2.waitKey(0)
cv2.destroyAllWindows()