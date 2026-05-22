import cv2

img = cv2.imread('large_image.png')
template = cv2.imread('Python_logo_(icon_only).svg.png')

if img is None:
    raise ValueError("Не удалось загрузить large_image.png")
if template is None:
    raise ValueError("Не удалось загрузить шаблон")

template = cv2.resize(template, (120, 120))

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

res = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

print("max_val =", max_val)
print("max_loc =", max_loc)

h, w = tpl_gray.shape[:2]
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()