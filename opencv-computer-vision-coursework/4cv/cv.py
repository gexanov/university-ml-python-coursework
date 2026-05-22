import cv2

# Загружаем видеофайл video.mp4
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: видео не открылось")
    exit()
# Читаем два последовательных кадра из видеофайла

ret1, frame1 = cap.read()
ret2, frame2 = cap.read()

if not ret1 or not ret2:
    print("Ошибка: не удалось прочитать кадры")
    exit()
# Начало бесконечного цикла обработки кадров
while True:
    # Вычисляем разницу между двумя кадрами (абсолютную разницу)
    diff = cv2.absdiff(frame1, frame2)

    # Преобразуем изображение разницы в черно-белое (серое)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Применяем пороговую фильтрацию для получения бинарного изображения
    _, thresh = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)

    # Находим контуры на бинарном изображении
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Проходим по всем найденным контурам
    for contour in contours:
        # Оставляем только крупные контуры (площадью больше 500 пикселей)
        if cv2.contourArea(contour) > 500:
            # Получаем координаты и размеры прямоугольника, охватывающего контур
            x, y, w, h = cv2.boundingRect(contour)
            # Рисуем зеленый прямоугольник вокруг области движения
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Показываем текущий кадр с обнаруженными движениями
    cv2.imshow('Motion Detection', frame1)

    # Текущий кадр становится предыдущим, а затем считываем новый кадр
    frame1 = frame2
    ret, frame2 = cap.read()

    # Если закончился видеофайл или нажата клавиша q, останавливаемся
    if not ret or cv2.waitKey(1) == ord('q'):
        break

# Освобождаем захваченный видеопоток и закрываем окна
cap.release()
cv2.destroyAllWindows()
