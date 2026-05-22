import cv2

# Загружаем видеофайл video.mp4
cap = cv2.VideoCapture(0)
MOG2 = cv2.createBackgroundSubtractorMOG2()
while True:
    ret, frame = cap.read()
    if ret == False:
        print('img not read')
    mog_img = MOG2.apply(frame)
    cv2.imshow('Motion Detection', frame)
    cv2.imshow('Motion Detection', mog_img)
    key = cv2.waitKey(1)
    if key  == ord('q'):
        break
cap.release()