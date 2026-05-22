import cv2

# Загружаем видеофайл video.mp4
cap = cv2.VideoCapture(0)
MOG2 = cv2.createBackgroundSubtractorMOG2()
while True:
    ret, frame = cap.read()
    if ret == False:
        print('img not read')
        break
    mog_img = MOG2.apply(frame)
    cv2.imshow('Motion Detection', frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    miner = cv2.erode(mog_img, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dil = cv2.dilate(miner, kernel, iterations=2)
    cv2.imshow('dil', dil)
    kernel_to_closing = cv2.getStructuringElement(cv2.MORPH_RECT, (15,15))
    f = cv2.morphologyEx(dil, cv2.MORPH_CLOSE, kernel_to_closing)
    cv2.imshow('close', f)
    f = cv2.dilate(f, kernel_to_closing, iterations=1)
    fndcntr, _ = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in fndcntr:
        if cv2.contourArea(i) < 2500:
            continue
        else:
            x,y,w,h = cv2.boundingRect(i)
            motion_detected = True
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    if motion_detected:
        cv2.putText(frame, "MOTION DETECTED", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        cv2.putText(frame, "NO MOTION", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    motion_detected = False
    cv2.imshow('final', frame)


    key = cv2.waitKey(1)
    if key  == ord('q'):
        break
cap.release()