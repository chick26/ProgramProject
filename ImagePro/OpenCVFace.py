import cv2
import numpy as np
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)                                             #调用0号摄像头

while True:
    ret, img = cap.read()                                             #从摄像头获取到图像，返回了第一个为布尔值表示成功与否，第二个是图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                      #转换成灰度图片
    faces = detector.detectMultiScale(gray, 1.3, 5)                   #检测人脸，返回(x, y, height, width)，即人脸的位置
    for (x, y, w, h) in faces:                                        #在图像中标记人脸
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):                              #q退出
        break

cap.release()                                                          #释放摄像头资源
cv2.destroyAllWindows()