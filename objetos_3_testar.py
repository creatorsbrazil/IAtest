import cv2
import os

#haar_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
#haar_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_apple.xml')
haar_cascade = cv2.CascadeClassifier('temp/data/cascade.xml')
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    objetos = haar_cascade.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in objetos:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow('img',img)
    k=cv2.waitKey(30)
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()