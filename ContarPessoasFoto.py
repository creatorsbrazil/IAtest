# Pacotes necessários
# pip install opencv-python opencv-contrib-python imutils
# Referencias e livre adaptação dos códigos exemplificados
# https://docs.opencv.org/2.4/doc/tutorials/introduction/display_image/display_image.html
# https://docs.opencv.org/3.0-beta/modules/objdetect/doc/cascade_classification.html?highlight=haar%20cascade
# https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
# https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/

import cv2
import numpy as np
import datetime
import imutils

pA = 105    # scaleFactor – Parameter specifying how much the image size is reduced at each image scale
pB = 2      # minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it
pC = 0      # Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade

vermelho = (0, 0, 255)
azul = (255, 0, 0)
branco = (255, 255, 255)

face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')

while True:

    img = cv2.imread("images/crowd-many-people-in-confined-space-at-a-festival-E2HD6H.jpg")
    # img = imutils.resize(img, width=900) # redimenciona a entrada

    dtStart = datetime.datetime.today()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,       # Matrix of the type CV_8U containing an image where objects are detected
        pA/100,     # scaleFactor – Parameter specifying how much the image size is reduced at each image scale
        pB,         # minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it
        pC,         # Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade
        # minSize – Minimum possible object size. Objects smaller than that are ignored.
        (30, 30),
        (100, 100))  # maxSize – Maximum possible object size. Objects larger than that are ignored.

    sampleNum = 0
    for (x, y, w, h) in faces:
        sampleNum = sampleNum + 1
        cv2.rectangle(img, (x, y), (x+w, y+h), azul, 2)
        cv2.putText(img, str(sampleNum), (x+1, y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, vermelho, 1)

    dtFrame = datetime.datetime.today()
    diff = dtFrame - dtStart
    elapsed_time = int((diff.seconds * 1000) + (diff.microseconds / 1000))

    info = 'Tempo: {}ms Total: {} A {} B {} C {}'.format(
        elapsed_time, sampleNum, pA / 100, pB, pC)
    cv2.putText(img, info, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, branco, 5)
    cv2.putText(img, info, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, azul, 2)

    img = imutils.resize(img, width=900)  # redimenciona a saída
    cv2.imshow("Contando pessoas", img)

    key = cv2.waitKey(300) & 0xFF
    if key == 27:
        break
    elif key == ord('z') and pA > 101:
        pA -= 1
    elif key == ord('a'):
        pA += 1
    elif key == ord('x') and pB > 0:
        pB -= 1
    elif key == ord('s'):
        pB += 1
    elif key == ord('c') and pC > 0:
        pC -= 1
    elif key == ord('d'):
        pC += 1
    elif key == ord('0'):
        pA = 105
        pB = 2
        pC = 0

cv2.destroyAllWindows()
