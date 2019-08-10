# Pacotes necessários
# pip install opencv-python opencv-contrib-python imutils
# Referencias e livre adaptação dos códigos exemplificados
# https://docs.opencv.org/2.4/doc/tutorials/introduction/display_image/display_image.html
# https://docs.opencv.org/3.0-beta/modules/objdetect/doc/cascade_classification.html?highlight=haar%20cascade
# https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
# https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
# https://stackoverflow.com/questions/22249579/opencv-detectmultiscale-minneighbors-parameter

import cv2
import numpy as np
import datetime
import imutils

pA = 105     # scaleFactor – Parameter specifying how much the image size is reduced at each image scale
pB = 20      # minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it
pS = 100     # Size

vermelho = (0, 0, 255)
azul = (255, 0, 0)
branco = (255, 255, 255)

areaX = 120
areaY = 200
areaW = 400
areaH = 200

# alem de pessoas, pode-se contar qualquer outro tipo de objeto baseado em qualquer arquivo
# cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')
cascade = cv2.CascadeClassifier('haarcascade/torre4.xml')
# cascade = cv2.CascadeClassifier('temp/data/cascade.xml')

# use a linha abaixo para habilitar a camera
cap = cv2.VideoCapture(0)
while True:

    ret, img = cap.read()

    # img = cv2.imread("images/crowd-many-people-in-confined-space-at-a-festival-E2HD6H.jpg")
    # img = imutils.resize(img, width=900) # redimenciona a entrada

    dtStart = datetime.datetime.today()

    img2 = img[areaY:(areaY+areaH), areaX:(areaX+areaW)]
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # faces = []
    if pS >= 30:
        faces = cascade.detectMultiScale(
            gray,             # Matrix of the type CV_8U containing an image where objects are detected
            pA/100,           # scaleFactor – Parameter specifying how much the image size is reduced at each image scale
            pB,               # minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it
            0,                # It is not used for a new cascade
            (pS, pS),         # minSize – Minimum possible object size.
            (pS*+50, pS+50))  # maxSize – Maximum possible object size.
    else:
        faces = cascade.detectMultiScale(gray, pA/100, pB)

    sampleNum = 0
    for (x, y, w, h) in faces:
        sampleNum = sampleNum + 1
        info = '{}: {}x{} '.format(sampleNum, w, h)
        cv2.rectangle(img, (areaX+x, areaY+y), (areaX+x+w, areaY+y+h), azul, 2)
        cv2.putText(img, info, (areaX+x+1, areaY+y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, vermelho, 1)

    dtFrame = datetime.datetime.today()
    diff = dtFrame - dtStart
    elapsed_time = int((diff.seconds * 1000) + (diff.microseconds / 1000))

    info = 'Tempo: {}ms Total: {} '.format(elapsed_time, sampleNum)
    cv2.putText(img, info, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, branco, 5)
    cv2.putText(img, info, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, azul, 2)

    info = 'A {} B '.format(pA / 100)

    if pB > 0:
        info += str(pB)
    else:
        info += 'all'

    info += ' S '

    if pS >= 30:
        info += str(pS)
    else:
        info += 'all'

    cv2.putText(img, info, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.1, branco, 5)
    cv2.putText(img, info, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.1, azul, 2)

    cv2.rectangle(img, (areaX, areaY), (areaX+areaW, areaY+areaH), vermelho, 3)

    # img = imutils.resize(img, width=400)  # redimenciona a saída
    cv2.imshow("Identicando", img)

    # gray = imutils.resize(gray, width=200)  # redimenciona a saída
    # cv2.imshow("Cinza", gray)

    key = cv2.waitKey(1) & 0xFF
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
    elif key == ord('c') and pS > 20:
        pS -= 10
    elif key == ord('d'):
        pS += 10
    elif key == ord('0'):
        pA = 105
        pB = 20
        pS = 100

cv2.destroyAllWindows()
