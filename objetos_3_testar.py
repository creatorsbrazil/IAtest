import cv2
import os

haarPath: str = 'haarcascade'
colors = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (127, 0, 0),
    (0, 127, 0),
    (0, 0, 127),
    (127, 127, 0),
    (0, 127, 127),
    (127, 0, 127)
)


class xmlHaarCascade:
    def __init__(self, xml: str, pos: int):
        self.name = xml.replace('haarcascade_', '').replace('.xml', '').split('_')[0]
        self.cascade = cv2.CascadeClassifier(haarPath+'/' + xml)
        self.color = colors[pos]
        print(self.name)


colorPos = 0
haars = []
for xml in os.listdir(haarPath):
    if not xml.endswith('.xml'):
        continue
    elif xml.upper().find('1600_HAAR')==-1:
        continue

    haars.append(xmlHaarCascade(xml, colorPos))
    colorPos += 1
    if colorPos >= len(colors):
        colorPos = 0

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for haar in haars:

        objetos = haar.cascade.detectMultiScale(gray, 1.05, 10) #, 0, (70, 70), (150, 150))

        for (x, y, w, h) in objetos:
            cv2.rectangle(img, (x, y), (x+w, y+h), haar.color, 2)
            cv2.putText(img, haar.name, (x+1, y+12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, haar.color, 1)

    cv2.imshow('img', img)
    k = cv2.waitKey(30)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
