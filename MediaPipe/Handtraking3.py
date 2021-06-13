"""
Hand Tracing Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone
"""
import math
import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList


def distPoints(A, B):
    return int(math.sqrt(math.pow(A[0] - B[0], 2) + math.pow(A[1] - B[1], 2)))


def calcAngPoint(A, B):
    dab = A[1] - B[1]
    if dab == 0:
        return 0
    else:
        return (A[0] - B[0]) / dab


def calcAngLine(A, B, C):
    g1 = calcAngPoint(A, B)
    g2 = calcAngPoint(B, C)

    if g2 == 0:
        g = 1
    else:
        g = g1 / g2
        if g < 0:
            g *= -1

    return g


def main():
    # circle parameters
    cx = 320
    cy = 240
    cr = 50
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img, draw=False)
        lmList = detector.findPosition(img, draw=False)

        cv2.circle(img, (cx, cy), cr, (128, 255, 0), cv2.FILLED)

        if len(lmList) != 0:

            for p in lmList:
                pn = (p[1], p[2])
                dc = distPoints(pn, (cx, cy))
                if dc < cr:

                    if cx > pn[0]:
                        cx += int((cr - dc) / 2)
                    else:
                        cx -= int((cr - dc) / 2)

                    if cy > pn[1]:
                        cy += int((cr - dc) / 2)
                    else:
                        cy -= int((cr - dc) / 2)

                    cv2.circle(img, pn, 5, (255, 255, 255), 3)

        cTime = time.time()
        dTime = cTime - pTime
        if dTime > 0:
            fps = 1 / dTime
            pTime = cTime
            cv2.putText(img, "FPS: " + str(int(fps)),
                        (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
