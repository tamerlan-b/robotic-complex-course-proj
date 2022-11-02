import numpy as np
import cv2

class ImageProcessor:
    def getContourCenters(contours):
        centers = []
        for c in contours:
            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
        return centers

    def drawCenters(img, centers):
        for c in centers:
            cv2.circle(img, (c[0], c[1]), 7, (0, 0, 0), -1)
            text = f"(U,V) = ({c[0]}, {c[1]})"
            text_pos = (c[0] - 30, c[1] - 30)
            cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return img

    # Находит красные кубики на изображении с камеры и возвращает их центры
    def findObjects(img):
        # Переводим в HSV
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
        # Бинаризуем
        hsv_min = np.array((30,0,0), np.uint8)
        hsv_max = np.array((255,255,255), np.uint8)
        thresh = cv2.inRange(hsv, hsv_min, hsv_max)
        # Детектируем контуры
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Вычисляем центры объектов
        centers = ImageProcessor.getContourCenters(contours)
        return centers

    def process_image(img):
        centers = ImageProcessor.findObjects(img)
        cnt_img = img.copy()
        cnt_img = ImageProcessor.drawCenters(cnt_img, centers)
        return cnt_img