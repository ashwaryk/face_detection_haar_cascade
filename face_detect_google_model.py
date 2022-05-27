import cv2
import mediapipe as mp
import time
import numpy as np

cap = cv2.VideoCapture(0)
pTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
#    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
                   
            # Create a mask
            blurred_img = cv2.GaussianBlur(img, (51, 51), 0)
            mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
            mask = cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 255, 255), -1)
            # Apply the mask
            img = np.where(mask!=np.array([255, 255, 255]), img, blurred_img)
                   
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()