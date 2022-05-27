import cv2
import numpy as np

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        blurred_img = cv2.GaussianBlur(img, (51, 51), 0)
        # Create a mask
        mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.uint8)
        mask = cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
        # Apply the mask
        img = np.where(mask!=np.array([255, 255, 255]), img, blurred_img)
    
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()