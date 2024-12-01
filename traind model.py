import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time

url = 'http://192.168.19.152:8080/video'

cap = cv2.VideoCapture(url)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# for training the model
folder = "images/C"
counter = 0

# Create a single white background image outside the loop
imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:  # Check if imgCrop has valid dimensions
            # Calculate the aspect ratio of the hand region
            aspect_ratio = imgCrop.shape[1] / imgCrop.shape[0]

            # Determine the dimensions to resize the hand region while preserving its aspect ratio
            new_width = int(min(imgSize, imgSize * aspect_ratio))
            new_height = int(min(imgSize, imgSize / aspect_ratio))

            # Resize the hand region
            imgCrop_resized = cv2.resize(imgCrop, (new_width, new_height))

            # Calculate the position to place the resized hand region on the white background
            x_offset = (imgSize - new_width) // 2
            y_offset = (imgSize - new_height) // 2

            # Clear the white background image
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Place the resized hand region onto the white background
            imgWhite[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = imgCrop_resized

            # Display the white background with the hand image overlay
            cv2.imshow("ImageWhite", imgWhite)

    # Display the original image
    cv2.imshow("Image", img)
# if press s image s are stored in this folder
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)