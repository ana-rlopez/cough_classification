import collections

import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# video_capture = cv2.VideoCapture('outpy.avi')
video_capture = cv2.VideoCapture(0)

QUEUE_LEN = 30
x_centers = None
y_centers = None
heights = None

signal = []

while video_capture.isOpened():
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=15,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        if isinstance(faces, tuple):
            continue

        (x, y, w, h) = faces[0]
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        x_center, y_center = (x + x + w) / 2, (y + y + h) / 2

        if x_centers is None:
            x_centers = collections.deque(QUEUE_LEN * [x_center], QUEUE_LEN)
            y_centers = collections.deque(QUEUE_LEN * [y_center], QUEUE_LEN)
            heights = collections.deque(QUEUE_LEN * [h], QUEUE_LEN)
        else:
            x_centers.append(x_center)
            y_centers.append(y_center)
            heights.append(h)

        x_center_adjusted = int(sum(x_centers) / QUEUE_LEN)
        y_center_adjusted = int(sum(y_centers) / QUEUE_LEN)
        height_adjusted = int(sum(heights) / QUEUE_LEN)

        forehead_x = x_center_adjusted
        forehead_y = int(y_center_adjusted - h / 3)

        # cv2.rectangle(frame, (forehead_x - 10, forehead_y - 10), (forehead_x + 10, forehead_y + 10), (255, 255, 255))

        forehead_green = frame[forehead_y - 5:forehead_y + 5, forehead_x - 5:forehead_x + 5, 1]
        # frame[forehead_y - 10:forehead_y + 10, forehead_x - 10:forehead_x + 10, 1] = 0

        # cv2.putText(frame, f'{x_center_adjusted}; {y_center_adjusted}', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        # Display the resulting frame
        cv2.imshow('Video', forehead_green)

        signal.append(forehead_green.sum())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

np.save('signal', np.array(signal))

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()