import cv2
import os
import time
import imutils

path = 'E:\LiveED\Dataset'

cap = cv2.VideoCapture('roopesh-outdoor1.mp4')
i = 1474
t = time.time()

while(time.time() - t < 5):
    _,frame = cap.read()
    resized = cv2.resize(frame,(100,100))
    rotated = imutils.rotate_bound(resized, -90)
    cv2.imwrite(os.path.join(path,'image%d.jpg'%i),rotated)
    i = i + 1


cap.release()
cv2.destroyAllWindows()

