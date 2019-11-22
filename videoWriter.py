#!/usr/bin/python3
import cv2
import time
import sys

if __name__ == '__main__':
    assert(len(sys.argv) == 4)
    cap = cv2.VideoCapture(int(sys.argv[1]))
    _, frame = cap.read()

    h,w = frame.shape[:2]
    framerate = int(sys.argv[3])
    frametime = 1.0 / framerate
    

    for i in range(3,0,-1):
        print('\rStarting in {}'.format(i), end='')
        time.sleep(1)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    writer = cv2.VideoWriter(sys.argv[2], fourcc, framerate, (w, h))

    while True:
        
        start_time = time.time()
        _, frame = cap.read()
        
        cv2.imshow('frame', frame)
        writer.write(frame)

        key = cv2.waitKey(1)
        if key & 0xFF in [27, ord('q')]:
            break

        delta = time.time() - start_time
        time.sleep(max(0, frametime - delta))

