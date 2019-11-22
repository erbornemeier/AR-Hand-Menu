#!/usr/bin/python3
import cv2
import time
import numpy as np


protoFile = "model/pose_deploy.prototxt"
weightsFile = "model/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

INPUT_PAIRS = {'thumb':(1,4), 
               'index':(5,8),
               'middle':(9,12),
               'ring':(13,16),
               'pinky':(17,20) }

MENU_KEYS =   {'thumb':4, 
               'index':3,
               'middle':2,
               'ring':1,
               'pinky':0 }

mainMenu = True 
subMenu = 0

mainMenuItem = {'pinky': 'Monday',
                'ring': 'Tuesday',
                'middle': 'Wednesday',
                'index': 'Thursday',
                'thumb': 'Friday'}

threshold = 0.2

input_source = "menu_demo2.avi"
cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

aspect_ratio = frameWidth/frameHeight

inHeight = 368
inWidth = int(((aspect_ratio*inHeight)*8)//8)

vid_writer = cv2.VideoWriter('menu_demo2_output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
k = 0
while 1:
    k+=1
    t = time.time()
    for i in range(10):
        hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()

    print("forward = {}".format(time.time() - t))

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold :
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else :
            points.append(None)

    '''
    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            #cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2, lineType=cv2.LINE_AA)
            #cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            #cv2.circle(frame, points[partB], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            pass
    '''

    def dist(a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1]) ** 2) ** 0.5

    THRESHOLD = 45 
    for key, pair in INPUT_PAIRS.items():
        base, tip = points[pair[0]], points[pair[1]]
        if base is not None and tip is not None:
            distance = dist(base, tip)
            print('{} -> {}'.format(key, distance))
            if distance < THRESHOLD or (key == 'thumb' and distance < THRESHOLD*1.5):
                cv2.circle(frame, tip, 15, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
                if key is not 'thumb':
                    mainMenu = False
                else:
                    mainMenu = True


            else:
                if mainMenu:
                    cv2.putText(frame, mainMenuItem[key], tip,
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                else:
                    #cv2.putText(frame, subMenuItem[subMenu])
                    pass
                  

    print("Time Taken for frame = {}".format(time.time() - t))

    #cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    #cv2.putText(frame, "Hand Pose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    cv2.imshow('Output-Skeleton', frame)
    cv2.imshow('copy', frameCopy)
    # cv2.imwrite("video_output/{:03d}.jpg".format(k), frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

    print("total = {}".format(time.time() - t))

    vid_writer.write(frame)

vid_writer.release()
