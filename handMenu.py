#!/usr/bin/python3
import sys
import cv2

class HandMenu():
    def __init__(self, input_video_file, output_video_file):

        # video input and output
        self.cap = cv2.VideoCapture(input_video_file)
        _, frame = self.cap.read()
        self.frameWidth = frame.shape[1]
        self.frameHeight = frame.shape[0]
        self.vid_writer = cv2.VideoWriter(output_video_file,
                                     cv2.VideoWriter_fourcc('M','J','P','G'),
                                     20,
                                     (frame.shape[1],frame.shape[0]))
        # handnet logic
        protoFile = "model/pose_deploy.prototxt"
        weightsFile = "model/pose_iter_102000.caffemodel"
        self.nPoints = 22
        aspect_ratio = self.frameWidth/self.frameHeight
        self.inHeight = 368
        self.inWidth = int(((aspect_ratio*self.inHeight)*8)//8)
        self.net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
        self.PROB_THRESHOLD = 0.2
        self.POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
                       [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],
                       [14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

        # menu interaction
        self.INPUT_PAIRS = {'thumb':(1,4), 
                            'index':(5,8),
                            'middle':(9,12),
                            'ring':(13,16),
                            'pinky':(17,20) }

        self.MENU_KEYS = {finger:False for finger in self.INPUT_PAIRS.keys()} 

        self.INPUT_THRESHOLD = 45 #pixels
        self.last_input = None
        self.on_main_menu = True
        self.sub_menu = 0
        self.sub_menu_triggers = {'thumb':0, 'index':1, 'middle':2, 'ring':3, 'pinky':4}

        # menu text
        self.main_menu_text = {'thumb': 'Friday',
                               'index': 'Thursday',
                               'middle': 'Wednesday',
                               'ring': 'Tuesday',
                               'pinky': 'Monday'}

        self.sub_menu_text = [  #Friday
                               {
                                'index': 'Thursday',
                                'middle': 'Wednesday',
                                'ring': 'Tuesday',
                                'pinky': 'Monday'},
                                #Thursday
                               {
                                'index': 'Vacuum',
                                'middle': 'Homework',
                                'ring': 'AR Project',
                                'pinky': 'Feed cats'},
                                #Wednesday
                               {
                                'index': 'Thursday',
                                'middle': 'Wednesday',
                                'ring': 'Tuesday',
                                'pinky': 'Monday'},
                                #Tuesday
                               {
                                'index': 'Grocery Store',
                                'middle': 'Trash Out',
                                'ring': 'Dishes Cleaned',
                                'pinky': 'Walk Dog'},
                                #Monday
                               {
                                'index': 'Thursday',
                                'middle': 'Wednesday',
                                'ring': 'Tuesday',
                                'pinky': 'Monday'} ]

        self.sub_menu_text = [ {key:(False, value) for key, value in m.items() } for m in self.sub_menu_text]

    def run(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        blob = cv2.dnn.blobFromImage(frame, 
                                     1.0/255,
                                     (self.inWidth, self.inHeight),
                                     (0, 0, 0),
                                     swapRB=False, crop=False)
        self.net.setInput(blob)
        output = self.net.forward()
        self.points = [None]*self.nPoints
        for p in range(self.nPoints):
            prob_map = cv2.resize(output[0, p, :, :], 
                                  (self.frameWidth, self.frameHeight))
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
            if prob > self.PROB_THRESHOLD:
                self.points[p] = tuple(map(int, point[:2]))

        self.detect_menu_inputs()
        self.show_menu(frame)
        return frame

    def detect_menu_inputs(self):

        def dist(a, b):
            return ((a[0] - b[0])**2 + (a[1] - b[1]) ** 2) ** 0.5

        for finger, pair in self.INPUT_PAIRS.items():
            base, tip = self.points[pair[0]], self.points[pair[1]]
            if base is not None and tip is not None:
                distance = dist(base, tip)
                threshold = self.INPUT_THRESHOLD*1.5 if finger is 'thumb' else self.INPUT_THRESHOLD 
                if distance < threshold:
                    #new input
                    if not self.MENU_KEYS[finger]:
                        self.MENU_KEYS[finger] = True 
                        if finger == 'thumb':
                            self.on_main_menu = True
                        else:
                            if self.on_main_menu:
                                self.on_main_menu = False
                                self.sub_menu = self.sub_menu_triggers[finger] 
                            else:
                                toggle = self.sub_menu_text[self.sub_menu][finger][0] ^ 1
                                text = self.sub_menu_text[self.sub_menu][finger][1]
                                self.sub_menu_text[self.sub_menu][finger] = (toggle, text)
                elif distance > threshold * 1.5:
                    self.MENU_KEYS[finger] = False 

            

    def show_menu(self, frame):
        for finger, pair in self.INPUT_PAIRS.items():
            base, tip = self.points[pair[0]], self.points[pair[1]]
            if base is not None and tip is not None:
                if self.on_main_menu:
                    cv2.putText(frame,
                                self.main_menu_text[finger],
                                tip,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255,0,0),
                                2)      
                elif finger == 'thumb':
                    cv2.putText(frame,
                                'BACK',
                                tip,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255,0,0),
                                2)      
                else:
                    enabled, text = self.sub_menu_text[self.sub_menu][finger]
                    cv2.putText(frame,
                                text,
                                tip,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.75,
                                (0,255,0) if enabled else (0,0,255),
                                2)     


if __name__ == '__main__':

    hand_menu = HandMenu(sys.argv[1], sys.argv[2])    
    while True:
        frame = hand_menu.run()
        if frame is None:
            break
        cv2.imshow('AR Hand Menu', frame)
        hand_menu.vid_writer.write(frame)
        #[hand_menu.cap.read()for i in range(5)] 

        key = cv2.waitKey(1)
        if key & 0xFF in [ord('q'), 27]:
            break

