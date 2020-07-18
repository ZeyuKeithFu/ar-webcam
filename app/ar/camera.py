import cv2
import os
import numpy as np
from ar.obj_loader import *
from ar.renderer import *

class ArCamera(object):

    def __init__(self):
        dir_name = os.path.dirname(os.path.abspath(__file__))
        self.camera_parameters = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])
        # Load reference model
        self.model = cv2.flip(cv2.imread(os.path.join(dir_name, 'img/model.jpg'), 0), 1)
        # Create ORB keypoint detector
        self.orb = cv2.ORB_create()
        # Create BFMatcher object based on hamming distance  
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Compute model keypoints and its descriptors
        self.kp_model, self.des_model = self.orb.detectAndCompute(self.model, None)
        # Load AR object
        self.obj = OBJ(os.path.join(dir_name, 'obj/fox.obj'), swapyz=True)
        # Init video capture
        self.cap = cv2.VideoCapture(0)
    
    def __del__(self):
        self.cap.release()
    
    def get_frame(self):
        """
        Get rendered AR frame
        """
        ret, frame = self.cap.read()
        frame = render(frame, self.orb, self.bf, self.model,
            self.kp_model, self.des_model, self.obj, self.camera_parameters, box=True)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
