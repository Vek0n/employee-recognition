import os
import cv2
from config import CAM_1_URL
from .employee_identifier import EmployeeIdentifier
from .employee_utils import read_coords_file, get_boundingbox_coords

class EmployeeInitializer:

    def __init__(self, cam_url, id):
        self.cap = cv2.VideoCapture(cam_url)
        self.id = id
        self.is_recording = False
        self.count = 0


    def save_employee_photo(self):
        ret, frame = self.cap.read()
        lines = read_coords_file()
        x,y,w,h = get_boundingbox_coords(lines[0])
        crop_img = frame[y:y+h, x:x+w]
        cv2.imwrite(
            os.path.sep.join(
                ["data", "temp", '{}_{}.jpg'.format(self.id, self.count)]
            ), crop_img
        )
        self.count+=1