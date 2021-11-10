import os
import cv2
from .employee_identifier import EmployeeIdentifier
from .employee_utils import read_coords_file, get_boundingbox_coords

class EmployeeInitializer:

    def __init__(self, cam_url, cam_id, employee_id):
        self.employee_id = employee_id
        self.cam_id = cam_id
        self.count = 0
        self.cap = cv2.VideoCapture(cam_url)


    def save_employee_photo(self):
        ret, frame = self.cap.read()
        lines = []
        try:
            lines = read_coords_file(self.cam_id)
        except:
            print("Error: Could not read coords file.")
        if lines:
            x,y,w,h = get_boundingbox_coords(lines[0]) #opakowac
            crop_img = frame[y:y+h, x:x+w]
            try:
                cv2.imwrite(
                    os.path.sep.join(
                        ["neural_network", 
                        "data", 
                        "train", 
                        str(self.employee_id), 
                        '{}_{}.jpg'.format(self.employee_id, self.count)
                        ]
                    ), crop_img
                )
            except:
                print("Error: Could not save a training photo.")
            self.count+=1