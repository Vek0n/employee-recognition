import cv2
from NeuralNetwork import NeuralNetwork
from utils import config

class EmployeeIdentifier:

    def __init__(self):
        self.nn = NeuralNetwork()
        self.model = self.nn.get_model()

    def read_coords_file(self):
        with open('cam1.coords', 'r') as reader:
            raw_data = reader.readlines()
        lines = []
        for i in raw_data:
            line = list(map(int, i.split()))
            lines.append(line)
        return lines


    def get_boundingbox_coords(self, line):
        w = line[2] - line[0]
        h = line[3] - line[1]
        y = line[1]
        x = line[0]
        return x,y,w,h


    def identify_employees_on_frame(self, cam_url):
        list_of_predictions = []
        cap = cv2.VideoCapture(cam_url)
        ret, frame = cap.read()
        lines = self.read_coords_file()
        for l in lines:
            x,y,w,h = self.get_boundingbox_coords(l)
            crop_img = frame[y:y+h, x:x+w]
            crop_img = cv2.resize(crop_img, (config.IMG_SIZE, config.IMG_SIZE))
            list_of_predictions.append(
                self.nn.get_predictions_for_image(crop_img, self.model)
            )

        cap.release()
        return list_of_predictions

