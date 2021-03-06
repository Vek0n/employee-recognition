import cv2
from neural_network.network import NeuralNetwork
from employees.employee_utils import read_coords_file, get_boundingbox_coords
from config import IMG_SIZE

class EmployeeIdentifier:

    def __init__(self):
        self.nn = NeuralNetwork()
        self.model = self.nn.get_model()


    def identify_employees_on_frame(self, cam_url):
        list_of_predictions = []
        cap = cv2.VideoCapture(cam_url)
        ret, frame = cap.read()
        lines = read_coords_file()
        if lines:
            for l in lines:
                x,y,w,h = get_boundingbox_coords(l)
                crop_img = frame[y:y+h, x:x+w]
                crop_img = cv2.resize(crop_img, (IMG_SIZE, IMG_SIZE))
                list_of_predictions.append(
                    self.nn.get_predictions_for_image(crop_img, self.model)
                )
        cap.release()
        return list_of_predictions