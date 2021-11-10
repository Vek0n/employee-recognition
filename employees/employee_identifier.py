import cv2
from neural_network.network import NeuralNetwork
from employees.employee_utils import read_coords_file, get_boundingbox_coords
from neural_network.network_config import IMG_DIM

class EmployeeIdentifier:

    def __init__(self, cam_url, cam_id):
        self.nn = NeuralNetwork()
        self.model = self.nn.get_model()
        self.cam_url = cam_url
        self.cam_id = cam_id


    def identify_employees_on_frame(self):
        list_of_predictions = []
        # cap = cv2.VideoCapture(self.cam_url)
        # ret, frame = cap.read()
        lines = [0] 
        # lines = read_coords_file(self.cam_id)
        if lines:
            for l in lines:
                # x,y,w,h = get_boundingbox_coords(l)
                crop_img = cv2.imread("img.jpg")
                # crop_img = frame[y:y+h, x:x+w] 
                crop_img = cv2.resize(crop_img, IMG_DIM)
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                list_of_predictions.append(
                    self.nn.get_predictions_for_image(crop_img, self.model)
                )
        # cap.release()
        return list_of_predictions