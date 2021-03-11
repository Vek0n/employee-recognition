from neural_network.network import NeuralNetwork
from employees.employee_identifier import EmployeeIdentifier
from employees.employee_initializer import EmployeeInitializer
import cv2

CAM_1_URL = "http://192.168.1.16:8081/video"

def main():

    # y = EmployeeIdentifier(CAM_1_URL, 1)
    # print(y.identify_employees_on_frame())

    x = NeuralNetwork()
    x.train_network()
    
    # model = x.get_model()
    # img = cv2.imread("1.png")
    # img = cv2.resize(img, (224,224))
    # print(x.get_predictions_for_image(image = img, model = model))
    
    
    # x = EmployeeInitializer(CAM_1_URL, cam_id = 1 ,employee_id = 3)
    # x.save_employee_photo()

if __name__ == "__main__":
    main()
