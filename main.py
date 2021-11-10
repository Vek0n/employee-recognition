from neural_network.network import NeuralNetwork
from employees.employee_identifier import EmployeeIdentifier
from employees.employee_initializer import EmployeeInitializer
import cv2
import time

CAM_1_URL = "http://192.168.2.103:8081/video"

def main():
    #1 - train, 2 - test, 3 - gather data
    mode = 1
    x = NeuralNetwork()
    if mode == 1:
        x.train_network()
    elif mode == 2:
        model = x.get_model()
        img = cv2.imread("3_60.jpg")
        img = cv2.resize(img, (224,224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(x.get_predictions_for_image(image = img, model = model))
    elif mode == 3:
        y = EmployeeInitializer(CAM_1_URL, cam_id = 1 ,employee_id = 3)
        time.sleep(1)
        for i in range(9999):
            # sleep(20)
            print("Saving photo number: " + str(i))
            y.save_employee_photo()
            time.sleep(0.1)
        
        
if __name__ == "__main__":
    main()







    # start1 = time.time() 
    
    # end1 = time.time()
    # print(end1 - start1) 
    # for i in range(50):
    # start = time.time()
    
    # end = time.time()
    # print(end - start)
 
    # y = EmployeeIdentifier(CAM_1_URL, 1)
    # emp = y.identify_employees_on_frame()
    # print(emp)
    
    # x = NeuralNetwork()
    # x.train_network()