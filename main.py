from neural_network.network import NeuralNetwork
from employees.employee_identifier import EmployeeIdentifier
from employees.employee_initializer import EmployeeInitializer

CAM_1_URL = "http://192.168.1.16:8081/video"

def main():

    # y = EmployeeIdentifier(CAM_1_URL, 1)
    # print(y.identify_employees_on_frame())

    # x = NeuralNetwork()
    # x.train_network()
    
    x = EmployeeInitializer(CAM_1_URL,1 ,0)
    x.save_employee_photo()

if __name__ == "__main__":
    main()
