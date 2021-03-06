from neural_network.network import NeuralNetwork
from employees.employee_identifier import EmployeeIdentifier
from employees.employee_initializer import EmployeeInitializer
from config import CAM_1_URL


def main():

    # y = EmployeeIdentifier()
    # print(y.identify_employees_on_frame(CAM_1_URL))

    # x = NeuralNetwork()
    # x.train_network()
    
    x = EmployeeInitializer(CAM_1_URL, "0")
    x.save_employee_photo()
    x.save_employee_photo()
    x.save_employee_photo()

if __name__ == "__main__":
    main()
