from NeuralNetwork import NeuralNetwork
from EmployeeIdentifier import EmployeeIdentifier
from utils import config
def main():

    y = EmployeeIdentifier()
    y.identify_employees_on_frame(config.CAM_1_URL)
    # x = NeuralNetwork()
    # x.train_network()

if __name__ == "__main__":
    main()
