# employee_recognition
 
 ### NeuralNetwork() class
- `train_network()` - starts training of neural netowork with data from `neural_network/data/` directory, with hyperparameters declared in `network_config.py`
 
 ### EmployeeInitializer(cam_url, cam_id, employee_id) class
- `save_employee_photo()` - saves one cropped image of employee in `neural_netowrk/data/train/{employee_id}/` directory from current frame of a livestream at `cam_url`. With every call of this method image names are incremented like so: `{employee_id}_{iterator}`

 ### EmployeeIdentifier(cam_url, cam_id) class
 - `identify_employees_on_frame()` - this method returns list of vectors. Each vector corresponds to one detected person on the frame and contains probabilities of being in given class. By class it is meant `employee_id`.
 
 **Example:**
 
 *Given output*: `[[0.003, 0.123, 0.851, 0,054],[0.783, 0.153, 0,007, 0,193]]`
 
 *What does it mean:* In the current frame there are two employees: first one is `employee_id = 2` and second is `employee_id = 0`.
 
 
 ### How to sort training data
 To start training place data in `neural_network/data/` directory:
 ```
 data
 |------train
 |      |---class1
 |      |---class2 ...
 |
 |------test
        |---class1
        |---class2 ...
 ```
