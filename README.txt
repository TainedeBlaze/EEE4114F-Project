README FOR ML ASSIGNMENT 1 
TAINE DE BUYS -DBYTAI001

Upon unzipping the tarbell, the following line will build the virtual enviroment needed to execute the python scriptsand include all relevant packages once someone is in the source directory in their commandline:
$make 

Once this is executed, the following command activates the virtual enviroment. This is needed to ensure that the python files can be executed: 
$source ./venv/bin/activate 

Once the vitrual enviroment is activated, (venv) will appear in brackets next to your terminals command prompt. 

To train the 1st model which consists of a single layer linear activation function, and see its validation results, type the following: 
$python3 Classifier1.py 

To train the 2nd model with 3 layers, and a middle layer being the sigmoid activation function type the following 
$python3 Classifier2.py 

If desired , the type of activation function usedin the middle can be changed by opening a text editor and uncommenting the relevant model definitions. 


To train the 3rd ANN with 5 layers, the following can be typed into the command line 
$python3 Classifier_3.py 

The type of activation function used here can also be changed through uncommenting different models in the source code. 


After the results are displayed for the specific model, the user may pick any image path to push through the algorithm, for simplicity, a testSamples folder has been added with 20 images that can be passsed through, and will recieve the models expected output. E.g. 
Please Enter a filepath: 
$./testSamples/img_1.jpg 
Classifier: 3 

The user may end the program through typing exit when prompted to enter a file path, like so: 
Please Enter a filepath: 
$exit 
Exiting... 


NOTE The virtual enviroment as well as all generate python bytecode can be deleted through typing the following:
$make clean 


