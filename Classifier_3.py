#Taine de Buys 
#DBYTAI001 
#ML Project EEE4114F 

#time import
from datetime import datetime 

#imports that are needed for plots
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np 

#ML imports 
import torch 
import torch.nn as nn
import torch.optim as optim 
import torch.utils.data as data
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader
from torchvision import datasets 
from torchvision.transforms import ToTensor 
from PIL import Image 

#flattening data for models 
def flatten(inp):
    return inp.reshape(-1) 
transform = transforms.Compose([transforms.ToTensor() , flatten])  

print("Hello World") 
#Reading in the training data 
transformation = transforms.Compose([
    transforms.Resize(300) , #defines length of shortest size of image , other size is scaled to match aspect ratio 
    transforms.CenterCrop(280), #Crops center of image to give a 280 by 280 output 
    transforms.ToTensor() ,
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

#loading training data 
training_data = datasets.ImageFolder(r"C:\Users\Taine\OneDrive\Documents\UCT2022\Semester 1\EEE4114F\ML-Project\data\training_set", transform=transformation) 

validation_data = datasets.ImageFolder(r"C:\Users\Taine\OneDrive\Documents\UCT2022\Semester 1\EEE4114F\ML-Project\data\validation_set", transform = transformation)

print("Size of training data: " , len(training_data)) 


print("Size of validation data: " , len(validation_data)) 


batch_size = 100 #numbers passsed before backpropagation occurs  
#preparing data for training with data loader, reshuffles the data at every epoch if shuffle = True 
train_dataLoader = DataLoader(training_data , batch_size = batch_size, shuffle = True) 
validation_dataLoader = DataLoader(validation_data , batch_size = len(validation_data)) 

print ("Batch size before back propogation: " , batch_size) 

#defining models 
p = 280 
h = 1024
num_labels = 4 

model= torch.nn.Sequential(nn.Linear(p, 1024),nn.ReLU() ,nn.Linear(1024 ,64 ), nn.Tanh() , nn.Linear(64 ,num_labels) , nn.LogSoftmax(dim =1))

#model= torch.nn.Sequential(nn.Linear(p, h),nn.ReLU() ,nn.Linear(h ,64 ), nn.ReLU() , nn.Linear(64 ,k)) 


#model= torch.nn.Sequential(nn.Linear(p, h),nn.Tanh() ,nn.Linear(h ,64 ), nn.ReLU() , nn.Linear(64 ,k)) 

#model= torch.nn.Sequential(nn.Linear(p, h),nn.ReLU() ,nn.Linear(h ,64 ), nn.Sigmoid() , nn.Linear(64 ,k)) 


#setting number of epochs it will be trained for and arrays of values to store
n_epoch = 10
loss_values = [] #stores loss values 
acc_values = [] #stores accuracy values 
loss_fn = torch.nn.NLLLoss() #defines loss function 
#optimizer = optim.SGD(model.parameters(), lr = 0.001) #defines optimizer used 
optimizer = optim.Adam(model.parameters())

#setting up gpu to pass model through 
device ="cuda" if torch.cuda.is_available() else "cpu"
print( "using {} device".format(device)) 




#defining train function 
def train(dataLoader, model , loss_fn , optimizer, n_epoch = n_epoch): 
   #loop dependant on epochs 
   for epoch in range (n_epoch):
        print(f"Epoch {epoch +1}\n --------------------") 
        epoch_loss=[] #stores values of this specific epochs loss 
        size = len(dataLoader.dataset)
        
        for batch, (X,y) in enumerate(dataLoader): 
            X,y =X.to(device) , y.to(device)  #sends the values read in to the cpu/gpu 
            print(X.size())
            pred = model.forward(X) 
            loss = loss_fn(pred,y)
            epoch_loss.append(loss.detach()) #appends loss value to epoch loss array 

            #Backpropogation 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

        loss_values.append(torch.tensor(epoch_loss).mean()) 
        print("Average loss of training data in epoch: ", round(loss_values[epoch].item(),5) )
        #validation of model after each test 
        validate(validation_dataLoader,model)    

def validate(dataLoader , model):
    size = len(dataLoader.dataset) 
    model.eval()
    test_loss , correct = 0 , 0 
    with torch.no_grad():
        for X, y in dataLoader:
            X , y = X.to(device) , y.to(device) 
            pred = model(X) 
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) ==y).type(torch.float).sum().item()
    test_loss/= size
    correct/=size 

    print( "Accuracy on data: ", correct*100 , "%" ) 
    acc_values.append(correct) 


#passing variables made into the function
start = datetime.now()
train(train_dataLoader, model, loss_fn , optimizer, n_epoch) 
end = datetime.now() 
print("Time taken to complete training and test validation data across Epochs: " , (end-start) ) 
plt.title("Loss of model on training set")
plt.xlabel("Epoch") 
plt.plot(loss_values) 
plt.show() 
plt.title("Accuracy of model on validation set") 
plt.xlabel("Epoch") 
plt.show() 

#test the output on the test set 
print ("\n") 
print("Testing algorithm on unseen test data set") 
 

#Allow for user input 
print("Done!")
Running = True # boolean to control input 
while (Running == True):  
    filepath = input("Please enter a filepath: \n ")  
    if (filepath == "exit"): 
        print ("Exiting...") 
        quit() 
    else:
        try:
            img =Image.open(filepath) 
            tensor = transform(img)
            tensor = tensor[None, :] 
            pred = model(tensor)
            output=torch.argmax(pred)
            print("Classifier: " , output.item())
        except FileNotFoundError:
            print ("The inputted file or command is incorrect, try again")
            continue 
