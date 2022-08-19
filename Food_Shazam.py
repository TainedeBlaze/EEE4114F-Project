#Taine de Buys 
#DBYTAI001 
#ML Project EEE4114F 
from __future__ import print_function
from datetime import datetime
from logging import root 

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
import torchvision.models as models


from PIL import Image 

x=torch.rand(2, 3) 
print(x) 


#defining transforms for imported images and resizes all of them to be the same 
transformation = transforms.Compose([
    transforms.Resize(300) , #defines length of shortest size of image , other size is scaled to match aspect ratio 
    transforms.CenterCrop(280), #Crops center of image to give a 280 by 280 output 
    transforms.ToTensor() 
    ])

#loading training data 
training_set = datasets.ImageFolder(r"C:\Users\Taine\OneDrive\Documents\UCT2022\Semester 1\EEE4114F\ML-Project\data\training_set", transform=transformation) 

validation_set = datasets.ImageFolder(r"C:\Users\Taine\OneDrive\Documents\UCT2022\Semester 1\EEE4114F\ML-Project\data\validation_set", transform = transformation)

test_set = datasets.ImageFolder(r"C:\Users\Taine\OneDrive\Documents\UCT2022\Semester 1\EEE4114F\ML-Project\data\test_set", transform = transformation)

#print statements to see size of data 
print("training set size: ", len(training_set))
print("test set size: ", len(validation_set))

#delcaring batch size to send into the data loaders 
batch_size = 200

#creating data loaders 
train_loader = DataLoader(training_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(validation_set , batch_size=batch_size , shuffle =True )
realtestloader = DataLoader(test_set , batch_size=batch_size , shuffle =True )

print ("Batch size before back propogation: " , batch_size) 


#Get pretrained model using torchvision.models as models library
model = models.densenet161(pretrained=True)
# Turn off training for their parameters
for param in model.parameters():
    param.requires_grad = False


#define model to be fed after convolutional model 
classifier_input = model.classifier.in_features 
num_labels =4 

classifier = torch.nn.Sequential(nn.Linear(classifier_input, 1024),nn.ReLU() ,nn.Linear(1024 ,4 ) , nn.LogSoftmax(dim =1)) 

# Replace default classifier with new classifier
model.classifier = classifier 

#setting up gpu to pass model through 
device ="cuda" if torch.cuda.is_available() else "cpu"

print( "using {} device".format(device)) 

#setting number of epochs it will be trained for and arrays of values to store
n_epoch = 10

loss_fn = torch.nn.CrossEntropyLoss() #defines loss function 
#optimizer = optim.SGD(model.parameters(), lr = 0.001) #defines optimizer used 
optimizer = optim.Adam(model.parameters())


epochs = 10
for epoch in range(epochs):
    training_loss = 0 
    validation_loss = 0 
    accuracy = 0 
    # Training the model
    model.train()
    batch_num = 0
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output = model.forward(inputs)
        # Loss
        loss = loss_fn(output, labels)
        
        # Calculate gradients (backpropogation)
        loss.backward()
        
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        training_loss += loss.item()*inputs.size(0)
        
        # Print the progress of our training
        batch_num += 1
        print("Batch " , batch_num, "/", len(train_loader) ," of Epoch " ,str(epoch) ) 
        
    # Evaluating the model
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            output = model.forward(inputs)
            # Calculate Loss
            valloss = loss_fn(output, labels)
            # Add loss to the validation set's running loss
            validation_loss += valloss.item()*inputs.size(0)
            
            # Since our model outputs a LogSoftmax, find the real 
            # percentages by reversing the log function
            output = torch.exp(output)
            # Get the top class of the output
            top_p, top_class = output.topk(1, dim=1)
            # See how many of the classes were correct?
            
            equals = top_class == labels.view(*top_class.shape)
            
            # Calculate the mean (get the accuracy for this batch)
            # and add it to the running accuracy for this epoch
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Print the progress of our evaluation
            counter += 1
            print(counter, "/", len(test_loader))
    
    # Get the average loss for the entire epoch
    training_loss = training_loss/len(train_loader.dataset)
    #get validation loss for the entire epoch 
    validation_loss = validation_loss/len(test_loader.dataset)
    # Print out the information
    print('Accuracy of epoch : ', accuracy/len(test_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, training_loss, validation_loss))
    
    
print("Testing algorithm on unseen test data set") 
model.eval()
    
    # Tell torch not to calculate gradients
with torch.no_grad():
    for inputs, labels in realtestloader:
          # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            output = model.forward(inputs)
            # Calculate Loss
            valloss = loss_fn(output, labels)
            # Add loss to the validation set's running loss
            validation_loss += valloss.item()*inputs.size(0)
            
            # Since our model outputs a LogSoftmax, find the real 
            # percentages by reversing the log function
            output = torch.exp(output)
            # Get the top class of the output
            top_p, top_class = output.topk(1, dim=1)
            # See how many of the classes were correct?
            
            equals = top_class == labels.view(*top_class.shape)
            
            # Calculate the mean (get the accuracy for this batch)
            # and add it to the running accuracy for this epoch
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            # Print the progress of our evaluation
            counter += 1
            print(counter, "/", len(test_loader))
    training_loss = training_loss/len(train_loader.dataset)
    #get validation loss for the entire epoch 
    validation_loss = validation_loss/len(test_loader.dataset)
    # Print out the information
    print('Accuracy of Unseen data of size: ',len(test_set)  , accuracy/len(test_loader))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, training_loss, validation_loss))
    