#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Aaron Skow
#ECGR 4106 HW2
#2/28/22

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as op

from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms


# In[113]:


#Problem 1
#Import housing data
housing = pd.DataFrame(pd.read_csv("Housing.csv")) 
housing.head() 


# In[114]:


#Split data into training/test sets
rand = np.random.seed(0) 
df_train, df_test = train_test_split(housing, train_size = 0.8, test_size = 0.2, random_state = rand)
df_train.head()


# In[115]:


#Create tensors from data and normalize
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

df_train_vars = df_train[num_vars]
df_train_price = df_train['price']
df_test_vars = df_test[num_vars]
df_test_price = df_test['price']

train_vars = torch.tensor(df_train_vars.values).float()
train_vars_norm = torch.nn.functional.normalize(train_vars, dim = 0)

train_price = torch.tensor(df_train_price.values).float().unsqueeze(-1)
train_price_norm = torch.nn.functional.normalize(train_price, dim = 0)

test_vars = torch.tensor(df_test_vars.values).float()
test_vars_norm = torch.nn.functional.normalize(test_vars, dim = 0)

test_price = torch.tensor(df_test_price.values).float().unsqueeze(-1)
test_price_norm = torch.nn.functional.normalize(test_price, dim = 0)


# In[120]:


#Create model with 1 hidden layer
model_1 = nn.Sequential(
          nn.Linear(5,8),
          nn.Tanh(),
          nn.Linear(8,1))


# In[121]:


optimizer = op.SGD(model_1.parameters(), lr = 0.01)


# In[130]:


def training_loop(n_epochs, optimizer, model, loss_fn, train_vars, test_vars, train_price, test_price):
    for epoch in range(1, n_epochs + 1):
        train = model(train_vars)
        train_loss = loss_fn(train, train_price)

        test = model(test_vars)
        test_loss = loss_fn(test, test_price)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f},"
                  f" Validation loss {test_loss.item():.4f}")


# In[131]:


training_loop(
    n_epochs = 200, 
    optimizer = optimizer,
    model = model_1,
    loss_fn = nn.MSELoss(),
    train_vars = train_vars_norm,
    test_vars = test_vars_norm, 
    train_price = train_price_norm,
    test_price = test_price_norm)


# In[132]:


#Create model with 2 additonal hidden layers
model_2 = nn.Sequential(
          nn.Linear(5,8),
          nn.Tanh(),
          nn.Linear(8,5),
          nn.Tanh(),
          nn.Linear(5,3),
          nn.Tanh(),
          nn.Linear(3,1))


# In[133]:


optimizer = op.SGD(model_2.parameters(), lr = 0.01)


# In[134]:


training_loop(
    n_epochs = 200, 
    optimizer = optimizer,
    model = model_2,
    loss_fn = nn.MSELoss(),
    train_vars = train_vars_norm,
    test_vars = test_vars_norm, 
    train_price = train_price_norm,
    test_price = test_price_norm)


# In[20]:


#Problem 2
#import CIFAR-10 datasets using provided textbook code
from torchvision import datasets
data_path = '../data-unversioned/p1ch7/'
transformed_cifar10 = datasets.CIFAR10(data_path, train=True, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))
transformed_cifar10_val = datasets.CIFAR10(data_path, train=False, download=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]))


# In[21]:


#Separate into training/test datasets
cifar10_train = [(img, label) for img, label in transformed_cifar10]
cifar10_test = [(img, label) for img, label in transformed_cifar10_val]


# In[22]:


#Create model with 1 hidden layer
model_3 = nn.Sequential(
          nn.Linear(3072, 512),
          nn.ReLU(),
          nn.Linear(512, 10))


# In[23]:


optimizer = op.SGD(model_3.parameters(), lr = 0.01)
loss_fn = nn.CrossEntropyLoss()


# In[24]:


#Create training/test data loaders for batching
train_loader = torch.utils.data.DataLoader(cifar10_train, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=64, shuffle=False)


# In[25]:


#Evaluate model 3
for epoch in range(200):
    for imgs, labels in train_loader:
        outputs = model_3(imgs.view(imgs.shape[0], -1))
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
      print("Epoch: %d, Loss: %f" % (epoch, float(loss)))


# In[26]:


#Calculate training accuracy for model 3
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in train_loader:
        outputs = model_3(imgs.view(imgs.shape[0], -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
        
print("Train Accuracy: %f" % (correct / total))


# In[27]:


#Calculate validation accuracy for model 3
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        outputs = model_3(imgs.view(imgs.shape[0], -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
        
print("Val Accuracy: %f" % (correct / total))


# In[35]:


#Create model with 2 additional hidden layers
model_4 = nn.Sequential(
          nn.Linear(3072, 512),
          nn.ReLU(),
          nn.Linear(512, 256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.ReLU(),
          nn.Linear(128, 10))


# In[36]:


optimizer = op.SGD(model_4.parameters(), lr = 0.01)


# In[37]:


#Evaluate model 4
for epoch in range(200):
    for imgs, labels in train_loader:
        outputs = model_4(imgs.view(imgs.shape[0], -1))
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
      print("Epoch: %d, Loss: %f" % (epoch, float(loss)))


# In[38]:


#Calculate training accuracy for model 4
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in train_loader:
        outputs = model_4(imgs.view(imgs.shape[0], -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
        
print("Train Accuracy: %f" % (correct / total))


# In[39]:


#Calculate validation accuracy for model 4
correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in val_loader:
        outputs = model_4(imgs.view(imgs.shape[0], -1))
        _, predicted = torch.max(outputs, dim=1)
        total += labels.shape[0]
        correct += int((predicted == labels).sum())
        
print("Val Accuracy: %f" % (correct / total))


# In[ ]:




