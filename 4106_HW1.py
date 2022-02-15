#!/usr/bin/env python
# coding: utf-8

# In[34]:


#Aaron Skow
#ECGR 4106 HW1
#ID: 801161395
#2/13/22

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# In[3]:


#Problem 1

#Preprocessing Pipeline for input images
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
#transforms.Normalize(
#mean=[0.485, 0.456, 0.406],
#std=[0.229, 0.224, 0.225])
])


# In[4]:


#Import chosen images
img1 = Image.open("red.jpg")
img2 = Image.open("green.jpg")
img3 = Image.open("blue.jpg")


# In[5]:


#Run images through pipeline
img1_t = preprocess(img1)
img2_t = preprocess(img2)
img3_t = preprocess(img3)


# In[6]:


img1_t.shape


# In[7]:


#Mean of image tensors
torch.mean(img1_t), torch.mean(img2_t), torch.mean(img3_t)


# In[8]:


#Means of red image channels
red_c0_mean = img1_t[:, :, 0].mean()
red_c1_mean = img1_t[:, :, 1].mean()
red_c2_mean = img1_t[:, :, 2].mean()


# In[9]:


red_c0_mean


# In[10]:


red_c1_mean


# In[11]:


red_c2_mean


# In[12]:


#Means of green image channels
green_c0_mean = img2_t[:, :, 0].mean()
green_c1_mean = img2_t[:, :, 1].mean()
green_c2_mean = img2_t[:, :, 2].mean()


# In[13]:


green_c0_mean


# In[14]:


green_c1_mean


# In[15]:


green_c2_mean


# In[16]:


#Means of blue image channels
blue_c0_mean = img3_t[:, :, 0].mean()
blue_c1_mean = img3_t[:, :, 1].mean()
blue_c2_mean = img3_t[:, :, 2].mean()


# In[17]:


blue_c0_mean


# In[18]:


blue_c1_mean


# In[19]:


blue_c2_mean


# In[199]:


#Problem 2
#Import code from temperature example
t_c = [0.5,  14.0, 15.0, 28.0, 11.0,  8.0,  3.0, -4.0,  6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
w = torch.ones(())
w2 = torch.ones(())
b = torch.zeros(())


# In[200]:


#Update model equation
def model(t_u, w, w2, b):
    return w2 * t_u ** 2 + w * t_u + b


# In[201]:


def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[202]:


def dloss_fn(t_p, t_c):
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs


# In[203]:


def dmodel_dw(t_u, w, w2, b):
    return t_u


# In[204]:


#Derivative of 2nd parameter
def dmodel_dw2(t_u, w, w2, b):
    return 2 * w2 * t_u


# In[205]:


def dmodel_db(t_u, w, w2, b):
    return 1.0


# In[206]:


def grad_fn(t_u, t_c, t_p, w, w2, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, w2, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, w2, b)
    dloss_dw2 = dloss_dtp * dmodel_dw2(t_u, w, w2, b)
    return torch.stack([dloss_dw2.sum(), dloss_dw.sum(), dloss_db.sum()]) 


# In[207]:


def training_loop(n_epochs, learning_rate, params, t_u, t_c, print_params=True):
    for epoch in range(1, n_epochs + 1):
        w, w2, b = params

        t_p = model(t_u, w, w2, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, w2, b)

        params = params - learning_rate * grad
        
        if epoch in {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000}:  # <3>
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            if print_params:
                print('    Params:', params)
                print('    Grad:  ', grad)

    return params


# In[208]:


#Scale input
t_un = 0.1 * t_u


# In[209]:


#Training loops with 5000 epochs, various learning rates
params1 = training_loop(
    n_epochs = 5000, 
    learning_rate = 0.1, 
    params = torch.tensor([1.0, 1.0, 0.0]), 
    t_u = t_un, 
    t_c = t_c)


# In[210]:


params2 = training_loop(
    n_epochs = 5000, 
    learning_rate = 0.01, 
    params = torch.tensor([1.0, 1.0, 0.0]), 
    t_u = t_un, 
    t_c = t_c)


# In[213]:


params3 = training_loop(
    n_epochs = 5000, 
    learning_rate = 0.001, 
    params = torch.tensor([1.0, 1.0, 0.0]), 
    t_u = t_un, 
    t_c = t_c)


# In[211]:


params4 = training_loop(
    n_epochs = 5000, 
    learning_rate = 0.0001, 
    params = torch.tensor([1.0, 1.0, 0.0]), 
    t_u = t_un, 
    t_c = t_c)
params


# In[214]:


#Model Linear vs Non-Linear on dataset
t_p = model(t_un, *params3)

fig = plt.figure(dpi=600)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
#Non-linear model
plt.plot(np.sort(t_u.numpy()), np.sort(t_p.detach().numpy()))
#Linear model
plt.plot(np.sort(t_u.numpy()), np.sort((5.3671*t_un - 17.3012)))
#Original
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.savefig("temp_unknown_plot.png", format="png")


# In[129]:


#Problem 3
#Import housing data
housing = pd.DataFrame(pd.read_csv("Housing.csv")) 
housing.head() 


# In[215]:


#Create tensors from data and normalize
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking'] 
price = torch.tensor(housing['price'].values, dtype=torch.float64)
vars = torch.tensor(housing[num_vars].values, dtype=torch.float64)
vars = torch.nn.functional.normalize(vars,p = 2, dim = 0)
vars


# In[216]:


#Update model equation
def model(vars, params):
    return params[1] * vars[:,0] + params[2] * vars[:,1] + params[3] * vars[:,2] + params[4] * vars[:,3] + params[5] * vars[:,4] + params[0]


# In[217]:


def loss_fn(pred, price):
    squared_diffs = (pred - price)**2
    return squared_diffs.mean()


# In[218]:


def dloss_fn(pred, price):
    dsq_diffs = 2 * (pred - price) / pred.size(0)
    return dsq_diffs


# In[219]:


#Create new derivatives for new parameters
def dmodel_dw(vars, params):
    return vars[:,0]


# In[220]:


def dmodel_dw2(vars, params):
    return vars[:,1]


# In[221]:


def dmodel_dw3(vars, params):
    return vars[:,2]


# In[222]:


def dmodel_dw4(vars, params):
    return vars[:,3]


# In[223]:


def dmodel_dw5(vars, params):
    return vars[:,4]


# In[224]:


def dmodel_db(vars, params):
    return 1.0


# In[258]:


def grad_fn(vars, price, pred, params):
    dloss_dtp = dloss_fn(pred, price)
    dloss_dw = dloss_dtp * dmodel_dw(vars, params)
    dloss_dw2 = dloss_dtp * dmodel_dw2(vars, params)
    dloss_dw3 = dloss_dtp * dmodel_dw3(vars, params)
    dloss_dw4 = dloss_dtp * dmodel_dw4(vars, params)
    dloss_dw5 = dloss_dtp * dmodel_dw5(vars, params)
    dloss_db = dloss_dtp * dmodel_db(vars, params)
    return torch.stack([dloss_dw5.sum(), dloss_dw4.sum(), dloss_dw3.sum(), dloss_dw2.sum(), dloss_dw.sum(), dloss_db.sum()]) 


# In[259]:


def training_loop(n_epochs, learning_rate, params, vars, price, print_params=True):
    for epoch in range(1, n_epochs + 1):

        pred = model(vars, params)
        loss = loss_fn(pred, price)
        grad = grad_fn(vars, price, pred, params)

        params = params - learning_rate * grad
        
        if epoch in {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000}:  # <3>
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            if print_params:
                print('    Params:', params)
                print('    Grad:  ', grad)

    return params


# In[270]:


#Perform training with 5000 epochs, various learning rates
params1 = training_loop(
n_epochs = 5000,
learning_rate = 0.1,
params = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
vars = vars,
price = price)


# In[265]:


params2 = training_loop(
n_epochs = 5000,
learning_rate = 0.01,
params = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
vars = vars,
price = price)


# In[262]:


params3 = training_loop(
n_epochs = 5000,
learning_rate = 0.001,
params = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
vars = vars,
price = price)


# In[266]:


params4 = training_loop(
n_epochs = 5000,
learning_rate = 0.0001,
params = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
vars = vars,
price = price)


# In[271]:


#Plot of noramlized area vs prediced housing cost
pred = model(vars, params4)

fig = plt.figure(dpi=600)
plt.xlabel("area")
plt.ylabel("price")
#Linear model
plt.plot(np.sort(vars[:,0].numpy()), np.sort((pred.detach().numpy())))
#Input Data
plt.plot(vars[:,0].numpy(), price.numpy(), 'o')
#plt.savefig("temp_unknown_plot.png", format="png")


# In[ ]:




