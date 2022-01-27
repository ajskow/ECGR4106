#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
from ptflops import get_model_complexity_info
from torchvision import models
from torchvision import transforms
from PIL import Image


# In[3]:


#Problem 1
#ImageNet Classification of random images using ResNet101 Model
resnet = models.resnet101(pretrained=True)


# In[4]:


#Preprocessing Pipeline for input images
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)])


# In[5]:


#Import chosen images
img1 = Image.open(r"C:\Users\Aaron\Downloads\4106_HW0_Pictures\deer.jpg")
img2 = Image.open(r"C:\Users\Aaron\Downloads\4106_HW0_Pictures\elephant.jpg")
img3 = Image.open(r"C:\Users\Aaron\Downloads\4106_HW0_Pictures\fox.jpg")
img4 = Image.open(r"C:\Users\Aaron\Downloads\4106_HW0_Pictures\parrot.jpg")
img5 = Image.open(r"C:\Users\Aaron\Downloads\4106_HW0_Pictures\turtle.jpg")


# In[6]:


#Run images through pipeline
img1_t = preprocess(img1)
img2_t = preprocess(img2)
img3_t = preprocess(img3)
img4_t = preprocess(img4)
img5_t = preprocess(img5)


# In[7]:


#Reformat image tensors
batch1_t = torch.unsqueeze(img1_t, 0)
batch2_t = torch.unsqueeze(img2_t, 0)
batch3_t = torch.unsqueeze(img3_t, 0)
batch4_t = torch.unsqueeze(img4_t, 0)
batch5_t = torch.unsqueeze(img5_t, 0)


# In[8]:


resnet.eval()


# In[9]:


#Run images through ResNet101 model
out1 = resnet(batch1_t)
out2 = resnet(batch2_t)
out3 = resnet(batch3_t)
out4 = resnet(batch4_t)
out5 = resnet(batch5_t)


# In[10]:


#Determine class labels from imageNet database
with open(r'C:\Users\Aaron\Downloads\4106_HW0_Pictures\imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]


# In[11]:


#Determine highest scoring class for image/print label
_, index = torch.max(out1, 1)
percentage = torch.nn.functional.softmax(out1, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()


# In[12]:


#Determine top 5 highest scoring classes
_, indices = torch.sort(out1, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[13]:


#Determine highest scoring class for image/print label
_, index = torch.max(out2, 1)
percentage = torch.nn.functional.softmax(out2, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()


# In[14]:


#Determine top 5 highest scoring classes
_, indices = torch.sort(out2, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[15]:


#Determine highest scoring class for image/print label
_, index = torch.max(out3, 1)
percentage = torch.nn.functional.softmax(out3, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()


# In[16]:


#Determine top 5 highest scoring classes
_, indices = torch.sort(out3, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[17]:


#Determine highest scoring class for image/print label
_, index = torch.max(out4, 1)
percentage = torch.nn.functional.softmax(out4, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()


# In[18]:


#Determine top 5 highest scoring classes
_, indices = torch.sort(out4, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[19]:


#Determine highest scoring class for image/print label
_, index = torch.max(out5, 1)
percentage = torch.nn.functional.softmax(out5, dim=1)[0] * 100
labels[index[0]], percentage[index[0]].item()


# In[20]:


#Determine top 5 highest scoring classes
_, indices = torch.sort(out5, descending=True)
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[ ]:





# In[21]:


#Problem 2
#ResNetGenerator class code provided from textbook git repository
class ResNetBlock(nn.Module):

    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=True),
                       nn.InstanceNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x) # <2>
        return out


# In[22]:


class ResNetGenerator(nn.Module):

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):

        assert(n_blocks >= 0)
        super(ResNetGenerator, self).__init__()

        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=True),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=True),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResNetBlock(ngf * mult)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input): # <3>
        return self.model(input)


# In[23]:


netG = ResNetGenerator()


# In[24]:


#Load pretraining weights for ResNetG model
model_path = r'C:\Users\Aaron\Downloads\4106_HW0_Pictures\horse2zebra_0.4.0.pth'
model_data = torch.load(model_path)
netG.load_state_dict(model_data)


# In[25]:


netG.eval()


# In[26]:


#Define preprocessing pipeline
preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.ToTensor()])


# In[27]:


#Import input images
img6 = Image.open(r'C:\Users\Aaron\Downloads\4106_HW0_Pictures\horse1.jpg')
img7 = Image.open(r'C:\Users\Aaron\Downloads\4106_HW0_Pictures\horse2.jpg')
img8 = Image.open(r'C:\Users\Aaron\Downloads\4106_HW0_Pictures\horses3.jpg')
img9 = Image.open(r'C:\Users\Aaron\Downloads\4106_HW0_Pictures\horse4.jpg')
img10 = Image.open(r'C:\Users\Aaron\Downloads\4106_HW0_Pictures\horse5.jpg')


# In[28]:


#Preprocess input images
img6_t = preprocess(img6)
img7_t = preprocess(img7)
img8_t = preprocess(img8)
img9_t = preprocess(img9)
img10_t = preprocess(img10)


# In[29]:


#Reformat image tensors
batch6_t = torch.unsqueeze(img6_t, 0)
batch7_t = torch.unsqueeze(img7_t, 0)
batch8_t = torch.unsqueeze(img8_t, 0)
batch9_t = torch.unsqueeze(img9_t, 0)
batch10_t = torch.unsqueeze(img10_t, 0)


# In[30]:


#Apply ResNetG model to each input
batch_out6 = netG(batch6_t)
batch_out7 = netG(batch7_t)
batch_out8 = netG(batch8_t)
batch_out9 = netG(batch9_t)
batch_out10 = netG(batch10_t)


# In[31]:


#Image 1
out6_t = (batch_out6.data.squeeze() + 1.0) / 2.0
out6_img = transforms.ToPILImage()(out6_t)
out6_img


# In[32]:


#Image 2
out7_t = (batch_out7.data.squeeze() + 1.0) / 2.0
out7_img = transforms.ToPILImage()(out7_t)
out7_img


# In[33]:


#Image 3
out8_t = (batch_out8.data.squeeze() + 1.0) / 2.0
out8_img = transforms.ToPILImage()(out8_t)
out8_img


# In[34]:


#Image 4
out9_t = (batch_out9.data.squeeze() + 1.0) / 2.0
out9_img = transforms.ToPILImage()(out9_t)
out9_img


# In[35]:


#Image 5
out10_t = (batch_out10.data.squeeze() + 1.0) / 2.0
out10_img = transforms.ToPILImage()(out10_t)
out10_img


# In[ ]:





# In[36]:


#Problem 3
#Obtain MACs/Model size for ResNet101
with torch.cuda.device(0):
  macs, params = get_model_complexity_info(resnet, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[37]:


#Obtain MACs/Model size for ResNetG(netG)
with torch.cuda.device(0):
  macs, params = get_model_complexity_info(netG, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[ ]:





# In[38]:


#Problem 4
#Reclassify inputs from problem 1 using MobileNet v2
mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)


# In[39]:


mobilenet.eval()


# In[40]:


#Apply model to inputs from problem 1
out11 = mobilenet(batch1_t)
out12 = mobilenet(batch2_t)
out13 = mobilenet(batch3_t)
out14 = mobilenet(batch4_t)
out15 = mobilenet(batch5_t)


# In[41]:


#Determine top 5 highest scoring classes with labels
_, indices = torch.sort(out11, descending=True)
percentage = torch.nn.functional.softmax(out11, dim=1)[0] * 100
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[42]:


#Determine top 5 highest scoring classes with labels
_, indices = torch.sort(out12, descending=True)
percentage = torch.nn.functional.softmax(out12, dim=1)[0] * 100
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[43]:


#Determine top 5 highest scoring classes with labels
_, indices = torch.sort(out13, descending=True)
percentage = torch.nn.functional.softmax(out13, dim=1)[0] * 100
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[44]:


#Determine top 5 highest scoring classes with labels
_, indices = torch.sort(out14, descending=True)
percentage = torch.nn.functional.softmax(out14, dim=1)[0] * 100
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[45]:


#Determine top 5 highest scoring classes with labels
_, indices = torch.sort(out15, descending=True)
percentage = torch.nn.functional.softmax(out15, dim=1)[0] * 100
[(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]


# In[46]:


#Obtain MACs/Model size for MobileNet v2
with torch.cuda.device(0):
  macs, params = get_model_complexity_info(mobilenet, (3, 224, 224), as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# In[ ]:




