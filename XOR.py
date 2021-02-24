#!/usr/bin/env python
# coding: utf-8

# In[126]:


import numpy as np


# In[127]:


########################################## FUNCTIONS ##########################################################


# In[128]:


### ACTIVATION ###

def Activation(a,x):
    yhat=1/(1+np.exp(-a*x)) 
    return yhat  


# In[129]:


### fprime ###

def differenceActivation(a,x):
    fprime=a*Activation(a,x)*(1-Activation(a,x))
    return fprime


# In[130]:


# Forward function
def forward(x,w1,w2,predict=False):
    v_input=np.matmul(x,w1)
    x2=Activation(a,v_input)
    
    #create and add bias
    bias=np.ones((len(x2),1))
    x2=np.concatenate((bias,x2),axis=1)
    
    v_hidden=np.matmul(x2,w2)
    yhat=Activation(a,v_hidden)
    if predict:
        return yhat
    return v_input,x2,v_hidden,yhat   


# In[131]:


# Back propagation
def backprop(v_hidden,x,x2,yhat,y):
    e2=yhat-y
    Delta2=np.matmul(x2.T,e2)
    e1=(e2.dot(w_hidden[1:,:].T))*differenceActivation(a,v_input)
    Delta1=np.matmul(x.T,e1)
    return e2,Delta1,Delta2   


# In[132]:


########################################## Architecture ##########################################################

#hidden layer=1
i=2  #number of input neurons excluding bias
j=5  #number of hidden neurons excluding bias 
o=1  #number of output neurons


# In[133]:


########################################## Initializations ##########################################################


# In[134]:


# XOR inputs

x=np.array([[1,-1,-1],
            [1,-1,1],
            [1,1,-1],
            [1,1,1]])
x


# In[135]:


# XOR Outputs

y=np.array([[0],[1],[1],[0]])
y


# In[136]:


# weight_input

w_input=np.random.random((i+1,j))
'''w_input=np.array([[0.34236424, 0.07035245],
                     [0.78841577, 0.19436278],
                     [0.29103036, 0.1711286 ]])'''
w_input


# In[137]:


# weight_hidden

w_hidden=np.random.random((j+1,o))
'''w_hidden=np.array([[0.39806725],
                      [0.82368992],
                      [0.03316563]])'''
w_hidden


# In[138]:


lr=0.09

a=1


# In[139]:


costs=[]

epochs=5000

m=len(x)
#m=1


# In[140]:


for i in range (epochs):
    #forward
    v_input,x2,v_hidden,yhat=forward(x,w_input,w_hidden)
    #backprop
    e2,Delta1,Delta2=backprop(v_hidden,x,x2,yhat,y)
    
    w_input -= lr*(1/m)*Delta1
    w_hidden-= lr*(1/m)*Delta2
    
    #add cost to c for plotting
    c=np.mean(np.abs(e2))
    costs.append(c)
    
    if i % 500 == 0:
        print(f"Iterartion:{i}.Error:{c}")

# Training complete
print("Training complete")

#Make predictions
output=forward(x,w_input,w_hidden,True)
print("Precentages: ")
print(output)
print("Predictions: ")
print(np.round(output))


# In[141]:


############################################### PLOT ####################################################
import matplotlib.pyplot as plt
import pandas as pd

#pot cost
plt.plot(costs)
plt.show()   
    

