# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:55:02 2020

@author: manas
"""

'''
FUNCTION DEFINITIONS

'''

import numpy as np

def ReLU(z):
   return(np.maximum(np.zeros(np.shape(z)),z))


def ReLU_prime(z1):
    z1[z1 >0] = 1
    z1[z1<=0] = 0
    return z1

def generate(layer,nodes,features,classes):
    w=[]
    b=[]
    c = 1/np.sqrt(nodes)
    for i in range(layer-1):
        if i==0:
            w_i=np.random.uniform(-c/2,c/2,(features,nodes))
            b_i=np.random.rand(1,nodes)
        elif i==layer-2:
            w_i=np.random.uniform(-c/2,c/2,(nodes,classes))
            b_i=np.random.rand(1,classes)
        else:
            w_i=np.random.uniform(-c/2,c/2,(nodes,nodes))
            b_i=np.random.rand(1,nodes)
        w.append(w_i)
        b.append(b_i)
    return(w,b)  
    
def softmax(z):
    yhat = np.zeros(z.shape)
    m_z = np.amax(z)
    for r in range(z.shape[0]):
        yhat[r] = np.exp(z[r]-m_z)/np.sum(np.exp(z[r]-m_z))
  
    #yhat=np.exp(z) / np.sum(np.exp(z), axis=0)
    return yhat         

def fun_yhat(X_mini_batch,w,b,layer):
    h=[]
    z=[]
    for k in range(0,layer-1):
        if k==0:
            z_i=np.dot(X_mini_batch,w[k])+b[k]            
        else:         
            z_i=np.dot(h[k-1],w[k])+b[k]
        h.append(ReLU(z_i))
        z.append(z_i)

    yhat=softmax(z[k])
    h[k]=yhat
    #print(yhat)
    return(yhat,h,z)

def backprop(y_mini_batch,X_mini_batch,yhat,layer,w,b,h,z):
    g=yhat.T - y_mini_batch.T
    h[layer-2]=yhat
    grad_b=[]
    grad_w=[]
    for i in range(layer-2,-1,-1):
        g=np.multiply(g.T,ReLU_prime(z[i]))
        grad_b.append(np.sum(g,axis=0))
        if i==0:
            grad_w.append(np.dot(g.T,X_mini_batch).T)
        else:
            grad_w.append(np.dot(g.T,h[i-1]).T)
        g=np.dot(w[i],g.T)
    grad_w=grad_w[::-1]
    grad_b=grad_b[::-1]
    return(grad_w,grad_b)

def SGD(grad_w,grad_b,eps,w,b):
    w_new=[]
    b_new=[]
    for l in range(len(grad_w)):
        w_new.append(w[l]-eps*grad_w[l])
        b_new.append(b[l]-eps*grad_b[l])
    return(w_new,b_new)
    
def fce(X_mini_batch,y_mini_batch,w,alpha,yhat):
    fce_val=0.
    reg_error=0.
    for row in range(yhat.shape[0]):
        for col in range(yhat.shape[1]):
            fce_val=fce_val+y_mini_batch[row,col]*np.log(yhat[row,col])
    #print(fce_val)
    fce_val=-(fce_val)/(yhat.shape[0])    
    
    for col in range(yhat.shape[1]):
        reg_error=reg_error+np.dot(w.T[col],w[:,col])
        
    fce_val=fce_val+(alpha*reg_error/2)
    
    return(fce_val)

def accuracy(X,Y,best_w,best_b,hidden_layer):
    Y_cal,h,z=fun_yhat(X,best_w,best_b,hidden_layer)
    Y_cal=np.argmax(Y_cal, axis=1)
    y_true = np.argmax(Y, axis=1)
    accuracy1 = sum(Y_cal == y_true)/(float(len(y_true)))
    #print(accuracy1*100)
    return(accuracy1*100)
    
    
def findBestHyperparameters():
    print("-----------------Validation------------------")
    import numpy as np
    X_tr =(np.load("mnist_train_images.npy"))   
    ytr = (np.load("mnist_train_labels.npy"))
    X_val =(np.load("mnist_validation_images.npy"))    
    yval = (np.load("mnist_validation_labels.npy"))
    X_te =(np.load("mnist_test_images.npy"))    
    yte = (np.load("mnist_test_labels.npy"))
    
    n=X_tr.shape[0]

# =============================================================================
#  #range for epoch 
#     set_epochs=[20,30]
#     #range for batch size
#     set_batch=[128,256]
#     #
#     set_alpha=[0.00001,0.00005]
#     #   
#     set_learning=[0.0004,0.0008]
#     #
#     set_hidden_nodes=[30,40,50]
#     #
#     set_hidden_layers=[3,4]
#     
# =============================================================================
    #range for epoch 
    set_epochs=[30]
    #range for batch size
    set_batch=[128]
    #
    set_alpha=[0.00001]
    #   
    set_learning=[0.0004]
    #
    set_hidden_nodes=[50]
    #
    set_hidden_layers=[4]

    best_fce=10**10
    best_epoch=0
    best_batch=0
    best_hidden_node=0
    best_hidden_layers=0
    best_alpha=0
    best_eps=0
    best_w=[]
    best_b=[]


    hidden_nodes=40
    hidden_layer=4
    for epoch in set_epochs: 
        for hidden_layer in set_hidden_layers:
            for hidden_nodes in set_hidden_nodes:
                for batch in set_batch:
                    for alpha in set_alpha:
                        for eps in set_learning:
                        
                            w,b=generate(hidden_layer,hidden_nodes,X_tr.shape[1],ytr.shape[1])
                            
                            for i in range(0,epoch):
                                for j in range(0,batch):
                                    #print(i)
                                    #print(j)
                                    example_cut=int(n/batch)
                                    #print("\n examples from :",range(i*example_cut,((i+1)*example_cut)))
                                    X_mini_batch=X_tr[range(j*example_cut,((j+1)*example_cut)),:]
                                    y_mini_batch=ytr[range(j*example_cut,((j+1)*example_cut))]
                                    #Forward_propagation
                                    yhat,h,z=fun_yhat(X_mini_batch,w,b,hidden_layer)
                                    grad_w,grad_b=backprop(y_mini_batch,X_mini_batch,yhat,hidden_layer,w,b,h,z)
                                    print("accuracy",accuracy(X_mini_batch,y_mini_batch,w,b,hidden_layer))
                                    print("cost",fce(X_mini_batch,y_mini_batch,w[-1],alpha,yhat))
                                    w_new,b_new=SGD(grad_w,grad_b,eps,w,b)
                                    w=w_new
                                    b=b_new
                            
                            yhat,h,z=fun_yhat(X_val,w,b,hidden_layer)
                            fce_validation=fce(X_val,yval,w[-1],alpha,yhat)
                            val_accuracy=accuracy(X_val,yval,w,b,hidden_layer)   
                            print("Validation")
                            print("Epoch",epoch,"Alpha",alpha,"Batch size",batch,"Learning Rate",eps,"Hidden layes",hidden_layer,"Number of hidden units",hidden_nodes,"Accuracy",val_accuracy,"FCE",fce_validation)
                            
                            if(fce_validation<best_fce):
                                best_fce=fce_validation  
                                best_epoch=epoch
                                best_batch=batch
                                best_alpha=alpha
                                best_eps=eps
                                best_w=w 
                                best_b=b
                                best_hidden_node=hidden_nodes
                                best_hidden_layers=hidden_layer
                                best_val_accuracy=val_accuracy

      
    print("-----------------Test------------------")
    print("Epoch",best_epoch,"Alpha",best_alpha,"Batch size",best_batch,"Learning Rate",best_eps,"Hidden layes",best_hidden_layers,"Number of hidden units",best_hidden_node,"Accuracy",best_val_accuracy,"FCE",best_fce)
    accuracy(X_te,yte,best_w,best_b,hidden_layer)   
    

if __name__== "__main__":
     findBestHyperparameters()


#ACC test 97.42
#Epoch 30
#Alpha 1e-05
#Batch 128
#Learning rate 0.0008
#Hidden Layer 3
#Number of hidden units: 50
