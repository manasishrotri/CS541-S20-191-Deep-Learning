# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 17:52:46 2020
HW 3
@author: Manasi Shrotri & Akshata Pore 
"""
import numpy as np
X_tr =(np.load("mnist_train_images.npy"))   
ytr = (np.load("mnist_train_labels.npy"))
X_val =(np.load("mnist_validation_images.npy"))    
yval = (np.load("mnist_validation_labels.npy"))
X_te =(np.load("mnist_test_images.npy"))    
yte = (np.load("mnist_test_labels.npy"))

n=X_tr.shape[0]

#range for epoch 
set_epochs=[100,200,300,400]
#range for batch size
set_batch=[50,100,200,300]#500,1000,2000
#
set_alpha=[0.00001,0.0001,0.00005,0.00002]
#   
set_learning=[0,2,0.3,0,4,0.5]

best_fce=10**10
best_epoch=0
best_batch=0
best_alpha=0
best_eps=0
best_w=0
best_b=0

def fun_yhat(X_mini_batch,w,b):
    z = np.dot(X_mini_batch,w)+b
    yhat=np.zeros(z.shape) 
    for rows in range(z.shape[0]):
        yhat[rows]=np.exp(z[rows])/np.sum(np.exp(z[rows]))
    return(yhat)
    
def fce(X_mini_batch,y_mini_batch,w,b,alpha,yhat):
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
    
    
for epoch in set_epochs: 
    for batch in set_batch:
        for alpha in set_alpha:
            for eps in set_learning:
                w=np.mat(np.random.randn(X_tr.shape[1],10))
                b=np.random.randn(1,10)
                
                for i in range(0,epoch):
                    for j in range(0,batch):
                        #print("\n Epoch",i)
                        #print("\n Batch",j)
                        #print("\n Alpha",alpha)
                        #print("\n eps",eps)
                        example_cut=int(n/batch)
                        #print("\n examples from :",range(i*example_cut,((i+1)*example_cut)))
                        X_mini_batch=X_tr[range(j*example_cut,((j+1)*example_cut)),:]
                        y_mini_batch=ytr[range(j*example_cut,((j+1)*example_cut))]
                        
                        yhat=fun_yhat(X_mini_batch,w,b)
                        
                        fce_val=fce(X_mini_batch,y_mini_batch,w,b,alpha,yhat)
                        
                        gradient = np.matmul(X_mini_batch.T,(yhat-y_mini_batch))/X_mini_batch.shape[0]
                        gradient=gradient+(alpha*w)
                        w_new=w-(eps*gradient)
                        w=w_new
                        b_new = b - (eps*((yhat-y_mini_batch)/X_mini_batch.shape[0]))
                        b=b_new
                        #print("Fmse for train data is :",fce_val)
                                             
                yhat=fun_yhat(X_val,w,b)
                fce_validation=fce(X_val,yval,w,b,alpha,yhat)
                print("epoch",epoch)
                print("batch",batch)
                print(fce_validation)
                print("Fmse for Val data is :",fce_validation)
 
                if(fce_validation<best_fce):
                    best_fce=fce_validation  
                    best_epoch=epoch
                    best_batch=batch
                    best_alpha=alpha
                    best_eps=eps
                    best_w=w 
                    best_b=b
        
Y_test_cal=fun_yhat(X_te,best_w,best_b)
Y_test_cal=np.argmax(Y_test_cal, axis=1)
y_true = np.argmax(yte, axis=1)
from sklearn.metrics import confusion_matrix
                        
cm = confusion_matrix(Y_test_cal, y_true)
accuracy1 = sum(Y_test_cal == y_true)/(float(len(y_true)))
print(accuracy1*100)


#epoch 200
#batch 100
#Fmse for Val data is : [[0.3123731]]
#Accuracy : 92.84