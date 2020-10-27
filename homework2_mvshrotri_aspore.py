# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 18:46:28 2020

@author: Manasi Shrotri, Akshata Pore
"""
import numpy as np
X_tr =np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))    
ytr = np.reshape(np.load("age_regression_ytr.npy"),(-1,1))
X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
yte = np.reshape(np.load("age_regression_yte.npy"),(-1,1))


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_tr,ytr, test_size=0.2, random_state=42)

n=X_train.shape[0]


def fmse(X,w,alpha,Y,b):
    Y_hat=np.dot(X,w)+b
    Num=(np.sum(np.square(Y_hat-Y))/(2*Y_hat.shape[0]))
    Den=np.dot(w.T,w)
    fmse=Num+(alpha*Den)/2
    return fmse


#range of epochs

set_epochs=[10,15,30,50]
#range for batch size
set_batch=[50,100,150,200]
#
set_alpha=[0.0001,0.0002,0.00001,0.00002]
#
#
set_learning=[0.0001,0.0005,0.00001,0.000001]

best_fmse=10**10
best_epoch=0
best_batch=0
best_alpha=0
best_eps=0
best_w=0
best_b=0


for epoch in set_epochs: 
    for batch in set_batch:
        for alpha in set_alpha:
            for eps in set_learning:
                w=np.mat(np.random.randn(X_train.shape[1])).T
                b=np.random.randn(1,1)
                
                for i in range(0,epoch):
                    for j in range(0,batch):
                        print("\n Epoch",i)
                        print("\n Batch",j)
                        print("\n Alpha",alpha)
                        print("\n eps",eps)
                        example_cut=int(n/batch)
                        print("\n examples from :",range(i*example_cut,((i+1)*example_cut)))
                        X_mini_batch=X_train[range(i*example_cut,((i+1)*example_cut)),:]
                        y_mini_batch=y_train[range(i*example_cut,((i+1)*example_cut))]
                        #w = gradient_descent(X_mini_batch,y_mini_batch,w,b,eps)
                       
                        yhat = np.dot(X_mini_batch,w)+b
                        gradient = np.matmul(X_mini_batch.T,(yhat-y_mini_batch))/X_mini_batch.shape[0]
                        w_new=w-(eps*gradient)
                        w=w_new
                        
                        grad_b=np.mean(yhat-y_mini_batch)
                        b_new=b-(eps*grad_b)
                        b=b_new
                        # Report fMSE cost on the training and testing data (separately)
                        # fmse for training
                        
                        fmse_training=fmse(X_mini_batch,w,alpha,y_mini_batch,b)
                        print("Fmse for train data is :",fmse_training)
                
                fmse_val=fmse(X_val,w,alpha,y_val,b)
                print("epoch",epoch)
                print("batch",batch)
                print(fmse_val)
                print("Fmse for Val data is :",fmse_val)
 
                if(fmse_val<best_fmse):
                    best_fmse=fmse_val  
                    best_epoch=i
                    best_batch=j
                    best_alpha=alpha
                    best_eps=eps
                    best_w=w
                    best_b=b
                
fmse_test=fmse(X_te,best_w,best_alpha,yte,b) 
print("Fmse for test data is :",fmse_test)
  
                   
                    
                