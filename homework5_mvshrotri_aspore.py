############### Assignment 5 ##################################
#Akshata Pore
#Manasi Shrotri
#__________________________________________________________________


import numpy as np
#import matplotlib.pyplot as plt
#import scipy.optimize  # For check_grad, approx_fprime

class RNN:
    def __init__ (self, numHidden, numInput, numOutput):
        self.numHidden = numHidden
        self.numInput = numInput
        self.U = np.random.randn(numHidden, numHidden) * 1e-1
        self.V = np.random.randn(numHidden, numInput) * 1e-1
        self.w = np.random.randn(numHidden) * 1e-1
        self.w = self.w.reshape(numHidden,numInput)   
        # TODO: IMPLEMENT ME
    
    def cal_error(self,yt,ys):
        Jsum=0        
        for i in range(len(ys)):
            Jsum+=0.5*(yt[i]-ys[i])**2
        return(Jsum)
    
    def backward (self,yt,ys,ht,numTimesteps):
        # TODO: IMPLEMENT ME
        dj_dU=0
        dj_dV=0
        dj_dw=0
        
        for t in range(0,50):
            gt=(1-(ht[t]**2))
            F=np.multiply((gt),ht[t-1])
            temp_du=F+np.dot((gt).T,self.U).T
            temp_du2=(yt[t]-ys[t])*np.dot(self.w,temp_du.T)
            dj_dU=dj_dU+temp_du2
            
            #dj_dV
            E=np.multiply((gt),xs[t])#6*1
            temp_dv=E+np.dot((gt).T,self.U).T
            temp_dv2=(yt[t]-ys[t])*np.multiply(self.w,temp_dv)
            dj_dV=dj_dV+temp_dv2
            
            #dj_dw
            dj_dw=dj_dw+np.multiply(yt[t]-ys[t],ht[t])
            
        return(dj_dU,dj_dV,dj_dw)
   
    
    def forward (self, xs):
        # TODO: IMPLEMENT ME
            ht=[]
            yt=[]
            for i in range(numTimesteps):
                if i==0:
                    ht.append(np.tanh(np.dot(self.U,np.zeros((numHidden,numInput)))+(np.dot(self.V,xs[i]))))
                else:
                    ht.append(np.tanh(np.dot(self.U,ht[i-1])+(np.dot(self.V,xs[i]))))    
                yt.append(np.dot(ht[i].T,self.w))
            return(ht,yt)    

    def SGD(self,ys,xs,numTimesteps):
        for iter in range(0,100):
            print(iter)
            ht,yt=self.forward(xs)
            loss=self.cal_error(yt,ys)
            print(loss)
            dj_dU,dj_dV,dj_dw=self.backward(yt,ys,ht,numTimesteps)
            loss=0
            alpha1=0.00001
            alpha2=0.0001
            alpha3=0.0008
            self.U=self.U-np.multiply(alpha1,dj_dU)
            self.V=self.V-np.multiply(alpha2,dj_dV)
            self.w=self.w-np.multiply(alpha3,dj_dw) 

# From https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
def generateData ():
    total_series_length = 50
    echo_step = 2  # 2-back task
    batch_size = 1
    x = np.random.choice(2, total_series_length, p=[0.5, 0.5])
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    y = list(y)
    return (x, y)

if __name__ == "__main__":
    xs, ys = generateData()
    #print xs
    #print ys
    numHidden = 6
    numInput = 1
    numTimesteps = len(xs)
    rnn = RNN(numHidden, numInput, 1)
    # TODO: IMPLEMENT ME
    rnn.SGD(ys,xs,numTimesteps)