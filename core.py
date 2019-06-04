"""
Core Function Code
"""
import numpy as np
import pandas as pd

class ResponseSurface:
    def __init__(self,inputs,output,intercept = True, interaction = True):
        self.degrees = [(1,0),(0,1),(2,0),(0,2)]
        if intercept:
            self.degrees.append((0,0))
        if interaction:
            self.degrees.append((1,1))
        self.coef_matrix = np.stack([np.prod(inputs**d, axis=1) for d in self.degrees], axis=-1)
        self.coef = np.linalg.lstsq(self.coef_matrix, output)[0]
        
    def predict(self, input_pred):
        grid_pred = np.stack([np.prod(input_pred**d, axis=1) for d in self.degrees], axis=-1)
        pred = np.dot(grid_pred, self.coef)
        return pred

def GenerateGrid(ResSur, varLims,selectvar):
    n = np.shape(selectvar)[0]
    if n is not 2:
            raise ValueError('`GenerateGrid` only takes 2 distinct variables.')
    gridVar1 = np.linspace(varLims[selectvar[0],0],varLims[selectvar[0],1],100)
    gridVar2 = np.linspace(varLims[selectvar[1],0],varLims[selectvar[1],1],100)
    inputVar1, inputVar2 = np.meshgrid(gridVar1, gridVar2)
    inputGrid = np.stack([np.ravel(inputVar1),np.ravel(inputVar2)], axis = -1)
    outputGrid = np.array(ResSur.predict(inputGrid))
    outputGrid = outputGrid.reshape(inputVar1.shape)
    return {'input1':inputVar1,'input2':inputVar2,'output':outputGrid}

def NewSamples(inputs,outputs,varLims,n):
    nvar = inputs.shape[-1]
    
    trials = 1000
    rdn = np.random.random((trials,nvar))
    test_var = np.zeros((trials,nvar))
    new = inputs.values
    
    for i in range(nvar):
        test_var[:,i] = rdn[:,i]*(varLims[i,1]-varLims[i,0])+varLims[i,0]
    
    test_var = np.round(test_var,1)
    
    for j in range(n):
        d = np.zeros((new.shape[0],test_var.shape[0]))
        for k in range(test_var.shape[0]):
            d[:,k] = np.sqrt(np.sum(abs(new-test_var[k,:])**2, axis = 1))    
        dist = np.min(d, axis = 0)
        adist = np.argmax(dist) 
        new = np.row_stack((new,test_var[adist,:]))
        idx = np.where(dist == dist[adist])
        test_var = np.delete(test_var, (idx), axis = 0)
        
    new_data = pd.DataFrame(new, columns = ["x{}".format(i+1) for i in range(nvar)])
    new_data['y'] = ''
    new_data['y'].values[0:inputs.shape[0]] = outputs
    return new_data   
    