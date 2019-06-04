"""
Core Function Code
"""
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class ResponseSurface:
    def __init__(self,inputs,output,degree = 2, intercept = True, interaction_only = False):
        X = inputs
        y = output
        
        polynomial_features = PolynomialFeatures(degree=degree,
                                                 interaction_only=interaction_only,
                                                 include_bias=intercept)
        x_poly = polynomial_features.fit_transform(X)
    
        model = LinearRegression()
        model.fit(x_poly, y)
        
        self._polynomial_features = polynomial_features
        self._x_poly = x_poly
        self._model  = model
        
    def predict(self, input_pred):
        input_poly = self._polynomial_features.fit_transform(input_pred)
        pred = self._model.predict(input_poly)
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
    