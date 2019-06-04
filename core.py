"""
Core Function Code
"""
import numpy as np
import pandas as pd

class ResponseSurface:
    def __init__(self,inputs,output,intercept = True, interaction = True):
        nvar = inputs.shape[-1]
        if nvar == 2:
            self.degrees = [(1,0),(0,1),(2,0),(0,2)]
            if intercept:
                self.degrees.append((0,0))
            if interaction:
                self.degrees.append((1,1))
        elif nvar == 3:
            self.degrees = [(1,0,0),(0,0,1),(0,1,0)]
            aux = [(1,0,0),(0,0,1),(0,1,0)]
            npower = 3
            for i in np.arange(2,npower+1):
                aux2 = np.dot(aux,i)
                for j in range(len(aux)):                    
                    self.degrees.append(tuple(aux2[j]))
            if intercept:
                self.degrees.append((0,0,0))
            if interaction:
                aux3 = [(1,1,1),(1,1,0),(0,1,1),(1,0,1)]
                for k in range(len(aux3)):
                    self.degrees.append(tuple(aux3[k]))
                for ii in np.arange(2,npower):
                    aux4 = np.dot(aux3,ii)
                    for jj in range(len(aux4)):                    
                        self.degrees.append(tuple(aux4[jj]))
        elif nvar == 4:
            self.degrees = [(1,0,0,0),(0,0,0,1),(0,1,0,0),(0,0,1,0)]
            aux = [(1,0,0,0),(0,0,0,1),(0,1,0,0),(0,0,1,0)]
            npower = 4
            for i in np.arange(2,npower+1):
                aux2 = np.dot(aux,i)
                for j in range(len(aux)):                    
                    self.degrees.append(tuple(aux2[j]))
            if intercept:
                self.degrees.append((0,0,0,0))
            if interaction:
                aux3 = [(1,1,1,1),(1,1,1,0),(1,1,0,1),(1,0,1,1),(0,1,1,1),(1,1,0,0),
                        (1,0,1,0),(1,0,0,1),(0,1,0,1),(0,0,1,1)]
                for k in range(len(aux3)):
                    self.degrees.append(tuple(aux3[k]))
                for ii in np.arange(2,npower):
                    aux4 = np.dot(aux3,ii)
                    for jj in range(len(aux4)):                    
                        self.degrees.append(tuple(aux4[jj]))
        elif nvar == 5:
            self.degrees = [(1,0,0,0,0),(0,0,0,0,1),(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0)]
            aux = [(1,0,0,0,0),(0,0,0,0,1),(0,1,0,0,0),(0,0,1,0,0),(0,0,0,1,0)]
            npower = 4
            for i in np.arange(2,npower+1):
                aux2 = np.dot(aux,i)
                for j in range(len(aux)):                    
                    self.degrees.append(tuple(aux2[j]))
            if intercept:
                self.degrees.append((0,0,0,0,0))
            if interaction:
                aux3 = [(1,1,1,1,1),(1,1,1,1,0),(1,1,1,0,1),(1,1,0,1,1),(1,0,1,1,1),(0,1,1,1,1),
                        (1,1,1,0,0),(1,1,0,1,0),(1,1,0,0,1),(1,0,1,0,1),(1,0,0,1,1),(0,1,0,1,1),
                        (0,0,1,1,1),(1,1,0,0,0),(1,0,1,0,0),(1,0,0,1,0),(1,0,0,0,1),(0,1,0,0,1),
                        (0,0,1,0,1),(0,0,0,1,1)]
                for k in range(len(aux3)):
                    self.degrees.append(tuple(aux3[k]))
                for ii in np.arange(2,npower):
                    aux4 = np.dot(aux3,ii)
                    for jj in range(len(aux4)):                    
                        self.degrees.append(tuple(aux4[jj]))
        else:
            raise ValueError('`ResponseSurface` can only manage a maximum of 5 distinct variables.')
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
    gridVar1 = np.linspace(varLims[selectvar,0],varLims[selectvar,1],100)
    gridVar2 = np.linspace(varLims[selectvar,0],varLims[selectvar,1],100)
    inputVar1, inputVar2 = np.meshgrid(gridVar1, gridVar2)
    inputGrid = np.stack([np.ravel(inputVar1),np.ravel(inputVar2)], axis = -1)
    outputGrid = np.array(ResSur.predict(inputGrid))
    outputGrid = outputGrid.reshape(inputVar1.shape)
    return {'input1':inputVar1,'input2':inputVar2,'output':outputGrid}

def NewSamples(inputs,outputs,varLims,n):
    nvar = inputs.shape[-1]
    
    trials = 1000
    rdn = np.random.random(trials)
    test_var = np.zeros((trials,nvar))
    new = inputs.values
    
    for i in range(nvar):
        test_var[:,i] = rdn*(varLims[i,1]-varLims[i,0])+varLims[i,0]
    
    test_var = np.round(test_var,1)
    
    d = np.zeros(test_var.shape[0])  
    for k in range(test_var.shape[0]):
        aux = abs(new-test_var[k,:])
        d[k] = np.sqrt(np.sum(aux**2))
    
    aux = np.argsort(d)
    d = d[aux]
    du,b = np.unique(d,return_index = True)
    if b.shape[0]-n > 0:
        new = np.row_stack((new,test_var[b[b.shape[0]-n:b.shape[0]],:]))  
    else:
        raise Warning('It was not possible to add '+str(n)+' new data points!')
        new = np.row_stack((new,test_var[b,:]))
        
    new_data = pd.DataFrame(new, columns = ["x{}".format(i+1) for i in range(nvar)])
    new_data['y'] = ''
    new_data['y'].values[0:inputs.shape[0]] = outputs
    return new_data   
    