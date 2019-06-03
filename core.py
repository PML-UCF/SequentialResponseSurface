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

def DiagSampling(Grid,inputs,output,var1Lims,var2Lims,npoints):
    aux = np.amax(Grid['output'],axis = 0)
    index1 = np.where(Grid['output'] == np.max(aux))
    index2 = np.where(Grid['output'] == np.min(aux))
    
    slope = (Grid['input2'][index1]-Grid['input2'][index2])/(Grid['input1'][index1]-
             Grid['input1'][index2])
    
    var1 = np.linspace(var1Lims[0],var1Lims[-1],npoints+1)
    var1 = var1[1:npoints+1]
    
    var2 = Grid['input2'][index2] + slope*(var1-Grid['input1'][index2])
    
    var1 = np.round(var1,1)
    var2 = np.round(var2,1)
    
    degrees = [(0,0),(1,0),(0,1),(1,1),(2,0),(0,2)]
    coef_matrix = np.stack([np.prod(inputs**d, axis=1) for d in degrees], axis=-1)   # stack monomials like columns
    coef = np.linalg.lstsq(coef_matrix, output)[0]
    
    pred = np.zeros(npoints)
    
    for i in range(npoints):
        pred[i] = coef[0]+coef[1]*var1[i]+coef[2]*var2[i]+coef[3]*var1[i]*var2[i]+coef[4]*var1[i]**2+coef[5]*var2[i]**2

    
    return {'x1':var1,'x2':var2,'y':pred}

def NewDOE(data,new):
    nvar = data.shape[-1]-2
    oldinput = data[["x{}".format(i+1) for i in range(nvar)]]
    oldoutput = data['y']
    new_length = new.shape[0]
    newinput = np.array([new['x1'],new['x2']])
    newinput = np.reshape(newinput, (new_length,2))
    
    inputs = np.row_stack((oldinput.values,newinput))
    outputs = np.concatenate((oldoutput.values,np.zeros(new_length)))
    
    new_data = np.column_stack((inputs,outputs))
    new_doe = pd.DataFrame(new_data, columns = ['x1','x2','y'])
    return new_doe    
    