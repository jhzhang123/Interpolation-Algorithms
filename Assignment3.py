# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pandas
import math
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

fig = plt.figure()
ax = fig.gca(projection='3d')
plt.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')

# Prepare arrays x, y
u = np.linspace(0,1,num=50)
x = np.exp(np.cos(6.2*u+51/30)+0.1)*np.cos(12.4*u)
y = np.exp(np.cos(6.2*u+51/30)+0.1)*np.sin(12.4*u)

ax.plot(x, y, label='parametric curve')
ax.legend()

plt.show()

 #%%
 
w = np.linspace(0,1,5)

data_pointx = []
data_pointy = []
deg = 3

data_pointx.append(np.exp(np.cos(6.2*w+51/30)+0.1)*np.cos(12.4*w))
data_pointy.append(np.exp(np.cos(6.2*w+51/30)+0.1)*np.sin(12.4*w))

#print(data_pointx)

d = pandas.DataFrame([[1,2],[3,4],[5,6]],columns=['data_pointx','data_pointy'])

x1 = np.array(d['data_pointx'])
y1 = np.array(d['data_pointy'])
r= np.polyfit(x1, y1, deg)

print('Polynom coeffs',r[::-1])
#building function from coefs
cc = np.poly1d(r)
plt.plot(x,y,color='red')
plt.plot(x,cc(x),color='blue')
print ('RMSE 1st order:',np.sqrt(np.mean((y-cc(x))**2)))

print('Polynom coeffs',r[::-1])
cc = np.poly1d(r)
plt.plot(x,cc(x),color='green')
plt.show()
print ('RMSE 2nd order:',np.sqrt(np.mean((y-cc(x))**2)))

#%% trigo interpolation

def P8(n,t):
    matA1=[0]*n
    matA1[0]=1.0/math.sqrt(8)
    
    for k in range(1,4):
        matA1[k]=math.cos(k*t*2*math.pi)*2/math.sqrt(8)

    matA1[4]=math.cos(4*t*2*math.pi)/math.sqrt(8)

    for j in range(5,8):
            matA1[j]=math.sin((j-4)*t*2*math.pi)*(-2)/math.sqrt(8)
    matA1=np.around(matA1,2)    
    print(t,"\t:", matA1)
    return matA1

def P8Evaluation(n,udata,MatAA):
    matA1=[0]*n
    for i in range(n):
        t=udata[i]
        matA1[i]+=MatAA[0]*1.0/math.sqrt(8)
    
        for j in range(1,4):
             matA1[i]+=  MatAA[j]*math.cos(j*t*2*math.pi)*2/math.sqrt(8)

        matA1[i]+=MatAA[4]*math.cos(4*t*2*math.pi)/math.sqrt(8)

        for j in range(5,8):
             matA1[i]+=MatAA[j]*math.sin((j-4)*t*2*math.pi)*(-2)/math.sqrt(8)
    return matA1

n=8
tdata=[]
c=0
d=1

udata=[1.0*u/n for u in range(n)]

xdata=[(math.exp(math.cos(6.2*u+(51/30)))+0.1)*math.cos(12.4*u)for u in udata]
ydata=[(math.exp(math.cos(6.2*u+(51/30)))+0.1)*math.sin(12.4*u)for u in udata]


MatA=[]
for t in range(0,n):
    MatA.append(P8(n,udata[t]))

print("udata=", udata)
MatA=np.array(MatA)
MatBx=np.array(xdata)
MatBy=np.array(ydata)
matAAx=np.linalg.solve(MatA,MatBx)
matAAy=np.linalg.solve(MatA,MatBy)

num=100
udataf=[1.0*u/num for u in range(num)]
xdataf=[(math.exp(math.cos(6.2*u+(51/30)))+0.1)*math.cos(12.4*u)for u in udataf]
ydataf=[(math.exp(math.cos(6.2*u+(51/30)))+0.1)*math.sin(12.4*u)for u in udataf]
testdatax=P8Evaluation(num,udataf,matAAx)
testdatay=P8Evaluation(num,udataf,matAAy)

print("MatA=", MatA)
print("MatBx=", MatBx)
print("Mataax=", matAAx)
print("testdatax=", testdatax)

print("MatBy=", MatBy)
print("Mataay=", matAAy)
print("testdatay=", testdatay)

fig = plt.figure()
aux = fig.add_subplot(311)
aux.plot(udataf,xdataf,color='r',linestyle='-',marker='.')
aux.plot(udataf,testdatax,color='g',linestyle='-',marker='x')

auy = fig.add_subplot(312)
auy.plot(udataf,ydataf,color='r',linestyle='-',marker='.')
auy.plot(udataf,testdatay,color='g',linestyle='-',marker='x')

axy = fig.add_subplot(313)
axy.plot(xdataf,ydataf,color='r',linestyle='-',marker='.')
axy.plot(testdatax,testdatay,color='g',linestyle='-',marker='x')

#aux.legend()
plt.show()

#%%

import scipy.integrate
from numpy import exp
f= lambda x: -0.0081886*x**3+ 0.0736974*x**2+0.811662*x+1.12283

result = scipy.integrate.quad(f, 0, 1)
print(result)
