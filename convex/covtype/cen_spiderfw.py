# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 22:10:53 2021

@author: JX
"""

from __future__ import division 
#fruture前后是两个下划线：__future__
#上述是为了希望只执行简单除法，例如1/2=0.5这样的,单斜线不再用作整除，双斜线实现整除操作
import numpy as np
from scipy.io import savemat
import scipy.io as scio
import math
import matplotlib.pyplot as plt
import time
 ##邻接矩阵
dataFile = 'F:/anaconda_spyder/data/C_meth1_smote_800.mat'
data = scio.loadmat(dataFile)
C=data['a']
#print(C[2][3])
##数据
a9asam=scio.loadmat('F:/anaconda_spyder/data/covtype/cov.mat') #载入A：A(48243x123):48243个数据
a9al=scio.loadmat('F:/anaconda_spyder/data/covtype/L_cov.mat') #载入L：A(1x48243):48243个结果
A=a9asam['A'];
L=a9al['L'];
L=L[0]
'''
a9asam=scio.loadmat('F:/anaconda_spyder/data/covtype/cov.mat') #载入A：A(48243x123):48243个数据
a9al=scio.loadmat('F:/anaconda_spyder/data/covtype/L_cov.mat') #载入L：A(1x48243):48243个结果
A=a9asam['A'];
L=a9al['L'];
L=L[0]
r={1:-1,2:1}
L=[r[i] if i in r else i for i in L]
'''
#print(np.size(A,0))
#智能体数目和总样本数
agent_n=10
sample_n=np.size(A,0)

x_dim=np.size(A,1)#变量维度
start = time.time()
#变量初始化
listx_i=[]
#X_i=np.zeros(x_dim)
X_i=np.ones(x_dim)
X_i=X_i/(np.linalg.norm(X_i))
barX_i=X_i
listx_i.append(X_i) 
#print("process {} initial x {}".format(rank,listx_i[0]))

#梯度函数
def gradient_com(vari_x):
    gradient_vari_x=[0.0]*x_dim
    for j in range(sample_n):
        mid=L[j]*A[j,:]
        #print(mid@vari_x)
        gradient_x=-mid*math.exp(-mid@vari_x)/(1+math.exp(-mid@vari_x));
        gradient_vari_x=gradient_vari_x+gradient_x
    return gradient_vari_x/sample_n

def gradient_com_set(vari_x,S):
    gradient_vari_x=[0.0]*x_dim
    for j in S:
        mid=L[j]*A[j,:]
        gradient_x=-mid*math.exp(-mid@vari_x)/(1+math.exp(-mid@vari_x));
        gradient_vari_x=gradient_vari_x+gradient_x
    return gradient_vari_x

#print("process {} initial grad {}".format(rank,gradientx_i_seq[0]))
v0=gradient_com(X_i)
#开始迭代
T=300
K=math.ceil(sample_n**0.5)###参数
c_k=2
R=20
q=K
#eta=1/(T**0.5)## 1/(c_k*T**0.5)
for t in range(T):
    k=0
    eta=2/(2**k+t)
    if t%q==0: #########
        if k!=0:
            k=k+1
        v1=gradient_com(X_i)
    else:
        S=np.random.choice(sample_n, K)
        v1=(gradient_com_set(listx_i[t],S)-gradient_com_set(listx_i[t-1],S))/K+v0       
    #resul=-c_k/(2*np.linalg.norm(v1))*v1
    sumd_abs=list(map(abs,v1))
    maxk=sumd_abs.index(max(sumd_abs))
    resul=np.zeros(x_dim)  #.reshape(-1,1)
    resul[maxk]=-R*np.sign(v1[maxk])
    X_i=(1-eta)*listx_i[t]+eta*resul
    v0=v1
    listx_i.append(X_i)


#迭代结果序列（函数值和一致性结果）10 T+1 123
'''
    print('rank {} is X_col is {}'.format(rank, X_col[1][1]))
    print('X_col dim 0 is {}'.format(np.size(X_col,0)))
    print('X_col dim 1 is {}'.format(np.size(X_col,1)))
    print('X_col dim 2 is {}'.format(np.size(X_col,2)))
'''

end = time.time()
runntime=end-start
print(runntime)

X_col=np.array(listx_i)
file_name='CENSPIDER.mat'
savemat(file_name,{'runtime_censpider':runntime,'x_col_censpider':X_col})

'''    
    consenseq=[]
    fvseq=[]
    for t in range(T):
        fv=0.0
        for i in range(sample_n):
            fv=fv+np.log(1+np.exp(-L[i]*A[i,:]@X_col[1][t]))
        #print('rank {} coll fv is {}'.format(rank,fv))
        fvseq.append(fv/sample_n)
        consen=0;
        for i in range(agent_n):
            xcon=[0]*x_dim
            for j in range(agent_n):
                xcon=xcon+C[i][j]*(X_col[i][t]-X_col[j][t])
            xdot=X_col[i][t]@xcon
            consen=consen+xdot
        #print('rank {} coll consensus is {}'.format(rank,consen))
        consenseq.append(consen) 
    #制图
    fig=plt.figure()
    ax=fig.add_subplot(1,2,1)
    ax.plot(range(T),consenseq)
    ax=fig.add_subplot(1,2,2)
    ax.plot(range(T),fvseq)
    #ax.plot(range(0,T,10),fvseq[::10],color='r', marker='o', markerfacecolor='blue', markersize=5)
    #for a,b in zip(range(0,T,10),fvseq[::10]):
    #    ax.text(a,b,b,ha='center',va='bottom',fontsize=8)
    plt.show()
'''
