# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:29:08 2021

@author: JX
FW-FAC algorithm
"""

from __future__ import division 
from mpi4py import MPI
from scipy.io import savemat
import numpy as np
import scipy.io as scio
import math
import matplotlib.pyplot as plte
import time
 ##邻接矩阵
dataFile = 'F:/anaconda_spyder/data/C_meth1_smote_800.mat'
data = scio.loadmat(dataFile)
C=data['a']
#print(C[2][2])
##数据
a9asam=scio.loadmat('F:/anaconda_spyder/data/a9a/a9a_smote.mat') #载入A：A(48243x123):48243个数据
a9al=scio.loadmat('F:/anaconda_spyder/data/a9a/L_a9a_smote.mat') #载入L：A(1x48243):48243个结果
A=a9asam['A1'];
L=a9al['L1'];
L=L[0]
#print(np.size(A,0))
#智能体数目和总样本数
agent_n=10
sample_n=np.size(A,0)
#根据智能体数目裁剪数据
# 这里divmod可以同时得到商和余数，如divmod(10,3)得到3和1
ave, res = divmod(sample_n, agent_n)
#counts得到的是每个进程计算的数量个数
counts = [ave + 1 if p < res else ave for p in range(agent_n)]
# determine the starting and ending indices of each sub-agent
starts = [sum(counts[:p]) for p in range(agent_n)]
ends = [sum(counts[:p+1]) for p in range(agent_n)]
#print(ave) #本地样本个数
start = time.time()
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
if rank==0:   
    send_data = [A[starts[p]:ends[p],:] for p in range(size)]
    send_data_l=[L[starts[p]:ends[p]] for p in range(size)]
else:
    send_data=None
    send_data_l=None
    
recv_data=comm.scatter(send_data,root=0)
recv_data_l=comm.scatter(send_data_l,root=0)

x_dim=np.size(A,1)#变量维度

#变量初始化
listx_i=[]
#X_i=np.random.randn(x_dim)
#X_i=X_i/(np.linalg.norm(X_i))
X_i=np.zeros(x_dim)
#X_i=np.ones(x_dim)
#X_i=X_i/(np.linalg.norm(X_i)) 
#for i in range(5):
#    X_i[10*i]=0;
listx_i.append(X_i)
np.save('F:/anaconda_spyder/non-convex/data/classify/xi0_a9a.npy',X_i) # 保存为.npy格式


#print("process {} initial x {}".format(rank,listx_i[0]))

#梯度函数
def gradient_com(vari_x):
    gradient_vari_x=[0.0]*x_dim
    for j in range(ave):
        mid=recv_data_l[j]*recv_data[j,:]
        #print(mid@vari_x)
        gradient_x=-mid*math.exp(mid@vari_x)/(1+math.exp(mid@vari_x))**2;
        gradient_vari_x=gradient_vari_x+gradient_x
    return gradient_vari_x/ave
#梯度初始化
gradientx_i_seq=[]
gradientx_i=gradient_com(X_i)
gradientx_i_seq.append(gradientx_i)
#print("process {} initial grad {}".format(rank,gradientx_i_seq[0]))
sumx_seq=[]
sumx_seq.append(X_i)
#开始迭代
T=300
c_k=2
R=20
BarnablaFi=[0]*x_dim
gradientcom0=[0]*x_dim
for t in range(T):
    etak=1/((t+1)**0.5)
    for j in range(agent_n):                
        if C[rank][j] != 0:
            comm.send(listx_i[t], dest=j, tag=rank)
#            comm.send(gradientx_i_seq[t],dest=j,tag=rank+10)
    neigh_x=[0]*agent_n
    for j in range(agent_n):
        if C[rank][j] != 0:
            neigh_x[j]=comm.recv(source=j, tag=j)
#            neigh_gra[j]=comm.recv(source=j, tag=j+10)
            #print("process {} recv samples {} and {} from agent {}...".format(rank,neigh_x,neigh_gra,j))
    sumx=[0]*x_dim
    for j in range(agent_n):
        sumx=sumx+C[rank][j]*neigh_x[j]
    gradientcom1=gradient_com(sumx)
    NablaFi=BarnablaFi+gradientcom1-gradientcom0
    gradientcom0=gradientcom1
    for j in range(agent_n):                
        if C[rank][j] != 0:
            comm.send(NablaFi,dest=j,tag=rank+10)
    neigh_gra=[0]*agent_n
    for j in range(agent_n):
        if C[rank][j] != 0:
            neigh_gra[j]=comm.recv(source=j, tag=j+10)
    sumd=[0]*x_dim
    for j in range(agent_n):
        sumd=sumd+C[rank][j]*neigh_gra[j]
    BarnablaFi=sumd  
    #resul=-c_k/(2*np.linalg.norm(sumd))*sumd  
    
    sumd_abs=list(map(abs,sumd))
    maxk=sumd_abs.index(max(sumd_abs))
    resul=np.zeros(x_dim) #.reshape(-1,1)
    resul[maxk]=-R*np.sign(sumd[maxk])
    
    X_i=(1-etak)*sumx+etak*resul
    
    '''
    nablaf0=gradient_com(listx_i[t])
    nablaf1=gradient_com(X_i)
    gradientx_i=sumd+nablaf1-nablaf0
    gradientx_i_seq.append(gradientx_i)
    '''
    listx_i.append(X_i)
    sumx_seq.append(sumx)


#迭代结果序列（函数值和一致性结果）10 T+1 123
'''
    print('rank {} is X_col is {}'.format(rank, X_col[1][1]))
    print('X_col dim 0 is {}'.format(np.size(X_col,0)))
    print('X_col dim 1 is {}'.format(np.size(X_col,1)))
    print('X_col dim 2 is {}'.format(np.size(X_col,2)))
'''
X_col = comm.gather(sumx_seq, root=0)
if rank==0: 
    end = time.time()
    runntime=end-start
    print(runntime)
    X_col=np.array(X_col)

    file_name='GTTAC3.mat'
    savemat(file_name,{'runtime_tac3':runntime,'x_col_tac3':X_col})

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
