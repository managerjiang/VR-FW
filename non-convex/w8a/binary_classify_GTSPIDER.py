# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 21:52:39 2021

@author: JX
"""

from __future__ import division 
#fruture前后是两个下划线：__future__
#上述是为了希望只执行简单除法，例如1/2=0.5这样的,单斜线不再用作整除，双斜线实现整除操作
from mpi4py import MPI
from scipy.io import savemat
import numpy as np
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
a9asam=scio.loadmat('F:/anaconda_spyder/data/w8a/w8a_smote.mat') #载入A：A(48243x123):48243个数据
a9al=scio.loadmat('F:/anaconda_spyder/data/w8a/L_w8a_smote.mat') #载入L：A(1x48243):48243个结果
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
#X_i=np.zeros(x_dim)
X_i=np.load('F:/anaconda_spyder/non-convex/data/classify/xi0_w8a.npy')
listx_i.append(X_i)
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
#针对集合S中的索引确定样本，并求这些样本的在变量vari_x处的梯度
def gradient_com_set(vari_x,S):
    gradient_vari_x=[0.0]*x_dim
    for j in S:
        mid=recv_data_l[j]*recv_data[j,:]
        gradient_x=-mid*math.exp(mid@vari_x)/(1+math.exp(mid@vari_x))**2;
        gradient_vari_x=gradient_vari_x+gradient_x
    return gradient_vari_x
#梯度g和v初始化
gradientx_i_seq=[]
v_i_seq=[]
gradientx_i=gradient_com(X_i)
gradientx_i_seq.append(gradientx_i)
v0=gradient_com(X_i)#v初始化
#print("process {} initial grad {}".format(rank,gradientx_i_seq[0]))
 
#开始迭代
T=300
c_k=2
R=20
q=math.ceil(sample_n**(1/4)) #迭代多少次计算一次完整本地梯度
#sumx=X_i
for t in range(T):
    etak=1/((t+1)**0.5)  #迭代步长
    for j in range(agent_n):                
        if C[rank][j] != 0:
            comm.send(listx_i[t], dest=j, tag=rank)
            comm.send(gradientx_i_seq[t],dest=j,tag=rank+10)
    neigh_x=[0]*agent_n
    neigh_gra=[0]*agent_n
    for j in range(agent_n):
        if C[rank][j] != 0:
            neigh_x[j]=comm.recv(source=j, tag=j)
            neigh_gra[j]=comm.recv(source=j, tag=j+10)
            #print("process {} recv samples {} and {} from agent {}...".format(rank,neigh_x,neigh_gra,j))
    sumd=[0]*x_dim
    for j in range(agent_n):
        sumd=sumd+C[rank][j]*neigh_gra[j]
    #resul=-c_k/(2*np.linalg.norm(sumd))*sumd
    
    sumd_abs=list(map(abs,sumd))
    maxk=sumd_abs.index(max(sumd_abs))
    resul=np.zeros(x_dim) #.reshape(-1,1)
    resul[maxk]=-R*np.sign(sumd[maxk])
    
    sumx=[0]*x_dim
    for j in range(agent_n):
        sumx=sumx+C[rank][j]*neigh_x[j]   
    X_i=(1-etak)*sumx+etak*resul
    if t%q==0: #########
        v1=gradient_com(X_i)
    else:
        etasub=1/((t)**0.5)
        nn=t//q+1
        etaup=1/((nn*q-1)**0.5)
        batch_size=q*q*etasub**2/etaup**2 #4*q*q #100  20*(q-(t%q)) #20*(q-t//q) ##可调节参数，采样个数
        batch_size=math.ceil(batch_size)
        #print('iter {} is {}'.format(t,batch_size))
        S=np.random.choice(ave, batch_size)
        v1=(gradient_com_set(X_i,S)-gradient_com_set(listx_i[t],S))/batch_size+v0       
    gradientx_i=sumd+v1-v0
    v0=v1
#    sumx=sumx1
    gradientx_i_seq.append(gradientx_i)
    listx_i.append(X_i)


#迭代结果序列（函数值和一致性结果）10 T+1 123
'''
    print('rank {} is X_col is {}'.format(rank, X_col[1][1]))
    print('X_col dim 0 is {}'.format(np.size(X_col,0)))
    print('X_col dim 1 is {}'.format(np.size(X_col,1)))
    print('X_col dim 2 is {}'.format(np.size(X_col,2)))
'''
X_col = comm.gather(listx_i, root=0)
if rank==0:  
    end = time.time()
    runntime=end-start
    print(runntime)

    X_col=np.array(X_col)

    file_name='GTSPIDER.mat'
    savemat(file_name,{'runtime_spider':runntime,'x_col_spider':X_col})
'''    
    consenseq=[]
    fvseq=[]
    for t in range(T):
        fv=0.0
        for i in range(sample_n):
            fv=fv+np.log(1+np.exp(-L[i]*A[i,:]@X_col[1][t]))
        #print('rank {} coll fv is {}'.format(rank,fv))
        fvseq.append(fv/sample_n)
        consen=0.0;
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