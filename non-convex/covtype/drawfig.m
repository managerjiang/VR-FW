close all;
clc;
clear;
load('F:/anaconda_spyder/data/C_meth1_smote_800.mat')
load('GTTAC.mat');%
load('GTSPIDER.mat');%
load('CENSPIDER.mat');%
load('GTTAC1.mat');%
load('GTSPIDER1.mat');%
load('CENSPIDER1.mat');%
load('GTTAC2.mat');%
load('GTSPIDER2.mat');%
load('CENSPIDER2.mat');%
load('GTTAC3.mat');%
load('GTSPIDER3.mat');%
load('CENSPIDER3.mat');%

load('F:/anaconda_spyder/data/covtype/cov.mat');
load('F:/anaconda_spyder/data/covtype/L_cov.mat');
L(L==0)=-1;
C=a;
c_k=2;
Maxgen1=size(x_col_spider,2)-1;%迭代次数
x_dim=size(x_col_spider,3);
agent_num=size(C,1);%智能体个数

for k=1:Maxgen1
    [fi0,fw0]=fungk(x_col_spider(:,k,:),x_dim,0,A,L,agent_num);    
    [fi1,fw1]=fungk(x_col_spider1(:,k,:),x_dim,0,A,L,agent_num);    
    [fi2,fw2]=fungk(x_col_spider2(:,k,:),x_dim,0,A,L,agent_num);   
    [fi3,fw3]=fungk(x_col_spider3(:,k,:),x_dim,0,A,L,agent_num);    
    fi=(fi0+fi1+fi2+fi3)/4;
    obj_m1(k)=fi;
    fw=(fw0+fw1+fw2+fw3)/4;
    fwgap1(k)=fw;
    
end
%%meth tac
for k=1:Maxgen1
    [fi0,fw0]=fungk(x_col_tac(:,k,:),x_dim,0,A,L,agent_num);    
    [fi1,fw1]=fungk(x_col_tac1(:,k,:),x_dim,0,A,L,agent_num);    
    [fi2,fw2]=fungk(x_col_tac2(:,k,:),x_dim,0,A,L,agent_num);   
    [fi3,fw3]=fungk(x_col_tac3(:,k,:),x_dim,0,A,L,agent_num);    
    fi=(fi0+fi1+fi2+fi3)/4;
    obj_m2(k)=fi;
    fw=(fw0+fw1+fw2+fw3)/4;
    fwgap2(k)=fw;
end


%%meth censpider
for k=1:Maxgen1
   [fi0,fw0]=fungk(x_col_censpider(k,:),x_dim,1,A,L,agent_num);    
     [fi1,fw1]=fungk(x_col_censpider1(k,:),x_dim,1,A,L,agent_num);    
     [fi2,fw2]=fungk(x_col_censpider2(k,:),x_dim,1,A,L,agent_num);   
     [fi3,fw3]=fungk(x_col_censpider3(k,:),x_dim,1,A,L,agent_num);
    fi=(fi0+fi1+fi2+fi3)/4;
    obj_m3(k)=fi;
    fw=(fw0+fw1+fw2+fw3)/4;
    fwgap3(k)=fw;
end

figure(1);
tt=1:5:Maxgen1;
plot(tt(1:2:length(tt)),obj_m1(tt(1:2:length(tt))),'r-x','linewidth',1),hold on;
plot(tt(1:2:length(tt)),obj_m2(tt(1:2:length(tt))),'k-^','linewidth',1),hold on;
plot(tt(1:2:length(tt)),obj_m3(tt(1:2:length(tt))),'b-o','linewidth',1),hold on;
legend("FontName","Times New Roman","FontSize",14); 
legend('DstoFW','DenFW','CenFW');
ylabel('$$F(\bar{x}^k)$$','Interpreter','latex',"FontSize",17)
xlabel('k',"FontSize",17);

figure(2);
plot(tt(1:2:length(tt)),fwgap1(tt(1:2:length(tt))),'r-x','linewidth',1),hold on;
plot(tt(1:2:length(tt)),fwgap2(tt(1:2:length(tt))),'k-^','linewidth',1),hold on;
plot(tt(1:2:length(tt)),fwgap3(tt(1:2:length(tt))),'b-o','linewidth',1),hold on;
legend("FontName","Times New Roman","FontSize",14); 
legend('DstoFW','DenFW','CenFW');
ylabel('$$\log g_k$$','Interpreter','latex',"FontSize",17)
xlabel('k',"FontSize",17);

% figure(3);
% plot(1:Maxgen1,const1,'r','linewidth',1),hold on;
% plot(1:Maxgen1,const2,'k-.','linewidth',1),hold on;
% plot(1:Maxgen1,const3,'b--','linewidth',1),hold on;
% legend('DstoFW','DenFW','CenFW');
% ylabel('$$\Omega$$','Interpreter','latex')
% xlabel('iterations');
