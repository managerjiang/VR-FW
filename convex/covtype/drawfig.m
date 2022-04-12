close all;
clc;
clear;
load('F:/anaconda_spyder/data/C_meth1_smote_800.mat')
load('GTTAC.mat');%加载保存的迭代梯度信息x_k_store{}
load('GTSPIDER.mat');%加载保存的迭代解信息gradient()
load('CENSPIDER.mat');%加载保存的邻接矩阵信息C_store

load('F:/anaconda_spyder/data/covtype/cov.mat');
load('F:/anaconda_spyder/data/covtype/L_cov.mat');
L(L==0)=-1;
C=a;
c_k=2;
R=20;
Maxgen1=size(x_col_spider,2)-1;%迭代次数
x_dim=size(x_col_spider,3);
agent_num=size(C,1);%智能体个数

for k=1:Maxgen1
    x_k_agent=x_col_spider(:,k,:);
    x_k_agent=reshape(x_k_agent,agent_num,x_dim);
    x_k_1=x_k_agent(1,:);
    Xave=sum(x_k_agent,1)/agent_num;
    fi=sum(log((1+exp(-L'.*A*Xave'))),1);
    obj_m1(k)=fi/size(A,1);
     XLX=0;
  for i=1:agent_num
    temp_X=zeros(x_dim,1);
    for j=1:agent_num
        temp_X=temp_X+C(i,j)*(x_k_agent(i,:)'-x_k_agent(j,:)');
    end
    XLX=XLX+x_k_agent(i,:)*temp_X;
  end
  w_m1s(k)=XLX;
  gg=sum(-L'.*A.*exp(-L'.*A*Xave')./(1+exp(-L'.*A*Xave')),1)/size(A,1);
  %resul=-c_k/(2*norm(gg))*gg;
   sumd_abs=abs(gg);
  maxk=find(sumd_abs==max(sumd_abs));
  resul=zeros(1,x_dim);
  resul(maxk)=-R*sign(gg(maxk));
  Fgap=gg*(Xave-resul)';
  fwgap1(k)=log10(Fgap);
  const1(k)=norm(Xave);
end
%%meth tac
for k=1:Maxgen1
    x_k_agent=x_col_tac(:,k,:);
    x_k_agent=reshape(x_k_agent,agent_num,x_dim);
    x_k_1=x_k_agent(1,:);
    Xave=sum(x_k_agent,1)/agent_num;
    fi=sum(log((1+exp(-L'.*A*Xave'))),1);
    obj_m2(k)=fi/size(A,1);
     XLX=0;
  for i=1:agent_num
    temp_X=zeros(x_dim,1);
    for j=1:agent_num
        temp_X=temp_X+C(i,j)*(x_k_agent(i,:)'-x_k_agent(j,:)');
    end
    XLX=XLX+x_k_agent(i,:)*temp_X;
  end
  w_m2s(k)=XLX;
  gg=sum(-L'.*A.*exp(-L'.*A*Xave')./(1+exp(-L'.*A*Xave')),1)/size(A,1);
  %resul=-c_k/(2*norm(gg))*gg;
   sumd_abs=abs(gg);
  maxk=find(sumd_abs==max(sumd_abs));
  resul=zeros(1,x_dim);
  resul(maxk)=-R*sign(gg(maxk));
  Fgap=gg*(Xave-resul)';
  fwgap2(k)=log10(Fgap);
  const2(k)=norm(Xave);
end


%%meth censpider
for k=1:Maxgen1
    x_k_1=x_col_censpider(k,:);
    fi=sum(log((1+exp(-L'.*A*x_k_1'))),1);
    obj_m3(k)=fi/size(A,1);
  gg=sum(-L'.*A.*exp(-L'.*A*x_k_1')./(1+exp(-L'.*A*x_k_1')),1)/size(A,1);
  %resul=-c_k/(2*norm(gg))*gg;
  sumd_abs=abs(gg);
  maxk=find(sumd_abs==max(sumd_abs));
  resul=zeros(1,x_dim);
  resul(maxk)=-R*sign(gg(maxk));
  Fgap=gg*(x_k_1-resul)';
  fwgap3(k)=log10(Fgap);
  const3(k)=norm(x_k_1);
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
