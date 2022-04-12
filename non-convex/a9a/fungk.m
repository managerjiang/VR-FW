% load('F:/anaconda_spyder/data/C_meth1_smote_800.mat')
% load('F:/anaconda_spyder/data/a9a/a9a_smote.mat');
% load('F:/anaconda_spyder/data/a9a/L_a9a_smote.mat');
% A=double(A1);
% L=double(L1);
% L(L==0)=-1;
% C=a;
% c_k=2;
% R=20;
% agent_num=size(C,1);%智能体个数
function [fv,gkv]=fungk(xx,x_dim,cen_flag,A,L,agent_num)
%     load('F:/anaconda_spyder/data/C_meth1_smote_800.mat')
%     load('F:/anaconda_spyder/data/a9a/a9a_smote.mat');
%     load('F:/anaconda_spyder/data/a9a/L_a9a_smote.mat');
%     A=double(A1);
%     L=double(L1);
%     L(L==0)=-1;
%     C=a;
%     c_k=2;
     R=20;
%     agent_num=size(C,1);%智能体个数
    fv=0;
    gkv=0;
    if cen_flag==0
        xx=reshape(xx,agent_num,x_dim);
        Xave=sum(xx,1)/agent_num;
        fv=sum(1./(1+exp(L'.*A*Xave')),1)/size(A,1);

        gg=sum(-L'.*A.*exp(L'.*A*Xave')./(1+exp(L'.*A*Xave')).^2,1)/size(A,1);
          %resul=-c_k/(2*norm(gg))*gg;
        sumd_abs=abs(gg);
        maxk=find(sumd_abs==max(sumd_abs));
        resul=zeros(1,x_dim);
        resul(maxk)=-R*sign(gg(maxk));
        Fgap=gg*(Xave-resul)';
        gkv=log10(Fgap);
    end
    if cen_flag==1
        fv=sum(1./(1+exp(L'.*A*xx')),1)/size(A,1);
        gg=sum(-L'.*A.*exp(L'.*A*xx')./(1+exp(L'.*A*xx')).^2,1)/size(A,1);
        sumd_abs=abs(gg);
        maxk=find(sumd_abs==max(sumd_abs));
        resul=zeros(1,x_dim);
        resul(maxk)=-R*sign(gg(maxk));
        Fgap=gg*(xx-resul)';
        gkv=log10(Fgap);
    end
end