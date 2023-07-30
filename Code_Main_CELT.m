clc;clear;close all;
rand('seed',100);
% addpath('ClusteringMeasure')
addpath('code_coregspectral')

addpath('LRR')
addpath('twist')
addpath('funs')
addpath('cluster_measures');
%% load dataset_name

dataset_name='BDGP';
load(dataset_name);

X=data;
gnd=truth;

V=length(X);
numClust=length(unique(gnd));
N=size(gnd,1); %sample number
for ii=1:V
    X{ii}=double(full(X{ii}));
end
%% parameters setting
iter=0;
max_iter=20;
tol=1e-6;
m=100;
mu=100;
rho=100;
tao=rho/mu;
lambda=0.6;
ratio=1;

for ii=1:V
    [X{ii}]=NormalizeData(X{ii});  %d*n
    sigma(ii)=ratio*optSigma(X{ii});
end


%% Construct the initial similarity matrix
K=[];
T=cell(1,V*2);
for ii=1:V
    options.KernelType = 'Gaussian';
    options.t = sigma(ii);
    K(:,:,ii) = constructKernel(X{ii}',X{ii}',options);
    T{ii}=K(:,:,ii);
    T{V+ii}=K(:,:,ii);
end
T_tensor = cat(3, T{:,:});
t = T_tensor(:);
all_dim=0;

for ii=1:V
    dim(ii)=size(X{ii},1);
    all_dim=all_dim+size(X{ii},1);
    G{ii} = zeros(N,N);
    G{V+ii}=zeros(N,N);
    Theta{ii}=ones(dim(ii),1)./dim(ii);
end
G_tensor = cat(3, G{:,:});
S_tensor=T_tensor;
sX = [N, N, V*2];
%
H = eye(N)-1/N*ones(N);
% St = X*H*X';
% invSt = inv(St);
for ii=1:V
    invXTX{ii}=inv(10^(-6)*eye(dim(ii),dim(ii))+X{ii}*H*X{ii}');
end

W_cat_matrix=zeros(all_dim,min(m,min(dim)));
Theta_cat_matrix=zeros(all_dim,1);

while iter < max_iter
    fprintf('----processing iter %d--------\n', iter+1);
    Gpre=G_tensor;
    Spre=S_tensor;
    Wpre=W_cat_matrix;
    Thetapre=Theta_cat_matrix;
    s=S_tensor(:);
    % update G
    [g, objV] = wshrinkObj(s,tao,sX,0,3);
    G_tensor = reshape(g,sX);
    for ii=1:V*2
        G{ii}=G_tensor(:,:,ii);
    end
    % update Theta
    Theta_cat_matrix=[];
    for ii=1:V
        tmp_STheta=S_tensor(:,:,ii);
        tmp_DTheta=diag(sum(tmp_STheta));
        tmp_LTheta=tmp_DTheta-tmp_STheta;
        H_Theta=diag(diag(X{ii}*tmp_LTheta*X{ii}'));
        F=zeros(1,dim(ii));
        Aeq=ones(1,dim(ii));
        Beq=1;
        Lb=zeros(dim(ii),1);
        options_Theta = optimoptions('quadprog', 'Algorithm','interior-point-convex','Display','off');
        [Theta{ii},Theta_fval,Theta_exitflag,Theta_output,Theta_lambda] = quadprog(H_Theta,F,[],[],Aeq,Beq,Lb,[],[],options_Theta);
        Theta_cat_matrix=[Theta_cat_matrix;Theta{ii}];
    end
    
    % update W
    W_cat_matrix=[];
    for ii=1:V
        tmp_SW=S_tensor(:,:,V+ii);
        tmp_DW=diag(sum(tmp_SW));
        tmp_LW=tmp_DW-tmp_SW;
        M = invXTX{ii}*(X{ii}*tmp_LW*X{ii}');
        %         tmp_W = eig1(M, d, 0, 0);
        [tmp_W,~] = eigs(M, min(m,min(dim)));
        W{ii} =tmp_W*diag(1./sqrt(diag(tmp_W'*tmp_W)));
        W_cat_matrix=[W_cat_matrix;W{ii}];
    end
    
    % update S_Theta
    for ii=1:V
        A=zeros(N);
        distx_T = L2_distance_1(diag(Theta{ii})*X{ii},diag(Theta{ii})*X{ii});
        [temp, idx] = sort(distx_T,2);
        tmp_G=G{ii};
        for jj=1:N
            %             idxa0 = 1:N;
            dxi = distx_T(jj,:);
            ad = -(lambda*dxi-mu*tmp_G(jj,:))/(mu);
            A(jj,:) = EProjSimplex_new(ad);
        end
        S_tensor(:,:,ii)=A;
    end
    % update S_W
    for ii=1:V
        A=zeros(N);
        distx_W = L2_distance_1(W{ii}'*X{ii},W{ii}'*X{ii});
        [temp, idx] = sort(distx_W,2);
        tmp_G=G{V+ii};
        for jj=1:N
            %             idxa0 = 1:N;
            dxi = distx_W(jj,:);
            ad = -((1-lambda)*dxi-mu*tmp_G(jj,:))/(mu);
            A(jj,:) = EProjSimplex_new(ad);
        end
        S_tensor(:,:,V+ii)=A;
    end
    % check convergence
    tp_rank=rank(G{1});
    if tp_rank<1.2*numClust
        break;
    end
    iter = iter+1;
end

S = zeros(N,N);
for ii=1:V*2
    S = S + G{ii};
end
S=(S+S')/2;

[groups,feature] = SpectralClustering(S,numClust);



for ii=1:20
    
    [A_nmi_value(ii),A_ACC(ii),A_f(ii),A_p(ii),A_r(ii),A_Purity(ii),A_AR(ii),A_RI(ii),A_MI(ii),A_HI(ii),A_MIhat(ii)] = Cluster_Evaluation(groups,gnd);
end



MA_nmi_value=mean(A_nmi_value)
MA_ACC=mean(A_ACC)
MA_f=mean(A_f)
MA_p=mean(A_p)
MA_r=mean(A_r)
MA_Purity=mean(A_Purity)
MA_AR=mean(A_AR)
MA_RI=mean(A_RI)
MA_HI=mean(A_HI)
MA_MIhat=mean(A_MIhat)



