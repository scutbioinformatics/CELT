
function [NMI,ACC,AR,F,P,R,Purity,CKSym] = diverse_msc(X,gt,lambda_s,lambda_v)
%data preparation
view_num = length(X);
for v=1:view_num
    D{v} = X{v};
end


N = size(D{1},2);
clusNum = size(unique(gt),1);

%representation
Z = diverse_rep(D, view_num, lambda_s, lambda_v);
%spectral Clustering
CKSym = zeros(N,N);
for v=1:view_num
    CKSym =CKSym + abs(Z{v})+abs(Z{v}');
end

C = SpectralClustering(CKSym,clusNum);
% %2019.12.12 start
rng('default');Y = tsne(CKSym,'Algorithm','exact','Distance','correlation' ,'Standardize',true,'Perplexity',26);% Plot the result  MSRC_v1  
figure;gscatter(Y(:,1),Y(:,2),C); % title('tsne   48patients')
displayClusters(CKSym,C,0);
figure; h=imagesc(CKSym);
% %2019.12.12 end
[A NMI avgent] = compute_nmi(gt,C);
[F,P,R] = compute_f(gt,C);
[AR,RI,MI,HI]=RandIndex(gt,C);
C = bestMap(gt,C);
ACC = length(find(gt == C))/length(gt);
Purity = purityMeasure(gt,C);
%[Acc,rand_index,match]=AccMeasure(gt,C);
%[confusion_matrix,trace_max]=confusion_compute(label_predict,num_each_class);

