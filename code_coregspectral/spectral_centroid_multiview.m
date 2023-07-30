function [MA_nmi_value,MA_ACC,MA_AR,MA_f,MA_p,MA_r,MA_Purity,similarityMatirx] = spectral_centroid_multiview(X,num_views,numClust,sigma,lambda,truth,numiter)
% INPUT:
% OUTPUT:
    
    if (min(truth)==0)
        truth = truth + 1;
    end
    N=size(truth,1);

    
    for ii=1:num_views
        %options(i) = [];
        options(ii).KernelType = 'Gaussian';
        options(ii).t = sigma(ii);
        options(ii).d = 4;
    end
        
    kmeans_avg_iter = 10;
    opts.disp = 0;

    numEV = numClust;
    numVects = numClust;
    for ii=1:num_views
    % Laplacian for the first view of the data
%         fprintf('computing kernel for X(%d)\n',i);
        K(:,:,ii) = constructKernel(X{ii}',X{ii}',options(ii));
        %K1 = X1*X1';
        D = diag(sum(K(:,:,ii),1));
        %L1 = D1 - K1; 
        L(:,:,ii) = sqrt(inv(D))*K(:,:,ii)*sqrt(inv(D));  
        L(:,:,ii)=(L(:,:,ii)+L(:,:,ii)')/2;
        [U(:,:,ii) E] = eigs(L(:,:,ii),numEV,'LA',opts);    
        objval(ii,1) = sum(diag(E));
    end
    
    %%do clustering for first view
    U1 = U(:,:,1);
    normvect = sqrt(diag(U1*U1'));
    normvect(find(normvect==0.0)) = 1;
    U1 = inv(diag(normvect)) * U1;    
    for j=1:kmeans_avg_iter
        C = kmeans(U1(:,1:numVects),numClust,'EmptyAction','drop'); 
        [Fj(j),Pj(j),Rj(j)] = compute_f(truth,C); 
        [Aj nmi_j(j) avgent_j(j)] = compute_nmi(truth,C);
        [ARj(j),RIj(j),MIj(j),HIj(j)]=RandIndex(truth,C);
        ACCj(j) = accuracyMeasure(truth,C);
        Purityj(j)=purityMeasure(truth,C);
    end
    F(1) = mean(Fj); std_F(1) = std(Fj);
    P(1) = mean(Pj); std_P(1) = std(Pj);
    R(1) = mean(Rj); std_R(1) = std(Rj);
    nmi(1) = mean(nmi_j); std_nmi(1) = std(nmi_j);
    avgent(1) = mean(avgent_j); std_avgent(1) = std(avgent_j);
    AR(1) = mean(ARj); std_AR(1) = std(ARj);
    acc(1) = mean(ACCj);std_ACC(1)=std(ACCj);
    purity(1)=mean(Purityj);std_Purity(1)=std(Purityj);
    
    i = 2;
    % now iteratively solve for all U's
    while(i<=numiter+1)
%         fprintf('Running iteration %d\n',i-1);
        
        L_ustar(1:N,1:N) = 0;
        for j=1:num_views
            L_ustar = L_ustar + lambda(j)*U(:,:,j)*U(:,:,j)';
        end
        L_ustar = (L_ustar+L_ustar')/2;
        [Ustar Estar] = eigs(L_ustar, numEV,'LA',opts);    
        
        L_ustar = Ustar*Ustar';
        L_ustar = (L_ustar+L_ustar')/2;
        for j=1:num_views            
            [U(:,:,j) E] = eigs(L(:,:,j) + lambda(j)*L_ustar, numEV,'LA',opts);    
            objval(j,i) = sum(diag(E));
        end

        objval(1,i) = sum(diag(E));
        
        if i == numiter+1
            similarityMatirx = L(:,:,1) + lambda(1)*L_ustar;
        end
        
                
        if (1)  %use view 1 in actual clustering
            U1 = Ustar;
            normvect = sqrt(diag(U1*U1'));    
            normvect(find(normvect==0.0)) = 1;
            U1 = inv(diag(normvect)) * U1;
            
            for j=1:kmeans_avg_iter
                C = kmeans(U1(:,1:numVects),numClust,'EmptyAction','drop'); 
                [Fj(j),Pj(j),Rj(j)] = compute_f(truth,C); 
                [Aj nmi_j(j) avgent_j(j)] = compute_nmi(truth,C);
                [ARj(j),RIj(j),MIj(j),HIj(j)]=RandIndex(truth,C);
                ACCj(j) = accuracyMeasure(truth,C);
                Purityj(j)=purityMeasure(truth,C);
            end
            F(i) = mean(Fj); std_F(i) = std(Fj);
            P(i) = mean(Pj); std_P(i) = std(Pj);
            R(i) = mean(Rj); std_R(i) = std(Rj); 
            nmi(i) = mean(nmi_j); std_nmi(i) = std(nmi_j);
            avgent(i) = mean(avgent_j); std_avgent(i) = std(avgent_j);
            AR(i) = mean(ARj); std_AR(i) = std(ARj);
            acc(i) = mean(ACCj);std_ACC(i)=std(ACCj);
            purity(i)=mean(Purityj);std_Purity(i)=std(Purityj);
        end
        i = i+1;
    end
    
    
    %%%CCA on U1 and U2
    %i = i+1;
    %[feats1 feats2 F_c P_c R_c nmi_c avgent_c] = multiviewccacluster(U1_norm, U2_norm, numClust, sigma1, sigma2, truth);
%     
%     fprintf('F:   ');
%     for i=1:numiter+1
%         fprintf('%f(%f)  ', F(i), std_F(i));
%     end
%     fprintf('\n\n');
%     fprintf('P:   ');    
%     for i=1:numiter+1      
%         fprintf('%f(%f)  ', P(i), std_P(i));
%     end
%     fprintf('\n\n');
%     fprintf('R:   ');    
%     for i=1:numiter+1      
%         fprintf('%f(%f)  ', R(i), std_R(i));
%     end
%     fprintf('\n\n');
%     fprintf('nmi:   ');    
%     for i=1:numiter+1      
%         fprintf('%f(%f)  ', nmi(i), std_nmi(i));
%     end
%     fprintf('\n\n');
%     fprintf('acc:   ');    
%     for i=1:numiter+1      
%         fprintf('%f(%f)  ', acc(i), std_ACC(i));
%     end
%     fprintf('\n\n');
%     fprintf('AR:   ');    
%     for i=1:numiter+1      
%         fprintf('%f(%f)  ', AR(i), std_AR(i));
%     end
%     fprintf('\n\n');
%     fprintf('Purity:   ');    
%     for i=1:numiter+1      
%         fprintf('%f(%f)  ', purity(i), std_Purity(i));
%     end
%     fprintf('\n\n');
%     for j=1:num_views
%         fprintf('objval_u%d:   ', j);    
%         for i=1:numiter+1
%             fprintf('%f  ', objval(j,i));
%         end
%         fprintf('\n');
%     end
        
    if (0)
    %%%%averaging of U1 and U2
    V = (U1_norm+U2_norm)/2;
    normvect = sqrt(diag(V*V'));
    normvect(find(normvect==0.0)) = 1;
    V = inv(diag(normvect)) * V;
    %U = U./repmat(sqrt(sum(U.*U,2)),1,numClust*2); % normalize
    for j=1:kmeans_avg_iter
        C = kmeans(V(:,1:numVects),numClust,'EmptyAction','drop'); 
        [Fj(j),Pj(j),Rj(j)] = compute_f(truth,C); 
        [Aj nmi_j(j) avgent_j(j)] = compute_nmi(truth,C);
        [ARj(j),RIj(j),MIj(j),HIj(j)]=RandIndex(truth+1,C);
        ACCj(j) = accuracyMeasure(truth,C);
        Purityj(j)=purityMeasure(truth,C);
    end
    i = i+1;
    F(i) = mean(Fj);
    P(i) = mean(Pj);
    R(i) = mean(Rj);
    nmi(i) = mean(nmi_j);
    avgent(i) = mean(avgent_j);
    AR(i) = mean(ARj);    
    acc(i) = mean(ACCj);
    purity(i)=mean(Purityj);
    C = kmeans(U,numClust,'EmptyAction','drop');  
    %[F(i),P(i),R(i)] = compute_f(truth,C); 
    [A nmi(i) avgent(i)] = compute_nmi(truth,C);
        
    end
    MA_nmi_value=mean(nmi);
    MA_ACC=mean(acc);
    MA_f=mean(F);
    MA_p=mean(P);
    MA_r=mean(R);
    MA_Purity=mean(purity);
    MA_AR=mean(AR);