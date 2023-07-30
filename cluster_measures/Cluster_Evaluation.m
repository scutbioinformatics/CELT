function [nmi_value,ACC,f,p,r,Purity,AR,RI,MI,HI,MIhat] = Cluster_Evaluation(G,T)

nmi_value = nmi(G,T);  %%% normalized mutural information

[ACC,MIhat,Purity] = ClusteringMeasure(T,G);

[f,p,r] = compute_f(T,G);

[AR,RI,MI,HI] = RandIndex(G,T);