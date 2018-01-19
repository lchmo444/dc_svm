function [model] = dcsvm_rbf_train(trainy, trainX, C, gamma, ncluster)
% [model] = dcsvm_rbf_train(trainy, trainX, C, gamma, ncluster)
% 
%
% Arguments:
% trainy       training labels, an n by 1 matrix.  
% trainX       training data, an n by d matrix, each row is a data point. 
% C            the balancing parameter in SVM
% gamma        the kernel parameter for Gaussian kernel
%              K(x,y) = exp(-gamma*||x-y||_2^2)
% ncluster     number of clusters 
%

rand('seed',0);

n = size(trainX,1);
d = size(trainX,2);

kernel_parameters.gamma = gamma;
tol = 1e-2;

method = 1; %% clustering by kmeans, can only be used for shift invariant kernels. 
%method = 0; %% clustering by kernel kmeans. 

kernel = 0; %% RBF kernel

model = dcsvm_core(trainy, trainX, C, kernel_parameters, ncluster, tol, method, kernel);

