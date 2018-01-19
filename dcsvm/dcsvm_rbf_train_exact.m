function [model] = dcsvm_rbf_train_exact(trainy, trainX, C, gamma)
% [model] = dcsvm_rbf_train_exact(trainy, trainX, C, gamma)
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

%mode = 0; %% Early Prediction (faster and stable)
mode = 1; %% A hierachical DC-SVM for exact solution, use this if you want an exact SVM solution. 

%method = 1; %% clustering by kmeans, can only be used for shift invariant kernels. 
method = 0; %% clustering by kernel kmeans. 

kernel = 0; %% RBF kernel

level = 4;  %% will be used if mode=1
level_stop = 1; %% will be used if mode=1 to get exact solution
% level_stop = 0; %% the final solution, usually not needed. 
kk = 4; %% will be used if mode=1
 ncluster = 10; %% can be anything if mode=1

model = dcsvm_core(trainy, trainX, C, kernel_parameters, ncluster, level, level_stop, kk, tol, mode, method, kernel);


