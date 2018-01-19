function [model] = dcsvm_poly_train(trainy, trainX, C, gamma, degree, ncluster)
% [model] = dcsvm_poly_train(trainy, trainX, C, gamma, degree, ncluster)
% 
%
% Arguments:
% trainy       training labels, an n by 1 matrix.  
% trainX       training data, an n by d matrix, each row is a data point. 
% C            the balancing parameter in SVM
% gamma        the kernel parameter for polynomial kernel
% degree       the kernel parameter for polynomial kernel
%              K(x,y) = (gamma*u'*v + 1)^degree
% ncluster     number of clusters 
%

rand('seed',0);

n = size(trainX,1);
d = size(trainX,2);

kernel_parameters.gamma = gamma;
kernel_parameters.degree = degree;
tol = 1e-3;

mode = 0; %% Early Prediction (faster and stable)
%mode = 1; %% A hierachical DC-SVM for exact solution, use this if you want an exact SVM solution. 

method = 0; %% clustering by kernel kmeans. 

kernel = 1; %% polynomial kernel

level = 4;  %% will be used if mode=1
level_stop = 1; %% will be used if mode=1 to get exact solution
kk = 4; %% will be used if mode=1

model = dcsvm_core(trainy, trainX, C, kernel_parameters, ncluster, level, level_stop, kk, tol, mode, method, kernel);


