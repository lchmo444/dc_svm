function [model] = dcsvm_core(trainy, trainX, C, kernel_parameters, numcluster, tol, method, kernel)
% [model] = dcsvm_core(trainy, trainX, num_cluster, tol, "mode", "method")
% 
% Arguments:
% trainy       training labels, an n by 1 matrix.  
% trainX       training data, an n by d matrix, each row is a data point. 
% C            the balancing parameter in SVM
% kernel_parameters:   for Gaussian kernel, set kernel_parameters.gamma
%                      for polynomial kernel, set kernel_parameters.gamma and kernel_parameters.degree
% numcluster   number of clusters
% tol          stopping tolerance for libsvm
% method       the clustering method, 
%              0: kernel kmeans   1: kmeans (can only used for shift-invariant kernels)
% kernel       0: Gaussian kernel 1: polynomial kernel
%

rand('seed',0);
addpath('libsvm_windows');
tol = 1e-2;
n = size(trainX,1);
d = size(trainX,2);
if kernel == 0
	libsvmcmd = sprintf('-c %g -g %g -m 8000 -e %g', C, kernel_parameters.gamma, tol);
elseif kernel==1
	libsvmcmd = sprintf('-t 1 -c %g -g %g -d %g -m 8000 -e %g -q', C, kernel_parameters.gamma, kernel_parameters.degree, tol );
end
randper = randperm(n);

if method == 1
	k = numcluster;
	%% modify this number if you want to change number of samples for clustering.  
	max_samples_for_cluster =5000;
	num = min(max_samples_for_cluster, n);
	[i centers] = kmeans(trainX(randper(1:num),:),k,'MaxIter',10,'emptyaction','singleton','Display','iter');
	dis = sum(trainX.*trainX,2)*ones(1,k)+ones(n,1)*(sum(centers.*centers,2))'-2*trainX*centers';
	[v idx] = min(dis');
	
	model.mode = 0;
	model.numcluster = numcluster;
	model.method = 1;
	model.kernel = 0;
	model.centers = centers;
elseif method ==0 
	k = numcluster;
	%% modify this number if you want to change number of samples for clustering. 
	max_samples_for_cluster = 2000;
	num = min(max_samples_for_cluster, n);
	Xsample = trainX(randper(1:num), :);

	%% Edit here if you want to add a new kernel
	if kernel==0
		Ksample = rbf(Xsample, Xsample, kernel_parameters.gamma);
		train_label = knkmeans(Ksample, k, 20);
		idx = knkmeans_rbf_predict(Xsample, trainX, train_label, kernel_parameters.gamma,  Ksample);
		model.kernel = 0;
		model.gamma = kernel_parameters.gamma;
	elseif kernel==1
		Ksample = poly(Xsample, Xsample,kernel_parameters.gamma, kernel_parameters.degree );
		train_label = knkmeans(Ksample, k, 20);
		idx = knkmeans_poly_predict(Xsample, trainX, train_label, kernel_parameters.gamma,  kernel_parameters.degree, Ksample);
		model.kernel = 1;
		model.gamma = kernel_parameters.gamma;
		model.degree = kernel_parameters.degree;
	end

	model.mode = 0;
	model.numcluster = numcluster;
	model.method = 0;
	model.train_label = train_label;
	model.Xsample = Xsample;
	model.Ksample = Ksample;
end

%% training
models={};
for i=1:k
	fprintf('Train model %g\n', i);
	models{i} = svmtrain(trainy(idx==i),trainX(idx==i,:),libsvmcmd);
end
%% fill in models
model.models = models;

