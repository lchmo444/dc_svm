function [model] = dcsvm_core(trainy, trainX, C, kernel_parameters, numcluster, level, level_stop, kk, tol, mode, method, kernel)
% [model] = dcsvm_core(trainy, trainX, num_cluster, level, tol, "mode", "method")
% 
% Arguments:
% trainy       training labels, an n by 1 matrix.  
% trainX       training data, an n by d matrix, each row is a data point. 
% C            the balancing parameter in SVM
% kernel_parameters:   for Gaussian kernel, set kernel_parameters.gamma
%                      for polynomial kernel, set kernel_parameters.gamma and kernel_parameters.degree
% numcluster   number of clusters
% level        total levels 
% level_stop   stopping level
% kk           size of branch
% tol          stopping tolerance for libsvm
% mode         0: dcsvm-early (default)    1: dcsvm (solving the exact kernel SVM problem)
% method       the clustering method, 
%              0: kernel kmeans   1: kmeans (can only used for shift-invariant kernels)
% kernel       0: Gaussian kernel 1: polynomial kernel
%

rand('seed',0);
addpath('../libsvm-3.14-nobias/matlab');
tol = 1e-2;
n = size(trainX,1);
d = size(trainX,2);
if kernel == 0
	libsvmcmd = sprintf('-c %g -g %g -m 8000 -e %g', C, kernel_parameters.gamma, tol);
elseif kernel==1
	libsvmcmd = sprintf('-t 1 -c %g -g %g -d %g -m 4000 -e %g -q', C, kernel_parameters.gamma, kernel_parameters.degree, tol );
end
randper = randperm(n);

if mode == 0
	if method == 1
		k = numcluster;
		%% modify this number if you want to change number of samples for clustering.  
		max_samples_for_cluster =5000;
		num = min(max_samples_for_cluster, n);
		tic;
		[i centers] = kmeans(trainX(randper(1:num),:),k,'MaxIter',10,'emptyaction','singleton','Display','iter');
		toc
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
	centers = zeros(k,size(trainX,2));
	for i=1:k
		centers(i,:) = mean(trainX(idx==i,:),1);
	end
	%% fill in models
	model.models = models;

elseif mode==1
	if method == 1
		k = numcluster;
		%% modify this number if you want to change number of samples for clustering.  
		max_samples_for_cluster = 20000;
		num = min(max_samples_for_cluster, n);
		Xsample = trainX(randper(1:num), :);

		totalnum = 1;
		all_ind = {};
		all_ind{1} = ones(n,1);
		sample_ind{1} = ones(num,1);
		mincluster = 500;

		for i=1:level-1
			nowid = 0;
			sample_ind{i+1} = zeros(num,1);
			for cid =1:max(sample_ind{i})
				subind = find(sample_ind{i} == cid);
				subind_all = find(all_ind{i} == cid);
				if numel(subind) < mincluster
					all_ind{i}(subind_all) = 0;
					sample_ind{i+1}(subind) = nowid+1;
					all_ind{i+1}(subind_all) = nowid+1;
					nowid = nowid+1;
					continue
				end
				[idx centers] = kmeans(Xsample(subind, :),kk,'MaxIter',10,'emptyaction','singleton');
				for ii=1:kk
					sample_ind{i+1}(subind(idx==ii)) = nowid+ii;
				end
				idx = kmeans_predict(trainX(subind_all, :), centers);
				for ii=1:kk
					all_ind{i+1}(subind_all(idx==ii)) = nowid+ii;
				end
				nowid = nowid + kk;
			end
		end
		model.kernel = 0;
	elseif method == 0
		k = numcluster;
		%% modify this number if you want to change number of samples for clustering.  
		max_samples_for_cluster = 2000;
		num = min(max_samples_for_cluster, n);
		Xsample = trainX(randper(1:num), :);
		if kernel == 0
			Ksample = rbf(Xsample, Xsample, kernel_parameters.gamma);
		elseif kernel == 1
			Ksample = poly(Xsample, Xsample, kernel_parameters.gamma, kernel_parameters.degree);
		end

		totalnum = 1;
		all_ind = {};
		all_ind{1} = ones(n,1);
		sample_ind{1} = ones(num,1);
		mincluster = ceil(num/(kk^level)*5);

		for i=1:level-1
			nowid = 0;
			sample_ind{i+1} = zeros(num,1);
			samecluster{i+1} = [];
			for cid =1:max(sample_ind{i})
				subind = find(sample_ind{i} == cid);
				if numel(subind) < mincluster
					sample_ind{i+1}(subind) = nowid+1;
					samecluster{i+1} = [samecluster{i+1}; cid, nowid+1];
					nowid = nowid+1;
					continue
				end
				idx = knkmeans(Ksample(subind, subind), kk, 20);
				for ii=1:kk
					sample_ind{i+1}(subind(idx==ii)) = nowid+ii;
				end
				nowid = nowid + kk;
			end
		end
		
		if kernel == 0
			all_ind = knkmeans_rbf_predict_alllevel(Xsample, trainX, level, sample_ind, kernel_parameters.gamma, Ksample, samecluster);
			model.kernel = 0;
		elseif kernel == 1
			all_ind = knkmeans_poly_predict_alllevel(Xsample, trainX, level, sample_ind, kernel_parameters.gamma, kernel_parameters.degree, Ksample, samecluster);
			model.kernel = 1;
		end
	end
	%% training
	alpha = zeros(n,1);
	for ll=level:-1:max(level_stop,2)
		fprintf('Training Level %g\n',ll);
		ksub = max(all_ind{ll});
		if ll == level_stop
			models = {};
			centers = [];
		end
		for i=1:ksub
%			libsvmcmd = sprintf('-c %g -g %g -m 4000 -e 0.05', C, gamma);
			nowindices = find(all_ind{ll}==i);
%			fprintf('training cluster %g size %g', i, numel(nowindices));
			if numel(nowindices)==0
				continue
			end
			if nnz(alpha(nowindices))==0
				[mm obj initial_time] = svmtrain(trainy(nowindices), trainX(nowindices,:),libsvmcmd);
			else
				[mm obj initial_time] = svmtrain(trainy(nowindices),trainX(nowindices,:),alpha(nowindices),libsvmcmd);
			end
			alpha(nowindices) = 0;
			alpha(nowindices(mm.sv_indices)) = abs(mm.sv_coef);
			if ll == level_stop
				models{i} = mm;
				centers = [centers; mean(trainX(nowindices,:),1)];
			end
		end
	end
	if level_stop >= 2
		model.mode = 0;
		model.numcluster = max(all_ind{level_stop});
		model.method = 1;
		model.centers = centers;
		model.models = models;
	else
		fprintf('Training Level 1\n');
		nowindices = find(alpha~=0);
		[mm_tmp obj initial_time] = svmtrain(trainy(nowindices), trainX(nowindices,:), alpha(nowindices), libsvmcmd);
		if level_stop == 0
			alpha(nowindices(mm_tmp.sv_indices)) = abs(mm_tmp.sv_coef);
			[mm_tmp obj initial_time] = svmtrain(trainy, trainX, alpha, libsvmcmd);
		end
		model.mode = 1;
		model.model = mm_tmp;
	end
end


