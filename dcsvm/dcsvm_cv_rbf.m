function [kernel_parameters] = dcsvm_cv_rbf(trainy, trainX, ncluster);

nfold = 3;
clist = 2.^(-10:5:10);
glist = 2.^(-10:5:10);
n = size(trainX, 1);
per = randperm(n);
del = ceil(n/nfold);
for i=1:nfold
	nowlist = (del*(i-1)+1):min(del*i, n);
	ind{i} = per(nowlist);
end

acc = zeros(numel(clist), numel(glist));
for f=1:nfold
	nowtrainlist = [];
	for i=1:nfold
		if i~=f
			nowtrainlist = [nowtrainlist ind{i}];
		end
	end
	nowtrainX = trainX(nowtrainlist, :);
	nowtrainy = trainy(nowtrainlist);
	nowtestX = trainX(ind{f},:);
	nowtesty = trainy(ind{f});

	for cc = 1:numel(clist)
		for gg = 1:numel(glist)
			model = dcsvm_rbf_train(nowtrainy, nowtrainX, clist(cc), glist(gg), ncluster);
			[labels accuracy] = dcsvm_test(nowtesty, nowtestX, model);
			acc(cc, gg) = acc(cc,gg) + accuracy;
			acc
		end
	end
end
acc = acc/nfold;

[val, I] = max(acc(:));
[bestc bestgamma] = ind2sub(size(acc), I(1));
kernel_parameters.gamma = glist(bestgamma);
kernel_parameters.C = clist(bestc);
