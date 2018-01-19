function [label] = knkmeans_poly_predict(Xtrain, Xtest, train_label, gamma, degree, Ktrain)
%% [label] = knkmeans_poly_predict(Xtrain, Xtest, train_label, gamma, degree, Ktrain)

chunk = 10000;
ntrain = size(Xtrain,1);
ntest = size(Xtest, 1);


i=0;
k = max(train_label);
E = sparse(train_label, 1:ntrain, 1, k, ntrain, ntrain);
E = bsxfun(@times, E, 1./sum(E,2));
T = E*Ktrain;
Z = diag(T*E');
E = E';
label = [];
while 1
	list = (chunk*i+1):min(ntest, chunk*(i+1));
	now_ntest = numel(list);
	Ktest = poly(Xtest(list,:),Xtrain, gamma, degree);
	T = repmat(Z',now_ntest, 1) - 2*Ktest*E;
	[~, now_label] = min(T,[],2);
	label = [label; now_label];
	if list(end) == ntest
		break;
	end

	i=i+1;
end




