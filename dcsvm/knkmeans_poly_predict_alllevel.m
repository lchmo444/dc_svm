function [label] = knkmeans_rbf_predict_alllevel(Xtrain, Xtest, level, sample_ind, gamma,degree, Ktrain, samecluster)

chunk = 10000;
ntrain = size(Xtrain,1);
ntest = size(Xtest, 1);

for ll = 2:level
	k = max(sample_ind{ll});
	E = sparse(sample_ind{ll}, 1:ntrain, 1, k, ntrain, ntrain);
	E = bsxfun(@times, E, 1./sum(E,2));
	T = E*Ktrain;
	ZZ{ll} = diag(T*E');
	EE{ll} = E';
end


i=0;
for ll=2:level
	label{ll} = [];
end
while 1
	list = (chunk*i+1):min(ntest, chunk*(i+1));
	now_ntest = numel(list);
	Ktest = poly(Xtest(list,:),Xtrain, gamma, degree);
	for ll = 2:level
		T = repmat(ZZ{ll}', now_ntest, 1) - 2*Ktest*EE{ll};
		[~, now_label] = min(T,[],2);
		label{ll} = [label{ll}; now_label];
	end
	if list(end) == ntest
		break;
	end
	i=i+1;
end

for ll=2:level
	for ii = 1:size(samecluster{ll}, 1)
%		keyboard
		idx = find(label{ll-1} == samecluster{ll}(ii,1));
		label{ll}(idx) = samecluster{ll}(ii,2);
		label{ll-1}(idx) = 0;
	end
end



