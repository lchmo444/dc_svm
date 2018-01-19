function [label] = knkmeans(K,init, maxiter)
% Perform kernel k-means clustering.
%   K: kernel matrix
%   init: k (1 x 1) or label (1 x n, 1<=label(i)<=k)
n = size(K,1);
if length(init) == 1
   label = ceil(init*rand(1,n));
elseif size(init,1) == 1 && size(init,2) == n
	label = init;
	else
	    error('ERROR: init is not valid.');
end
last = 0;

for i=1:maxiter
	if ~any(label ~= last)
		break
	end
	  [u,~,label] = unique(label);   % remove empty clusters
	  k = length(u);
	  E = sparse(label,1:n,1,k,n,n);
	  E = bsxfun(@times,E,1./sum(E,2));
	  T = E*K;
	  Z = repmat(diag(T*E'),1,n)-2*T;
	  last = label;
      [val, label] = min(Z,[],1);
end
[~,~,label] = unique(label);   % remove empty clusters
