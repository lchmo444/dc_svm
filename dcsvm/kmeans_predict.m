function [idx] = kmeans_predict(X, centers)

if size(centers,2) < size(X,2)
	centers = [centers zeros(size(centers,1),size(X,2)-size(centers,2))];
end
if size(X,2) < size(centers,2)
	X = [X zeros(size(X,1),size(centers,2)-size(X,2))];
end

k = size(centers,1);
dd = sum(X.*X,2)*ones(1,k)+ones(size(X,1),1)*(sum(centers.*centers,2))'-2*X*centers';
[val idx1] = min(dd');
idx = idx1';
