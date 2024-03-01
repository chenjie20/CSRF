function [ V ] = cal_embedding_by_spectral_clustering(W, k)

D = diag(1./sqrt(sum(W, 2)+ eps));
W = D * W * D;
[U, ~, ~] = svd(W);
V = U(:, 1 : k);
V = normr(V);

end
