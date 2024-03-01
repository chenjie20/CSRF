function [L_views, H_views, Z_views, max_eigen_values] = data_preprocess_scheme2(data_views, Mn, beta, num_clusters)
% data_views{1}: each column of data_views{1} represents a sample

    num_sample = size(data_views{1}, 2);
    nv = size(data_views, 2);

    L_views = cell(1, nv);
    H_views = cell(1, nv);
    Z_views = cell(1, nv);
    max_eigen_values = zeros(1, nv);

    for nv_idx = 1 : nv
        Z = zeros(num_sample, num_sample);
        %missing_ratio > 0
        cols = abs(Mn{nv_idx} - 1) < 1e-6;
        if  length(find(cols > 0)) < num_sample
            X = data_views{nv_idx}(:, cols);
            W = constructZ_sparsity(X, beta);
            Z(cols, cols) = W;
        else
            %missing_ratio = 0
            Z = constructZ_sparsity(data_views{nv_idx}, beta);
        end
        Z_views{nv_idx} = Z;

        D = diag(1./sqrt(sum(Z, 2)+ eps));  
        W = D * Z * D;
        L_views{nv_idx} = eye(num_sample) - W;

        [U, ~, ~] = svd(W);
        V = U(:, 1 : num_clusters);       
        VV = normr(V);
        H_views{nv_idx} = VV;
        [~, eigen_values, ~] = eig1(L_views{nv_idx}, 1);
        max_eigen_values(nv_idx) = eigen_values(1); 

%         [v, d] = eig(L_views{nv_idx});
%         d = diag(d);
%         [d1, idx] = sort(d);
%         idx1 = idx(1:num_clusters);       
%         max_eigen_values(nv_idx) = d1(end);
%         V = v(:,idx1);
%         VV = normr(real(V));
%         H_views{nv_idx} = VV;

    end
    