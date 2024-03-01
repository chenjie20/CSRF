function [L_views, H_views, Z_views, max_eigen_values] = data_preprocess(data_views, Mn, kn, num_clusters, concatenation)
% data_views{1}: each column of data_views{1} represents a sample

    if (nargin < 5)
        concatenation = false; 
    end

    num_sample = size(data_views{1}, 2);
    nv = size(data_views, 2);
    if concatenation
        extra_data_view = [];
        for nv_idx = 1 : nv
            extra_data_view = [diag(Mn{nv_idx}) * data_views{nv_idx}', extra_data_view];
        end
        data_views{nv+1} = extra_data_view';
        nv = nv + 1;
    end

    L_views = cell(1, nv);
    H_views = cell(1, nv);
    Z_views = cell(1, nv);
    max_eigen_values = zeros(1, nv);

    for nv_idx = 1 : nv
        Z = zeros(num_sample, num_sample);
        if nv_idx <= size(Mn, 2)
            %missing_ratio > 0
            cols = abs(Mn{nv_idx} - 1) < 1e-6;
            if  length(find(cols > 0)) < num_sample
                X = data_views{nv_idx}(:, cols);
                W = constructW_PKN(X, kn);
                Z(cols, cols) = W;
            else
                %missing_ratio = 0
                Z = constructW_PKN(data_views{nv_idx}, kn);
            end
        else
            % concatenation
            Z = constructW_PKN(data_views{nv_idx}, kn);
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
    