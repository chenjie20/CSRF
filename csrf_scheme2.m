function [W, iter1, errors_1, iter2_set] = csrf_scheme2(L_views, H_views, max_eigen_values, alpha, beta)
% data_views{1}: each column of data_views{1} represents a sample
% Mn: missing ratio
% kn:  the number of nearest neighbors
% num_clusters:  the number of clusters
% concatenation: an extra view is generated by concatenating all views

max_iter1 = 500;
max_iter2 = 3;
iter1 = 0;
tol = 1e-3;

fobj1 = zeros(max_iter1, 1);
fobj2 = zeros(max_iter2, 1);

errors_1 = zeros(max_iter1, 1);

nv = length(L_views);
gammas = ones(nv, 1) * sqrt(1 / nv);

FHn_results = zeros(nv, 1);

iter2_set = zeros(nv, max_iter1);

graph_matrices = cell(1, nv);
for nv_idx = 1 : nv
    graph_matrices{nv_idx} = max_eigen_values(nv_idx) * eye(size(L_views{nv_idx})) - L_views{nv_idx};
end

while iter1 < max_iter1

    iter1 = iter1 + 1;

    %update F
    H_sum = zeros(size(H_views{1}));
    for nv_idx = 1 : nv
        H_sum = H_sum + gammas(nv_idx) * H_views{nv_idx};
    end
    [Ur, ~, Vr] = svd(H_sum, 'econ');
    F = Ur * Vr';

     %update Hv
    for nv_idx = 1 : nv     
        iter2 = 0;
        while iter2 < max_iter2
            iter2 = iter2 + 1;

            M = graph_matrices{nv_idx} * H_views{nv_idx} + alpha * gammas(nv_idx) * F;
            [Um, ~, Vm] = svd(M, 'econ');
            H_views{nv_idx} = Um * Vm';            
            
            fobj2(iter2) = trace(H_views{nv_idx}' * graph_matrices{nv_idx} * H_views{nv_idx} + alpha * gammas(nv_idx) * trace(F' * H_views{nv_idx}));
            if iter2 > 1
                r = (fobj2(iter2)-fobj2(iter2-1))/fobj2(iter2);
%                 disp([iter2, fobj2(iter2), r]);  
                if r < 0.1
                    break;
                end
%             else
%                 disp([iter2, fobj2(iter2)]); 
            end            
        end
        iter2_set(nv_idx, iter1) = iter2;
    end

    %update gamma
    for nv_idx = 1 : nv
        FHn_results(nv_idx) = trace(F' * H_views{nv_idx});
    end
    gammas = FHn_results  / norm(FHn_results);
    
    H_sum = zeros(size(H_views{1}));
    for nv_idx = 1 : nv
        fobj1(iter1) = fobj1(iter1) + trace(H_views{nv_idx}' * L_views{nv_idx} * H_views{nv_idx});
        H_sum = H_sum + gammas(nv_idx) * H_views{nv_idx};
    end
    fobj1(iter1) = fobj1(iter1) - alpha * trace(F' * H_sum);
    if iter1 > 2
        err = abs(fobj1(iter1) - fobj1(iter1-1)) / abs(fobj1(iter1));
        errors_1(iter1-2) = err;
%         disp([iter1, fobj1(iter1), err]);
        if err < tol
            break;
        end
    end        
%     disp('-----------------');

end

% W = constructW_PKN(F', kn);

W = constructZ_sparsity(F', beta);

