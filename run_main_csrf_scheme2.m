close all;
clear;
clc;

addpath('data');
addpath('utility');

% The results reported in the paper can be validated 
% in the following parameter settings.
%---------------------- load data------------------------------------------
data_index = 1;

switch data_index
    case 1
        filename = "MSRCv1";
        load('MSRCv1.mat');
        n = length(Y);
        nv = size(X, 2);
        K = length(unique(Y));
        gnd = Y;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
            data_views{nv_idx} = X{nv_idx}';           
        end
        data_views = normalize_multiview_data(data_views);  
        csrf_parameters = [2e4, 1.5; 1e4, 1; 1e4, 2; 1e4, 2];

    case 2
        filename = "COIL20";
        load('COIL20.mat');
        n = length(Y);
        nv = size(X, 2);
        K = length(unique(Y));
        gnd = Y;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
            data_views{nv_idx} = X{nv_idx}';            
        end
        data_views = normalize_multiview_data(data_views);
        csrf_parameters = [50, 1; 2e3, 10; 500, 1; 2000, 10];

    case 3
        filename = "handwritten";
        load('handwritten.mat');
        n = length(Y);
        nv = size(X, 2);
        K = length(unique(Y));
        gnd = Y + 1;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
             data_views{nv_idx} = X{nv_idx}';
        end
        csrf_parameters = [5e3, 0.2; 2e3, 0.1; 2e4, 0.5; 5e4, 5];

     case 4
        filename = "flower17";
        load('flower17_Kmatrix.mat');
        n = length(Y);
        nv = size(KH, 3);
        K = length(unique(Y));
        gnd = Y;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
             data_views{nv_idx} = KH(:, :, nv_idx);
        end
        csrf_parameters = [20, 5; 10, 5; 10, 0.5; 10, 0.5]; 

    case 5
        filename = "scene";
        load('scene.mat');
        n = length(Y);
        nv = size(X, 2);
        K = length(unique(Y));
        gnd = Y;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv
             data_views{nv_idx} = X{nv_idx}';
        end
        csrf_parameters = [50, 5; 50, 1; 20, 1; 50, 1];

    case 6
        filename = "Caltech101";
        load('Caltech101.mat');
        nv = size(fea, 2);
        gnd = gt;

        %We removed the background category.
        positions = find(gnd > 1);
        gnd = gnd(positions);
        K = length(unique(gnd));
        gnd = gnd - 1;
        n = length(gnd);

        data_views = cell(1, nv);
        for nv_idx = 1 : nv
            tmp = fea{nv_idx}';
            data_views{nv_idx} = tmp(:, positions);
        end
        data_views = normalize_multiview_data(data_views);
        csrf_parameters = [5e4, 5; 2e4, 5; 1e4, 5; 1e5, 5];

    case 7
        % this dataset is used for reference only.
        filename = "ORL";
        load('ORL.mat');
        n = length(Y);
        nv = size(X, 2);
        K = length(unique(Y));
        gnd = Y;
        data_views = cell(1, nv);
        for nv_idx = 1 : nv            
            data_views{nv_idx} = X{nv_idx}';
        end
        csrf_parameters = [3e4, 0.5; 3e4, 1; 1e5, 0.1; 10, 0.1];

    case 8
        % this dataset is used for reference only.
        filename = "leaves";
        load('100leaves.mat');
        n = size(truelabel{1}, 1);
        nv = size(data, 2);
        K = length(unique(truelabel{1}));
        gnd = truelabel{1};
        data_views = cell(1, nv);
        for nv_idx = 1 : nv            
            data_views{nv_idx} = data{nv_idx};
        end
        csrf_parameters = [10, 5; 10, 50; 10, 100; 10, 100];

end

final_result = strcat(filename, '_result_by_scheme2.txt');
final_average_result = strcat(filename, '_average_result_by_scheme2.txt');

class_labels = zeros(1, K);
for idx =  1 : K
    class_labels(idx) = length(find(gnd == idx));
end

missing_raitos = [0, 0.1, 0.2, 0.3];
ratio_len = length(missing_raitos);
repeated_times = 10;

final_clustering_accs = zeros(ratio_len, repeated_times);
final_clustering_nmis = zeros(ratio_len, repeated_times);
final_clustering_purities = zeros(ratio_len, repeated_times);
final_clustering_fmeasures = zeros(ratio_len, repeated_times);
final_clustering_ris = zeros(ratio_len, repeated_times);
final_clustering_aris = zeros(ratio_len, repeated_times);
final_clustering_costs = zeros(ratio_len, repeated_times);
final_clustering_iters = zeros(ratio_len, repeated_times);
final_clustering_values = zeros(ratio_len,  repeated_times);

individual_view_sparsity_ratios = zeros(ratio_len, nv);
Mn = cell(1, nv);
for raito_idx = 1 : length(missing_raitos)    
    % a set of the incomplete data instances 
    stream = RandStream.getGlobalStream;
    reset(stream);
    missing_raito = missing_raitos(raito_idx);
    raito = 1 - missing_raito;    
    rand('state', 100);
    for nv_idx = 1 : nv        
        if raito < 1
            pos = randperm(n);
            num = floor(n * raito);
            sample_pos = zeros(1, n);
            % 1 represents that the corresponding features are available.
            sample_pos(pos(1 : num)) = 1; 
            Mn{nv_idx} = sample_pos;
        else
            Mn{nv_idx} = ones(1, n);
        end
    end
    
    beta = csrf_parameters(raito_idx, 1);
    alpha = csrf_parameters(raito_idx, 2);    

    tic;
    [L_views, H_views, Z_views, max_eigen_values] = data_preprocess_scheme2(data_views, Mn, beta, K);
    [W, iter1, fobj1, iter2_set] = csrf_scheme2(L_views, H_views, max_eigen_values, alpha, beta); 
    time_cost = toc;
    for nv_idx = 1 : nv
        num_nonzeros = sum(sum(abs(Z_views{nv_idx}) > 1e-6));
        individual_view_sparsity_ratios(raito_idx, nv_idx) = num_nonzeros / (n * n);
    end
    % Tips: spectral clustering is divided into two steps since we may repeat k-means many times.
    embedding_vectors = cal_embedding_by_spectral_clustering(W, K);            
    for time_idx = 1 : repeated_times 
        labels = kmeans(embedding_vectors, K, 'maxiter', 1000, 'replicates', 20, 'emptyaction', 'singleton'); 
        acc = accuracy(gnd, labels);  
        cluster_data = cell(1, K);
        for pos_idx =  1 : K
            cluster_data(1, pos_idx) = { gnd(labels == pos_idx)' };
        end
        [nmi, purity, fmeasure, ri, ari] = calculate_results(class_labels, cluster_data);
%         disp([missing_raito, beta, alpha, acc, nmi, purity, fmeasure, ri, ari, iter1]);

        final_clustering_accs(raito_idx, time_idx) = acc;
        final_clustering_nmis(raito_idx, time_idx) = nmi;
        final_clustering_purities(raito_idx, time_idx) = purity;
        final_clustering_fmeasures(raito_idx, time_idx) = fmeasure;
        final_clustering_ris(raito_idx, time_idx) = ri;
        final_clustering_aris(raito_idx, time_idx) = ari;
        final_clustering_costs(raito_idx, time_idx) = time_cost;
        final_clustering_iters(raito_idx, time_idx) = iter1;

%         writematrix([missing_raito, beta, alpha, roundn(acc, -2), roundn(nmi, -4), roundn(purity, -4), roundn(fmeasure, -4), roundn(ri, -4), roundn(ari, -4), roundn(time_cost, -2), iter1], final_result, "Delimiter", 'tab', 'WriteMode', 'append');       
    end

    averge_acc = mean(final_clustering_accs(raito_idx, :));
    std_acc = std(final_clustering_accs(raito_idx, :));
    averge_nmi = mean(final_clustering_nmis(raito_idx, :)); 
    std_nmi =std(final_clustering_nmis(raito_idx, :));
    averge_purity = mean(final_clustering_purities(raito_idx, :));
    std_purity =std(final_clustering_purities(raito_idx, :));
    averge_fmeasure = mean(final_clustering_fmeasures(raito_idx, :));
    std_fmeasure =std(final_clustering_fmeasures(raito_idx, :));
    averge_ri =  mean(final_clustering_ris(raito_idx, :));
    std_ri =std(final_clustering_ris(raito_idx, :));
    averge_ari = mean(final_clustering_aris(raito_idx, :));
    std_ari =std(final_clustering_aris(raito_idx, :));
    averge_cost = mean(final_clustering_costs(raito_idx, :)); 
    averge_iter = mean(final_clustering_iters(raito_idx, :));

%     writematrix([missing_raito, beta, alpha, roundn(averge_acc, -2), roundn(std_acc, -2), roundn(averge_nmi, -4), roundn(std_nmi, -4), ...
%         roundn(averge_purity, -4), roundn(std_purity, -4), roundn(averge_fmeasure, -4), roundn(std_fmeasure, -4), roundn(averge_ri, -4), roundn(std_ri, -4),...
%         roundn(averge_ari, -4), roundn(std_ari, -4), roundn(averge_cost, -2), roundn(averge_iter, -2), roundn(max(individual_view_sparsity_ratios(raito_idx, :)), -2), ...
%         roundn(min(individual_view_sparsity_ratios(raito_idx, :)),-2)], final_average_result, "Delimiter", 'tab', 'WriteMode', 'append'); 

    disp([missing_raito, beta, alpha, averge_acc, averge_nmi, averge_fmeasure]);

end
