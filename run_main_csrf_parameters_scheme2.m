close all;
clear;
clc;

addpath('data');
addpath('utility');

% 1. The extra multiview datasets can be included for comparsion.
% 2. The different combinations of the parameters for testing.

betas = [5, 10, 20, 50, 100, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 3e4, 5e4, 1e5];
alphas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100];

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

    case 7
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

    case 8
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

end

final_result = strcat(filename, '_parameter_result_by_scheme2.txt');
final_average_result = strcat(filename, '_parameter_average_result_by_scheme2.txt');

class_labels = zeros(1, K);
for idx =  1 : K
    class_labels(idx) = length(find(gnd == idx));
end

missing_raitos = [0, 0.1, 0.2, 0.3];
% missing_raitos = [0.2];
ratio_len = length(missing_raitos);

% We can set the varaiable "repeated_times" to 10 for choosing 
%   the best result in each dataset.
repeated_times = 1; % % finding the proper parameters quickly by setting repeated_times to 1

if repeated_times > 1
    final_clustering_accs = zeros(ratio_len, repeated_times);
    final_clustering_nmis = zeros(ratio_len, repeated_times);
    final_clustering_purities = zeros(ratio_len, repeated_times);
    final_clustering_fmeasures = zeros(ratio_len, repeated_times);
    final_clustering_ris = zeros(ratio_len, repeated_times);
    final_clustering_aris = zeros(ratio_len, repeated_times);
    final_clustering_costs = zeros(ratio_len, repeated_times);
    final_clustering_iters = zeros(ratio_len, repeated_times);
    final_clustering_values = zeros(ratio_len,  repeated_times);
end
        
Mn = cell(1, nv);
for raito_idx = 1 : length(missing_raitos)    
    % prepare for incomplete multiview data: a set of the incomplete data instances 
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

    % test the different combinations of the parameters
    for kn_idx = 1 : length(betas)
        beta = betas(kn_idx);
        tic;
        [L_views, H_views, ~, max_eigen_values] = data_preprocess_scheme2(data_views, Mn, beta, K);
        time_cost1 = toc; 
        for alpha_idx = 1 : length(alphas)
            alpha = alphas(alpha_idx); 
            tic;
            [W, iter1, fobj1, iter2_set] = csrf_scheme2(L_views, H_views, max_eigen_values, alpha, beta); 
            time_cost2 = toc;
            time_cost = time_cost1 + time_cost2;
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
                disp([missing_raito, beta, alpha, acc, nmi, purity, fmeasure, ri, ari, iter1]);
        
                final_clustering_accs(raito_idx, time_idx) = acc;
                final_clustering_nmis(raito_idx, time_idx) = nmi;
                final_clustering_purities(raito_idx, time_idx) = purity;
                final_clustering_fmeasures(raito_idx, time_idx) = fmeasure;
                final_clustering_ris(raito_idx, time_idx) = ri;
                final_clustering_aris(raito_idx, time_idx) = ari;
                final_clustering_costs(raito_idx, time_idx) = time_cost;
                final_clustering_iters(raito_idx, time_idx) = iter1;
        
%                 writematrix([missing_raito, beta, alpha, roundn(acc, -2), roundn(nmi, -4), roundn(purity, -4), roundn(fmeasure, -4), roundn(ri, -4), roundn(ari, -4), roundn(time_cost, -2), iter1], final_result, "Delimiter", 'tab', 'WriteMode', 'append');       
            end

            if repeated_times > 1        
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
            
%                 writematrix([missing_raito, beta, alpha, roundn(averge_acc, -2), roundn(std_acc, -2), roundn(averge_nmi, -4), roundn(std_nmi, -4), ...
%                     roundn(averge_purity, -4), roundn(std_purity, -4), roundn(averge_fmeasure, -4), roundn(std_fmeasure, -4), roundn(averge_ri, -4), roundn(std_ri, -4),...
%                     roundn(averge_ari, -4), roundn(std_ari, -4), roundn(averge_cost, -2), roundn(averge_iter, -2)], final_average_result, "Delimiter", 'tab', 'WriteMode', 'append'); 
                disp([missing_raito, beta, alpha, averge_acc, averge_nmi, averge_fmeasure]);
            end
        end
    end

end

