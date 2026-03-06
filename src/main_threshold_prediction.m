%% MAIN_THRESHOLD_PREDICTION  Master script for laminated glass impact analysis
%
%  Implements a data-driven predictive framework that:
%    1. Loads a hardcoded LS-DYNA simulation dataset
%    2. Trains SVM, Logistic Regression, and Random Forest classifiers
%    3. Determines the rebound–penetration threshold KE for each
%       configuration / mass / impact-location combination
%    4. Generates publication-quality plots, sensitivity analysis, and
%       exports all results to Excel
%
%  Requires: Statistics and Machine Learning Toolbox (MATLAB R2019b+)
%
%  Usage:
%    >> main_threshold_prediction
%
% =========================================================================

%% 0. Initialisation
% -------------------------------------------------------------------------
clear; clc; close all;

% Add src folder to path (script is expected to live in src/)
src_dir     = fileparts(mfilename('fullpath'));
results_dir = fullfile(src_dir, '..', 'results');
addpath(src_dir);

if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

fprintf('=======================================================\n');
fprintf('  LAMINATED GLASS IMPACT — THRESHOLD PREDICTION\n');
fprintf('  Started: %s\n', datestr(now));
fprintf('=======================================================\n\n');

% =========================================================================
%% 1. Load dataset
% =========================================================================
fprintf('STEP 1: Loading dataset...\n');

% Optionally supply path to the Excel file; falls back to hardcoded data
excel_path = fullfile(src_dir, '..', 'Combined_DATA_SET_TO_BE_USED.xlsx');
data = load_dataset(excel_path);

fprintf('  Columns: %s\n', strjoin(data.Properties.VariableNames, ', '));
fprintf('  Rows   : %d\n\n', height(data));

% =========================================================================
%% 2. Feature matrix and label vector
% =========================================================================
fprintf('STEP 2: Building feature matrix...\n');

feature_names = {'Mass_kg', 'KE_J', 'Velocity_mps', ...
                 'Glass1_mm', 'PVB1_mm', 'Glass2_mm', 'PVB2_mm', 'Glass3_mm', ...
                 'TotalGlassThickness', 'TotalPVBThickness', ...
                 'ImpactLocation'};

X = zeros(height(data), numel(feature_names));
for k = 1:numel(feature_names)
    X(:, k) = data.(feature_names{k});
end
y = data.Penetration_Status;

fprintf('  Feature matrix size: %d × %d\n', size(X,1), size(X,2));
fprintf('  Labels — Rebound: %d | Penetration: %d\n\n', sum(y==0), sum(y==1));

% =========================================================================
%% 3. Train all three classifiers
% =========================================================================
fprintf('STEP 3: Training classifiers...\n');

[svm_model, svm_train_acc, svm_cv_acc, svm_metrics] = train_svm_model(X, y);
[log_model, log_train_acc, log_cv_acc, log_metrics] = train_logistic_model(X, y);
[ens_model, ens_train_acc, ens_cv_acc, ens_metrics] = train_ensemble_model(X, y);

% Attach metadata used by plot_all_results / sensitivity_analysis
svm_model.feat_order    = feature_names;
svm_model.train_accuracy = svm_train_acc;
svm_model.cv_accuracy    = svm_cv_acc;
svm_model.metrics        = svm_metrics;

log_model.feat_order    = feature_names;
log_model.train_accuracy = log_train_acc;
log_model.cv_accuracy    = log_cv_acc;
log_model.metrics        = log_metrics;

ens_model.feat_order    = feature_names;
ens_model.train_accuracy = ens_train_acc;
ens_model.cv_accuracy    = ens_cv_acc;
ens_model.metrics        = ens_metrics;

% Compute probability scores for ROC curves
svm_scores = get_probs(svm_model, 'SVM', X);
log_scores = get_probs(log_model, 'Logistic', X);
ens_scores = get_probs(ens_model, 'Ensemble', X);

svm_model.roc_scores = svm_scores;  svm_model.roc_labels = y;
log_model.roc_scores = log_scores;  log_model.roc_labels = y;
ens_model.roc_scores = ens_scores;  ens_model.roc_labels = y;

% Attach ensemble feature importance
ens_model.feat_importance = ens_metrics.feature_importance;

% Collect in cell arrays
all_models      = {svm_model, log_model, ens_model};
all_model_names = {'SVM', 'Logistic', 'Ensemble'};

% =========================================================================
%% 4. Model comparison table
% =========================================================================
fprintf('\nSTEP 4: Model comparison\n');

fprintf('\n%-12s | %-12s | %-12s | %-10s | %-8s | %-8s\n', ...
    'Model','Train Acc','CV Acc','Precision','Recall','F1');
fprintf('%s\n', repmat('-', 1, 72));
accs   = [svm_train_acc, log_train_acc, ens_train_acc];
cv_acc = [svm_cv_acc,    log_cv_acc,    ens_cv_acc];
prec   = [svm_metrics.precision, log_metrics.precision, ens_metrics.precision];
rec    = [svm_metrics.recall,    log_metrics.recall,    ens_metrics.recall];
f1s    = [svm_metrics.f1_score,  log_metrics.f1_score,  ens_metrics.f1_score];
for m = 1:3
    fprintf('%-12s | %10.2f%% | %10.2f%% | %10.3f | %8.3f | %8.3f\n', ...
        all_model_names{m}, accs(m)*100, cv_acc(m)*100, prec(m), rec(m), f1s(m));
end

% Best model (by CV accuracy)
[~, best_idx]   = max(cv_acc);
best_model      = all_models{best_idx};
best_model_name = all_model_names{best_idx};
fprintf('\n  Best model (highest CV accuracy): %s (%.2f%%)\n\n', ...
    best_model_name, cv_acc(best_idx)*100);

% =========================================================================
%% 5. Compute threshold KE for every combination
% =========================================================================
fprintf('STEP 5: Computing threshold kinetic energies...\n');

configs = struct( ...
    'name',   {'5-layer','3-layer'}, ...
    'type',   {1, 2}, ...
    'params', { struct('Glass1_mm',3,'PVB1_mm',1.52,'Glass2_mm',3,'PVB2_mm',1.52,'Glass3_mm',3), ...
                struct('Glass1_mm',6,'PVB1_mm',1.52,'Glass2_mm',6,'PVB2_mm',0,'Glass3_mm',0) });

loc_labels = {'center', 'corner'};
loc_values = [1, 0];
masses     = [1, 2, 3, 4];

% Pre-allocate result storage
n_rows = numel(configs) * numel(loc_labels) * numel(masses);
Config_col          = cell(n_rows, 1);
Impact_Location_col = cell(n_rows, 1);
Mass_col            = zeros(n_rows, 1);
ThrKE_col           = zeros(n_rows, 1);
ThrV_col            = zeros(n_rows, 1);
ConfigType_col      = zeros(n_rows, 1);

row = 0;
for c = 1:numel(configs)
    for l = 1:numel(loc_labels)
        for m = 1:numel(masses)
            row = row + 1;
            fprintf('  Computing: Config=%s  Location=%-6s  Mass=%d kg ... ', ...
                configs(c).name, loc_labels{l}, masses(m));

            [thr_ke, thr_v] = compute_threshold_ke( ...
                best_model, masses(m), loc_values(l), ...
                configs(c).params, best_model_name, feature_names);

            if isnan(thr_ke)
                fprintf('No threshold found in [0, 2000] J\n');
            else
                fprintf('KE = %.1f J  |  V = %.2f m/s\n', thr_ke, thr_v);
            end

            Config_col{row}          = configs(c).name;
            Impact_Location_col{row} = loc_labels{l};
            Mass_col(row)            = masses(m);
            ThrKE_col(row)           = thr_ke;
            ThrV_col(row)            = thr_v;
            ConfigType_col(row)      = configs(c).type;
        end
    end
end

threshold_results = table(Config_col, Impact_Location_col, Mass_col, ...
                          ThrKE_col, ThrV_col, ConfigType_col, ...
                          'VariableNames', {'Config','Impact_Location','Mass_kg', ...
                                            'Threshold_KE_J','Threshold_Velocity_mps', ...
                                            'ConfigType'});

% =========================================================================
%% 6. Print formatted threshold table
% =========================================================================
fprintf('\n');
fprintf('=====================================================\n');
fprintf('  THRESHOLD KINETIC ENERGY RESULTS\n');
fprintf('=====================================================\n');
fprintf('%-10s | %-8s | %-8s | %-16s | %-16s\n', ...
    'Config', 'Impact', 'Mass(kg)', 'Threshold KE(J)', 'Threshold V(m/s)');
fprintf('%s\n', repmat('-', 1, 70));
for r = 1:height(threshold_results)
    ke_str = '       NaN';
    v_str  = '     NaN';
    if ~isnan(threshold_results.Threshold_KE_J(r))
        ke_str = sprintf('%14.1f', threshold_results.Threshold_KE_J(r));
        v_str  = sprintf('%14.2f', threshold_results.Threshold_Velocity_mps(r));
    end
    fprintf('%-10s | %-8s | %8d | %s | %s\n', ...
        threshold_results.Config{r}, ...
        threshold_results.Impact_Location{r}, ...
        threshold_results.Mass_kg(r), ...
        ke_str, v_str);
end
fprintf('=====================================================\n\n');

% =========================================================================
%% 7. Generate all plots
% =========================================================================
fprintf('STEP 6: Generating plots...\n');
plot_all_results(data, all_models, all_model_names, threshold_results, results_dir);

% =========================================================================
%% 8. Sensitivity analysis
% =========================================================================
fprintf('\nSTEP 7: Sensitivity analysis...\n');
baseline_x = mean(X, 1);
sensitivity_analysis(best_model, best_model_name, baseline_x, feature_names, results_dir);

% =========================================================================
%% 9. Export to Excel
% =========================================================================
fprintf('\nSTEP 8: Exporting to Excel...\n');

% --- Model comparison table
model_comparison = table( ...
    all_model_names', accs'*100, cv_acc'*100, prec', rec', f1s', ...
    'VariableNames', {'Model_Name','Training_Accuracy','CV_Accuracy', ...
                      'Precision','Recall','F1_Score'});

% --- Full predictions table
pred_svm = double(svm_scores >= 0.5);
pred_log = double(log_scores >= 0.5);
pred_ens = double(ens_scores >= 0.5);

full_predictions = data;
full_predictions.Predicted_SVM      = pred_svm;
full_predictions.Prob_SVM           = svm_scores;
full_predictions.Predicted_Logistic = pred_log;
full_predictions.Prob_Logistic      = log_scores;
full_predictions.Predicted_Ensemble = pred_ens;
full_predictions.Prob_Ensemble      = ens_scores;

% --- Feature importance table (from ensemble)
fi_vals  = ens_model.feat_importance;
feat_imp_table = table(feature_names', fi_vals', ...
    'VariableNames', {'Feature_Name', 'Importance_Score'});

export_results_to_excel(threshold_results, model_comparison, ...
                         full_predictions, feat_imp_table, results_dir);

% =========================================================================
%% Done
% =========================================================================
fprintf('\n=======================================================\n');
fprintf('  Analysis Complete — %s\n', datestr(now));
fprintf('  Results saved to: %s\n', results_dir);
fprintf('=======================================================\n');


% =========================================================================
%% Local helper: get probability scores for all samples
% =========================================================================
function scores = get_probs(model, model_type, X)
n = size(X, 1);
scores = zeros(n, 1);
for i = 1:n
    x = X(i, :);
    switch upper(model_type)
        case 'SVM'
            mu = model.mu; sigma = model.sigma;
            xs = (x - mu) ./ sigma;
            [~, post] = predict(model.svm, xs);
            cn  = model.svm.ClassNames;
            idx = find(cn == 1, 1);
            if isempty(idx), idx = 2; end
            scores(i) = post(idx);
        case 'LOGISTIC'
            mu = model.mu; sigma = model.sigma;
            xs = (x - mu) ./ sigma;
            T  = array2table(xs, 'VariableNames', model.col_names);
            scores(i) = predict(model.glm, T);
        case 'ENSEMBLE'
            [~, sc] = predict(model.tb, x);
            if iscell(sc), sc = cell2mat(sc); end
            cn  = model.tb.ClassNames;
            idx = find(strcmp(cn,'1'), 1);
            if isempty(idx), idx = 2; end
            scores(i) = sc(idx);
    end
end
scores = min(max(scores, 0), 1);
end
