function sensitivity_analysis(model, model_type, baseline_features, feat_names, results_dir)
% SENSITIVITY_ANALYSIS  Perform one-at-a-time and partial-dependence analysis.
%
%   sensitivity_analysis(model, model_type, baseline_features, feat_names, results_dir)
%
%   Inputs:
%     model             - trained model struct
%     model_type        - 'SVM', 'Logistic', or 'Ensemble'
%     baseline_features - (1 x F) baseline feature vector (e.g. dataset mean)
%     feat_names        - (1 x F) cell array of feature name strings
%     results_dir       - directory to save output figures

if nargin < 5 || isempty(results_dir)
    results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
end
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

FS = 12; LW = 2; MS = 8;
n_feat = numel(feat_names);
delta  = 0.20;   % ±20%

% =========================================================================
% One-at-a-time (OAT) sensitivity
% =========================================================================
fprintf('\n  Running one-at-a-time sensitivity analysis (±20%%)...\n');

p_base = predict_prob_sa(model, model_type, baseline_features);

sens_plus  = zeros(1, n_feat);
sens_minus = zeros(1, n_feat);

for k = 1:n_feat
    x_plus          = baseline_features;
    x_minus         = baseline_features;

    step = delta * abs(baseline_features(k));
    if step < eps, step = delta; end

    x_plus(k)  = baseline_features(k) + step;
    x_minus(k) = baseline_features(k) - step;

    p_plus  = predict_prob_sa(model, model_type, x_plus);
    p_minus = predict_prob_sa(model, model_type, x_minus);

    sens_plus(k)  = p_plus  - p_base;
    sens_minus(k) = p_minus - p_base;
end

% Rank by absolute sensitivity (average of + and −)
sensitivity = (abs(sens_plus) + abs(sens_minus)) / 2;
[sens_sorted, sort_idx] = sort(sensitivity, 'descend');
feat_sorted = feat_names(sort_idx);

fprintf('  Feature sensitivity ranking (OAT, ±20%% perturbation):\n');
fprintf('  %-30s  %10s  %10s\n', 'Feature', '+20% delta-P', '-20% delta-P');
fprintf('  %s\n', repmat('-',1,55));
for k = 1:n_feat
    fprintf('  %-30s  %+10.4f  %+10.4f\n', feat_sorted{k}, ...
        sens_plus(sort_idx(k)), sens_minus(sort_idx(k)));
end

% Plot OAT bar chart
fig1 = figure('Name','OAT Sensitivity','Visible','off');
set(fig1, 'Position', [100 100 900 600]);
barh(flip(sens_sorted), 'FaceColor', [0.3 0.6 0.9]);
set(gca, 'YTick', 1:n_feat, 'YTickLabel', flip(feat_sorted), 'FontSize', FS);
xlabel('Mean |ΔP(penetration)| for ±20% perturbation', 'FontSize', FS);
title('One-at-a-Time Feature Sensitivity', 'FontSize', FS+2);
grid on;
save_fig(fig1, results_dir, 'OAT_Sensitivity');

% =========================================================================
% Partial dependence plots for Mass, KE, Velocity
% =========================================================================
pdp_features = {'Mass_kg', 'KE_J', 'Velocity_mps'};
pdp_ranges   = {linspace(0.5, 5, 40), linspace(0, 1500, 40), linspace(5, 30, 40)};

fprintf('  Generating partial dependence plots...\n');
for j = 1:numel(pdp_features)
    feat_j   = pdp_features{j};
    feat_idx = find(strcmpi(feat_names, feat_j), 1);
    if isempty(feat_idx)
        continue;
    end

    range_j = pdp_ranges{j};
    pdp_val = zeros(1, numel(range_j));

    for r = 1:numel(range_j)
        x_r         = baseline_features;
        x_r(feat_idx) = range_j(r);
        pdp_val(r)  = predict_prob_sa(model, model_type, x_r);
    end

    fig_pdp = figure('Name', sprintf('PDP — %s', feat_j), 'Visible', 'off');
    set(fig_pdp, 'Position', [100 100 700 500]);
    plot(range_j, pdp_val, 'b-', 'LineWidth', LW, 'MarkerSize', MS);
    yline(0.5, 'k--', 'Threshold', 'FontSize', FS-2, 'LineWidth', 1);
    xlabel(feat_j, 'FontSize', FS);
    ylabel('P(Penetration)', 'FontSize', FS);
    title(sprintf('Partial Dependence: %s', feat_j), 'FontSize', FS+2);
    ylim([0 1]); grid on; set(gca, 'FontSize', FS);
    save_fig(fig_pdp, results_dir, sprintf('PDP_%s', feat_j));
end

fprintf('  Sensitivity analysis complete.\n');
end


% =========================================================================
function p = predict_prob_sa(model, model_type, x)
% PREDICT_PROB_SA  Predict P(penetration) for one feature row.
switch upper(model_type)
    case 'SVM'
        mu = model.mu; sigma = model.sigma;
        xs = (x - mu) ./ sigma;
        [~, post] = predict(model.svm, xs);
        cn  = model.svm.ClassNames;
        idx = find(cn == 1, 1);
        if isempty(idx), idx = 2; end
        p = post(idx);
    case 'LOGISTIC'
        mu = model.mu; sigma = model.sigma;
        xs = (x - mu) ./ sigma;
        T  = array2table(xs, 'VariableNames', model.col_names);
        p  = predict(model.glm, T);
    case 'ENSEMBLE'
        [~, scores] = predict(model.tb, x);
        if iscell(scores), scores = cell2mat(scores); end
        cn  = model.tb.ClassNames;
        idx = find(strcmp(cn, '1'), 1);
        if isempty(idx), idx = 2; end
        p = scores(idx);
    otherwise
        p = 0.5;
end
p = min(max(double(p), 0), 1);
end


% =========================================================================
function save_fig(fig, results_dir, name)
try
    saveas(fig, fullfile(results_dir, [name '.png']));
    savefig(fig, fullfile(results_dir, [name '.fig']));
catch ME
    warning('Could not save %s: %s', name, ME.message);
end
close(fig);
end
