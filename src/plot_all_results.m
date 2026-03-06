function plot_all_results(data, models, model_names, threshold_results, results_dir)
% PLOT_ALL_RESULTS  Generate all publication-quality plots.
%
%   plot_all_results(data, models, model_names, threshold_results, results_dir)
%
%   Inputs:
%     data             - full dataset table (from load_dataset)
%     models           - cell array of trained model structs
%     model_names      - cell array of model name strings
%     threshold_results - table with columns:
%                          Config, Impact_Location, Mass_kg,
%                          Threshold_KE_J, Threshold_Velocity_mps
%     results_dir      - path to output directory for saving figures

if nargin < 5 || isempty(results_dir)
    results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
end
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

% Common plot settings
FS  = 12;   % FontSize
LW  = 2;    % LineWidth
MS  = 8;    % MarkerSize

config_names = {'5-layer (3-3-3)', '3-layer (6-6)'};
loc_names    = {'center', 'corner'};
masses       = [1, 2, 3, 4];

% =========================================================================
% 1. KE vs Penetration Status scatter (per config × location)
% =========================================================================
fprintf('  Generating: KE vs Penetration Status scatter plots...\n');
fig1 = figure('Name','KE vs Penetration Status','Visible','off');
set(fig1, 'Position', [100 100 1200 800]);
sp = 0;
for cfg = 1:2
    for loc = 0:1
        sp = sp + 1;
        subplot(2, 2, sp);
        idx = (data.ConfigType == cfg) & (data.ImpactLocation == loc);
        d   = data(idx, :);

        reb = d(d.Penetration_Status == 0, :);
        pen = d(d.Penetration_Status == 1, :);

        scatter(reb.Mass_kg, reb.KE_J, MS^2, 'b', 'filled', ...
            'DisplayName', 'Rebound'); hold on;
        scatter(pen.Mass_kg, pen.KE_J, MS^2, 'r', 'filled', ...
            'DisplayName', 'Penetration');

        % Overlay threshold line if available
        if ~isempty(threshold_results)
            loc_str = loc_names{loc+1};
            cfg_val = cfg;
            thr_sub = threshold_results( ...
                (threshold_results.ConfigType == cfg_val) & ...
                strcmp(threshold_results.Impact_Location, loc_str), :);
            if ~isempty(thr_sub) && any(~isnan(thr_sub.Threshold_KE_J))
                valid = thr_sub(~isnan(thr_sub.Threshold_KE_J), :);
                plot(valid.Mass_kg, valid.Threshold_KE_J, 'k--', ...
                    'LineWidth', LW, 'DisplayName', 'Threshold KE');
            end
        end

        xlabel('Mass (kg)', 'FontSize', FS);
        ylabel('Kinetic Energy (J)', 'FontSize', FS);
        title(sprintf('Config %d — %s', cfg, loc_names{loc+1}), 'FontSize', FS);
        legend('Location', 'northwest', 'FontSize', FS-2);
        grid on; set(gca, 'FontSize', FS);
    end
end
sgtitle('KE vs Penetration Status by Configuration and Impact Location', 'FontSize', FS+2);
save_figure(fig1, results_dir, 'KE_vs_PenetrationStatus');

% =========================================================================
% 2. Threshold KE vs Mass
% =========================================================================
fprintf('  Generating: Threshold KE vs Mass curves...\n');
if ~isempty(threshold_results)
    fig2 = figure('Name','Threshold KE vs Mass','Visible','off');
    set(fig2, 'Position', [100 100 900 600]);
    colors = {'b','r'};
    lstyle = {'-o', '--s'};
    hold on;
    lgd_entries = {};
    for cfg = 1:2
        for loc = 0:1
            loc_str = loc_names{loc+1};
            thr_sub = threshold_results( ...
                (threshold_results.ConfigType == cfg) & ...
                strcmp(threshold_results.Impact_Location, loc_str), :);
            if ~isempty(thr_sub)
                thr_sub = sortrows(thr_sub, 'Mass_kg');
                plot(thr_sub.Mass_kg, thr_sub.Threshold_KE_J, ...
                    lstyle{cfg}, 'Color', colors{loc+1}, ...
                    'LineWidth', LW, 'MarkerSize', MS);
                lgd_entries{end+1} = sprintf('%s — %s', config_names{cfg}, loc_str); %#ok<AGROW>
            end
        end
    end
    xlabel('Impactor Mass (kg)', 'FontSize', FS);
    ylabel('Threshold Kinetic Energy (J)', 'FontSize', FS);
    title('Threshold KE vs Impactor Mass', 'FontSize', FS+2);
    legend(lgd_entries, 'Location', 'northwest', 'FontSize', FS-2);
    grid on; set(gca, 'FontSize', FS);
    save_figure(fig2, results_dir, 'Threshold_KE_vs_Mass');

    % =====================================================================
    % 3. Threshold Velocity vs Mass
    % =====================================================================
    fprintf('  Generating: Threshold Velocity vs Mass curves...\n');
    fig3 = figure('Name','Threshold Velocity vs Mass','Visible','off');
    set(fig3, 'Position', [100 100 900 600]);
    hold on;
    lgd_entries = {};
    for cfg = 1:2
        for loc = 0:1
            loc_str = loc_names{loc+1};
            thr_sub = threshold_results( ...
                (threshold_results.ConfigType == cfg) & ...
                strcmp(threshold_results.Impact_Location, loc_str), :);
            if ~isempty(thr_sub)
                thr_sub = sortrows(thr_sub, 'Mass_kg');
                plot(thr_sub.Mass_kg, thr_sub.Threshold_Velocity_mps, ...
                    lstyle{cfg}, 'Color', colors{loc+1}, ...
                    'LineWidth', LW, 'MarkerSize', MS);
                lgd_entries{end+1} = sprintf('%s — %s', config_names{cfg}, loc_str); %#ok<AGROW>
            end
        end
    end
    xlabel('Impactor Mass (kg)', 'FontSize', FS);
    ylabel('Threshold Velocity (m/s)', 'FontSize', FS);
    title('Threshold Velocity vs Impactor Mass', 'FontSize', FS+2);
    legend(lgd_entries, 'Location', 'northeast', 'FontSize', FS-2);
    grid on; set(gca, 'FontSize', FS);
    save_figure(fig3, results_dir, 'Threshold_Velocity_vs_Mass');
end

% =========================================================================
% 4. Confusion matrix heatmap for each model
% =========================================================================
fprintf('  Generating: Confusion matrix heatmaps...\n');
for m = 1:numel(models)
    if ~isfield(models{m}, 'metrics'), continue; end
    CM   = models{m}.metrics.confusion_matrix;
    fig4 = figure('Name', sprintf('Confusion Matrix — %s', model_names{m}), 'Visible','off');
    set(fig4, 'Position', [100 100 500 450]);
    imagesc(CM);
    colormap(parula);
    colorbar;
    set(gca, 'XTick', 1:2, 'XTickLabel', {'Rebound','Penetration'}, ...
             'YTick', 1:2, 'YTickLabel', {'Rebound','Penetration'}, ...
             'FontSize', FS);
    xlabel('Predicted', 'FontSize', FS);
    ylabel('Actual',    'FontSize', FS);
    title(sprintf('Confusion Matrix — %s', model_names{m}), 'FontSize', FS+2);
    % Annotate cells
    for r = 1:2
        for c = 1:2
            text(c, r, num2str(CM(r,c)), ...
                'HorizontalAlignment','center', 'FontSize', FS+2, ...
                'FontWeight','bold', 'Color','w');
        end
    end
    save_figure(fig4, results_dir, sprintf('ConfusionMatrix_%s', model_names{m}));
end

% =========================================================================
% 5. Decision boundary in (Mass, KE) space — per config
% =========================================================================
fprintf('  Generating: Decision boundary plots...\n');
if ~isempty(models)
    best_model = models{1};   % use first (best) model
    best_type  = model_names{1};
    feat_order = best_model.feat_order;
    for cfg = 1:2
        if cfg == 1
            cfg_p = struct('Glass1_mm',3,'PVB1_mm',1.52,'Glass2_mm',3,'PVB2_mm',1.52,'Glass3_mm',3);
        else
            cfg_p = struct('Glass1_mm',6,'PVB1_mm',1.52,'Glass2_mm',6,'PVB2_mm',0,'Glass3_mm',0);
        end

        mass_vec = linspace(0.5, 5, 50);
        ke_vec   = linspace(0, 1500, 50);
        [MM, KK] = meshgrid(mass_vec, ke_vec);
        PP       = zeros(size(MM));

        for r = 1:numel(mass_vec)
            for c = 1:numel(ke_vec)
                x = build_fv(ke_vec(c), mass_vec(r), 1, cfg_p, feat_order);
                PP(c, r) = predict_prob_plot(best_model, best_type, x);
            end
        end

        fig5 = figure('Name', sprintf('Decision Boundary — Config %d', cfg), 'Visible','off');
        set(fig5, 'Position', [100 100 900 650]);
        contourf(MM, KK, PP, [0.5 0.5], 'LineWidth', LW); hold on;
        colormap([0.8 0.9 1; 1 0.8 0.8]);

        idx_d = data.ConfigType == cfg;
        d_cfg = data(idx_d, :);
        reb = d_cfg(d_cfg.Penetration_Status == 0, :);
        pen = d_cfg(d_cfg.Penetration_Status == 1, :);
        scatter(reb.Mass_kg, reb.KE_J, MS^2, 'b', 'filled', 'DisplayName','Rebound');
        scatter(pen.Mass_kg, pen.KE_J, MS^2, 'r', 'filled', 'DisplayName','Penetration');

        xlabel('Mass (kg)', 'FontSize', FS);
        ylabel('Kinetic Energy (J)', 'FontSize', FS);
        title(sprintf('Decision Boundary — Config %d (%s)', cfg, config_names{cfg}), 'FontSize', FS+2);
        legend('Location','northwest','FontSize',FS-2);
        grid on; set(gca,'FontSize',FS);
        save_figure(fig5, results_dir, sprintf('DecisionBoundary_Config%d', cfg));
    end
end

% =========================================================================
% 6. ROC curve for each model
% =========================================================================
fprintf('  Generating: ROC curves...\n');
fig6 = figure('Name','ROC Curves','Visible','off');
set(fig6, 'Position', [100 100 700 600]);
hold on;
roc_colors = {'b','r','g'};
for m = 1:numel(models)
    if ~isfield(models{m}, 'roc_scores'), continue; end
    scores = models{m}.roc_scores;
    labels = models{m}.roc_labels;
    [X_roc, Y_roc, ~, AUC] = perfcurve(labels, scores, 1);
    plot(X_roc, Y_roc, roc_colors{mod(m-1,3)+1}, 'LineWidth', LW, ...
        'DisplayName', sprintf('%s (AUC=%.3f)', model_names{m}, AUC));
end
plot([0 1],[0 1],'k--','LineWidth',1,'HandleVisibility','off');
xlabel('False Positive Rate','FontSize',FS);
ylabel('True Positive Rate','FontSize',FS);
title('ROC Curves','FontSize',FS+2);
legend('Location','southeast','FontSize',FS-2);
grid on; set(gca,'FontSize',FS);
save_figure(fig6, results_dir, 'ROC_Curves');

% =========================================================================
% 7. 3D surface: Mass × KE × P(penetration) per config
% =========================================================================
fprintf('  Generating: 3D probability surface plots...\n');
if ~isempty(models)
    best_model = models{1};
    best_type  = model_names{1};
    feat_order = best_model.feat_order;
    for cfg = 1:2
        if cfg == 1
            cfg_p = struct('Glass1_mm',3,'PVB1_mm',1.52,'Glass2_mm',3,'PVB2_mm',1.52,'Glass3_mm',3);
        else
            cfg_p = struct('Glass1_mm',6,'PVB1_mm',1.52,'Glass2_mm',6,'PVB2_mm',0,'Glass3_mm',0);
        end

        mass_vec = linspace(0.5, 5, 30);
        ke_vec   = linspace(0, 1500, 30);
        [MM, KK] = meshgrid(mass_vec, ke_vec);
        PP       = zeros(size(MM));
        for r = 1:numel(mass_vec)
            for c = 1:numel(ke_vec)
                x = build_fv(ke_vec(c), mass_vec(r), 1, cfg_p, feat_order);
                PP(c,r) = predict_prob_plot(best_model, best_type, x);
            end
        end

        fig7 = figure('Name', sprintf('3D Surface — Config %d', cfg), 'Visible','off');
        set(fig7, 'Position', [100 100 900 700]);
        surf(MM, KK, PP, 'EdgeAlpha', 0.2);
        colorbar; colormap(jet);
        xlabel('Mass (kg)',     'FontSize', FS);
        ylabel('KE (J)',        'FontSize', FS);
        zlabel('P(Penetration)','FontSize', FS);
        title(sprintf('P(Penetration) Surface — Config %d (%s)', cfg, config_names{cfg}), ...
            'FontSize', FS+2);
        grid on; set(gca,'FontSize',FS);
        save_figure(fig7, results_dir, sprintf('3DSurface_Config%d', cfg));
    end
end

% =========================================================================
% 8. Bar chart comparing model performance
% =========================================================================
fprintf('  Generating: Model comparison bar chart...\n');
if ~isempty(models)
    n_models = numel(models);
    train_acc = zeros(1, n_models);
    cv_acc    = zeros(1, n_models);
    f1        = zeros(1, n_models);
    for m = 1:n_models
        train_acc(m) = models{m}.train_accuracy  * 100;
        cv_acc(m)    = models{m}.cv_accuracy     * 100;
        f1(m)        = models{m}.metrics.f1_score * 100;
    end

    fig8 = figure('Name','Model Comparison','Visible','off');
    set(fig8, 'Position', [100 100 800 550]);
    bar_data = [train_acc; cv_acc; f1]';
    b = bar(bar_data);
    b(1).FaceColor = [0.2 0.4 0.8];
    b(2).FaceColor = [0.8 0.3 0.2];
    b(3).FaceColor = [0.2 0.7 0.3];
    set(gca, 'XTickLabel', model_names, 'FontSize', FS);
    ylabel('Score (%)', 'FontSize', FS);
    title('Model Performance Comparison', 'FontSize', FS+2);
    legend({'Training Accuracy','CV Accuracy','F1 Score'}, ...
        'Location','southeast','FontSize',FS-2);
    ylim([0 110]); grid on;
    save_figure(fig8, results_dir, 'Model_Comparison');
end

% =========================================================================
% 9. Feature importance bar chart (ensemble model)
% =========================================================================
fprintf('  Generating: Feature importance bar chart...\n');
ens_idx = find(strcmpi(model_names, 'Ensemble'), 1);
if ~isempty(ens_idx) && isfield(models{ens_idx}, 'feat_order')
    fi     = models{ens_idx}.feat_importance;
    fo     = models{ens_idx}.feat_order;
    [fi_s, fi_ord] = sort(fi, 'descend');

    fig9 = figure('Name','Feature Importance','Visible','off');
    set(fig9, 'Position', [100 100 900 600]);
    barh(fi_s, 'FaceColor', [0.2 0.6 0.8]);
    set(gca, 'YTick', 1:numel(fo), 'YTickLabel', fo(fi_ord), 'FontSize', FS);
    xlabel('OOB Importance Score', 'FontSize', FS);
    title('Feature Importance (Random Forest)', 'FontSize', FS+2);
    grid on;
    save_figure(fig9, results_dir, 'Feature_Importance');
end

fprintf('  All plots saved to: %s\n', results_dir);
end


% =========================================================================
% Helper: save figure as .png and .fig
% =========================================================================
function save_figure(fig, results_dir, name)
png_path = fullfile(results_dir, [name '.png']);
fig_path = fullfile(results_dir, [name '.fig']);
try
    saveas(fig, png_path);
    savefig(fig, fig_path);
catch ME
    warning('Could not save figure %s: %s', name, ME.message);
end
close(fig);
end


% =========================================================================
% Helper: build feature vector (duplicated here for self-containment)
% =========================================================================
function x = build_fv(KE, mass, impact_loc, cfg, feat_order)
velocity             = sqrt(2 * KE / max(mass, eps));
TotalGlassThickness  = cfg.Glass1_mm + cfg.Glass2_mm + cfg.Glass3_mm;
TotalPVBThickness    = cfg.PVB1_mm   + cfg.PVB2_mm;
TotalPanelThickness  = TotalGlassThickness + TotalPVBThickness;
NumLayers            = double(cfg.Glass1_mm > 0) + ...
                       double(cfg.Glass2_mm > 0) + ...
                       double(cfg.Glass3_mm > 0);
ConfigType           = 2 - double(cfg.Glass3_mm > 0);

feat_map = containers.Map( ...
    {'Mass_kg','KE_J','Velocity_mps', ...
     'Glass1_mm','PVB1_mm','Glass2_mm','PVB2_mm','Glass3_mm', ...
     'TotalGlassThickness','TotalPVBThickness','TotalPanelThickness', ...
     'NumLayers','ConfigType','ImpactLocation'}, ...
    {mass, KE, velocity, ...
     cfg.Glass1_mm, cfg.PVB1_mm, cfg.Glass2_mm, cfg.PVB2_mm, cfg.Glass3_mm, ...
     TotalGlassThickness, TotalPVBThickness, TotalPanelThickness, ...
     NumLayers, ConfigType, double(impact_loc)});

x = zeros(1, numel(feat_order));
for k = 1:numel(feat_order)
    if isKey(feat_map, feat_order{k})
        x(k) = feat_map(feat_order{k});
    end
end
end


% =========================================================================
% Helper: predict probability (mirrors compute_threshold_ke logic)
% =========================================================================
function p = predict_prob_plot(model, model_type, x)
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
p = min(max(p, 0), 1);
end
