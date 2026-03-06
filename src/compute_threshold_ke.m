function [threshold_KE, threshold_velocity] = compute_threshold_ke( ...
        model, mass, impact_loc, config_params, model_type, feat_order)
% COMPUTE_THRESHOLD_KE  Find the KE at which P(penetration) = 0.5.
%
%   Uses a bisection method over the range [KE_lo, KE_hi] to locate the
%   decision boundary of the trained classifier.
%
%   Inputs:
%     model         - trained model struct (from train_*_model)
%     mass          - impactor mass in kg  (scalar)
%     impact_loc    - impact location: 1 = center, 0 = corner
%     config_params - struct with glass configuration fields:
%                       .Glass1_mm, .PVB1_mm, .Glass2_mm, .PVB2_mm, .Glass3_mm
%     model_type    - string: 'SVM', 'Logistic', or 'Ensemble'
%     feat_order    - cell array of feature names defining column order of X
%                     (must match the order used during training)
%
%   Outputs:
%     threshold_KE       - kinetic energy at decision boundary (J), or NaN
%     threshold_velocity - corresponding velocity (m/s), or NaN

% -------------------------------------------------------------------------
% Bisection settings
% -------------------------------------------------------------------------
KE_lo  = 0;
KE_hi  = 2000;   % J
tol    = 0.1;    % J
max_it = 1000;

% -------------------------------------------------------------------------
% Check boundary behaviour — skip if no threshold exists in this range
% -------------------------------------------------------------------------
p_lo = predict_prob(model, model_type, ...
    build_feature_vector(KE_lo, mass, impact_loc, config_params, feat_order));
p_hi = predict_prob(model, model_type, ...
    build_feature_vector(KE_hi, mass, impact_loc, config_params, feat_order));

if p_lo >= 0.5 && p_hi >= 0.5
    % Always penetrates over this range
    threshold_KE       = NaN;
    threshold_velocity = NaN;
    fprintf('    [WARNING] Model always predicts penetration for this config. Threshold < %.1f J.\n', KE_lo);
    return;
end

if p_lo < 0.5 && p_hi < 0.5
    % Always rebounds over this range
    threshold_KE       = NaN;
    threshold_velocity = NaN;
    fprintf('    [WARNING] Model always predicts rebound for this config. Threshold > %.1f J.\n', KE_hi);
    return;
end

% -------------------------------------------------------------------------
% Bisection
% -------------------------------------------------------------------------
for iter = 1:max_it
    KE_mid = (KE_lo + KE_hi) / 2;
    p_mid  = predict_prob(model, model_type, ...
        build_feature_vector(KE_mid, mass, impact_loc, config_params, feat_order));

    if (KE_hi - KE_lo) < tol
        break;
    end

    if p_mid < 0.5
        KE_lo = KE_mid;
    else
        KE_hi = KE_mid;
    end
end

threshold_KE       = (KE_lo + KE_hi) / 2;
threshold_velocity = sqrt(2 * threshold_KE / mass);
end


% =========================================================================
function x = build_feature_vector(KE, mass, impact_loc, cfg, feat_order)
% BUILD_FEATURE_VECTOR  Assemble one feature row given scalar KE and mass.

velocity              = sqrt(2 * KE / max(mass, eps));
TotalGlassThickness   = cfg.Glass1_mm + cfg.Glass2_mm + cfg.Glass3_mm;
TotalPVBThickness     = cfg.PVB1_mm   + cfg.PVB2_mm;
TotalPanelThickness   = TotalGlassThickness + TotalPVBThickness;
NumLayers             = double(cfg.Glass1_mm > 0) + ...
                        double(cfg.Glass2_mm > 0) + ...
                        double(cfg.Glass3_mm > 0);
ConfigType            = 2 - double(cfg.Glass3_mm > 0);  % 1 or 2

% Map feature names to values
feat_map = containers.Map( ...
    {'Mass_kg','KE_J','Velocity_mps', ...
     'Glass1_mm','PVB1_mm','Glass2_mm','PVB2_mm','Glass3_mm', ...
     'TotalGlassThickness','TotalPVBThickness','TotalPanelThickness', ...
     'NumLayers','ConfigType','ImpactLocation'}, ...
    {mass, KE, velocity, ...
     cfg.Glass1_mm, cfg.PVB1_mm, cfg.Glass2_mm, cfg.PVB2_mm, cfg.Glass3_mm, ...
     TotalGlassThickness, TotalPVBThickness, TotalPanelThickness, ...
     NumLayers, ConfigType, double(impact_loc)});

n_feat = numel(feat_order);
x = zeros(1, n_feat);
for k = 1:n_feat
    if isKey(feat_map, feat_order{k})
        x(k) = feat_map(feat_order{k});
    end
end
end


% =========================================================================
function p = predict_prob(model, model_type, x)
% PREDICT_PROB  Return P(penetration=1) for feature row x.

switch upper(model_type)

    case 'SVM'
        mu    = model.mu;
        sigma = model.sigma;
        xs    = (x - mu) ./ sigma;
        [~, post] = predict(model.svm, xs);
        % post columns correspond to ClassNames order
        class_names = model.svm.ClassNames;
        idx1 = find(class_names == 1, 1);
        if isempty(idx1), idx1 = 2; end
        p = post(idx1);

    case 'LOGISTIC'
        mu        = model.mu;
        sigma     = model.sigma;
        xs        = (x - mu) ./ sigma;
        col_names = model.col_names;
        T         = array2table(xs, 'VariableNames', col_names);
        p         = predict(model.glm, T);

    case 'ENSEMBLE'
        [~, scores] = predict(model.tb, x);
        % TreeBagger returns scores as cell of probabilities per class
        % ClassNames are sorted: {'0','1'}
        if iscell(scores)
            scores = cell2mat(scores);
        end
        class_names = model.tb.ClassNames;
        idx1 = find(strcmp(class_names, '1'), 1);
        if isempty(idx1), idx1 = 2; end
        p = scores(idx1);

    otherwise
        error('compute_threshold_ke: unknown model_type ''%s''.', model_type);
end

% Clamp to [0,1]
p = min(max(p, 0), 1);
end
