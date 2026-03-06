function [model, accuracy, cv_accuracy, metrics] = train_logistic_model(features, labels)
% TRAIN_LOGISTIC_MODEL  Train a logistic regression (binomial GLM) classifier.
%
%   [model, accuracy, cv_accuracy, metrics] = train_logistic_model(features, labels)
%
%   Inputs:
%     features  - (N x F) numeric feature matrix
%     labels    - (N x 1) binary label vector (0 = rebound, 1 = penetration)
%
%   Outputs:
%     model       - struct containing the fitted GLM and feature scaler
%     accuracy    - training set accuracy
%     cv_accuracy - Leave-One-Out cross-validation accuracy
%     metrics     - struct: precision, recall, f1_score, confusion_matrix

fprintf('\n--- Training Logistic Regression ---\n');

% -------------------------------------------------------------------------
% Standardise features
% -------------------------------------------------------------------------
mu    = mean(features, 1);
sigma = std(features,  0, 1);
sigma(sigma < eps) = 1;
features_scaled = (features - mu) ./ sigma;

% Build a table for fitglm
n_feat  = size(features_scaled, 2);
col_names = arrayfun(@(k) sprintf('X%d', k), 1:n_feat, 'UniformOutput', false);
T = array2table(features_scaled, 'VariableNames', col_names);
T.y = labels;

% Fit logistic regression
formula  = ['y ~ ' strjoin(col_names, ' + ')];
glm_raw  = fitglm(T, formula, ...
    'Distribution', 'binomial', ...
    'Link',         'logit');

% Pack into struct
model.type      = 'Logistic';
model.glm       = glm_raw;
model.mu        = mu;
model.sigma     = sigma;
model.col_names = col_names;

% -------------------------------------------------------------------------
% Training accuracy (threshold = 0.5)
% -------------------------------------------------------------------------
prob_train = predict(glm_raw, T);
pred_train = double(prob_train >= 0.5);
accuracy   = mean(pred_train == labels);
fprintf('  Training accuracy : %.2f%%\n', accuracy * 100);

% -------------------------------------------------------------------------
% Leave-One-Out Cross-Validation
% -------------------------------------------------------------------------
n = size(features_scaled, 1);
loocv_pred = zeros(n, 1);
for i = 1:n
    idx_tr = setdiff(1:n, i);
    X_tr   = features_scaled(idx_tr, :);
    y_tr   = labels(idx_tr);
    X_te   = features_scaled(i, :);

    T_tr = array2table(X_tr, 'VariableNames', col_names);
    T_tr.y = y_tr;
    T_te = array2table(X_te, 'VariableNames', col_names);

    try
        tmp_glm   = fitglm(T_tr, formula, 'Distribution', 'binomial', 'Link', 'logit');
        prob_te   = predict(tmp_glm, T_te);
        loocv_pred(i) = double(prob_te >= 0.5);
    catch
        % If fitting fails (e.g. perfect separation on small fold), fall back
        loocv_pred(i) = mode(y_tr);
    end
end
cv_accuracy = mean(loocv_pred == labels);
fprintf('  LOOCV accuracy    : %.2f%%\n', cv_accuracy * 100);

% -------------------------------------------------------------------------
% Metrics
% -------------------------------------------------------------------------
metrics = compute_metrics(labels, pred_train);
fprintf('  Precision: %.3f  |  Recall: %.3f  |  F1: %.3f\n', ...
    metrics.precision, metrics.recall, metrics.f1_score);
end


% =========================================================================
function metrics = compute_metrics(y_true, y_pred)
TP = sum((y_pred == 1) & (y_true == 1));
TN = sum((y_pred == 0) & (y_true == 0));
FP = sum((y_pred == 1) & (y_true == 0));
FN = sum((y_pred == 0) & (y_true == 1));

precision = TP / max(TP + FP, 1);
recall    = TP / max(TP + FN, 1);
f1_score  = 2 * precision * recall / max(precision + recall, eps);

metrics.precision        = precision;
metrics.recall           = recall;
metrics.f1_score         = f1_score;
metrics.confusion_matrix = [TN, FP; FN, TP];
metrics.TP = TP; metrics.TN = TN; metrics.FP = FP; metrics.FN = FN;
end
