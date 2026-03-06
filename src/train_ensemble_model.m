function [model, accuracy, cv_accuracy, metrics] = train_ensemble_model(features, labels)
% TRAIN_ENSEMBLE_MODEL  Train a Random Forest ensemble classifier.
%
%   [model, accuracy, cv_accuracy, metrics] = train_ensemble_model(features, labels)
%
%   Inputs:
%     features  - (N x F) numeric feature matrix
%     labels    - (N x 1) binary label vector (0 = rebound, 1 = penetration)
%
%   Outputs:
%     model       - struct with TreeBagger model, scaler, and feature importance
%     accuracy    - training set accuracy
%     cv_accuracy - Out-of-Bag (OOB) error estimate (1 − OOB error)
%     metrics     - struct: precision, recall, f1_score, confusion_matrix,
%                           feature_importance

fprintf('\n--- Training Random Forest Ensemble (100 trees) ---\n');

% -------------------------------------------------------------------------
% No standardisation required for tree-based methods, but we store the
% scaler anyway so that compute_threshold_ke can use a uniform interface.
% -------------------------------------------------------------------------
mu    = mean(features, 1);
sigma = std(features,  0, 1);
sigma(sigma < eps) = 1;

% -------------------------------------------------------------------------
% Fit TreeBagger (Random Forest)
% -------------------------------------------------------------------------
n_trees = 100;
tb = TreeBagger(n_trees, features, labels, ...
    'Method',                  'classification', ...
    'OOBPrediction',           'on', ...
    'OOBPredictorImportance',  'on', ...
    'MinLeafSize',             1, ...
    'NumPredictorsToSample',   'sqrt');

% -------------------------------------------------------------------------
% Training accuracy
% -------------------------------------------------------------------------
[pred_cell, ~] = predict(tb, features);
pred_train     = str2double(pred_cell);
accuracy       = mean(pred_train == labels);
fprintf('  Training accuracy : %.2f%%\n', accuracy * 100);

% -------------------------------------------------------------------------
% OOB-based CV accuracy
% -------------------------------------------------------------------------
oob_error   = oobError(tb);
cv_accuracy = 1 - oob_error(end);
fprintf('  OOB CV accuracy   : %.2f%%\n', cv_accuracy * 100);

% -------------------------------------------------------------------------
% Feature importance
% -------------------------------------------------------------------------
feat_importance = tb.OOBPermutedPredictorImportance;

% -------------------------------------------------------------------------
% Pack into struct
% -------------------------------------------------------------------------
model.type              = 'Ensemble';
model.tb                = tb;
model.mu                = mu;
model.sigma             = sigma;
model.feat_importance   = feat_importance;

% -------------------------------------------------------------------------
% Metrics
% -------------------------------------------------------------------------
metrics                   = compute_metrics(labels, pred_train);
metrics.feature_importance = feat_importance;
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
