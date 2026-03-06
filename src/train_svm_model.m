function [model, accuracy, cv_accuracy, metrics] = train_svm_model(features, labels)
% TRAIN_SVM_MODEL  Train an SVM classifier with RBF kernel.
%
%   [model, accuracy, cv_accuracy, metrics] = train_svm_model(features, labels)
%
%   Inputs:
%     features  - (N x F) numeric feature matrix
%     labels    - (N x 1) binary label vector (0 = rebound, 1 = penetration)
%
%   Outputs:
%     model       - trained SVM model (with posterior probabilities)
%     accuracy    - training set accuracy (fraction correct)
%     cv_accuracy - Leave-One-Out cross-validation accuracy
%     metrics     - struct with fields:
%                     precision, recall, f1_score, confusion_matrix

fprintf('\n--- Training SVM (RBF kernel) ---\n');

% -------------------------------------------------------------------------
% Standardise features (zero mean, unit variance)
% -------------------------------------------------------------------------
[features_scaled, mu, sigma] = zscore(features);
% Replace near-zero std columns to avoid NaN
sigma(sigma < eps) = 1;
features_scaled = (features - mu) ./ sigma;

% Store scaling parameters inside the model struct for later use
% We wrap everything in a struct to keep things self-contained
svm_raw = fitcsvm(features_scaled, labels, ...
    'KernelFunction',  'rbf', ...
    'Standardize',     false, ...   % already standardised manually
    'ClassNames',      [0, 1], ...
    'OptimizeHyperparameters', 'none');

% Enable probability estimates via Platt scaling
model_raw = fitPosterior(svm_raw);

% Pack into a struct so we can carry the scaler parameters along
model.type       = 'SVM';
model.svm        = model_raw;
model.mu         = mu;
model.sigma      = sigma;
model.feat_names = [];   % populated by caller if desired

% -------------------------------------------------------------------------
% Training accuracy
% -------------------------------------------------------------------------
pred_train = predict(model_raw, features_scaled);
accuracy   = mean(pred_train == labels);
fprintf('  Training accuracy : %.2f%%\n', accuracy * 100);

% -------------------------------------------------------------------------
% Leave-One-Out Cross-Validation
% -------------------------------------------------------------------------
n = size(features_scaled, 1);
loocv_pred = zeros(n, 1);
for i = 1:n
    idx_train = setdiff(1:n, i);
    X_tr = features_scaled(idx_train, :);
    y_tr = labels(idx_train);
    X_te = features_scaled(i, :);

    % Fit a temporary SVM on the leave-one-out training set
    tmp = fitcsvm(X_tr, y_tr, ...
        'KernelFunction', 'rbf', ...
        'Standardize',    false, ...
        'ClassNames',     unique(y_tr));
    loocv_pred(i) = predict(tmp, X_te);
end
cv_accuracy = mean(loocv_pred == labels);
fprintf('  LOOCV accuracy    : %.2f%%\n', cv_accuracy * 100);

% -------------------------------------------------------------------------
% Metrics (on full training set)
% -------------------------------------------------------------------------
metrics = compute_metrics(labels, pred_train);
fprintf('  Precision: %.3f  |  Recall: %.3f  |  F1: %.3f\n', ...
    metrics.precision, metrics.recall, metrics.f1_score);
end


% =========================================================================
function metrics = compute_metrics(y_true, y_pred)
% COMPUTE_METRICS  Compute precision, recall, F1, and confusion matrix.
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
