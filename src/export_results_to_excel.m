function export_results_to_excel(threshold_results, model_comparison, ...
                                  full_predictions, feat_importance, results_dir)
% EXPORT_RESULTS_TO_EXCEL  Write analysis results to an Excel workbook.
%
%   Inputs:
%     threshold_results  - table: Config, Impact_Location, Mass_kg,
%                                 Threshold_KE_J, Threshold_Velocity_mps
%     model_comparison   - table: Model_Name, Training_Accuracy, CV_Accuracy,
%                                 Precision, Recall, F1_Score
%     full_predictions   - table: original data + predicted classes + probs
%     feat_importance    - table: Feature_Name, Importance_Score
%     results_dir        - directory path for saving the Excel file

if nargin < 5 || isempty(results_dir)
    results_dir = fullfile(fileparts(mfilename('fullpath')), '..', 'results');
end
if ~exist(results_dir, 'dir')
    mkdir(results_dir);
end

xlsx_path = fullfile(results_dir, 'Threshold_Results.xlsx');
fprintf('\n  Exporting results to: %s\n', xlsx_path);

% -------------------------------------------------------------------------
% Sheet 1: Threshold_KE
% -------------------------------------------------------------------------
try
    writetable(threshold_results, xlsx_path, 'Sheet', 'Threshold_KE');
    fprintf('  Sheet "Threshold_KE" written (%d rows).\n', height(threshold_results));
catch ME
    warning('Could not write Threshold_KE sheet: %s', ME.message);
end

% -------------------------------------------------------------------------
% Sheet 2: Model_Comparison
% -------------------------------------------------------------------------
try
    writetable(model_comparison, xlsx_path, 'Sheet', 'Model_Comparison');
    fprintf('  Sheet "Model_Comparison" written (%d rows).\n', height(model_comparison));
catch ME
    warning('Could not write Model_Comparison sheet: %s', ME.message);
end

% -------------------------------------------------------------------------
% Sheet 3: Full_Predictions
% -------------------------------------------------------------------------
try
    writetable(full_predictions, xlsx_path, 'Sheet', 'Full_Predictions');
    fprintf('  Sheet "Full_Predictions" written (%d rows).\n', height(full_predictions));
catch ME
    warning('Could not write Full_Predictions sheet: %s', ME.message);
end

% -------------------------------------------------------------------------
% Sheet 4: Feature_Importance
% -------------------------------------------------------------------------
try
    if ~isempty(feat_importance)
        writetable(feat_importance, xlsx_path, 'Sheet', 'Feature_Importance');
        fprintf('  Sheet "Feature_Importance" written (%d rows).\n', height(feat_importance));
    end
catch ME
    warning('Could not write Feature_Importance sheet: %s', ME.message);
end

fprintf('  Excel export complete.\n');
end
