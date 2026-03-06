%% =========================================================================
%  LaminatedGlass_ThresholdPrediction.m
%
%  Comprehensive data-driven predictive framework for determining the
%  rebound–penetration threshold of laminated glass panels subjected to
%  lumbar impact.
%
%  Dataset source: Finite element simulations performed in LS-DYNA.
%
%  Configurations supported
%  -------------------------
%   Config 1 (3-ply)  : Glass / PVB interlayer / Glass
%   Config 2 (5-ply)  : Glass / PVB interlayer / Glass / PVB interlayer / Glass
%
%  Impactor masses : 1, 2, 3, 4  kg
%
%  Methodology
%  -----------
%  1. Load and preprocess the LS-DYNA simulation dataset.
%  2. Engineer composite features (total thickness, thickness ratio, etc.).
%  3. Train three classification models:
%       (a) Support Vector Machine (SVM) with RBF kernel  [PRIMARY]
%       (b) Fine decision tree
%       (c) Logistic regression (linear discriminant)
%  4. Evaluate models using 5-fold stratified cross-validation.
%  5. For each (configuration, impactor-mass) pair, perform a fine KE sweep
%     and locate the decision boundary → threshold KE.
%  6. Produce publication-quality figures.
%
%  Usage
%  -----
%   >> LaminatedGlass_ThresholdPrediction
%
%  The script reads  'Data_LaminatedGlass.csv'  from the same folder.
%  To use your own Excel file, replace the readtable call in Section 1 with:
%   >> T = readtable('YourFile.xlsx');
%  and ensure the column names match those listed in Section 1.
%
%  Author  : Generated for the Laminated Glass impact prediction project
%  Date    : 2025
% =========================================================================

clc; clear; close all;

fprintf('=================================================================\n');
fprintf(' Laminated Glass – Rebound/Penetration Threshold Prediction\n');
fprintf('=================================================================\n\n');

%% =========================================================================
%  SECTION 1 – DATA LOADING
% =========================================================================
fprintf('[1/6] Loading dataset...\n');

% ---- Adjust this path or filename as needed ----
dataFile = 'Data_LaminatedGlass.csv';

if ~isfile(dataFile)
    error(['Dataset file "%s" not found.\n' ...
           'Please place the CSV/Excel file in the same directory as this script,\n' ...
           'or update the ''dataFile'' variable above.'], dataFile);
end

% Read table (works for .csv and .xlsx)
T = readtable(dataFile, 'VariableNamingRule', 'preserve');

fprintf('   Loaded %d rows × %d columns.\n', height(T), width(T));
fprintf('   Columns: %s\n\n', strjoin(T.Properties.VariableNames, ', '));

% ---- Required column names (rename here if your file differs) ----
colConfig   = 'Config';
colTg1      = 't_g1_mm';
colTil1     = 't_il1_mm';
colTg2      = 't_g2_mm';
colTil2     = 't_il2_mm';
colTg3      = 't_g3_mm';
colTglTotal = 'Total_Glass_mm';
colTilTotal = 'Total_Interlayer_mm';
colMass     = 'Mass_kg';
colVel      = 'Velocity_ms';
colKE       = 'KE_J';
colStatus   = 'Status';         % 0 = Rebound, 1 = Penetration
colDefl     = 'Central_Deflection_mm';
colResV     = 'Residual_Velocity_ms';

% Verify all required columns are present
requiredCols = {colConfig, colTg1, colTil1, colTg2, colTil2, colTg3, ...
                colTglTotal, colTilTotal, colMass, colVel, colKE, ...
                colStatus, colDefl, colResV};
missingCols = setdiff(requiredCols, T.Properties.VariableNames);
if ~isempty(missingCols)
    error('Missing columns in dataset: %s', strjoin(missingCols, ', '));
end

%% =========================================================================
%  SECTION 2 – EXPLORATORY DATA ANALYSIS & FEATURE ENGINEERING
% =========================================================================
fprintf('[2/6] Feature engineering and exploratory analysis...\n');

% Extract raw arrays
Config    = T.(colConfig);
t_g1      = T.(colTg1);
t_il1     = T.(colTil1);
t_g2      = T.(colTg2);
t_il2     = T.(colTil2);
t_g3      = T.(colTg3);
T_glass   = T.(colTglTotal);    % total glass thickness
T_il      = T.(colTilTotal);    % total interlayer thickness
Mass      = T.(colMass);
Velocity  = T.(colVel);
KE        = T.(colKE);
Status    = T.(colStatus);      % binary label: 0=Rebound, 1=Penetration
Defl      = T.(colDefl);
ResV      = T.(colResV);

% ---- Engineered features ----
T_total     = T_glass + T_il;              % total panel thickness (mm)
thickness_r = T_glass ./ (T_il + eps);    % glass-to-interlayer thickness ratio
nLayers     = 3 + 2*(t_g3 > 0);           % 3 for 3-ply, 5 for 5-ply
KE_per_mass = KE ./ Mass;                  % specific kinetic energy (J/kg)

% Assemble feature matrix
%   Features: [t_g1, t_il1, t_g2, t_il2, t_g3, T_glass, T_il, T_total,
%              thickness_r, nLayers, Mass, Velocity, KE, KE_per_mass]
X = [t_g1, t_il1, t_g2, t_il2, t_g3, T_glass, T_il, T_total, ...
     thickness_r, nLayers, Mass, Velocity, KE, KE_per_mass];

featureNames = {'t_g1','t_il1','t_g2','t_il2','t_g3','T_glass','T_il', ...
                'T_total','thickness_ratio','nLayers','Mass','Velocity', ...
                'KE','KE_per_mass'};

y = Status;  % 0 or 1

fprintf('   Total samples  : %d\n', numel(y));
fprintf('   Rebound  (0)  : %d\n', sum(y==0));
fprintf('   Penetration(1): %d\n\n', sum(y==1));

% ---- Derived index arrays (used throughout) ----
configIDs = unique(Config);
massVals  = unique(Mass);
colors    = lines(numel(massVals));   % one colour per impactor mass

fig1 = figure('Name','Dataset Overview','Position',[50 50 1200 500]);
for ci = 1:numel(configIDs)
    ax = subplot(1,2,ci);
    hold(ax,'on');
    cfgMask = (Config == configIDs(ci));
    for mi = 1:numel(massVals)
        mask = cfgMask & (Mass == massVals(mi));
        rbMask  = mask & (y==0);
        penMask = mask & (y==1);
        scatter(ax, KE(rbMask),  Velocity(rbMask),  40, colors(mi,:), 'o', ...
            'DisplayName', sprintf('m=%dkg Rebound',   massVals(mi)));
        scatter(ax, KE(penMask), Velocity(penMask), 40, colors(mi,:), '^', ...
            'filled', 'DisplayName', sprintf('m=%dkg Penetration', massVals(mi)));
    end
    xlabel(ax,'Initial KE (J)'); ylabel(ax,'Impact Velocity (m/s)');
    cfgStr = getConfigStr(configIDs(ci), configs_info(configIDs(ci)));
    title(ax, sprintf('Config %d – %s', configIDs(ci), cfgStr));
    legend(ax,'Location','northwest','FontSize',7);
    grid(ax,'on'); box(ax,'on');
end
sgtitle('LS-DYNA Simulation Dataset: Rebound (o) vs Penetration (▲)');

%% =========================================================================
%  SECTION 3 – MODEL TRAINING
% =========================================================================
fprintf('[3/6] Training classification models...\n');

% ---- Normalise features (z-score) ----
[X_norm, mu_X, sigma_X] = zscore(X);

% ----- (a) SVM with RBF (Gaussian) kernel -----
fprintf('   Training SVM (RBF kernel)...\n');
svmModel = fitcsvm(X_norm, y, ...
    'KernelFunction', 'rbf', ...
    'BoxConstraint',  10, ...
    'KernelScale',    'auto', ...
    'Standardize',    false, ...     % already normalised
    'ClassNames',     [0,1], ...
    'OptimizeHyperparameters', 'none');

% Compute SVM scores (positive → penetration)
[~, svmScores] = predict(svmModel, X_norm);

% ----- (b) Fine decision tree -----
fprintf('   Training Decision Tree...\n');
treeModel = fitctree(X_norm, y, ...
    'MinLeafSize', 1, ...
    'SplitCriterion', 'gdi');

% ----- (c) Logistic regression (linear classifier with logistic loss) -----
fprintf('   Training Logistic Regression...\n');
lrModel = fitclinear(X_norm, y, ...
    'Learner',       'logistic', ...
    'Regularization','ridge', ...
    'Lambda',         1e-3);

fprintf('\n');

%% =========================================================================
%  SECTION 4 – MODEL EVALUATION (5-FOLD CROSS-VALIDATION)
% =========================================================================
fprintf('[4/6] Evaluating models (5-fold cross-validation)...\n');

cvPartition = cvpartition(y, 'KFold', 5, 'Stratify', true);

models     = {svmModel,  treeModel,  lrModel};
modelNames = {'SVM (RBF)', 'Decision Tree', 'Logistic Regression'};
cvAccuracy = zeros(1,3);

for m = 1:3
    switch m
        case 1
            cvMdl = crossval(svmModel,  'CVPartition', cvPartition);
        case 2
            cvMdl = crossval(treeModel, 'CVPartition', cvPartition);
        case 3
            cvMdl = crossval(lrModel,   'CVPartition', cvPartition);
    end
    cvErr = kfoldLoss(cvMdl);
    cvAccuracy(m) = (1 - cvErr) * 100;
    fprintf('   %-22s  CV Accuracy = %.1f%%\n', modelNames{m}, cvAccuracy(m));
end
fprintf('\n');

% ---- Confusion matrix for primary model (SVM) ----
y_pred_svm = predict(svmModel, X_norm);
cm = confusionmat(y, y_pred_svm);

fig2 = figure('Name','SVM Confusion Matrix','Position',[50 600 450 380]);
confusionchart(cm, {'Rebound','Penetration'}, ...
    'Title','SVM Confusion Matrix (Training Set)', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

% ---- Bar chart of model accuracies ----
fig3 = figure('Name','Model Comparison','Position',[520 600 500 350]);
bar(cvAccuracy, 'FaceColor','flat', 'CData', lines(3));
set(gca,'XTickLabel', modelNames, 'XTickLabelRotation', 15);
ylabel('5-Fold CV Accuracy (%)');
ylim([0 105]);
title('Classification Model Comparison');
grid on;
for k = 1:3
    text(k, cvAccuracy(k)+1, sprintf('%.1f%%', cvAccuracy(k)), ...
        'HorizontalAlignment','center','FontSize',9);
end

%% =========================================================================
%  SECTION 5 – THRESHOLD KE COMPUTATION
% =========================================================================
fprintf('[5/6] Computing threshold KE for each (Config, Mass) pair...\n');
fprintf('   Using SVM decision boundary (score = 0 crossing).\n\n');

% Resolution of KE sweep (J)
KE_sweep_min = 1.0;
KE_sweep_max = 80.0;
KE_step      = 0.01;   % fine resolution for accurate threshold detection
KE_sweep     = (KE_sweep_min : KE_step : KE_sweep_max)';

% Struct to hold results
results = struct();
resultIdx = 0;

fprintf('   %-10s %-10s %-18s %-18s %-12s\n', ...
    'Config','Mass(kg)','Threshold KE(J)','Threshold V(m/s)','Model');
fprintf('   %s\n', repmat('-',1,72));

for ci = 1:numel(configIDs)
    cfgID = configIDs(ci);
    % Representative geometry for this configuration
    cfgRows  = find(Config == cfgID, 1);
    g1  = t_g1(cfgRows);   il1 = t_il1(cfgRows);
    g2  = t_g2(cfgRows);   il2 = t_il2(cfgRows);
    g3  = t_g3(cfgRows);
    Tg  = T_glass(cfgRows);
    Ti  = T_il(cfgRows);
    Tt  = T_total(cfgRows);
    thR = thickness_r(cfgRows);
    nL  = nLayers(cfgRows);

    for mi = 1:numel(massVals)
        mVal = massVals(mi);

        % Build sweep feature matrix for this (config, mass, KE sweep)
        V_sweep      = sqrt(2 * KE_sweep / mVal);
        KEpm_sweep   = KE_sweep / mVal;

        N = numel(KE_sweep);
        X_sweep = [repmat(g1,N,1),  repmat(il1,N,1), repmat(g2,N,1), ...
                   repmat(il2,N,1), repmat(g3,N,1),  repmat(Tg,N,1), ...
                   repmat(Ti,N,1),  repmat(Tt,N,1),  repmat(thR,N,1),...
                   repmat(nL,N,1),  repmat(mVal,N,1), V_sweep, ...
                   KE_sweep,        KEpm_sweep];

        % Normalise using training statistics
        X_sweep_norm = (X_sweep - mu_X) ./ sigma_X;

        % Get raw SVM scores (positive side → predicted Penetration)
        [~, scores] = predict(svmModel, X_sweep_norm);
        % scores(:,2) corresponds to class 1 (Penetration)
        svmScore = scores(:,2);

        % Find first crossing from negative to positive (rebound → penetration)
        crossIdx = find(diff(sign(svmScore)) > 0, 1);

        if ~isempty(crossIdx)
            % Linear interpolation for sub-step accuracy
            s1 = svmScore(crossIdx);
            s2 = svmScore(crossIdx+1);
            ke1 = KE_sweep(crossIdx);
            ke2 = KE_sweep(crossIdx+1);
            ke_thresh = ke1 - s1*(ke2-ke1)/(s2-s1);
            v_thresh  = sqrt(2*ke_thresh/mVal);
        else
            % Fallback: find minimum of |score| (closest to boundary)
            [~,minIdx] = min(abs(svmScore));
            ke_thresh  = KE_sweep(minIdx);
            v_thresh   = sqrt(2*ke_thresh/mVal);
        end

        resultIdx = resultIdx + 1;
        results(resultIdx).Config      = cfgID;
        results(resultIdx).Mass        = mVal;
        results(resultIdx).KE_thresh   = ke_thresh;
        results(resultIdx).V_thresh    = v_thresh;
        results(resultIdx).SVM_scores  = svmScore;
        results(resultIdx).KE_sweep    = KE_sweep;

        fprintf('   Config %-4d  %d kg     %8.2f J          %8.3f m/s      SVM\n', ...
            cfgID, mVal, ke_thresh, v_thresh);
    end
end
fprintf('\n');

%% =========================================================================
%  SECTION 6 – VISUALISATIONS
% =========================================================================
fprintf('[6/6] Generating visualisations...\n');

% ---- 6a  Threshold KE vs Mass for each configuration ----
fig4 = figure('Name','Threshold KE vs Mass','Position',[50 50 700 500]);
hold on;
cfgColors  = [0.2 0.4 0.8 ; 0.9 0.3 0.1];
cfgMarkers = {'o-','s-'};
for ci = 1:numel(configIDs)
    cfgID = configIDs(ci);
    ke_vals = arrayfun(@(r) r.KE_thresh, results([results.Config]==cfgID));
    m_vals  = massVals;
    plot(m_vals, ke_vals, cfgMarkers{ci}, 'Color', cfgColors(ci,:), ...
        'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', cfgColors(ci,:), ...
        'DisplayName', sprintf('Config %d (%s)', cfgID, ...
            getConfigStr(cfgID, configs_info(cfgID))));
end
xlabel('Impactor Mass (kg)', 'FontSize',13);
ylabel('Threshold Kinetic Energy (J)', 'FontSize',13);
title('Rebound–Penetration Threshold KE vs Impactor Mass', 'FontSize',14);
legend('Location','northwest','FontSize',11);
grid on; box on;
xticks(massVals);
set(gca,'FontSize',12);

% ---- 6b  Threshold velocity vs Mass ----
fig5 = figure('Name','Threshold Velocity vs Mass','Position',[760 50 700 500]);
hold on;
for ci = 1:numel(configIDs)
    cfgID = configIDs(ci);
    v_vals = arrayfun(@(r) r.V_thresh, results([results.Config]==cfgID));
    m_vals = massVals;
    plot(m_vals, v_vals, cfgMarkers{ci}, 'Color', cfgColors(ci,:), ...
        'LineWidth', 2, 'MarkerSize', 10, 'MarkerFaceColor', cfgColors(ci,:), ...
        'DisplayName', sprintf('Config %d (%s)', cfgID, ...
            getConfigStr(cfgID, configs_info(cfgID))));
end
xlabel('Impactor Mass (kg)', 'FontSize',13);
ylabel('Threshold Velocity (m/s)', 'FontSize',13);
title('Rebound–Penetration Threshold Velocity vs Impactor Mass', 'FontSize',14);
legend('Location','northeast','FontSize',11);
grid on; box on;
xticks(massVals);
set(gca,'FontSize',12);

% ---- 6c  KE-Mass decision boundary plots (one subplot per config) ----
fig6 = figure('Name','Decision Boundary KE-Mass','Position',[50 550 1200 480]);
nMasses = numel(massVals);
for ci = 1:numel(configIDs)
    cfgID   = configIDs(ci);
    cfgMask = (Config == cfgID);
    ax      = subplot(1,2,ci);
    hold(ax,'on');

    % Plot raw data points; capture first handle of each type for legend
    hRb  = gobjects(0);
    hPen = gobjects(0);
    for mi = 1:nMasses
        mVal   = massVals(mi);
        mask   = cfgMask & (Mass == mVal);
        rbMask = mask & (y==0);
        pnMask = mask & (y==1);
        c = colors(mi,:);
        h1 = scatter(ax, Mass(rbMask)+randn(sum(rbMask),1)*0.02, KE(rbMask), ...
            50, c, 'o', 'LineWidth', 1.2);
        h2 = scatter(ax, Mass(pnMask)+randn(sum(pnMask),1)*0.02, KE(pnMask), ...
            50, c, '^', 'filled');
        if mi == 1
            hRb  = h1;
            hPen = h2;
        end
    end

    % Overlay threshold line
    ke_thresh_vec = arrayfun(@(r) r.KE_thresh, results([results.Config]==cfgID));
    hThr = plot(ax, massVals, ke_thresh_vec, 'k-o', 'LineWidth', 2.5, ...
        'MarkerSize', 9, 'MarkerFaceColor', 'k');

    xlabel(ax, 'Impactor Mass (kg)', 'FontSize', 12);
    ylabel(ax, 'Kinetic Energy (J)',  'FontSize', 12);
    title(ax,  sprintf('Config %d – %s\nDecision Boundary (SVM)', ...
        cfgID, getConfigStr(cfgID, configs_info(cfgID))), 'FontSize',11);
    legend(ax, [hRb, hPen, hThr], {'Rebound','Penetration','SVM Threshold'}, ...
        'Location','northwest','FontSize',9);
    grid(ax,'on'); box(ax,'on');
    xticks(ax, massVals);
    ylim(ax, [0 max(KE(cfgMask))*1.1]);
end
sgtitle('Decision Boundary: Rebound (o) vs Penetration (▲) — SVM Predicted Threshold (—●)', ...
    'FontSize', 13);

% ---- 6d  SVM score vs KE for each (config, mass) ----
fig7 = figure('Name','SVM Score Profile','Position',[50 50 1200 650]);
nCfg = numel(configIDs);
nMas = numel(massVals);
pIdx = 0;
for ci = 1:nCfg
    cfgID = configIDs(ci);
    cfgRes = results([results.Config]==cfgID);
    for mi = 1:nMas
        pIdx = pIdx + 1;
        r    = cfgRes(mi);
        ax   = subplot(nCfg, nMas, pIdx);
        plot(ax, r.KE_sweep, r.SVM_scores, 'b-', 'LineWidth',1.5);
        hold(ax,'on');
        yline(ax, 0, 'r--', 'LineWidth',1.5);
        xline(ax, r.KE_thresh, 'g-', 'LineWidth',2, ...
            'Label', sprintf('%.1fJ',r.KE_thresh), ...
            'LabelHorizontalAlignment','right');
        xlabel(ax,'KE (J)','FontSize',9);
        ylabel(ax,'SVM Score','FontSize',9);
        title(ax, sprintf('C%d  m=%dkg\nKE_{th}=%.1fJ', ...
            cfgID, massVals(mi), r.KE_thresh), 'FontSize',9);
        grid(ax,'on');
        xlim(ax,[KE_sweep_min KE_sweep_max/2]);
    end
end
sgtitle('SVM Decision Score vs KE (threshold at score = 0)', 'FontSize',12);

% ---- 6e  Central deflection vs KE ----
fig8 = figure('Name','Deflection vs KE','Position',[50 50 1100 420]);
for ci = 1:nCfg
    cfgID   = configIDs(ci);
    cfgMask = (Config == cfgID);
    ax      = subplot(1,2,ci);
    hold(ax,'on');
    for mi = 1:nMas
        mVal   = massVals(mi);
        mask   = cfgMask & (Mass == mVal);
        c = colors(mi,:);
        plot(ax, KE(mask), Defl(mask), 'o-', 'Color', c, 'LineWidth', 1.5, ...
            'MarkerFaceColor', c, 'MarkerSize', 6, ...
            'DisplayName', sprintf('m=%dkg', mVal));
    end
    xlabel(ax,'KE (J)','FontSize',12);
    ylabel(ax,'Central Deflection (mm)','FontSize',12);
    title(ax, sprintf('Config %d – %s', cfgID, ...
        getConfigStr(cfgID, configs_info(cfgID))),'FontSize',11);
    legend(ax,'Location','northwest','FontSize',9);
    grid(ax,'on'); box(ax,'on');
end
sgtitle('Central Panel Deflection vs Initial KE', 'FontSize',13);

%% =========================================================================
%  SUMMARY TABLE
% =========================================================================
fprintf('=================================================================\n');
fprintf(' SUMMARY: Predicted Threshold Kinetic Energies\n');
fprintf('=================================================================\n');
fprintf(' %-10s %-10s %-22s %-18s\n', 'Config','Mass(kg)', ...
    'Threshold KE (J)', 'Threshold V (m/s)');
fprintf(' %s\n', repmat('-',1,62));
for ri = 1:resultIdx
    r = results(ri);
    fprintf(' Config %-4d  %d kg       %8.2f J          %8.3f m/s\n', ...
        r.Config, r.Mass, r.KE_thresh, r.V_thresh);
end
fprintf('=================================================================\n\n');

% Save results to a CSV
outputTable = table(...
    [results.Config]', [results.Mass]', ...
    [results.KE_thresh]', [results.V_thresh]', ...
    'VariableNames', {'Config','Mass_kg','Threshold_KE_J','Threshold_V_ms'});
writetable(outputTable, 'Threshold_KE_Results.csv');
fprintf('Results saved to: Threshold_KE_Results.csv\n');
fprintf('Figures generated: %d\n\n', 8);

%% =========================================================================
%  LOCAL HELPER FUNCTIONS
% =========================================================================

function s = configs_info(cfgID)
% Returns a short label for each configuration ID.
switch cfgID
    case 1, s = '5/1.52/5 mm';
    case 2, s = '4/0.76/4/0.76/4 mm';
    otherwise, s = 'Custom';
end
end

function s = getConfigStr(cfgID, info)
% Returns a readable config description string.
switch cfgID
    case 1, s = sprintf('3-ply (%s Glass/PVB)', info);
    case 2, s = sprintf('5-ply (%s Glass/PVB)', info);
    otherwise, s = sprintf('Config %d (%s)', cfgID, info);
end
end
