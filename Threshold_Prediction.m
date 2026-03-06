%% Threshold_Prediction.m — Rebound/Penetration Threshold Analysis for Laminated Glass
clc; clear; close all;

%% Section 1: Load Data from Excel
t1 = readtable('Combined_DATA_SET_TO_BE_USED.xlsx','Sheet',1,'VariableNamingRule','preserve');
t2 = readtable('Combined_DATA_SET_TO_BE_USED.xlsx','Sheet',2,'VariableNamingRule','preserve');
t1.Config = ones(height(t1),1);
t2.Config = 2*ones(height(t2),1);
T = [t1; t2];

cols = T.Properties.VariableNames;
% Map columns by index: Sr.no(1),SampleNo(2),Impact(3),Mass(4),KE(5),Velocity(6),
%   PenetrationStatus(7),G1(8),PVB1(9),G2(10),PVB2(11),G3(12),Config(13)
Mass   = T{:,4};
KE     = T{:,5};
Vel    = T{:,6};
Status = T{:,7};
G1     = T{:,8};  PVB1 = T{:,9};
G2     = T{:,10}; PVB2 = T{:,11};
G3     = T{:,12};
ImpactRaw = T{:,3};
if iscell(ImpactRaw)
    ImpactLoc = double(strcmpi(strtrim(ImpactRaw),'center'));
else
    ImpactLoc = double(ImpactRaw == 'center');
end
TotalGlass = G1+G2+G3;
TotalPVB   = PVB1+PVB2;
Config     = T{:,end};

%% Section 2: Prepare Features & Train Models
X = [Mass, KE, Vel, G1, PVB1, G2, PVB2, G3, TotalGlass, TotalPVB, ImpactLoc];
y = Status;
n = length(y);

% SVM with RBF + posterior
mdlSVM = fitcsvm(X,y,'KernelFunction','rbf','Standardize',true,'BoxConstraint',1);
mdlSVM = fitPosterior(mdlSVM);

% Logistic regression
mdlLR  = fitglm(X,y,'Distribution','binomial','Link','logit');

% Random Forest
mdlRF  = TreeBagger(100,X,y,'OOBPrediction','on','Method','classification');

% LOOCV accuracy
accSVM=0; accLR=0; accRF=0;
for i=1:n
    tr=[1:i-1,i+1:n];
    svm_i = fitcsvm(X(tr,:),y(tr),'KernelFunction','rbf','Standardize',true,'BoxConstraint',1);
    svm_i = fitPosterior(svm_i);
    [~,ps] = predict(svm_i,X(i,:));
    accSVM = accSVM + (round(ps(2)>0.5)==y(i));

    lr_i  = fitglm(X(tr,:),y(tr),'Distribution','binomial','Link','logit');
    pl    = predict(lr_i,X(i,:));
    accLR = accLR + ((pl>0.5) == y(i));

    rf_i  = TreeBagger(50,X(tr,:),y(tr),'Method','classification');
    pr    = predict(rf_i,X(i,:));
    accRF = accRF + (str2double(pr{1})==y(i));
end
accSVM=accSVM/n; accLR=accLR/n; accRF=accRF/n;

fprintf('\n--- Model Accuracy (LOOCV, n=%d) ---\n',n);
fprintf('SVM:      %.1f%%\nLogistic: %.1f%%\nRandForest:%.1f%%\n', ...
    accSVM*100, accLR*100, accRF*100);

% Pick best model for threshold bisection
[~,bIdx] = max([accSVM,accLR,accRF]);
modelNames = {'SVM','Logistic','RandomForest'};
fprintf('Best model: %s\n', modelNames{bIdx});

% Prediction function using best model
predictP = @(Xq) getPenetrationProb(Xq, bIdx, mdlSVM, mdlLR, mdlRF);

%% Section 3: Compute Threshold KE via Bisection
configs = [1,2];
cparams = {[3,1.52,3,1.52,3]; [6,1.52,6,0,0]};  % G1,PVB1,G2,PVB2,G3
impacts = [1,0]; impNames = {'center','corner'};
masses  = [1,2,3,4];
results = [];
fprintf('\n%-8s %-8s %-8s %-12s %-16s\n','Config','Impact','Mass(kg)','KE_thresh(J)','Vel_thresh(m/s)');
fprintf('%s\n',repmat('-',1,55));
for ci=1:2
    cp = cparams{ci};
    tg = cp(1)+cp(3)+cp(5); tp = cp(2)+cp(4);
    for ii=1:2
        for mi=1:4
            m = masses(mi);
            lo=0; hi=2000; tol=0.1; ke_th=NaN;
            Plo = predictP([m,lo,sqrt(2*lo/m),cp(1),cp(2),cp(3),cp(4),cp(5),tg,tp,impacts(ii)]);
            Phi = predictP([m,hi,sqrt(2*hi/m),cp(1),cp(2),cp(3),cp(4),cp(5),tg,tp,impacts(ii)]);
            if Plo>=0.5, ke_th=0;
            elseif Phi<0.5, ke_th=Inf;
            else
                while (hi-lo)>tol
                    mid=(lo+hi)/2;
                    Pm=predictP([m,mid,sqrt(2*mid/m),cp(1),cp(2),cp(3),cp(4),cp(5),tg,tp,impacts(ii)]);
                    if Pm<0.5, lo=mid; else, hi=mid; end
                end
                ke_th=(lo+hi)/2;
            end
            if isfinite(ke_th), vt=sqrt(2*ke_th/m); else, vt=NaN; end
            results(end+1,:)=[ci, impacts(ii), m, ke_th, vt]; %#ok<AGROW>
            fprintf('%-8d %-8s %-8d %-12.1f %-16.2f\n',ci,impNames{ii},m,ke_th,vt);
        end
    end
end
RT = array2table(results,'VariableNames',{'Config','ImpactLoc','Mass_kg','KE_thresh_J','Vel_thresh_ms'});

%% Section 4: Plots
clr = [0.2 0.6 1; 1 0.3 0.3];
mk  = {'o','s'};

% Figure 1: KE vs Status by config
figure('Name','KE vs Penetration Status');
for ci=1:2
    subplot(1,2,ci);
    idx = Config==ci;
    sc=Status(idx); ke=KE(idx);
    scatter(ke(sc==0),zeros(sum(sc==0),1),60,'b','filled'); hold on;
    scatter(ke(sc==1),ones(sum(sc==1),1),60,'r','filled');
    xlabel('KE (J)'); ylabel('Status'); title(sprintf('Config %d',ci));
    yticks([0 1]); yticklabels({'Rebound','Penetration'});
    legend('Rebound','Penetration'); grid on; set(gca,'FontSize',11);
end

% Figure 2: Threshold KE vs Mass
figure('Name','Threshold KE vs Mass');
for ci=1:2
    for ii=1:2
        ri = RT.Config==ci & RT.ImpactLoc==impacts(ii);
        subplot(2,2,(ci-1)*2+ii);
        plot(RT.Mass_kg(ri),RT.KE_thresh_J(ri),'-','Marker',mk{ii},'LineWidth',1.5,'MarkerSize',8);
        xlabel('Mass (kg)'); ylabel('KE_{thresh} (J)');
        title(sprintf('Config %d — %s',ci,impNames{ii}));
        grid on; set(gca,'FontSize',11);
    end
end

% Figure 3: Threshold Velocity vs Mass
figure('Name','Threshold Velocity vs Mass');
lsym={'-o','-s','--o','--s'}; li=0;
for ci=1:2
    for ii=1:2
        li=li+1;
        ri = RT.Config==ci & RT.ImpactLoc==impacts(ii);
        plot(RT.Mass_kg(ri),RT.Vel_thresh_ms(ri),lsym{li},'LineWidth',1.5,'MarkerSize',8); hold on;
    end
end
xlabel('Mass (kg)'); ylabel('Threshold Velocity (m/s)');
title('Threshold Velocity vs Mass');
legend('C1-center','C1-corner','C2-center','C2-corner','Location','best');
grid on; set(gca,'FontSize',11);

% Figure 4: Confusion matrix for best model
yPred = zeros(n,1);
for i=1:n
    yPred(i) = double(predictP(X(i,:))>0.5);
end
CM = confusionmat(y,yPred);
figure('Name','Confusion Matrix');
imagesc(CM); colorbar; colormap(flipud(gray));
text(1,1,num2str(CM(1,1)),'HorizontalAlignment','center','FontSize',14,'FontWeight','bold');
text(2,1,num2str(CM(1,2)),'HorizontalAlignment','center','FontSize',14,'FontWeight','bold');
text(1,2,num2str(CM(2,1)),'HorizontalAlignment','center','FontSize',14,'FontWeight','bold');
text(2,2,num2str(CM(2,2)),'HorizontalAlignment','center','FontSize',14,'FontWeight','bold');
xlabel('Predicted'); ylabel('Actual');
xticks([1 2]); xticklabels({'Rebound','Penetration'});
yticks([1 2]); yticklabels({'Rebound','Penetration'});
title(sprintf('Confusion Matrix — %s',modelNames{bIdx}));
set(gca,'FontSize',11);

% Figure 5: Decision boundary in (Mass, KE) space
figure('Name','Decision Boundary');
massGrid = linspace(0.5,5,60); keGrid = linspace(0,1500,60);
[MG,KG] = meshgrid(massGrid,keGrid);
for ci=1:2
    cp=cparams{ci}; tg=cp(1)+cp(3)+cp(5); tp=cp(2)+cp(4);
    subplot(1,2,ci);
    Pgrid=zeros(numel(MG),1);
    for k=1:numel(MG)
        m=MG(k); ke=KG(k);
        v=sqrt(max(0,2*ke/m));
        Pgrid(k)=predictP([m,ke,v,cp(1),cp(2),cp(3),cp(4),cp(5),tg,tp,1]);
    end
    contourf(MG,KG,reshape(Pgrid,size(MG)),[0.5,0.5],'LineWidth',2); hold on;
    colormap(gca,[0.8 0.9 1; 1 0.8 0.8]); colorbar;
    idxc=Config==ci;
    scatter(Mass(idxc&Status==0),KE(idxc&Status==0),50,'b','filled');
    scatter(Mass(idxc&Status==1),KE(idxc&Status==1),50,'r','filled');
    xlabel('Mass (kg)'); ylabel('KE (J)');
    title(sprintf('Decision Boundary — Config %d',ci));
    legend('Boundary','Rebound','Penetration'); grid on; set(gca,'FontSize',11);
end

% Figure 6: ROC curve
[~,scSVM] = predict(mdlSVM,X);
[xSVM,ySVM,~,aucSVM] = perfcurve(y,scSVM(:,2),1);
plLR = predict(mdlLR,X);
[xLR,yLR,~,aucLR]   = perfcurve(y,plLR,1);
[~,scRF] = predict(mdlRF,X);
[xRF,yRF,~,aucRF] = perfcurve(y,scRF(:,2),1);
figure('Name','ROC Curve');
plot(xSVM,ySVM,'b-','LineWidth',1.5); hold on;
plot(xLR,yLR,'r-','LineWidth',1.5);
plot(xRF,yRF,'g-','LineWidth',1.5);
plot([0 1],[0 1],'k--');
xlabel('False Positive Rate'); ylabel('True Positive Rate'); title('ROC Curves');
legend(sprintf('SVM (AUC=%.2f)',aucSVM),sprintf('Logistic (AUC=%.2f)',aucLR), ...
    sprintf('RF (AUC=%.2f)',aucRF),'Random','Location','southeast');
grid on; set(gca,'FontSize',11);

% Save all figures
figNames = {'KE_vs_Status','Threshold_KE','Threshold_Vel','Confusion_Matrix','Decision_Boundary','ROC_Curve'};
for f=1:6
    figure(f); saveas(figure(f),sprintf('%s.png',figNames{f}));
end

%% Section 5: Export Results
writetable(RT,'Threshold_Results.xlsx','Sheet',1);
modelComp = table(modelNames(:),round([accSVM;accLR;accRF]*100,1), ...
    'VariableNames',{'Model','LOOCV_Accuracy_pct'});
writetable(modelComp,'Threshold_Results.xlsx','Sheet',2);
fprintf('\nResults exported to Threshold_Results.xlsx\n');

%% Local helper function
function p = getPenetrationProb(Xq, bIdx, mdlSVM, mdlLR, mdlRF)
    if bIdx==1
        [~,ps] = predict(mdlSVM,Xq); p=ps(2);
    elseif bIdx==2
        p = predict(mdlLR,Xq);
        p = min(max(p,0),1);
    else
        [~,ps] = predict(mdlRF,Xq); p=ps(2);
    end
end
