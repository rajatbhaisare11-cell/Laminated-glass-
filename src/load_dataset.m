function data = load_dataset(excel_file)
% LOAD_DATASET  Load the laminated glass impact dataset.
%
%   data = load_dataset()            uses the hardcoded dataset (default)
%   data = load_dataset(excel_file)  attempts to read from an Excel file;
%                                    falls back to hardcoded data on failure
%
%   Returns a MATLAB table with the following columns:
%     Sr_no, Sample_no, Impact, Mass_kg, KE_J, Velocity_mps,
%     Penetration_Status, Glass1_mm, PVB1_mm, Glass2_mm, PVB2_mm, Glass3_mm
%   Plus derived features:
%     TotalGlassThickness, TotalPVBThickness, TotalPanelThickness,
%     NumLayers, ConfigType, ImpactLocation

% -------------------------------------------------------------------------
% Try to load from Excel file if one is supplied
% -------------------------------------------------------------------------
if nargin >= 1 && ~isempty(excel_file)
    try
        fprintf('Attempting to read dataset from: %s\n', excel_file);
        raw = readtable(excel_file);
        % Basic sanity check
        if height(raw) >= 32
            fprintf('Successfully loaded %d rows from Excel file.\n', height(raw));
            data = add_derived_features(raw);
            return;
        else
            warning('Excel file has fewer than 32 rows — falling back to hardcoded data.');
        end
    catch ME
        warning('Could not read Excel file (%s) — falling back to hardcoded data.', ME.message);
    end
end

% -------------------------------------------------------------------------
% Hardcoded dataset (32 rows from LS-DYNA FE simulations)
% -------------------------------------------------------------------------
fprintf('Loading hardcoded dataset...\n');

%  Sr_no  Samp  Impact     Mass    KE      Vel  Pen  G1  PVB1  G2  PVB2  G3
raw_data = [
    1,  25, 1, 1,  312.5, 25, 0, 3, 1.52, 3, 1.52, 3;
    2,  26, 1, 1,  112.5, 15, 0, 3, 1.52, 3, 1.52, 3;
    3,  29, 0, 1,  312.5, 25, 0, 3, 1.52, 3, 1.52, 3;
    4,  30, 0, 1,  112.5, 15, 0, 3, 1.52, 3, 1.52, 3;
    5,   9, 1, 2,  625.0, 25, 1, 3, 1.52, 3, 1.52, 3;
    6,  10, 1, 2,  225.0, 15, 0, 3, 1.52, 3, 1.52, 3;
    7,  13, 0, 2,  625.0, 25, 1, 3, 1.52, 3, 1.52, 3;
    8,  14, 0, 2,  225.0, 15, 0, 3, 1.52, 3, 1.52, 3;
    9,  27, 1, 3,  937.5, 25, 1, 3, 1.52, 3, 1.52, 3;
   10,  28, 1, 3,  337.5, 15, 0, 3, 1.52, 3, 1.52, 3;
   11,  31, 0, 3,  937.5, 25, 1, 3, 1.52, 3, 1.52, 3;
   12,  32, 0, 3,  337.5, 15, 0, 3, 1.52, 3, 1.52, 3;
   13,  11, 1, 4, 1250.0, 25, 1, 3, 1.52, 3, 1.52, 3;
   14,  12, 1, 4,  450.0, 15, 0, 3, 1.52, 3, 1.52, 3;
   15,  15, 0, 4, 1250.0, 25, 1, 3, 1.52, 3, 1.52, 3;
   16,  16, 0, 4,  450.0, 15, 1, 3, 1.52, 3, 1.52, 3;
   17,  17, 1, 1,  312.5, 25, 0, 6, 1.52, 6,    0, 0;
   18,  18, 1, 1,  112.5, 15, 0, 6, 1.52, 6,    0, 0;
   19,  21, 0, 1,  312.5, 25, 0, 6, 1.52, 6,    0, 0;
   20,  22, 0, 1,  112.5, 15, 0, 6, 1.52, 6,    0, 0;
   21,   1, 1, 2,  625.0, 25, 1, 6, 1.52, 6,    0, 0;
   22,   2, 1, 2,  225.0, 15, 0, 6, 1.52, 6,    0, 0;
   23,   5, 0, 2,  625.0, 25, 1, 6, 1.52, 6,    0, 0;
   24,   6, 0, 2,  225.0, 15, 0, 6, 1.52, 6,    0, 0;
   25,  19, 1, 3,  937.5, 25, 1, 6, 1.52, 6,    0, 0;
   26,  20, 1, 3,  337.5, 15, 0, 6, 1.52, 6,    0, 0;
   27,  23, 0, 3,  937.5, 25, 1, 6, 1.52, 6,    0, 0;
   28,  24, 0, 3,  337.5, 15, 1, 6, 1.52, 6,    0, 0;
   29,   3, 1, 4, 1250.0, 25, 1, 6, 1.52, 6,    0, 0;
   30,   4, 1, 4,  450.0, 15, 1, 6, 1.52, 6,    0, 0;
   31,   7, 0, 4, 1250.0, 25, 1, 6, 1.52, 6,    0, 0;
   32,   8, 0, 4,  450.0, 15, 1, 6, 1.52, 6,    0, 0;
];
% Column 3: Impact location — 1 = center, 0 = corner (numeric encoding)
% (the text labels are reconstructed below)

% Build the table
Sr_no             = raw_data(:,  1);
Sample_no         = raw_data(:,  2);
impact_numeric    = raw_data(:,  3);   % 1=center, 0=corner
Mass_kg           = raw_data(:,  4);
KE_J              = raw_data(:,  5);
Velocity_mps      = raw_data(:,  6);
Penetration_Status = raw_data(:, 7);
Glass1_mm         = raw_data(:,  8);
PVB1_mm           = raw_data(:,  9);
Glass2_mm         = raw_data(:, 10);
PVB2_mm           = raw_data(:, 11);
Glass3_mm         = raw_data(:, 12);

% Text impact labels
impact_labels = cell(32, 1);
for i = 1:32
    if impact_numeric(i) == 1
        impact_labels{i} = 'center';
    else
        impact_labels{i} = 'corner';
    end
end

data = table(Sr_no, Sample_no, impact_labels, Mass_kg, KE_J, Velocity_mps, ...
             Penetration_Status, Glass1_mm, PVB1_mm, Glass2_mm, PVB2_mm, Glass3_mm, ...
             'VariableNames', {'Sr_no','Sample_no','Impact','Mass_kg','KE_J', ...
             'Velocity_mps','Penetration_Status','Glass1_mm','PVB1_mm', ...
             'Glass2_mm','PVB2_mm','Glass3_mm'});

data = add_derived_features(data);

fprintf('Dataset loaded: %d rows, %d columns.\n', height(data), width(data));
end


% =========================================================================
function data = add_derived_features(data)
% ADD_DERIVED_FEATURES  Append engineered columns to the dataset table.

% --- Numeric impact location (1 = center, 0 = corner) --------------------
n = height(data);
ImpactLocation = zeros(n, 1);
for i = 1:n
    if strcmpi(data.Impact{i}, 'center')
        ImpactLocation(i) = 1;
    else
        ImpactLocation(i) = 0;
    end
end

% --- Thickness summaries --------------------------------------------------
TotalGlassThickness  = data.Glass1_mm + data.Glass2_mm + data.Glass3_mm;
TotalPVBThickness    = data.PVB1_mm   + data.PVB2_mm;
TotalPanelThickness  = TotalGlassThickness + TotalPVBThickness;

% --- Number of non-zero glass layers -------------------------------------
NumLayers = double(data.Glass1_mm > 0) + ...
            double(data.Glass2_mm > 0) + ...
            double(data.Glass3_mm > 0);

% --- Configuration type --------------------------------------------------
%   1 = 5-layer (Glass3 > 0)   Config 1: 3-1.52-3-1.52-3
%   2 = 3-layer (Glass3 == 0)  Config 2: 6-1.52-6
ConfigType = ones(n, 1) * 2;
ConfigType(data.Glass3_mm > 0) = 1;

% --- Append to table ------------------------------------------------------
data.TotalGlassThickness  = TotalGlassThickness;
data.TotalPVBThickness    = TotalPVBThickness;
data.TotalPanelThickness  = TotalPanelThickness;
data.NumLayers            = NumLayers;
data.ConfigType           = ConfigType;
data.ImpactLocation       = ImpactLocation;
end
