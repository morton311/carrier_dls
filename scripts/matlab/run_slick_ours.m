function run_slick_ours()
%RUN_SLICK_OURS The authors' SLICK Challenge 2.1 benchmark, unmodified, on our split.
%
%   matlab -batch "addpath('scripts/matlab'); run_slick_ours"     (from the repository root)
%
% Uses papers/challenge_2.1/SLICK/challenge21, byte-identical to github.com/SLICK-model/SLICK_matlab
% release SLICK_v1.0.1 / v2.0.1 (commit d61ba838; File Exchange 182867). Their run_benchmark fits SLICK
% to snapshots 1-10368, selects nDFT, rank and both ridge parameters on the 1152 validation snapshots,
% and scores our 30 held-out sequences (0-based starts 11520:40:12700). Their slick_challenge_predict,
% with the selected model, then forecasts the same 30 sequences and 7 calibration sequences
% (10800:100:11400) and a free rollout through the held-out block. scripts/slick_matlab_score.py
% scores the saved forecasts; nothing here is our re-implementation.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'papers', 'challenge_2.1', 'SLICK', 'challenge21'));
out = fullfile(root, 'results', 'slick_challenge21', 'matlab_ours');
cfg = challenge21_config('dataFile', fullfile(root, 'data', 'Challenge2_1_train.h5'), 'outputDir', out, ...
    'nFit', 10368, 'nValidation', 1152, 'nTest', 1280, 'sequenceStride', 40, ...
    'runLSTM', false, 'makePlots', false);
run_benchmark(cfg);

S = load(fullfile(out, 'best_slick_model.mat'), 'bestModel');
model = S.bestModel;
% snapshots in their layout, [all ux; all uy] x time, as run_benchmark's read_data
u = h5read(cfg.dataFile, '/ux'); sz = size(u); ns = sz(1) * sz(2); nt = sz(3);
q = zeros(2 * ns, nt, 'single'); q(1:ns, :) = reshape(single(u), ns, nt); clear u
v = h5read(cfg.dataFile, '/uy'); q(ns + 1:end, :) = reshape(single(v), ns, nt); clear v

ours = 11520:40:12700;
calib = 10800:100:11400;
test = forecast(model, q, ours, 30);
cal = forecast(model, q, calib, 30);
% free rollout of snapshots lo..hi-1 (0-based), seeded by the 70 frames before lo, as scripts/flow_stats.py
lo = 11524; hi = 12800;
rollout = single(slick_challenge_predict(model, q(:, lo - 69:lo), hi - lo));
save(fullfile(out, 'slick_forecasts.mat'), 'test', 'cal', 'rollout', 'ours', 'calib', 'lo', 'hi', 'sz', '-v7.3');
fprintf('Saved forecasts to %s\n', fullfile(out, 'slick_forecasts.mat'));
end

function F = forecast(model, q, starts, horizon)
% (space, horizon, sequence) raw fields; starts are 0-based, each sequence gives 70 frames
F = zeros(size(q, 1), horizon, numel(starts), 'single');
for k = 1:numel(starts)
    s = starts(k) + 1;
    F(:, :, k) = single(slick_challenge_predict(model, q(:, s:s + 69), horizon));
end
end
