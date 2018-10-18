function [res, newim] = aba_MIRT_reg(fixed, rigid_aligned)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% Main settings
main.similarity = 'ssd';  % similarity measure, e.g. SSD, CC, SAD, RC, CD2, MS, MI
main.subdivide = 3;       % use 3 hierarchical levels
main.okno = 5;            % mesh window size
main.lambda = 0.005;      % transformation regularization weight, 0 for none
main.single = 0;          % show mesh transformation at every iteration

% Optimization settings
optim.maxsteps = 50;   % maximum number of iterations at each hierarchical level
optim.fundif = 1e-5;    % tolerance (stopping criterion)
optim.gamma = 1;        % initial optimization step size
optim.anneal=0.8;       % annealing rate on the optimization step

[res, newim]=mirt2D_register(fixed, rigid_aligned, main, optim);

end

