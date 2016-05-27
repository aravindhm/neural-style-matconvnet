function [x, fx, state] = adam(opfunc, x, config, state)
% [x, fx, state] = adam(opfunc, x, config, state) - ADAM 
% from http://arxiv.org/pdf/1412.6980.pdf
% Based on https://github.com/torch/optim/blob/master/adam.lua 
% Retrieved on [10:29, GMT, 24th May 2016]
%
% ARGS:
% opfunc  : a function that takes a single input (X), the point
%            of a evaluation, and returns f(X) and df/dX
% x       : the initial point
% config  : a table with configuration parameters for the optimizer
% config.learningRate       : learning rate
% config.beta1              : first moment coefficient
% config.beta2              : second moment coefficient
% config.epsilon            : for numerical stability
% state                     : a table describing the state of optimizer; 
%                             after each call the state is modified
% RETURN:
% x       : the new x vector
% f(x)    : the function, evaluated before the update
% state   : updated state

% (0) get/update state
if(~exist('config', 'var')), config = struct();  end
if(~exist('state', 'var')), state = config; end
lr = getConfig(config, 'learningRate', 1e-3);

beta1 = getConfig(config, 'beta1', 0.9);
beta2 = getConfig(config, 'beta2', 0.999);
epsilon = getConfig(config, 'epsilon', 1e-8);

% (1) evaluate f(x) and df/dx
[fx, dfdx] = opfunc(x);

% Initialization
state.t = getConfig(state, 't', 0);

% Exponential moving average of gradient values
state.m = getConfig(state, 'm', zeros(size(dfdx), 'like', x));

% Exponential moving average of squared gradient values
state.v = getConfig(state, 'v', zeros(size(dfdx), 'like', x));

% A tmp tensor to hold the sqrt(v) + epsilon
state.denom = getConfig(state, 'demon', zeros(size(dfdx), 'like', x));

state.t = state.t + 1;

% Update the loss history
state.loss_history(state.t) = fx;

% Decay the first and second moment running average coefficient
state.m = beta1 * state.m + (1-beta1) * dfdx;
state.v = beta2 * state.v + (1-beta2) * (dfdx.*dfdx);

state.denom = state.v.^0.5 + epsilon;

biasCorrection1 = 1 - beta1^state.t;
biasCorrection2 = 1 - beta2^state.t;
stepSize = lr * sqrt(biasCorrection2) / biasCorrection1;

% (2) Update x
x = x - stepSize * state.m ./ state.denom;

% Note that x, fx and the updated state are returned

% -------------------------------------------------------------------------
function val = getConfig(config, fieldname, default)
% -------------------------------------------------------------------------
if isfield(config, fieldname)
    val = config.(fieldname);
else
    val = default;
end