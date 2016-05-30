function neural_style(varargin)
% Perceptual loss based neural_style
% Use key value pairs in the arguments to specify the following options
%
% For default values see the code below (sorry :( )
% style_image - Image(s) to use for style ( ',' separated list )
% style_blend_weights - The relative weights for different style images
% style_weight - weight of style loss in objective
% style_scale - scaling for style images.
% style_layers - layers from which style loss statistics are computed
% (',' separated list)
%
% content_image - image from which content information is taken
% content_weight - weight of content loss in objective
% content_layers - layer from which content information is taken
%
% tv_weight - Weight of total variational norm (actually the quadratic norm)
% image_size - the content image will be rescaled such that the largest
%     dimension is image_size big. The result image is of the same size as this
% gpu - 0 indexed gpu number for gpu usage (NaN) for cpu mode
% pooling - replace max pooling layers with avg pooling by setting this to 'avg'
% model_file - the matconvnet model of the network to use for your experiment
% seed - seed for randomly initializing the image ( < 0 for no seed)
%
% dsloss_weight - weight of downsampling loss (for superresolution problems)
% upsampling_factor - upsampling involved in superresolution problems
% num_iterations - Number of iterations in the optimization
% normalize_gradients - Whether or not to normalize the gradients (disable this
%    when gradient checking)
% init - 'random'/'image' - random initialization / content image based initialization
% optimizer - 'lbfgs'/'adam' - currently only adam is supported
% learning_rate - Initial learning rate for adam optimization
% print_iter - Number of iterations after which to print loss values
% save_iter - Number of iterations after which to save the current image
% output_image - the output at each iteration will be decided based on this

% Setup matconvnet and vlfeat
run ~/toolboxes/vlfeat/toolbox/vl_setup.m
run ~/toolboxes/matconvnet/matlab/vl_setupnn.m

% Gradient checking.
% gtest(@tvloss_forward, @tvloss_backward, [32,32,7], [32,32,7]); % Get's a
% NaN in the end
% gtest(@contentloss_forward, @contentloss_backward, [32,32,7], [32,32,7]);
% gtest(@styleloss_forward, @styleloss_backward, [32,32,21], [21,21]);
% gtest(@dsloss_forward, @dsloss_backward, [32,32,3], [8,8,3], 'upsampling_factor', 4);
% return;

% Parse the arguments
% Style loss options
opts.style_image = 'examples/inputs/starry_night.jpg';
opts.style_blend_weights = '';
opts.style_weight = 1e2;
opts.style_scale = 1.0;
opts.style_layers = 'relu1_2,relu2_2,relu3_3,relu4_3';

% Content loss options
opts.content_image = 'examples/inputs/tubingen.jpg';
opts.content_weight = 5;
opts.content_layers = 'relu2_2';

% Misc options
opts.tv_weight = 1e-3;
opts.image_size = 512;
opts.gpu = 0;
opts.pooling = 'max';
opts.model_file = '~/data/nets/imagenet-vgg-verydeep-16.mat';
opts.seed = -1;

% Superresolution loss options
opts.dsloss_weight = 0;
opts.upsampling_factor = 4;

% Optimization options
opts.num_iterations = 1000;
opts.normalize_gradients = false;
opts.init = 'random';
opts.optimizer = 'adam';
opts.learning_rate = 1e1;

% Output options
opts.print_iter = 50;
opts.save_iter = 100;
opts.output_image = 'out.png';

[opts, varargin] = vl_argparse(opts, varargin); % Parse all the arguments into opts

% Load the CNN model
cnn = load(opts.model_file);
if isfield(cnn, 'net'), cnn = cnn.net; end
cnn = vl_simplenn_tidy(cnn);

% Get the mean pixel from the CNN as default value.
opts.mean_pixel = cnn.meta.normalization.averageImage;
opts = vl_argparse(opts, varargin); % If the mean_pixel is hard coded.

% Load the content image and resize it
content_image = imread(opts.content_image);
if(numel(opts.image_size) == 1)
    opts.image_rescale = opts.image_size / max(size(content_image));
end
content_image = imresize(content_image, opts.image_rescale, 'bilinear');
content_image_pp = preprocess(content_image, opts.mean_pixel);

% Load style image(s) and preprocess them
style_size = ceil(opts.style_scale * opts.image_size);
style_images_list = strsplit(opts.style_image, ',');
style_images_pp = cell(numel(style_images_list), 1);
for i=1:numel(style_images_list)
    img = imread(style_images_list{i});
    style_scale = style_size / max(size(img));
    img = imresize(img, style_scale, 'bilinear');
    style_images_pp{i} = preprocess(img, opts.mean_pixel);
end

% Handle style blending weights for multiple style inputs
if isempty(opts.style_blend_weights) % Uniform weights
    style_blend_weights = ones(numel(style_images_pp), 1, 'single');
else
    style_blend_weights = cellfun(@(x) str2num(x), ...
              strsplit(opts.style_blend_weights, ','));
    assert(numel(style_blend_weights) == numel(style_images_pp), ...
         'Number of Style blend weights is different from number of style images');
end

% L1 Normalize the style_blend_weights
style_blend_weights = style_blend_weights / sum(style_blend_weights);

% Which content layers and style layers from opts
content_layers = strsplit(opts.content_layers, ',');
style_layers = strsplit(opts.style_layers, ',');

% Setup the network using style and content loss modules
if(opts.content_weight > 0)
  contentlosses_index = zeros(numel(content_layers),1); % This will store the layer index for content loss
else
  contentlosses_index = [];
end
if(opts.style_weight > 0)
  stylelosses_index = zeros(numel(style_layers),1); % This will store the layer indexes for style losses
else
  stylelosses_index = [];
end
next_content_idx = 1; % To keep track of which content loss to add next
next_style_idx = 1; % To keep track of which style loss to add next

net.layers = {};
net.meta = cnn.meta;

% First add the TV norm loss
if opts.tv_weight > 0
    ly.name = 'tvloss';
    ly.type = 'custom';
    ly.precious = 0;
    ly.strength = opts.tv_weight;
    ly.forward = @tvloss_forward;
    ly.backward = @tvloss_backward;

    net.layers{end+1} = ly;

    tvloss_index = numel(net.layers);
else
    tvloss_index = NaN;
end

% Next the per pixel loss for downsampling
if opts.dsloss_weight > 0
    ly.name = 'dsloss';
    ly.type = 'custom';
    ly.normalize = opts.normalize_gradients;
    ly.strength = opts.dsloss_weight;
    ly.upsampling_factor = opts.upsampling_factor;
    ly.target = vl_nnpool(content_image_pp, opts.upsampling_factor, ...
        'stride', opts.upsampling_factor, 'Method', 'avg');
    ly.forward = @dsloss_forward;
    ly.backward = @dsloss_backward;

    net.layers{end+1} = ly;

    dsloss_index = numel(net.layers);
else
    dsloss_index = NaN;
end

% Next add the content and style losses
for i=1:numel(cnn.layers)
    if (next_content_idx <= numel(content_layers) && opts.content_weight > 0) ...
            || (next_style_idx <= numel(style_layers) && opts.style_weight > 0)
        ly = cnn.layers{i};
        name = ly.name;
        type = ly.type;

        % Check if it is a max pooling layer and whether I need to replace
        % it with an average pooling one.
        if(strcmp(type, 'pool') ...
                && strcmp(ly.method, 'max') ...
                && strcmp(opts.pooling, 'avg'))
            ly.method = 'avg';
            fprintf(1, 'Replacing max pooling at layer %d with average pooling\n', i);
            net.layers{end+1} = ly;
        else
            net.layers{end+1} = ly;
           % res_temp = vl_simplenn(net, content_image_pp);
        end

        % See if I should put a content layer here
        if next_content_idx <= numel(content_layers) ...
                && strcmp(name, content_layers{next_content_idx}) ...
                && opts.content_weight > 0
            fprintf(1, 'Setting up content layer %d: %s\n', i, name);

            % I've run the net forward to get the content target
            res = vl_simplenn(net, content_image_pp);
            ly.target = res(end).x;
            ly.name = [name, '_contentloss'];
            ly.type = 'custom';
            ly.strength = opts.content_weight;
            ly.normalize = opts.normalize_gradients;
            ly.forward = @contentloss_forward;
            ly.backward = @contentloss_backward;

            net.layers{end+1} = ly;

            contentlosses_index(next_content_idx) = numel(net.layers);
            next_content_idx = next_content_idx + 1;
        end

        % See if I should put a style layer here
        if next_style_idx <= numel(style_layers) ...
                && strcmp(name, style_layers{next_style_idx}) ...
                && opts.style_weight > 0
            fprintf(1, 'Setting up style layer %d: %s\n', i, name);

            % I've run the net forward to get the content target
            ly.target = 0;
            for style_images_pp_idx = 1:numel(style_images_pp)
              res = vl_simplenn(net, style_images_pp{style_images_pp_idx});
              target_i = grammatrix_forward(res(end).x);
              ly.target = ly.target + ...
                  (target_i * style_blend_weights(style_images_pp_idx) ) / numel(res(end).x) ;
            end

            ly.name = [name, '_styleloss'];
            ly.type = 'custom';
            ly.strength = opts.style_weight;
            ly.normalize = opts.normalize_gradients;
            ly.forward = @styleloss_forward;
            ly.backward = @styleloss_backward;

            net.layers{end+1} = ly;

            stylelosses_index(next_style_idx) = numel(net.layers);
            next_style_idx = next_style_idx + 1;
        end

    end
end % End for i=1:numel(cnn.layers)

% We are done with creating the network. Let's tidy it up.
net = vl_simplenn_tidy(net);

% We don't need to base cnn anymore, so clean it up to save memory
clear cnn;

% Initialize the image
if opts.seed >= 0
    rng(opts.seed);
end
if strcmp(opts.init, 'random')
    img = randn(size(content_image), 'single') * 0.001;
elseif strcmp(opts.init, 'image')
    img = content_image_pp;
else
    error('Invalid init type');
end

% Move to img and net to gpu.
if ~isnan(opts.gpu)
    g = gpuDevice(opts.gpu + 1);

    img = gpuArray(img);
    net = vl_simplenn_move(net, 'gpu');
    for i=1:numel(net.layers)
        if strcmp(net.layers{i}.type, 'custom')
            nms = fieldnames(net.layers{i});
            for j=1:numel(nms)
                if( isnumeric(net.layers{i}.(nms{j})) && numel(net.layers{i}.(nms{j})) > 1 )
                    net.layers{i}.(nms{j}) = gpuArray(net.layers{i}.(nms{j}));
                end
            end
        end
    end
end

% Precompute the size of the output from the last layer
res = vl_simplenn(net, img);
y = res(end).x;
sz_y = size(y);


% Optimization state
if strcmp(opts.optimizer, 'lbfgs')
    optim_config = struct('maxIter', opts.num_iterations, 'verbose', true);
elseif strcmp(opts.optimizer, 'adam')
    optim_config = struct('learningRate', opts.learning_rate);
elseif strcmp(opts.optimizer, 'sgd')
    optim_config = struct('learningRate', opts.learning_rate, 'momentum', 0.9);
else
    error('Unrecognized optimizer %s', opts.optimizer);
end
optim_state = struct();

global num_calls;
num_calls = 0; % The number of function calls

% Run the optimization
if strcmp(opts.optimizer, 'lbfgs')
    error('L-BFGS current unsupported');
elseif strcmp(opts.optimizer, 'adam')
    fprintf('Running optimization with ADAM\n');
    objfunc = @(x) feval(x, net, sz_y, contentlosses_index, stylelosses_index, ...
                            dsloss_index, tvloss_index, opts);
    for t = 1:opts.num_iterations
        [img, ~, optim_state] = optim.adam(objfunc, img, optim_config, optim_state);
        change_current_figure(2304);
        subplot(1,2,1);
          semilogy(optim_state.loss_history);
        subplot(1,2,2);
          img_cpu = gather(img);
          imshow(deprocess(img_cpu, opts.mean_pixel));
        drawnow;
    end
elseif strcmp(opts.optimizer, 'sgd')
    fprintf('Running optimization with SGD\n');
    objfunc = @(x) feval(x, net, sz_y, contentlosses_index, stylelosses_index, ...
                            dsloss_index, tvloss_index, opts);
    for t = 1:opts.num_iterations
        [img, ~, optim_state] = optim.sgd(objfunc, img, optim_config, optim_state);
        %change_current_figure(2304);
        %clf;
        subplot(1,2,1);
          semilogy(optim_state.loss_history);
        subplot(1,2,2);
          img_cpu = gather(img);
          imshow(deprocess(img_cpu, opts.mean_pixel));
        drawnow;
    end
end

% Save the final result
img = deprocess(img, opts.mean_pixel);
img = max(min(img, 255), 0);
imwrite(gather(img), opts.output_image);


% -------------------------------------------------------------------------
function img_pp = preprocess(img, mean_pixel)
% -------------------------------------------------------------------------
% Preprocess the image to be ready for input to the CNN
img_pp = bsxfun(@minus, single(img), mean_pixel);


% -------------------------------------------------------------------------
function img = deprocess(img_pp,mean_pixel)
% -------------------------------------------------------------------------
% Deprocess the image from network input to regular uint8 image
img = uint8(bsxfun(@plus, img_pp, mean_pixel));


% -------------------------------------------------------------------------
function maybe_print(t, loss, res, contentlosses_index, stylelosses_index, ...
                            dsloss_index, tvloss_index, opts)
% -------------------------------------------------------------------------
% This function is called when evaluating the objective function
% It prints the current value of loss function every few iterations

verbose = (opts.print_iter > 0) && mod(t, opts.print_iter) == 0;
if verbose
    fprintf(1, 'Iteration %d/%d %f\n', t, opts.num_iterations, loss);
    if(~isnan(dsloss_index))
        fprintf(1, '  DSLoss: %f\n', res(dsloss_index + 1).aux);
    end
    if(~isnan(tvloss_index))
        fprintf(1, '  TVLoss: %f\n', res(tvloss_index + 1).aux);
    end
    for i=1:numel(contentlosses_index)
        fprintf(1, '  Content %d Loss %f\n', i, ...
            res(contentlosses_index(i)+1).aux);
    end
    for i=1:numel(stylelosses_index)
        fprintf(1, '  Style %d Loss %f\n', i, ...
            res(stylelosses_index(i)+1).aux);
    end
    fprintf(1, 'Total Loss: %f\n', loss);
end


% -------------------------------------------------------------------------
function maybe_save(t, x, opts)
% -------------------------------------------------------------------------
should_save = (opts.save_iter > 0) && mod(t, opts.save_iter) == 0 ;
if should_save
    img = deprocess(x, opts.mean_pixel);
    img = max(min(img, 255), 0);
    filename = build_filename(opts.output_image, t);
    imwrite(gather(img), filename);
end

% -------------------------------------------------------------------------
function filename = build_filename(output_image, iteration)
% -------------------------------------------------------------------------
[folder, basename, ext] = fileparts(output_image);
filename = fullfile(folder, sprintf('%s_%d%s', basename, iteration, ext));


% -------------------------------------------------------------------------
function [f, dfdx] = feval(x, net, sz_y, contentlosses_index, stylelosses_index, ...
                            dsloss_index, tvloss_index, opts)
% -------------------------------------------------------------------------
% Function to evaluate loss and gradient. We run the net forward and
% backward to get the gradient, and sum up losses from the loss modules.
% optim.lbfgs internally handles iteration and calls this fucntion many
% times, so we manually count the number of iterations to handle printing
% and saving intermediate results.

% This is the number of times the function has been called. Sorry to hack
% this in as a global variable.
global num_calls ;
num_calls = num_calls + 1;

% Updating global num_calls to reflect the iteration count
res = vl_simplenn(net, x, zeros(sz_y, 'like', x));
% DZDY is set to zero because all the derivative come from the losses
% rather than the error at the end of the network.

% Get the gradient
dfdx = res(1).dzdx;

% Calculate the loss
f = 0;
% First the loss from content layers
for i=1:numel(contentlosses_index)
    f = f + res(contentlosses_index(i) + 1).aux;
end
% Then the loss from style layers
for i=1:numel(stylelosses_index)
    f = f + res(stylelosses_index(i) + 1).aux;
end
% DS loss layer's loss
if(~isnan(dsloss_index)), f = f + res(dsloss_index + 1).aux; end
% TV norm layer's loss
if(~isnan(tvloss_index)), f = f + res(tvloss_index + 1).aux; end

maybe_print(num_calls, f, res, contentlosses_index, stylelosses_index, ...
                            dsloss_index, tvloss_index, opts);
maybe_save(num_calls, x, opts);



% -------------------------------------------------------------------------
% In this section of the code I implement all the content and style loss
% functions to be used in the custom layers. Also the dsloss and tvnorm are
% implemented below.
%
% We need a forward and backward pass for each custom layer type.
% The only outlier is the GramMatrix which computes the GramMatrix of input
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
function resi_1 = contentloss_forward(ly, resi, resi_1)
% -------------------------------------------------------------------------
if numel(resi.x) == numel(ly.target)
    % MSECriterion
    resi_1.aux = sum( (resi.x(:) - ly.target(:)).^2 ) / numel(ly.target);
    % Multiply by appropriate weight for this loss
    resi_1.aux = resi_1.aux * ly.strength;
    % The pass through for the actual output
    resi_1.x = resi.x;
else
    warning('Skipping content loss');
    resi_1.x = resi.x;
end

% -------------------------------------------------------------------------
function resi = contentloss_backward(ly, resi, resi_1)
% -------------------------------------------------------------------------
if  numel(resi.x) == numel(ly.target)
    resi.dzdx = 2 / numel(ly.target) * (resi.x - ly.target);
    if ly.normalize
        resi.dzdx = resi.dzdx / ( norm(resi.dzdx(:)) + 1e-8 );
    end
    resi.dzdx = ly.strength * resi.dzdx;
    resi.dzdx = resi.dzdx + resi_1.dzdx;
else
    warning('Skipping content loss');
    resi.dzdx = resi_1.dzdx;
end

% -------------------------------------------------------------------------
function ccc = grammatrix_forward(x)
% -------------------------------------------------------------------------
% Function to compute the cross channel correlation from x
% Input: x is nRows x nCols x nChannels
% Result: ccc is nRows x nCols
x_reshaped = reshape(x, [size(x,1)*size(x,2), size(x,3)]);
ccc = x_reshaped' * x_reshaped;

% -------------------------------------------------------------------------
function dzdx = grammatrix_backward(x, dzdccc)
% -------------------------------------------------------------------------
% Function to compute derivative of the cross channel correlation op.
% Inputs: x is nRows x nCols x nChannels,
%         dzdccc is nRows x nCols
% Result: dzdx is nRows x nCols x nChannels
sz = size(x);
x = reshape(x, [], sz(3));
dzdx = reshape(x * dzdccc + x * dzdccc', sz);

% -------------------------------------------------------------------------
function resi_1 = styleloss_forward(ly, resi, resi_1)
% -------------------------------------------------------------------------
G = grammatrix_forward(resi.x);
G = G / numel(resi.x);
resi_1.aux = sum( (G(:) - ly.target(:)).^2 ) / numel(ly.target);
resi_1.aux = resi_1.aux * ly.strength;
resi_1.x = resi.x;

% -------------------------------------------------------------------------
function resi = styleloss_backward(ly, resi, resi_1)
% -------------------------------------------------------------------------
G = grammatrix_forward(resi.x);
G = G / numel(resi.x);

dG = 2 / numel(ly.target) * (G - ly.target) ;
dG = dG / numel(resi.x);

resi.dzdx = grammatrix_backward(resi.x, dG);
if ly.normalize
    resi.dzdx = resi.dzdx / ( norm(resi.dzdx(:)) + 1e-8 );
end
resi.dzdx = ly.strength * resi.dzdx;
resi.dzdx = resi.dzdx + resi_1.dzdx;

% -------------------------------------------------------------------------
function resi_1 = tvloss_forward(ly, resi, resi_1)
% -------------------------------------------------------------------------

% First compute the loss
% d1 = resi.x(:,[2:end end],:) - resi.x ;
% d2 = resi.x([2:end end],:,:) - resi.x ;

% v = (d1.*d1 + d2.*d2) ;
% resi_1.aux = sum(sum(sum(sum(v)))) * ly.strength;

xdiff = resi.x(1:end-1, 1:end-1, :) - resi.x(1:end-1, 2:end  , :) ;
ydiff = resi.x(1:end-1, 1:end-1, :) - resi.x(2:end  , 1:end-1, :) ;

v = xdiff.^2 + ydiff.^2;
resi_1.aux = 0.5*sum(v(:)) * ly.strength;

% Then put shortcut connection
resi_1.x = resi.x;

% -------------------------------------------------------------------------
function resi = tvloss_backward(ly, resi, resi_1)
% -------------------------------------------------------------------------

% First the derivative due to the tv norm
xdiff = resi.x(1:end-1, 1:end-1, :) - resi.x(1:end-1, 2:end  , :) ;
ydiff = resi.x(1:end-1, 1:end-1, :) - resi.x(2:end  , 1:end-1, :) ;

resi.dzdx = zeros(size(resi.x), 'like', resi.x);
resi.dzdx(1:end-1,1:end-1,:) = xdiff + ydiff;
resi.dzdx(1:end-1,2:end,:) = resi.dzdx(1:end-1,2:end,:) - xdiff;
resi.dzdx(2:end,1:end-1,:) = resi.dzdx(2:end,1:end-1,:) - ydiff;

resi.dzdx = ly.strength * resi.dzdx;

% Then add the derivative owning to the shortcut
resi.dzdx = resi.dzdx + resi_1.dzdx;

% -------------------------------------------------------------------------
function resi_1 = dsloss_forward(ly, resi, resi_1)
% -------------------------------------------------------------------------
small_input = vl_nnpool(resi.x, [ly.upsampling_factor, ly.upsampling_factor],...
    'stride', [ly.upsampling_factor, ly.upsampling_factor], 'Method', 'avg');
if(numel(small_input) ~= numel(ly.target))
    warning('Skipping the DSloss');
else
    resi_1.aux = sum( (small_input(:) - ly.target(:)).^2 ) / numel(ly.target);
    resi_1.aux = resi_1.aux * ly.strength;
end

% The shortcut connection
resi_1.x = resi.x;


% -------------------------------------------------------------------------
function resi = dsloss_backward(ly, resi, resi_1)
% -------------------------------------------------------------------------
small_input = vl_nnpool(resi.x, [ly.upsampling_factor, ly.upsampling_factor],...
    'stride', [ly.upsampling_factor, ly.upsampling_factor], 'Method', 'avg');

gradCrit = 2/numel(ly.target) * (small_input - ly.target) * ly.strength ;
resi.dzdx = vl_nnpool(resi.x, [ly.upsampling_factor, ly.upsampling_factor],...
    gradCrit, 'stride', [ly.upsampling_factor, ly.upsampling_factor], 'Method', 'avg');
resi.dzdx = resi.dzdx + resi_1.dzdx;


% -------------------------------------------------------------------------
function change_current_figure(fig_id)
% -------------------------------------------------------------------------
% To change the figure without shifting focus everytime.
try
    set(0, 'CurrentFigure', fig_id);
catch
    figure(fig_id);
end



% -------------------------------------------------------------------------
function gtest(forward, backward, inputSize, targetSize, varargin)
% -------------------------------------------------------------------------
% This function does gradient testing for the new custom layers
% Input: forward  - function handle to the forward function call
%        backward - function handle to the backward function call
%        inputSize - size of the input to these functions
%        targetSize - size of the target as these are actually loss layers

ly = struct(varargin{:});
ly.normalize = false;
ly.strength = 6;
resi.x = randn(inputSize, 'double') *10;
%resi.x = reshape(single(1:prod(inputSize)), inputSize);
ly.target = randn(targetSize, 'double') *10;
%ly.target = vl_nnpool(resi.x, [2, 2], 'stride', 2, 'Method', 'avg');

delta_x = 1e-2;

% These are all pass through layers so the outputSize == inputSize
resi_1.dzdx = randn(inputSize, 'single');

for i=1:numel(resi.x)
    % f(x+delta_x)
    resi.x(i) = resi.x(i) + delta_x;
    resi_1 = forward(ly, resi, resi_1);
    y1 = resi_1.aux;

    % f(x-delta_x)
    resi.x(i) = resi.x(i) - 2*delta_x;
    resi_1 = forward(ly, resi, resi_1);
    y2 = resi_1.aux;

    % Numerical estimate = (f(x+delta_x) - f(x-delta_x)) / (2*delta_x)
    numerical_estimate = resi_1.dzdx(i) + (y1 - y2) / (2*delta_x);

    % Analytical estimate
    resi.x(i) = resi.x(i) + delta_x;
    resi = backward(ly, resi, resi_1);
    analytical_estimate = resi.dzdx(i);
    assert((numerical_estimate - analytical_estimate) / (abs(numerical_estimate) + abs(analytical_estimate)) < 1e-4, '%f - %f = %f: Relative: %f\n', numerical_estimate, analytical_estimate, ...
        numerical_estimate - analytical_estimate, (numerical_estimate - analytical_estimate) / (abs(numerical_estimate) + abs(analytical_estimate)));
%    fprintf('%f - %f = %f: Relative: %f\n', numerical_estimate, analytical_estimate, ...
%        numerical_estimate - analytical_estimate, (numerical_estimate - analytical_estimate) / (abs(numerical_estimate) + abs(analytical_estimate)));
end


