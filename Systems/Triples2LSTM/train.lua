----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree.
----
----  Parts of this code are adapted from: https://github.com/wojzaremba/lstm/blob/master/main.lua
----

local params = {
    -- The path to the folder with all the required dataset-related files.
    -- Modify this variable according to the dataset with which you would like to
    -- train (i.e. D1 or D2), and whether you want to train with surface form
    -- tuples (with-Surface-Form-Tuples) or URIs (with-URIs)

    dataset_path = '../../D1/processed/with-Surface-Form-Tuples/',
    -- dataset_path = '../../D2/processed/with-URIs/',


    -- Periodically (i.e. after the sixth epoch, every half an epoch), this file
    -- will be saving checkpoints of the trained model.
    -- IMPORTANT: For consistency and future reference reasons, make sure that
    -- the checkpoint path's parts that refer to the training dataset
    -- (i.e. D1/ or D2/) and the selected training setup (surf_form_tuples.model
    -- or uris.model) are the same with the ones that have been set at the
    -- dataset_path variable.

    checkpoint_path = './checkpoints/D1/epoch_%.2f.error_%.4f.surf_form_tuples.model.t7', -- use when dataset_path = '../../D1/processed/with-Surface-Form-Tuples/'
    -- checkpoint_path = './checkpoints/D2/epoch_%.2f.error_%.4f.uris.model.t7', -- use when dataset_path = '../../D2/processed/with-URIs/'

    batch_size = 85,
    layers = 1,
    decay = 0.8,
    rnn_size = 650,
    dropout = 0.0,
    init_weight = 0.001,
    learningRate = 2e-3,
    max_epoch = 12,
    gpuidx = 1
}


package.path = '../utils/?.lua;' .. package.path
local ok1, cunn = pcall(require, 'cunn')
local ok2, cutorch = pcall(require, 'cutorch')
if not (ok1 and ok2) then
    print('Warning: Either cunn or cutorch was not found. Falling gracefully to CPU...')
    params.gpuidx = 0
    pcall(require, 'nn')
else
    cutorch.setDevice(params.gpuidx)
    print(string.format("GPU: %d", params.gpuidx) .. string.format(' out of total %d is being currently used.', cutorch.getDeviceCount()))
end

require('nngraph')
require('optim')
require('utilities')
dataset = require('dataset')
require('LookupTableMaskZero')
require('MaskedClassNLLCriterionInheritingNLLCriterion')

-- Adapted from: https://github.com/wojzaremba/lstm
local function lstm(x, prev_c, prev_h)
    local n2i                   = nn.BatchNormalization(4 * params.rnn_size, 1e-5, 0.1, true)
    local n2h                   = nn.BatchNormalization(4 * params.rnn_size, 1e-5, 0.1, true)
    local i2h                   = n2i(nn.Linear(params.rnn_size, 4 * params.rnn_size)(x))
    local h2h                   = n2h(nn.Linear(params.rnn_size, 4 * params.rnn_size)(prev_h))
    local gates                 = nn.CAddTable()({i2h, h2h})
    
    local reshapedGates         = nn.Reshape(4, params.rnn_size)(gates)
    local splitGates            = nn.SplitTable(2)(reshapedGates)
   
    local in_gate               = nn.Sigmoid()(nn.SelectTable(1)(splitGates))
    local input                 = nn.Tanh()(nn.SelectTable(2)(splitGates))
    local forget_gate           = nn.Sigmoid()(nn.SelectTable(3)(splitGates))
    local out_gate              = nn.Sigmoid()(nn.SelectTable(4)(splitGates))
   
    local next_c                = nn.CAddTable()({
            nn.CMulTable()({forget_gate, prev_c}),
            nn.CMulTable()({in_gate,     input})
                                                })
    local next_h                = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
   
    return next_c, next_h
end


local function create_encoder()
    local x                     = nn.Identity()() 
    local i                     = {}
    i[0]                        = nn.Reshape(params.numAlignedTriples * params.batch_size, 3 * params.rnn_size, false)(
        nn.LookupTableMaskZero(params.source_vocab_size, params.rnn_size)(x))

    local n2i                   = nn.BatchNormalization(params.rnn_size, 1e-5, 0.1, true)
    
    i[1]                        = nn.ReLU()(n2i(nn.Linear(3 * params.rnn_size, params.rnn_size, false)(nn.Dropout(params.dropout)(i[0]))))
   for layeridx = 2, params.layers do
       i[layeridx] = nn.ReLU()(nn.BatchNormalization(params.rnn_size, 1e-5, 0.1, true)(
				   nn.Linear(params.rnn_size, params.rnn_size, false)(
				       nn.Dropout(params.dropout)(i[layeridx - 1]))))
   end
   local tripleEmbeddings       = nn.SplitTable(1)(
       {nn.Reshape(params.numAlignedTriples, params.batch_size, params.rnn_size, false)(i[params.layers])})
   local triplesConcat          = nn.JoinTable(2)(tripleEmbeddings)
   local output                 = nn.Linear(params.numAlignedTriples * params.rnn_size, params.rnn_size)(triplesConcat)
   local module                 = nn.gModule({x}, {output})
   
   return module
end



local function create_decoder()
    local x                     = nn.Identity()()
    local prev_state            = nn.Identity()()

    local i                     = {}

    i[0]                        = nn.LookupTableMaskZero(params.target_vocab_size, params.rnn_size)(x)
    local next_state            = {}
    local splitPrevState        = {prev_state:split(2 * params.layers)}
    for layeridx = 1, params.layers do
        local prev_c            = splitPrevState[2 * layeridx - 1]
        local prev_h            = splitPrevState[2 * layeridx]
        local next_c, next_h    = lstm(i[layeridx - 1], prev_c, prev_h)
        table.insert(next_state, next_c)
        table.insert(next_state, next_h)
        i[layeridx] = next_h
    end
    
    
    local h1y                   = nn.Linear(params.rnn_size, params.target_vocab_size)
    local dropped               = nn.Dropout(params.dropout)
    local pred                  = nn.LogSoftMax()(h1y(dropped(i[params.layers])))
    
    local module                = nn.gModule({x, prev_state}, {pred, nn.Identity()(next_state)})
    return module

end

local function setup()
   
    print("Creating a Triples2LSTM network...")
    local encoderNetwork = transfer_to_gpu(create_encoder(), params.gpuidx)
    local decoderNetwork = transfer_to_gpu(create_decoder(), params.gpuidx)
    
    x, dx = combine_all_parameters(encoderNetwork, decoderNetwork)
    x:uniform(-params.init_weight, params.init_weight)

    encoder = {}
    decoder = {}
    encoder.network = encoderNetwork
    decoder.network = decoderNetwork
    decoder.rnns = cloneManyTimes(decoderNetwork, params.timesteps)
    decoder.s = {}
    decoder.pred = {}
    decoder.ds = {}

    encoder.s = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
    encoder.ds = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)

    for j = 0, params.timesteps do
        decoder.s[j] = {}
        for d = 1, 2 * params.layers do
            decoder.s[j][d] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
        end
    end
    
    for j = 0, params.timesteps do
        decoder.pred[j] = transfer_to_gpu(torch.zeros(params.batch_size, params.target_vocab_size), params.gpuidx)
    end
    
    for d = 1, 2 * params.layers do
        decoder.ds[d] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
    end

    local criterion = transfer_to_gpu(nn.MaskedClassNLLCriterion(), params.gpuidx)
    decoder.criterion = criterion
    decoder.criterions = cloneManyTimes(criterion, params.timesteps)
    
    decoder.err = transfer_to_gpu(torch.zeros(params.timesteps), params.gpuidx)
    collectgarbage()
    collectgarbage()
end

local function reset_state(state)
    state.batchidx = 1
    print('State: ' .. string.format("%s", state.name) .. ' has been reset.')
end


local function forward(state, x_new)
    
    if x ~= x_new then x:copy(x_new) end
    
    if state.batchidx > state.triples:size(1) then reset_state(state) end
    
    local batchTriples = state.triples[{state.batchidx, {}, {}, {}}]:reshape(params.batch_size * params.numAlignedTriples, 3)
    encoder.s = encoder.network:forward(batchTriples)
    if params.gpuidx > 0 then cutorch.synchronize() end
    
    -- We initialise the decoder.
    for d = 1, #decoder.s[0] do 
        if d == 2 then decoder.s[0][2]:copy(encoder.s)
        else decoder.s[0][d]:zero() end
    end

    for i = 1, params.timesteps do
	
        decoder.pred[i], decoder.s[i] = unpack(decoder.rnns[i]:forward({state.summaries[{state.batchidx, {}, i}],
									decoder.s[i - 1]}))
        if params.gpuidx > 0 then cutorch.synchronize() end
        decoder.err[i] = decoder.criterions[i]:forward(decoder.pred[i], state.summaries[{state.batchidx, {}, i + 1}])
        if params.gpuidx > 0 then cutorch.synchronize() end
        
    end

    local validBatches = state.summaries[{state.batchidx, {}, {2, params.timesteps + 1}}]:ne(0):eq(1):sum(1)
    local mask = validBatches:ne(0)
    
    return torch.cdiv(decoder.err[mask]:contiguous():typeAs(decoder.err), validBatches[mask]:contiguous():typeAs(decoder.err)):mean()
    
end


local function backward(state)

    dx:zero()
    -- Reset gradients
    -- Gradients are always accumulated to accomodate batch methods.

    for d = 1, #decoder.ds do decoder.ds[d]:zero() end
    encoder.ds:zero()
    
    for i = params.timesteps, 1, -1 do
        
        local tempPrediction = decoder.criterions[i]:backward(decoder.pred[i],
							      state.summaries[{state.batchidx, {}, i + 1}])
        local tempDecoder = decoder.rnns[i]:backward({state.summaries[{state.batchidx, {}, i}],
						      decoder.s[i - 1]}, {tempPrediction, decoder.ds})[2]
        copyTable(tempDecoder, decoder.ds)
        if params.gpuidx > 0 then cutorch.synchronize() end
    end
    encoder.ds:copy(decoder.ds[2])
    assert(encoder.ds:norm() > 1e-10)
    local batchTriples = state.triples[{state.batchidx, {}, {}, {}}]:reshape(params.batch_size * params.numAlignedTriples, 3)
    encoder.network:backward(batchTriples, encoder.ds)
    state.batchidx = state.batchidx + 1

end


local function feval(x_new)
    forward(training, x_new)
    backward(training)
    return 0, dx
end

-- Evaluate by computing perplexity on either the validation set.
local function evaluate(state)
    reset_state(state)
    
    encoder.network:evaluate()
    decoder.network:evaluate()
    for j = 1, #decoder.rnns do decoder.rnns[j]:evaluate() end

    local perplexity = 0

    while state.batchidx <= state.triples:size(1) do
        perplexity = perplexity + forward(state, x)
        print (string.format("%d", state.batchidx).. '\t/ '.. string.format("%d", state.triples:size(1)))
        state.batchidx = state.batchidx + 1
        collectgarbage()
        collectgarbage()
    end
    perplexity = torch.exp(perplexity / state.triples:size(1))
    print('Validation Set Perplexity: ' .. string.format("%.4f", perplexity))

    encoder.network:training()
    decoder.network:training()
    for j = 1, #decoder.rnns do decoder.rnns[j]:training() end

    return perplexity

end

local function main()
    
    sgd_params = {
	learningRate = params.learningRate,
	-- learningRateDecay = 0,
	weightDecay = 0.2,
	momentum = 0
    }

    rmsprop_params = {
	learningRate = params.learningRate,
	weightDecay = 0.1,
	alpha = 0.9
    }
    -- Initialising the dataset.
    dataset.init_dataset(params.dataset_path)
    
    local triples_dictionary = dataset.triples_dictionary()
    local summaries_dictionary = dataset.summaries_dictionary()
    assert(length(triples_dictionary['item2id']) == length(triples_dictionary['id2item']))
    assert(length(summaries_dictionary['word2id']) == length(summaries_dictionary['id2word']))
    params.source_vocab_size = length(triples_dictionary['item2id']) - 1
    params.target_vocab_size = length(summaries_dictionary['word2id']) - 1
    params.numAlignedTriples = triples_dictionary['max_num_triples']
    
    training = {
	triples = transfer_to_gpu(dataset.train_triples(params.numAlignedTriples, params.batch_size), params.gpuidx),
	summaries = transfer_to_gpu(dataset.train_summaries(params.batch_size), params.gpuidx),
	name = 'Training'
    }
    validation = {
	triples = transfer_to_gpu(dataset.validate_triples(params.numAlignedTriples, params.batch_size), params.gpuidx),
	summaries = transfer_to_gpu(dataset.validate_summaries(params.batch_size), params.gpuidx),
	name = 'Validation'
    }
    testing = {
	triples = transfer_to_gpu(dataset.test_triples(params.numAlignedTriples, params.batch_size), params.gpuidx),
	summaries = transfer_to_gpu(dataset.test_summaries(params.batch_size), params.gpuidx),
	name = 'Testing'
    }


    assert(training.summaries:size(3) == validation.summaries:size(3))
    assert(training.summaries:size(3) == testing.summaries:size(3))
    -- The number of timesteps that we unroll for.
    params.timesteps = training.summaries:size(3) - 1
    
    reset_state(training)
    reset_state(validation)
    reset_state(testing)


    local details = {
	params = params,
	epoch = 0, 
	err = 0
    }
 
    print('Network Parameters')
    print(params)
    setup()
    print_gpu_usage(params.gpuidx)
    

    local epoch_size = training.triples:size(1)

    local step = details.epoch * epoch_size
    
    validationErrors = {}
    while details.epoch < params.max_epoch do
	optim.rmsprop(feval, x, rmsprop_params)
	step = step + 1
	details.epoch = step / epoch_size
	print (string.format("%d", step).. '\t/ '.. string.format("%d", epoch_size).. ': '.. string.format('Training loss is %.2f.', decoder.err:sum()))
	if step % torch.round(epoch_size / 2) == 0 then
	    details.err = evaluate(validation)
	    table.insert(validationErrors, details.err)
	    print('Below are the Validation Errors for every half training epoch until now: ')	
	    print(validationErrors)
	    if details.epoch >= 1  then 
		encoder.network:clearState()
		for j = 1, #decoder.rnns do decoder.rnns[j]:clearState() end
		local tempEncoder = {}
		local tempDecoder = {}
		tempEncoder.network = encoder.network
		tempDecoder.network = decoder.network
		tempDecoder.rnns = decoder.rnns
		-- Saving a checkpoint file of the trained model in the ./checkpoints directory.
		generateCheckpoint(params.checkpoint_path, {encoder = tempEncoder, decoder = tempDecoder}, details)
	    end
	end
                
	if details.epoch >= 2 then
	    if step % torch.round(epoch_size) == 0 then
		rmsprop_params.learningRate = rmsprop_params.learningRate * params.decay
		params.learningRate = rmsprop_params.learningRate
		print('Learning rate has changed to '.. string.format("%f. Carrying on with the training...", rmsprop_params.learningRate))
	    end
	end
	
	if step % 33 == 0 then
	    collectgarbage()        
	    collectgarbage()
	end
    end
    
    print('Training has been completed.')
end

main()
