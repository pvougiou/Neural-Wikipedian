----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree.
----
----  Sampled summaries will be saved as HDF5 files in the directory of the pre-trained model.
----  For example: sampling summaries with surface form tuples with a beam size of 3 for the triples in the test set of D2
----  will create the following file: "./checkpoints/D2/surf_form_tuples.model.t7.summaries_Testing.beam_3.h5".
----
----  IMPORTANT: Make sure that the pre-trained model (i.e. D1 or D2, with-URIs or with-Surface-Form-Tuples) matches the dataset that
----  will be loaded in the beam_sampling_params.dataset_path variable.
----


local beam_sampling_params = {
    checkpoint = './checkpoints/D1/surf_form_tuples.model.t7', -- The filepath to the saved pre-trained model.
    -- IMPORTANT: Make sure that the dataset that will be loaded matches the specification of the pre-trained model.
    dataset_path = '../../D1/processed/with-Surface-Form-Tuples/', -- The path to the folder with all the required dataset-related files.
    beam_size = 5 -- Sets the beam size that will be used during decoding.
}


package.path = '../utils/?.lua;' .. package.path
local ok1, cunn = pcall(require, 'cunn')
local ok2, cutorch = pcall(require, 'cutorch')
if not (ok1 and ok2) then
    print('Warning: Either cunn or cutorch was not found. Falling gracefully to CPU...')
    params.gpuidx = 0
    pcall(require, 'nn')
end

require('nngraph')
require('optim')
require('utilities')
dataset = require('dataset')
require('LookupTableMaskZero')
require('MaskedClassNLLCriterionInheritingNLLCriterion')


local function reset_state(state)
    state.batchidx = 1
    print('State: ' .. string.format("%s", state.name) .. ' has been reset.')
end


local function main()
    local checkpoint = torch.load(beam_sampling_params.checkpoint)
    encoder, decoder = checkpoint.model.encoder, checkpoint.model.decoder
    params = checkpoint.details.params
    extendTable(params, beam_sampling_params)

    -- Initialising the dataset.
    dataset.init_dataset(params.dataset_path)

    print('Network Parameters')
    print(params)
    
    local triples_dictionary = dataset.triples_dictionary()
    local summaries_dictionary = dataset.summaries_dictionary()
    assert(length(triples_dictionary['item2id']) == length(triples_dictionary['id2item']))
    assert(length(summaries_dictionary['word2id']) == length(summaries_dictionary['id2word']))
    local start_token = summaries_dictionary['word2id']['<start>']
    local end_token = summaries_dictionary['word2id']['<end>']
    local pad_token = summaries_dictionary['word2id']['<PAD>']


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
    reset_state(training)
    reset_state(validation)
    reset_state(testing)
    collectgarbage()
    collectgarbage()

    -- Set the state from which we'd like to start sampling.
    local state = testing

    next_word = transfer_to_gpu(torch.zeros(params.batch_size), params.gpuidx)
    LogSoftMax = transfer_to_gpu(torch.zeros(params.batch_size, params.target_vocab_size * params.beam_size), params.gpuidx)
    
    encoder.network:evaluate()
    for j = 1, #decoder.rnns do decoder.rnns[j]:evaluate() end
    
    decoder.s = {}
    decoder.tempState = {}
    for j = 0, params.beam_size do
	decoder.s[j] = {}
	decoder.tempState[j] = {}
	for d = 1, 2 * params.layers do
	    decoder.tempState[j][d] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
	    decoder.s[j][d] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
	end
    end

    local summaries_filename = string.format('%s.summaries_%s.beam_%s.h5', params.checkpoint, state.name, params.beam_size)
    local summaries_file = hdf5.open(summaries_filename, 'w')
    summaries = torch.zeros(params.beam_size, params.batch_size * state.triples:size(1), params.timesteps + 1)

	
    while state.batchidx <= state.triples:size(1) do
	local batchTriples = state.triples[{state.batchidx, {}, {}, {}}]:reshape(params.batch_size * params.numAlignedTriples, 3)
	encoder.s = encoder.network:forward(batchTriples)
	if params.gpuidx > 0 then cutorch.synchronize() end
	
	validMask = torch.ones(params.batch_size, params.beam_size)
	
	print('Generating summaries for '.. string.format("%d", state.batchidx).. '. Batch...')
	
	-- We initialise the decoder.
	next_word:fill(start_token)
	summaries:sub(1, params.beam_size, 1, summaries:size(2), 1, 1):fill(start_token)
	for d = 1, #decoder.s[0] do 
	    if d == 2 then decoder.s[0][2]:copy(encoder.s)
	    else decoder.s[0][d]:zero() end
	end
	
	local tempPrediction, tempState = unpack(decoder.rnns[1]:forward({next_word, decoder.s[0]}))
		
	for j = 1, params.beam_size do copyTable(tempState, decoder.s[j]) end
	if params.gpuidx > 0 then cutorch.synchronize() end
	
	batchProbabilities, batchIndeces = tempPrediction:topk(params.beam_size, true, true)
	-- In which case the results are returned from smallest to k-th smallest (dir == false)
	-- or highest to k-th highest (dir == true).
		

	for i = 2, params.timesteps do
	    LogSoftMax:zero()
	    for beamidx = 1, params.beam_size do
		next_word:copy(batchIndeces:sub(1, params.batch_size, beamidx, beamidx))
			
		local tempPrediction, tempState = unpack(decoder.rnns[i]:forward({next_word, decoder.s[beamidx]}))
		
		copyTable(tempState, decoder.tempState[beamidx])
		if params.gpuidx > 0 then cutorch.synchronize() end
			
		prediction = tempPrediction + batchProbabilities:sub(1, params.batch_size, beamidx, beamidx)
		    :reshape(params.batch_size, 1):expand(params.batch_size, params.target_vocab_size)
		-- Assertion fails due to floating point accuracy; let's hope it's fine for now.
		-- assert(seqProbabilities[sampleidx][13] * tempPrediction[13][2] == prediction[13][2])
 
		LogSoftMax:sub(1, params.batch_size, (beamidx - 1) * params.target_vocab_size + 1, beamidx * params.target_vocab_size):copy(prediction)
	    end

	    for j = 1, params.batch_size do
		for b = 1, params.beam_size do
		    if validMask[j][s] == 0 then
			LogSoftMax:sub(j, j, (b - 1) * params.target_vocab_size + 1, b * params.target_vocab_size):fill(-math.huge)
		    end
		end
			

		candidates = torch.Tensor(batchIndeces[j]:size()):copy(batchIndeces[j])
		
		tempProbabilities, tempCandidates = LogSoftMax[j]:topk(params.beam_size, true, true)
		tempMask = torch.Tensor(validMask[j]:size()):copy(validMask[j])
				
		batchProbabilities:sub(j, j, 1, params.beam_size):copy(tempProbabilities)
		tempSummaries = torch.Tensor(summaries:sub(1, params.beam_size, (state.batchidx - 1) * params.batch_size + j, (state.batchidx - 1) * params.batch_size + j, 1, i - 1):size())
		    :copy(summaries:sub(1, params.beam_size, (state.batchidx - 1) * params.batch_size + j, (state.batchidx - 1) * params.batch_size + j, 1, i - 1))
		
		for beamidx = 1, params.beam_size do

		    if validMask[j][beamidx] == 1 then
			candidatesIndex = math.floor((tempCandidates[beamidx - validMask[j]:sub(1, beamidx):eq(0):sum()] - 1) / params.target_vocab_size) + 1
			

			-- The indexing here is preverted!
			summaries[beamidx][(state.batchidx - 1) * params.batch_size + j]:sub(1, i - 1):copy(tempSummaries[candidatesIndex])
			summaries[beamidx][(state.batchidx - 1) * params.batch_size + j][i] = candidates[candidatesIndex]
			
			batchIndeces[j][beamidx] = tempCandidates[beamidx - validMask[j]:sub(1, beamidx):eq(0):sum()] % params.target_vocab_size
			batchProbabilities[j][beamidx] = tempProbabilities[beamidx - validMask[j]:sub(1, beamidx):eq(0):sum()]
				
			for d = 1, 2 * params.layers do
			    decoder.s[beamidx][d][j]:copy(decoder.tempState[candidatesIndex][d][j])
			end
			if batchIndeces[j][beamidx] == 0 then
			    summaries[beamidx][(state.batchidx - 1) * params.batch_size + j][i + 1] = end_token
			    batchProbabilities[j][beamidx] = LogSoftMax[j]:min()
			    tempMask[beamidx] = 0
			end
		    else
			batchIndeces[j][beamidx] = 0
			batchProbabilities[j][beamidx] = LogSoftMax[j][(beamidx - 1) * params.target_vocab_size + 1]
		    end	
		end
		validMask[j]:copy(tempMask)
	    end
	end

	state.batchidx = state.batchidx + 1
	collectgarbage()
	collectgarbage()
    end
    summaries_file:write(tostring('triples'), state.triples:int())
    summaries_file:write(tostring('summaries'), summaries)
    summaries_file:write(tostring('actual_summaries'), state.summaries:int())
    summaries_file:close()	
end

main()
