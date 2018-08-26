----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree.
----
----  Sampled summaries will be saved as HDF5 files in the directory of the pre-trained model.
----  For example: sampling summaries with surface form tuples with a beam size of 3 for the triples in the test set of D2
----  will create the following file: "./checkpoints/D2/surf_form_tuples.model.t7.batch_size_85.beam_size_3.summaries_Testing.h5".
----
----  IMPORTANT: Make sure that the pre-trained model (i.e. D1 or D2, with-URIs or with-Surface-Form-Tuples) matches the dataset that
----  will be loaded in the beam_sampling_params.dataset_path variable.
----


local beam_sampling_params = {
    checkpoint = './checkpoints/D1/surf_form_tuples.model.t7', -- The filepath to the saved pre-trained model.
    -- IMPORTANT: Make sure that the dataset that will be loaded matches the specification of the pre-trained model.
    dataset_path = '../../D1/processed/with-Surface-Form-Tuples/', -- The path to the folder with all the required dataset-related files.
    beam_size = 5, -- Sets the beam size that will be used during decoding.
    -- IMPORTANT: A model that has been trained on a GPU should be loaded in a GPU; a model that has been trained
    -- on the CPU should be loaded in the CPU. Use the variable below to set the ID of the GPU on which
    -- you would like to load a model that has been training on a GPU.
    -- In case the model has been trained using the CPU, please set beam_sampling_params.gpuidx to 0.
    gpuidx = 1
}


package.path = '../utils/?.lua;' .. package.path
local ok1, cunn = pcall(require, 'cunn')
local ok2, cutorch = pcall(require, 'cutorch')
if not (ok1 and ok2) then
    print('Warning: Either cunn or cutorch was not found. Falling gracefully to CPU...')
    beam_sampling_params.gpuidx = 0
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


local function errorHandler(err)
    if string.find(err, 'unknown Torch class <torch.CudaTensor>') and beam_sampling_params.gpuidx == 0 then
	print("You cannot sample from a model that has been trained on the GPU without cunn and cutorch.")
    else
	print(err)
    end
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
    params.num_aligned_triples = triples_dictionary['max_num_triples']
    params.source_vocab_size = length(triples_dictionary['item2id']) - 1
    params.target_vocab_size = length(summaries_dictionary['word2id']) - 1

    local start_token = summaries_dictionary['word2id']['<start>']
    local end_token = summaries_dictionary['word2id']['<end>']
    local pad_token = summaries_dictionary['word2id']['<PAD>']

    
    training = {
	triples = transfer_to_gpu(dataset.train_triples(params.num_aligned_triples, params.batch_size), params.gpuidx),
	summaries = transfer_to_gpu(dataset.train_summaries(params.batch_size), params.gpuidx),
	name = 'Training'
    }
    validation = {
	triples = transfer_to_gpu(dataset.validate_triples(params.num_aligned_triples, params.batch_size), params.gpuidx),
	summaries = transfer_to_gpu(dataset.validate_summaries(params.batch_size), params.gpuidx),
	name = 'Validation'
    }
    testing = {
	triples = transfer_to_gpu(dataset.test_triples(params.num_aligned_triples, params.batch_size), params.gpuidx),
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



    encoder.network:evaluate()
    for j = 1, #decoder.rnns do decoder.rnns[j]:evaluate() end


    decoder.s = {}
    decoder.tempState = {}


    for j = 0, params.beam_size do
	decoder.s[j] = {}
	decoder.tempState[j] = {}
	if params.layers == 1 then
	    decoder.tempState[j] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
	    decoder.s[j] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
	else
	    for d = 1, params.layers do
		decoder.tempState[j][d] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
		decoder.s[j][d] = transfer_to_gpu(torch.zeros(params.batch_size, params.rnn_size), params.gpuidx)
	    end
	end
    end


    next_word = transfer_to_gpu(torch.zeros(params.batch_size), params.gpuidx)
    candidates = torch.zeros(params.beam_size, params.batch_size * state.triples:size(1), params.timesteps + 1)
    beam_probabilities = transfer_to_gpu(torch.zeros(params.batch_size, params.target_vocab_size * params.beam_size), params.gpuidx):fill(-math.huge)
    summaries = torch.zeros(params.beam_size, params.batch_size * state.triples:size(1), params.timesteps + 1)
    probabilities = torch.zeros(params.beam_size, params.batch_size * state.triples:size(1)):fill(-math.huge)
    
    while state.batchidx <= state.triples:size(1) do
	local batchTriples = state.triples[{state.batchidx, {}, {}, {}}]:reshape(params.batch_size * params.num_aligned_triples, 3)

	encoder.s = encoder.network:forward(batchTriples)
	if params.gpuidx > 0 then cutorch.synchronize() end
	
	is_beam_active = torch.ByteTensor(params.batch_size, params.beam_size):fill(1)
	
	print('Generating summaries for '.. string.format("%d", state.batchidx).. '. Batch...')
	
	-- We initialise the decoder.
	next_word:fill(start_token)
	candidates:sub(1, params.beam_size, 1, candidates:size(2), 1, 1):fill(start_token)
	if params.layers == 1 then
	    decoder.s[0]:copy(encoder.s)
	else
	    for d = 1, #decoder.s[0] do 
		if d == 1 then decoder.s[0][1]:copy(encoder.s)
		else decoder.s[0][d]:zero() end
	    end
	end
	
	local tempPrediction, tempState = unpack(decoder.rnns[1]:forward({next_word,
									  decoder.s[0]}))
	
	for beamidx = 1, params.beam_size do
	    if params.layers == 1 then decoder.s[beamidx]:copy(tempState)
	    else copyTable(tempState, decoder.s[beamidx]) end
	end
	if params.gpuidx > 0 then cutorch.synchronize() end

	
	-- In which case the results are returned from smallest to k-th smallest (dir == false)
	-- or highest to k-th highest (dir == true).
	batch_probabilities, batch_indices = tempPrediction:topk(params.beam_size, true, true)


	for j = 2, params.timesteps do    
	    beam_probabilities:fill(-math.huge)
	    for beamidx = 1, params.beam_size do
		next_word:copy(batch_indices:sub(1, params.batch_size, beamidx, beamidx))
		
		
		local tempPrediction, tempState = unpack(decoder.rnns[j]:forward({next_word,
										  decoder.s[beamidx]}))

		if params.layers == 1 then decoder.tempState[beamidx]:copy(tempState)
		else copyTable(tempState, decoder.tempState[beamidx]) end
		if params.gpuidx > 0 then cutorch.synchronize() end

		local prediction = tempPrediction + batch_probabilities:sub(1, params.batch_size, beamidx, beamidx)
		    :reshape(params.batch_size, 1):expand(params.batch_size, params.target_vocab_size)
		
		beam_probabilities:sub(1, params.batch_size, (beamidx - 1) * params.target_vocab_size + 1, beamidx * params.target_vocab_size):copy(prediction)
	    end
	    
	    for itemidx_in_batch = 1, params.batch_size do
		-- This part is computationally too expensive, but it outputs the
		-- the proper result. Need to fix at a later stage.
		for beamidx = 1, params.beam_size do
		    if is_beam_active[itemidx_in_batch][beamidx] == 0 then
			
			beam_probabilities:sub(itemidx_in_batch, itemidx_in_batch, (beamidx - 1) * params.target_vocab_size + 1, beamidx * params.target_vocab_size):fill(-math.huge)
		    end
		end

		-- These are the words for which a prediction for their next one
		-- has been computed.
		local candidate_words = torch.Tensor(batch_indices[itemidx_in_batch]:size()):copy(batch_indices[itemidx_in_batch])


		-- The winning indices of the tokens along in the aggregated dictionary
		-- that lead to the sequences with the highest probabilities. next_word_probabilities
		-- contains the probabilities of the resultant sequences up to those predicted tokens
		-- (i.e. timestep: j + 1).
		next_word_probabilities, next_word_indices = beam_probabilities[itemidx_in_batch]:topk(params.beam_size, true, true)

		-- This is a deep copy of the is_beam_active mask.
		local tmp_is_beam_active = torch.ByteTensor(is_beam_active[itemidx_in_batch]:size()):copy(is_beam_active[itemidx_in_batch])

		-- Deep copy of the running candidates just before we set them to zero.
		local tmp_candidates = torch.Tensor(candidates[{{}, (state.batchidx - 1) * params.batch_size + itemidx_in_batch, {}}]:size())
:copy(candidates[{{}, (state.batchidx - 1) * params.batch_size + itemidx_in_batch, {}}])


		
		-- We are setting the candidates of this particular item in the batch (i.e. itemidx_in_batch) to zero.
		-- We are re-filling it based on the tmp_candidates matrix according to the indices of the most
		-- probable sequences.
		candidates[{{1, params.beam_size}, (state.batchidx - 1) * params.batch_size + itemidx_in_batch}]:zero()


		local num_active_beams = is_beam_active[itemidx_in_batch]:eq(1):sum()
		local num_completed_beams = is_beam_active[itemidx_in_batch]:eq(0):sum()

		for beamidx = 1, params.beam_size do

		    if is_beam_active[itemidx_in_batch][beamidx] == 1 then
				
			
			winning_sequence_index = math.floor((next_word_indices[beamidx - is_beam_active[itemidx_in_batch]:sub(1, beamidx):eq(0):sum()] - 1) / params.target_vocab_size) + 1
			winning_latter_words = next_word_indices[beamidx - is_beam_active[itemidx_in_batch]:sub(1, beamidx):eq(0):sum()] % params.target_vocab_size
			if winning_latter_words == 0 then winning_latter_words = params.target_vocab_size end

			
			-- Update the hidden states of the winning beams.
			if params.layers == 1 then 
			    decoder.s[beamidx][itemidx_in_batch]:copy(decoder.tempState[winning_sequence_index][itemidx_in_batch])
			else
			    for d = 1, params.layers do
				decoder.s[beamidx][d][itemidx_in_batch]:copy(decoder.tempState[winning_sequence_index][d][itemidx_in_batch])
			    end
			end

			
			if winning_latter_words == end_token then

			    tmp_is_beam_active[beamidx] = 0
			    summaries[tmp_is_beam_active:eq(0):sum()][(state.batchidx - 1) * params.batch_size + itemidx_in_batch]:copy(tmp_candidates[winning_sequence_index])
			    summaries[tmp_is_beam_active:eq(0):sum()][(state.batchidx - 1) * params.batch_size + itemidx_in_batch][j] = candidate_words[winning_sequence_index]
			    summaries[tmp_is_beam_active:eq(0):sum()][(state.batchidx - 1) * params.batch_size + itemidx_in_batch][j + 1] = end_token
			    probabilities[tmp_is_beam_active:eq(0):sum()][(state.batchidx - 1) * params.batch_size + itemidx_in_batch] = next_word_probabilities[beamidx - is_beam_active[itemidx_in_batch]:sub(1, beamidx):eq(0):sum()]


			    -- Update batch_indices and batch_probabilities for the
			    -- prediction of the i + 1 word.
			    batch_indices[itemidx_in_batch][beamidx] = 0
			    batch_probabilities[itemidx_in_batch][beamidx] = -math.huge			    
			else
			
			    -- The indexing is preverted here as well!
			    candidates[beamidx][(state.batchidx - 1) * params.batch_size + itemidx_in_batch]:copy(tmp_candidates[winning_sequence_index])
			    candidates[beamidx][(state.batchidx - 1) * params.batch_size + itemidx_in_batch][j] = candidate_words[winning_sequence_index]


			    -- Update batch_indices and batch_probabilities for the
			    -- prediction of the i + 1 word. batch_indices are used as an input to the neural
			    -- network for the prediction of the word at j + 1.
			    batch_indices[itemidx_in_batch][beamidx] = winning_latter_words
			    batch_probabilities[itemidx_in_batch][beamidx] = next_word_probabilities[beamidx - is_beam_active[itemidx_in_batch]:sub(1, beamidx):eq(0):sum()]
			end


		    else
			batch_indices[itemidx_in_batch][beamidx] = 0
			batch_probabilities[itemidx_in_batch][beamidx] = -math.huge
		    end
		end
		is_beam_active[itemidx_in_batch]:copy(tmp_is_beam_active)
	    end
	end
    
	state.batchidx = state.batchidx + 1
	collectgarbage()
	collectgarbage()
    end


    -- Creating the HDF5 file that will be used in order to store
    -- the generated summaries.
    local summaries_filename = string.format('%s.batch_size_%d.beam_size_%d.summaries_%s.h5', params.checkpoint, params.batch_size, params.beam_size, state.name)
    -- Previous format of storing the summary.
    -- local summaries_filename = string.format('%s.summaries_%s.beam_%s.h5', params.checkpoint, state.name, params.beam_size)
    local summaries_file = hdf5.open(summaries_filename, 'w')

    
    summaries_file:write(tostring('triples'), state.triples:int())
    summaries_file:write(tostring('summaries'), summaries)
    summaries_file:write(tostring('probabilities'), probabilities)
    summaries_file:write(tostring('actual_summaries'), state.summaries:int())
    summaries_file:close()    
end

xpcall(main, errorHandler)












    
