----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree.
----
----  Parts of this code are adapted from: https://github.com/wojzaremba/lstm/blob/master/data.lua
----

require 'hdf5'
local cjson = require 'cjson'


local function init_dataset(dataset_path)
    summaries_filename = dataset_path.. 'summaries.h5'
    summaries_dictionary_filename = dataset_path.. 'summaries_dictionary.json'
    triples_filename = dataset_path.. 'triples.h5'
    triples_dictionary_filename = dataset_path.. 'triples_dictionary.json'
    summaries = hdf5.open(summaries_filename, 'r')
    triples = hdf5.open(triples_filename, 'r')
    collectgarbage()
    collectgarbage()
 end

local function sequence_batch_splitter(input, batch_size)
    local height = input:size(1)
    local width = input:size(2)
    local x = torch.zeros(torch.floor(height / batch_size), batch_size, width)
    for i = 1, batch_size do
	local start = torch.round((i - 1) * height / batch_size) + 1
	local finish = start + x:size(1) - 1
	x:sub(1, x:size(1), i, i, 1, x:size(3)):copy(input:sub(start, finish, 1, width))
    end
    collectgarbage()
    return x
end

local function triples_batch_splitter(input, numAlignedTriples, batch_size)
    local height = input:size(1)
    local width = input:size(2)
    local x = torch.zeros(torch.floor(height / (batch_size * numAlignedTriples)), numAlignedTriples, batch_size, width)
    for i = 1, batch_size do
	local start = torch.round((i - 1) * height / (batch_size * numAlignedTriples)) + 1
	local finish = start + x:size(1) - 1
	x:sub(1, x:size(1), 1, x:size(2), i, i, 1, x:size(4))
	    :copy(input:sub((start - 1) * numAlignedTriples + 1, finish * numAlignedTriples, 1, width))
    end
    collectgarbage()
    return x
end

local function train_summaries(batch_size)
    local x = summaries:read('/train'):all()
    x = sequence_batch_splitter(x, batch_size)
    collectgarbage()
    return x
end

local function train_summaries_len(batch_size)
    local x = summaries:read('/train_len'):all()
    x = sequence_batch_splitter(x, batch_size)
    return x
end

local function train_triples(numAlignedTriples, batch_size)
    local x = triples:read('/train'):all()
    x = triples_batch_splitter(x, numAlignedTriples, batch_size)
    collectgarbage()
    return x
end

local function validate_summaries(batch_size)
    local x = summaries:read('/validate'):all()
    x = sequence_batch_splitter(x, batch_size)
    collectgarbage()
    return x
end

local function validate_summaries_len(batch_size)
    local x = summaries:read('/validate_len'):all()
    x = sequence_batch_splitter(x, batch_size)
    collectgarbage()
    return x
end

local function validate_triples(numAlignedTriples, batch_size)
    local x = triples:read('/validate'):all()
    x = triples_batch_splitter(x, numAlignedTriples, batch_size)
    collectgarbage()
    return x
end

local function test_summaries(batch_size)
    local x = summaries:read('/test'):all()
    x = sequence_batch_splitter(x, batch_size)
    collectgarbage()
    return x
end

local function test_summaries_len(batch_size)
    local x = summaries:read('/test_len'):all()
    x = sequence_batch_splitter(x, batch_size)
    collectgarbage()
    return x
end

local function test_triples(numAlignedTriples, batch_size)
    local x = triples:read('/test'):all()
    x = triples_batch_splitter(x, numAlignedTriples, batch_size)
    collectgarbage()
    return x
end

local function triples_dictionary()
    local f = io.open(triples_dictionary_filename, 'r')
    local temp = f:read('*all')
    f:close()
    return cjson.decode(temp)
end

local function summaries_dictionary()
    local f = io.open(summaries_dictionary_filename, 'r')
    local temp = f:read('*all')
    f:close()
    return cjson.decode(temp)
end


return {init_dataset = init_dataset,
	train_triples = train_triples, train_summaries = train_summaries,
	validate_triples = validate_triples, validate_summaries = validate_summaries,
	test_triples = test_triples, test_summaries = test_summaries,
	triples_dictionary = triples_dictionary, summaries_dictionary = summaries_dictionary
}
