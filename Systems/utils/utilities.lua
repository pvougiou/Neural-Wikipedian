----
----  The code is based on: https://github.com/oxford-cs-ml-2015/practical6
----  
----  Creates clones of the given network.
----  The clones share all weights and gradWeights with the original network.
----  Accumulating of the gradients sums the gradients properly.
----  The clone also allows parameters for which gradients are never computed
----  to be shared. Such parameters must be returns by the parametersNoGrad
----  method, which can be null.
----

function cloneManyTimes(net, T)
    local clones = {}
    local params, gradParams
    if net.parameters then
	params, gradParams = net:parameters()
	if params == nil then
	    params = {}
	end
    end
    local paramsNoGrad
    if net.parametersNoGrad then
	paramsNoGrad = net:parametersNoGrad()
    end
    
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)
    for t = 1, T do
	-- We need to use a new reader for each clone.
	-- We don't want to use the pointers to already read objects.
	local reader = torch.MemoryFile(mem:storage(), "r"):binary()
	local clone = reader:readObject()
	reader:close()
	
	if net.parameters then
	    local cloneParams, cloneGradParams = clone:parameters()
	    local cloneParamsNoGrad
	    for i = 1, #params do
		cloneParams[i]:set(params[i])
		cloneGradParams[i]:set(gradParams[i])
	    end
	    
	    if paramsNoGrad then
		cloneParamsNoGrad = clone:parametersNoGrad()
		for i =1, #paramsNoGrad do
		    
		    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
		end
	    end
	end
	clones[t] = clone
	collectgarbage()
    end
    mem:close()
    return clones
end


----  Creates a contiguous 1D-Tensor large enough to hold all the parameters,
----  and points all the modules to this one 1D-Tensor. 
----  The existing weights and gradients tensors are discarded.
----  Suppose you had 3 modules, m1, m2 and m3, where m1 and m2 shared their 
----  parameters (and their gradients), but m3 did not. 
----  Calling combine_all_parameters on these 3 will produce a 1D-Tensor with
----  space for the params of m1 and m2 and M3, with the parameters for the
----  first two both being references to the same place in memory for m1 and m2.
----  The tensor for the parameters for m3 will point to the latter half of
----  the new 1D-Tensor. The 1D-Tensor gets returned as the first return
----  value of combine_all_parameters(...), and the corresponding gradParams
----  is returned too.


function combine_all_parameters(...)

    local networks = {...}
    local parameters = {}
    local gradParameters = {}
    for i = 1, #networks do
	local net_params, net_grads = networks[i]:parameters()
	
	if net_params then
	    for _, p in pairs(net_params) do
		parameters[#parameters + 1] = p
	    end
	    for _, g in pairs(net_grads) do
		gradParameters[#gradParameters + 1] = g
	    end
	end
    end

    local function storageInSet(set, storage)
	local storageAndOffset = set[torch.pointer(storage)]
	if storageAndOffset == nil then
	    return nil
	end
	local _, offset = unpack(storageAndOffset)
	return offset
    end

    -- This function flattens arbitrary lists of parameters, even complex shared ones.
    local function flatten(parameters)
	if not parameters or #parameters == 0 then
	    return torch.Tensor()
	end
	local Tensor = parameters[1].new
	
	local storages = {}
	local nParameters = 0
	for k = 1,#parameters do
	    local storage = parameters[k]:storage()
	    if not storageInSet(storages, storage) then
		storages[torch.pointer(storage)] = {storage, nParameters}
		nParameters = nParameters + storage:size()
	    end
	end
	
	local flatParameters = Tensor(nParameters):fill(1)
	local flatStorage = flatParameters:storage()

	for k = 1,#parameters do
	    local storageOffset = storageInSet(storages, parameters[k]:storage())
	    parameters[k]:set(flatStorage,
			      storageOffset + parameters[k]:storageOffset(),
			      parameters[k]:size(),
			      parameters[k]:stride())
	    parameters[k]:zero()
	end

	local maskParameters=  flatParameters:float():clone()
	local cumSumOfHoles = flatParameters:float():cumsum(1)
	local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
	local flatUsedParameters = Tensor(nUsedParameters)
	local flatUsedStorage = flatUsedParameters:storage()

	for k = 1,#parameters do
	    local offset = cumSumOfHoles[parameters[k]:storageOffset()]
	    parameters[k]:set(flatUsedStorage,
			      parameters[k]:storageOffset() - offset,
			      parameters[k]:size(),
			      parameters[k]:stride())
	end
	
	for _, storageAndOffset in pairs(storages) do
	    local k, v = unpack(storageAndOffset)
	    flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
	end
	
	if cumSumOfHoles:sum() == 0 then
	    flatUsedParameters:copy(flatParameters)
	else
	    local counter = 0
	    for k = 1,flatParameters:nElement() do
		if maskParameters[k] == 0 then
		    counter = counter + 1
		    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
		end
	    end
	    assert (counter == nUsedParameters)
	end
	return flatUsedParameters
    end
    
    -- Flatten parameters and gradients.
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)
    
    -- Return new flat vector that contains all discrete parameters.
    return flatParameters, flatGradParameters
end

function copyTable(fromTable, toTable)
    assert(#fromTable == #toTable)
    for i = 1, #toTable do
	toTable[i]:copy(fromTable[i])
    end
end

function extendTable(extendedTable, inputTable)
    for k, v in pairs(inputTable) do 
	if (type(extendedTable[k]) == 'table' and type(v) == 'table') then
	    extend(extendedTable[k], v)
	else
	    extendedTable[k] = v 
	end
    end
end

function argmax(vector)
    if vector:dim() == 1 then
	for i = 1, vector:size(1) do
	    if vector[i] == vector:max() then
		return i
	    end
	end
    end
end

function argmin(vector)
    if vector:dim() == 1 then
	for i = 1, vector:size(1) do
	    if vector[i] == vector:min() then
		return i
	    end
	end
    end
end

-- Transfer input tensor to GPU
function transfer_to_gpu(x, gpuidx)
    if gpuidx > 0 then
	return x:cuda()
    else
	return x
    end
end

-- Prints GPU memory usage
function print_gpu_usage(gpuidx)
    if gpuidx > 0 then
	freeMemory, totalMemory = cutorch.getMemoryUsage(gpuidx)
	print(string.format("Memory Usage: %.2f", (freeMemory / (1024 * 1024))).. '\t/ '.. string.format("%.2f", (totalMemory / (1024 * 1024))))
    end
end

-- Returns the length of an arbitrary table
-- Adapted from: https://stackoverflow.com/questions/2705793/how-to-get-number-of-entries-in-a-lua-table
function length(array)
    count = 0
    for _ in pairs(array) do
	count = count + 1
    end
    return count
end

function generateCheckpoint(path, model, details)
    local savefile = string.format(path, details.epoch, details.err)
    local checkpoint = {model = model, details = details}   
    print('Storing checkpoint file at: ' .. savefile)
    torch.save(savefile, checkpoint)	
end
