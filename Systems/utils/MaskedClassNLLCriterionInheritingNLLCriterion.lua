----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree.
----

local MaskedClassNLLCriterion, parent = torch.class('nn.MaskedClassNLLCriterion', 'nn.ClassNLLCriterion')

function MaskedClassNLLCriterion:__init()
	self._gradInput = torch.Tensor()
   	parent.__init(self)
	self.sizeAverage = false
	
end


function MaskedClassNLLCriterion:updateOutput(input, target)
	self.validMask = target:ne(0)
	self.validTarget = target[self.validMask]:contiguous()
	if self.validTarget:nElement() == 0 then
		return 0
	end
	self.inputMask = self.validMask:resize(self.validMask:nElement(), 1):expandAs(input)
	self.validInput = input[self.inputMask]:contiguous():reshape(self.validTarget:size(1), input:size(2))
	return parent.updateOutput(self, self.validInput, self.validTarget)
end


function MaskedClassNLLCriterion:updateGradInput(input, target)
	self._gradInput = self._gradInput:typeAs(input):resizeAs(input)
	self._gradInput:zero()
	if self.validTarget:nElement() > 0 then
		local gradInput = parent.updateGradInput(self, self.validInput, self.validTarget)
		self._gradInput[self.inputMask] = gradInput
	end
	return self._gradInput
end
