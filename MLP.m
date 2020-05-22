classdef MLP < handle
    properties (SetAccess=private)
        inputDim
        hiddenDim
        outputDim
        
        hiddenWeights
        outputWeights
    end
    
    methods
        function obj=MLP(inputD,hiddenD,outputD)
            obj.inputDim=inputD;
            obj.hiddenDim=hiddenD;
            obj.outputDim=outputD;
            obj.hiddenWeights=zeros(hiddenD,inputD+1);
            obj.outputWeights=zeros(outputD,hiddenD+1);
        end
        
        function obj=initWeight(obj, variance)
            obj.hiddenWeights = variance .* randn([obj.hiddenDim,obj.inputDim+1]);
            obj.outputWeights = variance .* randn([obj.outputDim,obj.hiddenDim+1]);
        end
        
        function [hiddenNet,hidden,outputNet,output]=compute_net_activation(obj, input)
            input = [input; 1]; % add the bias node
            hiddenNet = obj.hiddenWeights * input; % activation threshold
            hidden = 1 ./ (1 + exp(-hiddenNet)); % sigmoid function for hidden layer output
            
            hidden = [hidden; 1]; % add the bias node
            outputNet = obj.outputWeights * hidden; % activation threshold
            output = exp(outputNet) ./ sum(exp(outputNet)); % softmax function for output
            output = output';
        end
        
        function output=compute_output(obj,input)
            [hN,h,oN,output] = obj.compute_net_activation(input);
        end
        
        function [error, obj]=adapt_to_target(obj,input,target,rate)
            [hN,h,oN,o] = obj.compute_net_activation(input);
            error_before_train = o - target; % cross entropy loss
            
            % chain rule (output layer)
            out_der = error_before_train;
            out_change = out_der .* h;
            
            % chain rule (hidden layer)
            hiddenErr = obj.outputWeights' .* out_der;
            hidden_der = hiddenErr .* (h .* (1 - h));
            hidden_change = mean(hidden_der, 2) * [input' 1];
            
            for i = 1:size(obj.outputWeights) % for each output...
                for j = 1:size(obj.outputWeights, 2) % for each connection...
                    obj.outputWeights(i,j) = obj.outputWeights(i,j) - (rate * out_change(j,i)');
                end
            end
            
            for i = 1:size(obj.hiddenWeights) % for each node...
                for j = 1:size(obj.hiddenWeights,2) % for each connection...
                    obj.hiddenWeights(i,j) = obj.hiddenWeights(i,j) - (rate * hidden_change(i,j)');
                end
            end
            
            [hN,h,oN,o] = obj.compute_net_activation(input);
            error_after_train = abs(o) - target;
            error = abs(error_after_train);
        end
    end
end
