--[[ Will accept a tensor of training data,
clean it up and make a network which will classify the n
words in the set.

we will start with 'yes' 'no' 'maybe'
]]

require 'nn'

-- load training and testing data sets
-- TODO: make it take raw tensor data, not cleaned up
-- TODO: Also maybe have it use command line args
trainset = torch.load('../data/all_ynm.t7')
testset = torch.load('../data/testset.t7')

-- TODO: Trim and normalize data. Lowercase all labels



-- TODO: Analyze data. Output number of classes and list of unique labels

-- TODO: Create NN based on number of classes

-- TODO: Train and Save!

-- initialize Sequential nn
net = nn.Sequential()

-- add layers (disregard comments next to layers)
net:add(nn.SpatialConvolution(2, 6, 5, 5)) -- 92 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5)) -- 92 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.View(16*1021*49))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*1021*49, 120))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.Linear(120, 84))
net:add(nn.Linear(84, 3))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems

print('Simple feedforward word classification net:\n' .. net:__tostring());

classes = {'yes', 'no', 'maybe'}

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,2 do
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    -- print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    -- print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end



-- required for training functions
setmetatable(trainset,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]}
                end}
);

criterion = nn.ClassNLLCriterion()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 10 epochs of training.

function train()
  trainer:train(trainset)
end


-- prints out percent of correct out of total
function testCorrect(n)
  if n==nil then
    n=net
  end
  correct = 0
  for i=1,3 do
    local groundtruth = testset.label[i]
    local prediction = n:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
      correct = correct + 1
    end
  end
  print(correct .. '   (out of 3)')
end
