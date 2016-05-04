require 'nn'


trainset = torch.load('./data/spok_nums_all.t7')

net = nn.Sequential()

-- inputs are 4097x211 DoubleTensors
net:add(nn.TemporalConvolution(512, 1024, 5)) -- 92 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.TemporalMaxPooling(10))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.TemporalConvolution(1024, 2048, 5)) -- 92 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.TemporalMaxPooling(10))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.View(254*844))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(254*844, 120))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.Linear(120, 84))
net:add(nn.Linear(84, 3))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())

print('Simple feedforward word classification net:\n' .. net:__tostring());

classes = {'0','1','2','3','4','5','6','7','8','9'}

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,512 do
    mean[i] = trainset.data[{ {}, {i}, {} }]:mean() -- mean estimation
    -- print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {} }]:add(-mean[i]) -- mean subtraction

    stdv[i] = trainset.data[{ {}, {i}, {}  }]:std() -- std estimation
    -- print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}}]:div(stdv[i]) -- std scaling
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
  for i=1,4684 do
    local groundtruth = trainset.label[i]
    local prediction = n:forward(trainset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
      correct = correct + 1
    end
  end
  print(correct .. '   (out of 4684)')
end
