require 'nn'
require 'cunn'
require 'cutorch'

trainset = torch.load('../data/spok_nums_v2.t7')

-- torch.save('./data/spok_nums_v2.t7',trainset)

net = nn.Sequential()

-- inputs are 4097x211 DoubleTensors
net:add(nn.TemporalConvolution(64, 256, 5)) -- 92 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.TemporalMaxPooling(5))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.TemporalConvolution(256, 512, 5)) -- 92 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.TemporalMaxPooling(5))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
-- net:add(nn.TemporalConvolution(512, 1024, 5))
-- net:add(nn.TemporalMaxPooling(5))
net:add(nn.View(512))
-- net:add(nn.Linear(512, 256))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(512, 300))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.Linear(300, 84))
net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())

net = net:cuda()

-- net = torch.load('../data/spoknums_tnet_20x.t7')

print('Simple feedforward word classification net:\n' .. net:__tostring());

classes = {'0','1','2','3','4','5','6','7','8','9'}

function trainset:size()
  return self.data:size(1)
end

--randomize data
order = torch.randperm(trainset:size())
tempdat = trainset.data
templabel = trainset.label

for i=1,trainset:size() do
  trainset.data[i] = tempdat[order[i]]
  trainset.label[i] = templabel[order[i]]
end

trainset.data = trainset.data:double()

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,64 do
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

-- trainset.label = trainset.label:cuda()
-- trainset.data = trainset.data:cuda()

criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.

t1 = {}
t1.data = trainset.data:sub(1,2500)
t1.label = trainset.label:sub(1,2500)
function t1:size()
  return self.data:size(1)
end
setmetatable(t1,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]}
                end}
);


t2 = {}
t2.data = trainset.data:sub(2501,trainset:size())
t2.label = trainset.label:sub(2501,trainset:size())
function t2:size()
  return self.data:size(1)
end
setmetatable(t2,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]}
                end}
);

-- function train90test10()
--
-- end

function train()
  t1.data = t1.data:cuda()
  t1.label = t1.label:cuda()
  trainer:train(t1)
  t1.data = t1.data:double()
  t1.label = t1.label:byte()

  -- t2.data = t2.data:cuda()
  -- t2.label = t2.label:cuda()
  -- trainer:train(t2)
  -- t2.data = t2.data:double()
  -- t2.label = t2.label:byte()
end

function getCorrect()
  -- t1.data = t1.data:cuda()
  -- t1.label = t1.label:cuda()
  -- correct = testCorrect(t1)
  -- t1.data = t1.data:double()
  -- t1.label = t1.label:byte()

  t2.data = t2.data:cuda()
  t2.label = t2.label:cuda()
  correct = testCorrect(t2)
  t2.data = t2.data:double()
  t2.label = t2.label:byte()

  print(correct .. " correct out of "..t2:size().." = "..100*correct/t2:size().."%")
end

-- returns # correct
function testCorrect(set, n)
  if n==nil then
    n=net
  end
  correct = 0
  for i=1,set:size() do
    local groundtruth = set.label[i]
    local prediction = n:forward(set.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
      correct = correct + 1
    end
  end
  return correct
end

-- tests performance by class
function testClassPerformance(set)
  class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
  for i=1,set:size() do
    local groundtruth = set.label[i]
    local prediction = net:forward(set.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
  end

  for i=1,#classes do
    print(classes[i], 100*class_performance[i]/set:size() .. ' %')
  end
end

function getConfMat()
  -- t1.data = t1.data:cuda()
  -- t1.label = t1.label:cuda()
  -- confmat = confMat(t1)
  -- t1.data = t1.data:double()
  -- t1.label = t1.label:byte()

  t2.data = t2.data:cuda()
  t2.label = t2.label:cuda()
  confmat = confMat(t2)
  t2.data = t2.data:double()
  t2.label = t2.label:byte()

  print("X axis: Predicted Values")
  print("Y axis: Actual Values")
  print(confmat)
end

-- Returns confusion matrix for loaded network on loaded dataset
function confMat(set, mat)
  if(mat == nil) then
    confmat = torch.ByteTensor(11,11)
    confmat:zero()

    for i=2,11 do
      confmat[1][i] = i-2
      confmat[i][1] = i-2
    end
  else
    confmat = mat
  end

  for i=1,set:size() do
    local groundtruth = set.label[i]
    local prediction = net:forward(set.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order

    confmat[groundtruth+1][indices[1]+1] = confmat[groundtruth+1][indices[1]+1] + 1
  end

  return confmat
end
