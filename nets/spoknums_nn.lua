require 'nn'
require 'cunn'
require 'cutorch'
require 'csvigo'

dataset = torch.load('../data/spok_nums_v2.t7')

-- torch.save('./data/spok_nums_v2.t7',dataset)

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

print('Simple feedforward word classification net:\n' .. net:__tostring());

classes = {'0','1','2','3','4','5','6','7','8','9'}

function dataset:size()
  return self.data:size(1)
end

--randomize data
order = torch.randperm(dataset:size())
tempdat = dataset.data
templabel = dataset.label

for i=1,dataset:size() do
  dataset.data[i] = tempdat[order[i]]
  dataset.label[i] = templabel[order[i]]
end

dataset.data = dataset.data:double()

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,64 do
    mean[i] = dataset.data[{ {}, {i}, {} }]:mean() -- mean estimation
    -- print('Channel ' .. i .. ', Mean: ' .. mean[i])
    dataset.data[{ {}, {i}, {} }]:add(-mean[i]) -- mean subtraction

    stdv[i] = dataset.data[{ {}, {i}, {}  }]:std() -- std estimation
    -- print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    dataset.data[{ {}, {i}, {}}]:div(stdv[i]) -- std scaling
end

-- required for training functions
setmetatable(dataset,
    {__index = function(t, i)
                    return {t.data[i], t.label[i]}
                end}
);

criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.

-- takes percent of dataset to use as training (e.g. 90)
function trainTest(train_percent)
  if (train_percent >= 1) then
    train_percent = train_percent/100
  end

  last_index = math.floor(dataset:size()*train_percent)

  -- create training dataset of size train_percent % of dataset size
  trainset = {}
  trainset.data = dataset.data:sub(1,last_index)
  trainset.label = dataset.label:sub(1,last_index)
  function trainset:size()
    return self.data:size(1)
  end
  setmetatable(trainset,
      {__index = function(t, i)
                      return {t.data[i], t.label[i]}
                  end}
  );

  -- create test dataset of size remaining % of dataset size
  testset = {}
  testset.data = dataset.data:sub(last_index+1,dataset:size())
  testset.label = dataset.label:sub(last_index+1,dataset:size())
  function testset:size()
    return self.data:size(1)
  end
  setmetatable(testset,
      {__index = function(t, i)
                      return {t.data[i], t.label[i]}
                  end}
  );

  -- google stochastic gradient parameters!!
  trainer.maxIteration = 25;

  -- get % correct after each iteration
  pCorrect = {{}}
  i=1
  function trainer.hookIteration()
      pCorrect[1][i] = getCorrect()
      i = i+1
  end

  train()

  csvigo.save('./trainon'..train_percent..'_pCorrect',pCorrect)

  confmat = getConfMat()

  outputCSV('./trainon'..train_percent..'_confmat',confmat)
end

function train()
  trainset.data = trainset.data:cuda()
  trainset.label = trainset.label:cuda()
  trainer:train(trainset)
  trainset.data = trainset.data:double()
  trainset.label = trainset.label:byte()
end

function getCorrect()
  testset.data = testset.data:cuda()
  testset.label = testset.label:cuda()
  correct = testCorrect(testset)
  testset.data = testset.data:double()
  testset.label = testset.label:byte()

  return 100*correct/testset:size()

  -- print(correct .. " correct out of "..testset:size().." = "..100*correct/testset:size().."%")
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
  testset.data = testset.data:cuda()
  testset.label = testset.label:cuda()
  confmat = confMat(testset)
  testset.data = testset.data:double()
  testset.label = testset.label:byte()

  return confmat

  -- print("X axis: Predicted Values")
  -- print("Y axis: Actual Values")
  -- print(confmat)
end

function outputCSV(filepath,matrix)
  io.output(filepath)

  splitter = ','

  for i=1,matrix:size(1) do
    for j=1,matrix:size(2) do
      io:write(matrix[i][j])
      if (j ~= matrix:size(2)) then
        io:write(splitter)
      end
    end
    io:write('\n')
  end
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
