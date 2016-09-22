--[[ Will accept a tensor of training data,
clean it up and make a network which will classify the n
words in the set.

we will start with 'yes' 'no' 'maybe'
]]

cuda = false

require 'nn'
require 'csvigo'

if cuda then
  require 'cutorch'
  require 'cunn'
end

dataset = nil
trainset = {}
classes = {}




-- load training and testing data sets
-- TODO: make it take raw tensor data, not cleaned up
-- TODO: Also maybe have it use command line args
dataset = torch.load('../data/spok_nums_v2.t7')


--[[Normalize data and labels]]

function dataset:size()
  return self.data:size(1)
end

--randomize data and lowercase labels
order = torch.randperm(dataset:size())
tempdat = dataset.data
templabel = dataset.label

for i=1,dataset:size() do
  dataset.data[i] = tempdat[order[i]]
  dataset.label[i] = string.lower(templabel[order[i]])
end

-- normalize data
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


--[[Analyze data. Create table of unique labels]]

for i=1,dataset:size() do
  if not classes[dataset.label[i]] then
    classes.insert(dataset.label[i])
  end
end

print(classes)

--[[Create NN based on number of classes]]

net = nn.Sequential()

-- inputs are 4097x211 DoubleTensors

-- 92 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.TemporalConvolution(64, 256, 5)) 

-- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.TemporalMaxPooling(5))     

-- 92 input image channel, 6 output channels, 5x5 convolution kernel
net:add(nn.TemporalConvolution(256, 512, 5)) 

-- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.TemporalMaxPooling(5))
-- net:add(nn.TemporalConvolution(512, 1024, 5))
-- net:add(nn.TemporalMaxPooling(5))
net:add(nn.View(512))

-- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
-- net:add(nn.Linear(512, 256))

-- fully connected layer (matrix multiplication between input and weights)
net:add(nn.Linear(512, 300))
net:add(nn.Linear(300, 84))

-- #classes is the number of outputs of the network
-- taken from the number of labels in the data
net:add(nn.Linear(84, #classes))
net:add(nn.LogSoftMax())

--put net on GPU
if cuda then
  net = net:cuda()
end

print('Simple feedforward word classification net:\n' .. net:__tostring());

criterion = nn.ClassNLLCriterion()
if cuda then
  criterion = criterion:cuda()
end

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.

-- TODO: Train and Save!

function train()
  if cuda then
    trainset.data = trainset.data:cuda()
    trainset.label = trainset.label:cuda()
  end
  trainer:train(trainset)
  if cuda then
    trainset.data = trainset.data:double()
    trainset.label = trainset.label:byte()
  end
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

function run()
  -- we will splipt the dataset in half in order to avoid taking up too much space on the GPU
  middle = math.floor(dataset:size()*0.5)

  trainset = {}
  trainset.data = dataset.data:sub(1,middle)
  trainset.label = dataset.label:sub(1,middle)
  function trainset:size()
    return self.data:size(1)
  end 
  setmetatable(trainset,
      {__index = function(t, i)
                      return {t.data[i], t.label[i]}
                  end}
  );  

  train() -- train on first half

  trainset = {}
  trainset.data = dataset.data:sub(middle+1,dataset:size())
  trainset.label = dataset.label:sub(middle+1,dataset:size())
  function trainset:size()
    return self.data:size(1)
  end
  setmetatable(trainset,
      {__index = function(t, i)
                      return {t.data[i], t.label[i]}
                  end}
  );

  train() --train on second half


end

-- takes percent of dataset to use as training (e.g. 90)
function trainTest(train_percent)
  if (train_percent > 1) then
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

  if (last_index >= dataset:size()) then
    last_index = 0
  end
  
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

  --csvigo.save('./trainon' .. train_percent .. '_pCorrect',pCorrect)

  confmat = getConfMat()

  --outputCSV('./trainon' .. train_percent .. '_confmat',confmat)
end



-- returns # correct
function testCorrect(set, n)
  if not n then
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
  class_performance = {}
  
  -- init each class to 0
  for class in classes do
    class_performance[class] = 0
  end

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
  local out = assert(io.open(filepath,"w"))

  splitter = ','

  for i=1,matrix:size(1) do
    for j=1,matrix:size(2) do
      out:write(tostring(matrix[i][j]))
      if (j ~= matrix:size(2)) then
        out:write(splitter)
      end
    end
    out:write('\n')
  end

  assert(out:close())
end

-- Returns confusion matrix for loaded network on loaded dataset
-- NOTE: THIS ONLY WORKS FOR CLASS SIZE OF 10
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
