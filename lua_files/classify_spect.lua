--[[Takes a filepath as an argument and passes it to trained network,
which classifies the recorded word and outputs 'yes', 'no', or 'maybe']]

require 'nn'
require 'audio'



function run(path)
    classify(normalize(load(path)))
end


function load(path)
  -- Step 1: Load audio file and turn it into acceptable tensor
  -- test word
  local word = audio.spectrogram(audio.load(path),8192,'hann',512)

  -- resize to fit neural net
  if word:size(2)<211 then
    local size = word:size(2)
    word:resize(4097,211)
    for a=1,4097 do
      for b=size+1,211 do
        word[{a,b}]=0
      end
    end

  elseif word:size(2)>211 then
    word = word:narrow(2,1,211)
  end

  return word
end

function normalize(w)
  -- Step 2: Normalize data, using previously calculated mean and stdv
  norm = torch.load('../data/spect_normvars.t7')

  for i=1,4097 do
    w[{{i}, {}}]:add(-norm.mean[i])
    w[{{i}, {}}]:div(norm.stdv[i])
  end
  return w
end

-- save to dataset for batch training
function save_to_set (input,label,setpath)
  set = torch.load(setpath)
  if set.data:dim() == 0 then
    set.data = torch.Tensor(1,4097,211)
    set.label = torch.ByteTensor(1)
  else
    new_data = torch.Tensor(set:size()+1,4097,211)
    new_label = torch.ByteTensor(set:size()+1)
    for i=1,set:size() do
      new_data = set.data[i]
      new_label = set.label[i]
    end
    set.data = new_data
    set.label = new_label
  end
  set.data[set:size()] = input
  set.label[set:size()] = label
  torch.save(setpath,set)
end


-- Step 3: Load trained network and perform necessary setup
-- load net
-- net = torch.load('./data/new_trained_spect.t7')
net = torch.load('../data/spect_trained_net.t7')
classes = {'yes', 'no', 'maybe'}

function classify(w)
  local prediction = net:forward(w)
  print (prediction)

  -- sort by likelihood
  local confidences, indices = torch.sort(prediction, true)

  which = classes[indices[1]]

  print ('This sounds a lot like \'' .. which .. '\'!')

  return which
end

function train_single(output)
  local input = normalize(load('../data/word.wav'))
  criterion = nn.ClassNLLCriterion()

  iter=5
  print ('Training...')
  for i=1,iter do

    local pred = net:forward(input)
    local err = criterion:forward(pred, output)
    net:zeroGradParameters()
    local t = criterion:backward(pred, output)
    net:backward(input, t)
    net:updateParameters(0.001)

    print ('Iteration '..i..'/'..iter..', complete!')
  end

  print ('done!')
  torch.save('../data/new_trained_spect.t7',net)
  print ('Updated net saved.')
end

function train_set (setpath)
end

-- working with command line arguments
if arg[1] != nil then
  a = tonumber(arg[1])
  if a == nil then
    run(arg[1])
  else
    for i=1,3 do
      if a == i then
        print ('Output = '..a)
        train(a)
        break
      end
    end
  end
end
