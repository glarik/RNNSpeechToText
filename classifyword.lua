--[[Takes a filepath as an argument and passes it to trained network,
which classifies the recorded word and outputs 'yes', 'no', or 'maybe']]

require 'nn'
require 'audio'



function run(path)
    classify(normalize(load(path)))
end


function load(path)
  -- Step 1: Load audio file and turn it into acceptable tensor
  -- word = audio.stft(audio.load(path),8192,'hann',512)
  -- test word
  local word = audio.stft(audio.load(path),8192,'hann',512)
  -- swaps dimensions 1 and 3 (works better with neural net)
  word = word:transpose(1,3)

  -- resize to fit neural net
  if word:size(3)<211 then
    local size = word:size(3)
    word:resize(2,4097,211)
    for a=1,2 do
      for b=1,4097 do
        for c=size+1,211 do
          word[{a,b,c}]=0
        end
      end
    end

  elseif word:size(3)>211 then
    word = word:narrow(3,1,211)
  end

  return word
end

function normalize(w)
  -- Step 2: Normalize data, using previously calculated mean and stdv
  norm = torch.load('./data/normvars.t7')

  for i=1,2 do
    w[{{i}, {}, {}}]:add(-norm.mean[i])
    w[{{i}, {}, {}}]:div(norm.stdv[i])
  end
  return w
end

-- Step 3: Load trained network and perform necessary setup
-- load net
-- net = torch.load('./data/new_trained_net.t7')
-- net = torch.load('./data/trained_net.t7')
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

function train(output)
  local input = normalize(load('../word.wav'))
  -- local output = torch.ByteTensor(1)
  -- output[1] = label
  criterion = nn.ClassNLLCriterion()

  iter=5
  print ('Training...')
  -- for i=1,iter do

    local pred = net:forward(input)
    local err = criterion:forward(pred, output)
    net:zeroGradParameters()
    local t = criterion:backward(pred, output)
    net:backward(input, t)
    net:updateParameters(0.001)

    -- print ('Iteration '..i..'/'..iter..', complete!')
  -- end

  print ('done!')
  torch.save('./data/new_trained_net.t7',net)
  print ('Updated net saved.')
end

-- working with command line arguments
-- a = tonumber(arg[1])
-- if a == nil then
--   run(arg[1])
-- else
--   for i=1,3 do
--     if a == i then
--       print ('Output = '..a)
--       train(a)
--       break
--     end
--   end
-- end
