require 'audio'

--[[
Assortment of functions for making dataset from files.
]]

-- devset = {}
-- devset.data = {}
-- devset.label = {}

-- loads a single audio file and returns the stft
function calc_stft(filepath)
  return audio.stft(audio.load(filepath),8192,'hann',512)
end

function calc_spect(filepath)
  return audio.spectrogram(audio.load(filepath),8192,'hann',512)
end

function init_dataset(d)
  function d:size()
    return 18
  end

  d.data = torch.Tensor(18,157,4097)
  d.label = torch.ByteTensor(18)
end

-- creates and fills dataset with 15 recordings, 5 each  of yes, no, maybe
function make_dataset(d)
  -- function devset:size()
  --   return 15
  -- end

  data={}
  label={}

  dir = '/home/george/Desktop/research/yesnomaybe/'

  for i=0,17 do
    if i<6 then
      path = dir .. 'yes/'
      output = 1
    elseif i<12 then
      path = dir .. 'no/'
      output = 2
    else
      path = dir .. 'maybe/'
      output = 3
    end

    input = calc_spect(path .. (i%6) .. '.wav')
    -- input = input:transpose(1,3) -- only useful for stft
    input = input:t() --swap dims 1 and 2 of 2D vector


    data[i+1] = input
    label[i+1] = output
    -- devset.data[i] = input
    -- devset.label[i] = label
  end
end

function pad_spect_data(data)
  for n=1,18 do
    data[n] = pad_spect(data[n]);
  end
end

function pad_spect (spect)
  local cur = spect
  local size = cur:size(1)
  cur:resize(211,4097)
  for b=1,4097 do
    for a=size+1,211 do
      cur[{a,b}]=0
    end
  end
  return cur
end

function pad_stft(data)
  for n=1,18 do
    local cur = data.data[n]
    local size = cur:size(3)
    cur:resize(2,4097,211)
    for a=1,2 do
      for b=1,4097 do
        for c=size+1,211 do
          cur[{a,b,c}]=0
        end
      end
    end
  end
end

function make_spect_from_sound (filepath)
  return pad_spect(calc_spect(filepath))
end


-- saves dataset
-- function devset:save(filename)
--   torch.save(filename,self)
-- end
