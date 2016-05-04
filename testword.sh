#!/bin/bash

echo 'Press r to start recording'
echo 'Press CTRL-C to stop recording, press q to quit'


read -t 1 -n 1 key

until [[ $key = q ]];
do
        if [[ $key = r ]];
        then
          arecord -f cd -c 1 -t wav > ../word.wav
          th classifyword.lua '../word.wav'
          echo 'Please help me learn! If you said yes type y,'
          echo 'if you said no type n, and if you said maybe type m. Thanks!'

        elif [[ $key = y ]];
        then
          echo 'Okay, training on output yes...'
          th classifyword.lua 1

        elif [[ $key = n ]];
        then
          echo 'Okay, training on output no...'
          th classifyword.lua 2

        elif [[ $key = m ]];
        then
          echo 'Okay, training on output maybe...'
          th classifyword.lua 3

        fi
        read -n 1 key
done
