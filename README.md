# RNNSpeechToText

**Note:** This repository is under construction. Please don't attempt to use it until it is finished, because it will not do anything useful.

## About
Offline, standalone speech recognition using recurrent neural networks in Torch

## Runme
While this isn't very usable as of yet, there are several things one can play around with, although they may require some technical knowledge (please note that the data files used in many of these files are not included as they are too large for github):

1. The scripts directory contains two scripts that run code which allows the user to speak a word and see what the network spits out. spect_testword.sh will store words in spectrogram form, while stft_testword.sh will store words in stft form.

2. The lua_files directory contain some lua files to run existing neural nets. Two of these files, classify_spect.lua and classify_stft.lua are the workhorses behind the scripts mentioned above. data_ops.lua is full of useful data operations (hence the name).

3. The nets directory contains files to initialize and train various neural nets. spoknums_nn.lua contains the latest iteration which is meant to work with a spoken digits spectrogram dataset. The other two are for spects and stfts, as labeled.

### Purpose
The purpose of this repository is to create a more or less one-stop solution for anyone looking to use speech recognition in their work. The program will not require internet access, but will require training using datasets, which can be found online. I will include links to several of these as the project progresses.

### Usability
The ideal program allows for ease of use for a non-technical user, while simultaneously providing options for tweaking and expandability for advanced users. While this is not always feasible, this project will be geared more towards ease of use, since most of the time a program that 'just works' is more valuable than one which takes a week to get up and running.

### Software and Compatibility
This program will be written and run on Ubuntu 15.10 (15.04 should also work), using Torch. See the [Torch Website] (http://torch.ch/ "Torch") for information on installing and using Torch.

### GPU Support
Yes! Super fast!!
