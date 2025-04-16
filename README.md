# Overview

The following repository presents a Deep Learning Model that follows a novel approach to Speech Separation by making use of state-of-the-art transformer architectures. The model demonstrates exceptional results in the domain of speech separation and speaker recognition.

# Usage

1. Each of the folders contain a file description.txt that provides an overview of the contents stored in the folder. Read through each before proceeding further
2. The four files FusionNet 1-4 are used for speech separation in various scenarios <br/>
   a. FusionNet1.py - used to separate a mixture of 2 speakers in the foreground and one noise from the background <br/>
   b. FusionNet2.py - used to separate a mixture of 3 speakers in the foreground and one noise from the background <br/>
   c. FusionNet3.py - used to separate a mixture of 2 speakers in the foreground and two noises from the background <br/>
   d. FusionNet4.py - used to separate a mixture of 3 speakers in the foreground and two noises from the background <br/>

The Checkpoints folder is empty and a link to the checkpoints for all 4 models will soon be provided here

To run:
  ```
    python -W ignore FusionNet(the exact model you are using).py
  ```
