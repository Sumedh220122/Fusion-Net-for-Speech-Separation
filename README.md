# Overview

The following repository presents a Deep Learning Model that follows a novel approach to Speech Separation by making use of state-of-the-art transformer architectures. The model demonstrates exceptional results in the domain of speech separation and speaker recognition.

# Usage

1. Each of the folders contain a file description.txt that provides an overview of the contents stored in the folder. Read through each before proceeding further
2. The four files Combined_model 1-4 are used for speech separation in various scenarios
   a. Combined_model1.py - used to separate a mixture of 2 speakers in the foreground and one noise from the background <br/>
   b. Combined_model2.py - used to separate a mixture of 3 speakers in the foreground and one noise from the background
   c. Combined_model3.py - used to separate a mixture of 2 speakers in the foreground and two noises from the background
   d. Combined_model4.py - used to separate a mixture of 3 speakers in the foreground and two noises from the background

The Checkpoints folder is empty and a link to the checkpoints for all 4 models will soon be provided here

To run:
  ```
    python -W ignore Combined_model(the exact model you are using).py
  ```
