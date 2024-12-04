from Models.Waveformer import Net as Wave

import torch
import torchaudio
import os
from pydub.playback import play
from pydub import AudioSegment
import gradio as gr
import speechbrain as sb
import soundfile as sf
from Models.Sepformer_2spk import Net as sep
import matplotlib.pyplot as plt
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
import numpy as np
import librosa

from Models.Noise_reduction import Background_Reduction
from speechbrain.inference.separation import SepformerSeparation as separator

TARGETS = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus",
    "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass",
    "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks",
    "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling",
    "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter",
    "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle",
    "Writing", "People"]

checkpoints_path = 'Checkpoints/'

waveformer = Wave(label_len = 42, device = None, L=32,
                 enc_dim=256, num_enc_layers=10,
                 dec_dim=128, dec_buf_len=13, num_dec_layers=1,
                 dec_chunk_size=13, out_buf_len=4,
                 use_pos_enc=True, skip_connection=True, proj=True, lookahead=True)

sepformer1 = sep1()
#sepformer2 = sep2()
nr = Background_Reduction()

waveformer.load_state_dict(
    torch.load(checkpoints_path + '', map_location=torch.device('cpu'))['model_state_dict']) # Train the model on your dataset and add the Checkpoints files this way
waveformer.eval()

sepformer1.load_state_dict(
        torch.load(checkpoints_path + '', map_location=torch.device('cpu'))['model_state_dict']) # Train the model on your dataset and add the Checkpoints files this way
sepformer1.eval()

resampler = torchaudio.transforms.Resample(orig_freq = 44100, new_freq = 8000)

def combined_model(audio, label_choices, num_spk):

    mixture, sr = torchaudio.load(audio)

    mixture = mixture.unsqueeze(0)

    query = torch.zeros(1, len(TARGETS))
    for t in label_choices:
        query[:, TARGETS.index(t)] = 1

    with torch.no_grad():
        output, mask = waveformer(mixture, query)
    
    output = output / torch.max(torch.abs(output))

    torchaudio.save('output_audio.wav', output.squeeze(0).detach().cpu(), 44100)

    enhanced_signal = nr.spectral_subtraction(audio, 'output_audio.wav', 'Predictions', 0)
    
    human_mix, _ = torchaudio.load('Predictions/enhanced.wav')

    human_mix = resampler(human_mix)

    outputs = ['output_audio.wav']

    est_sources = sepformer1(human_mix)

    est_sources[0, :, :] = est_sources[0, :, :] / torch.max(torch.abs(est_sources[0, :, :]))
    est_sources[1, :, :] = est_sources[1, :, :] / torch.max(torch.abs(est_sources[1, :, :]))

    torchaudio.save('spk1.wav', est_sources[0, :, :].detach().cpu(), 8000)
    torchaudio.save('spk2.wav', est_sources[1, :, :].detach().cpu(), 8000)
    outputs.extend(['spk1.wav', 'spk2.wav'])
    
    return outputs

input_audio = gr.Audio(label="Input audio", type = "filepath")
label_checkbox = gr.CheckboxGroup(choices=TARGETS, label="Input target selection(s)")
search_bar = gr.Textbox(label="Search (Enter the number of speakers)", placeholder="Enter a number")

outputs = [gr.Audio(label=f"Audio {i + 1}") for i in range(3)]

demo = gr.Interface(fn=combined_model, inputs=[input_audio, label_checkbox, search_bar],
                    outputs = outputs)

demo.launch(show_error=True)




