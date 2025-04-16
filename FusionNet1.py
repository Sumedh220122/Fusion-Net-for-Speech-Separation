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


class FusionNet:
    def __init__(self):
        self.waveformer = Wave(label_len = 42, device = None, L=32,
                 enc_dim=256, num_enc_layers=10,
                 dec_dim=128, dec_buf_len=13, num_dec_layers=1,
                 dec_chunk_size=13, out_buf_len=4,
                 use_pos_enc=True, skip_connection=True, proj=True, lookahead=True)

        self.sepformer = sep()
        self.nr = Background_Reduction()
        self.resampler = torchaudio.transforms.Resample(orig_freq = 44100, new_freq = 8000)

        self.checkpoints_path = 'Checkpoints/'

    def load_state_dict(self):
        self.waveformer.load_state_dict(
            torch.load(self.checkpoints_path + '', map_location=torch.device('cpu'))['model_state_dict']) # Train the model on your dataset and add the Checkpoints files this way
        self.waveformer.eval()

        self.sepformer.load_state_dict(
            torch.load(self.checkpoints_path + '', map_location=torch.device('cpu'))['model_state_dict']) # Train the model on your dataset and add the Checkpoints files this way
        self.sepformer.eval()

    def separate_audio(self, audio, label_choices, num_spk):

        mixture, sr = torchaudio.load(audio)

        mixture = mixture.unsqueeze(0)

        query = torch.zeros(1, len(TARGETS))
        for t in label_choices:
            query[:, TARGETS.index(t)] = 1

        with torch.no_grad():
            output, mask = self.waveformer(mixture, query)
        
        output = output / torch.max(torch.abs(output))

        torchaudio.save('Predictions/output_audio.wav', output.squeeze(0).detach().cpu(), 44100)

        enhanced_signal = self.nr.spectral_subtraction(audio, 'Predictions/output_audio.wav', 'Predictions', 0)
        
        human_mix, _ = torchaudio.load('Predictions/enhanced.wav')

        human_mix = self.resampler(human_mix)

        outputs = ['Predictions/output_audio.wav']

        est_sources = self.sepformer(human_mix)

        est_sources[0, :, :] = est_sources[0, :, :] / torch.max(torch.abs(est_sources[0, :, :]))
        est_sources[1, :, :] = est_sources[1, :, :] / torch.max(torch.abs(est_sources[1, :, :]))

        torchaudio.save('Predictions/spk1.wav', est_sources[0, :, :].detach().cpu(), 8000)
        torchaudio.save('Predictions/spk2.wav', est_sources[1, :, :].detach().cpu(), 8000)
        outputs.extend(['Predictions/spk1.wav', 'Predictions/spk2.wav'])
        
        return outputs

if __name__ == "__main__":

    model = FusionNet()
    model.load_state_dict()

    input_audio = gr.Audio(label="Input audio", type = "filepath")
    label_checkbox = gr.CheckboxGroup(choices=TARGETS, label="Input target selection(s)")
    search_bar = gr.Textbox(label="Search (Enter the number of speakers)", placeholder="Enter a number")

    outputs = [gr.Audio(label=f"Audio {i + 1}") for i in range(3)]

    demo = gr.Interface(fn=model.separate_audio, inputs=[input_audio, label_checkbox, search_bar],
                        outputs = outputs)

    demo.launch(show_error=True)




