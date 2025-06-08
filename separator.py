import os
import torchaudio
import torch
import torch.nn.functional as F
from Models.Waveformer import Net as Waveformer
from Models.Noise_reduction import Background_Reduction
import warnings

warnings.filterwarnings("ignore")


mix_path = 'C:/Users/ravi_/Music/Mixed2/Mixed2'
labels_path = 'C:/Users/ravi_/Music/labels/labels'
checkpoints_path = 'Checkpoints/'

def load_and_sort(mix_path):
    """
    Load and sort the files in the mix_path
    """
    audio_files = []
    supported_formats = ['wav', 'pt']

    for root, _, files in os.walk(mix_path):
            for file in files:
                if file.split('.')[-1] in supported_formats:
                    path = os.path.join(root, file)
                    audio_files.append(path)
            break

    audio_files = sorted(audio_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    return audio_files

def separate_files(audio_files, labels, checkpoints_path):
     """
     Separate the files into foreground and background
     """
     waveformer = Waveformer(label_len=42, device=torch.device('cpu'))
     background_reduction = Background_Reduction()

     target_folder = 'C:/Users/ravi_/Music/Enhanced'
     os.makedirs(target_folder, exist_ok=True)

     waveformer.load_state_dict(
     torch.load(checkpoints_path + 'checkpoint_wave_2spk1bg.pth', map_location=torch.device('cpu'))['model_state_dict'])
     waveformer.eval()


     if len(audio_files) != len(labels):
          raise ValueError("The number of audio files and labels must be the same")
     
     for i in range(len(audio_files)):
          audio = audio_files[i]
          label = torch.load(labels[i])
          label = F.pad(label, (0, 1))

          label = label.unsqueeze(0)

          audio_waveform, _ = torchaudio.load(audio)
          audio_waveform = audio_waveform.unsqueeze(0)

          with torch.no_grad():
               output, _ = waveformer(audio_waveform, label)

          output = output / torch.max(torch.abs(output))

          torchaudio.save('output_audio.wav', output.squeeze(0).detach().cpu(), 44100)

          background_reduction.spectral_subtraction(mixture = audio_files[i], 
                                                    background = 'output_audio.wav', 
                                                    target_folder = target_folder, 
                                                    i = i
                                                    )
          
if __name__ == "__main__":
     audio_files = load_and_sort(mix_path)
     labels = load_and_sort(labels_path)
     separate_files(audio_files=audio_files, labels=labels, checkpoints_path=checkpoints_path)
        








