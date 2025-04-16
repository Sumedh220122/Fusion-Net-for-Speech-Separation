import os
import warnings
import scaper
import numpy as np
from pydub import AudioSegment
import torch

warnings.filterwarnings("ignore")

class ScaperSourceGenerator:
    """
    Generate source files for Scaper
    """
    def __init__(self, foreground_folder, background_folder, base_output_folder):
        self.foreground_folder = foreground_folder
        self.background_folder = background_folder
        self.target_fg_folder = "3spk_mix"
        self.target_bg_folder = ""

        self.output_freesound_folder = os.path.join(base_output_folder, "freesound")
        self.output_labels_folder = os.path.join(base_output_folder, "labels")

        self.soundscape_duration = 10
        self.mixed_output_folder = os.path.join(base_output_folder, "freesound_mix")
        self.seed = 123

        self.freesound_labels = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus",
                    "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass",
                    "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks",
                    "Flute", "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling",
                    "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter",
                    "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle",
                    "Writing"]

    def pad_or_truncate_audio(self, audio_path, desired_duration, output_folder, i, start_time=0):
        """
        Pad or truncate an audio file to the desired duration in seconds.
        Args:
            audio_path: Path to input audio file
            desired_duration: Total duration of output audio in seconds
            output_folder: Folder to save the output
            i: Index for the output filename
            start_time: When the audio should start in the final mix (in seconds)
        """
        audio = AudioSegment.from_file(audio_path)
        current_duration = len(audio) // 1000  # Convert milliseconds to seconds

        # Create silence for the initial offset
        initial_silence = AudioSegment.silent(duration=start_time * 1000)
        
        # Calculate how much audio we can fit after the start time
        available_duration = desired_duration - start_time
        
        if current_duration < available_duration:
            # If audio is shorter than available duration, pad with silence
            remaining_duration = available_duration - current_duration
            end_silence = AudioSegment.silent(duration=remaining_duration * 1000)
            padded_audio = initial_silence + audio + end_silence
        else:
            # If audio is longer, truncate it to fit the available duration
            truncated_audio = audio[:available_duration * 1000]
            padded_audio = initial_silence + truncated_audio

        # Ensure exact duration by creating a new audio segment of desired duration
        final_audio = AudioSegment.silent(duration=desired_duration * 1000)
        final_audio = final_audio.overlay(padded_audio)

        final_audio.export(os.path.join(output_folder, f"freesound_{i}.wav"), format="wav")

    def get_audio_files(self, folder, target_folder):
        """
        Get all audio files in a folder and subfolders
        """

        audio_files = []
        supported_formats = ['wav']
        target_subfolder_path = os.path.join(folder, target_folder)

        for root, _, files in os.walk(folder):
            if root == target_subfolder_path:
                for file in files:
                    if file.split('.')[-1] in supported_formats:
                        path = os.path.join(root, file)
                        audio_files.append(path)
                break

        return audio_files

    def gen_audio_files(self):
        """
        Generate audio files for Scaper
        """
        foreground_files = self.get_audio_files(self.foreground_folder, self.target_fg_folder)

        foreground_files = sorted(foreground_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        j = 20

        self.target_bg_folder = self.freesound_labels[j]

        background_files = self.get_audio_files(self.background_folder, self.target_bg_folder)

        for i in range(2000, 4000):
            sc = scaper.Scaper(
                self.soundscape_duration, 
                self.foreground_folder, 
                self.background_folder, 
                random_state = self.seed
            )

            label_vector =  torch.zeros(len(self.freesound_labels))
            label_vector[j] = 1

            torch.save(label_vector, os.path.join(self.output_labels_folder, f"label_{i}.pt"))

            sc.ref_db = -20
            sc.duration = 10

            chosen_background_file = background_files[int(np.random.randint(0, len(background_files)))]

            self.pad_or_truncate_audio(chosen_background_file, 10, self.output_freesound_folder, i)

            padded_bg_file = os.path.join(self.output_freesound_folder, f"freesound_{i}.wav")

            sc.add_background(label=('const', self.target_bg_folder),
                        source_file=('choose', [padded_bg_file]),
                        source_time=('const', 0),
            )

            chosen_foreground_file =  foreground_files[i]

            sc.add_event(label=('const', '3spk_mix'),
                        source_file=('choose', [chosen_foreground_file]),
                        source_time=('const', 0),
                        event_time=('const', 3),
                        event_duration=('const', 10),
                        snr=('uniform', 15.0, 20.0),
                        pitch_shift=None,
                        time_stretch=None)

            audiofile = os.path.join(self.mixed_output_folder, f"fmix_{i}.wav")
            jamsfile = os.path.join(self.mixed_output_folder, 'mysoundscape.jams')
            txtfile = os.path.join(self.mixed_output_folder, 'mysoundscape.txt')

            sc.generate(audiofile, jamsfile,
                    allow_repeated_label=True,
                    allow_repeated_source=True,
                    reverb=0.1,
                    disable_sox_warnings=True,
                    no_audio=False,
                    peak_normalization=True,
                    txt_path=txtfile)
            
            if i % 100 == 0:
                print(f"Done with {self.freesound_labels[j]}")
                j += 1
                self.target_bg_folder = self.freesound_labels[j]
                background_files = self.get_audio_files(self.background_folder, self.target_bg_folder)

        
if __name__ == "__main__":

    foreground_folder = "3speaker_mixtures"
    background_folder = "freesound"
    base_output_folder = os.path.join("3speaker_mixtures")

    os.makedirs(base_output_folder, exist_ok=True)

    Audio_generator = ScaperSourceGenerator(foreground_folder, background_folder, base_output_folder)

    Audio_generator.gen_audio_files()
    