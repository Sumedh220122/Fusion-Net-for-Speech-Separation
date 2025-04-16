import os
import warnings
import scaper
import numpy as np
from pydub import AudioSegment

warnings.filterwarnings("ignore")

class ScaperSourceGenerator:
    """
    Generate source files for Scaper
    """
    def __init__(self, foreground_folder, background_folder, base_output_folder):
        self.foreground_folder = foreground_folder
        self.background_folder = background_folder
        self.target_fg_folder = "irish_male"
        self.target_bg_folder = "northern_english_female"

        self.output_fg_folder = os.path.join(base_output_folder, "Spk1")
        self.output_bg_folder = os.path.join(base_output_folder, "Spk2")

        self.soundscape_duration = 6
        self.mixed_output_folder = os.path.join(base_output_folder, "2spk_mix")
        self.seed = 123

    def pad_or_truncate_audio(self, audio_path, desired_duration, output_folder, i, category):
        """
        Pad or truncate an audio file to the desired duration in seconds.
        """
        audio = AudioSegment.from_file(audio_path)
        current_duration = len(audio) // 1000  # Convert milliseconds to seconds

        if current_duration < desired_duration:
            # Pad with silence
            silence = AudioSegment.silent(duration=(desired_duration - current_duration) * 1000)
            padded_audio = audio + silence
            if category == "fg":
                padded_audio.export(os.path.join(output_folder, f"Spk1_{i}.wav"), format="wav")
            elif category == "bg":
                padded_audio.export(os.path.join(output_folder, f"Spk2_{i}.wav"), format="wav")
        elif current_duration >= desired_duration:
            # Truncate
            truncated_audio = audio[:desired_duration * 1000]
            if category == "fg":
                truncated_audio.export(os.path.join(output_folder, f"Spk1_{i}.wav"), format="wav")
            elif category == "bg":
                truncated_audio.export(os.path.join(output_folder, f"Spk2_{i}.wav"), format="wav")

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
        background_files = self.get_audio_files(self.background_folder, self.target_bg_folder)

        for i in range(2000, 4000):
            sc = scaper.Scaper(
                self.soundscape_duration, 
                self.foreground_folder, 
                self.background_folder, 
                random_state = self.seed
            )

            sc.ref_db = -20
            sc.duration = 6


            chosen_background_file = background_files[int(np.random.randint(len(background_files)))]

            self.pad_or_truncate_audio(chosen_background_file, sc.duration, self.output_bg_folder, i, "bg")

            sc.add_background(label=('const', 'northern_english_female'),  # Ensure 'cat' is a label in your background folder
                        source_file=('choose', [chosen_background_file]),
                        source_time=('const', 0),
            )

            chosen_foreground_file =  foreground_files[int(np.random.randint(len(foreground_files)))]

            self.pad_or_truncate_audio(chosen_foreground_file, sc.duration, self.output_fg_folder, i, "fg")

            sc.add_event(label=('const', 'irish_male'),  # Ensure 'People' is a label in your foreground folder
                        source_file=('choose', [chosen_foreground_file]),
                        source_time=('const', 0),
                        event_time=('const', 0),
                        event_duration=('const', 6),
                        snr=('uniform', 15.0, 20.0),
                        pitch_shift=None,
                        time_stretch = None)

            audiofile = os.path.join(self.mixed_output_folder, f"2spkmix_{i}.wav")
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

        
if __name__ == "__main__":

    foreground_folder = "foreground"
    background_folder = "background"
    base_output_folder = os.path.join("output")

    os.makedirs(base_output_folder, exist_ok=True)

    Audio_generator = ScaperSourceGenerator(foreground_folder, background_folder, base_output_folder)

    Audio_generator.gen_audio_files()
    