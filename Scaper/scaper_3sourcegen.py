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
        self.target_fg_folder = "midlands_female"
        self.target_bg_folder = "midlands_english_male"

        self.output_fg_folder = os.path.join(base_output_folder, "Spk1")
        self.output_fg2_folder = os.path.join(base_output_folder, "Spk2")
        self.output_bg_folder = os.path.join(base_output_folder, "Spk3")

        self.soundscape_duration = 6
        self.mixed_output_folder = os.path.join(base_output_folder, "3spk_mix")
        self.seed = 123

    def pad_or_truncate_audio(self, audio_path, desired_duration, output_folder, i, category, start_time=0):
        """
        Pad or truncate an audio file to the desired duration in seconds.
        Args:
            audio_path: Path to input audio file
            desired_duration: Total duration of output audio in seconds
            output_folder: Folder to save the output
            i: Index for the output filename
            category: Type of audio (fg1, fg2, or bg)
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

        if category == "fg1":
            final_audio.export(os.path.join(output_folder, f"Spk1_{i}.wav"), format="wav")
        elif category == "fg2":
            final_audio.export(os.path.join(output_folder, f"Spk2_{i}.wav"), format="wav")
        elif category == "bg":
            final_audio.export(os.path.join(output_folder, f"Spk3_{i}.wav"), format="wav")

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
            # First pad/truncate all audio files
            chosen_background_file = background_files[int(np.random.randint(len(background_files)))]
            chosen_foreground_file1 = foreground_files[int(np.random.randint(len(foreground_files)))]
            
            # Get a different foreground file for speaker 2
            chosen_foreground_file2 = chosen_foreground_file1
            while chosen_foreground_file2 == chosen_foreground_file1:
                chosen_foreground_file2 = foreground_files[int(np.random.randint(len(foreground_files)))]

            # Pad/truncate all files first
            self.pad_or_truncate_audio(chosen_background_file, 8, self.output_bg_folder, i, "bg")
            self.pad_or_truncate_audio(chosen_foreground_file1, 8, self.output_fg_folder, i, "fg1", start_time=3)
            self.pad_or_truncate_audio(chosen_foreground_file2, 8, self.output_fg2_folder, i, "fg2", start_time=2)

            # Now use the padded files for mixing
            padded_bg_file = os.path.join(self.output_bg_folder, f"Spk3_{i}.wav")
            padded_fg1_file = os.path.join(self.output_fg_folder, f"Spk1_{i}.wav")
            padded_fg2_file = os.path.join(self.output_fg2_folder, f"Spk2_{i}.wav")

            sc = scaper.Scaper(
                self.soundscape_duration, 
                self.foreground_folder, 
                self.background_folder, 
                random_state = self.seed
            )

            sc.ref_db = -20
            sc.duration = 8

            # Add background using padded file
            sc.add_background(label=('const', 'midlands_english_male'),
                        source_file=('choose', [padded_bg_file]),
                        source_time=('const', 0),
            )

            # Add first foreground using padded file
            sc.add_event(label=('const', 'midlands_female'),
                        source_file=('choose', [padded_fg1_file]),
                        source_time=('const', 0),
                        event_time=('const', 3),
                        event_duration=('const', 8),
                        snr=('uniform', 15.0, 20.0),
                        pitch_shift=None,
                        time_stretch=None)
            
            # Add second foreground using padded file
            sc.add_event(label=('const', 'midlands_female'),
                        source_file=('choose', [padded_fg2_file]),
                        source_time=('const', 0),
                        event_time=('const', 2),
                        event_duration=('const', 8),
                        snr=('uniform', 15.0, 20.0),
                        pitch_shift=None,
                        time_stretch=None)

            audiofile = os.path.join(self.mixed_output_folder, f"3spkmix_{i}.wav")
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
    base_output_folder = os.path.join("3speaker_mixtures")

    os.makedirs(base_output_folder, exist_ok=True)

    Audio_generator = ScaperSourceGenerator(foreground_folder, background_folder, base_output_folder)

    Audio_generator.gen_audio_files()
    