import gradio as gr
from FusionNet1 import FusionNet, TARGETS
import os

def process_audio(audio_file, selected_sounds):
    # Initialize the model
    model = FusionNet()
    model.load_state_dict()
    
    # Process the audio
    outputs = model.separate_audio(audio_file, selected_sounds, num_spk=2)
    
    # Return the separated audio files
    return outputs[0], outputs[1], outputs[2]

# Create the Gradio interface
with gr.Blocks(title="Audio Separation Interface") as demo:
    gr.Markdown("# Audio Separation Interface")
    gr.Markdown("Upload an audio file and select the sounds you want to separate.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Input Audio", type="filepath")
            sound_choices = gr.CheckboxGroup(
                choices=TARGETS,
                label="Select sounds to separate",
                info="Choose the sounds you want to separate from the audio"
            )
            process_btn = gr.Button("Process Audio")
        
        with gr.Column():
            background_output = gr.Audio(label="Background/Noise")
            speaker1_output = gr.Audio(label="Speaker 1")
            speaker2_output = gr.Audio(label="Speaker 2")
    
    process_btn.click(
        fn=process_audio,
        inputs=[audio_input, sound_choices],
        outputs=[background_output, speaker1_output, speaker2_output]
    )

if __name__ == "__main__":
    # Create Predictions directory if it doesn't exist
    os.makedirs("Predictions", exist_ok=True)
    
    # Launch the Gradio interface
    demo.launch(share=True) 