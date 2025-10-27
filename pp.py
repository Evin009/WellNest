"PART-II TRANSCRIBING WITH WHISPER"

import whisper

# Loading whisper model
whisper_model = whisper.load_model("small", device="cpu")

# transcribing function
def transcribe_audio(audio_file):
    # empty input/ no recording
    if not audio_file:
        return "No audio provided."
    # transcribing and saving in result variable
    result = whisper_model.transcribe(audio_file)
    # returning result
    return result.get("text", "")

# UI for audio transcribing
with gr.Blocks() as transcribe_ui:
    # heading
    gr.Markdown("### Transcribe")
    # transcribe btn
    transcribe_btn = gr.Button("Click here to Transcribe")
    # Output Box
    transcript_box = gr.Textbox(
        label="Transcribed Text Output",
        interactive=False,
        placeholder="Voice to text"
    )

    # onlclick action
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=transcript_box
    )

transcribe_ui.launch()

"PART - III IMAGE GENERATION"

from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch


# initilizing for mac
device = "mps" 

# Loading the pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "stablediffusionapi/anything-v5", torch_dtype=torch.float32
).to(device)

pipe.enable_attention_slicing()
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# prompt refining due to token restriction (75-77)
def clean_prompt(prompt, max_len=200):
    return prompt[:max_len] if prompt else ""

# image generation fn
def generate_image(prompt, steps=40, scale=7.5):
    # passing prompt refining
    short_prompt = clean_prompt(prompt)
    if short_prompt == "":
        return None
    styled_prompt = f"In Studio Ghibli style, {short_prompt}, highly detailed, 8K resolution"
    negative_prompt = "blurry, low quality, bad anatomy, extra limbs, distorted, text, watermark"

    image = pipe(
        prompt=styled_prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        num_inference_steps=steps,
        guidance_scale=scale
    ).images[0]
    return image

# UI component
with gr.Blocks() as generate_ui:
    gr.Markdown("### 🎨 3) Image Generation")
    steps_slider = gr.Slider(
        minimum=10, maximum=80, value=40, step=5,
        label="Inference Steps (Quality vs Speed)"
    )
    scale_slider = gr.Slider(
        minimum=5.0, maximum=15.0, value=7.5, step=0.5,
        label="Guidance Scale (Prompt Strength)"
    )

    gen_btn = gr.Button("Generate Image")
    image_out = gr.Image(label="Generated Image")

    #  on click
    gen_btn.click(
        fn=generate_image,
        inputs=[transcript_box, steps_slider, scale_slider],
        outputs=image_out
    )
