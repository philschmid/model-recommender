import json
import gradio as gr
from recommender.main import get_tgi_config


def greet(model_id, gpu_memory, num_gpus):
    try:
        configs = get_tgi_config(model_id, gpu_memory, num_gpus)
    except Exception as e:
        return json.dumps({"error": str(e)})
    if configs is None:
        return json.dumps({"error": f"Couldn't generate TGI config for {model_id}"})
    return json.dumps(configs)


theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[
        gr.themes.GoogleFont("Open Sans"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
)
DESCRIPTION = """
<div style="text-align: center; max-width: 650px; margin: 0 auto; display:grid; gap:25px;">
    <h1 style="font-weight: 900; margin-bottom: 7px;margin-top:5px">
       Hugging Face TGI Configuration Creator
    </h1> 
    <p style="margin-bottom: 10px; font-size: 94%; line-height: 23px;">
    This Space helps you generate and validate Hugging Face TGI configurations for your model. Provide you model ID and the amount of GPU memory you have available and we will generate a configuration for you, which you can use to run your model on TGI.
    </p>
</div>
"""


demo = gr.Interface(
    fn=greet,
    description=DESCRIPTION,
    inputs=[
        gr.Textbox(label="Model ID", placeholder="meta-llama/Llama-2-7b-chat-hf"),
        gr.Slider(
            step=4,
            minimum=16,
            maximum=640,
            value=24,
            label="GPU memory",
            info="Select how much GPU memory you have available",
        ),
        gr.Slider(
            step=1,
            minimum=1,
            maximum=8,
            value=1,
            label="# of GPUs",
            info="Select how many GPUs you have available",
        ),
    ],
    theme=theme,
    outputs=[gr.JSON()],
)

demo.launch()
