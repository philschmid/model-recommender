import json
import gradio as gr
from recommender.main import get_recommendation


def greet(model_id):
    configs = get_recommendation(model_id)
    return json.dumps(configs)


demo = gr.Interface(
    fn=greet,
    inputs=[
        gr.Textbox(label="Model ID", placeholder="meta-llama/Llama-2-7b-chat-hf"),
        # gr.Slider(
        #     step=4000,
        #     minimum=16_000,
        #     maximum=640_000,
        #     value=24_000,
        #     label="GPU memory",
        #     info="Select how much GPU memory you have available",
        # ),
    ],
    outputs=[gr.JSON()],
)

demo.launch()
