import gradio as gr
from utils.inference import predict_condition

with gr.Blocks() as demo:
    gr.Markdown("# 💛 Skin Cancer Detection Chatbot")

    with gr.Row():
        image_input = gr.Image(type="numpy", label="Upload Skin Lesion Image")
        output_text = gr.Textbox(label="Prediction & Info")

    submit_btn = gr.Button("Diagnose")
    submit_btn.click(fn=predict_condition, inputs=image_input, outputs=output_text)

demo.launch()

