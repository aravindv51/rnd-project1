# gradio_app.py
import gradio as gr
import requests

def call_remote_function(user_input):
    response = requests.post("http://10.195.100.6:5000/process", json={"input": user_input})
    return response.json()["result"]

interface = gr.Interface(fn=call_remote_function, inputs="text", outputs="text")
interface.launch()
