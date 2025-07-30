# !pip install -q transformers accelerate gradio # install these before running the code

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Load model
model_name = "utk-2oo4/DeepShiva-finetuned-model"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
# Chat function
def chat_interface(message, history):
    system_prompt = "<|system|>\nYou are an assistant who replies to agricultural related question.\n"
    history_prompt = ""

    for user, bot in history:
        history_prompt += f"<|user|>\n{user}\n<|assistant|>\n{bot}\n"

    full_prompt = system_prompt + history_prompt + f"<|user|>\n{message}\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = decoded_output.split("<|assistant|>")[-1].strip()

    return reply

# Gradio UI
gr.ChatInterface(fn=chat_interface,
                 title="DeepShiva Chatbot",
                 description="Ask anything to the DeepShiva model.",
                 chatbot=gr.Chatbot(height=500),
                 theme="soft").launch()
