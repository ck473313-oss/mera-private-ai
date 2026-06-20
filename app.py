import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("[+] Loading uncensored model...")

model_name = "cognitivecomputations/WizardLM-7B-Uncensored"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def chat(message, history):
    inputs = tokenizer.encode(message, return_tensors="pt")
    # Agar GPU hai to wahan bhejo
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    outputs = model.generate(
        inputs,
        max_length=800,
        temperature=0.7,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Sirf answer dikhao
    if message in response:
        response = response.replace(message, "").strip()
    
    return response

# Launch karo
gr.ChatInterface(
    chat,
    title="Private Unfiltered AI",
    description="Koi restriction nahi. Kuch bhi poochh."
).launch(server_name="0.0.0.0", server_port=10000)
