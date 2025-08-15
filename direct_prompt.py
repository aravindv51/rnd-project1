import os
import pandas as pd
import tqdm

from huggingface_hub import login
login(hf_token)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#for progress bar
from tqdm import tqdm
tqdm.pandas()


model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def correct_grammar(sentence):
    prompt = f"Rewrite this paragraph to make it objective and neutral, keeping structure similar: {sentence}\nFixed sentence:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:7" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(**inputs, max_length=500, pad_token_id=tokenizer.eos_token_id )
    decoded =  tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Fixed sentence:" in decoded:
        fixed_part = decoded.split("Fixed sentence:")[-1].strip()
    else:
        fixed_part = decoded.strip()  # Fallback in case the model omits prefix

    return fixed_part

sentence = "The prime minister’s decision to dissolve parliament has generated debate across the nation. Critics state that the decision is an attempt to maintain authority, while supporters describe it as a step to restore order. The announcement has led to protests, with many expressing concern about its implications for democratic norms."
print("Corrected:", correct_grammar(sentence))

# #apply the above mistral based grammar correction model on the dataset
# data = pd.read_excel("parallel_sent_new.xlsx")

# data['CORRECTED SENTENCES'] = data['OBJECTIVE SENTENCES'].progress_apply(correct_grammar)

# #store this in a corrected sentences excel file
# data.to_excel("parallel_final.xlsx")