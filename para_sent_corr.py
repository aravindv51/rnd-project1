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

def correct_grammar(context):
    # Adjust the prompt to reflect the task of improving fluency while keeping the meaning
    prompt = f"""
    Task:
    You are given a text where subjective language (e.g., adjectives, adverbs) has been removed. Your task is to rephrase the sentence to improve fluency and coherence, ensuring that the original meaning is preserved. Do not introduce any new adjectives, adverbs, or subjective phrasing. Reduce subjectivity and make paragrph more neutral toned
    
    Instruction:
    - Focus on improving sentence fluency and logical flow.
    - Retain the exact meaning of the original passage.
    - Avoid adding any subjective language or unnecessary descriptors.
    - Ensure the rephrased passage is clear, coherent, and grammatically correct.
    - Try to reduce the subjective tone and make it more neutral
    Input:
    {context}
    
    Output:
    """

    # Tokenize the input and move it to the appropriate device (GPU or CPU)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:7" if torch.cuda.is_available() else "cpu")

    # Generate output from the model
    outputs = model.generate(**inputs, max_length=1500, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the output to a readable string
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Ensure the model has produced an output section
    if "Output:" in decoded:
        fixed_part = decoded.split("Output:")[-1].strip()
    else:
        fixed_part = decoded.strip()  # If the model omits the "Output:" label, fallback to the entire decoded string

    # Return the final fixed sentence
    return fixed_part


sentence = '''
Following Jannik Sinner ’s 3 - 6 , 3 - 6 , 6 - 4 , 6 - 4 , 6 - 4 comeback win over Daniil Medvedev to win a maiden Grand Slam title at the Australian Open on Sunday , men ’s tennis players born in the 2000s have now won majors ( three ) than those born in the 1990s ( two ) . 
 Sinner , the 22 - old World No 4 from Italy , joins 20 - old Spaniard Carlos Alcaraz — who won two majors over the past year — as members of the generation that has captured tennis fans ’ imagination and leapfrogged the generation of players above it . 
 The sport is going through a moment . The fading of the Federer - Nadal duopoly — the Spaniard will still be planning one last push for the French Open where he is a 13 - time champion this year — has allowed Novak Djokovic to establish himself as the player in the world . But now at 36 , the Serb is facing and issues of his , and while the generation below him , including Sunday ’s finalist Medvedev , who became the first player in history to fail to win a final from a two - set lead , has failed to take advantage of his lapses , Gen Z are facing no issue . 
 Sinner defeated Djokovic in the semifinal in Melbourne this week , while Alcaraz defeated him in a Wimbledon final last year . Both have attacking games , and both showed the resilience to deal with the pressure of playing five sets in a Grand Slam final . 
 Djokovic remains the leader of this pack . He lost his first Wimbledon match since 2017 against Alcaraz in the final last year , and bounced to end his year on a record high . As 10 - time champion of Melbourne Park , since 2018 until Sinner took him down this year , expect him to return with a fury . expect the Spaniard and the laid - back to resist .
'''
 
print("Corrected:", correct_grammar(sentence))

# #apply the above mistral based grammar correction model on the dataset
# data = pd.read_excel("parallel_sent_new.xlsx")

# data['CORRECTED SENTENCES'] = data['OBJECTIVE SENTENCES'].progress_apply(correct_grammar)

# #store this in a corrected sentences excel file
# data.to_excel("parallel_final.xlsx")