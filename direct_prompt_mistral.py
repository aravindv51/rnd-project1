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
    Task Description:
    You are an AI tasked with transforming subjective paragraphs into their objective versions. Your output must retain only factual, observable, or verifiable information explicitly stated in the original text. Your goal is to neutralize the tone without introducing new content or removing essential meaning.

    Here is the sample example pair:
    Subjective version:
    The conference was a resounding success, drawing brilliant minds from around the globe to share groundbreaking ideas. The keynote speaker delivered an inspiring talk that left the audience energized and hopeful. Attendees were treated to a beautifully organized event with seamless logistics and exceptional hospitality. Many left feeling intellectually enriched and personally motivated. It was, without a doubt, one of the most memorable gatherings of the year.
    Objective version:
    The conference included participants from multiple countries who shared their ideas. A keynote speaker gave a talk. The event included planned logistics and hospitality. Attendees reported intellectual benefits. Many described the event as notable.

    Constraints:
    Factual Grounding: Do not invent or infer any details that are not present in the input text. Avoid assumptions, generalizations, or added interpretations.
    Neutral Tone: Remove emotionally charged, evaluative, or opinionated language such as adjectives, adverbs, metaphors, and subjective expressions. If they are very much needed, try to neutralize tone of that word
    Preserve Meaning: Only remove or rephrase subjective words when their absence does not alter the core meaning of the sentence.
    No Additions: Do not expand, embellish, or reinterpret the original content. Only return a strictly objective version of the input text. Try avoid using the unnecessary adverbs and adjectives and subjective words in the generated text. 
    Sentence-Level Clarity: Rewrite the paragraph as a coherent whole, not as bullet points or sentence fragments. Maintain grammatical correctness and logical flow.
    Data consistency: Reflect all the facts and data statistics in the original content. Do not miss any of the original content until it is redundant

    Instruction:
    Generate an objective version of the given passage. It should be very neutral tone with unnecessary subjective words removed and the subjectivity reduced. Do not include any explanations, reasoning, or additional content beyond the target objective text. The objective version should cover all information in the subjective version.

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
Following Jannik Sinner’s 3-6, 3-6, 6-4, 6-4, 6-4 comeback win over Daniil Medvedev to win a maiden Grand Slam title at the Australian Open on Sunday, men’s tennis players born in the 2000s have now won more majors (three) than those born in the 1990s (two). Sinner, the unassuming 22-year-old World No 4 from Italy, joins 20-year-old Spaniard Carlos Alcaraz — who won two majors over the past year — as members of the generation that has captured tennis fans’ imagination and leapfrogged the generation of players above it.
The sport is unquestionably going through a transitional moment. The fading of the Federer-Nadal duopoly — the Spaniard will still be planning one last big push for the French Open where he is a 13-time champion this year — has allowed Novak Djokovic to establish himself as the best player in the world. But now at 36, the Serb is facing physical and technical issues of his own, and while the generation below him, including Sunday’s finalist Medvedev, who became the first player in history to fail to win a major final from a two-set lead twice, has failed to take advantage of his lapses, Gen Z are facing no such issue.Sinner defeated Djokovic in the semifinal in Melbourne this week, while Alcaraz defeated him in a memorable Wimbledon final last year. Both have well-rounded attacking games, and both showed the mental resilience to deal with the pressure of playing five sets in a Grand Slam final. Djokovic remains the leader of this pack. He lost his first Wimbledon match since 2017 against Alcaraz in the final last year, and bounced back to end his year on a record high. As 10-time champion of Melbourne Park, unbeaten there since 2018 until Sinner took him down this year, expect him to return with a similar fury. Also expect the fiery Spaniard and the laid-back Italian to resist.
'''
 
print("Corrected:", correct_grammar(sentence))

# #apply the above mistral based grammar correction model on the dataset
# data = pd.read_excel("parallel_sent_new.xlsx")

# data['CORRECTED SENTENCES'] = data['OBJECTIVE SENTENCES'].progress_apply(correct_grammar)

# #store this in a corrected sentences excel file
# data.to_excel("parallel_final.xlsx")