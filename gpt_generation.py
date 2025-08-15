import openai
import os
import re

# For OpenAI v1.x: use the `OpenAI` client
from openai import OpenAI

# Initialize client with your API key
client = OpenAI(api_key=getkey())

# Input filename
input_filename = "energy_editorial_8.txt"

# Read and parse file content
with open(input_filename, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

# Extract title and passage
title_line = lines[0].strip()
assert title_line.lower().startswith("title:"), "File must start with 'title:'"

title = title_line[len("title:"):].strip()

blank_line_index = lines.index('')
passage_lines = lines[blank_line_index + 1:]
joined_text = "\n".join(passage_lines).strip()
input_passage = re.split(r'Views expressed above', joined_text)[0]

# Construct the prompt
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

Instruction:
Generate an objective version of the given passage. It should be very neutral tone with unnecessary subjective words removed and the subjectivity reduced. Do not include any explanations, reasoning, or additional content beyond the target objective text. The objective version should cover all information in the subjective version.

Input:
{input_passage}

Output:
"""

# Make the API call
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.3,
)

# Extract the generated text
objective_text = response.choices[0].message.content.strip()

# Write output to new file
base, ext = os.path.splitext(input_filename)
output_filename = f"{base}_obj{ext}"

with open(output_filename, 'w', encoding='utf-8') as f:
    f.write(f"title: {title}\n\n{objective_text}\n")

print(f"✅ Output written to: {output_filename}")
