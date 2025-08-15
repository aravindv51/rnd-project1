import openai
import os
import re
import time
# import tiktoken

from openai import OpenAI

# Set up client (you can set key via env var or insert directly here)

# Input file
input_filename = "energy_editorial_8.txt"

# Read and extract title + passage
with open(input_filename, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

title_line = lines[0].strip()
assert title_line.lower().startswith("title:"), "File must start with 'title:'"
title = title_line[len("title:"):].strip()

blank_line_index = lines.index('')
passage_lines = lines[blank_line_index + 1:]
joined_text = "\n".join(passage_lines).strip()

# Optional: truncate at "Views expressed above" if present
split_match = re.split(r'Views expressed above', joined_text, flags=re.IGNORECASE)
input_passage = split_match[0].strip() if split_match else joined_text

# Prompt template
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

# # Token counter
# def count_tokens(text, model="gpt-4"):
#     enc = tiktoken.encoding_for_model(model)
#     return len(enc.encode(text))

# print(f"🔢 Prompt token count: {count_tokens(prompt)}")

# Retry logic
def get_response_with_retry(prompt, max_retries=4, timeout=90):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                timeout=timeout
            )
        except openai.APITimeoutError:
            print(f"⏱️ Timeout on attempt {attempt + 1}. Retrying...")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"❌ Error: {e}")
            break
    return None

# Call API
response = get_response_with_retry(prompt)

# Handle output
if response:
    objective_text = response.choices[0].message.content.strip()
    base, ext = os.path.splitext(input_filename)
    output_filename = f"{base}_obj{ext}"

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(f"title: {title}\n\n{objective_text}\n")

    print(f"✅ Output written to: {output_filename}")
else:
    print("❌ Failed to generate response after retries.")
