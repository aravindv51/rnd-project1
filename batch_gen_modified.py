import os
import time
from tqdm import tqdm
import openai
from openai import OpenAI
from bert_score import score
import pandas as pd

# Initialize OpenAI client
client = OpenAI(api_key=getkey())

# Root input and output directories
input_root = "indian_express_editorials"
output_root = "indian_express_editorials_objective"
bert_score_file = os.path.join(output_root, "bert_scores.csv")

# Retry wrapper for API
def call_openai_with_retry(prompt, max_retries=5, delay=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise e

# Optional: Retry wrapper for BERTScore (rarely needed)
def compute_bertscore_with_retry(cands, refs, max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            P, R, F1 = score(cands, refs, lang="en", verbose=False)
            return round(F1[0].item(), 4)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                raise e

# Prepare progress
input_files = []
for root, _, files in os.walk(input_root):
    for file in files:
        if file.endswith(".txt"):
            input_files.append(os.path.join(root, file))

# Store scores
bert_records = []

# Process each file
for input_path in tqdm(input_files, desc="Processing files"):
    # Read input
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    # Validate format
    if not lines or not lines[0].lower().startswith("title:"):
        continue

    title = lines[0][len("title:"):].strip()
    try:
        blank_index = lines.index('')
    except ValueError:
        continue

    passage = "\n".join(lines[blank_index + 1:]).strip()

    # Skip empty passages
    if not passage:
        continue

    # Prepare prompt
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
{passage}

Output:
"""

    # API call with retry
    try:
        objective_passage = call_openai_with_retry(prompt)
    except Exception as e:
        print(f"\n Failed for {input_path}: {e}")
        continue

    # Compute BERTScore
    try:
        bert_f1 = compute_bertscore_with_retry([objective_passage], [passage])
    except Exception as e:
        print(f"\n  BERTScore failed for {input_path}: {e}")
        bert_f1 = None

    # Prepare output path
    relative_path = os.path.relpath(input_path, input_root)
    base_name, ext = os.path.splitext(relative_path)
    output_path = os.path.join(output_root, base_name + "_obj" + ext)

    # Create output directories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write objective file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"title: {title}\n\n{objective_passage.strip()}\n")

    # Record BERTScore
    bert_records.append({
        "file": relative_path,
        "bert_f1": bert_f1
    })

# Save BERT scores
os.makedirs(output_root, exist_ok=True)
pd.DataFrame(bert_records).to_csv(bert_score_file, index=False)
print(f"\n All done. BERT scores saved to: {bert_score_file}")
