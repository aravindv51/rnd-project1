import os
import csv
from openai import OpenAI
import bert_score

# === Configuration ===
openai_api_key = getkey()
input_folder = "indian_express_opinions/indian_express_opinions"
output_folder = "output_samples"
os.makedirs(output_folder, exist_ok=True)
score_csv = "bert_scores.csv"

# === OpenAI Client ===
client = OpenAI(api_key=openai_api_key)

# === Prompt Builder ===
def make_prompt(text):
    return f"""
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
{text}

Output:
"""

# === CSV setup ===
with open(score_csv, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "bert_score"])

    # === Loop through all .txt files ===
    for filename in os.listdir(input_folder):
        if not filename.endswith(".txt"):
            continue

        input_path = os.path.join(input_folder, filename)

        # Read and parse file content
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

        title_line = lines[0].strip()
        assert title_line.lower().startswith("title:"), f"{filename} must start with 'title:'"
        title = title_line[len("title:"):].strip()

        # Find blank line and read passage
        blank_line_index = lines.index('')
        passage_lines = lines[blank_line_index + 1:]
        input_passage = "\n".join(passage_lines).strip()

        # Build and send prompt
        prompt = make_prompt(input_passage)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        objective_text = response.choices[0].message.content.strip()

        # Save output to new file
        output_filename = os.path.splitext(filename)[0] + "_obj.txt"
        output_path = os.path.join(output_folder, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"title: {title}\n\n{objective_text}")

        # Compute BERTScore (F1)
        _, _, F1 = bert_score.score([objective_text], [input_passage], lang="en", verbose=False)
        score = round(F1[0].item(), 4)

        # Log result
        print(f"✅ {filename} → {output_filename} | BERTScore: {score}")
        writer.writerow([filename, score])
