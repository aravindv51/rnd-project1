def load_annotations(ann_file_path):
    sentence_spans = []
    subjective_spans = []

    with open(ann_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue 
            parts = line.split()
            if not parts:
                continue
            start, end = int(parts[1].split(',')[0]), int(parts[1].split(',')[1])
            tag_type = parts[3]

            if tag_type == 'GATE_inside':
                sentence_spans.append((start, end))
            elif tag_type in ['GATE_direct-subjective', 'GATE_expressive-subjectivity']:
                subjective_spans.append((start, end))
    return sentence_spans, subjective_spans

def extract_subjective_sentences(text_file, ann_file):
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    sentence_spans, subjective_spans = load_annotations(ann_file)
    subjective_sentences = []

    for sent_start, sent_end in sentence_spans:
        for subj_start, subj_end in subjective_spans:
            if subj_start >= sent_start and subj_end <= sent_end:
                sentence = text[sent_start:sent_end].strip()
                subjective_sentences.append(sentence)
                break  # No need to check other subjective spans for this sentence
    return subjective_sentences

text_file = 'database.mpqa.2.0/docs/20010620/13.40.05-15087'
ann_file = 'database.mpqa.2.0/man_anns/20010620/13.40.05-15087/gateman.mpqa.lre.2.0'

subjective_sents = extract_subjective_sentences(text_file, ann_file)

for sent in subjective_sents:
    print(sent)