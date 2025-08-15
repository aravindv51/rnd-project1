import spacy
import string
import pandas as pd
from textblob import TextBlob

#these subjective words are context related and connot be removed
adverbs_of_time = [
    "before", "after", "soon", "earlier", "recently", "tomorrow",
    "yesterday", "now", "then", "already", "just", "immediately",
    "finally", "still", "yet", "lately", "nowadays", "today",
    "eventually", "thereafter", "ago"
]

number_adjectives = [
    "one", "two", "three", "several", "few", "many",
    "hundred", "thousand", "single", "multiple"
]

position_adjectives = [
    "first", "second", "third", "fourth", "last", "next",
    "previous", "middle", "final", "top", "bottom",
    "beginning", "end", "upper", "lower"
]

adverbs_with_difficulty = [
    "barely", "hardly", "rarely", "merely", "scarcely", "sparsely", "faintly"
]

descriptive_adjectives = [
    "decent", "good", "bad", "high", "low", "near", "far",
    "past", "present", "future", "human", "mediocre", "heavy",
    "light", "lightweight", "new", "old", "mostly", "recent",
    "recently", "most", "unique", "least", "highest", 
    "north", "south", "east", "west",
    "northern", "southern", "eastern", "western",
    "northeast", "northwest", "southeast", "southwest",
    "northeastern", "northwestern", "southeastern", "southwestern"
]


country_adjectives = [
    "Afghan", "Albanian", "Algerian", "American", "Andorran", "Angolan", "Argentine", "Armenian",
    "Australian", "Austrian", "Azerbaijani", "Bahraini", "Bangladeshi", "Barbadian", "Belarusian",
    "Belgian", "Belizean", "Beninese", "Bhutanese", "Bolivian", "Bosnian", "Brazilian", "British",
    "Bruneian", "Bulgarian", "Burkinabe", "Burmese", "Burundian", "Cambodian", "Cameroonian", "Canadian",
    "Cape Verdean", "Central African", "Chadian", "Chilean", "Chinese", "Colombian", "Comorian",
    "Congolese", "Costa Rican", "Croatian", "Cuban", "Cypriot", "Czech", "Danish", "Djiboutian",
    "Dominican", "Dutch", "East Timorese", "Ecuadorean", "Egyptian", "Emirati", "English", "Equatorial Guinean",
    "Eritrean", "Estonian", "Ethiopian", "Fijian", "Finnish", "French", "Gabonese", "Gambian", "Georgian",
    "German", "Ghanaian", "Greek", "Grenadian", "Guatemalan", "Guinean", "Guyanese", "Haitian", "Honduran",
    "Hungarian", "Icelandic", "Indian", "Indonesian", "Iranian", "Iraqi", "Irish", "Israeli", "Italian",
    "Ivorian", "Jamaican", "Japanese", "Jordanian", "Kazakh", "Kenyan", "Kiribati", "Korean", "Kosovar",
    "Kuwaiti", "Kyrgyz", "Laotian", "Latvian", "Lebanese", "Liberian", "Libyan", "Liechtensteiner",
    "Lithuanian", "Luxembourgish", "Macedonian", "Malagasy", "Malawian", "Malaysian", "Maldivian", "Malian",
    "Maltese", "Marshallese", "Mauritanian", "Mauritian", "Mexican", "Micronesian", "Moldovan", "Monegasque",
    "Mongolian", "Montenegrin", "Moroccan", "Mozambican", "Namibian", "Nauruan", "Nepalese", "New Zealander",
    "Nicaraguan", "Nigerien", "Nigerian", "Norwegian", "Omani", "Pakistani", "Palauan", "Palestinian",
    "Panamanian", "Papua New Guinean", "Paraguayan", "Peruvian", "Philippine", "Polish", "Portuguese",
    "Qatari", "Romanian", "Russian", "Rwandan", "Saint Lucian", "Salvadoran", "Samoan", "San Marinese",
    "Saudi", "Scottish", "Senegalese", "Serbian", "Seychellois", "Sierra Leonean", "Singaporean", "Slovak",
    "Slovenian", "Somali", "South African", "Spanish", "Sri Lankan", "Sudanese", "Surinamese", "Swazi",
    "Swedish", "Swiss", "Syrian", "Taiwanese", "Tajik", "Tanzanian", "Thai", "Togolese", "Tongan",
    "Trinidadian", "Tunisian", "Turkish", "Turkmen", "Tuvaluan", "Ugandan", "Ukrainian", "Uruguayan",
    "Uzbek", "Vanuatuan", "Vatican", "Venezuelan", "Vietnamese", "Welsh", "Yemeni", "Zambian", "Zimbabwean"
]



# Load spaCy model
nlp = spacy.load("en_core_web_lg")

# function to convert subjective sentence into objective by removing some words
def convert_to_objective(sentence):
    # Process the sentence using spaCy
    doc = nlp(sentence)

    # Define subjective words (adjectives, adverbs, emotional/subjective phrases)
    subjective_tags = ["ADJ", "ADV"]

    # List to store the words for the objective sentence
    objective_sentence = []

    for token in doc:
        # If the word is an adjective or adverb, ignore it for objectivity
        # ignore the word if it is in subjective_tags and next word is a punctuation

        #if next token is fullstop, then don't remove that word
        if token.i < len(doc) - 1 and doc[token.i + 1].text == ".":
            objective_sentence.append(token.text)
            continue

        # don't remove the adverb if the previous token is ADJ
        if token.pos_ ==  "ADJ":
            if token.i > 0 and doc[token.i - 1].pos_ == "ADV":
                objective_sentence.append(token.text)
                continue
        if token.pos == "PART":
            if token.i > 0 and doc[token.i - 1].pos_ == "AUX":
                objective_sentence.append(token.text)
                continue

        # don't remove adverbs of time
        if token.text.lower() in adverbs_of_time:
            objective_sentence.append(token.text)
            continue

        # don't remove number_adjectives and position_adjectives
        if token.text.lower() in number_adjectives or token.text.lower() in position_adjectives:
            objective_sentence.append(token.text)
            continue
        
        # don't remove country related adjectives or descriptive adjectives form the above list
        if token.text.lower() in country_adjectives or token.text.lower() in descriptive_adjectives:
            objective_sentence.append(token.text)
            continue

        if token.text.lower() in adverbs_with_difficulty:
            objective_sentence.append("not much")
            continue

        #replace vaguely with somewhat
        if token.text.lower() in  ["roughly", "vaguely", "mildly"]:
            objective_sentence.append("somewhat")
            continue


        #below 3 if conditions are to remove joined adjectives like fast-paced, small-scale
        if token.i < len(doc) - 2:
            if doc[token.i + 1].tag_ == "HYPH" and doc[token.i + 2].pos_ in subjective_tags:
                continue;
        if token.i > 1:
            if doc[token.i - 1].tag_ == "HYPH" and doc[token.i - 2].pos_ in subjective_tags:
                continue;
        if token.tag_ == "HYPH":
            if (token.i < len(doc) - 1 and doc[token.i + 1].pos_ in subjective_tags) or (token.i > 0 and doc[token.i - 1].pos_ in subjective_tags):
                continue

        if token.pos_ in subjective_tags:
            if (token.i < len(doc) - 1 and doc[token.i + 1].tag_ == "HYPH") or (token.i > 0 and doc[token.i-1].tag_== "HYPH"):
                continue

        #generally multiple adjectives are joined by punctuations, so any p=such punctuations then remove them
        if token.pos_ == "PUNCT":
            if (token.i > 0 and doc[token.i - 1].pos_ in subjective_tags) and (token.i < len(doc) - 1 and doc[token.i + 1].pos_ in subjective_tags):
                continue

        #sentence should not start with punctuation

        if token.pos_ in subjective_tags:
            continue
        else:
            # Otherwise, keep the word
            objective_sentence.append(token.text)

    # Join the words into a single sentence
    return " ".join(objective_sentence).strip(string.punctuation + " ")



#read the excel file
data = pd.read_excel("data_subj_obj.xlsx")



#from textblob find subjectivity level of each sentence
def subj_level(sentence):
  blob = TextBlob(sentence)
  return blob.sentiment.subjectivity

#total sentences
print("Total Sentences : ", end = ' ')
print(data["SUBJECTIVE SENTENCES"].count())

#find subj_level of each sentence in data
data["INITIAL_SUBJECTIVITY_LEVEL"] = data["SUBJECTIVE SENTENCES"].apply(subj_level)

#find total counts of sentences having subjectivity level more than 0.5
print("Sentences with subjectivity > 0.5 before conversion :", end = ' ')
print(data[data["INITIAL_SUBJECTIVITY_LEVEL"] > 0.5].shape[0])



#write all objective sentences into a new excel file
data["OBJECTIVE SENTENCES"] = data["SUBJECTIVE SENTENCES"].apply(convert_to_objective)

#find subjectivity level of this objective sentences
data["FINAL_SUBJ_LEVEL"] = data["OBJECTIVE SENTENCES"].apply(subj_level)

#find total counts of objective sentences having subjectivity level more than 0.5
print("Sentences with subjectivity > 0.5 after conversion : ", end = ' ')
print(data[data["FINAL_SUBJ_LEVEL"] > 0.5].shape[0])

#print the description of the data
print(data.describe())

#write this updated data into a new excel file called parallel_sent
data.to_excel("parallel_sent_new_nettest.xlsx")
