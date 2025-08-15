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
    "recently", "most", "unique",
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


#from textblob find subjectivity level of each sentence
def subj_level(sentence):
  blob = TextBlob(sentence)
  return blob.sentiment.subjectivity

input_data = '''
Following Jannik Sinner’s 3-6, 3-6, 6-4, 6-4, 6-4 comeback win over Daniil Medvedev to win a maiden Grand Slam title at the Australian Open on Sunday, men’s tennis players born in the 2000s have now won more majors (three) than those born in the 1990s (two).
Sinner, the unassuming 22-year-old World No 4 from Italy, joins 20-year-old Spaniard Carlos Alcaraz — who won two majors over the past year — as members of the generation that has captured tennis fans’ imagination and leapfrogged the generation of players above it.
The sport is unquestionably going through a transitional moment. The fading of the Federer-Nadal duopoly — the Spaniard will still be planning one last big push for the French Open where he is a 13-time champion this year — has allowed Novak Djokovic to establish himself as the best player in the world. But now at 36, the Serb is facing physical and technical issues of his own, and while the generation below him, including Sunday’s finalist Medvedev, who became the first player in history to fail to win a major final from a two-set lead twice, has failed to take advantage of his lapses, Gen Z are facing no such issue.
Sinner defeated Djokovic in the semifinal in Melbourne this week, while Alcaraz defeated him in a memorable Wimbledon final last year. Both have well-rounded attacking games, and both showed the mental resilience to deal with the pressure of playing five sets in a Grand Slam final.
Djokovic remains the leader of this pack. He lost his first Wimbledon match since 2017 against Alcaraz in the final last year, and bounced back to end his year on a record high. As 10-time champion of Melbourne Park, unbeaten there since 2018 until Sinner took him down this year, expect him to return with a similar fury. Also expect the fiery Spaniard and the laid-back Italian to resist.
'''

print("Before : ", subj_level(input_data))
output_data = convert_to_objective(input_data)
print("Objective data : ", output_data)
print("After : ", subj_level(output_data))
