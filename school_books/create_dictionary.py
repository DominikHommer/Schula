from collections import defaultdict
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9äöüßáéíóúàèìòùâêîôûçñ]+', ' ', text)
    #with open("test.txt", "w", encoding="utf-8") as file:
    #    file.write(text)    
    return text

def word_frequencies_from_file(file_path):
    word_freq = defaultdict(int)

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        cleaned_text = clean_text(text)
        words = cleaned_text.split()

        for word in words:
            if word.isdigit():
                continue

            word_freq[word] += 1

    return dict(word_freq)

file_path = 'Pädagogik_Psychologie_Band_3.txt'
frequenzen = word_frequencies_from_file(file_path)

print(frequenzen)
with open('../models/symspell/de-100k.txt', encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            word = clean_text(parts[0])
            if word.isdigit():
                continue
            
            freq = int(parts[1])

            if frequenzen.get(word, None) is None:
                frequenzen[word] = freq
            else:
                frequenzen[word] = max(frequenzen[word], 0) + freq

with open('../models/symspell/de-100k_schulbuch.txt', "w", encoding="utf-8") as out:
    for word in sorted(frequenzen):
        out.write(f"{word} {frequenzen[word]}\n")

