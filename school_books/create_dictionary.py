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
            word_freq[word] += 1

    return dict(word_freq)

file_path = 'Pädagogik_Psychologie_Band_1.txt'
frequenzen = word_frequencies_from_file(file_path)

for wort, freq in sorted(frequenzen.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{wort}: {freq}")

