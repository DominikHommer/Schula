import os
from spellchecker import SpellChecker
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForMaskedLM,
    pipeline
)
from Levenshtein import ratio, distance
import re
import torch.nn.functional as F
from spylls.hunspell import Dictionary
from jarowinkler import jarowinkler_similarity
from symspellpy import SymSpell, Verbosity

from .module_base import Module

class TextCorrector(Module):
    def __init__(self, debug=False, debug_folder="debug/debug_textcorrector"):
        super().__init__("text-corrector")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.debug = debug
        self.debug_folder = debug_folder
        if self.debug:
            os.makedirs(self.debug_folder, exist_ok=True)

    def _warmup(self):
        self.hunspell = Dictionary.from_files('models/hunspell/de_DE_frami')
        self.symspell = SymSpell(max_dictionary_edit_distance=4)
        self.symspell.load_dictionary('models/symspell/de-100k_schulbuch.txt', 0, 1)

        self.possible_per_names = ["Tim", "Marcia"]

        self.checker = SpellChecker(language="de")
        # Here load dictionary of all words from Schulbuch
        self.checker.word_frequency.load_words(["Werthaltungen", "Identitätstypen", "Marcia", "Sprechblase", "Sprechblasen", "Selbstwert", "Arzt", "vgl", "vgl.", 
                                                "Identitätsmodell", "Fachoberschule", "Moratoriumsidentität", "Experimentierfreudigkeit", "Selbstkonzeptes", "Idealselbst", "Realselbst"
                                                "Kunstlehrer", "Urteilsbildungen", "Beratungskonzept", "Therapeuten"])
        
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert/distilbert-base-german-cased")
        self.model = DistilBertForMaskedLM.from_pretrained("distilbert/distilbert-base-german-cased")
        
        self.ner = pipeline(
            "ner",
            model="FacebookAI/xlm-roberta-large-finetuned-conll03-german",
            tokenizer="FacebookAI/xlm-roberta-large-finetuned-conll03-german",
            grouped_entities=True
        )

        self.fill_mask = pipeline(
            "fill-mask",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if 'cuda' in self.device else -1
        )
    
    def get_preconditions(self) -> list[str]:
        return ['text-recognizer']
    
    def score_candidates_batch(
        self,
        pre: list[str],
        post: list[str],
        candidates: list[str]
    ) -> dict[str, float]:
        mask_index_lengths: list[tuple] = []
        batch_ids = []
        for cand in candidates:
            # First tokenize all parts
            tokenized_pre: list = self.tokenizer.tokenize(" ".join(pre))
            tokenized_cand: list = self.tokenizer.tokenize(cand)
            tokenized_post: list = self.tokenizer.tokenize(" ".join(post))

            # Convert to corresponding ids
            base_ids = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokenized_pre + tokenized_cand + tokenized_post + ['[SEP]'])

            # Add ids and save "masks" start and their length
            batch_ids.append(base_ids)
            mask_index_lengths.append((cand, tokenized_pre.__len__() + 1, tokenized_cand.__len__()))

        max_len = max(len(ids) for ids in batch_ids)
        batch_tensor = torch.zeros((len(batch_ids), max_len), dtype=torch.long, device=self.device)
        for i, ids in enumerate(batch_ids):
            batch_tensor[i, :len(ids)] = torch.tensor(ids, device=self.device)
        
        with torch.no_grad():
            logits = self.model(batch_tensor).logits  # (batch, seq, vocab)
            log_probs = F.log_softmax(logits, dim=-1)

        scores = {}
        for i, cand in enumerate(candidates):
            ids = batch_tensor[i]

            total_prob = 0
            mask_start = mask_index_lengths[i][1]
            mask_len = mask_index_lengths[i][2]
            for idx in range(mask_start, mask_start + mask_len):
                token_id = ids[idx].item()

                total_prob += log_probs[i, idx, token_id].item()
                
            scores[cand] = total_prob / mask_len # We normalize here to have an even probability

        return scores

    def _map_ner_to_per(self, ner_result: dict) -> str:
        total_names = len(self.possible_per_names)
        for per in self.possible_per_names:
            if ratio(ner_result['word'], per) > 1 / total_names:
                return per
            
            if jarowinkler_similarity(ner_result['word'], per) > 1 / total_names:
                return per
        
        return ner_result['word']

    def process(self, data: dict) -> list:
        texts: list = data.get('text-recognizer', [])

        text = " new_line ".join(texts)
        # Fix linebreaks with wo-rd
        pattern = re.compile(r'(\w+)-\snew_line\s+(\w+)', re.UNICODE)
        while True:
            text, count = pattern.subn(r'\1\2 new_line ', text)
            if count == 0:
                break

        # Use NER to map possible names
        ner_result = {res['word']: self._map_ner_to_per(res) for res in self.ner(text) if res['score'] > 0.8 and res['entity_group'] == "PER"}

        # Add space after and before symbols. Else: BERT has problems
        words = (re.sub(r'([.,!?;:()\[\]{}"“”])', r' \1 ', text)).split()

        ## Algos:
        # - Wenn Wort in pyspellchecker == original Wort, nehmen wir das Wort
        # - Ansonsten: Lass BERT nächstes Wort predicten, lass pyspellchecker wort beheben
        #   - Danach: Lassen wir BERT die Wörter scoren
        #   - Danach: Gewichten wir den Score mit der Levenstein-Distant zum original
        #   - Danach: Bester Score gewinnt
        # - Beginne wieder von vorne für nächstes Wort

        corrected_words = list()
        for i in range(len(words)):
            original_word = words[i]

            # Add line breaks to their original position
            if original_word == "new_line":
                corrected_words.append("\n")

                continue

            # Symbols are symobls, no need for predictions
            if re.fullmatch(r'(?:[.,!?;:()\[\]{}"“”]|\d+)', original_word):
                corrected_words.append(original_word)

                continue

            # Mostly just noise
            if len(original_word) <= 1:
                corrected_words.append(original_word)
                continue

            found_per = False
            for possible_per, mapped_value in ner_result.items():
                # TODO: This has to be done better
                if not found_per and possible_per in original_word:
                    found_per = True
                    corrected_words.append(mapped_value)

            if found_per:
                continue

            corrected_candidates = self.checker.candidates(original_word)
            if corrected_candidates and len(corrected_candidates) == 1:
                corrected = list(corrected_candidates)[0]
            
                if corrected == original_word and len(corrected) >= 4:
                    corrected_words.append(corrected)
                    
                    continue

            try:
                pre_context = corrected_words[max(0, i - 40) : i]
                post_context = words[i + 1 : min(len(words) - 1, i + 40)]
                post_context = list(filter(lambda x: x != 'new_line', post_context))

                context = pre_context + ['[MASK]'] + post_context
                masked_text = ' '.join(context)

                canditates = []
                #score_corrected = float('-inf')
                if corrected_candidates is not None:
                    amount = 0
                    for canditate in list(corrected_candidates):
                        amount += 1
                        if amount <= 3:
                            canditates.append(canditate)
                    #corrected = list(corrected_candidates)[0]
                    #score_corrected = self._score_sentence_bert_batch(" ".join(pre_context + [corrected] + post_context))
                    #canditates.append(corrected)
                else:
                    pass
                    #corrected = original_word
                    #score_corrected = self._score_sentence_bert_batch(" ".join(pre_context + [original_word] + post_context))

                amount = 0
                for suggest in self.hunspell.suggest(original_word):
                    if suggest not in canditates and suggest is not original_word:
                        amount += 1
                        if amount <= 3:
                            canditates.append(suggest)

                amount = 0
                for suggest in self.symspell.lookup(original_word, Verbosity.CLOSEST, max_edit_distance=4, transfer_casing=True):
                    if suggest.term not in canditates and suggest.term is not original_word:
                        amount += 1
                        if amount <= 3:
                            canditates.append(suggest.term)
                
                top = self.fill_mask(masked_text, top_k=3)
                for t in top:
                    w = t['token_str'].strip()
                    if w and w not in canditates:
                        canditates.append(w)

                    #new_score = self._score_sentence_bert_batch(" ".join(pre_context + [suggest] + post_context))

                    #if new_score > score_corrected:
                    #    corrected = suggest
                    #    score_corrected = new_score


                #score_predicted = self._score_sentence_bert_batch(" ".join(pre_context + [predicted_token] + post_context))

                #print(masked_text)
                result = self.symspell.word_segmentation(original_word)
                if result.distance_sum == 1:
                    result_string = result.corrected_string

                    if len(result_string.split(" ")) >= 2:
                        canditates.append(result.corrected_string)

                bert_scores = self.score_candidates_batch(pre_context, post_context, canditates)
                # We filter candidates with a too bad score
                bert_scores = {k: v for k, v in bert_scores.items() if v > -0.1}

                lm_weight = 0.5
                lev_weight = 1.5

                scores = bert_scores.values()
                min_s, max_s = min(scores), max(scores)
                eps = 1e-6
                norm_scores = {w: (s - min_s) * 2 / (max_s - min_s + eps) for w, s in bert_scores.items()}
                #print(bert_scores)
                #print(norm_scores)

                final_scores = {}
                for w, s in norm_scores.items():
                    length = len(w)
                    org_length = len(original_word)

                    # Compute similarity
                    jaro = jarowinkler_similarity(w.lower(), original_word.lower())
                    lev = ratio(original_word.lower(), w.lower())

                    if jaro == 0 or lev == 0:
                        continue

                    proportion = length  / org_length
                    if length > org_length:
                        proportion = (org_length / length)

                    # Short words
                    if org_length <= 5:
                        proportion = 1

                    similarity = (jaro + lev) * proportion * lev_weight

                    score = (lm_weight * s) + similarity

                    final_scores[w] = score
                    #print(lev, w, s * lm_weight, similarity, f'Final: {final_scores[w]}')
                
                best = max(final_scores, key=lambda w: final_scores[w])
                if distance(original_word, best) > 4:
                    corrected_words.append(original_word)

                    continue

                #print(best)

                #lev_score_corrected = score_corrected * (1 - ratio(corrected, original_word))
                #lev_score_predicted = score_predicted * (1 - ratio(predicted_token, original_word))

                #_, best_word = max([
                #    (score_corrected + lev_score_corrected, corrected),
                #    (score_predicted + lev_score_predicted, predicted_token),
                #], key=lambda x: x[0])

                corrected_words.append(best)
            except:
                print("[TextCorrector] an Error occured during text reparation")
    
        if self.debug:
            debug_path = os.path.join(self.debug_folder, "debug_textcorrector.txt")
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(" ".join(corrected_words))
            print(f"[TextCorrector] Debug-Log gespeichert in: {debug_path}")

        return corrected_words 