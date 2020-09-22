import torch
from transformers import MarianMTModel, MarianTokenizer
import string
import pandas as pd
import spacy
from spacy import displacy
import difflib
from difflib import Differ, SequenceMatcher
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

en_ROMANCE_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
en_ROMANCE_tokenizer = MarianTokenizer.from_pretrained(en_ROMANCE_model_name)
en_ROMANCE = MarianMTModel.from_pretrained(en_ROMANCE_model_name).to(device)

ROMANCE_en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
ROMANCE_en_tokenizer = MarianTokenizer.from_pretrained(ROMANCE_en_model_name)
ROMANCE_en = MarianMTModel.from_pretrained(ROMANCE_en_model_name).to(device)

class CustomMTModel(MarianMTModel):
    def postprocess_next_token_scores(self, scores, input_ids, *a, **kw):
        if not ROMANCE_en.original_postprocess:
            batch_size, vocab_size = scores.shape
            cur_len = input_ids.shape[1]
            for hypothesis_idx in range(batch_size):
                cur_hypothesis = input_ids[hypothesis_idx]

            if 0 < cur_len <= len(ROMANCE_en.selected_tokens):
                force_token_id = ROMANCE_en.selected_tokens[cur_len-1]
                self._force_token_ids_generation(scores, force_token_id)

        return MarianMTModel.postprocess_next_token_scores(self, scores, input_ids, *a, **kw)

ROMANCE_en.__class__ = CustomMTModel

# parameters: machine_translation, the spanish translation
#             start, the forced beginning of the english. 
#             prefix_only, if true no new tokens will be generated after param 'start'
# returns:the final text (will be the same as 'start' if prefix_only)
#         the expected result (machine translation to english of the spanish input)
#         list of tokens in the final sequence
#         list of top 10 predictions for each token
#         score for average predictability
#######################################################################################
def incremental_generation(machine_translation, start, prefix_only):
    tokenizer = ROMANCE_en_tokenizer
    model = ROMANCE_en
    tokenized_prefix = tokenizer.convert_tokens_to_ids(en_ROMANCE_tokenizer.tokenize(start.strip()))
    prefix = torch.LongTensor(tokenized_prefix).to(device)

    batch = tokenizer.prepare_seq2seq_batch([machine_translation.replace("<pad> ", '')]).to(device)
    original_encoded = model.get_encoder()(**batch)
    decoder_start_token = model.config.decoder_start_token_id
    partial_decode = torch.LongTensor([decoder_start_token]).to(device).unsqueeze(0)
    past = None

    #machine translation for comparative purposes
    translation_tokens = model.generate(**batch)
    auto_translation = tokenizer.decode(translation_tokens[0]).split("<pad>")[1]

    num_tokens_generated = 0
    prediction_list = []
    MAX_LENGTH = 100
    total = 0

    #generate tokens incrementally 
    while True:
        model_inputs = model.prepare_inputs_for_generation(
            partial_decode, past=past, encoder_outputs=original_encoded, attention_mask=batch['attention_mask'], use_cache=model.config.use_cache
        )
        with torch.no_grad():
            model_outputs = model(**model_inputs)

        next_token_logits = model_outputs[0][:, -1, :]
        past = model_outputs[1]
        
        #start with designated beginning
        if num_tokens_generated < len(prefix):
            next_token_to_add = prefix[num_tokens_generated]
        elif prefix_only == True:
            break
        else:
            next_token_to_add = next_token_logits[0].argmax()

        #calculate score
        next_token_logprobs = next_token_logits - next_token_logits.logsumexp(1, True)
        token_score = next_token_logprobs[0][next_token_to_add].item()
        total += token_score

        #append top 10 predictions for each token to list
        decoded_predictions = []
        for tok in next_token_logits[0].topk(10).indices:
            decoded_predictions.append(tokenizer.convert_ids_to_tokens(tok.item()).replace('\u2581', '\u00a0'))
        
        #list of lists of predictions
        prediction_list.append(decoded_predictions)

        #add new token to tokens so far
        partial_decode = torch.cat((partial_decode, next_token_to_add.unsqueeze(0).unsqueeze(0)), -1)
        num_tokens_generated += 1

        #stop generating at </s>, or when max num tokens exceded
        if next_token_to_add.item() == 0 or not (num_tokens_generated < MAX_LENGTH):
            break

    #list of tokens used to display sentence
    decoded_tokens = [sub.replace('\u2581', '\u00a0') for sub in tokenizer.convert_ids_to_tokens(partial_decode[0])]
    decoded_tokens.remove("<pad>")

    final = tokenizer.decode(partial_decode[0]).replace("<pad>", '')
    score = round(total/(len(decoded_tokens)), 3)
    print(final)
    
    return {"final": final.lstrip(),
            "expected" : auto_translation,
            "tokens" : decoded_tokens,
            "predictions" : prediction_list,
            "score" : score
            }

def translate(tokenizer, model, text, num_outputs):   
    """Use beam search to get a reasonable translation of 'text'"""
    # Tokenize the source text
    tokenizer.current_spm = tokenizer.spm_source ### HACK!
    batch = tokenizer.prepare_seq2seq_batch([text]).to(model.device)
    
    # Run model
    num_beams = num_outputs
    translated = model.generate(**batch, num_beams=num_beams, num_return_sequences=num_outputs, max_length=40, no_repeat_ngram_size=5)
    
    # Untokenize the output text.
    tokenizer.current_spm = tokenizer.spm_target
    return [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in translated]

#get prepositional phrases
#adapted from https://stackoverflow.com/questions/39100652/python-chunking-others-than-noun-phrases-e-g-prepositional-using-spacy-etc
def get_pps(doc):
    pps = []
    for token in doc:
        if token.pos_ == 'ADP':
            pp = ' '.join([tok.orth_ for tok in token.subtree])
            pps.append(pp)
        if token.dep_ == 'prep':
            ROMANCE_en.off_limits.append(' '.join([tok.orth_ for tok in token.subtree]))
    return pps

def get_adv_clause(doc):
    clauses = []
    for token in doc:
        if token.dep_ == 'advcl' or token.dep_ == 'npadvmod' or token.dep_ == 'advmod':
            clause = ' '.join([tok.orth_ for tok in token.subtree])
            clauses.append(clause)
    return clauses

def generate_alternatives(english):
    nlp = spacy.load("en_core_web_sm")
    sentence =  english
    doc = nlp(sentence)
    phrases = []

    #get prepositional phrases and blacklist OPs 
    ROMANCE_en.off_limits = []
    for pphrase in get_pps(doc):
        #messy way to capitalize the first word without lowercasing the others
        capitalized = pphrase.split(' ')[0].capitalize() + ' ' + ' '.join(pphrase.split(' ')[1:])
        phrases.append(capitalized)

    #get noun chunks that aren't OPs
    for chunk in doc.noun_chunks:
        valid = True
        for phr in ROMANCE_en.off_limits:
            if chunk.text in phr:
                valid = False
        if valid:
            capitalized = chunk.text.split(' ')[0].capitalize() + ' ' + ' '.join(chunk.text.split(' ')[1:])
            phrases.append(capitalized)

    #get adverbial modifiers and clauses
    for clause in get_adv_clause(doc):
        capitalized = clause.split(' ')[0].capitalize() + ' ' + ' '.join(clause.split(' ')[1:])
        phrases.append(capitalized)
        
    #get clause beginnings
    wordlist = [t.orth_ for t in doc]
    for token in doc:
        if token.dep_ == 'nsubj':
            mystr = ' '.join([t.orth_ for t in token.lefts]) + token.text + ' ' + token.head.orth_

            phraselist = wordlist[wordlist.index(token.orth_):wordlist.index(token.head.orth_)+1]
            phrases.append(' '.join(phraselist).capitalize())

            wordlist.remove(token.orth_)
        
    print(phrases)

    #prepare input for translation
    ROMANCE_en.original_postprocess = True;
    english = ">>es<<" + sentence
    engbatch = en_ROMANCE_tokenizer.prepare_seq2seq_batch([english]).to(device)
    eng_to_spanish = en_ROMANCE.generate(**engbatch).to(device)
    machine_translation = en_ROMANCE_tokenizer.decode(eng_to_spanish[0]).replace("<pad> ", '')

    #generate alternatives starting with each selected phrase
    results = []
    for selection in set(phrases):
        resultset = []
        ROMANCE_en_tokenizer.current_spm = ROMANCE_en_tokenizer.spm_target
        tokens = ROMANCE_en_tokenizer.tokenize(selection)
        ROMANCE_en.selected_tokens = ROMANCE_en_tokenizer.convert_tokens_to_ids(tokens)

        ROMANCE_en.original_postprocess = False
        top50 = translate(ROMANCE_en_tokenizer, ROMANCE_en, ">>en<<" + machine_translation, 50)
        for element in top50[0:3]:
            res = incremental_generation(machine_translation, element, prefix_only=False)
            resultset.append((res['score'], res['final']))
        results.append(resultset)

    #count content words in original and each alternative to catch options that repeat or leave off important phrases
    important_words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
    wordcount = []
    for word in important_words:
        wordcount.append((word, sentence.count(word))) 
    for resultset in results:
        idx = 0
        for score, sen in resultset:
            resdoc = nlp(sen)
            important = [token.text for token in resdoc if token.is_stop != True and token.is_punct != True]
            if len(important) - len(important_words) not in [-1, 0, 1]:
                resultset[idx] = (score-10, sen)
            else:
                for el in wordcount:
                    if sen.count(el[0]) > el[1]:
                        resultset[idx] = (score-10, sen)
            idx += 1
        
    #sort results with highest score first                
    all_sorted = sorted(results, key=lambda x: x[0])[::-1]

    #select prepositional and noun phrases to be highlighted 
    top = nlp(all_sorted[0][0][1])
    highlight = []
    for pphrase in get_pps(top):
        highlight.append(pphrase)
    for chunk in doc.noun_chunks:
        if chunk.text not in ' '.join(highlight):
            highlight.append(chunk.text)
    color_code_chunks = []
    for optionset in all_sorted:
        color_code_subset = []
        for score, text in optionset:
            if score > -10:
                ph_and_idx = []
                for ph in highlight:
                    starting_idx = text.lower().find(ph.lower())
                    ph_and_idx.append((ph, starting_idx))

                order = sorted((score, text) for text, score in ph_and_idx)
                ordered_phrases = [phrase for score, phrase in order]

                new_sentence = text
                final_sentence = []
                x = 0
                for phrase in ordered_phrases:
                    if phrase.lower() in text.lower():
                        starting_idx = text.lower().find(phrase.lower())
                        final_sentence.append((new_sentence.lower().split(phrase.lower())[0], 0))
                        new_sentence = new_sentence.lower().split(phrase.lower())[-1]
                        final_sentence.append((phrase, highlight.index(phrase) + 1))
                    x += 1
                final_sentence.append((new_sentence, 0))
                color_code_subset.append(final_sentence)
        color_code_chunks.append(color_code_subset)
    
    #messy way to capitalize sentences
    for group in color_code_chunks:
        for chunk in group:
            if chunk[0][0] == '':
                first = chunk[1][0]
                capitalized = first.split(' ')[0].capitalize() + ' ' + ' '.join(first.split(' ')[1:])
                chunk[1] = (capitalized, chunk[1][1])
            else:
                first = chunk[0][0]
                capitalized = first.split(' ')[0].capitalize() + ' ' + ' '.join(first.split(' ')[1:])
                chunk[0] = (capitalized, chunk[0][1])

    alternatives = []
    scores = []
    for subset in all_sorted:
        altgroup = []
        for score, result in subset:
            altgroup.append(result)
        alternatives.append(altgroup)

    return {"alternatives" : alternatives,
            "colorCoding" : color_code_chunks
        }

def incremental_alternatives(sentence, prefix, recalculation):
    ROMANCE_en.original_postprocess = True
    english = ">>es<<" + sentence
    engbatch = en_ROMANCE_tokenizer.prepare_seq2seq_batch([english]).to(device)
    eng_to_spanish = en_ROMANCE.generate(**engbatch).to(device)
    machine_translation = en_ROMANCE_tokenizer.decode(eng_to_spanish[0]).replace("<pad> ", '')
    if recalculation:
        sentence = prefix
    print(machine_translation)
    print(sentence)
    return(incremental_generation(machine_translation, sentence, False))

def completion(sentence, prefix):
    prefix = prefix.replace(" ", '', 1)
    ROMANCE_en.original_postprocess = True
    english = ">>es<<" + sentence
    engbatch = en_ROMANCE_tokenizer.prepare_seq2seq_batch([english]).to(device)
    eng_to_spanish = en_ROMANCE.generate(**engbatch).to(device)
    machine_translation = en_ROMANCE_tokenizer.decode(eng_to_spanish[0]).replace("<pad> ", '')

    ROMANCE_en_tokenizer.current_spm = ROMANCE_en_tokenizer.spm_target
    tokens = ROMANCE_en_tokenizer.tokenize(prefix)
    ROMANCE_en.selected_tokens = ROMANCE_en_tokenizer.convert_tokens_to_ids(tokens)

    ROMANCE_en.original_postprocess = False
    top5 = translate(ROMANCE_en_tokenizer, ROMANCE_en, ">>en<<" + machine_translation, 5)

    differences = []
    for option in top5:
        diffs = []
        a = sentence.split()
        print(a)
        b = option.split()
        print(b)
        s = SequenceMatcher(None, a, b)
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            if tag != 'equal':
                x = j1 - len(prefix.split()) + 1
                for string in b[j1:j2]:
                    print(string)
                    diffs.append(x)
                    x += 1
        differences.append(diffs)
    print("prefix length: ", len(prefix.split()))

    endings = []
    for s in top5:
        endings.append(s.replace(prefix, ''))

    return {'endings' : endings,
            'differences' : differences
            }