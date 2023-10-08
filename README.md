# Speedbolt
This is a paraphrasing model, that is capable of generating humanized texts.

 
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_text(prompt, model, tokenizer, max_length=100, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    # Load pre-trained GPT-2 model and tokenizer
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Set the seed for reproducibility
    torch.manual_seed(42)

    # Prompt for text generation
    prompt = "In a world where robots"

    # Generate unique humanized text
    generated_text = generate_text(prompt, model, tokenizer)

    # Print the generated text
    print("Prompt: ", prompt)
    print("Generated Text: ", generated_text)


// ==UserScript==
// @name         Quillbot Premium Unlocker
// @namespace    quillbot.taozhiyu.gitee.io
// @version      0.1
// @description  Unlocks Quillbot Premium so that you don't have to pay.
// @author       longkidkoolstar
// @match        https://quillbot.com/*
// @icon         https://quillbot.com/favicon.png
// @require      https://greasyfork.org/scripts/455943-ajaxhooker/code/ajaxHooker.js?version=1124435
// @run-at       document-start
// @grant        none
// @license      WTFPL
// ==/UserScript==
/* global ajaxHooker*/
(function() {
    'use strict';
    // How's it going filthy code looker
    ajaxHooker.hook(request => {
        if (request.url.endsWith('get-account-details')) {
            request.response = res => {
                const json=JSON.parse(res.responseText);
                const a="data" in json?json.data:json;
                a.profile.accepted_premium_modes_tnc=true;
                a.profile.premium=true;
                res.responseText=JSON.stringify("data" in json?(json.data=a,json):a);
            };
        }
    });
})();


import openai

openai.api_key = 'your-api-key'  # Replace with your OpenAI GPT-3 API key

def paraphrase_with_gpt3(prompt, temperature=0.7, max_tokens=100):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].text.strip()

if __name__ == "__main__":
    # Prompt for text paraphrasing
    prompt = "In a world where robots"

    # Generate paraphrased text using GPT-3
    paraphrased_text = paraphrase_with_gpt3(prompt)

    # Print the paraphrased text
    print("Prompt: ", prompt)
    print("Paraphrased Text: ", paraphrased_text)


from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import random

def synonym_replacement(sentence, n=2):
    words = word_tokenize(sentence)
    for _ in range(n):
        word = random.choice(words)
        synonyms = get_synonyms(word)
        if synonyms:
            synonym = random.choice(synonyms)
            sentence = sentence.replace(word, synonym, 1)
    return sentence

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return set(synonyms)

if __name__ == "__main__":
    original_sentence = "This is a sample sentence."
    modified_sentence = synonym_replacement(original_sentence)
    
    print("Original Sentence: ", original_sentence)
    print("Modified Sentence: ", modified_sentence)