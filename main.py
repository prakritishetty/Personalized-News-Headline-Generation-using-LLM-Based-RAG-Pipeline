import pandas as pd
import random
import json
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import faiss
import google.generativeai as genai
import torch
from tqdm import tqdm
import random
import requests
import os
from numpy import mean
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import ast
from rouge_score import rouge_scorer


def load_lamp4_dataset(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        return None

top_k_articles = []

lamp4_data = []
lamp4_subset=[]


import random
from tqdm import tqdm

url = "https://ciir.cs.umass.edu/downloads/LaMP/LaMP_4/train/train_questions.json"

lamp4_data = load_lamp4_dataset(url)

if lamp4_data is None:
    print("Failed to load dataset. Exiting.")
else:
    subset_size = 300
    lamp4_subset = random.sample(lamp4_data, subset_size)

file_path = 'https://drive.google.com/file/d/1-6JejVHCH_cSQEnSHLItk9Be2L51puYO/view?usp=drive_link'

with open(file_path, "w", encoding="utf-8") as outfile:
    json.dump(lamp4_subset, outfile, indent=4)

with open(file_path, "r", encoding="utf-8") as file:
        lamp4_subset = json.load(file)

lamp4_subset = random.sample(lamp4_subset, 100)

genai.configure(api_key="AIzaSyDb361_mnQ_6qrckEv_eFgu1mB5dO9II0E")


def generate(given_line, num_variants=3):

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = []
    for _ in range(num_variants):
        response_t = model.generate_content(given_line)
        response.append(response_t.text)
    return response

def encode(model, tokenizer, text):

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.pooler_output[0].numpy()

def index_in_batches(df, encoder, tokenizer, batch_size=100):

    index = None
    for start in range(0, len(df), batch_size):
        batch = df[start: start + batch_size]
        embeddings = [encode(encoder, tokenizer, text) for text in batch['text']]
        embeddings = np.array(embeddings).astype("float32")

        if index is None:
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index = faiss.IndexIDMap(index)


        index.add_with_ids(embeddings, batch['id'].values)

    return index

def initialize_encoder():
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever")
    return model, tokenizer

encoder_model, encoder_tokenizer = initialize_encoder()

def process_data_in_batches(
    lamp4_subset, encoder, tokenizer, batch_size=100, output_file="https://drive.google.com/file/d/1D9HVAQyHyp3U0Ha-jMyAyumTVzeg7JrF/view?usp=drive_link"
):

    top_k_articles = []  

    for idx, item in enumerate(tqdm(lamp4_subset, desc="Processing items")):
        try:
            original_query = item['input']  
            given_line = original_query[:47]  
            new_query = f"Paraphrase the article: {given_line}"

            response = generate(new_query)

            user_profile = item['profile']

            if isinstance(user_profile, str):
                user_profile = json.loads(user_profile.replace("'", "\""))  
            elif not isinstance(user_profile, list):
                raise ValueError(f"Expected a list for user_profile, got {type(user_profile)}")

            if len(user_profile) == 0:
                raise ValueError("user_profile is empty.")

            df_profile = pd.DataFrame(user_profile)

            index = index_in_batches(df_profile, encoder, tokenizer, batch_size)

            dense_vector = np.mean(
                [encode(encoder, tokenizer, t) for t in [given_line] + response], axis=0
            )
            dense_vector = dense_vector.reshape(1, -1).astype("float32")

            length_profile = len(user_profile)
            k = min(3, max(1, length_profile // 2))  
            distances, indices = index.search(dense_vector, k)

            top_k_indices = [str(idx) for idx in indices[0]]

            matched_articles = [
                {"text": profile['text'], "title": profile['title'], "id": profile['id']}
                for profile in user_profile if str(profile['id']) in top_k_indices
            ]

            top_k_articles.append({
                "id": item['id'], 
                "input": item['input'],  
                "top_k_articles": matched_articles  
            })

            if (idx + 1) % 50 == 0 or (idx + 1) == len(lamp4_subset):
                temp_output_file = output_file.replace(".json", f"_{idx + 1}.json")
                with open(temp_output_file, "w", encoding="utf-8") as outfile:
                    json.dump(top_k_articles, outfile, indent=4)
                print(f"Saved progress to {temp_output_file}")

        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            continue  

    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(top_k_articles, outfile, indent=4)

    print(f"Final results saved to {output_file}")

process_data_in_batches(lamp4_subset, encoder_model, encoder_tokenizer, batch_size=50)


file_path = 'https://drive.google.com/file/d/1-6JejVHCH_cSQEnSHLItk9Be2L51puYO/view?usp=drive_link'
def save_dataset(data):
  sampled_data = random.sample(data, 100)
  data_df = pd.DataFrame(sampled_data)
  data_df.to_csv(file_path)

def fetch_and_print_json(url):
    try:

        response = requests.get(url)
        response.raise_for_status()

        data = response.json()



        if isinstance(data, dict):
            print("JSON is a dictionary. Here are the keys:")
            print(data.keys())
            print("\nSample content:")
            print(data)
            save_dataset(data)

        elif isinstance(data, list):
            print("JSON is a list. Length of list:", len(data))
            print("\nFirst item in the list:")
            print(data[0])
            save_dataset(data)

        else:
            print("Unexpected JSON format. Printing the data:")
            print(data)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching JSON from URL: {e}")
    except Exception as e:
        print(f"Error processing JSON: {e}")


reference_url = "https://ciir.cs.umass.edu/downloads/LaMP/LaMP_4/train/train_questions.json"

fetch_and_print_json(reference_url)

model = SentenceTransformer('AnnaWegmann/Style-Embedding')
print("Model Loaded succesfully!")

def compute_average_embedding(profile, model):
    profile_texts = [article['text'] for article in profile]

    embeddings = model.encode(profile_texts, convert_to_tensor=True)

    avg_embedding = embeddings.mean(axis=0)

    return avg_embedding


def find_top_k_articles(profile, avg_embedding, model, k=5):
    
    profile_texts = [article['text'] for article in profile]
    profile_embeddings = model.encode(profile_texts, convert_to_tensor=True)


    similarities = util.cos_sim(profile_embeddings, avg_embedding).squeeze(1)
    top_k_indices = similarities.argsort(descending=True)[:k]
    top_k_articles = [profile[i] for i in top_k_indices]
    return top_k_articles

def safe_json_loads(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        print(f"Invalid JSON: {value}")
        return None


with open(file_path, 'r') as f:
    data = json.load(f)

def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        print(f"Invalid format in profile: {value}")
        return value  

for entry in data:
    if isinstance(entry.get('profile'), str):
        entry['profile'] = safe_literal_eval(entry['profile'])

loaded_df = pd.DataFrame(data)


data = loaded_df.to_dict(orient="records")

results = []
k = 5

for item in tqdm(data, desc="Finding top-k articles from user profiles:"):
    profile = item['profile']
    avg_embedding = compute_average_embedding(profile, model)
    top_k_articles = find_top_k_articles(profile, avg_embedding, model, k)
    results.append({
        "id": item['id'],
        "input": item['input'],
        "top_k_articles": top_k_articles
    })

print(results[0])

top_k_path = 'https://drive.google.com/file/d/1Zg7ML3wArE5xtSfUro4zMlWRm-lW10YM/view?usp=drive_link'

with open(top_k_path, "w") as f:
    json.dump(results, f, indent=4)

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def load_reference_outputs(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['golds']
    except requests.exceptions.RequestException as e:
        print(f"Error downloading reference outputs: {e}")
        return None

reference_url = "https://ciir.cs.umass.edu/downloads/LaMP/LaMP_4/train/train_outputs.json"
reference_outputs = load_reference_outputs(reference_url)

def generate_headline(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=64, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_output_using_LLM(top_k_data, top_k_data_bm25):
    generated_headlines = []
    counter = 0

    for item in tqdm(top_k_data, desc="Generating headlines using LLM"):
        input_article = item["input"]
        top_k_articles = item["top_k_articles"]

        
        item_id = str(item["id"])  

        top_k_articles_bm25_item = next(
            (bm25_item for bm25_item in top_k_data_bm25 if bm25_item["id"] == item_id),
            None
        )


        if not top_k_articles_bm25_item:
            print(f"No matching BM25 data for id: {item_id}")
            continue
        counter+=1
        top_k_articles_bm25 = top_k_articles_bm25_item["top_k_articles"]

        context_articles = top_k_articles[:1] + top_k_articles_bm25[:1]
        context = "\n".join(
            [f"Title: {a['title']}\nText: {a['text']}" for a in context_articles]
        )

        input_text = f"{input_article}\nGiven past user profile context:\n{context}"
        print(input_text)
        headline = generate_headline(input_text)

        generated_headlines.append({
            "id": item["id"],
            "output": headline
        })
    print(f"Genearted headlines for {counter} articles ! ")
    return generated_headlines

def evaluate_rouge(generated_headlines, reference_outputs):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": []
    }

    for gen_headline in generated_headlines:
        ref_output = next(
            (ref for ref in reference_outputs if ref["id"] == str(gen_headline["id"])),
            None
        )

        if ref_output:
            score = scorer.score(gen_headline["output"], ref_output["output"])
            scores["rouge1"].append(score["rouge1"].fmeasure)
            scores["rouge2"].append(score["rouge2"].fmeasure)
            scores["rougeL"].append(score["rougeL"].fmeasure)
        else:
            print(f"Warning: No reference found for ID {gen_headline['id']}")

    avg_scores = {metric: (sum(scores[metric]) / len(scores[metric]) if scores[metric] else 0)
                  for metric in scores}

    return avg_scores

top_k_path = "https://drive.google.com/file/d/1Zg7ML3wArE5xtSfUro4zMlWRm-lW10YM/view?usp=sharing"
with open(top_k_path, "r") as f:
    top_k_data = json.load(f)

top_k_path_bm25 = "https://drive.google.com/file/d/1D9HVAQyHyp3U0Ha-jMyAyumTVzeg7JrF/view?usp=drive_link"
with open(top_k_path_bm25, "r") as f:
    top_k_data_bm25 = json.load(f)

generated_headlines = generate_output_using_LLM(top_k_data, top_k_data_bm25)
rouge_scores = evaluate_rouge(generated_headlines, reference_outputs)
generated_headlines_write = f"/generated_headlines_dense_100.json"
with open(generated_headlines_write, "w") as f:
    json.dump(generated_headlines, f, indent=4)

generated_headlines_read = (f"/generated_headlines_hybrid.json")
with open(generated_headlines_read, "r") as f:
    generated_headlines = json.load(f)
rouge_scores = evaluate_rouge(generated_headlines, reference_outputs)
print("ROUGE Evaluation Results:", rouge_scores)

print(reference_outputs)
