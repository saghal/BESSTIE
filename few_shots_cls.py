"""
This file have for objective to implement a RAG solution to improve the performance of the MISTRAL LLM in classification of sentiment and sarcasm.
To do so, instead of using zero-shot like the original paper, we are going to use a few-shot by using a RAG system in 2 different way :
- Closest sentence : we find the k closest sentences and add them as examples
- Contrastive sentence : we add k (even) closest sentences as examples, with k/2 sentences from each labels
"""
from mistralai import Mistral
from dotenv import load_dotenv
import os
import certifi
import pandas as pd
from tqdm import tqdm
from BESSTIE_RAG import BESSTIE_RAG
import numpy as np
import argparse
import sys
from sklearn.metrics import classification_report
os.environ['SSL_CERT_FILE'] = certifi.where()

# Fix the seed for reproducibility
np.random.seed(42)

# Load the API key
load_dotenv(dotenv_path=".env")

# Store the classification results
results = []

sentiment_prompt = """Generate the sentiment of the given text.
1 for positive sentiment, and 0 for negative sentiment. Do not give an explanation.
"""
sarcasm_prompt = """Predict if the given text is sarcastic.
1 if the text is sarcastic, and 0 if the text is not sarcastic. Do not give an explanation."""

def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Classification of Sentiment or Sarcasm sentences with Few-shots")
    
    parser.add_argument("--sarcasm", action="store_true", help="Switch the classification from Sentiment to Sarcasm")
    parser.add_argument("-K", type=int, default=4, help="Number of K examples for the few shots")
    parser.add_argument("--zero_shot", action="store_true", help="Use zero shot")
    parser.add_argument("--contrastive", action="store_true", help="Switch the retrieving methods from the closest to contrastive")
    parser.add_argument("--variety", action="store", type=str, default="en-UK", help="You can select the variety between [en-UK, en-IN, en-AU] (Default:en-UK)")
    parser.add_argument("--source", action="store", type=str, default="BOTH", help="You can select the source between [BOTH, Google, Reddit] (Default:BOTH)")

    return parser.parse_args(args)

if (__name__ == "__main__"):
    args = parse_args()

    # Create the client
    client = Mistral(api_key=os.getenv("API_KEY"))
    # Select the prompt according to the task
    prefix_prompt = sentiment_prompt if not(args.sarcasm) else sarcasm_prompt

    train_df = pd.read_csv("train.csv")
    valid_df = pd.read_csv("valid.csv")

    # Filter the train and valid dataset for Sentiment or Sarcasm entries only and for the specified variety
    train_df_filtered = train_df[(train_df["task"] == "Sentiment") & (train_df["variety"] == args.variety)] if not(args.sarcasm) else train_df[(train_df["task"] == "Sarcasm") & (train_df["variety"] == args.variety)]
    valid_df_filtered = valid_df[(valid_df["task"] == "Sentiment") & (valid_df["variety"] == args.variety)] if not(args.sarcasm) else valid_df[(valid_df["task"] == "Sarcasm") & (valid_df["variety"] == args.variety)]

    # Filter the source if needed
    if (args.source != "BOTH"):
        train_df_filtered = train_df_filtered[train_df_filtered["source"] == args.source]
        valid_df_filtered = valid_df_filtered[valid_df_filtered["source"] == args.source]

    # Instantiate the RAG system on the train datasets
    RAG_system = BESSTIE_RAG(train_df_filtered)

    for _, row in tqdm(valid_df_filtered.iterrows(), total=len(valid_df_filtered)):
        text = row["text"]
        label = row["label"]
        variety = row["variety"]
        source = row["source"]
        predicted_label = -1
        is_correct = False
        try:
            # Get the examples from the RAG system
            if (not(args.zero_shot)):
                examples = RAG_system.retrieve_closest(text, args.K) if not(args.contrastive) else RAG_system.retrieve_contrastive(text, args.K)

            prompt = f"{prefix_prompt}\n\n{examples}\nNow it is your turn.\nText: {text}\nLabel:" if not(args.zero_shot) else f"{prefix_prompt}\nText: {text}\nLabel:"
            
            # Create the prompt and generate the response
            response = client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {"role": "user",
                    "content": prompt}
                ],
                temperature=0
            )

            # Get the result from the response
            result = response.choices[0].message.content.strip()
            
            # Verification, since temperature is 0, return the most probable token (similar to the MLE of the BESSTIE Paper)
            if (result == "1"):
                predicted_label = 1
            elif (result == "0"):
                predicted_label = 0

            if (predicted_label != -1):
                results.append({
                    "text" : text,
                    "label" : label,
                    "prediction" : predicted_label,
                    "is_correct" : label == predicted_label,
                    "variety" : variety,
                    "source" : source

                })

        except Exception as e:
            print(f"Error : {e}")

    # Custom name crafting
    CLS = "Sentiment" if not(args.sarcasm) else "Sarcasm"
    FS_technique = "closest" if not(args.contrastive) else "contrastive"
    FS_technique = FS_technique if not(args.zero_shot) else "None"
    filename = f"{FS_technique}_{args.K}_{args.variety}_{CLS}_{args.source}_few_shots_results"

    # Convert the results in a DF and store it as a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results/{filename}.csv", index=False)

    # Generate a report with sklearn metrics and store it
    report_output = classification_report(results_df["label"], results_df["prediction"], digits=2)
    with open(f"results/{filename}.txt", "w", encoding="utf-8") as f:
        f.write(report_output)