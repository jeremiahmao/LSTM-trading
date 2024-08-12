from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news):
    if not news:
        # Return a DataFrame with zero probabilities if no input is provided
        return pd.DataFrame([[0] * len(labels)], columns=labels)

    # Tokenize the batch of news
    tokens = tokenizer(news, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Forward pass through the model
    with torch.no_grad():  # No need to compute gradients
        logits = model(**tokens).logits
    
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Average probabilities over the batch
    avg_probabilities = probabilities.mean(dim=0).cpu().numpy()
    
    # Create a DataFrame with probabilities and labels
    df = pd.DataFrame([avg_probabilities], columns=labels)
    
    return df

if __name__ == "__main__":
    
    df = estimate_sentiment([
        'markets responded POSITIVE to the news!',
        'traders were pleased!'
    ])
    print(df)
    print("CUDA available:", torch.cuda.is_available())