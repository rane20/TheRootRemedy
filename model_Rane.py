import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch import nn
from transformers import BertModel
import os


# Load the dataset
df = pd.read_csv('og_intent_dataset.csv')

# Check unique values for Sentiment
unique_sentiments = df['Sentiment'].unique()
print("Unique sentiments:", unique_sentiments)

# Check unique values for Intent
unique_intents = df['Intent'].unique()
print("Unique intents:", unique_intents)

# Step 1: Integer Encoding for Sentiment
le_sentiment = LabelEncoder()
df['Sentiment'] = le_sentiment.fit_transform(df['Sentiment'])  # Convert Sentiment into integer labels

# Step 2: Integer Encoding for Intent (One label per row)
le_intent = LabelEncoder()
df['Intent'] = le_intent.fit_transform(df['Intent'])  # Convert Intent into integer labels

# Tokenize sentences
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(df['Sentence'].tolist(), padding=True, truncation=True, return_tensors="pt", max_length=128)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    inputs['input_ids'], df['Sentiment'], test_size=0.2, random_state=42
)
intent_train, intent_test = train_test_split(df['Intent'], test_size=0.2, random_state=42)

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, input_ids, sentiment_labels, intent_labels=None):
        self.input_ids = input_ids
        self.sentiment_labels = sentiment_labels
        self.intent_labels = intent_labels

    def __len__(self):
        return len(self.sentiment_labels)

    def __getitem__(self, idx):
        data = {
            'input_ids': self.input_ids[idx],
            'labels': self.sentiment_labels[idx]
        }
        if self.intent_labels is not None:
            data['intent_labels'] = self.intent_labels[idx]
        return data

# Convert to tensors
X_train_tensor = X_train
X_test_tensor = X_test
y_train_tensor = torch.tensor(y_train.values).long()
y_test_tensor = torch.tensor(y_test.values).long()
intent_train_tensor = torch.tensor(intent_train.values).long()
intent_test_tensor = torch.tensor(intent_test.values).long()

# Create training and evaluation datasets
train_dataset = CustomDataset(X_train_tensor, y_train_tensor, intent_train_tensor)
eval_dataset = CustomDataset(X_test_tensor, y_test_tensor, intent_test_tensor)

# Define MultiTaskBERT model- used on top of BERT to extract rich embeddings(vector representations of data)
class MultiTaskBERT(nn.Module):
    def __init__(self, num_sentiment_labels, num_intent_labels):
        super(MultiTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased') #loads pre-trained BERT model
        self.sentiment_classifier = nn.Linear(self.bert.config.hidden_size, num_sentiment_labels) 
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intent_labels)

    def forward(self, input_ids, attention_mask=None, labels=None, intent_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        sentiment_logits = self.sentiment_classifier(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        loss = None
        if labels is not None and intent_labels is not None:
            # Compute sentiment loss using CrossEntropyLoss
            loss_fn_sentiment = nn.CrossEntropyLoss()
            loss_sentiment = loss_fn_sentiment(sentiment_logits, labels)
            
            # Compute intent loss using CrossEntropyLoss for multi-class
            loss_fn_intent = nn.CrossEntropyLoss()
            loss_intent = loss_fn_intent(intent_logits, intent_labels)

            loss = loss_sentiment + loss_intent

        return {'loss': loss, 'sentiment_logits': sentiment_logits, 'intent_logits': intent_logits}

# Initialize the model
model = MultiTaskBERT(num_sentiment_labels=2, num_intent_labels=len(le_intent.classes_))

# Define training arguments for the model (Hyperparameters)
training_args = TrainingArguments(
    output_dir='./results', #Where checkpoints are being saved
    num_train_epochs=4, #number of times training of dataset will be passed through
    per_device_train_batch_size=8, #processes 8 samples at a time
    per_device_eval_batch_size=8, #evaluates 8 processes at a time
    learning_rate=5e-5,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2
)

# Define custom training loop- evaluate model's performance with metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    sentiment_logits, intent_logits = logits
    # Add accuracy or F1 computation here if needed
    return {}

#handles training loop by automating the following arguments
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)


os.makedirs('./trained_model', exist_ok=True)
# Save model state dictionary
torch.save(model.state_dict(), './trained_model/multitask_bert.pth')

# Save tokenizer (this part remains the same as tokenizer is from Hugging Face)
tokenizer.save_pretrained('./trained_model')


# Reinitialize the model and load saved weights
# Initialize the model
model = MultiTaskBERT(num_sentiment_labels=2, num_intent_labels=len(le_intent.classes_))
model.load_state_dict(torch.load('./trained_model/multitask_bert.pth'))
model.eval()  # Set the model to evaluation mode

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('./trained_model')


# Premade responses based on sentiment and inquiry intent
responses = {
    "Positive": {
        "Gut Health Inquiry": "Great to hear you're curious about gut health! It's essential for overall wellness.",
        "Wellness Inquiry": "Glad you're exploring wellness. A balanced lifestyle is key.",
        "Gut Supplements Inquiry": "Gut supplements can support your digestive health. It's always good to consult a professional.",
        "General Inquiry": "Thank you for your question. Let's explore more!",
        "Beauty Inquiry": "Beauty starts from within, and gut health plays a big role in it.",
        "Skin Health Inquiry": "Healthy skin starts with a healthy gut. Keep taking care of yourself!"
    },
    "Negative": {
        "Gut Health Inquiry": "Sorry to hear you're having gut issues. It might be worth discussing with a healthcare provider.",
        "Wellness Inquiry": "Wellness challenges can be tough, but you're not alone in this journey.",
        "Gut Supplements Inquiry": "Supplements might help, but consult a professional to ensure they suit your needs.",
        "General Inquiry": "We're here to help. Don't hesitate to ask more questions.",
        "Beauty Inquiry": "Beauty is complex, and sometimes external factors can affect it. Keep a positive mindset!",
        "Skin Health Inquiry": "Skin concerns can be frustrating. Focus on hydration and a balanced diet to improve skin health."
    }
}

# Function to get premade response
def get_response(sentiment, intent):
    sentiment_label = le_sentiment.inverse_transform([sentiment])[0]
    intent_label = le_intent.inverse_transform([intent])[0]
    
    if sentiment_label == 'Positive':
        return responses["Positive"].get(intent_label, "I don't have a response for that.")
    elif sentiment_label == 'Negative':
        return responses["Negative"].get(intent_label, "I don't have a response for that.")
    else:
        return "I don't have a response for that."

# Take user input
while True:
    user_input = input("Enter a sentence (or 'exit' to quit): ")
    
    if user_input.lower() == 'exit':
        break

    inputs = tokenizer([user_input], padding=True, truncation=True, return_tensors="pt", max_length=128)

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        sentiment_predictions = torch.argmax(outputs['sentiment_logits'], dim=1)
        intent_predictions = torch.argmax(outputs['intent_logits'], dim=1)

    # Convert predictions back to original labels
    predicted_sentiments = le_sentiment.inverse_transform(sentiment_predictions.numpy())
    predicted_intents = le_intent.inverse_transform(intent_predictions.numpy())

    print(f"Predicted Sentiment: {predicted_sentiments[0]}")
    print(f"Predicted Intent: {predicted_intents[0]}")

    # Get premade response based on predictions
    response = get_response(sentiment_predictions.item(), intent_predictions.item())
    print("Response:", response)