
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import re
from nltk.corpus import brown, stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# Download required nltk packages
nltk.download('brown')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Safe download for POS tagger
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Hyperparameters
CONTEXT_SIZE = 4
EMBEDDING_DIM = 50
HIDDEN_DIM = 100
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.001
VOCAB_SIZE = 10000

# Helper function for POS tag conversion
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Load and preprocess corpus
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

sentences = brown.sents()
preprocessed_sents = []
for sent in sentences:
    tokens = [re.sub(r"\W+", "", word.lower()) for word in sent if word.isalpha()]
    tokens = [t for t in tokens if t and t not in stop_words]
    tagged = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged]
    if lemmatized:
        preprocessed_sents.append(lemmatized)

# Build vocabulary
word_counts = Counter(word for sent in preprocessed_sents for word in sent)
most_common = word_counts.most_common(VOCAB_SIZE - 1)
vocab = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}
vocab['<UNK>'] = 0
inv_vocab = {idx: word for word, idx in vocab.items()}

# Create training data
training_data = []
for sent in preprocessed_sents:
    indexed = [vocab.get(word, 0) for word in sent]
    if len(indexed) >= CONTEXT_SIZE + 1:
        for i in range(len(indexed) - CONTEXT_SIZE):
            context = indexed[i:i+CONTEXT_SIZE]
            target = indexed[i+CONTEXT_SIZE]
            training_data.append((context, target))

class NGramDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context), torch.tensor(target)

train_dataset = NGramDataset(training_data)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Neural Probabilistic Language Model
class NNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
        super(NNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embeds = self.embedding(x)
        embeds = embeds.view(embeds.size(0), -1)
        hidden_out = self.tanh(self.hidden(embeds))
        output = self.output(hidden_out)
        return output

model = NNLM(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for contexts, targets in train_loader:
        outputs = model(contexts)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# Evaluation: compute perplexity
with torch.no_grad():
    total_loss = 0
    total_count = 0
    for contexts, targets in train_loader:
        outputs = model(contexts)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * targets.size(0)
        total_count += targets.size(0)
    avg_loss = total_loss / total_count
    perplexity = np.exp(avg_loss)
    print(f"Perplexity: {perplexity:.2f}")

# Test example
context_example = random.choice(training_data)[0]
context_words = [inv_vocab[idx] for idx in context_example]
print("Context:", context_words)
with torch.no_grad():
    output = model(torch.tensor([context_example]))
    predicted_idx = torch.argmax(output, dim=1).item()
    predicted_word = inv_vocab[predicted_idx]
    print("Predicted next word:", predicted_word)
