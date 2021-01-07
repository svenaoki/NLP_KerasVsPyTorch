import math
import pandas as pd
import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


train_data = pd.read_csv('train.csv')
train_X = train_data["text"].values
target = train_data["target"].values

# create the tokenizer
tokenizer = Tokenizer()

# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_X)
# index2word = tokenizer.index_word
# word2index = tokenizer.word_index
# encode sequences
encoded = tokenizer.texts_to_sequences(train_X)

# pad sequences
max_length = max(len(tokens) for tokens in encoded)
padded = pad_sequences(encoded, maxlen=max_length, padding='post')

# custom dataset to load data in batches


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, inputs, target):
        'Initialization'
        self.inputs = inputs
        self.target = target

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        input_sample, target_sample = self.inputs[index], self.target[index]

        return torch.tensor(input_sample).to(torch.long), torch.tensor(target_sample).to(torch.long)

# GRU model


class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_classes):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.GRU = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embeddings = self.embeddings(x)
        # [64, 33, 32] [batch_size, seq_length, embedding_size]
        hidden = torch.zeros(1, x.size(0), self.hidden_size)
        # [1, 64, 100] [num_layer, batch_size, hidden_size]
        output, hidden = self.GRU(embeddings, hidden)
        # output = [64, 33 100] [batch_size, seq_length, hidden_size];
        # hiddenn = [1, 64, 100] [num_layers, batch_size, hidden_size]
        output = output[:, -1, :]
        # [64, 100] [batch_size, (last) hidden_size]
        output = self.fc(output)
        # output = [64, 1] [batch_size, num_classes]
        return output, hidden


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
vocab_size = len(tokenizer.index_word)+1
embedding_size = 50
hidden_size = 100
num_classes = 1
batch_size = 128
num_epochs = 10

# Set up pyTsorch's Dataloader
train_data = Dataset(padded, target)
train_loader = torch.utils.data.DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=True)


# initialise the model and criteria
rnn = GRU(vocab_size, embedding_size, hidden_size,
          num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.005)

n_samples = 0
n_correct = 0
for epoch in range(num_epochs):
    for i, (X, y) in enumerate(train_loader):
        y = y.float().to(device)
        X = X.to(device)

        optimizer.zero_grad()
        y_hat, _ = rnn(X)

        loss = criterion(y_hat, y.view(-1, 1))
        loss.backward()
        optimizer.step()

        # sigmoid was not applied in the model class
        sig_yhat = torch.sigmoid(y_hat)
        predictions = torch.round(sig_yhat)
        n_samples += y.size(0)
        n_correct += (predictions.view(-1) == y).sum().item()

        if i == 10:
            acc = 100.0 * n_correct / n_samples
            print(loss.detach().item())
            print(f'Accuracy of the network: {acc} %')
