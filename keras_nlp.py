import math
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D, Embedding, GRU, RNN


train_data = pd.read_csv('train.csv')
train_X = train_data["text"].values
target = train_data["target"].values

# create the tokenizer
tokenizer = Tokenizer()

# fit the tokenizer on the documents
tokenizer.fit_on_texts(train_X)

# encode sequences
encoded = tokenizer.texts_to_sequences(train_X)

# pad sequences
max_length = max(len(tokens) for tokens in encoded)
padded = pad_sequences(encoded, maxlen=max_length, padding='post')

train_X, test_X, train_y, test_y = train_test_split(
    padded, target, test_size=0.2)

# Hyperparameters
# +1 to avoid "index out of range" error in the vocab_size
vocab_size = len(tokenizer.index_word)+1
embedding_size = 50
hidden_size = 100
num_classes = 1
dropout = 0.2
batch_size = 64
learning_rate = 0.005
num_epochs = 5


model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=max_length))
model.add(GRU(hidden_size, dropout=dropout))
model.add(Dense(num_classes, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

print(
    f'...Vocabulary length is {vocab_size}, Embedding Vector is therefore of dimensions [{vocab_size}, {embedding_size}]...')

history = model.fit(train_X, train_y,
                    epochs=num_epochs, batch_size=batch_size, validation_data=(test_X, test_y))
