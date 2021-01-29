import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Embedding
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.compat.v1.keras.layers import CuDNNLSTM

from PreProcessings import tokenize_and_add_padding, stemming_and_lemmatization, normalizer, one_hot_encoder

dataset_link = "https://drive.google.com/file/d/1Re3OYrevmlMscyNSBsDeIdX2xfndnM7C/view?usp=sharing"
df = pd.read_csv("/content/train.csv")

raw_texts = df['Text']
raw_lables = df['Category']

X = tokenize_and_add_padding(stemming_and_lemmatization(normalizer(raw_texts)))
Y = one_hot_encoder(raw_lables)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

# The maximum number of words to be used.
MAX_NB_WORDS = 350000

# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250

EMBEDDING_DIM = 100

## creating model
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(Conv1D(200, 2, activation='softmax'))
model.add(SpatialDropout1D(0.25))
model.add(CuDNNLSTM(128))
model.add(Dense(34, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

epochs = 8
batch_size = 128
history = model.fit(X_train, Y_train, epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)]
                    )

accr = model.evaluate(X_test, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))
