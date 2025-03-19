# %%
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dropout, Dense, LSTM
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# %%
text = open('text_data.txt', 'r', encoding='utf-8').read()

# %%
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# %%
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)
        
# %%
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# %%
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = to_categorical(y, num_classes=total_words)

# %%
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)
        predicted_word = tokenizer.index_word[predicted_word_index[0]]
        seed_text += " " + predicted_word
        
    return seed_text

# %%
for emdeding_size in [300]:
    for epochs in [200]: 
        model = Sequential()
        model.add(Embedding(total_words, emdeding_size, input_length=max_sequence_len-1))
        model.add(LSTM(150, return_sequences=True))
        model.add(LSTM(100))
        model.add(Dense(total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        history = model.fit(X, y, epochs=epochs, verbose=1)
        
        chart_params_label = 'Embedding Size=%d, Epochs=%d' % (emdeding_size, epochs)
        plt.plot(history.history['loss'])
        plt.title('[' + chart_params_label + ']: Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

        print(generate_text("Three quarks for Muster Mark", 50, model, max_sequence_len))
        print(generate_text("Three quarks", 50, model, max_sequence_len))
        print(generate_text("at the end of this age", 50, model, max_sequence_len))
# %%
