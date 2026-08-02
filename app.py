import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM Model
model=load_model("best_next_word_model.keras")

#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_top_words(model, tokenizer, text, max_sequence_len, top_k=5):

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return []

    # Keep only the latest words if sentence is long
    token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences(
        [token_list],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )
    prediction = model.predict(token_list, verbose=0)[0]
    top_indices = np.argsort(prediction)[-top_k:][::-1]
    results = []
    for idx in top_indices:

        word = tokenizer.index_word.get(idx, "<UNK>")

        probability = prediction[idx]

        results.append((word, probability))

    return results

# streamlit app
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="📝",
    layout="centered"
)
st.title("📝 Next Word Prediction using LSTM")
st.write(
    "Enter a sentence and the model will predict the most probable next words."
)
text = st.text_input(
    "Enter text",
    placeholder="Example: to be or not"
)
if st.button("Predict Next Word"):

    if text.strip() == "":

        st.warning("Please enter some text.")

    else:

        predictions = predict_top_words(
            model,
            tokenizer,
            text.lower(),
            max_sequence_len,
            top_k=5
        )

        if len(predictions) == 0:

            st.error("No valid words found in tokenizer.")

        else:

            st.success("Top Predictions")

            for i, (word, prob) in enumerate(predictions, start=1):

                st.write(
                    f"**{i}. {word}**  —  {prob*100:.2f}%"
                )

