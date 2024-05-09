from tensorflow.keras.models import load_model
import pickle 
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import pandas as pd
import io
# Load the pre-trained model
model = load_model("/Users/meetshah/Desktop/model_final/model2.h5")

# Load the tokenizer
with open("/Users/meetshah/Desktop/model_final/tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

print(model.summary())
# Function to generate lyrics
def generate_lyrics(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list)[0]

        # Sample the next word based on the predicted probabilities
        predicted_index = np.random.choice(len(predicted_probs), p=predicted_probs)

        # Map the index back to the word
        output_word = tokenizer.index_word.get(predicted_index, "")

        seed_text += " " + output_word
    return seed_text


df = pd.read_csv("final_final_data.csv")


desc = "Uses an LSTM neural network trained on songs by various artist to generate new lyrics" 
model.summary()

def main():
    st.set_page_config(page_title="Generate Lyrics With Ai",layout="wide",initial_sidebar_state="collapsed")
    # Set title with an attractive font and spacing
    st.title("Generate Lyrics Using Neural Networks🎶",help=desc)

    
    st.sidebar.title("Options")
    
    show_dataset = st.sidebar.checkbox("Show Dataset", False, help="Click to show the dataset on which the model is trained.")
    if show_dataset:
        st.sidebar.dataframe(df)
        
    show_result = st.sidebar.checkbox("Show Training Accuracy",False,help="Click to show how model learned over time")
    if show_result:
        st.sidebar.image('/Users/meetshah/Desktop/model_final/__results___16_0.png')
        
        
    show_model = st.sidebar.checkbox("Show Model Architecture",False,help="Click to show how model was built")
    if show_model:
        summary_str = io.StringIO()
        model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
        st.sidebar.code(summary_str.getvalue())
        
        
    st.subheader("Enter few words to start the lyrics:",help="Please DON'T Enter Capital Letters, Numbers, Special Charachters Or Any Punctuation")
    seed_text = st.text_input("")
    st.subheader("Chose the lenght of output: ",help="Determines the number of words in the output")
    next_words = st.select_slider("",options=[10,20,30,40,50,60,70,80,90,100])

 # Button to generate lyrics with custom styling
    if st.button("Generate Lyrics", key="generate_btn", help="Click to generate lyrics"):
        spinner = st.spinner("Please wait, generating lyrics...")
        with spinner:
            generated_lyrics = generate_lyrics(seed_text, next_words, max_sequence_len=1003)
        st.text_area("", generated_lyrics, height=None, help="Generated lyrics will appear here.")


if __name__ == "__main__":
    main()