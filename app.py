import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load the model and tokenizer
@st.cache_resource
def load_components():
    model = load_model('rnn_text_generator_model.h5')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer

def generate_text(model, tokenizer, seed_text, max_length, num_words):
    """Generate text using the trained model"""
    generated_text = seed_text
    for _ in range(num_words):
        # Tokenize the current text
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        # Pad the sequence
        token_list = pad_sequences([token_list], maxlen=max_length-1, padding='pre')
        # Predict the next word
        predicted_probs = model.predict(token_list, verbose=0)[0]
        # Get the index of the word with the highest probability
        predicted_index = np.argmax(predicted_probs)
        # Convert index to word
        predicted_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                predicted_word = word
                break
        # Add the word to the generated text
        generated_text += " " + predicted_word
    return generated_text

def main():
    st.title("RNN Text Generator")
    st.write("This app generates text using a trained Recurrent Neural Network")
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_components()
        max_seq_len = 10  # This should match what you used during training
        
        # User input
        seed_text = st.text_input("Enter starting text:", "The weather is")
        num_words = st.slider("Number of words to generate:", 1, 20, 5)
        
        if st.button("Generate Text"):
            if seed_text:
                with st.spinner("Generating text..."):
                    generated = generate_text(model, tokenizer, seed_text, max_seq_len, num_words)
                    st.success("Generated Text:")
                    st.write(generated)
            else:
                st.warning("Please enter some starting text.")
                
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Make sure you have the model files (rnn_text_generator_model.h5 and tokenizer.pickle) in the same directory as this app.")

if __name__ == "__main__":
    main()