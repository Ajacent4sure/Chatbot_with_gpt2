import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 model and tokenizer from Hugging Face
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Function to generate a response
def generate_response(user_input):
    prompt = f"Question: {user_input}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.5,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Answer:" in response:
        response = response.split("Answer:")[1].strip()
    
    return response

# Streamlit UI
st.title("GPT-2 Chatbot")

user_input = st.text_input("You:", key="input")

if st.button("Send"):
    if user_input.strip():
        response = generate_response(user_input)
        st.text_area("Bot:", value=response, height=100, disabled=True)
    else:
        st.warning("Please enter a message.")
