# app.py
import streamlit as st
import requests
import json

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:1b"


def initialize_ollama():
    """Verify Ollama is running and model is available"""
    try:
        response = requests.get("http://localhost:11434")
        response.raise_for_status()

        # Check if model is available
        models = requests.get("http://localhost:11434/api/tags").json()
        if not any(m["name"].startswith(MODEL_NAME) for m in models.get("models", [])):
            st.error(
                f"Model {MODEL_NAME} not found in Ollama. Run 'ollama pull {MODEL_NAME}' first.")
            st.stop()
    except requests.ConnectionError:
        st.error("Ollama not running. Please start Ollama first.")
        st.stop()


def generate_response(prompt, context=None):
    """Generate response using Ollama API"""
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "context": context,
            "options": {
                "temperature": 0.7,
                "top_p": 0.95,
                "num_predict": 512
            }
        }

        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()

        result = response.json()
        return result["response"], result.get("context")

    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return "Sorry, I'm having trouble responding right now.", context


# Initialize Ollama connection
initialize_ollama()

# Set up the Streamlit UI
st.title("💎 CPU-Friendly Chatbot with Gemma 3B")
st.markdown("This demo uses Gemma 3B via Ollama for local CPU inference")

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "context" not in st.session_state:
    st.session_state.context = None


def display_conversation():
    for chat in st.session_state.conversation:
        if chat["role"] == "user":
            st.markdown(f"**You:** {chat['content']}")
        else:
            st.markdown(f"**Gemma:** {chat['content']}")


# Chat input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message", placeholder="Type here...")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input:
    # Add user message to conversation
    st.session_state.conversation.append(
        {"role": "user", "content": user_input})

    # Generate response
    with st.spinner("Gemma is thinking..."):
        response, new_context = generate_response(
            user_input, st.session_state.context)

        # Update context and conversation
        if response:
            st.session_state.context = new_context
            st.session_state.conversation.append(
                {"role": "assistant", "content": response})

# Display conversation
display_conversation()


# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM


# def initialize_model(model_name="microsoft/DialoGPT-small"):
#     """
#     Load and return the tokenizer and model for DialoGPT-small.
#     The model is explicitly loaded on CPU.
#     """
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     model.to("cpu")
#     return tokenizer, model


# def generate_response(tokenizer, model, chat_history, user_input, max_length=1000):
#     """
#     Generate a response from the model using the current chat history and latest user input.
#     Returns the model's response text and updated chat history.
#     """
#     # Encode user input with the EOS token
#     new_input_ids = tokenizer.encode(
#         user_input + tokenizer.eos_token, return_tensors="pt")

#     # Create the full bot input by concatenating the prior history (if any) with the new user input
#     if chat_history is not None:
#         bot_input_ids = torch.cat([chat_history, new_input_ids], dim=-1)
#     else:
#         bot_input_ids = new_input_ids

#     # Save the length of the input to correctly slice out the generated response later
#     input_length = bot_input_ids.shape[-1]

#     # Generate a response using the concatenated input
#     output_ids = model.generate(
#         bot_input_ids,
#         max_length=max_length,
#         pad_token_id=tokenizer.eos_token_id,
#         no_repeat_ngram_size=3,
#         do_sample=True,
#         top_k=50,
#         top_p=0.95,
#         temperature=0.7
#     )

#     # Extract only the newly generated tokens after the entire input
#     response_ids = output_ids[:, input_length:]
#     response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

#     # Set the updated chat history to the full output (input + new response)
#     updated_history = output_ids
#     return response_text, updated_history


# # Initialize the model and tokenizer
# tokenizer, model = initialize_model()

# # Set up the Streamlit UI
# st.title("CPU Friendly Chatbot with Streamlit")
# st.markdown(
#     "This demo uses DialoGPT‑small running on the CPU for a responsive conversational experience.")

# # Initialize session state for conversation if not already initialized
# if "chat_history" not in st.session_state:
#     st.session_state["chat_history"] = None
# if "conversation" not in st.session_state:
#     st.session_state["conversation"] = []


# def display_conversation():
#     for chat in st.session_state.conversation:
#         if chat["sender"] == "You":
#             st.markdown(f"**You:** {chat['message']}")
#         else:
#             st.markdown(f"**Bot:** {chat['message']}")


# # Input form for chat messages
# with st.form(key="chat_form", clear_on_submit=True):
#     user_input = st.text_input(
#         "Enter your message", placeholder="Type here...")
#     submit_button = st.form_submit_button(label="Send")

# if submit_button and user_input:
#     # Append the user's message to the conversation history
#     st.session_state.conversation.append(
#         {"sender": "You", "message": user_input})

#     # Generate a response and update chat history
#     response_text, updated_history = generate_response(
#         tokenizer, model, st.session_state.chat_history, user_input
#     )
#     st.session_state.chat_history = updated_history

#     # Append the bot's response to the conversation history
#     st.session_state.conversation.append(
#         {"sender": "Bot", "message": response_text})

# # Display the conversation
# display_conversation()
