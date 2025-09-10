import streamlit as st
import json
from test_Inference import generate_response


st.set_page_config(page_title="LLM Playground", page_icon="🤖")
st.title("LLM Playground") #[cite: 46]
st.subheader("Conversational AI Web Application")
st.write("A simple chat application powered by Hugging Face Transformers and Streamlit.")
models = [
    "mistralai/Mistral-7B-Instruct-v0.3",  # Example: a supported conversational model
    # Add other supported conversational models here if needed
]
selected_model = st.selectbox("Select a model:", models)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am a conversational AI. How can I help you today?"}
    ]

with st.sidebar:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am a conversational AI. How can I help you today?"}
        ]
        st.rerun()

    chat_history_json = json.dumps(st.session_state.messages, indent=2)
    st.download_button(
        label="Export Conversation",
        data=chat_history_json,
        file_name="conversation.txt",
        mime="text/plain",
        use_container_width=True
    )
    
    st.markdown("---")
    st.markdown("Optional Enhancements (Bonus Marks):")
    st.markdown("- **Time/Date Stamp:** Add timestamps to chat messages.")
    st.markdown("- **Light/Dark Theme:** Add a theme switcher.")
    st.markdown("- **Typing Indicator:** Use `st.spinner` for an animated indicator.") #[cite: 42]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("What is on your mind?"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."): 
            try:
                response = generate_response(selected_model, st.session_state.messages)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")