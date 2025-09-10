import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
# --- FINAL FIX ---
# 1. PASTE YOUR NEW, VALID TOKEN HERE. This is the one we just tested.
HF_TOKEN = os.getenv("hugging_face")

# 2. Make sure the token is available for the client
if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN is not set. Please paste your token.")

# 3. Initialize the client globally or within the function.
# Here, we will do it inside the function for clarity.

def generate_response(model_name: str, messages: list) -> str:
    """
    Connects to a Hugging Face model and generates a response for a conversation.
    """
    try:
        # Initialize the client with the model and the confirmed working token
        client = InferenceClient(model=model_name, token=HF_TOKEN)

        # Use the correct method for conversational models
        response = client.chat_completion(
            messages=messages,
            max_tokens=200,
            stream=False,
            temperature=0.7,
            top_p=0.95
        )
        
        # Extract the response content correctly
        return response.choices[0].message.content

    except Exception as e:
        # Return a detailed error if something still goes wrong
        return f"An error occurred during inference: {e}"

# This part is for testing the file directly
if __name__ == '__main__':
    test_messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    test_model = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"Testing response from model: {test_model}")
    
    # Check if the token was pasted before running the test
    if HF_TOKEN == "hf_YourNewTokenGoesHere":
        print("Please paste your real Hugging Face token into the HF_TOKEN variable.")
    else:
        test_response = generate_response(test_model, test_messages)
        print(f"AI: {test_response}")