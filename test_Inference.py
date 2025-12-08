import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("hugging_face")

if not HF_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN is not set. Please paste your token.")

def generate_response(model_name: str, messages: list) -> str:
    """
    Connects to a Hugging Face model and generates a response for a conversation.
    """
    try:
        client = InferenceClient(model=model_name, token=HF_TOKEN)
        response = client.chat_completion(
            messages=messages,
            max_tokens=200,
            stream=False,
            temperature=0.7,
            top_p=0.95
        )
        
        return response.choices[0].message.content

    except Exception as e:
        return f"An error occurred during inference: {e}"

if __name__ == '__main__':
    test_messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    test_model = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"Testing response from model: {test_model}")
    
    if HF_TOKEN == "hf_YourNewTokenGoesHere":
        print("Please paste your real Hugging Face token into the HF_TOKEN variable.")
    else:
        test_response = generate_response(test_model, test_messages)
        print(f"AI: {test_response}")
