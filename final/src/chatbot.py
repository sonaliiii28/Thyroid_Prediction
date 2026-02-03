import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_thyroid_assistant_response(user_query, chat_history=None):
    """
    Generates a response from OpenAI's GPT-4o model specialized for thyroid health.
    """
    if chat_history is None:
        chat_history = []

    system_prompt = {
        "role": "system",
        "content": (
            "You are a specialized Thyroid Health AI Assistant for the 'ThyroidAI Pro' platform. "
            "Your goal is to help users understand thyroid disorders, interpret clinical measurements "
            "(like TSH, T3, T4, T4U, FTI), and explain how machine learning models help in prediction. "
            "\n\nRules:\n"
            "1. ALWAYS provide a medical disclaimer at the end of your response: 'Disclaimer: I am an AI, not a doctor. Consult a medical professional for diagnosis.'\n"
            "2. If the user asks about their specific prediction from the app, explain the clinical relevance of the features used (e.g., how high TSH often indicates hypothyroidism).\n"
            "3. Use a professional, empathetic, and clear tone.\n"
            "4. Keep responses concise but informative."
        )
    }

    # Prepare messages for API
    messages = [system_prompt]
    # Add recent history (last 5 exchanges to keep it concise)
    messages.extend(chat_history[-10:])
    messages.append({"role": "user", "content": user_query})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using the latest reliable model
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"I'm sorry, I'm having trouble connecting right now. Error: {str(e)}"
