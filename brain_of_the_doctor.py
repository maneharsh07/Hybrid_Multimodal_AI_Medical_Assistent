#-------------------------------------------DOCTOR BRAIN (IMAGE ANALYSIS)---------------------------------------------

import os
import base64
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Default model
DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


#-------------------------------------------IMAGE ENCODING---------------------------------------------

def encode_image(image_path):
    """
    Convert image to base64 format for API usage
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None


#-------------------------------------------IMAGE ANALYSIS FUNCTION---------------------------------------------

def analyze_image_with_query(query, encoded_image, model=DEFAULT_MODEL):
    """
    Analyze medical image with user query

    NOTE:
    - Always pass English query (translation handled in main file)
    """

    try:
        if not GROQ_API_KEY:
            return "GROQ API key missing. Please check your .env file."

        client = Groq(api_key=GROQ_API_KEY)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ]

        response = client.chat.completions.create(
            messages=messages,
            model=model
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error analyzing image: {str(e)}"


#-------------------------------------------TEST (OPTIONAL)---------------------------------------------

if __name__ == "__main__":
    test_image = "test.jpg"  # Replace with your image path

    encoded = encode_image(test_image)

    if encoded:
        result = analyze_image_with_query(
            query="Check if there is any medical issue in this image",
            encoded_image=encoded
        )
        print("Doctor AI Response:\n", result)