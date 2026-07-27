#-------------------------------------------MULTI LANGUAGE SUPPORT---------------------------------------------

from deep_translator import GoogleTranslator

def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

def translate_from_english(text, target_lang):
    try:
        if target_lang == "en":
            return text
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except:
        return text


#-------------------------------------------BRAIN OF THE DOCTOR---------------------------------------------

import os
import base64
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except:
        return None


def analyze_image_with_query(query, model, encoded_image):
    client = Groq(api_key=GROQ_API_KEY)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": query},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
        ]
    }]

    response = client.chat.completions.create(
        messages=messages,
        model=model
    )

    return response.choices[0].message.content


#--------------------------------------------VOICE OF THE DOCTOR---------------------------------------------

from gtts import gTTS
from elevenlabs import save
from elevenlabs.client import ElevenLabs

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

def text_to_speech_with_gtts(text, path, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tts.save(path)
    return path

def text_to_speech_with_elevenlabs(text, path):
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.text_to_speech.convert(
        voice_id="21m00Tcm4TlvDq8ikWAM",
        model_id="eleven_turbo_v2",
        text=text,
        output_format="mp3_22050_32"
    )
    save(audio, path)
    return path


#------------------------------------VOICE OF THE PATIENT-----------------------------------------------------

import speech_recognition as sr
from groq import Groq

def transcribe_with_groq(audio_filepath, lang="en"):
    try:
        client = Groq(api_key=GROQ_API_KEY)
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                language=lang
            )
        return transcription.text
    except:
        return "Error in transcription"


#--------------------------------------------GRADIO APP--------------------------------------------------

import gradio as gr

LANG_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr"
}

system_prompt = "Act as a doctor and give short medical advice."


def process_inputs(audio_filepath, image_filepath, language):
    lang_code = LANG_MAP[language]

    # 🎤 Speech → Text
    if audio_filepath:
        speech_text = transcribe_with_groq(audio_filepath, lang=lang_code)
    else:
        speech_text = ""

    # 🌍 Translate → English
    english_input = translate_to_english(speech_text)

    # 🧠 AI Image Analysis
    if image_filepath:
        doctor_response = analyze_image_with_query(
            query=system_prompt + " " + english_input,
            encoded_image=encode_image(image_filepath),
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
    else:
        doctor_response = "No image provided"

    # 🌍 Translate back
    final_response = translate_from_english(doctor_response, lang_code)

    # 🔊 Voice Output
    audio_path = "doctor_response.mp3"
    if ELEVENLABS_API_KEY:
        text_to_speech_with_elevenlabs(final_response, audio_path)
    else:
        text_to_speech_with_gtts(final_response, audio_path, lang=lang_code)

    return speech_text, final_response, audio_path


# UI
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="filepath", label="🎤 Speak"),
        gr.Image(type="filepath", label="🖼 Upload Image"),
        gr.Dropdown(["English", "Hindi", "Marathi"], value="English", label="🌍 Language")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Doctor Response"),
        gr.Audio(label="Voice Output")
    ],
    title="🌍 Multilingual AI Doctor",
    description="AI Doctor with Vision + Voice + Multi-language Support"
)


if __name__ == "__main__":
    iface.launch(debug=True)