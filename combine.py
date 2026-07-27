#-------------------------------------------Combine My own model-------------------------------------------

from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load models
pneumonia_model = load_model("pneumonia_model.h5")
skin_model = load_model("skin_model.h5")

skin_classes = ["Acne", "Eczema", "Psoriasis"]

# pneumonia prediction
def predict_xray(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))

    pred = pneumonia_model.predict(img)[0][0]

    disease = "Pneumonia" if pred > 0.5 else "Normal"
    confidence = pred if pred > 0.5 else 1 - pred

    return disease, round(float(confidence)*100,2)

# Skin prediction 
def predict_skin(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.reshape(img, (1,224,224,3))

    pred = skin_model.predict(img)[0]
    idx = np.argmax(pred)

    disease = skin_classes[idx]
    confidence = pred[idx]

    return disease, round(float(confidence)*100,2)

# detect image type 
def detect_image_type(image_path):
    name = image_path.lower()

    if "xray" in name or "chest" in name:
        return "xray"
    else:
        return "skin"
    
def detect_image_with_ai(image_path):
    result = analyze_image_with_query(
        query="Is this a chest X-ray or a skin image? Answer only: xray or skin.",
        encoded_image=encode_image(image_path),
        model="meta-llama/llama-4-scout-17b-16e-instruct"
    )

    return "xray" if "xray" in result.lower() else "skin"

#-------------------------------------------MULTI LANGUAGE SUPPORT---------------------------------------------
def convert_to_local_language(text, lang_code):
    try:
        if lang_code == "en":
            return text

        llm = load_llm()

        prompt = f"""
        Translate the following medical sentence into {lang_code}.

        Rules:
        - Keep EXACT meaning same
        - Do NOT add extra information
        - Do NOT add disclaimer
        - Keep it short and clear
        - Only translation, nothing else

        Text:
        {text}
        """

        response = llm.invoke(prompt)
        return response.content.strip()

    except:
        return text
    
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
from dotenv import load_dotenv
load_dotenv()

import re

def clean_text(text):
    text = re.sub(r'[\*\#\@\!]', '', text)   # remove symbols
    text = re.sub(r'\d+\.', '', text)        # remove 1. 2. 3.
    text = text.replace("मोइरायझर", "मॉइश्चरायझर")
    text = text.replace("ओम3", "ओमेगा 3")
    text = text.replace(":", "")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

from groq import Groq

def analyze_image_with_query(query, model, encoded_image):
    client = Groq(api_key=GROQ_API_KEY)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": query},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
        ]
    }]

    chat_completions = client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completions.choices[0].message.content


#--------------------------------------------VOICE OF THE DOCTOR---------------------------------------------

from gtts import gTTS
from elevenlabs import save
from elevenlabs.client import ElevenLabs

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

def text_to_speech_with_gtts(input_text, output_filepath, lang="en"):
    audioobj = gTTS(text=input_text, lang=lang, slow=False)
    audioobj.save(output_filepath)
    return output_filepath

def text_to_speech_with_elevenlabs(input_text, output_filepath):
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    voice_id = "21m00Tcm4TlvDq8ikWAM"

    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        model_id="eleven_turbo_v2",
        text=input_text,
        output_format="mp3_22050_32"
    )
    save(audio, output_filepath)
    return output_filepath


#------------------------------------VOICE OF THE PATIENT-----------------------------------------------------

import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO

logging.basicConfig(level=logging.INFO)

def record_audio(file_path, timeout=20):
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio_data = recognizer.listen(source, timeout=timeout)

        wav_data = audio_data.get_wav_data()
        audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
        audio_segment.export(file_path, format="mp3", bitrate="128k")


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
    except Exception as e:
        return "Error in transcription"


#------------------------------------CHATBOT BRAIN-----------------------------------------------------

from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

DB_FAISS_PATH = "Medical_chatbot/vectorstore/db_faiss"

def get_vectorstore():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

def load_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)

def get_chatbot_response(question):
    try:
        # 🔹 Try FAISS
        vectorstore = get_vectorstore()

        if vectorstore:
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 2})
            )

            response = qa_chain.invoke({'query': question})
            result = response.get("result", "")

            if result and "don't know" not in result.lower():
                return result

        # 🔥 Fallback → ChatGPT
        llm = load_llm()
        response = llm.invoke(question)

        return response.content

    except:
        return "Something went wrong"


#--------------------------------------------GRADIO APP--------------------------------------------------

import gradio as gr

system_prompt = "Act as a doctor and give short medical advice."

LANG_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr"
}

def process_inputs(audio_filepath, image_filepath, language):
    lang_code = LANG_MAP[language]

    # Speech → text
    if audio_filepath:
        speech_text = transcribe_with_groq(audio_filepath, lang=lang_code)
    else:
        speech_text = ""

    # Translate → English
    english_input = translate_to_english(speech_text)

    # Image AI (YOUR MODEL + GPT)

    if image_filepath:

        # Step 1: Detect image type
        # img_type = detect_image_type(image_filepath)
        img_type = detect_image_with_ai(image_filepath)

        # Step 2: Predict using your models
        if img_type == "xray":
            disease, confidence = predict_xray(image_filepath)
        else:
            disease, confidence = predict_skin(image_filepath)

        # Step 3: Send result to GPT for explanation
        llm = load_llm()

        prompt = f"""
        A medical AI model predicted:

        Disease: {disease}
        Confidence: {confidence}%

        Patient question: {english_input}

        Give short, clear medical advice and precautions.
        """

        response = llm.invoke(prompt)
        doctor_response = response.content

    else:
        doctor_response = "No image provided"

    # Translate back
    # final_response = clean_text(
    #     translate_from_english(doctor_response, lang_code)
    # )
    final_response = clean_text(
        convert_to_local_language(doctor_response, lang_code)
    )
    # Voice output
    audio_path = "doctor.mp3"
    cleaned = clean_text(final_response)
    text_to_speech_with_gtts(cleaned, audio_path, lang=lang_code)

    return speech_text, final_response, audio_path


def chatbot_function(message, history, language):
    lang_code = LANG_MAP[language]

    # 🌍 Translate input
    english_msg = translate_to_english(message)

    # 🧠 STRONG CONTEXT (last 5 messages)
    context = ""
    for user, bot in history[-5:]:
        context += f"User: {user}\nDoctor: {bot}\n"

    # 🔥 SYSTEM PROMPT (VERY IMPORTANT)
    system_prompt = f"""
        You are a professional medical doctor assistant.

        Rules:
        - Always reply in this language: {lang_code}
        - Give practical advice and home remedies
        - Speak like a real doctor (friendly and helpful)
        - Keep answer short (3-5 lines)
        - If user asks follow-up question, use previous context
        - Never say "I don't know"

        Focus:
        - Explain problem simply
        - Tell what patient should do
        """
    full_query = f"""
    {system_prompt}

    Conversation:
    {context}

    User question: {english_msg}

    Give helpful medical advice and home remedies.
    """

    # 🤖 Get response
    response = get_chatbot_response(full_query)

    # 🌍 Translate back
    # Force translation properly
    # translated = translate_from_english(response, lang_code)
    # final_response = clean_text(translated)
    final_response = clean_text(
    convert_to_local_language(response, lang_code)
    )

    history.append([message, final_response])

    # 🔊 Voice
    audio_path = "chat.mp3"
    text_to_speech_with_gtts(final_response, audio_path, lang=lang_code)

    return "", history, audio_path


# UI
with gr.Blocks() as iface:
    gr.Markdown("# 🌍 Multilingual AI Doctor")

    language = gr.Dropdown(["English", "Hindi", "Marathi"], value="English", label="Select Language")

    with gr.Tab("Vision Doctor"):
        audio = gr.Audio(type="filepath")
        image = gr.Image(type="filepath")
        btn = gr.Button("Analyze")

        out1 = gr.Textbox()
        out2 = gr.Textbox()
        out3 = gr.Audio()

        btn.click(process_inputs, [audio, image, language], [out1, out2, out3])

    with gr.Tab("Chatbot"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        send = gr.Button("Send")
        audio_out = gr.Audio()

        send.click(chatbot_function, [msg, chatbot, language], [msg, chatbot, audio_out])


if __name__ == "__main__":
    iface.launch(debug=True)