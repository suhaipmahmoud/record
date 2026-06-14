import streamlit as st
import pandas as pd
from docx import Document
from openai import OpenAI
import os

st.title("🎙️ Lightning Fast Transcriber")

# يجب وضع OPENAI_API_KEY في إعدادات Streamlit Secrets
if "OPENAI_API_KEY" not in st.secrets:
    st.error("يرجى إضافة OPENAI_API_KEY في إعدادات Secrets")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

uploaded_file = st.file_uploader("ارفع ملف صوتي", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    file_path = "temp_audio_file"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("تحليل سريع جداً"):
        with st.spinner("جاري المعالجة على سيرفرات OpenAI..."):
            try:
                # استخدام Whisper API (سريع جداً ودقيق جداً)
                audio_file = open(file_path, "rb")
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    language="ar"
                )
                
                st.write("النتيجة:")
                st.text_area("النص:", transcript.text, height=300)
                
                # تصدير للـ Word
                doc = Document()
                doc.add_paragraph(transcript.text)
                doc.save("result.docx")
                st.download_button("تحميل Word", open("result.docx", "rb"), "result.docx")
                
                os.remove(file_path)
            except Exception as e:
                st.error(f"خطأ: {e}")
