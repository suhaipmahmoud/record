import streamlit as st
import pandas as pd
from faster_whisper import WhisperModel
import os

st.title("🎙️ Free Audio Transcriber")

uploaded_file = st.file_uploader("ارفع ملف صوتي", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    file_path = "temp_audio_file"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("تحليل مجاني"):
        with st.spinner("جاري المعالجة..."):
            try:
                # استخدام نموذج صغير وخفيف جداً ليعمل في السيرفر المجاني
                model = WhisperModel("tiny", device="cpu", compute_type="int8")
                segments, _ = model.transcribe(file_path, language="ar")
                
                results = [{"Text": s.text} for s in segments]
                df = pd.DataFrame(results)
                st.table(df)
                
                os.remove(file_path)
            except Exception as e:
                st.error(f"خطأ: {e}")
