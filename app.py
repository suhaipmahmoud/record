import streamlit as st
import pandas as pd
from docx import Document
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import os
import librosa
import soundfile as sf

st.title("🎙️ Audio Converter Pro")

# التأكد من وجود التوكن
if "HF_TOKEN" not in st.secrets:
    st.error("يرجى إضافة HF_TOKEN في إعدادات Streamlit Secrets")
    st.stop()

HF_TOKEN = st.secrets["HF_TOKEN"]

def load_audio(uploaded_file):
    """تحميل الملف الصوتي وتحويله لصيغة تدعمها المعالجة"""
    temp_path = "input_audio.wav"
    # قراءة الملف باستخدام librosa (يدعم mp3, wav, m4a إلخ)
    y, sr = librosa.load(uploaded_file, sr=16000)
    # حفظه كملف WAV موحد
    sf.write(temp_path, y, sr)
    return temp_path

uploaded_file = st.file_uploader("ارفع ملف صوتي", type=["wav", "mp3", "m4a", "ogg"])

if uploaded_file is not None:
    if st.button("بدء التحليل"):
        with st.spinner("جاري المعالجة..."):
            try:
                # 1. تحضير الملف
                file_path = load_audio(uploaded_file)
                
                # 2. تحديد المتحدثين
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
                diarization = pipeline(file_path)
                
                # 3. تحويل الصوت لنص
                model = WhisperModel("small", device="cpu", compute_type="int8")
                segments, _ = model.transcribe(file_path, language="ar")
                
                results = []
                for segment in segments:
                    speaker = "Unknown"
                    for turn, _, spk in diarization.itertracks(yield_label=True):
                        if turn.start <= segment.start and turn.end >= segment.end:
                            speaker = spk
                            break
                    results.append({"Speaker": speaker, "Start": round(segment.start, 2), "Text": segment.text})
                
                df = pd.DataFrame(results)
                st.table(df)

                # التصدير
                df.to_excel("result.xlsx", index=False)
                st.download_button("تصدير Excel", open("result.xlsx", "rb"), "result.xlsx")
                
                doc = Document()
                for _, row in df.iterrows():
                    doc.add_paragraph(f"{row['Speaker']}: {row['Text']}")
                doc.save("result.docx")
                st.download_button("تصدير Word", open("result.docx", "rb"), "result.docx")
                
            except Exception as e:
                st.error(f"حدث خطأ أثناء المعالجة: {e}")
