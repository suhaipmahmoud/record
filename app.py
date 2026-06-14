import streamlit as st
import pandas as pd
from docx import Document
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import os

st.title("🎙️ Audio Processor (Stable)")

# التأكد من التوكن
if "HF_TOKEN" not in st.secrets:
    st.error("يرجى إضافة HF_TOKEN في إعدادات Secrets")
    st.stop()

HF_TOKEN = st.secrets["HF_TOKEN"]

uploaded_file = st.file_uploader("ارفع ملف صوتي", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    # حفظ الملف مؤقتاً
    file_path = "temp_audio"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("تحليل"):
        with st.spinner("جاري المعالجة..."):
            try:
                # 1. تحويل الصوت لنص (يعمل مباشرة مع mp3/wav)
                # استخدمنا model='tiny' لسرعة أعلى وتوافق أفضل مع الذاكرة
                model = WhisperModel("tiny", device="cpu", compute_type="int8")
                segments, _ = model.transcribe(file_path, language="ar")
                
                # 2. تحديد المتحدثين
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
                diarization = pipeline(file_path)
                
                results = []
                for segment in segments:
                    speaker = "Unknown"
                    for turn, _, spk in diarization.itertracks(yield_label=True):
                        # نعتمد على تداخل الوقت
                        if turn.start <= segment.start and turn.end >= segment.end:
                            speaker = spk
                            break
                    results.append({"Speaker": speaker, "Start": round(segment.start, 2), "Text": segment.text})
                
                df = pd.DataFrame(results)
                st.table(df)
                
                # تصدير النتائج
                df.to_excel("result.xlsx", index=False)
                st.download_button("تصدير Excel", open("result.xlsx", "rb"), "result.xlsx")
                
                # تنظيف الملفات
                os.remove(file_path)
                
            except Exception as e:
                st.error(f"خطأ في المعالجة: {e}")
