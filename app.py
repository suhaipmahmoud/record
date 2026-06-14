import streamlit as st
import pandas as pd
from docx import Document
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import os
import ffmpeg

# إعداد واجهة البرنامج
st.title("🎙️ Audio Converter Pro - Developed by Suhaib")

# استدعاء التوكن بأمان
HF_TOKEN = st.secrets["HF_TOKEN"]

def convert_to_wav(input_file):
    """تحويل أي ملف صوتي إلى WAV باستخدام ffmpeg-python"""
    output_file = "temp_audio.wav"
    try:
        # تحويل الصوت لتردد 16kHz أحادي القناة (مثالي لـ Whisper)
        ffmpeg.input(input_file).output(output_file, acodec='pcm_s16le', ac=1, ar='16000').run(overwrite_output=True)
        return output_file
    except Exception as e:
        st.error(f"خطأ في تحويل الصوت: {e}")
        return None

def get_diarization(audio_path):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
    return pipeline(audio_path)

uploaded_file = st.file_uploader("ارفع ملف صوتي (MP3, WAV, M4A, OGG)", type=["wav", "mp3", "m4a", "ogg"])

if uploaded_file is not None:
    # حفظ الملف المرفوع مؤقتاً
    with open("temp_upload", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # تحويله
    file_path = convert_to_wav("temp_upload")
    
    if file_path and st.button("بدء التحليل"):
        with st.spinner("جاري المعالجة... (قد يستغرق وقتاً)"):
            # 1. تحديد المتحدثين
            diarization = get_diarization(file_path)
            
            # 2. تحويل الصوت لنص
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
            
            # تنظيف
            if os.path.exists("temp_audio.wav"): os.remove("temp_audio.wav")
            if os.path.exists("temp_upload"): os.remove("temp_upload")
