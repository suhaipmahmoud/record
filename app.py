import streamlit as st
import pandas as pd
from docx import Document
from faster_whisper import WhisperModel
import os

st.title("🎙️ High Accuracy Transcriber")

uploaded_file = st.file_uploader("ارفع ملف صوتي", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    file_path = "temp_audio_file"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("بدء التحليل بدقة عالية"):
        with st.spinner("جاري المعالجة... (هذا النموذج يتطلب وقتاً أكبر للدقة)"):
            try:
                # استخدام نموذج 'medium' للحصول على أفضل دقة ممكنة
                # beam_size=5 تعني أن النموذج سيقارن بين عدة احتمالات للجملة ليختار الأصح
                model = WhisperModel("medium", device="cpu", compute_type="int8")
                
                segments, _ = model.transcribe(file_path, language="ar", beam_size=5)
                
                results = [{"Start": round(s.start, 2), "Text": s.text} for s in segments]
                
                df = pd.DataFrame(results)
                st.table(df)
                
                # تصدير
                doc = Document()
                for _, row in df.iterrows():
                    doc.add_paragraph(f"{row['Start']}s: {row['Text']}")
                doc.save("result.docx")
                st.download_button("تحميل النص", open("result.docx", "rb"), "result.docx")
                
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
            except Exception as e:
                st.error(f"حدث خطأ: {e}")
