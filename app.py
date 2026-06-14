from docx import Document
from faster_whisper import WhisperModel
import os
import librosa
import soundfile as sf

st.title("🎙️ Audio Converter - Stable Version")
st.title("🎙️ Audio Transcriber - Clean Version")

uploaded_file = st.file_uploader("ارفع ملف صوتي", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    # حفظ الملف مؤقتاً
    input_path = "temp_input"
    wav_path = "temp_output.wav"
    
    with open(input_path, "wb") as f:
    file_path = "temp_audio_file"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("بدء التحويل إلى نص"):
    if st.button("بدء التحليل"):
        with st.spinner("جاري المعالجة..."):
            try:
                # تحويل الصوت باستخدام librosa و soundfile (بدون مكتبات نظام معقدة)
                y, sr = librosa.load(input_path, sr=16000)
                sf.write(wav_path, y, sr)
                # استخدام النموذج 'small' ليكون متوازناً بين السرعة والدقة
                # لا يحتاج لـ torchudio أو librosa
                model = WhisperModel("small", device="cpu", compute_type="int8")

                # معالجة النص
                model = WhisperModel("tiny", device="cpu", compute_type="int8")
                segments, _ = model.transcribe(wav_path, language="ar")
                # المعالجة
                segments, _ = model.transcribe(file_path, language="ar", beam_size=5)

                results = [{"Start": round(s.start, 2), "Text": s.text} for s in segments]

                df = pd.DataFrame(results)
                st.table(df)

                # تصدير
                # تصدير النتائج
                doc = Document()
                for _, row in df.iterrows():
                    doc.add_paragraph(f"{row['Start']}s: {row['Text']}")
                doc.save("result.docx")
                st.download_button("تحميل النص", open("result.docx", "rb"), "result.docx")

                # تنظيف
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
            except Exception as e:
                st.error(f"حدث خطأ: {e}")
                st.error(f"خطأ: {e}")
