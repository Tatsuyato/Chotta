import gradio as gr
import subprocess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ซ่อนแจ้งเตือน Warning สีแดงจาก TensorFlow/JAX
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
import sys
import json
import time
import torch
import gc
import scipy.io.wavfile
import cv2
import shutil
import soundfile as sf
import re
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    import yt_dlp
except ImportError:
    pass

# Try to import F5-TTS-TH
try:
    from f5_tts_th.tts import TTS
    F5_AVAILABLE = True
except ImportError:
    F5_AVAILABLE = False
    print("F5-TTS-TH is not installed. Please install it using: pip install f5-tts-th")

# --- 1. Helper Functions & Cleanup ---
def format_time(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def format_ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def time_to_sec(time_str):
    try:
        parts = str(time_str).split(':')
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except:
        return 0.0

def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Failed to remove {path}: {e}")

def get_drive_videos():
    drive_path = "/content/drive/MyDrive"
    if not os.path.exists(drive_path):
        return []
    
    video_exts = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
    videos = []
    
    try:
        for root, dirs, files in os.walk(drive_path):
            depth = root[len(drive_path):].count(os.sep)
            if depth > 1: # สแกนแค่ MyDrive และโฟลเดอร์ย่อย 1 ชั้น เพื่อไม่ให้ระบบโหลดนานเกินไป
                dirs.clear()
            for f in files:
                if f.lower().endswith(video_exts):
                    videos.append(os.path.join(root, f))
    except Exception:
        pass
    
    return sorted(videos)

def mount_google_drive():
    try:
        # ตรวจสอบก่อนว่ามีการเมานท์ Drive ไว้แล้วหรือไม่
        if os.path.exists('/content/drive'):
            os.makedirs('/content/drive/MyDrive/AI_Videos', exist_ok=True)
            return "✅ Google Drive ถูกเชื่อมต่อไว้แล้ว! วิดีโอจะถูกบันทึกที่โฟลเดอร์ AI_Videos", gr.update(choices=get_drive_videos())
            
        from google.colab import drive
        drive.mount('/content/drive')
        os.makedirs('/content/drive/MyDrive/AI_Videos', exist_ok=True)
        return "✅ เชื่อมต่อ Google Drive สำเร็จ! วิดีโอจะถูกบันทึกที่โฟลเดอร์ AI_Videos", gr.update(choices=get_drive_videos())
    except ImportError:
        return "⚠️ ไม่สามารถเชื่อมต่อ Google Drive ได้ (ฟีเจอร์นี้ใช้ได้บน Google Colab เท่านั้น)", gr.update()
    except Exception as e:
        if "'NoneType' object has no attribute 'kernel'" in str(e):
            return "⚠️ กรุณาเมานท์ Google Drive จากเซลล์ในสมุดโน้ต (Notebook) โดยตรงก่อน (ไม่สามารถเมานท์ผ่านปุ่มนี้ได้เมื่อรันด้วยคำสั่ง python app.py)", gr.update()
        return f"⚠️ เกิดข้อผิดพลาด: {str(e)}", gr.update()

# --- 2. Smart Auto-Crop (Face Tracking) ---
def get_smart_crop_center(video_path, target_ratio=9/16, progress=gr.Progress()):
    try:
        import mediapipe as mp
    except ImportError:
        print("MediaPipe not installed, falling back to center crop.")
        return None

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    face_centers_x = []
    
    progress(0, desc="🔎 สแกนหาใบหน้าในวิดีโอ (Smart Crop)...")
    
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        frame_count = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            
            # สุ่มเช็คทุกๆ 15 เฟรมเพื่อความรวดเร็ว
            if frame_count % 15 == 0:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image_rgb)
                
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        x_center = bboxC.xmin + bboxC.width / 2
                        face_centers_x.append(x_center * width)
                        
            if frame_count % 30 == 0:
                progress(frame_count / total_frames, desc=f"🔎 สแกนใบหน้าเฟรมที่ {frame_count}/{total_frames}...")
                
            frame_count += 1
            
    cap.release()
    
    if face_centers_x:
        avg_x = sum(face_centers_x) / len(face_centers_x)
    else:
        avg_x = width / 2  # หากไม่เจอใบหน้า ให้ตัดตรงกลาง
        
    crop_w = int(height * target_ratio)
    crop_h = height
    crop_x = int(avg_x - crop_w / 2)
    crop_y = 0
    
    # ป้องกันกรอบทะลุขอบวิดีโอ
    if crop_x < 0: crop_x = 0
    if crop_x + crop_w > width: crop_x = width - crop_w
    
    return crop_w, crop_h, crop_x, crop_y

# --- 3. Subtitles Generation (Customizable) ---
def generate_ass_subtitles(audio_path, font_color="Yellow", font_size=85, outline_size=5, progress=gr.Progress()):
    progress(0, desc="📝 สร้างซับไตเติ้ลคำต่อคำ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    whisper_model = WhisperModel("small", device=device, compute_type=compute_type)
    
    segments, info = whisper_model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    # เปิด vad_filter เพื่อข้ามช่วงเงียบ เร่งความเร็วการทำซับไตเติ้ล
    segments, info = whisper_model.transcribe(audio_path, beam_size=5, word_timestamps=True, vad_filter=True)
    total_duration = info.duration
    
    # กำหนดรหัสสี BGR (ASS format is AABBGGRR but usually just BBGGRR in V4+)
    # Yellow = 00FFFF, White = FFFFFF, Green = 00FF00
    color_codes = {
        "Yellow": "&H0000FFFF",
        "White": "&H00FFFFFF",
        "Green": "&H0000FF00"
    }
    primary_color = color_codes.get(font_color, "&H0000FFFF")
    
    ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TikTok,Arial,{font_size},{primary_color},&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline_size},3,2,20,20,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    for i, segment in enumerate(segments):
        start = format_ass_time(segment.start)
        end = format_ass_time(segment.end)
        text = segment.text.strip()
        ass_content += f"Dialogue: 0,{start},{end},TikTok,,0,0,0,,{text}\n"
        
        percent = segment.end / total_duration if total_duration > 0 else 0
        progress(percent, desc=f"📝 สร้างซับไตเติ้ลคำต่อคำ... {format_time(segment.end)} / {format_time(total_duration)}")
        
    del whisper_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    subs_path = f"temp_subs_{int(time.time())}.ass"
    with open(subs_path, "w", encoding="utf-8") as f:
        f.write(ass_content)
    return subs_path


# --- 4. Advanced Analysis (Chunking) ---
def analyze_video_chunked(upload_method, video_path, drive_path, url_input, isolate_vocal, progress=gr.Progress()):
    actual_video_path = None
    if upload_method == "วางลิงก์จากเว็บ (URL)" and url_input and url_input.strip():
        progress(0, desc="🌐 กำลังดาวน์โหลดวิดีโอจากลิงก์...")
        
        last_update = [0]
        def yt_dlp_hook(d):
            if d['status'] == 'downloading':
                now = time.time()
                if now - last_update[0] > 1.0: # อัปเดต UI ทุกๆ 1 วินาทีเพื่อไม่ให้ Gradio ค้าง
                    progress(0, desc=f"🌐 กำลังดาวน์โหลดวิดีโอ... {d.get('_percent_str', '').strip()} ({d.get('_speed_str', '').strip()})")
                    last_update[0] = now
                    
        output_filename = f"dl_video_{int(time.time())}.mp4"
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 
            'outtmpl': output_filename, 
            'quiet': True,
            'progress_hooks': [yt_dlp_hook]
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url_input.strip()])
            actual_video_path = output_filename
        except Exception as e:
            return f"เกิดข้อผิดพลาดในการดาวน์โหลดวิดีโอ: {str(e)}", gr.update(choices=[], visible=False), {}, {}, ""
    elif upload_method == "เลือกจาก Google Drive":
        actual_video_path = drive_path.strip() if drive_path and drive_path.strip() else None
    else:
        actual_video_path = video_path
        
    if not actual_video_path or not os.path.exists(actual_video_path):
        return "กรุณาอัปโหลดวิดีโอ, ระบุพาธไฟล์ หรือใส่ลิงก์ให้ถูกต้องตามช่องทางที่เลือก", gr.update(choices=[], visible=False), {}, {}, ""
    
    audio_path = f"temp_audio_{int(time.time())}.wav"
    original_audio_path = audio_path
    
    try:
        progress(0.05, desc="🎵 1/3 กำลังสกัดเสียงจากวิดีโอ...")
        subprocess.run(["ffmpeg", "-y", "-i", actual_video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if isolate_vocal:
            progress(0.1, desc="🧹 กำลังสกัดเฉพาะเสียงพูด (ลบดนตรี/เสียงรบกวนด้วย AI)... อาจใช้เวลาสักครู่")
            try:
                subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", audio_path], check=True)
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
                clean_audio_path = os.path.join("separated", "htdemucs", base_name, "vocals.wav")
                if os.path.exists(clean_audio_path):
                    audio_path = clean_audio_path
            except Exception as demucs_e:
                print(f"Demucs failed: {demucs_e}")
                
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        
        progress(0.15, desc="🗣️ กำลังโหลดโมเดลถอดเสียง...")
        whisper_model = WhisperModel("small", device=device, compute_type=compute_type)
        
        progress(0.2, desc="📝 กำลังถอดเสียง (Transcription) อาจใช้เวลานานสำหรับคลิป 1 ชม...")
        segments_generator, info = whisper_model.transcribe(audio_path, beam_size=5)
        progress(0.2, desc="📝 กำลังถอดเสียง (ใช้ VAD เร่งความเร็วเต็มประสิทธิภาพ)...")
        # ลด beam_size ลงและเปิด VAD พร้อมปิด condition_on_previous_text เพื่อรีดความเร็วสูงสุดในขั้นวิเคราะห์
        segments_generator, info = whisper_model.transcribe(audio_path, beam_size=2, vad_filter=True, condition_on_previous_text=False)
        
        transcript = ""
        segments = []
        valid_ref_segments = []
        total_duration = info.duration
        
        for segment in segments_generator:
            transcript += f"[{format_time(segment.start)} - {format_time(segment.end)}] {segment.text}\n"
            segments.append(segment)
            
            # อัปเดตสถานะให้ผู้ใช้เห็นแบบเรียลไทม์ (วินาทีต่อวินาที)
            percent = segment.end / total_duration if total_duration > 0 else 0
            progress(0.2 + (0.2 * percent), desc=f"📝 กำลังถอดเสียง... {format_time(segment.end)} / {format_time(total_duration)}")
            
            # เก็บตัวเลือกเสียงอ้างอิง (4-12 วินาที) โดยเว้นระยะห่างกันอย่างน้อย 30 วินาที เพื่อให้ได้เสียงตัวละครที่หลากหลาย
            duration = segment.end - segment.start
            if 4 <= duration <= 12 and len(segment.text.strip()) > 10:
                if not valid_ref_segments or (segment.start - valid_ref_segments[-1].start > 30):
                    if len(valid_ref_segments) < 10: # เก็บสูงสุด 10 ตัวเลือก
                        valid_ref_segments.append(segment)
                
        # หากไม่เจอช่วงที่ยาวพอ ให้เอาช่วงแรกมาเลย
        if not valid_ref_segments and segments:
            valid_ref_segments.append(segments[0])
            
        ref_candidates = {}
        choices_ref = []
        
        if valid_ref_segments:
            progress(0.4, desc="🎙️ กำลังบันทึกตัวเลือกเสียงอ้างอิงหลายรายการ (Auto-Reference Candidates)...")
            for i, seg in enumerate(valid_ref_segments):
                ref_path = f"auto_ref_{int(time.time())}_{i}.wav"
                subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ss", str(seg.start), "-to", str(seg.end), "-c:a", "pcm_s16le", "-ar", "24000", ref_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                label = f"เสียงที่ {i+1} [{format_time(seg.start)}] {seg.text.strip()[:40]}..."
                ref_candidates[label] = {"path": ref_path, "text": seg.text.strip()}
                choices_ref.append(label)
            
        del whisper_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        progress(0.5, desc="🧠 2/3 โหลด Local LLM (Qwen) เพื่อวิเคราะห์เนื้อหา...")
        model_id = "Qwen/Qwen2.5-1.5B-Instruct" 
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        llm_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

        # Chunking System: แบ่งทีละ 12,000 ตัวอักษร (ประมาณ 10-15 นาที) เพื่อป้องกัน RAM ระเบิดและอ่านไม่จบ
        chunk_size = 12000
        chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]
        
        all_topics = []
        
        for idx, chunk in enumerate(chunks):
            progress(0.6 + (0.3 * (idx/len(chunks))), desc=f"🔎 กำลังวิเคราะห์คลิปส่วนที่ {idx+1}/{len(chunks)}...")
            prompt = f"""คุณคือครีเอเตอร์นักตัดต่อวิดีโอสั้นและผู้กำกับมือทองชาวไทย อ่านสคริปต์ต่อไปนี้และสร้าง "เนื้อเรื่อง" ที่น่าสนใจที่สุด 1 ถึง 2 เรื่องเพื่อทำเป็นคลิปสั้น (TikTok/Reels)
โดยคุณสามารถ "เลือกตัดเอาหลายๆ ช่วงเวลาที่เกี่ยวข้องกัน" (Clips) จากในสคริปต์มาประกอบร่างกันเป็นคลิปเดียวได้ เพื่อให้เนื้อหากระชับและน่าติดตาม
กรุณาตอบกลับเป็น JSON เท่านั้น โดยห้ามมีคำอธิบายอื่นใด รูปแบบตามนี้เป๊ะๆ:
{{
    "topics": [
        {{
            "title": "ตั้งชื่อคลิปให้น่าสนใจ (ภาษาไทย)", 
            "clips": [
                {{"start_time": "00:00:10", "end_time": "00:00:25", "voiceover_script": "บทพากย์สำหรับฉากที่ 1 (ภาษาไทย)"}},
                {{"start_time": "00:01:10", "end_time": "00:01:30", "voiceover_script": "บทพากย์สำหรับฉากที่ 2 (ภาษาไทย)"}}
            ]
        }}
    ]
}}
สคริปต์วิดีโอ:
{chunk}
"""
            messages = [{"role": "system", "content": "You are a helpful assistant. Output ONLY valid JSON in Thai."}, {"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
            
            outputs = llm_model.generate(**inputs, max_new_tokens=2048, temperature=0.7)
            response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            try:
                # Regex to extract JSON carefully
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)
                    all_topics.extend(data.get("topics", []))
            except Exception as parse_e:
                print(f"Failed to parse JSON in chunk {idx}: {parse_e}")
                continue

        del llm_model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        progress(1.0, desc="✅ 3/3 วิเคราะห์เสร็จสิ้น!")

        if not all_topics:
            raise ValueError("AI ไม่สามารถหาหัวข้อที่น่าสนใจในวิดีโอได้ (Invalid JSON or no topics)")

        choices = []
        topics_dict = {}
        for i, t in enumerate(all_topics):
            title = t.get('title', 'ไม่ระบุชื่อหัวข้อ')
            clips = t.get('clips', [])
            time_str = f"({len(clips)} ฉากประกอบกัน เริ่ม {clips[0].get('start_time', '00:00:00')})" if clips else f"({t.get('start_time', '00:00:00')} - {t.get('end_time', '00:00:00')})"
            
            key = f"[{i+1}] {title} {time_str}"
            choices.append(key)
            topics_dict[key] = t
        
        # ส่งคืนค่ากลับไปเก็บใน UI โดยอัปเดต Dropdown เลือกเสียง
        return f"วิเคราะห์เสร็จสิ้น พบ {len(all_topics)} หัวข้อตลอดความยาวคลิป!", gr.update(choices=choices, visible=True), topics_dict, gr.update(choices=choices_ref, value=choices_ref[0] if choices_ref else None, visible=True), ref_candidates, actual_video_path

    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}", gr.update(choices=[], visible=False), {}, gr.update(choices=[], visible=False), {}, ""
    finally:
        safe_remove(original_audio_path)
        try:
            if os.path.exists("separated"):
                shutil.rmtree("separated")
        except:
            pass

# --- 5. Test Voice Cloning ---
def test_voice_clone(selected_auto_ref, ref_candidates_dict, custom_ref_audio, custom_ref_text, test_script, isolate_custom_ref, progress=gr.Progress()):
    auto_ref = ref_candidates_dict.get(selected_auto_ref, {}) if ref_candidates_dict and selected_auto_ref else {}
    actual_ref_audio = custom_ref_audio if custom_ref_audio else auto_ref.get("path", "")
    actual_ref_text = custom_ref_text if custom_ref_text else auto_ref.get("text", "")
    
    if not actual_ref_audio or not actual_ref_text:
        return None, "⚠️ กรุณาอัปโหลดเสียงต้นแบบและใส่ข้อความที่พูดในเสียงต้นแบบ หรือทำการวิเคราะห์วิดีโอใน Tab 1 ก่อน"
        
    if isolate_custom_ref and actual_ref_audio:
        progress(0.2, desc="🧹 กำลังสกัดเฉพาะเสียงพูดจากเสียงต้นแบบ (Demucs)...")
        try:
            subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", actual_ref_audio], check=True)
            base_name = os.path.splitext(os.path.basename(actual_ref_audio))[0]
            clean_audio_path = os.path.join("separated", "htdemucs", base_name, "vocals.wav")
            if os.path.exists(clean_audio_path):
                actual_ref_audio = clean_audio_path
        except Exception as demucs_e:
            print(f"Demucs failed on custom ref: {demucs_e}")
            
    if not test_script or not test_script.strip():
        return None, "⚠️ กรุณาใส่ข้อความสำหรับทดสอบเสียง"
        
    audio_path = f"test_voice_{int(time.time())}.wav"
    
    try:
        if F5_AVAILABLE:
            progress(0.5, desc="🎙️ กำลังทดสอบโคลนเสียง (F5-TTS)...")
            tts = TTS(model="v2")
            wav = tts.infer(
                ref_audio=actual_ref_audio,
                ref_text=actual_ref_text,
                gen_text=test_script,
                step=32,
                cfg=2.0,
                speed=1.0
            )
            sf.write(audio_path, wav, 24000)
            del tts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            progress(0.5, desc="🎙️ กำลังสร้างเสียงทดสอบ (MMS-TTS)...")
            from transformers import VitsModel, AutoProcessor
            processor = AutoProcessor.from_pretrained("facebook/mms-tts-tha")
            tts_model = VitsModel.from_pretrained("facebook/mms-tts-tha")
            inputs = processor(text=test_script, return_tensors="pt")
            with torch.no_grad():
                audio_output = tts_model(**inputs).waveform
            scipy.io.wavfile.write(audio_path, rate=tts_model.config.sampling_rate, data=audio_output[0].numpy())
            del tts_model
            del processor
            gc.collect()
            
        return audio_path, "✅ ทดสอบเสียงสำเร็จ! คุณสามารถกดฟังได้เลย"
    except Exception as e:
        return None, f"⚠️ เกิดข้อผิดพลาด: {str(e)}"

# --- 6. Main Process Video (With Garbage Collection & Progress) ---
def process_video_local(video_path, selected_topic_key, topics_dict, orientation, vertical_mode, resolution, bgm_path, bgm_vol, font_color, font_size, save_to_drive, selected_auto_ref, ref_candidates_dict, custom_ref_audio, custom_ref_text, isolate_custom_ref, watermark_path, b_roll_path, progress=gr.Progress()):
    if not video_path or not os.path.exists(video_path):
        return None, "ไม่พบไฟล์วิดีโอต้นฉบับ กรุณากลับไปที่ Tab 1 เพื่ออัปโหลดและวิเคราะห์วิดีโอใหม่อีกครั้ง"
        
    if not selected_topic_key or not topics_dict:
        return None, "กรุณาวิเคราะห์วิดีโอและเลือกหัวข้อก่อน"
    
    temp_files = [] # Track files for garbage collection
    
    try:
        topic_info = topics_dict[selected_topic_key]
        
        clips = topic_info.get('clips', [])
        if not clips and 'start_time' in topic_info:
            clips = [{'start_time': topic_info['start_time'], 'end_time': topic_info['end_time']}]
            
        if not clips:
            raise ValueError("ไม่พบข้อมูลเวลาเริ่มต้นและสิ้นสุดของวิดีโอ (Invalid Topic Data)")
            
        # รวบรวมบทพากย์จากแต่ละฉากมาต่อกัน หากใช้รูปแบบใหม่
        script_parts = [c.get('voiceover_script', '').strip() for c in clips if c.get('voiceover_script')]
        if script_parts:
            script = " ".join(script_parts)
        else:
            script = topic_info.get('voiceover_script', '')
        
        audio_path = f"temp_voice_{int(time.time())}.wav"
        temp_files.append(audio_path)
        
        # 1. สร้างเสียงพากย์ด้วย F5-TTS-TH-V2
        if F5_AVAILABLE:
            progress(0.1, desc="🎙️ 1/4 กำลังโคลนเสียงพากย์ (F5-TTS Voice Cloning)...")
            
            auto_ref = ref_candidates_dict.get(selected_auto_ref, {}) if ref_candidates_dict and selected_auto_ref else {}
            actual_ref_audio = custom_ref_audio if custom_ref_audio else auto_ref.get("path", "")
            actual_ref_text = custom_ref_text if custom_ref_text else auto_ref.get("text", "")
            
            if not actual_ref_audio or not actual_ref_text:
                raise ValueError("ไม่พบเสียงอ้างอิง (Reference Audio) กรุณาอัปโหลดเสียงต้นแบบ")
                
            if isolate_custom_ref:
                progress(0.15, desc="🧹 กำลังสกัดเฉพาะเสียงพูดจากเสียงต้นแบบ (Demucs)...")
                try:
                    subprocess.run([sys.executable, "-m", "demucs", "--two-stems=vocals", actual_ref_audio], check=True)
                    base_name = os.path.splitext(os.path.basename(actual_ref_audio))[0]
                    clean_audio_path = os.path.join("separated", "htdemucs", base_name, "vocals.wav")
                    if os.path.exists(clean_audio_path):
                        actual_ref_audio = clean_audio_path
                except Exception as demucs_e:
                    print(f"Demucs failed on custom ref: {demucs_e}")
                    
            tts = TTS(model="v2")
            wav = tts.infer(
                ref_audio=actual_ref_audio,
                ref_text=actual_ref_text,
                gen_text=script,
                step=32,
                cfg=2.0,
                speed=1.0
            )
            sf.write(audio_path, wav, 24000)
            
            del tts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            progress(0.1, desc="🎙️ 1/4 กำลังสร้างเสียงพากย์สำรอง (MMS-TTS-Thai)...")
            from transformers import VitsModel, AutoProcessor
            processor = AutoProcessor.from_pretrained("facebook/mms-tts-tha")
            tts_model = VitsModel.from_pretrained("facebook/mms-tts-tha")
            inputs = processor(text=script, return_tensors="pt")
            with torch.no_grad():
                audio_output = tts_model(**inputs).waveform
            scipy.io.wavfile.write(audio_path, rate=tts_model.config.sampling_rate, data=audio_output[0].numpy())
            del tts_model
            del processor
            gc.collect()

        # 2. สร้าง Subtitles (.ass) จากเสียงพากย์
        progress(0.3, desc="📝 2/4 กำลังสร้าง Subtitle ตามสไตล์ที่ตั้งค่า...")
        subs_path = generate_ass_subtitles(audio_path, font_color=font_color, font_size=font_size, progress=progress)
        temp_files.append(subs_path)

        # 3. ตัดและปรับสัดส่วนวิดีโอ (Auto-Crop / Blur Background)
        progress(0.5, desc="✂️ 3/4 กำลังตัดประกอบวิดีโอ (Video Assembly) และปรับสัดส่วน...")
        temp_subclip = f"temp_sub_{int(time.time())}.mp4"
        temp_files.append(temp_subclip)
        
        if len(clips) == 1:
            st = clips[0]['start_time']
            et = clips[0]['end_time']
            subprocess.run(["ffmpeg", "-y", "-i", video_path, "-ss", str(st), "-to", str(et), "-c:v", "copy", "-c:a", "copy", temp_subclip], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            filter_complex_concat = ""
            concat_inputs = ""
            for idx, clip in enumerate(clips):
                st_sec = time_to_sec(clip['start_time'])
                et_sec = time_to_sec(clip['end_time'])
                filter_complex_concat += f"[0:v]trim=start={st_sec}:end={et_sec},setpts=PTS-STARTPTS[v{idx}];"
                filter_complex_concat += f"[0:a]atrim=start={st_sec}:end={et_sec},asetpts=PTS-STARTPTS[a{idx}];"
                concat_inputs += f"[v{idx}][a{idx}]"
                
            filter_complex_concat += f"{concat_inputs}concat=n={len(clips)}:v=1:a=1[v_out][a_out]"
            
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-filter_complex", filter_complex_concat,
                "-map", "[v_out]", "-map", "[a_out]",
                "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", temp_subclip
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if orientation == "Vertical (9:16)":
            if vertical_mode == "Blur Background (ขอบเบลอ)":
                target_w = 1080 if resolution == "1080p" else (720 if resolution == "720p" else 1080)
                target_h = 1920 if resolution == "1080p" else (1280 if resolution == "720p" else 1920)
                video_filters_prefix = f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},boxblur=20:20[bg];[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v_base]"
            else:
                crop_params = get_smart_crop_center(temp_subclip, target_ratio=9/16, progress=progress)
                if crop_params:
                    cw, ch, cx, cy = crop_params
                    video_filter = f"crop={cw}:{ch}:{cx}:{cy}"
                else:
                    video_filter = "crop=ih*(9/16):ih"
                    
                if resolution == "1080p":
                    video_filter += ",scale=1080:1920"
                elif resolution == "720p":
                    video_filter += ",scale=720:1280"
                video_filters_prefix = f"[0:v]{video_filter}[v_base]"
        else:
            video_filter = "crop=iw:iw*(9/16)"
            if resolution == "1080p":
                video_filter += ",scale=1920:1080"
            elif resolution == "720p":
                video_filter += ",scale=1280:720"
            video_filters_prefix = f"[0:v]{video_filter}[v_base]"

        # 4. รวมภาพ, เสียงพากย์, Subtitles และ BGM
        progress(0.8, desc="🎞️ 4/4 กำลังเรนเดอร์และผสานทุกองค์ประกอบ...")
        final_output = f"final_output_{int(time.time())}.mp4"
        
        # Convert bgm_vol percentage to float (e.g., 10 -> 0.1)
        vol_float = bgm_vol / 100.0 if bgm_vol else 0.1
        
        cmd_inputs = ["-i", temp_subclip, "-i", audio_path]
        audio_filter = ""
        audio_map = "1:a"
        
        bgm_idx = None
        wm_idx = None
        broll_idx = None
        curr_idx = 2
        
        if bgm_path:
            cmd_inputs.extend(["-i", bgm_path])
            bgm_idx = curr_idx
            curr_idx += 1
            audio_filter = f";[1:a]volume=1.0[a1];[{bgm_idx}:a]volume={vol_float}[a2];[a1][a2]amix=inputs=2:duration=first[a]"
            audio_map = "[a]"
            
        if watermark_path:
            cmd_inputs.extend(["-i", watermark_path])
            wm_idx = curr_idx
            curr_idx += 1
            
        if b_roll_path:
            cmd_inputs.extend(["-i", b_roll_path])
            broll_idx = curr_idx
            curr_idx += 1
            
        video_filters = video_filters_prefix
        current_v = "[v_base]"
        
        if b_roll_path:
            if "Vertical" in orientation:
                b_w = 800 if resolution == "1080p" else 540
            else:
                b_w = 1400 if resolution == "1080p" else 960
            video_filters += f";[{broll_idx}:v]scale={b_w}:-2[broll];{current_v}[broll]overlay=(W-w)/2:(H-h)/2:eof_action=pass[v_broll]"
            current_v = "[v_broll]"
            
        video_filters += f";{current_v}ass={subs_path}[v_sub]"
        current_v = "[v_sub]"
        
        if watermark_path:
            video_filters += f";[{wm_idx}:v]scale=150:-2[wm];{current_v}[wm]overlay=W-w-20:20[v_out]"
            current_v = "[v_out]"
            
        filter_complex = video_filters + audio_filter
        
        cmd = ["ffmpeg", "-y"] + cmd_inputs + [
            "-filter_complex", filter_complex,
            "-map", current_v, "-map", audio_map,
            "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", final_output
        ]
            
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg Error: {result.stderr.decode('utf-8')}")

        # 5. Save to Google Drive
        progress(0.95, desc="💾 กำลังบันทึกไฟล์...")
        status_msg = "✨ สร้างวิดีโอเสร็จสมบูรณ์!"
        if save_to_drive:
            try:
                drive_path = f"/content/drive/MyDrive/AI_Videos/{os.path.basename(final_output)}"
                shutil.copy(final_output, drive_path)
                status_msg += f"\n📁 บันทึกลง Google Drive แล้ว: {drive_path}"
            except Exception as e:
                status_msg += f"\n⚠️ บันทึกลง Google Drive ไม่สำเร็จ (อาจยังไม่ได้เชื่อมต่อ)"

        progress(1.0, desc="✅ เสร็จสมบูรณ์")
        return final_output, status_msg

    except Exception as e:
        return None, f"เกิดข้อผิดพลาดในการสร้างวิดีโอ: {str(e)}"
    finally:
        # Garbage Collection - Clean up all temporary files safely
        for file in temp_files:
            safe_remove(file)

# --- 7. Gallery Management ---
def get_generated_videos():
    videos = []
    for f in os.listdir('.'):
        if f.startswith('final_output_') and f.endswith('.mp4'):
            videos.append(os.path.abspath(f))
            
    drive_path = '/content/drive/MyDrive/AI_Videos'
    if os.path.exists(drive_path):
        for f in os.listdir(drive_path):
            if f.endswith('.mp4'):
                videos.append(os.path.join(drive_path, f))
                
    videos.sort(key=os.path.getmtime, reverse=True)
    return videos

def update_gallery():
    videos = get_generated_videos()
    choices = [(os.path.basename(v), v) for v in videos]
    return gr.update(choices=choices, value=videos[0] if videos else None)

def load_video_preview(video_path):
    if not video_path or not os.path.exists(video_path):
        return None
    return video_path

def delete_video(video_path):
    if not video_path or not os.path.exists(video_path):
        return "ไฟล์ไม่พบหรือไม่สามารถลบได้", update_gallery()
    try:
        os.remove(video_path)
        return f"🗑️ ลบไฟล์ {os.path.basename(video_path)} สำเร็จ", update_gallery()
    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}", gr.update()

# --- Gradio UI ---
with gr.Blocks(title="The Ultimate Pro AI Video Agent") as app:
    gr.Markdown("# 🎬 The Ultimate Pro AI Video Agent (F5-TTS 🇹🇭)")
    gr.Markdown("🌟 **[New!]** อัปเกรดระบบ **AI อ่านคลิปได้จบคลิป (1 ชม+)** มี **Progress Bar แสดงสถานะเรียลไทม์**, **ล้างไฟล์ขยะออโต้**, และตั้งค่า **สี/ขนาดซับไตเติ้ล** ได้แล้ว!")
    
    with gr.Row():
        drive_btn = gr.Button("🔗 เชื่อมต่อ Google Drive (เพื่อบันทึกไฟล์อัตโนมัติ)", variant="secondary")
        drive_status = gr.Textbox(label="สถานะ Google Drive", interactive=False)
        drive_btn.click(fn=mount_google_drive, outputs=[drive_status])

    with gr.Tab("1. Analyze Video (วิเคราะห์และดึงเสียงต้นแบบ)"):
        upload_method = gr.Radio(
            label="เลือกวิธีการนำเข้าวิดีโอ", 
            choices=["อัปโหลดจากเครื่อง (Local)", "เลือกจาก Google Drive", "วางลิงก์จากเว็บ (URL)"], 
            value="อัปโหลดจากเครื่อง (Local)"
        )
        gr.Markdown("💡 **ทริคสำหรับไฟล์ขนาดใหญ่ (1 ชม.+)**: แนะนำให้เลือกช่องทาง **\"เลือกจาก Google Drive\"** จะรวดเร็วและเสถียรกว่ามาก!")
        
        video_input = gr.Video(label="อัปโหลดวิดีโอจากเครื่อง", visible=True)
        
        with gr.Row(visible=False) as drive_group:
            drive_path_input = gr.Dropdown(label="📂 เลือกไฟล์จาก Google Drive", choices=get_drive_videos(), allow_custom_value=True, scale=4)
            refresh_drive_btn = gr.Button("🔄 โหลดชื่อไฟล์", scale=1)
            
        url_input = gr.Textbox(label="🌐 วางลิงก์วิดีโอจากเว็บ (YouTube, TikTok ฯลฯ)", visible=False)
        
        isolate_vocal_checkbox = gr.Checkbox(label="🧹 ลบเสียงดนตรี/เสียงรบกวน (Vocal Isolation - แนะนำหากคลิปต้นฉบับมีเพลง)", value=False)
            
        analyze_btn = gr.Button("🧠 วิเคราะห์วิดีโอด้วย Local AI", variant="primary")
        analysis_status = gr.Textbox(label="สถานะการประมวลผล", interactive=False)
        topics_state = gr.State({})
        ref_candidates_state = gr.State({})
        current_video_path = gr.State("")
        
    with gr.Tab("2. Generate Pro Video (สร้างคลิปสั้น)"):
        topic_dropdown = gr.Dropdown(label="เลือกหัวข้อคลิปที่น่าสนใจ", choices=[], visible=False)
        
        with gr.Row():
            auto_ref_dropdown = gr.Dropdown(label="🗣️ เลือกเสียงต้นแบบจากวิดีโอ (เลือกว่าจะโคลนเสียงใคร)", choices=[], visible=False, scale=3)
            auto_ref_preview = gr.Audio(label="ฟังเสียงต้นแบบ", interactive=False, visible=False, scale=1)
        
        with gr.Row():
            orientation_radio = gr.Radio(label="รูปแบบวิดีโอ", choices=["Horizontal (16:9)", "Vertical (9:16)"], value="Vertical (9:16)")
            vertical_mode_radio = gr.Radio(label="สไตล์แนวตั้ง (เฉพาะ 9:16)", choices=["Smart Auto-Crop (เต็มจอ)", "Blur Background (ขอบเบลอ)"], value="Smart Auto-Crop (เต็มจอ)")
            resolution_dropdown = gr.Dropdown(label="ความละเอียดวิดีโอ", choices=["Original", "1080p", "720p"], value="1080p")
            bgm_input = gr.Audio(label="🎵 อัปโหลดเพลงพื้นหลัง (BGM) - ออปชันเสริม", type="filepath")
            
        with gr.Accordion("⚙️ ตั้งค่าขั้นสูง (Advanced Settings) ซับไตเติ้ล & เสียงเพลง", open=False):
            with gr.Row():
                sub_color = gr.Dropdown(label="สีซับไตเติ้ล", choices=["Yellow", "White", "Green"], value="Yellow")
                sub_size = gr.Slider(label="ขนาดตัวอักษร", minimum=40, maximum=150, value=85, step=5)
                bgm_vol = gr.Slider(label="ความดังเพลงพื้นหลัง BGM (%)", minimum=1, maximum=100, value=10, step=1)
                
            with gr.Row():
                watermark_input = gr.Image(label="🖼️ โลโก้ลายน้ำ (Watermark)", type="filepath")
                b_roll_input = gr.Video(label="🎞️ อัปโหลด B-Roll (ภาพแทรก)")
                
            custom_ref_audio = gr.Audio(label="อัปโหลดเสียงต้นแบบ (Custom Voice Cloning)", type="filepath")
            isolate_custom_ref_checkbox = gr.Checkbox(label="🧹 ลบเสียงดนตรี/เสียงรบกวนในเสียงต้นแบบ", value=False)
            custom_ref_text = gr.Textbox(label="ข้อความที่พูดในเสียงต้นแบบ (ภาษาไทย)")
            
            with gr.Row():
                test_gen_text = gr.Textbox(label="ข้อความสำหรับทดสอบเสียง (Test Script)", value="สวัสดีครับ นี่คือเสียงทดสอบระบบโคลนเสียงครับ", scale=3)
                test_voice_btn = gr.Button("🎧 ทดสอบเสียงโคลน", scale=1)
            with gr.Row():
                test_voice_output = gr.Audio(label="ผลลัพธ์เสียงทดสอบ", interactive=False)
                test_voice_status = gr.Textbox(label="สถานะทดสอบ", interactive=False)

        save_drive_checkbox = gr.Checkbox(label="บันทึกลง Google Drive อัตโนมัติ", value=True)
        generate_btn = gr.Button("🚀 สร้างวิดีโอขั้นสูง (พร้อม Progress Bar)", variant="primary")
        
        final_video_output = gr.Video(label="ผลลัพธ์วิดีโอ (Final Output)")
        generation_status = gr.Textbox(label="สถานะการสร้าง", interactive=False)

    with gr.Tab("3. Video Gallery (จัดการและดาวน์โหลดวิดีโอ)"):
        gr.Markdown("ดูและจัดการวิดีโอคลิปสั้นทั้งหมดที่คุณสร้างไว้ในระบบ (ทั้งไฟล์ชั่วคราวในเครื่อง และไฟล์ที่บันทึกบน Google Drive)")
        with gr.Row():
            refresh_btn = gr.Button("🔄 รีเฟรชรายการวิดีโอ", variant="secondary")
            video_list = gr.Dropdown(label="เลือกวิดีโอที่ต้องการจัดการ", choices=[])
            
        with gr.Row():
            preview_video = gr.Video(label="เล่นวิดีโอที่เลือก (Preview)")
            with gr.Column():
                delete_btn = gr.Button("🗑️ ลบวิดีโอนี้", variant="stop")
                manage_status = gr.Textbox(label="สถานะการจัดการ", interactive=False)
                download_file = gr.File(label="ดาวน์โหลดไฟล์วิดีโอลงเครื่อง")

        refresh_btn.click(fn=update_gallery, outputs=[video_list])
        video_list.change(fn=load_video_preview, inputs=[video_list], outputs=[preview_video])
        video_list.change(fn=lambda x: x, inputs=[video_list], outputs=[download_file])
        delete_btn.click(fn=delete_video, inputs=[video_list], outputs=[manage_status, video_list])
        
    drive_btn.click(fn=mount_google_drive, outputs=[drive_status, drive_path_input])
    
    refresh_drive_btn.click(fn=lambda: gr.update(choices=get_drive_videos()), outputs=[drive_path_input])

    def toggle_upload_ui(method):
        return (
            gr.update(visible=(method == "อัปโหลดจากเครื่อง (Local)")),
            gr.update(visible=(method == "เลือกจาก Google Drive")),
            gr.update(visible=(method == "วางลิงก์จากเว็บ (URL)"))
        )
        
    upload_method.change(
        fn=toggle_upload_ui,
        inputs=[upload_method],
        outputs=[video_input, drive_group, url_input]
    )

    analyze_btn.click(
        fn=analyze_video_chunked,
        inputs=[upload_method, video_input, drive_path_input, url_input, isolate_vocal_checkbox],
        outputs=[analysis_status, topic_dropdown, topics_state, auto_ref_dropdown, ref_candidates_state, current_video_path]
    )
    
    test_voice_btn.click(
        fn=test_voice_clone,
        inputs=[auto_ref_dropdown, ref_candidates_state, custom_ref_audio, custom_ref_text, test_gen_text, isolate_custom_ref_checkbox],
        outputs=[test_voice_output, test_voice_status]
    )

    generate_btn.click(
        fn=process_video_local,
        inputs=[
            current_video_path, topic_dropdown, topics_state, 
            orientation_radio, vertical_mode_radio, resolution_dropdown, bgm_input, bgm_vol, sub_color, sub_size, save_drive_checkbox, 
            auto_ref_dropdown, ref_candidates_state, custom_ref_audio, custom_ref_text, isolate_custom_ref_checkbox, watermark_input, b_roll_input
        ],
        outputs=[final_video_output, generation_status]
    )

if __name__ == "__main__":
    app.launch(share=True, debug=True)
