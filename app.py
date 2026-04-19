import gradio as gr
import subprocess
import os
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

def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Failed to remove {path}: {e}")

def mount_google_drive():
    try:
        # ตรวจสอบก่อนว่ามีการเมานท์ Drive ไว้แล้วหรือไม่
        if os.path.exists('/content/drive'):
            os.makedirs('/content/drive/MyDrive/AI_Videos', exist_ok=True)
            return "✅ Google Drive ถูกเชื่อมต่อไว้แล้ว! วิดีโอจะถูกบันทึกที่โฟลเดอร์ AI_Videos"
            
        from google.colab import drive
        drive.mount('/content/drive')
        os.makedirs('/content/drive/MyDrive/AI_Videos', exist_ok=True)
        return "✅ เชื่อมต่อ Google Drive สำเร็จ! วิดีโอจะถูกบันทึกที่โฟลเดอร์ AI_Videos"
    except ImportError:
        return "⚠️ ไม่สามารถเชื่อมต่อ Google Drive ได้ (ฟีเจอร์นี้ใช้ได้บน Google Colab เท่านั้น)"
    except Exception as e:
        if "'NoneType' object has no attribute 'kernel'" in str(e):
            return "⚠️ กรุณาเมานท์ Google Drive จากเซลล์ในสมุดโน้ต (Notebook) โดยตรงก่อน (ไม่สามารถเมานท์ผ่านปุ่มนี้ได้เมื่อรันด้วยคำสั่ง python app.py)"
        return f"⚠️ เกิดข้อผิดพลาด: {str(e)}"

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
    
    mp_face_detection = mp.solutions.face_detection
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
        
    del whisper_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    subs_path = f"temp_subs_{int(time.time())}.ass"
    with open(subs_path, "w", encoding="utf-8") as f:
        f.write(ass_content)
    return subs_path


# --- 4. Advanced Analysis (Chunking) ---
def analyze_video_chunked(video_path, drive_path, url_input, progress=gr.Progress()):
    if url_input and url_input.strip():
        progress(0, desc="🌐 กำลังดาวน์โหลดวิดีโอจากลิงก์...")
        output_filename = f"dl_video_{int(time.time())}.mp4"
        ydl_opts = {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', 'outtmpl': output_filename, 'quiet': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url_input.strip()])
            actual_video_path = output_filename
        except Exception as e:
            return f"เกิดข้อผิดพลาดในการดาวน์โหลดวิดีโอ: {str(e)}", gr.update(choices=[], visible=False), {}, {}, ""
    else:
        actual_video_path = drive_path.strip() if drive_path and drive_path.strip() else video_path
        
    if not actual_video_path or not os.path.exists(actual_video_path):
        return "กรุณาอัปโหลดวิดีโอ, ระบุพาธไฟล์ หรือใส่ลิงก์ให้ถูกต้อง", gr.update(choices=[], visible=False), {}, {}, ""
    
    audio_path = f"temp_audio_{int(time.time())}.wav"
    ref_audio_path = f"auto_ref_{int(time.time())}.wav"
    
    try:
        progress(0.05, desc="🎵 1/3 กำลังสกัดเสียงจากวิดีโอ...")
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["ffmpeg", "-y", "-i", actual_video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        
        progress(0.15, desc="🗣️ กำลังโหลดโมเดลถอดเสียง...")
        whisper_model = WhisperModel("small", device=device, compute_type=compute_type)
        
        progress(0.2, desc="📝 กำลังถอดเสียง (Transcription) อาจใช้เวลานานสำหรับคลิป 1 ชม...")
        segments_generator, info = whisper_model.transcribe(audio_path, beam_size=5)
        segments = list(segments_generator)
        
        transcript = ""
        ref_segment = None
        
        for segment in segments:
            transcript += f"[{format_time(segment.start)} - {format_time(segment.end)}] {segment.text}\n"
            # หาช่วงที่ยาว 4-10 วินาทีเพื่อใช้เป็น Reference สำหรับ F5-TTS
            duration = segment.end - segment.start
            if not ref_segment and 4 <= duration <= 10 and len(segment.text.strip()) > 10:
                ref_segment = segment
                
        # หากไม่เจอช่วงที่ยาวพอ ให้เอาช่วงแรกมาเลย
        if not ref_segment and segments:
            ref_segment = segments[0]
            
        ref_text = ""
        
        if ref_segment:
            progress(0.4, desc="🎙️ กำลังบันทึกเสียงอ้างอิง (Reference Audio) สำหรับ AI Voice...")
            subprocess.run(["ffmpeg", "-y", "-i", audio_path, "-ss", str(ref_segment.start), "-to", str(ref_segment.end), "-c:a", "pcm_s16le", "-ar", "24000", ref_audio_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ref_text = ref_segment.text.strip()
            
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
            prompt = f"""You are an expert Thai video editor. Read this transcript segment and find 1 to 2 interesting highlights to make short vertical videos.
Respond ONLY in valid JSON format exactly like this:
{{
    "topics": [
        {{"title": "ชื่อคลิปสั้นๆ", "start_time": "00:00:10", "end_time": "00:01:00", "voiceover_script": "บทพากย์สั้นๆ สำหรับช่วงนี้ (ภาษาไทย)"}}
    ]
}}
Transcript Segment:
{chunk}
"""
            messages = [{"role": "system", "content": "You are a helpful assistant. Output ONLY valid JSON."}, {"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(llm_model.device)
            
            outputs = llm_model.generate(**inputs, max_new_tokens=1024, temperature=0.7)
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

        choices = [f"{t['title']} ({t['start_time']} - {t['end_time']})" for t in all_topics]
        topics_dict = {f"{t['title']} ({t['start_time']} - {t['end_time']})": t for t in all_topics}
        
        # ส่งคืนค่ากลับไปเก็บใน UI
        ref_state = {"path": ref_audio_path if os.path.exists(ref_audio_path) else "", "text": ref_text}

        return f"วิเคราะห์เสร็จสิ้น พบ {len(all_topics)} หัวข้อตลอดความยาวคลิป!", gr.update(choices=choices, visible=True), topics_dict, ref_state, actual_video_path

    except Exception as e:
        return f"เกิดข้อผิดพลาด: {str(e)}", gr.update(choices=[], visible=False), {}, {}, ""
    finally:
        safe_remove(audio_path)

# --- 5. Test Voice Cloning ---
def test_voice_clone(ref_state, custom_ref_audio, custom_ref_text, test_script, progress=gr.Progress()):
    actual_ref_audio = custom_ref_audio if custom_ref_audio else ref_state.get("path", "")
    actual_ref_text = custom_ref_text if custom_ref_text else ref_state.get("text", "")
    
    if not actual_ref_audio or not actual_ref_text:
        return None, "⚠️ กรุณาอัปโหลดเสียงต้นแบบและใส่ข้อความที่พูดในเสียงต้นแบบ หรือทำการวิเคราะห์วิดีโอใน Tab 1 ก่อน"
        
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
def process_video_local(video_path, selected_topic_key, topics_dict, orientation, resolution, bgm_path, bgm_vol, font_color, font_size, save_to_drive, ref_state, custom_ref_audio, custom_ref_text, watermark_path, b_roll_path, progress=gr.Progress()):
    if not video_path or not os.path.exists(video_path):
        return None, "ไม่พบไฟล์วิดีโอต้นฉบับ กรุณากลับไปที่ Tab 1 เพื่ออัปโหลดและวิเคราะห์วิดีโอใหม่อีกครั้ง"
        
    if not selected_topic_key or not topics_dict:
        return None, "กรุณาวิเคราะห์วิดีโอและเลือกหัวข้อก่อน"
    
    temp_files = [] # Track files for garbage collection
    
    try:
        topic_info = topics_dict[selected_topic_key]
        start_time = topic_info['start_time']
        end_time = topic_info['end_time']
        script = topic_info['voiceover_script']
        
        audio_path = f"temp_voice_{int(time.time())}.wav"
        temp_files.append(audio_path)
        
        # 1. สร้างเสียงพากย์ด้วย F5-TTS-TH-V2
        if F5_AVAILABLE:
            progress(0.1, desc="🎙️ 1/4 กำลังโคลนเสียงพากย์ (F5-TTS Voice Cloning)...")
            
            actual_ref_audio = custom_ref_audio if custom_ref_audio else ref_state.get("path", "")
            actual_ref_text = custom_ref_text if custom_ref_text else ref_state.get("text", "")
            
            if not actual_ref_audio or not actual_ref_text:
                raise ValueError("ไม่พบเสียงอ้างอิง (Reference Audio) กรุณาอัปโหลดเสียงต้นแบบ")
                
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

        # 3. ตัดและครอปวิดีโอ (Smart Auto-Crop)
        progress(0.5, desc="✂️ 3/4 กำลังตัดวิดีโอและวิเคราะห์ใบหน้า (Smart Auto-Crop)...")
        temp_subclip = f"temp_sub_{int(time.time())}.mp4"
        temp_files.append(temp_subclip)
        
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-ss", start_time, "-to", end_time, "-c:v", "copy", "-c:a", "copy", temp_subclip], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if orientation == "Vertical (9:16)":
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
        else:
            video_filter = "crop=iw:iw*(9/16)"
            if resolution == "1080p":
                video_filter += ",scale=1920:1080"
            elif resolution == "720p":
                video_filter += ",scale=1280:720"

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
            
        video_filters = f"[0:v]{video_filter}[v_base]"
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
with gr.Blocks(title="The Ultimate Pro AI Video Agent", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎬 The Ultimate Pro AI Video Agent (F5-TTS 🇹🇭)")
    gr.Markdown("🌟 **[New!]** อัปเกรดระบบ **AI อ่านคลิปได้จบคลิป (1 ชม+)** มี **Progress Bar แสดงสถานะเรียลไทม์**, **ล้างไฟล์ขยะออโต้**, และตั้งค่า **สี/ขนาดซับไตเติ้ล** ได้แล้ว!")
    
    with gr.Row():
        drive_btn = gr.Button("🔗 เชื่อมต่อ Google Drive (เพื่อบันทึกไฟล์อัตโนมัติ)", variant="secondary")
        drive_status = gr.Textbox(label="สถานะ Google Drive", interactive=False)
        drive_btn.click(fn=mount_google_drive, outputs=[drive_status])

    with gr.Tab("1. Analyze Video (วิเคราะห์และดึงเสียงต้นแบบ)"):
        video_input = gr.Video(label="อัปโหลดวิดีโอเต็ม (1 ชม. ได้ ไม่จำกัดขนาด)")
        gr.Markdown("💡 **ทริคสำหรับไฟล์ขนาดใหญ่ (1 ชม.+)**: ระบบอัปโหลดผ่านเว็บอาจไม่แสดงสถานะเป็น MB ที่ชัดเจน แนะนำให้อัปโหลดวิดีโอลง Google Drive แล้วนำ **พาธไฟล์ (Path)** มาวางในช่องด้านล่าง จะรวดเร็วและเสถียรกว่ามาก!")
        with gr.Row():
            video_input = gr.Video(label="อัปโหลดวิดีโอจากเครื่อง")
            with gr.Column():
                drive_path_input = gr.Textbox(label="📂 ใส่พาธไฟล์จาก Google Drive (เช่น /content/drive/MyDrive/ep1.mp4)")
                url_input = gr.Textbox(label="🌐 หรือวางลิงก์วิดีโอจากเว็บ (YouTube, TikTok ฯลฯ)")
            
        analyze_btn = gr.Button("🧠 วิเคราะห์วิดีโอด้วย Local AI", variant="primary")
        analysis_status = gr.Textbox(label="สถานะการประมวลผล", interactive=False)
        topics_state = gr.State({})
        ref_audio_state = gr.State({})
        current_video_path = gr.State("")
        
    with gr.Tab("2. Generate Pro Video (สร้างคลิปสั้น)"):
        topic_dropdown = gr.Dropdown(label="เลือกหัวข้อคลิปที่น่าสนใจ", choices=[], visible=False)
        
        with gr.Row():
            orientation_radio = gr.Radio(label="รูปแบบวิดีโอ", choices=["Horizontal (16:9)", "Vertical (9:16)"], value="Vertical (9:16)")
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
        
    analyze_btn.click(
        fn=analyze_video_chunked,
        inputs=[video_input, drive_path_input, url_input],
        outputs=[analysis_status, topic_dropdown, topics_state, ref_audio_state, current_video_path]
    )
    
    test_voice_btn.click(
        fn=test_voice_clone,
        inputs=[ref_audio_state, custom_ref_audio, custom_ref_text, test_gen_text],
        outputs=[test_voice_output, test_voice_status]
    )

    generate_btn.click(
        fn=process_video_local,
        inputs=[
            current_video_path, topic_dropdown, topics_state, 
            orientation_radio, resolution_dropdown, bgm_input, bgm_vol, sub_color, sub_size, save_drive_checkbox, 
            ref_audio_state, custom_ref_audio, custom_ref_text, watermark_input, b_roll_input
        ],
        outputs=[final_video_output, generation_status]
    )

if __name__ == "__main__":
    app.launch(share=True, debug=True)
