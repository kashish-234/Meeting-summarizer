from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont
import uuid

def _make_slide(text, size=(1280,720), font_size=36):
    img = Image.new("RGB", size, color=(18,18,18))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()
    lines = []
    words = text.split()
    cur = ""
    for w in words:
        if len(cur)+len(w)+1 <= 60:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    y = 80
    for line in lines:
        draw.text((60, y), line, font=font, fill=(255,255,255))
        y += font_size + 12
    path = f"slide_{uuid.uuid4().hex}.png"
    img.save(path)
    return path

def compose_video(summary_text, audio_path, out_path="summary_video.mp4"):
    sentences = [s.strip() for s in summary_text.split(".") if s.strip()]
    if not sentences:
        sentences = [summary_text]
    def _image_clip_with_duration(path, dur):
        """Create an ImageClip and set duration in a backward-compatible way.

        Some moviepy versions expose ImageClip.set_duration(), others expose
        with_duration(). As a last resort set the .duration attribute.
        """
        clip = ImageClip(path)
        # Prefer methods if available
        if hasattr(clip, "set_duration"):
            return clip.set_duration(dur)
        if hasattr(clip, "with_duration"):
            return clip.with_duration(dur)
        # Fallback: set attribute
        try:
            clip.duration = dur
        except Exception:
            # If even that fails, return the clip unchanged (may error later)
            pass
        return clip

    clips = [_image_clip_with_duration(_make_slide(s), 4) for s in sentences]
    video = concatenate_videoclips(clips)
    audio = AudioFileClip(audio_path)
    # Backward/forward compatible: some moviepy versions use set_audio(),
    # others expose with_audio(). Fall back to setting the attribute.
    if hasattr(video, "set_audio"):
        video = video.set_audio(audio)
    elif hasattr(video, "with_audio"):
        video = video.with_audio(audio)
    else:
        try:
            video.audio = audio
        except Exception:
            # If this fails, leave video unchanged and let moviepy raise later
            pass
    video.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac")
    return out_path
