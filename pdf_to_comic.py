import os
import io
import json
import re

from google import genai
from google.genai import types
from PyPDF2 import PdfReader
from PIL import Image, ImageDraw, ImageFont
from tenacity import retry, wait_random_exponential
import gradio as gr
import textwrap
from dotenv import load_dotenv
load_dotenv() 

client = genai.Client(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))
# extract text from PDF
def extract_pdf_text(path):
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)

# creates the driving points that make up a story
def make_beats(text):
    prompt = (
        """Turn this research paper into comic-story beats. Each beat should focus on a single event in the paper. Together, the beats should form a clear
        setup, process, turning point, insight, and conclusion. Return a JSON array of strings. Paper text:"""
        + text
    )
    resp = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(temperature=0.3)
    )
    raw = "".join(part.text for part in resp.candidates[0].content.parts if part.text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return [line.strip() for line in raw.splitlines() if line.strip()]

# parse json
def extract_json(raw_text):
    clean = re.sub(r'```(?:json)?', '', raw_text).strip()
    decoder = json.JSONDecoder()
    pos = 0
    results = []
    length = len(clean)

    while pos < length:
        while pos < length and clean[pos] in " \t\r\n,":
            pos += 1
        if pos >= length:
            break

        obj, end = decoder.raw_decode(clean, pos)
        results.append(obj)
        pos = end

    return results

# use the plot points (beats) to create specific scenes and dialogue
def make_script(beat):
    prompt = (
        f"Create a comic panel script for this story {beat}. "  
        """Describe a visual scene and write relevant character dialogue. 
        Make sure the dialogue is natural and concise. Return JSON: {\"scene\":, \"dialogue\":}."""
    )
    resp = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(temperature=0.3)
    )
    raw = "".join(part.text for part in resp.candidates[0].content.parts if part.text)
    clean_json = raw.strip().removeprefix("```json").removesuffix("```").strip()
    panels = extract_json(clean_json)
    return {"scene": panels[0]['scene'], "dialogue": panels[0]['dialogue']}

# given a scene, generate an image of it
def make_image(scene, beat):
    img_query = (
        f"""Create an image prompt for a comic panel given a {scene} and a {beat}. 
        Make sure to include consistent characters and key ideas from the beat."""
    )
    resp_prompt = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[types.Part.from_text(text=img_query)],
        config=types.GenerateContentConfig(temperature=0.4)
    )
    img_prompt = "".join(part.text for part in resp_prompt.candidates[0].content.parts if part.text).strip()
    resp_img = client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=img_prompt,
        config=types.GenerateContentConfig(response_modalities=["Text","Image"], temperature=0.3)
    )
    for part in resp_img.candidates[0].content.parts:
        if part.inline_data:
            return Image.open(io.BytesIO(part.inline_data.data))

# combine all images and dialogues into one comic strip
def make_comic(panels, images, scale=3):
    widths, heights = zip(*(img.size for img in images))
    total_w, max_h = sum(widths), max(heights)+300
    canvas = Image.new("RGB", (total_w * scale, max_h * scale), "white")
    draw = ImageDraw.Draw(canvas)
    x_off = []
    x = 0
    for img in images:
        hi = img.resize((img.width*scale, img.height*scale), Image.LANCZOS)
        canvas.paste(hi, (x*scale, 0))
        x_off.append(x*scale)
        x += img.width

    font = ImageFont.truetype("arial.ttf", size=18*scale)
    margin = 10*scale

    for i, panel in enumerate(panels):
        raw = panel.get("dialogue", [])
        if isinstance(raw, str):
            dlg = [{"character": "", "line": raw}]
        elif isinstance(raw, dict):
            dlg = []
            for name, text in raw.items():
                dlg.append({"character": name, "line": text.strip()})
        elif isinstance(raw, list):
            dlg = raw
        else:
            dlg = []
            
        try:
            lines = [f"{d.get('character','')}: {d['line']}".strip(": ") for d in dlg]
        except:
            lines = [f"{d.get('character','')}: {d['text']}".strip(": ") for d in dlg]
        
        text_block = "\n".join(lines)

        panel_w = images[i].width*scale
        avg_c = font.getlength("A") or 1
        wrap_w = max(int((panel_w-2 * margin) / avg_c),1)
        wrapped = textwrap.wrap(text_block, width=wrap_w)

        text_h = sum(font.getbbox(l)[3] for l in wrapped)
        bx = x_off[i] + margin
        by = images[i].height * scale + margin
        bw = panel_w - 2 * margin
        bh = text_h + 2 * margin

        draw.rounded_rectangle([bx,by, bx + bw, by + bh], radius=10 * scale, fill="white", outline="black", width=2)
        tail = [(bx + 10 * scale, by + bh), (bx+20 * scale, by + bh), (bx + 10 * scale, by + bh + 10 * scale)]
        draw.polygon(tail, fill="white", outline="black")
        draw.multiline_text((bx + margin, by + margin), "\n".join(wrapped), font=font, fill="black", spacing=4 * scale)

    return canvas.resize((total_w, max_h), Image.LANCZOS)

unwanted_text = {'json', '```json', '', '[', ']', '{', '}', '```'} # strip these tokens from the beats

# convert a PDF into a comic strip
@retry(wait=wait_random_exponential(multiplier=1, max=60))
def pdf_to_comic(pdf_path):
    text = extract_pdf_text(pdf_path)
    raw_beats = make_beats(text)
    raw_beats = [b.strip().rstrip(',') for b in raw_beats if b.strip()]
    beats = []
    for beat in raw_beats:
        for token in unwanted_text:
            beat = beat.replace(token, "")
        beat = beat.strip().rstrip(",")
        if beat:
            beats.append(beat)
    panels, images = [], []
    for beat in beats:
        panel = make_script(beat)

        panels.append(panel)

        img = make_image(panel["scene"], beat)
        images.append(img)

    return make_comic(panels, images)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("PDF to Comic")
    with gr.Row():
        pdf_in = gr.File(label="Upload PDF", file_count="single", type="filepath")
    comic_out = gr.Image(label="Comic Strip")
    convert_btn = gr.Button("Convert to Comic")
    convert_btn.click(fn=pdf_to_comic, inputs=[pdf_in], outputs=[comic_out])

if __name__ == "__main__":
    demo.launch()