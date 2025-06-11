import os
from flask import Flask, request, jsonify, render_template, Response, stream_with_context, send_from_directory
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextIteratorStreamer
from threading import Thread
from pathlib import Path
from itertools import groupby
from PIL import Image
import torch
import gc
import sys
from werkzeug.utils import secure_filename

app = Flask(__name__)

# --- Configuration ---
model_name = "google/gemma-3-27b-it"  # Adjust as needed.
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
LIBRARY_BASE = "library"
EVAL_BASE = "eval"

# Global session variables
session_active = False
session_status = "Not started"
model_instance = None
processor_instance = None

# Global conversation (for follow-up chat; text-only)
conversation = []
# Global dictionary for temporary eval uploads (filename -> (data, mimetype))
temp_eval_images = {}

# --- Helper Functions for Directory Browsing ---

def get_directory_contents(rel_path):
    base = Path(LIBRARY_BASE) / rel_path
    dirs = []
    files = []
    if not base.exists() or not base.is_dir():
        return {"directories": dirs, "files": files}
    for item in base.iterdir():
        if item.is_dir():
            dirs.append(item.name)
        elif item.is_file() and item.suffix.lower() in valid_extensions:
            files.append(item.name)
    return {"directories": sorted(dirs), "files": sorted(files)}

def get_images_recursive(rel_path):
    if not rel_path:
        return []
    base = Path(LIBRARY_BASE) / rel_path
    image_list = []
    if not base.exists():
        return image_list
    for root, dirs, files in os.walk(base):
        for file in files:
            if Path(file).suffix.lower() in valid_extensions:
                image_list.append(os.path.join(root, file))
    return image_list

def get_eval_images_data(root=EVAL_BASE):
    eval_dir = Path(root)
    image_paths = []
    final_question = ""
    eval_txt = eval_dir / "eval.txt"
    if eval_txt.exists():
        final_question = eval_txt.read_text().strip()
    for file in eval_dir.iterdir():
        if file.is_file() and file.suffix.lower() in valid_extensions:
            if file.name.lower() == "eval.txt":
                continue
            image_paths.append(os.path.join(root, file.name))
    image_paths.sort()
    return image_paths, final_question

# --- Helper Functions for Prompt Generation ---

def generate_reference_prompt(data):
    groups = {}
    for rel_dir, file_path in data:
        groups.setdefault(rel_dir, []).append(file_path)
    descriptions = []
    image_counter = 1
    for rel_dir, file_list in groups.items():
        count = len(file_list)
        start = image_counter
        end = image_counter + count - 1
        image_counter += count
        desc = f"Pictures {start} to {end} come from folder: {rel_dir}"
        descriptions.append(desc)
    return "\n".join(descriptions), image_counter - 1

def generate_eval_prompt_from_list(count, start_index):
    if count == 1:
        prompt = f"Picture {start_index} contains images after a mixture of CCCP and LLOMe treatment."
    else:
        prompt = f"Pictures {start_index} to {start_index+count-1} contain images after a mixture of CCCP and LLOMe treatment."
    return prompt

def trim_conversation(convo):
    trimmed = []
    for msg in convo:
        new_content = [item for item in msg["content"] if item["type"] == "text"]
        if new_content:
            trimmed.append({"role": msg["role"], "content": new_content})
    return trimmed

# --- Endpoints for Directory Browsing & Eval Question ---

@app.route("/dir", methods=["GET"])
def directory():
    rel_path = request.args.get("path", "")
    contents = get_directory_contents(rel_path)
    return jsonify({"path": rel_path, "contents": contents})

@app.route("/eval_question", methods=["GET"])
def default_eval_question():
    eval_txt = Path(EVAL_BASE) / "eval.txt"
    if eval_txt.exists():
        return jsonify({"question": eval_txt.read_text().strip()})
    return jsonify({"question": ""})

# --- Endpoints for Uploading Files ---

@app.route("/upload_library", methods=["POST"])
def upload_library():
    target = request.form.get("target", "")
    target_path = Path(LIBRARY_BASE) / target
    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)
    files = request.files.getlist("files")
    saved_files = []
    for file in files:
        filename = secure_filename(file.filename)
        file.save(os.path.join(target_path, filename))
        saved_files.append(filename)
    return jsonify({"status": "success", "files": saved_files})

@app.route("/upload_eval", methods=["POST"])
def upload_eval():
    global temp_eval_images
    files = request.files.getlist("files")
    saved_files = []
    for file in files:
        filename = secure_filename(file.filename)
        file_data = file.read()
        temp_eval_images[filename] = (file_data, file.mimetype)
        saved_files.append(filename)
    return jsonify({"status": "success", "files": saved_files})

# --- Endpoint to Return Eval Images List ---
@app.route("/eval_images", methods=["GET"])
def eval_images():
    persistent_paths, _ = get_eval_images_data(EVAL_BASE)
    images = []
    for path in persistent_paths:
        filename = os.path.basename(path)
        images.append({"filename": filename, "url": f"/eval_image/{filename}"})
    for filename in temp_eval_images.keys():
        images.append({"filename": filename, "url": f"/temp_eval_image/{filename}"})
    return jsonify(images)

# --- Endpoints for Serving Images ---
@app.route("/library_image/<path:filename>")
def library_image(filename):
    return send_from_directory(LIBRARY_BASE, filename)

@app.route("/eval_image/<path:filename>")
def eval_image(filename):
    return send_from_directory(EVAL_BASE, filename)

@app.route("/temp_eval_image/<filename>")
def temp_eval_image(filename):
    if filename in temp_eval_images:
        data, mimetype = temp_eval_images[filename]
        return Response(data, mimetype=mimetype)
    return "Not found", 404

# --- Session-Aware Model Loading Endpoints ---

@app.route("/start_session_interactive", methods=["POST"])
def start_session_interactive():
    global model_instance, processor_instance, session_active, session_status, conversation
    if session_active:
        return jsonify({"status": "already active", "session_status": session_status})
    session_status = "Loading"
    conversation = []  # clear previous history
    def load_model():
        global model_instance, processor_instance, session_active, session_status
        model_instance = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"
        )
        processor_instance = AutoProcessor.from_pretrained(model_name)
        session_active = True
        session_status = "Ready"
    Thread(target=load_model).start()
    return jsonify({"status": "session starting", "session_status": session_status})

@app.route("/session_status", methods=["GET"])
def get_session_status():
    return jsonify({"session_active": session_active, "session_status": session_status})

@app.route("/stop_session_interactive", methods=["POST"])
def stop_session_interactive():
    global model_instance, processor_instance, session_active, session_status, conversation
    try:
        del model_instance
        del processor_instance
    except Exception as e:
        print("Error during deletion:", e)
    torch.cuda.empty_cache()
    gc.collect()
    session_active = False
    session_status = "Not started"
    conversation = []
    return jsonify({"status": "session stopped", "session_status": session_status})

# --- Streaming Function for Real-Time Output ---
def stream_response(local_convo):
    if not session_active:
        yield "data: Session not active\n\n"
        return
    inputs = processor_instance.apply_chat_template(
        local_convo, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    streamer = TextIteratorStreamer(
        processor_instance.tokenizer, skip_special_tokens=True, skip_prompt=True
    )
    thread = Thread(target=model_instance.generate, kwargs={
        "inputs": inputs,
        "streamer": streamer,
        "max_new_tokens": 1024
    })
    thread.start()
    full_response = ""
    for token in streamer:
        token = token.replace("\\n", "\n")
        sys.stdout.write(token)
        sys.stdout.flush()
        full_response += token
        sse_token = token.replace("\n", "\\n")
        yield f"data: {sse_token}\n\n"
    thread.join()  # wait for generation thread to finish
    torch.cuda.synchronize()
    local_convo.append({"role": "assistant", "content": [{"type": "text", "text": full_response}]})
    torch.cuda.empty_cache()
    gc.collect()

# --- Endpoints for Processing and Chat ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    global conversation
    if not session_active:
        return jsonify({"error": "Session not active"}), 400
    data = request.get_json()
    selected_dirs = data.get("selected_dirs", [])
    final_question = data.get("final_question", "")
    selected_data = []
    for d in selected_dirs:
        imgs = get_images_recursive(d)
        for img_path in imgs:
            try:
                rel = str(Path(img_path).relative_to(LIBRARY_BASE))
            except Exception:
                rel = img_path
            rel_dir = str(Path(rel).parent)
            selected_data.append((rel_dir, img_path))
    ref_images = []
    for _, f in selected_data:
        try:
            img = Image.open(f).convert("RGB")
            ref_images.append(img)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    ref_prompt_text, ref_count = generate_reference_prompt(selected_data)
    persistent_eval_paths, _ = get_eval_images_data(EVAL_BASE)
    eval_images = []
    for f in persistent_eval_paths:
        try:
            img = Image.open(f).convert("RGB")
            eval_images.append(img)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    eval_count = len(eval_images)
    eval_prompt_text = generate_eval_prompt_from_list(eval_count, ref_count + 1)
    print(f"Reference images: {ref_count}, Evaluation images: {eval_count}")
    local_convo = []
    conversation_content = (
        [{"type": "image", "image": img} for img in ref_images] +
        [{"type": "text", "text": ref_prompt_text}] +
        [{"type": "image", "image": img} for img in eval_images] +
        [{"type": "text", "text": eval_prompt_text + "\n" + final_question}]
    )
    local_convo.append({"role": "user", "content": conversation_content})
    response = stream_with_context(stream_response(local_convo))
    conversation = local_convo.copy()
    return Response(response, mimetype="text/event-stream")

@app.route("/chat", methods=["POST"])
def chat():
    global conversation
    if not session_active:
        return jsonify({"error": "Session not active"}), 400
    data = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    if conversation and conversation[-1]["role"] == "user":
        last_text = conversation[-1]["content"][-1]["text"]
        if not last_text.endswith("\n"):
            conversation[-1]["content"][-1]["text"] += "\n" + user_message
        else:
            conversation[-1]["content"][-1]["text"] += user_message
    else:
        conversation.append({"role": "user", "content": [{"type": "text", "text": user_message}]})
    return Response(stream_with_context(stream_response(conversation)), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
