import os
import subprocess
import time
import requests
import gc
from flask import Flask, request, jsonify, render_template, Response, stream_with_context, send_from_directory
from pathlib import Path
from werkzeug.utils import secure_filename
from PIL import Image
import shutil
import base64
import io

app = Flask(__name__)

# --- Configuration ---
LIBRARY_BASE = "library"
EVAL_BASE = "eval"
MODEL_WORKER_URL = "http://localhost:8001"  # Model worker runs on port 8001

# Global variable to hold the model worker process handle
model_worker_process = None

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
        elif item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            files.append(item.name)
    return {"directories": sorted(dirs), "files": sorted(files)}

def get_images_recursive(rel_path):
    if not rel_path:
        return []
    base = Path(LIBRARY_BASE) / rel_path
    image_list = []
    if not base.exists():
        return image_list
    for root, _, files in os.walk(base):
        for file in files:
            if Path(file).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
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
        if file.is_file() and file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            if file.name.lower() == "eval.txt":
                continue
            image_paths.append(os.path.join(root, file.name))
    image_paths.sort()
    return image_paths, final_question

# --- Helper Functions for Prompt Generation ---
def generate_reference_prompt(data):
    # Data is a list of tuples: (rel_dir, file_path)
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
        prompt = f"Picture {start_index} contains images for evaluations."
    else:
        prompt = f"Pictures {start_index} to {start_index+count-1} contain images for evaluation."
    return prompt

# --- Image Upload with Resize (300px width) ---

def resize_and_save_image(file, target_path):
    try:
        # Open the image
        img = Image.open(file)

        # Check extension from file path if available
        if hasattr(file, 'filename'):
            original_ext = os.path.splitext(file.filename)[1].lower()
        else:
            original_ext = os.path.splitext(str(file))[1].lower()

        # Convert TIFF/TIF to JPEG
        if img.format.lower() == 'tiff' or original_ext in ['.tif', '.tiff']:
            img = img.convert('RGB')  # Required for JPEG
            target_path = os.path.splitext(target_path)[0] + '.jpg'

        # Resize to 300px width while maintaining aspect ratio
        width_percent = 300 / float(img.size[0])
        new_height = int(float(img.size[1]) * width_percent)
        img = img.resize((300, new_height), Image.Resampling.LANCZOS)

        # Save image
        img.save(target_path)
        print(f"Saved resized image to {target_path}")
        return True
    except Exception as e:
        print("Error resizing image:", e)
        return False


# --- Delete Endpoint ---
@app.route("/delete", methods=["POST"])
def delete_item():
    item_type = request.args.get("type", "")
    path = request.args.get("path", "")
    if item_type not in {"dir", "file", "eval"} or not path:
        return jsonify({"status": "error", "error": "Invalid parameters"}), 400
    try:
        if item_type in {"dir", "file"}:
            full_path = Path(LIBRARY_BASE) / path
        elif item_type == "eval":
            full_path = Path(EVAL_BASE) / path
        if not full_path.exists():
            return jsonify({"status": "error", "error": "Path not found"}), 404
        if item_type == "dir":
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
    
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
        full_path = os.path.join(target_path, filename)
        # Resize before saving
        if resize_and_save_image(file, full_path):
            saved_files.append(filename)
    return jsonify({"status": "success", "files": saved_files})

@app.route("/upload_eval", methods=["POST"])
def upload_eval():
    files = request.files.getlist("files")
    saved_files = []
    for file in files:
        filename = secure_filename(file.filename)
        full_path = os.path.join(EVAL_BASE, filename)
        # For eval images, we also resize before saving
        if resize_and_save_image(file, full_path):
            saved_files.append(filename)
    return jsonify({"status": "success", "files": saved_files})

@app.route("/eval_images", methods=["GET"])
def eval_images():
    persistent_paths, _ = get_eval_images_data(EVAL_BASE)
    images = []
    for path in persistent_paths:
        filename = os.path.basename(path)
        images.append({"filename": filename, "url": f"/eval_image/{filename}"})
    return jsonify(images)

@app.route("/library_image/<path:filename>")
def library_image(filename):
    return send_from_directory(LIBRARY_BASE, filename)

@app.route("/eval_image/<path:filename>")
def eval_image(filename):
    return send_from_directory(EVAL_BASE, filename)

# --- Session Management: Start/Stop Model Worker ---
import subprocess

@app.route("/session_status", methods=["GET"])
def get_session_status():
    try:
        r = requests.get(f"{MODEL_WORKER_URL}/session_status")
        return jsonify(r.json())
    except Exception:
        return jsonify({"session_active": False, "session_status": "Not started"})

@app.route("/start_session_interactive", methods=["POST"])
def start_session_interactive():
    global model_worker_process
    if model_worker_process is not None:
        return jsonify({"status": "already active"})
    # Start the model worker process
    model_worker_process = subprocess.Popen(["python", "model_worker.py"])
    # Poll until worker reports session_active==True and session_status=="Ready"
    for i in range(30):
        try:
            r = requests.get(f"{MODEL_WORKER_URL}/session_status")
            data = r.json()
            if data.get("session_active") and data.get("session_status") == "Ready":
                break
        except Exception:
            pass
        time.sleep(1)
    return jsonify({"status": "session started", "session_status": "Ready"})

@app.route("/stop_session_interactive", methods=["POST"])
def stop_session_interactive():
    global model_worker_process
    if model_worker_process:
        model_worker_process.terminate()
        model_worker_process.wait()
        model_worker_process = None
    return jsonify({"status": "session stopped", "session_status": "Not started"})



def pil_image_to_base64_str(img, format="JPEG"):
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Forward Inference Requests to Model Worker ---
def forward_inference(payload):
    r = requests.post(f"{MODEL_WORKER_URL}/inference", json=payload, stream=True)
    for line in r.iter_lines(decode_unicode=True):
        if line:
            yield line + "\n"

# --- Endpoints for Processing and Chat ---
conversation = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    global conversation
    data = request.get_json()
    selected_dirs = data.get("selected_dirs", [])
    final_question = data.get("final_question", "")
    # Also get selected eval image filenames
    selected_eval = data.get("selected_eval", [])
    
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
    # Filter persistent eval images based on selected filenames
    if selected_eval:
        persistent_eval_paths = [p for p in persistent_eval_paths if os.path.basename(p) in selected_eval]
    eval_count = len(persistent_eval_paths)
    eval_prompt_text = generate_eval_prompt_from_list(eval_count, ref_count + 1)
    print(f"Reference images: {ref_count}, Evaluation images: {eval_count}")
    eval_images = []
    for f in persistent_eval_paths:
        try:
            img = Image.open(f).convert("RGB")
            eval_images.append(img)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    local_convo = []
    conversation_content = (
        [{"type": "image", "image": pil_image_to_base64_str(img)} for img in ref_images] +
        [{"type": "text", "text": ref_prompt_text}] +
        [{"type": "image", "image": pil_image_to_base64_str(img)} for img in eval_images] +
        [{"type": "text", "text": eval_prompt_text + "\n" + final_question}]
    )
    local_convo.append({"role": "user", "content": conversation_content})
    payload = {"conversation": local_convo}
    response = Response(forward_inference(payload), mimetype="text/event-stream")
    conversation = local_convo.copy()
    return response
    # local_convo = []
    # conversation_content = (
    #     [{"type": "image", "image": img} for img in ref_images] +
    #     [{"type": "text", "text": ref_prompt_text}] +
    #     [{"type": "image", "image": img} for img in eval_images] +
    #     [{"type": "text", "text": eval_prompt_text + "\n" + final_question}]
    # )
    # local_convo.append({"role": "user", "content": conversation_content})
    # payload = {"conversation": local_convo}
    # response = Response(forward_inference(payload), mimetype="text/event-stream")
    # conversation = local_convo.copy()
    # return response

@app.route("/chat", methods=["POST"])
def chat():
    global conversation
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
    payload = {"conversation": conversation}
    return Response(forward_inference(payload), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
