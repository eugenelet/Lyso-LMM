import os
import gc
import sys
import torch
from flask import Flask, request, jsonify, Response, stream_with_context
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, TextIteratorStreamer
from threading import Thread
from pathlib import Path

app = Flask(__name__)

# --- Configuration ---
model_name = "google/gemma-3-27b-it"
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
LIBRARY_BASE = "library"
EVAL_BASE = "eval"

# Global model and processor variables
model = None
processor = None
session_active = False
session_status = "Not started"

@app.route("/session_status", methods=["GET"])
def session_status_endpoint():
    return jsonify({"session_active": session_active, "session_status": session_status})

def load_model():
    global model, processor, session_active, session_status
    session_status = "Loading"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    # Warm-up dummy inference
    dummy_conv = [{"role": "user", "content": [{"type": "text", "text": "dummy"}]}]
    dummy_inputs = processor.apply_chat_template(
        dummy_conv, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    with torch.no_grad():
        _ = model.generate(dummy_inputs, max_new_tokens=1)
    session_active = True
    session_status = "Ready"

@app.route("/inference", methods=["POST"])
def inference():
    if not session_active:
        return jsonify({"error": "Session not active"}), 400
    data = request.get_json()
    conversation = data.get("conversation", [])
    def generate():
        inputs = processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        streamer = TextIteratorStreamer(
            processor.tokenizer, skip_special_tokens=True, skip_prompt=True
        )
        thread = Thread(target=model.generate, kwargs={
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
        thread.join()
        torch.cuda.empty_cache()
        gc.collect()
    return Response(stream_with_context(generate()), mimetype="text/event-stream")

if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=8001, debug=False)
