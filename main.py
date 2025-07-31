from flask import Flask, request, jsonify
import replicate
import os
import time

app = Flask(__name__)

# ✅ Replicate API Token kontrolü
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN is not set")

replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

@app.route("/")
def home():
    return "✅ Sac AI Server with FLUX & HairCLIP is Live"

# ✅ FLUX: Prompt ile saç stilini değiştir
@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    image_url = data.get("image")
    prompt = data.get("prompt", "Add a short black hairstyle, keep the face unchanged")

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    try:
        print("💬 FLUX PROMPT:", prompt)
        output = replicate_client.run(
            "flux-kontext-apps/change-haircut:bcb74d7c4db17efb87e2b8ddf0a2c152d69a9b1cfdb5cd8b3770b3c8cb1fcbe8",
            input={
                "input_image": image_url,
                "prompt": prompt
            }
        )

        if isinstance(output, list) and output:
            return jsonify({"result": output[0]})
        return jsonify({"result": output})

    except Exception as e:
        print("❌ FLUX ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

# ✅ HairCLIP: Stil ve renk ile saç düzenle
@app.route("/hairclip", methods=["POST"])
def hairclip():
    data = request.json
    image_url = data.get("image")
    style = data.get("style", "short hair")
    color = data.get("color", "black")

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    try:
        print(f"🎨 HairCLIP Style: {style}, Color: {color}")
        output = replicate_client.run(
            "wty-ustc/hairclip:b95cb2a16763bea87ed7ed851d5a3ab2f4655e94bcfb871edba029d4814fa587",
            input={
                "image": image_url,
                "editing_type": "both",
                "hairstyle_description": style,
                "color_description": color
            }
        )

        if isinstance(output, list) and output:
            return jsonify({"result": output[0]})
        return jsonify({"result": output})

    except Exception as e:
        print("❌ HairCLIP ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
