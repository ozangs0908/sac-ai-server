from flask import Flask, request, jsonify
import replicate
import os
import time

app = Flask(__name__)

# ✅ Replicate API token'ı ortamdan al
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN is not set")

replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

@app.route("/")
def home():
    return "✅ Sac AI Flux Server is Live"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    image_url = data.get("image")
    prompt = data.get("prompt", "add short black hairstyle, do not modify the face")

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    try:
        print("📸 IMAGE URL:", image_url)
        print("💬 PROMPT:", prompt)
        start_time = time.time()

        output = replicate_client.run(
            "flux-kontext-apps/change-haircut",
            input={
                "input_image": image_url,
                "prompt": prompt
            }
        )

        elapsed = round(time.time() - start_time, 2)
        print(f"🕐 ELAPSED TIME: {elapsed} seconds")
        print("✅ OUTPUT:", output)

        if isinstance(output, list) and output:
            return jsonify({"result": output[0]})
        elif isinstance(output, str):
            return jsonify({"result": output})
        else:
            return jsonify({"error": "Model ran but returned no usable result."}), 500

    except replicate.exceptions.ReplicateError as e:
        print("🔥 REPLICATE ERROR:", e)
        return jsonify({"error": f"ReplicateError: {str(e)}"}), 500

    except Exception as e:
        print("💥 UNHANDLED EXCEPTION:", e)
        return jsonify({"error": f"Exception: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
