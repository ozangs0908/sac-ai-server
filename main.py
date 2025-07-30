from flask import Flask, request, jsonify
import replicate
import os
import logging

app = Flask(__name__)

# Loglama ayarları
logging.basicConfig(level=logging.INFO)

# Ortam değişkeninden API token al
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN is not set")

# Replicate istemcisi başlat
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

@app.route("/")
def home():
    return "Sac AI Server is live"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    image_url = data.get("image")
    prompt = data.get("prompt", "add short hair to the person")  # Varsayılan prompt

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    try:
        logging.info(f"Image: {image_url}")
        logging.info(f"Prompt: {prompt}")

        output = replicate_client.run(
            "cjwbw/instruct-pix2pix:latest",
            input={
                "image": image_url,
                "prompt": prompt,
                "num_inference_steps": 30,
                "guidance_scale": 7.5,
                "image_guidance_scale": 1.5
            }
        )

        logging.info(f"Output: {output}")

        if isinstance(output, list):
            return jsonify({"result": output[0]})
        return jsonify({"result": str(output)})

    except replicate.exceptions.ReplicateError as e:
        logging.error(f"ReplicateError: {e}")
        return jsonify({"error": f"ReplicateError: {str(e)}"}), 500

    except Exception as e:
        logging.error(f"Unhandled Exception: {e}")
        return jsonify({"error": f"Exception: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
