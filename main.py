from flask import Flask, request, jsonify
import replicate
import os
import logging

app = Flask(__name__)

# 🚨 Loglar terminalde çıksın
logging.basicConfig(level=logging.INFO)

# ✅ Ortam değişkeninden API token al
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise RuntimeError("REPLICATE_API_TOKEN is not set")

replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

@app.route("/")
def home():
    return "Sac AI Server is live"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    image_url = data.get("image")

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    try:
        # 📤 Girdi logu
        logging.info(f"Received image URL: {image_url}")

        output = replicate_client.run(
            "tencentarc/gfpgan:0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c",
            input={
                "img": image_url,
                "scale": 2
            }
        )

        # 🧪 Output tipi logla
        logging.info(f"Raw output: {output}")
        logging.info(f"Output type: {type(output)}")

        # 🛠️ Eğer liste ise ilk elemanı al
        if isinstance(output, list):
            return jsonify({"result": output[0]})
        
        # 🛠️ Eğer string değilse stringe çevir
        return jsonify({"result": str(output)})
    
    except replicate.exceptions.ReplicateError as e:
        logging.error(f"Replicate API Error: {e}")
        return jsonify({"error": f"ReplicateError: {str(e)}"}), 500

    except Exception as e:
        logging.error(f"Unhandled Exception: {e}")
        return jsonify({"error": f"Exception: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
