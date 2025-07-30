from flask import Flask, request, jsonify
import replicate
import os

app = Flask(__name__)

# Replicate API Token ortam değişkeninden alınır
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

# Replicate'ı API key ile başlat
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    image_url = data.get("image")

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    try:
        # ✅ Replicate modeli çalıştır
        output = replicate_client.run(
            "tencentarc/gfpgan:0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c",
            input={
                "img": image_url,
                "scale": 2
            }
        )

        # 🔁 Liste döner, ilk elemanı al
        return jsonify({"result": output[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
