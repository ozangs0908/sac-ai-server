from flask import Flask, request, jsonify
import replicate
import os

app = Flask(__name__)

# 🔐 Replicate API anahtarını ortam değişkeninden al
os.environ["REPLICATE_API_TOKEN"] = "r8_senin_tokenin_buraya"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    image_url = data.get("image")

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    try:
        output = replicate.run(
            "tencentarc/gfpgan:0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c",
            input={"img": image_url}
        )

        # Çıktıyı string URL olarak dön
        result_url = str(output[0]) if isinstance(output, list) else str(output)
        return jsonify({"result": result_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
