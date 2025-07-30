from flask import Flask, request, jsonify
import replicate
import os

app = Flask(__name__)

# ✅ Replicate API token ortam değişkeninden alınır
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    image_url = data.get("image")

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    try:
        # ✅ Replicate modelini çalıştır (GFPGAN v1.4 daha iyi sonuç verir)
        output = replicate_client.run(
            "tencentarc/gfpgan:0fbacf7afc6c144e5be9767cff80f25aff23e52b0708f17e20f9879b2f21516c",
            input={
                "img": image_url,
                "scale": 2
            }
        )

        # ✅ output genellikle liste döner → ilk elemanı al
        if isinstance(output, list) and len(output) > 0:
            return jsonify({"result": output[0]})
        else:
            return jsonify({"error": "No result returned from model"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
