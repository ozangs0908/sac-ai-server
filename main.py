from flask import Flask, request, jsonify
import replicate
import os

app = Flask(__name__)

REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN is not set.")

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

@app.route("/")
def home():
    return "Sac AI Server aktif!"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    image_url = data.get("image")

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    try:
        output = replicate.run(
            "tencentarc/gfpgan:1.4",  # ✅ doğru model versiyonu
            input={"img": image_url}
        )
        return jsonify({"result": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
