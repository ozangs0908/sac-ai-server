from flask import Flask, request, jsonify
import replicate
import os

app = Flask(__name__)
os.environ["REPLICATE_API_TOKEN"] = "r8_QAHHlvSXCud4CfoEw61Klq7ZDprTdJy3k8ylT"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    image_url = data.get("image")
    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400
    try:
        output = replicate.run("tencentarc/gfpgan:1.4", input={"img": image_url})
        return jsonify({"result": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
