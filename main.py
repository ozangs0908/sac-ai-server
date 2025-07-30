from flask import Flask, request, jsonify
import replicate
import os

app = Flask(__name__)

# Replicate API token'ı Render panelinden Environment Variables olarak eklenecek!
# os.environ["REPLICATE_API_TOKEN"] = "xxx"  # BUNU BURADA TUTMA!

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    image_url = data.get("image")

    if not image_url:
        return jsonify({"error": "Image URL is required"}), 400

    try:
        output = replicate.run(
            "tencentarc/gfpgan:92895e48c621c3f19aa9e584cff0980483c0a801",
            input={"img": image_url}
        )
        return jsonify({"result": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
