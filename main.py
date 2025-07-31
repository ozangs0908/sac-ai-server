from flask import Flask, request, jsonify
import replicate, os

app = Flask(__name__)
replicate_client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

@app.route("/flux", methods=["POST"])
def flux():
    data = request.json
    image = data.get("image")
    prompt = data.get("prompt", "Add a short black haircut, keep face unchanged")
    if not image:
        return jsonify({"error": "Image URL required"}),400

    try:
        output = replicate_client.run(
            "flux-kontext-apps/change-haircut:48f03523665cabe9a2e832ea9cc2d7c30ad5079cb5f1c1f07890d40596fe1f87",
            input={"input_image": image, "haircut": prompt, "hair_color":"Black", "gender":"none"}
        )
        return jsonify({"result": output})
    except Exception as e:
        return jsonify({"error": str(e)}),500

@app.route("/hairclip", methods=["POST"])
def hairclip():
    data = request.json
    image = data.get("image")
    style = data.get("style", "short hair")
    color = data.get("color", "black")
    if not image:
        return jsonify({"error": "Image URL required"}),400

    try:
        output = replicate_client.run(
            "wty-ustc/hairclip:b95cb2a16763bea87ed7ed851d5a3ab2f4655e94bcfb871edba029d4814fa587",
            input={
                "image": image,
                "editing_type": "both",
                "hairstyle_description": style,
                "color_description": color
            }
        )
        return jsonify({"result": output})
    except Exception as e:
        return jsonify({"error": str(e)}),500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
