from flask import Flask, request, jsonify
import replicate, os

app = Flask(__name__)
replicate_client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

@app.route("/flux", methods=["POST"])
def flux():
    data = request.json
    image = data.get("image")
    prompt = data.get("prompt", "Crew Cut")  # default saç modeli
    hair_color = data.get("hair_color", "Black")  # istersen bu da sabitlenebilir

    if not image:
        return jsonify({"error": "Image URL required"}), 400

    try:
        output = replicate_client.run(
            "flux-kontext-apps/change-haircut:48f03523665cabe9a2e832ea9cc2d7c30ad5079cb5f1c1f07890d40596fe1f87",
            input={
                "input_image": image,
                "haircut": prompt,
                "hair_color": hair_color,
                "gender": "male"  # GENDER SABİT
            }
        )
        return jsonify({"result": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
