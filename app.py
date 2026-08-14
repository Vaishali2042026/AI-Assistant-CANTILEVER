from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os

# Import NLP Chatbot model function
from chatbot.chatbot_model import generate_response

# Import BLIP / Image Captioning module
from image_captioning import caption

# ============================================================
# FLASK APP SETUP
# ============================================================

app = Flask(__name__)
CORS(app)  # Prevents CORS errors across routes

# Create temporary upload directory if it doesn't exist
UPLOAD_FOLDER = 'image_captioning/temp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# 1. HOMEPAGE ROUTE (UNTOUCHED)
# ============================================================

@app.route("/")
def home():
    return render_template("home.html")


# ============================================================
# 2. CHATBOT ROUTES (UNTOUCHED)
# ============================================================

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")


@app.route("/api/chat", methods=["POST"])
def chat_api():
    try:
        data = request.get_json(silent=True) or {}
        user_message = data.get("message", "")

        if not user_message.strip():
            return jsonify({
                "response": "Please type something."
            }), 400

        response = generate_response(user_message)

        return jsonify({
            "response": response
        })

    except Exception as error:
        print("CHAT ERROR:", error)
        return jsonify({
            "response": "I'm sorry, something went wrong. Please try again."
        }), 500


# ============================================================
# 3. IMAGE CAPTION PAGE ROUTE (UNTOUCHED)
# ============================================================

@app.route("/image-caption")
def image_caption():
    return render_template("image_caption.html")


# ============================================================
# 4. IMAGE CAPTION API ENDPOINTS
# Supports both '/api/image-caption' and '/predict' endpoints
# so your frontend JavaScript works without HTML edits.
# ============================================================

@app.route("/api/image-caption", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    temp_path = None
    try:
        temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(temp_path)

        # Generate real caption using the module
        predicted_caption = caption.generate_caption(temp_path)

        # Return JSON with both keys for frontend compatibility
        return jsonify({
            "caption": predicted_caption,
            "response": predicted_caption
        })

    except Exception as e:
        print(f"IMAGE CAPTION ERROR: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temporary file safely
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)