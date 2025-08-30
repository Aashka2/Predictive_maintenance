from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load your model (make sure path is correct)
model_path = os.path.join(os.path.dirname(
    __file__), '../model/trained_model.pkl')
model = joblib.load(model_path)

# Feature names used during training
FEATURE_COLUMNS = ["Air temperature [K]", "Process temperature [K]",
                   "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

# Home page (static UI)


@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict-page")
def predict_page():
    return render_template("predict.html") 




@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Check for JSON input
        if request.is_json:
            data = request.get_json()

            # Option 1: Full feature set sent from a web form or API
            if all(key in data for key in FEATURE_COLUMNS):
                df = pd.DataFrame([data])
                prediction = model.predict(df)[0]
                return jsonify({"maintenance_needed": bool(prediction)})

            # Option 2: Just temperature from ESP8266 or simple IoT device
            elif "temperature" in data:
                # Dummy values for other features
                full_input = {
                    "Air temperature [K]": float(data["temperature"]),
                    # Adjust as needed
                    "Process temperature [K]": float(data["temperature"]) + 10,
                    "Rotational speed [rpm]": 1500,
                    "Torque [Nm]": 40,
                    "Tool wear [min]": 5
                }
                df = pd.DataFrame([full_input])
                prediction = model.predict(df)[0]
                return jsonify({"maintenance_needed": bool(prediction)})

            else:
                return jsonify({"error": "Insufficient data in JSON"}), 400

        # If not JSON, try form data (optional)
        else:
            form_data = request.form.to_dict()
            df = pd.DataFrame([form_data])[FEATURE_COLUMNS].astype(float)
            prediction = model.predict(df)[0]
            return jsonify({"maintenance_needed": bool(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
