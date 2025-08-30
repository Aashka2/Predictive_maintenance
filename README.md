Tech Stack Used:
1 IoT & Embedded: ESP8266 (NodeMCU), DHT11 sensor
2 Programming: Python
3 Machine Learning: Random Forest Classifier (sklearn), Pandas, NumPy
4 Web Development: Flask (Backend), HTML/CSS/JavaScript (Frontend)
5 Model Storage: Joblib (for saving trained model)

Steps Followed
1. Data Preparation & ML Model:
Collected predictive_maintenance.csv dataset.
Cleaned & preprocessed data (pandas, numpy).
Encoded categorical features using OneHotEncoding.
Trained Random Forest Classifier to predict Target (Maintenance Needed vs Not Needed).
Saved trained model as trained_model.pkl.
2. IoT Setup:
Configured ESP8266 + DHT11 sensor to read:
Temperature (C/F)
Humidity
Sent sensor data via Serial / Wi-Fi .
3. Flask Backend:
Created Flask API with /predict route.
API takes JSON input (sensor readings) and passes to ML model.
Returns response:
"maintenance_needed": true/false.
4. Web Frontend (predict.html):
Built an interactive form to input:
Air Temperature (K)
Process Temperature (K)
Rotational Speed (RPM)
Torque (Nm)
Tool Wear (min)
Sent data via fetch() API request to Flask backend.
Displayed real-time result:
- No Maintenance Needed
- Maintenance Required
5. Threshold Integration:
Added custom thresholding: if Air Temp > 315K OR Torque > 60Nm → Maintenance alert.
Combined ML + Rule-Based approach for higher accuracy.


-Results
1.Real-Time Monitoring of machine health using IoT sensors.
2.Accurate Predictions for preventive maintenance using ML.
3.User-Friendly Dashboard with responsive design.
4.Demonstrates cross-domain expertise in IoT + ML + Web Dev.


