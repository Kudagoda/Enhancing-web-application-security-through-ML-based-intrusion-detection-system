from flask import Flask, jsonify, request, render_template
import numpy as np
import joblib
import pandas as pd
import logging
import flask
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logging.basicConfig(filename='Run_Log.log', level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')

app = Flask(__name__)

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        traffic = pd.read_csv("Attack1.csv")  # Read CSV directly from the file

        # Check if the CSV file has the expected number of features (columns)
        if traffic.shape[1] != 79:
            return "Invalid Input: The CSV has the wrong number of features. Input should hold 79 features."

        bool_val = traffic.isnull().any().any()
        if bool_val:
            logging.error('Some column values are left empty')
            return flask.render_template('empty_columns.html')

        feature_count = traffic.columns
        size = len(feature_count)

        if size < 79:
            logging.error('This file has fewer features in it')
            return flask.render_template('less_feat.html')

        value = traffic['Timestamp'].isin([0]).any().any()
        if value:
            logging.error('The Timestamp is zero')
            return flask.render_template('timestamp_zero.html')

        # Data PreProcessing
        traffic['Timestamp'] = pd.to_datetime(traffic['Timestamp']).astype(np.int64)
        columns = traffic.columns

        for i in columns:
            traffic[i] = traffic[i].astype(float)

        # Dropping the least important features (analyzed via EDA)
        traffic.drop(['Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 
                    'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg'], axis=1, inplace=True)

        traffic = traffic.drop_duplicates(keep="first")
        traffic['Flow Byts/s'] = traffic['Flow Byts/s'].replace([np.inf, -np.inf], np.nan)
        traffic['Flow Pkts/s'] = traffic['Flow Pkts/s'].replace([np.inf, -np.inf], np.nan)
        traffic = traffic.replace([np.inf, -np.inf], np.nan)
        traffic = traffic.replace(np.nan, 0)
        columns = traffic.columns

        # Feature Reduction
        perm_imp = joblib.load('perm_imp')  # Loading the perm imp model
        coefficients = perm_imp.feature_importances_
        absCoefficients = abs(coefficients)
        Perm_imp_features = pd.concat((pd.DataFrame(columns, columns=['Variable']),
                                    pd.DataFrame(absCoefficients, columns=['absCoefficient'])),
                                    axis=1).sort_values(by='absCoefficient', ascending=False)
        least_features = Perm_imp_features.iloc[50:, 0]

        # Dropping the least important features identified via PermutationImportance
        data = least_features.tolist()
        for i in data:
            traffic.drop(labels=[i], axis=1, inplace=True)

        # Load RandomForestClassifier model
        model = joblib.load('models/Decision_Tree_Classifier_Model')  # Load RandomForestClassifier model
        pred = model.predict(traffic)

        if pred.any() > 0:
            pred_result = 'Not Malware'
            send_email(pred_result)  # Send email with prediction result
            return jsonify({'prediction': pred_result})
        else:
            # Additional information about the attack when it's detected as malicious
            is_malicious = pred.any() == 1
            attack_info = {
                'prediction': 'Not Malware' if is_malicious else 'Malware',
                'attack_type': 'DDoS & DoS Attack' if is_malicious else 'DDoS & DoS  Attack',
                'severity': 'High' if is_malicious else 'Low',
                'description': 'This is a DDoS & DoS Attack.' if is_malicious else 'This is a malicious request.',
                'feature_values': traffic.iloc[0, :20].to_dict(),
                'used_features': list(traffic.columns)
            }
            response = app.response_class(
                response=json.dumps(attack_info, indent=4),  # Format the JSON with indentation
                status=200,
                mimetype='application/json'
            )
            return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8070, debug=True)
