from flask import Flask, request, jsonify, render_template
import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

def query_granite(prompt_text):
    api_key = os.getenv("IBM_API_KEY")
    project_id = os.getenv("IBM_PROJECT_ID")

    if not api_key or not project_id:
        return "Missing IBM credentials."

    iam_url = "https://iam.cloud.ibm.com/identity/token"
    iam_headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    iam_data = f"apikey={api_key}&grant_type=urn:ibm:params:oauth:grant-type:apikey"

    try:
        iam_response = requests.post(iam_url, headers=iam_headers, data=iam_data)
        iam_response.raise_for_status()
        access_token = iam_response.json()["access_token"]
    except Exception as e:
        return f"Error fetching access token: {e}"

    chat_url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}"
    }

    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text}
                ]
            }
        ],
        "project_id": project_id,
        "model_id": "ibm/granite-3-3-8b-instruct",
        "temperature": 0.7,
        "top_p": 1
    }

    try:
        response = requests.post(chat_url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Granite API call failed: {e}"

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)

            spike_info = None
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    diffs = df[col].diff().fillna(0)
                    spike_idx = diffs.abs().idxmax()
                    if abs(diffs.iloc[spike_idx]) > 50:
                        spike_info = {
                            "product": col,
                            "message": f"I found a spike in {col}'s data at row {spike_idx}. Want to explore why?"
                        }
                        break

            summary = {
                "columns": df.columns.tolist(),
                "row_count": len(df),
                "preview": df.head().to_dict()
            }

            prompt = f"You are an AI business strategist. Analyze this data:\n{summary['preview']}"
            insight = query_granite(prompt)

            chart_data = {
                "labels": df.index.astype(str).tolist(),
                "datasets": []
            }
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    chart_data["datasets"].append({
                        "label": col,
                        "data": df[col].tolist()
                    })

            return jsonify({
                "type": "sales_data",
                "summary": summary,
                "ai_insight": insight,
                "spike_info": spike_info,
                "chart_data": chart_data
            })

        elif file.filename.endswith('.txt'):
            text = file.read().decode('utf-8')
            prompt = f"Analyze this customer review and give 2-3 suggestions:\n{text[:500]}"
            insight = query_granite(prompt)
            return jsonify({"type": "review_data", "content": text[:500], "ai_insight": insight})

        else:
            return jsonify({"error": "Unsupported file type."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/explore-spike', methods=['POST'])
def explore_spike():
    data = request.get_json()
    product = data.get("product")
    reason = data.get("reason")

    if not product:
        return jsonify({"error": "Product not specified."}), 400

    prompt = f"There was a sudden spike in {product}. Explore possible reasons for this change and suggest actions."
    insight = query_granite(prompt)
    return jsonify({"insight": insight})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
