from flask import Flask, request, jsonify
import pandas as pd
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai import Credentials
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 


# IBM Cloud Credentials
creds = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key="hkBYvuD79tNMg0BJH6Gc92PhJlijUQC779gauY7EceY_"
)
project_id = "44c89f17-ab07-4d83-a597-8b9c19e3b3d9"

model = Model(
    model_id="google/flan-t5-xl",
    params={"decoding_method": "greedy", "max_new_tokens": 150},
    credentials=creds,
    project_id=project_id
)

# Load course data
df = pd.read_csv("courses.csv")

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json.get("user_input", "")
    keywords = user_input.lower().split()
    matches = df[df['Tags'].str.contains('|'.join(keywords), case=False, na=False)]

    if matches.empty:
        return jsonify({"response": "Sorry, I couldn't find any matching courses."})

    course_list = '\n'.join([
        f"{row['Title']} ({row['Domain']}, {row['Level']})"
        for _, row in matches.iterrows()
    ])

    prompt = f"""
You are a helpful course advisor.

Based on the following courses:
{course_list}

A user says: "{user_input}"

Suggest the top 2 best-fit courses with short reasons.
"""

    response = model.generate(prompt)
    return jsonify({"response": response['results'][0]['generated_text']})

if __name__ == '__main__':
    app.run()
