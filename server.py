from flask import Flask, request, jsonify
import openai
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from the frontend

# Set your OpenAI API Key
openai.api_key = "your_openai_api_key"

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_input = data.get("prompt", "")

    if not user_input:
        return jsonify({"error": "Prompt cannot be empty"}), 400

    # Measure start time
    start_time = time.time()

    try:
        # Call OpenAI's LLM API
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Adjust as needed
            messages=[{"role": "user", "content": user_input}]
        )

        # Measure end time
        end_time = time.time()
        time_taken = end_time - start_time

        return jsonify({
            "response": response["choices"][0]["message"]["content"],
            "time_taken": time_taken
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
