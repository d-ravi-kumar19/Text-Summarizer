from flask import Flask, render_template, request, jsonify
from textsummerizer import generate_summary

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data["text"]
    num_sentences = int(data["num_sentences"])
    
    # Call your summarizer function with the text
    summary = generate_summary(text,num_sentences)  # Change parameters accordingly
    
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(debug=True)
