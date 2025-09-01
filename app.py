from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
import json


#import custom function

from pgvdemo import text_summary

app = Flask(__name__) 
CORS(app)

@app.route('/bot', methods=['POST'])
def summarize_text():
    text = request.json['text']
    isNew = request.json['isNew']
    conversation_history = request.json.get('conversationHistory', [])  # Get conversation history if provided
    
    # Pass conversation history to the text_summary function
    response = text_summary(text, isNew, conversation_history)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)