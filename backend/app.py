import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)
limiter = Limiter(get_remote_address, app=app)

logging.basicConfig(level=logging.INFO)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# System prompt for responsible health information
SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a helpful AI assistant providing general health information. Always emphasize that your responses are for informational purposes only and are not a substitute for professional medical advice, diagnosis, or treatment. Encourage users to consult qualified healthcare professionals for personalized advice. Be empathetic, accurate, and promote healthy lifestyles."
}

# In-memory conversation history (for demo purposes; use database in production)
conversation_history = []

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

@app.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Add user message to history
        conversation_history.append({"role": "user", "content": user_message})
        
        # Prepare messages for OpenAI
        messages = [SYSTEM_PROMPT] + conversation_history
        
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        # Add AI response to history
        conversation_history.append({"role": "assistant", "content": ai_response})
        
        # Return response with disclaimer
        return jsonify({
            'response': ai_response,
            'disclaimer': 'This response is for informational purposes only and is not medical advice. Please consult a healthcare professional for personalized guidance.',
            'timestamp': request.json.get('timestamp')  # Optional client timestamp
        })
    
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500

@app.route('/clear-history', methods=['POST'])
def clear_history():
    global conversation_history
    conversation_history = []
    return jsonify({'status': 'Conversation history cleared'})

if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true', host='0.0.0.0', port=5000)