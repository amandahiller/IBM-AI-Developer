from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import request
import json

model_name = "facebook/blenderbot-400M-distill"

# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Keeping track of conversation history
conversation_history = []

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    # Read prompt from HTTP request body
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']

    #Encoding the conversation history
    history_string = "\n".join(conversation_history)

    #Tokenization of user prompt and chat history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")

    #Generate output from the model
    outputs = model.generate(**inputs, max_length= 60)  # max_length will acuse model to crash at some point as history grows

    #Decode output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    #Update conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)

    return response