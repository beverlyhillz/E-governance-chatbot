from main_bot import *
from dialogue_manager import *
from flask import Flask, render_template, request
app = Flask(__name__)

simple_manager = DialogueManager(RESOURCE_PATH)
simple_manager.create_chitchat_bot()
bot = BotHandler(simple_manager)
inp="Heelo"
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/get")
def get_bot_response():
    print("fjafa")
    userText = request.args.get('msg')
    return str(simple_manager.generate_answer(userText))

if __name__ == "__main__":
    app.run(debug='true')
