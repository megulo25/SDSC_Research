from flask import Flask, jsonify, render_template
from sqlalchemy import create_engine

# Create an app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)