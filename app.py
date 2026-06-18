from flask import Flask, render_template, request
from summarizer import summarize

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    summary = ""

    if request.method == "POST":

        article = request.form["article"]

        if article.strip():

            summary = summarize(article)

    return render_template(
        "index.html",
        summary=summary
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)