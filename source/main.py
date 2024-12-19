from flask import Flask, request,render_template

app = Flask(__name__,
            template_folder="assets",
            static_folder="assets")

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="localhost",port=5000)