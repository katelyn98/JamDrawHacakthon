from flask import Flask, render_template
app = Flask(__name__, template_folder='C:/Users/bgooder/PycharmProjects/JamDraw/templates', static_folder='C:/Users/bgooder/PycharmProjects/JamDraw/static')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<songFile>')
def jam(songFile):
    return app.send_static_file(songFile)

if __name__ == '__main__':
    app.run()