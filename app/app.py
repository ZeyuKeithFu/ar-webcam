from flask import Flask, render_template, request, g, Response
from ar.camera import ArCamera

app = Flask(__name__)


@app.route('/')
def index():
    """
    Homepage
    """
    return render_template('base.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/ar')
def ar_webcam():
    return Response(gen(ArCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
