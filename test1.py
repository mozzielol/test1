from flask import Flask, Response
import cv2
class Camera(object):
	def __init__(self):
		self.cap = cv2.VideoCapture(0)
	def get_frame(self):
		_,img = self.cap.read()
		frame = cv2.imencode(’.jpg’,img)[1]
		return frame.tobytes()
	def __del__(self):
		self.cap.release()

app = Flask(__name__)

def gen(camera):
	while True:
	frame = camera.get_frame()
	yield (b’--frame\r\n’
		b’Content-Type: image/jpeg\r\n\r\n’ + frame + b’\r\n\r\n’)

@app.route(’/’)
def index():
	return Response(gen(Camera()),
		mimetype=’multipart/x-mixed-replace; boundary=frame’)

if __name__ == ’__main__’:
	#default port number is 5000
	app.run(host=’0.0.0.0’, debug=True)
