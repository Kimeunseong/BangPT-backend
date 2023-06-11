from flask import *

app = Flask(__name__)
Video_show_bp = Blueprint('video_show', __name__) # 따로 선언하기

# 여기 부분 반드시 수정하기.
@Video_show_bp.route('/video_show1', methods=['GET'])
def send_video1():
    video_path = f'static/video_split/output_segment_1.mp4'
    return send_file(video_path, mimetype='video/mp4')



@Video_show_bp.route('/video_show2', methods=['GET'])
def send_video2():
    video_path = f'static/video_split/output_segment_4.mp4'
    return send_file(video_path, mimetype='video/mp4')

