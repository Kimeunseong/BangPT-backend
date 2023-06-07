from datetime import datetime
from pymongo import MongoClient

import os
import time
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import tensorflow as tf


#--------------------------------------------------------------------------------------------------------------------#
client = MongoClient("mongodb://localhost:27017/")
db = client.capstone_design
Exercise_result = db['exercise_result'] # 유저 정보 관리하는 컬렉션
Users = db['users'] # 유저의 배지 관리 
# Exercise_result_bp = Blueprint('user', __name__)
User_Badges = db['user_badge'] # 유저의 배지 관리 
Badge = db['badges'] # 배지 리스트
#--------------------------------------------------------------------------------------------------------------------#


from flask import *

Model_bp = Blueprint('model', __name__) # 따로 선언하기

@Model_bp.route('/model', methods=['POST'])
def model():

    user_id = request.form['user_id'] # (추가)


    # 1. full video를 5개로 분할하여 경로(split_video_path)에 저장한다.
    def video_split(input_file, output_prefix, segment_count):
        # 동영상 파일 열기
        cap = cv2.VideoCapture(input_file)
        # 동영상 속성 가져오기
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 세그먼트 길이 계산
        segment_length_frames = total_frames // segment_count
        # 저장할 세그먼트 번호 초기화
        segment_number = 1
        frame_count = 0
        while segment_number <= segment_count:
            # 동영상의 현재 프레임 설정
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            # 세그먼트 파일 이름 설정
            output_file = f"{output_prefix}_{segment_number}.mp4"
            # 세그먼트 파일 생성
            out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
            # 세그먼트 길이만큼 프레임 저장
            for i in range(segment_length_frames):
                # 동영상의 현재 프레임 읽기
                ret, frame = cap.read()
                # 프레임 저장
                out.write(frame)
            # 세그먼트 파일 닫기
            out.release()
            # print(f'segment_count: {segment_count}') # 확인용 넘버링
            # 다음 세그먼트 번호로 이동
            segment_number += 1
            # 다음 세그먼트의 시작 프레임 설정
            frame_count += segment_length_frames
        # 동영상 파일 닫기
        cap.release()


    # 2. 각 동영상을 15장의 이미지셋으로 변환하여 어레이로 저장한다.
    def video_to_images(N, video_PATH) :
        # 1. 동영상의 모든 프레임을 리스트(temp_list)에 저장하기
        temp_list = []
        video = cv2.VideoCapture(video_PATH)
        fps = video.get(cv2.CAP_PROP_FPS)
        while(video.isOpened()):
            ret, frame = video.read()
            if not ret:
                break
            if ret:
                temp_list.append(frame)
        video.release()
        # 2. 건너 뛸 간격(skip)을 계산해서 넘파이 어레이로 저장하기
        images_list = []
        skip = len(temp_list) / N
        cnt = 0
        for i in range(len(temp_list)):
            if i == np.floor(skip*cnt):
                images_list.append(temp_list[i])
                cnt += 1
        images_array = np.array(images_list)
        
        return images_array
    
    
    # 3. 이미지에서 스켈레톤 좌표를 추출하여 df로 저장한다.(df00, df01, df02, df03, df04, shape = (15, 22))
    def img_to_skeleton(img_array):
        # 모듈 로드
        mp_pose = mp.solutions.pose
        # 이미지에서 프레임을 읽어온다.
        image = img_array
        # HumanPose 모듈을 사용하여 스켈레톤을 추출한다.
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            # 추출된 스켈레톤을 저장할 리스트를 생성한다.
            landmarks_list = []
            # 스켈레톤을 추출한다.
            results = pose.process(image)
            landmark = results.pose_landmarks.landmark
            if len(landmark) == 33: # 33보다 적거나 많다는 것은 잘못된 추출이다.
                # 필요한 관절 좌표만 선별적으로 저장한다.
                need = [0, 11, 12, 23, 24, 25, 26, 27, 28, 31, 32]
                for i in need:
                    # 스켈레톤의 좌표를 저장한다.
                    x = round(landmark[i].x, 5)
                    y = round(landmark[i].y, 5)
                    landmarks_list.append((x, y)) # x,y좌표만 저장한다.    
            else:
                return 'There is a problem.'
        return landmarks_list


    # 6. mse 계산 함수
    # mse 계산
    def mse_loss(y_true, y_pred):
        err = y_true - y_pred
        loss = tf.math.reduce_mean(tf.math.square(err))
        return loss


    # 7. 점수로 환산
    def convert_to_score(score):
        score_clean = [float(s.strip().rstrip(',')) for s in score]
        score_100 = []
        for i in score_clean:
            score_100.append((100 - i*1000)-25)
        return score_100


    # 8. best, worst reps 계산
    def find_max_min_index(lst):
        min_value = min(lst)
        max_value = max(lst)
        min_index = lst.index(min_value)
        max_index = lst.index(max_value)
        return min_index, max_index


    # 9. 한 문장 피드백 제공
    def oneLineFeedback(tot_score):
        if 96 <= tot_score:
            return '대단해요! 거의 트레이너 수준입니다! 운동 자세가 매우 정확하고 균형도 잘 잡혀 있습니다. 전문가 수준의 자세를 유지하고 있으며, 안정적이고 효과적인 운동을 하고 있습니다.'
        elif 75 <= tot_score <= 95:
            return '멋져요! 운동 자세가 상당히 훌륭합니다. 자세의 안정성과 균형을 유지하며, 기본적인 운동 동작을 정확하게 수행하고 있습니다.'
        elif 65 <= tot_score <= 74:
            return '잘 하셨어요! 운동 자세가 괜찮습니다. 기본적인 운동 동작을 어느 정도 정확하게 수행하고 있으며, 자세의 안정성도 일정 수준을 유지하고 있습니다. 조금 더 정확한 동작을 위해 세심한 주의가 필요합니다.'
        elif 50 <= tot_score <= 64:
            return '아쉬워요! 기본적인 운동 동작을 대체로 잘 수행하고 있지만, 몇몇 부분에서 개선이 필요합니다. 조금 더 정확하고 자세한 동작을 위해 주의와 연습이 더욱 필요합니다.'
        else:
            return '앗, 야생의 헬린이가 나타났다! 운동 자세가 아직 미흡한 부분이 많습니다. 자세의 안정성과 정확성을 향상시키기 위해 더 많은 연습과 주의가 필요합니다.'
        
    # ============================================================================================================



    input_video_path = '/Users/estar-kim/Desktop/2023/mju/캡스톤디자인/flask/bangpt-flask_0604/static/video/video.mp4' # 인풋 동영상 경로
    split_video_path = '/Users/estar-kim/Desktop/2023/mju/캡스톤디자인/flask/bangpt-flask_0604/static/video_split/output_segment'


    # 1. full video를 5개로 분할하여 경로(split_video_path)에 저장한다.
    video_split(input_video_path, split_video_path, 5)
    print('1번 완료')

    # 2. 각 동영상을 15장의 이미지셋으로 변환하여 어레이로 저장한다.(images_list00, images_list01, ..., images_list04)
    images_list = []
    for i in range(5):
        video_path = f'{split_video_path}_{i+1}.mp4'
        images = video_to_images(15, video_path)
        images_list.append(images)
    print('2번 완료')

    # 3. 이미지에서 스켈레톤 좌표를 추출하여 df로 저장한다.(df00, df01, df02, df03, df04, shape = (15, 22))
    result_dict = {}
    for n in range(5):
        total_skeleton = []
        for j in range(15):
            temp_img = img_to_skeleton(images_list[n][j])
            total_skeleton.append(temp_img)  # 여기서 스켈레톤 추출
        df = pd.DataFrame()
        for i in range(len(total_skeleton)):
            temp = np.array(total_skeleton[i])
            temp = temp.flatten()
            df[i] = temp
        result_dict[f'df{n}'] = df.T

    print('3번 완료')

    # 4. 모델 인풋에 맞게 shape을 조정한다. inputs shape = (5,22), outputs shape = (5, 308)
    def shape_control(inputs_dict):
        arrX = []
        arrY = []
        for df_key in inputs_dict:
            df = inputs_dict[df_key]
            tempX = df.iloc[0]
            tempY = df.iloc[1:]
            tempX = np.array(tempX)
            tempY = np.array(tempY)
            tempY = tempY.flatten()
            arrX.append(tempX)
            arrY.append(tempY)
        inputs = np.array(arrX)
        outputs = np.array(arrY)
        return inputs, outputs

    inputs_dict = result_dict
    inputs, outputs = shape_control(inputs_dict)
    print('4번 완료')

    # 5. 모델 예측 실행하여 pred에 저장한다. pred.shape = (5, 308)
    model = load_model('/Users/estar-kim/Desktop/2023/mju/캡스톤디자인/model/learned_models/model_0520_132_ver1.h5')
    pred = model.predict(inputs)
    print('5번 완료')

    # 6. pred와 ture를 비교하여 mse를 계산한다.
    lst_loss = []
    for i in range(len(pred)):
        loss = str(mse_loss(outputs[i], pred[i]))[10:30]
        lst_loss.append(loss) 
    print('6번 완료')

    # 7. 5회의 스쿼트에 대한 100점 만점 기준 점수와 최종 점수를 제공한다.
    score_100 = convert_to_score(lst_loss)
    score_100_total = sum(convert_to_score(lst_loss))//5
    print('7번 완료')

    # 8. 최고, 최저 랩스의 번호와 해당 동영상의 경로를 찾는다.
    worst_rep, best_rep = find_max_min_index(score_100)
    best_rep_video = f'{split_video_path}_{best_rep+1}.mp4'
    worst_rep_video = f'{split_video_path}_{worst_rep+1}.mp4'
    print('8번 완료')

    # 9. 운동에 대한 피드백을 보여준다.
    feedback = oneLineFeedback(score_100_total)
    print('9번 완료')

    # score_100 = [76.22372312913811, 56.58484131633456, 59.895442483959, 63.84220034202431, -51.834721821715306]
    # score_100_total = 40.0
    # best_rep, worst_rep = (0, 4)
    # feedback = '앗, 야생의 헬린이가 나타났다! 운동 자세가 아직 미흡한 부분이 많습니다. 자세의 안정성과 정확성을 향상시키기 위해 더 많은 연습과 주의가 필요합니다.'
    # best_rep_video = '/Users/estar-kim/Desktop/2023/mju/캡스톤디자인/flask/bangpt-flask_0604/static/video_split/output_segment_1.mp4'
    # worst_rep_video = '/Users/estar-kim/Desktop/2023/mju/캡스톤디자인/flask/bangpt-flask_0604/static/video_split/output_segment_5.mp4'




    # ++++++ 결과 확인 ++++++
    print(f'score_100 : {score_100}')
    print(f'score_100_total : {score_100_total}')
    print(f'best_rep, worst_rep : {best_rep, worst_rep}')
    print(f'best_rep_video : {best_rep_video}')
    print(f'worst_rep_video : {worst_rep_video}')
    print(f'feedback : {feedback}')


    # import shutil

    # b_copy_path = '/Users/estar-kim/Desktop/2023/mju/캡스톤디자인/AndroidStudio/bangPT-master_0601/app/src/main/res/raw/best_video.mp4'
    # shutil.copy(best_rep_video, b_copy_path)

    # w_copy_path = '/Users/estar-kim/Desktop/2023/mju/캡스톤디자인/AndroidStudio/bangPT-master_0601/app/src/main/res/raw/worst_video.mp4'
    # shutil.copy(worst_rep_video, w_copy_path)


# =======================================================================================
    num = Exercise_result.count_documents({'user_id' : user_id})
    now = datetime.now()
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

    result = {
        "num": num + 1,
        "user_id": user_id,  # Get userid from the request
        "score_100": score_100,
        "score_100_total": score_100_total,
        "best_rep": best_rep,
        "worst_rep": worst_rep,
        "feedback": feedback,
        "timestamp": timestamp
    }

    Exercise_result.insert_one(result)
    # Exercise_result = db['exercise_result'] # 유저 운동 결과 정보 관리하는 컬렉션
    # user_id1 = request.form['userID'] # (추가)
    user_exercise(user_id)
    

    return 'model predect successfully'


# =======================


# 배지 3~5번 (운동 결과 도큐먼트 추가할 때 이 함수 호출해야됨)
def user_exercise (user_id) :
    try:
        exercise = Exercise_result.count_documents({'user_id': user_id}) # user exercise result의 개수 계산
    
        # 운동 처음 마치고 나면 !!! 
        if exercise >= 1:
            badge3 = Badge.find_one({'num': 3})
            if badge3:
                # 중복 체크
                existing_badge = User_Badges.find_one({'user_id': user_id, 'badges.badge_num': badge3['num']})
                if not existing_badge:
                    user_badge = {
                        'badge_num': badge3['num'],
                        'badge_name': badge3['badge_name'],
                        'badge_desc': badge3['badge_desc'],
                        'badge_image_url' : badge3['badge_image_url'],
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    user_document = User_Badges.find_one({'user_id': user_id})
                    if user_document:
                        User_Badges.update_one(
                            {'user_id': user_id},
                            {'$push': {'badges': user_badge}}
                        )
                    else:
                        user_document = {
                            'user_id': user_id,
                            'badges': [user_badge]
                        }
                        User_Badges.insert_one(user_document)
                        
        if exercise >= 20:
            badge3 = Badge.find_one({'num': 4})
            if badge3:
                # 중복 체크
                existing_badge = User_Badges.find_one({'user_id': user_id, 'badges.badge_num': badge3['num']})
                if not existing_badge:
                    user_badge = {
                        'badge_num': badge3['num'],
                        'badge_name': badge3['badge_name'],
                        'badge_desc': badge3['badge_desc'],
                        'badge_image_url' : badge3['badge_image_url'],
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    user_document = User_Badges.find_one({'user_id': user_id})
                    if user_document:
                        User_Badges.update_one(
                            {'user_id': user_id},
                            {'$push': {'badges': user_badge}}
                        )
                    else:
                        user_document = {
                            'user_id': user_id,
                            'badges': [user_badge]
                        }
                        User_Badges.insert_one(user_document)
                        
        if exercise >= 100:
            badge3 = Badge.find_one({'num': 5})
            if badge3:
                # 중복 체크
                existing_badge = User_Badges.find_one({'user_id': user_id, 'badges.badge_num': badge3['num']})
                if not existing_badge:
                    user_badge = {
                        'badge_num': badge3['num'],
                        'badge_name': badge3['badge_name'],
                        'badge_desc': badge3['badge_desc'],
                        'badge_image_url' : badge3['badge_image_url'],
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    user_document = User_Badges.find_one({'user_id': user_id})
                    if user_document:
                        User_Badges.update_one(
                            {'user_id': user_id},
                            {'$push': {'badges': user_badge}}
                        )
                    else:
                        user_document = {
                            'user_id': user_id,
                            'badges': [user_badge]
                        }
                        User_Badges.insert_one(user_document)

    
    except Exception as e:
        print(e)
        return jsonify({'success': False})
    

@Model_bp.route('/get_video/<int:video_num>', methods=['GET'])
def get_video(video_num):

    # filepath = f'/Users/estar-kim/Desktop/2023/mju/캡스톤디자인/flask/bangpt-flask_0604/static/video_split/output_segment_{video_num+1}.mp4'
    filepath = f'static/video_split/output_segment_{video_num}.mp4'
    return send_file(filepath)