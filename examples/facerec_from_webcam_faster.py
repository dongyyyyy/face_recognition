import face_recognition
import cv2
import numpy as np
import os
import time
import pymysql
from datetime import datetime
import requests
import json
import csv
from examples.make_csvfile import *
# conn = pymysql.connect(host='localhost',port=3306,user='root',password='pass',db='face_recognition',charset='utf8')
# curs = conn.cursor()

known_face_encodings = []
known_face_names = []
known_face_names_check = dict()


def encoding_func():
    global known_face_encodings
    global known_face_names
    global known_face_names_check
    # Load a sample picture and learn how to recognize it.
    train_path = 'C:/Users/dongyoung/Desktop/Git/face_recognition_project/examples/knn_examples/train/'
    file_list = os.listdir(train_path)
    # image_list = []
    known_face_encodings = []
    known_face_names = []
    known_face_names_check = dict()
    # for filename in file_list:
    #     current_filename_list = os.listdir(train_path+filename)
    #     for current_filename in current_filename_list:
    #         now_filename = train_path + f'{filename}/' +current_filename
    #         print('filename : ',now_filename)
    #         known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file(now_filename))[0])
    #
    #         known_face_names.append(filename)
    #         known_face_names_check[filename] = 0
    known_face_encodings, known_face_names, known_face_names_check = read_csv_encodingVector()
            # print(f'known_face_names == {known_face_names}')
    # exit(1)
    print(f'known_face_names == {known_face_names}')

# from examples.face_recognition_knn import train
# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)


encoding_func()
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    _,origin = video_capture.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        if name in known_face_names_check:
            known_face_names_check[name] += 1
            if known_face_names_check[name] >= 20:
                check_checks = f"select ct from checks where name='{name}'"
                params = {
                    "select": check_checks,
                    'name' : name,
                }
                # print(f'parmas ==> {params}')
                res = requests.post("http://210.115.230.164:80/select_checks", data=json.dumps(params))
                known_face_names_check[name] = 0
                print(res.text)
    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('i'):
        name = input('Input New Person Name : ')
        print('name : ',name)
        # SQL문 실행
        sql = f"select * from user where name='{name}'"
        params = {
            "select": sql,
            'name': name,
        }

        res = requests.post("http://210.115.230.164:80/new_picture", data=json.dumps(params))
        print('res ==> ',res.text)
        if int(res.text) > 0:
            path = f'C:/Users/dongyoung/Desktop/Git/face_recognition/examples/knn_examples/train/{name}/'
            os.makedirs(path, exist_ok=True)
            now = time.localtime()
            filename = f'{name}_%04d_%02d_%02d_%02d_%02d_%02d.jpg' % (
                now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
            cv2.imwrite(path + filename, origin)
            time.sleep(1)
            print(path + filename)
            save_csv_encodingVector()
            encoding_func()
        else:
            print('등록되지 않은 사람입니다.')




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
