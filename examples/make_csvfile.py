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

def save_csv_encodingVector():
    train_path = 'C:/Users/dongyoung/Desktop/Git/face_recognition_project/examples/knn_examples/train/'
    file_list = os.listdir(train_path)
    # image_list = []
    csv_path = 'C:/Users/dongyoung/Desktop/Git/face_recognition_project/examples/encoding_csv.csv'
    f = open(csv_path,'w',newline='')
    wr = csv.writer(f)
    for filename in file_list:
        current_filename_list = os.listdir(train_path + filename)
        for current_filename in current_filename_list:
            now_filename = train_path + f'{filename}/' + current_filename
            print('filename : ', now_filename)
            vector = face_recognition.face_encodings(face_recognition.load_image_file(now_filename))[0]
            wr.writerow([filename,vector])

    f.close()


def read_csv_encodingVector():
    csv_path = 'C:/Users/dongyoung/Desktop/Git/face_recognition_project/examples/encoding_csv.csv'
    f = open(csv_path, 'r')
    lines = csv.reader(f)
    known_face_encodings = []
    known_face_names = []
    known_face_names_check = dict()
    for line in lines:
        name = line[0]
        vector = line[1]
        # print(f'name = {name} // vector = {np.array(vector[1:-1].split()).astype(np.float).shape}')
        known_face_encodings.append(np.array(vector[1:-1].split()).astype(np.float))
        known_face_names.append(name)
        known_face_names_check[name] = 0
    return known_face_encodings, known_face_names,known_face_names_check

