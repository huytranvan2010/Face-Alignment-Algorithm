import cv2 
import numpy as np
import argparse
import imutils

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--face", required=True, help="path to the face cascade")
parser.add_argument("-e", "--eye", required=True, help="path to the eye cascade")
parser.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(parser.parse_args())

# tạo face cascade và eye cascade 
# cascasde này nhẹ nhưng mặt nghiêng cái là không phát hiện được
face_cascade = cv2.CascadeClassifier(args["face"])
eye_cascade = cv2.CascadeClassifier(args["eye"])

""" load image, convert to grayscale """
image = cv2.imread(args["image"])
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # cần đưa vào cascade
cv2.imshow("original", image)
cv2.waitKey(0)

""" Phát hiện khuôn mặt, trả về list of tuples (x, y, w, h) các khuôn mặt nếu có """
rects_face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
# Vẽ rectangle quang khuôn mặt
for (x, y, w, h) in rects_face:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
cv2.imshow("face", image)
cv2.waitKey(0)

""" 
    Phát hiện khuôn mặt xong đến phát hiện mắt 
    Trong phần này chúng ta tách riêng khuôn mặt ra khỏi ảnh, rồi sử dụng để phát hiện eyes
"""
# Lấy face ROI để phát hiện khuôn mặt trên đó
face_ROI_gray = gray[y:y+h, x:x+w]      # để phát hiện eye trong cascade
face_ROI_color = original[y:y+h, x:x+w]    # để vẽ rectangle

rects_eye = eye_cascade.detectMultiScale(face_ROI_gray, scaleFactor=1.1, minNeighbors=4)
index = 0   # để phân chia hai mắt
for (ex, ey, ew, eh) in rects_eye:
    if index == 0:
        eye_1 = (ex, ey, ew, eh)
    else:
        eye_2 = (ex, ey, ew, eh)
    
    # vẽ hai mắt
    cv2.rectangle(face_ROI_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    index = 1
cv2.imshow("eyes", face_ROI_color)
cv2.waitKey(0)

""" Xác định mắt trái, phải """
if eye_1[0] < eye_2[0]:     # tọa độ x nhỏ hơn là mắt bên trái (từ phía người quan sát)
    left_eye = eye_1
    right_eye = eye_2
else:
    left_eye = eye_2
    right_eye = eye_1

""" Xác định tâm hai mắt và vẽ đường thẳng qua đó """
center_left_eye = (int(left_eye[0] + left_eye[2] / 2), int(left_eye[1] + left_eye[3] / 2))
(x_left_eye, y_left_eye) = center_left_eye

center_right_eye = (int(right_eye[0] + right_eye[2] / 2), int(right_eye[1] + right_eye[3] / 2))
(x_right_eye, y_right_eye) = center_right_eye

cv2.circle(face_ROI_color, center_left_eye, 4, (255, 0, 0), -1)
cv2.circle(face_ROI_color, center_right_eye, 4, (255, 0, 0), -1)
cv2.line(face_ROI_color, center_left_eye, center_right_eye, (0, 220, 220), 2)
cv2.imshow("Line between the eyes", face_ROI_color)
cv2.waitKey(0)

""" 
    Vẽ đường nằm ngang và xác định góc giữa đường đó và đường nối 2 mắt để phục vụ xoay ảnh
"""
if y_left_eye > y_right_eye:    # mắt trái thấp hơn mắt phải
    point_A = (x_right_eye, y_left_eye)
    # Ảnh phải xoay theo chiều kim đồng hộ clockwise direction
    direction = -1
else: 
    point_A = (x_left_eye, y_right_eye)
    # Ảnh cần xoay ngược chiều kim đồng hồ counter clockwise direction
    direction = 1
 
cv2.circle(face_ROI_color, point_A, 4, (255, 0, 0), -1)

cv2.line(face_ROI_color, point_A, center_right_eye, (0, 220, 220), 2)
cv2.line(face_ROI_color, point_A, center_left_eye, (0, 220, 220), 2)
cv2.imshow("Lines", face_ROI_color)
cv2.waitKey(0)

""" Tính góc để chuyển, ở đây mình chia luôn 2 trường hợp cho dễ""" 
dist_A_righteye = np.linalg.norm(np.array(point_A) - np.array(center_right_eye))
dist_A_lefteye = np.linalg.norm(np.array(point_A) - np.array(center_left_eye))
if direction == -1:     # quay theo chiều kim đồng hồ clockwise direction
    angle = np.arctan(dist_A_righteye / dist_A_lefteye)
    angle = - angle     # quay theo chiều kim đồng hồ nên góc < 0
else:   # quay theo chiều ngược kim đồng hồ counter clockwise direction
    angle = np.arctan(dist_A_lefteye / dist_A_righteye)

# chuyển về degree
angle = (angle * 180.) / np.pi 
print(angle)
print(direction)

""" Xoay ảnh """
# Tâm của face ROI
(h, w) = face_ROI_color.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_face_ROI = cv2.warpAffine(face_ROI_color, M, (w, h))    # có thể dùng thư viện imutils cho nhanh

cv2.imshow("rotated", rotated_face_ROI)
cv2.waitKey(0)


(h, w) = original.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_face_ROI = cv2.warpAffine(original, M, (w, h))    # có thể dùng thư viện imutils cho nhanh

cv2.imshow("rotated", rotated_face_ROI)
cv2.waitKey(0)
