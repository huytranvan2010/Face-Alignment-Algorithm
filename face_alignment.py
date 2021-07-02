from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import cv2
import imutils
import dlib

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--shape_predictor", required=True, help="path to facial landmark predictor")     # dlib’s pre-trained facial landmark detector (phát hiện 68 landmarks)
parser.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(parser.parse_args())

# khởi tạo dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()   # dựa trên HOG + Linear SVM tìm face, xem thêm bài face recognition

# Tạo the facial landmerk predictor
predictor = dlib.shape_predictor(args["shape_predictor"])

# Tạo face aligner
fa = FaceAligner(predictor, desiredFaceWidth=256)

# Vẫn phải detect được khuôn mặt trước khi tìm facial landmarks
# load ảnh, resize, convert to gray (cần cho HOG)
image = cv2.imread(args["image"])
image = imutils.resize(image, width=800)    # giữ nguyên aspect ratio
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   # dùng cho HOG detector bên dưới

# detect faces in the grayscale image
# nhận 2 tham số ảnh (có thể ảnh màu), 2-nd parameter - số image pyramids tạo ra trước khi detect faces (upsample)
# nó giúp phóng ảnh lên để có thể phát hiện khuôn mặt nhỏ hơn, dùng thì chạy lâu hơn
cv2.imshow("Input", image)
rects = detector(gray, 2)   # trả về list các rectangle chứa khuôn mặt (left, top, right, bottom) <=> (xmin, ymin, xmax, ymax)

# duyệt qua các detections
for rect in rects:
    # Chuyển dlib's rectange (left, top, right, botttom) = (xmin, ymin, xmax, ymax) to OpenCV-style bounding box (xmin, ymin, w, h)
    # Dễ dàng chuyển được 
    (x, y, w, h) = rect_to_bb(rect)
    
    # trích xuất face ROI gốc
    face_Orig = imutils.resize(image[y:y+h, x:x+w], width=256)

    # align the face bằng cách sử dụng facial landmarks
    face_Aligned = fa.align(image, gray, rect)

    # hiển thị output images
    cv2.imshow("Original", face_Orig)
    cv2.imshow("Aligned", face_Aligned)
    cv2.waitKey(0)