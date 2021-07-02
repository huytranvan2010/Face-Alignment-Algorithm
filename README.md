# Face-Alignment
Face alignment là quá trình sắp đặt khuôn mặt sao cho nó thẳng đứng trong ảnh. Face alignment thường được thực hiện như bước tiền xử lý cho các thuật toán nhận diện khuôn mặt. Để thực hiện việc này cần trải qua 2 bước:
* Xác định cấu trúc hình học của khuôn mặt trong ảnh
* Thực hiện face alignment thông qua các pháp biến đổi như translation (dịch chuyển), scale, rotation.

Có một số phương pháp để thực hiện face alignment như sử dụng pre-trained 3D model sau đó chuyển ảnh đầu vào sao cho các landmarks trên khuôn mặt ban đầu khớp với landmarks trên 3D model... Trong bài này chúng ta sẽ thực hiện face alignment dựa trên facial landmarks.

Nhiệm vụ chính của chúng ta trong face alignment là:
* Tất cả khuôn mặt phải ở tâm của bức ảnh
* Tất cả khuôn mặt được xoay sao cho các mắt nằm trên đường nằm ngang (có cùng tọa độ y)
* Được scale sao cho các khuôn mặt có tỉ lệ tương đương

Dưới đây là chi tiết các bước thực hiện:
* Phát hiện khuôn mặt và mắt trong ảnh (qua ficial landmarks)
* Tính tâm của hai mắt (dựa trên tọa đọ facial landmarks)
* Vẽ đường nối tâm của hai mắt
* Vẽ đường nằm ngang giữa hai mắt
* Tính độ dài 2 canh của tam giác
* Tính góc
* Xoay ảnh
* Scale ảnh



http://datahacker.rs/010-how-to-align-faces-with-opencv-in-python/

https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/ 
