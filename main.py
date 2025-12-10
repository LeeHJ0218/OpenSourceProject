import cv2
import numpy as np

# 이미지 경로 설정
IMAGE_PATH = "images/demo.jpg"   # <-- 이미지 파일 이름 바꾸면 여기만 수정하면 됨

# 이미지 불러오기
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("이미지를 찾을 수 없습니다. 경로를 확인하세요.")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# 윤곽선 검출
edged = cv2.Canny(blur, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 가장 큰 물체 하나만 사용
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# 물체의 경계 박스 구하기
for c in contours:
    if cv2.contourArea(c) < 100:
        continue

    x, y, w, h = cv2.boundingRect(c)

    print(f"물체의 가로 너비: {w} px")
    print(f"물체의 세로 높이: {h} px")

    # 박스 그리기
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    break

# 결과 출력 (이미지가 너무 크면 화면에 맞게 자동 축소)
max_width = 800  # 원하는 가로 크기
scale = max_width / image.shape[1]
resized = cv2.resize(image, None, fx=scale, fy=scale)

cv2.imshow("Measured Image", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

