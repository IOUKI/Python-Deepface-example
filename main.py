from deepface import DeepFace
import threading
import cv2

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("無法打開攝像頭")
else:
    print("攝像頭已經打開")

counter = 0

faceMatch = False

referenceImg = cv2.imread('face.jpg')

# 檢測是否符合特定臉部圖片
def checkFace(frame):
    global faceMatch
    try:
        if DeepFace.verify(frame, referenceImg, model_name='Facenet')['verified']:
            faceMatch = True
        else:
            faceMatch = False

    except ValueError:
        print('value error')
        faceMatch = False

while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=checkFace, args=(frame,)).start()

            except ValueError:
                pass

        counter += 1

        # 顯示判斷結果
        if faceMatch:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video show", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()