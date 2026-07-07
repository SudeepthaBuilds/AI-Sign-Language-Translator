import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3

# ---------------------------
# Load Model
# ---------------------------
model = tf.keras.models.load_model("models/final_model.keras")

# Labels
labels = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'del','nothing','space'
]

# ---------------------------
# Text To Speech
# ---------------------------
engine = pyttsx3.init()

# ---------------------------
# MediaPipe
# ---------------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

# ---------------------------
# Webcam
# ---------------------------
cap = cv2.VideoCapture(0)

sentence = ""

previous_letter = ""

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            h,w,c = frame.shape

            x=[]
            y=[]

            for lm in hand.landmark:

                x.append(int(lm.x*w))
                y.append(int(lm.y*h))

            xmin=max(min(x)-20,0)
            ymin=max(min(y)-20,0)
            xmax=min(max(x)+20,w)
            ymax=min(max(y)+20,h)

            crop=frame[ymin:ymax,xmin:xmax]

            if crop.size!=0:

                img=cv2.resize(crop,(224,224))

                img=img.astype("float32")/255.0

                img=np.expand_dims(img,axis=0)

                prediction = model.predict(img, verbose=0)

                print(prediction)

                index = np.argmax(prediction)

                confidence = float(np.max(prediction))

                letter = labels[index]

                print(f"Predicted: {letter}, Confidence: {confidence:.4f}")
                
                if confidence>0.90:

                    if letter!=previous_letter:

                        previous_letter=letter

                        if letter=="space":
                            sentence+=" "

                        elif letter=="del":
                            sentence=sentence[:-1]

                        elif letter!="nothing":
                            sentence+=letter

                cv2.putText(frame,
                            f"{letter} {confidence:.2f}",
                            (xmin,ymin-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            2)

                cv2.rectangle(frame,
                              (xmin,ymin),
                              (xmax,ymax),
                              (255,0,0),
                              2)

            mpDraw.draw_landmarks(frame,
                                  hand,
                                  mpHands.HAND_CONNECTIONS)

    cv2.rectangle(frame,(0,420),(640,480),(0,0,0),-1)

    cv2.putText(frame,
                sentence,
                (20,460),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,255),
                2)

    cv2.imshow("AI Sign Language Translator",frame)

    key=cv2.waitKey(1)

    if key==ord('s'):

        engine.say(sentence)

        engine.runAndWait()

    if key==ord('c'):

        sentence=""

    if key==27:

        break

cap.release()

cv2.destroyAllWindows()