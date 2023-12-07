import cv2
import streamlit as st
from ultralytics import YOLO


cap= cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)


def captureRealtimefeed():

    model=YOLO('best.pt')
    classNames = ['hat', 'no hat', 'no vest', 'vest']
    stframe = st.empty()
    while True:
        _,frame =cap.read()
        results = model.predict(frame, stream=True, verbose=False,conf=0.67)
        for result in results:
            boxes = result.boxes
            for box in boxes:

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])

                if cls==0 or cls==3:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(frame, classNames[cls], [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 2)

                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, classNames[cls], [x1, y1], cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255), 2)


        print(results)
        stframe.image(frame, channels="BGR")

st.title("Realtime Feed")
captureRealtimefeed()
