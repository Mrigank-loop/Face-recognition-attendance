import cv2
import mediapipe as mp
import pandas as pd
from datetime import datetime
import os

CSV_FILE = "Attendance.csv"

if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["Name", "Time"]).to_csv(
        CSV_FILE,
        index=False
    )

name = input("Mrigank ")

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

attendance_marked = False

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_detection.process(rgb)

    if results.detections:

        for detection in results.detections:

            bbox = detection.location_data.relative_bounding_box

            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            cv2.rectangle(
                frame,
                (x, y),
                (x + bw, y + bh),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            if not attendance_marked:

                now = datetime.now()

                time_now = now.strftime("%H:%M:%S")

                df = pd.read_csv(CSV_FILE)

                if name not in df["Name"].values:

                    new_row = pd.DataFrame(
                        [[name, time_now]],
                        columns=["Name", "Time"]
                    )

                    df = pd.concat(
                        [df, new_row],
                        ignore_index=True
                    )

                    df.to_csv(
                        CSV_FILE,
                        index=False
                    )

                attendance_marked = True

    cv2.imshow(
        "Face Attendance System",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()