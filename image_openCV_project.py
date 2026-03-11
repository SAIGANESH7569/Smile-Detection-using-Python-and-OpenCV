import cv2

# ---- Load Haar cascades correctly ----
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)
#smile place lo "eye",hands,lowerbody ala chala detect cheyochu
# If cascades not loaded, quit with message
if face_cascade.empty() or smile_cascade.empty():
    print("Error: could not load Haar cascade XML files.")
    exit()

# ---- Open camera ----
cap = cv2.VideoCapture(0)   # 0 = default camera

if not cap.isOpened():
    print("Error: cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: cannot read frame from camera")
        break

    # Convert frame to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- Detect faces ----
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # ---- Detect smile inside the face region ----
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.8,
            minNeighbors=20
        )

        if len(smiles) > 0:
            cv2.putText(frame, "Smiling", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # ---- Show result ----
    cv2.imshow("Smile Detector", frame)

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---- Cleanup ----
cap.release()
cv2.destroyAllWindows()