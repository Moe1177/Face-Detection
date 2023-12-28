import cv2
import pathlib

# Link the cascade path
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data"/"haarcascade_frontalface_default.xml"

# Create classifier path
classifier_path = cv2.CascadeClassifier(str(cascade_path))

# Choose camera
camera = cv2.VideoCapture(0)

while True:

    # Frame is what the camera can record
    _, frame = camera.read()

    # Create a gray picture
    gray_picture = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    faces = classifier_path.detectMultiScale(
        gray_picture,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (40,40),
    )

    # Create rectangles to see if faces detected
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 4)

    # Display frame
    cv2.imshow("Face Recorder", frame)

    # Click q to exit program
    if cv2.waitKey(1) == ord('q'):
        break

# When exit program cut camera and close windows
camera.release()
cv2.destroyAllWindows()


    

