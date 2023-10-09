import cv2
import face_recognition

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Create a list to store face encodings
known_face_encodings = []

while True:
    ret, frame = video_capture.read()

    # Find face locations in the current frame
    face_locations = face_recognition.face_locations(frame)

    # Encode the detected faces
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]
        face_encodings = face_recognition.face_encodings(face_image)
        
        if len(face_encodings) > 0:
            known_face_encodings.append(face_encodings[0])

    # Display face rectangles
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    # Press 'q' to exit the video loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the face encodings to a text file
with open('face_encodings.txt', 'w') as file:
    for encoding in known_face_encodings:
        encoding_str = ','.join(map(str, encoding))
        file.write(encoding_str + '\n')

video_capture.release()
cv2.destroyAllWindows()
