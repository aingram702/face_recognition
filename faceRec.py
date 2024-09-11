import face_recognition
import cv2
import numpy as np
# MUST have active webcam or script will error
# refers to default webcam
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it
# OBAMA
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it
# BIDEN
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Load a second sample picture and learn how to recognize it
# TAJ my dog
taj_image = face_recognition.load_image_file("taj.jpg")
taj_face_encoding = face_recognition.face_encodings(taj_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    taj_face_encoding
    ]
known_face_names = [
    "Barack Obama",
    "Joe Biden"
    "taj"
    ]

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # grab single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # convert from grayscale to color
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # find all faces and encodings in current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # see if face is match for known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    # display results
    for(top, right, bottom, left), name in zip(face_locations, face_names):
        # bring scale back up from 1/4 to 1, by multiplying by 4
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # draw box around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # draw label to display persons name
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # display the image
    cv2.imshow('Video', frame)

    # Hit 'q' to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
        
