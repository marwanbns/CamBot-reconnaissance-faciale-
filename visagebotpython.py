import cv2
import dlib
import numpy as np
import time
import pyttsx3

engine = pyttsx3.init()
#On utilise des modèles pré-entrainée
pose_predictor_68_point = dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("pretrained_model/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()

def charger_visages_connus():
    visages = [
        ("Zuckerberg", r"visage\Zuckerberg.png"),
        ("Biden", r"visage\Biden.png")
    ]
    encodages_visages_connus = []
    noms_visages_connus = []
    
    for nom, chemin in visages:
        image = cv2.imread(chemin)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray) #On détecte les visages
        for face in faces:
            shape = pose_predictor_68_point(image, face)
            encodage = np.array(face_encoder.compute_face_descriptor(image, shape))
            encodages_visages_connus.append(encodage)
            noms_visages_connus.append(nom)
    
    return encodages_visages_connus, noms_visages_connus

def traiter_detection(frame, encodages_visages_connus, noms_visages_connus, personnes_detectees, detection_threshold, faces_times):
    #Détection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    for face in faces:
        shape = pose_predictor_68_point(frame, face)
        encodage_visage = np.array(face_encoder.compute_face_descriptor(frame, shape))
        #On compare avec les visages dans la base de données
        correspondances = np.linalg.norm(encodages_visages_connus - encodage_visage, axis=1)
        tolerance = 0.6
        nom = "Inconnu"
        #On sélectionne la correspondance la plus proche
        if np.any(correspondances <= tolerance):
            index = np.argmin(correspondances)
            nom = noms_visages_connus[index]
        #Rectangle d'affichage avec le nom de la personne
        (x, y, w, h) = (face.left(), face.top(), face.right(), face.bottom())
        cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
        cv2.putText(frame, nom, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
         #Gérer la détection de la personne et le délai de 6 secondes
        if nom not in faces_times:
            faces_times[nom] = time.time()
        return nom, frame, faces_times
    return None, frame, faces_times  #Si aucun visage n'est détecté

def reconnaissance_faciale():
    stop=False
    #On charge les visages connus
    encodages_visages_connus, noms_visages_connus = charger_visages_connus()
    #On réduit la résolution pour améliorer les performances
    capture_video = cv2.VideoCapture(0)
    capture_video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture_video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    personnes_detectees = []  #Liste pour suivre les personnes détectées récemment
    detection_threshold = 4  #Temps de détection de 4 secondes
    faces_times = {} 
    face_detected = False
    while not stop:
        ret, frame = capture_video.read()
        if not ret:
            break
        #On commence la detection
        nom, frame, faces_times = traiter_detection(frame, encodages_visages_connus, noms_visages_connus, personnes_detectees, detection_threshold, faces_times)
        if nom:  #Si un visage a été détecté
            #Vérifier si 4 secondes se sont écoulées depuis que le visage a été détecté
            if time.time() - faces_times[nom] >= detection_threshold:
                if nom not in personnes_detectees:
                    print(f'Bonjour {nom}')
                    engine.say(f"Bonjour {nom}")
                    engine.runAndWait()
                    personnes_detectees.append(nom)
                    faces_times[nom] = time.time()
        #Afficher la vidéo avec les visages et les noms détectés
        cv2.imshow("Cambot 2", frame)
        #Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop = True
            break
    capture_video.release()
    cv2.destroyAllWindows()

#Exécuter la reconnaissance faciale
reconnaissance_faciale()
