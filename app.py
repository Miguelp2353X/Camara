from flask import Flask, request, jsonify, render_template
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def detectar_puno(hand_landmarks):
    """
    Detecta si la mano está en forma de puño cerrado.
    Retorna True si es un puño, False si no.
    """
    dedos_estirados = 0
    dedos_tips = [4, 8, 12, 16, 20]  # Puntas de los dedos
    dedos_pips = [2, 6, 10, 14, 18]  # Nudillos

    for tip, pip in zip(dedos_tips[1:], dedos_pips[1:]):  
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            dedos_estirados += 1

    return dedos_estirados == 0  # Si no hay dedos estirados, es un puño

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/procesar_frame', methods=['POST'])
def procesar_frame():
    """
    Recibe una imagen del cliente, procesa la detección de manos y verifica si hay un puño cerrado.
    Retorna un JSON con {"puño": True/False}.
    """
    file = request.files['frame']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(img_rgb)
    puño_detectado = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if detectar_puno(hand_landmarks):
                puño_detectado = True
                break

    return jsonify({"puño": puño_detectado})

if __name__ == '__main__':
    app.run(debug=True)