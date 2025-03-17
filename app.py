from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

puño_detectado = False  # Variable para indicar si el puño está cerrado

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

def generar_frames():
    global puño_detectado  # Permite modificar la variable global
    cap = cv2.VideoCapture(0)  

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_hands = hands.process(img_rgb)

        puño_detectado = False  # Reiniciar estado

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                if detectar_puno(hand_landmarks):
                    puño_detectado = True
                    color_puntos = (0, 0, 255)  # Rojo
                else:
                    color_puntos = (255, 0, 0)  # Azul
                
                for lm in hand_landmarks.landmark:
                    ih, iw, _ = frame.shape
                    cx, cy = int(lm.x * iw), int(lm.y * ih)
                    cv2.circle(frame, (cx, cy), 5, color_puntos, cv2.FILLED)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_fist')
def check_fist():
    """Devuelve el estado del puño (cerrado o no) en JSON."""
    return jsonify({"puño": puño_detectado})

if __name__ == '__main__':
    app.run(debug=True)