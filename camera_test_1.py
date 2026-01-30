import cv2
import numpy as np
import onnxruntime as ort
import albumentations as A
import time

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
IP_CAMERA_URL = "http://10.7.46.63:5000/video"

ONNX_MODEL_PATH = "cbam4cnn_emotion_model.onnx"
INPUT_HEIGHT = 48
INPUT_WIDTH = 48

EMOTION_LABELS = [
    "Neutral", "Happiness", "Sadness", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt"
]

# -----------------------------
# CARGAR MODELO
# -----------------------------
print("🔄 Cargando modelo ONNX...")

try:
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"✓ Modelo cargado: {ONNX_MODEL_PATH}")
    print(f"✓ Input name: {input_name}")
    print(f"✓ Output name: {output_name}")
except Exception as e:
    print("❌ Error cargando el modelo ONNX:", e)
    exit()


# -----------------------------
# TRANSFORMACIÓN PRE-PROCESADO
# -----------------------------
preprocess = A.Compose([
    A.Resize(height=INPUT_HEIGHT, width=INPUT_WIDTH, interpolation=cv2.INTER_AREA),
    A.ToGray(num_output_channels=1),
    A.Normalize(mean=[0.5], std=[0.5]),
])


def preprocess_face(face_img):
    """Convierte un rostro a tensor compatible ONNX."""
    transformed = preprocess(image=face_img)
    img = transformed["image"]

    # (H, W, 1) → (1, H, W)
    img = np.transpose(img, (2, 0, 1))
    # (1, 1, H, W)
    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img


# -----------------------------
# CLASIFICACIÓN DEL ROSTRO
# -----------------------------
def classify_face(face_img):
    try:
        img_tensor = preprocess_face(face_img)
        outputs = session.run([output_name], {input_name: img_tensor})
        preds = outputs[0][0]            # (8,)
        return preds
    except Exception as e:
        print("⚠ Error en inferencia ONNX:", e)
        return None


# -----------------------------
# DETECTOR DE CARAS
# -----------------------------
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

if face_detector.empty():
    print("❌ No se pudo cargar HaarCascade.")
    exit()

# -----------------------------
# FUNCIÓN PRINCIPAL DE VIDEO
# -----------------------------
def main():
    print("📷 Intentando conectar a la cámara...")
    cap = None

    try:
        while True:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(IP_CAMERA_URL)
                time.sleep(1)
                if not cap.isOpened():
                    print("❌ No se pudo conectar. Reintento en 2s...")
                    time.sleep(2)
                    continue
                print("✓ Cámara conectada.")

            ret, frame = cap.read()
            if not ret:
                print("⚠ No se puede leer frame. Reconectando...")
                cap.release()
                cap = None
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]

                # Inferencia
                preds = classify_face(face_img)
                if preds is None:
                    continue

                # Argmax emoción
                idx = np.argmax(preds)
                emotion = EMOTION_LABELS[idx]
                conf = preds[idx]

                # =======================
                # DIBUJAR RESULTADOS
                # =======================

                # Bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Emoción principal
                cv2.putText(frame, f"{emotion} ({conf:.2f})",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2, cv2.LINE_AA)

                # Probabilidades completas
                start_y = y + h + 20
                for i, label in enumerate(EMOTION_LABELS):
                    prob = preds[i]
                    color = (0, 255, 0) if i == idx else (255, 255, 255)
                    cv2.putText(
                        frame,
                        f"{label}: {prob:.2f}",
                        (x, start_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA
                    )
                    start_y += 22

                # Debug en consola
                print("------------------------------")
                print("Rostro detectado")
                print(f"Emoción: {emotion}  Confianza: {conf:.2f}")
                for l, p in zip(EMOTION_LABELS, preds):
                    print(f"{l:10s}: {p:.2f}")

            cv2.imshow("Emotion Recognition - ONNX", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:   # ESC
                print("🛑 Salida solicitada.")
                break

    except KeyboardInterrupt:
        print("\n🛑 Interrupción manual (Ctrl+C).")

    finally:
        print("🔻 Liberando recursos...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

