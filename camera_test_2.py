import tensorrt as trt
import numpy as np
import cv2
import time

# ============================================================
# CONFIGURACIONES
# ============================================================
IP_CAMERA_URL = "http://10.7.46.63:5000/video"
ENGINE_PATH = "cbam4cnn_emotion_model.engine"

INPUT_H = 48
INPUT_W = 48

EMOTION_LABELS = [
    "Neutral", "Happiness", "Sadness", "Surprise",
    "Fear", "Disgust", "Anger"
]

# ============================================================
# CARGA DEL MOTOR TRT10
# ============================================================
def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("No se pudo deserializar el engine")
        return engine


def create_context_and_io(engine):
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("No se pudo crear el contexto")

    nb = engine.num_io_tensors

    input_name = None
    output_name = None

    for i in range(nb):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)

        if mode == trt.TensorIOMode.INPUT:
            input_name = name
        else:
            output_name = name

    return context, input_name, output_name


# ============================================================
# PREPROCESAMIENTO (sin albumentations)
# ============================================================
def preprocess_face(img):
    """
    Convierte el rostro a tensor estilo (1,1,48,48)
    """
    face = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5

    # (48,48) → (1,1,48,48)
    face = np.expand_dims(face, axis=[0, 1])
    return face.astype(np.float32)


# ============================================================
# INFERENCIA TRTensorRT 10.x
# ============================================================
def classify_face(context, engine, input_name, output_name, face_img):
    try:
        input_data = preprocess_face(face_img)
        context.set_input_shape(input_name, input_data.shape)

        # Direcciones
        context.set_tensor_address(input_name, input_data.ctypes.data)

        out_shape = context.get_tensor_shape(output_name)
        output_data = np.empty(out_shape, dtype=np.float32)
        context.set_tensor_address(output_name, output_data.ctypes.data)

        # Ejecutar
        context.execute_async_v3(stream_handle=0)

        return output_data[0]

    except Exception as e:
        print("⚠ Error en inferencia TRT:", e)
        return None


# ============================================================
# DETECTOR DE CARAS
# ============================================================
face_detector = cv2.CascadeClassifier(
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("❌ No se pudo cargar HaarCascade.")
    exit()


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================
def main():
    print("🔄 Cargando modelo TensorRT...")

    try:
        engine = load_engine(ENGINE_PATH)
        context, input_name, output_name = create_context_and_io(engine)
        print("✓ Modelo cargado correctamente")
    except Exception as e:
        print("❌ Error cargando el modelo:", e)
        return

    print("📷 Intentando conectar a la cámara...")
    cap = None

    try:
        while True:

            # ----------------------------------------------------
            # Reconexión automática
            # ----------------------------------------------------
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(IP_CAMERA_URL)
                time.sleep(1)

                if not cap.isOpened():
                    print("❌ No se pudo conectar. Reintento...")
                    time.sleep(2)
                    continue

                print("✓ Cámara conectada.")

            ret, frame = cap.read()
            if not ret:
                print("⚠ No se puede leer frame. Reconectando...")
                cap.release()
                cap = None
                continue

            # ----------------------------------------------------
            # DETECCIÓN DE ROSTROS
            # ----------------------------------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]

                preds = classify_face(context, engine, input_name, output_name, face_img)
                if preds is None:
                    continue

                idx = np.argmax(preds)
                emotion = EMOTION_LABELS[idx]
                conf = preds[idx]

                # ------------------- DIBUJAR --------------------
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, f"{emotion} ({conf:.2f})",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0), 2)

                # Probabilidades detalladas
                yy = y + h + 20
                for i, label in enumerate(EMOTION_LABELS):
                    color = (0,255,0) if i == idx else (255,255,255)
                    cv2.putText(frame, f"{label}: {preds[i]:.2f}",
                                (x, yy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    yy += 22

                # Debug
                print("------------------------------")
                print("Rostro detectado")
                print(f"Emoción: {emotion}  Confianza: {conf:.2f}")
                for lab, p in zip(EMOTION_LABELS, preds):
                    print(f"{lab:10s}: {p:.2f}")

            # ----------------------------------------------------
            # Mostrar ventana
            # ----------------------------------------------------
            cv2.imshow("Emotion Recognition - TensorRT", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                print("🛑 Salida solicitada.")
                break

    except KeyboardInterrupt:
        print("\n🛑 Interrupción manual.")

    finally:
        print("🔻 Liberando recursos...")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()


