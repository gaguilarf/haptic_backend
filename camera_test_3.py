import tensorrt as trt
import numpy as np
import cv2
import time

ENGINE_PATH = "cbam4cnn_emotion_model.engine"
INPUT_H = 48
INPUT_W = 48

EMOTION_LABELS = [
    "Neutral", "Happiness", "Sadness", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt"
]

# ---------------------------------------------------------
# Cargar ENGINE TensorRT 10.x
# ---------------------------------------------------------
def load_engine(engine_path):
    logger = trt.Logger(trt.Logger.ERROR)

    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("No se pudo deserializar el engine")
        return engine


# ---------------------------------------------------------
# Crear contexto + IO
# ---------------------------------------------------------
def create_context_and_io(engine):
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("No se pudo crear el ExecutionContext")

    nb = engine.num_io_tensors

    input_idx = None
    output_idx = None

    for i in range(nb):
        name = engine.get_tensor_name(i)
        is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

        if is_input:
            input_idx = i
            input_name = name
        else:
            output_idx = i
            output_name = name

    return context, input_name, output_name


# ---------------------------------------------------------
# Preprocesamiento con numpy (sin albumentations)
# ---------------------------------------------------------
def preprocess(img):
    face = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5

    face = np.expand_dims(face, axis=[0, 1])  # (1,1,48,48)
    return face.astype(np.float32)


# ---------------------------------------------------------
# Inferencia TensorRT 10
# ---------------------------------------------------------
def infer(context, engine, input_name, output_name, img):
    input_np = preprocess(img)

    # Prepara tensores
    context.set_input_shape(input_name, input_np.shape)

    # Asignar buffers vinculados
    context.set_tensor_address(input_name, input_np.ctypes.data)

    # Crear salida
    out_shape = context.get_tensor_shape(output_name)
    output_np = np.empty(out_shape, dtype=np.float32)
    context.set_tensor_address(output_name, output_np.ctypes.data)

    # Ejecutar
    context.execute_async_v3(stream_handle=0)

    return output_np


# ---------------------------------------------------------
# PROGRAMA PRINCIPAL
# ---------------------------------------------------------
print("🔄 Cargando modelo TensorRT...")

try:
    engine = load_engine(ENGINE_PATH)
    context, input_name, output_name = create_context_and_io(engine)
    print("✓ Modelo cargado correctamente")
except Exception as e:
    print("❌ Error cargando modelo:", e)
    exit()

print("📷 Abriendo cámara...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    preds = infer(context, engine, input_name, output_name, frame)[0]
    idx = np.argmax(preds)
    emotion = EMOTION_LABELS[idx]

    cv2.putText(frame, f"{emotion} {preds[idx]:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Emotion Recognition TRT", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


