import tensorrt as trt
import numpy as np
import cupy as cp
import cv2
import time
from collections import deque

# ============================================================
# CONFIGURACIONES
# ============================================================
IP_CAMERA_URL = "http://192.168.0.104:5000/video"
ENGINE_PATH = "cbam4cnn_emotion_model.engine"

INPUT_H = 48
INPUT_W = 48

EMOTION_LABELS = [
    "Neutral", "Happiness", 
    "Sadness", "Surprise",+
    "Fear", "Disgust", "Anger"
]

HEADLESS_MODE = True  # Cambiar a False si hay monitor disponible
HEADLESS_MODE = False  # Cambiar a False si hay monitor disponible
STATS_INTERVAL = 5.0  # Mostrar estadísticas cada N segundos

# ============================================================
# CLASE PARA ESTADÍSTICAS
# ============================================================
class PerformanceStats:
    def __init__(self, window_size=100):
        self.frame_times = deque(maxlen=window_size)
        self.detection_times = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        self.preprocess_times = deque(maxlen=window_size)
        
        self.total_frames = 0
        self.frames_with_faces = 0
        self.total_faces = 0
        self.dropped_frames = 0
        
        self.last_stats_time = time.time()
        self.start_time = time.time()
    
    def add_frame(self, frame_time, detection_time, num_faces):
        self.frame_times.append(frame_time)
        self.detection_times.append(detection_time)
        self.total_frames += 1
        
        if num_faces > 0:
            self.frames_with_faces += 1
            self.total_faces += num_faces
    
    def add_inference(self, preprocess_time, inference_time):
        self.preprocess_times.append(preprocess_time)
        self.inference_times.append(inference_time)
    
    def should_print(self):
        return (time.time() - self.last_stats_time) >= STATS_INTERVAL
    
    def print_stats(self):
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calcular FPS y tiempos promedio
        fps = self.total_frames / elapsed if elapsed > 0 else 0
        
        avg_frame = np.mean(self.frame_times) * 1000 if self.frame_times else 0
        avg_detection = np.mean(self.detection_times) * 1000 if self.detection_times else 0
        avg_preprocess = np.mean(self.preprocess_times) * 1000 if self.preprocess_times else 0
        avg_inference = np.mean(self.inference_times) * 1000 if self.inference_times else 0
        
        # Calcular throughput teórico
        total_processing = avg_detection + avg_preprocess + avg_inference
        max_fps = 1000 / total_processing if total_processing > 0 else 0
        
        # Detección de cuellos de botella
        bottleneck = "N/A"
        if avg_frame > 50:
            bottleneck = "⚠ CAMERA (lento/red)"
        elif avg_detection > avg_inference * 2:
            bottleneck = "⚠ DETECTION (HaarCascade)"
        elif avg_inference > 30:
            bottleneck = "⚠ INFERENCE (GPU/modelo)"
        else:
            bottleneck = "✓ OK"
        
        print("\n" + "="*70)
        print(f"📊 ESTADÍSTICAS - Runtime: {elapsed:.1f}s")
        print("="*70)
        print(f"Frames: {self.total_frames} | FPS: {fps:.1f} | Max teórico: {max_fps:.1f}")
        print(f"Rostros: {self.total_faces} en {self.frames_with_faces} frames ({self.frames_with_faces/max(self.total_frames,1)*100:.1f}%)")
        print("-"*70)
        print(f"⏱ Tiempos promedio (últimos {len(self.frame_times)} frames):")
        print(f"  Captura frame:    {avg_frame:6.2f} ms")
        print(f"  Detección caras:  {avg_detection:6.2f} ms")
        print(f"  Preprocesamiento: {avg_preprocess:6.2f} ms")
        print(f"  Inferencia GPU:   {avg_inference:6.2f} ms")
        print(f"  TOTAL procesado:  {total_processing:6.2f} ms")
        print("-"*70)
        print(f"Cuello de botella: {bottleneck}")
        print("="*70 + "\n")
        
        self.last_stats_time = current_time

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
# PREPROCESAMIENTO
# ============================================================
def preprocess_face(img):
    face = cv2.resize(img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5
    face = np.expand_dims(face, axis=[0, 1])
    return face.astype(np.float32)

# ============================================================
# INFERENCIA CON CUPY
# ============================================================
def classify_face(context, engine, input_name, output_name, face_img, d_input, d_output, stats):
    try:
        t_preprocess = time.time()
        input_data = preprocess_face(face_img)
        context.set_input_shape(input_name, input_data.shape)
        cp.copyto(d_input, cp.asarray(input_data))
        preprocess_time = time.time() - t_preprocess
        
        t_inference = time.time()
        context.set_tensor_address(input_name, d_input.data.ptr)
        context.set_tensor_address(output_name, d_output.data.ptr)
        context.execute_async_v3(stream_handle=0)
        cp.cuda.Stream.null.synchronize()
        output_data = cp.asnumpy(d_output)
        inference_time = time.time() - t_inference
        
        stats.add_inference(preprocess_time, inference_time)
        return output_data[0]

    except Exception as e:
        print(f"⚠ Error inferencia: {e}")
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
    print("🔄 Inicializando TensorRT...")

    try:
        engine = load_engine(ENGINE_PATH)
        context, input_name, output_name = create_context_and_io(engine)
        
        input_shape = (1, 1, INPUT_H, INPUT_W)
        output_shape = context.get_tensor_shape(output_name)
        
        d_input = cp.zeros(input_shape, dtype=cp.float32)
        d_output = cp.zeros(output_shape, dtype=cp.float32)
        
        print(f"✓ Modelo cargado | Input: {input_shape} | Output: {output_shape}")

    except Exception as e:
        print(f"❌ Error: {e}")
        return

    stats = PerformanceStats()
    cap = None
    reconnect_attempts = 0

    try:
        print("📷 Conectando a cámara...")
        
        while True:
            # Reconexión automática
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(IP_CAMERA_URL)
                print(cv2.VideoCapture(IP_CAMERA_URL))
                time.sleep(0.5)

                if not cap.isOpened():
                    reconnect_attempts += 1
                    if reconnect_attempts % 5 == 0:
                        print(f"⚠ Reconexión #{reconnect_attempts}...")
                    time.sleep(2)
                    continue

                print("✓ Cámara conectada")
                reconnect_attempts = 0

            # Capturar frame
            t_frame = time.time()
            ret, frame = cap.read()
            frame_time = time.time() - t_frame
            
            if not ret:
                stats.dropped_frames += 1
                cap.release()
                cap = None
                continue

            # Detección de rostros
            t_detection = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            detection_time = time.time() - t_detection
            
            stats.add_frame(frame_time, detection_time, len(faces))

            # Procesar rostros detectados
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                preds = classify_face(context, engine, input_name, output_name, 
                                     face_img, d_input, d_output, stats)
                
                if preds is None:
                    continue

                idx = np.argmax(preds)
                emotion = EMOTION_LABELS[idx]
                conf = preds[idx]

                # Log compacto por rostro
                print(f"👤 {emotion:10s} ({conf:.2f}) | Frame: {stats.total_frames}")

                # Dibujar en frame
                if not HEADLESS_MODE:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(frame, f"{emotion} ({conf:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Mostrar estadísticas periódicamente
            if stats.should_print():
                stats.print_stats()

            # Visualización
            if not HEADLESS_MODE:
                cv2.imshow("Emotion Recognition - TensorRT", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("🛑 ESC presionado")
                    break
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por usuario")

    finally:
        print("\n🔻 Finalizando...")
        stats.print_stats()  # Mostrar estadísticas finales
        
        if cap is not None:
            cap.release()
        if not HEADLESS_MODE:
            cv2.destroyAllWindows()
        
        del d_input
        del d_output
        cp.get_default_memory_pool().free_all_blocks()
        print("✓ Recursos liberados")

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()
