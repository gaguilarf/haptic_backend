#!/usr/bin/env python3
"""
FER Deployment con servidor BLE integrado
Envía predicciones de emociones vía Bluetooth
"""

import tensorrt as trt
import numpy as np
import cupy as cp
import cv2
import time
import threading
import multiprocessing as mp
import argparse
import queue
from collections import deque
from collections import Counter, defaultdict
import socket
import traceback

import dbus
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib

# ============================================================
# CONFIGURACIONES GLOBALES
# ============================================================
INPUT_H = 48
INPUT_W = 48

EMOTION_LABELS = [
    "Neutral", "Happiness", 
    "Sadness", "Surprise",
    "Fear", "Disgust", "Anger"
]

# Mapeo a 4 etiquetas: Neutral(0), Happiness(1), Sadness(2), Anger(3)
# Ignoramos Surprise, Fear, Disgust
EMOTION_TO_4 = {
    0: 0,
    1: 1,
    2: 2,
    6: 3,
}
DETECTION_INTERVAL = 10.0  # segundos entre detecciones
VIBRATION_INTERVAL = 15.0  # segundos entre vibraciones enviadas

# Cola para comunicación entre procesos (BLE server puede correr en otro proceso)
ble_queue = mp.Queue(maxsize=10)
stop_event = mp.Event()

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
        return (time.time() - self.last_stats_time) >= 5.0
    
    def print_stats(self):
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        fps = self.total_frames / elapsed if elapsed > 0 else 0
        
        avg_frame = np.mean(self.frame_times) * 1000 if self.frame_times else 0
        avg_detection = np.mean(self.detection_times) * 1000 if self.detection_times else 0
        avg_preprocess = np.mean(self.preprocess_times) * 1000 if self.preprocess_times else 0
        avg_inference = np.mean(self.inference_times) * 1000 if self.inference_times else 0
        
        total_processing = avg_detection + avg_preprocess + avg_inference
        max_fps = 1000 / total_processing if total_processing > 0 else 0
        
        print("\n" + "="*70)
        print(f"📊 ESTADÍSTICAS - Runtime: {elapsed:.1f}s")
        print("="*70)
        print(f"Frames: {self.total_frames} | FPS: {fps:.1f} | Max teórico: {max_fps:.1f}")
        print(f"Rostros: {self.total_faces} en {self.frames_with_faces} frames")
        print("-"*70)
        print(f"⏱ Tiempos promedio (últimos {len(self.frame_times)} frames):")
        print(f"  Captura frame:    {avg_frame:6.2f} ms")
        print(f"  Detección caras:  {avg_detection:6.2f} ms")
        print(f"  Preprocesamiento: {avg_preprocess:6.2f} ms")
        print(f"  Inferencia GPU:   {avg_inference:6.2f} ms")
        print("="*70 + "\n")
        
        self.last_stats_time = current_time

# ============================================================
# SERVIDOR BLE SIMPLIFICADO
# ============================================================
class BLECharacteristic(dbus.service.Object):
    """Característica BLE para enviar emociones"""
    
    def __init__(self, bus, index, service):
        self.path = service.path + '/char' + str(index)
        self.bus = bus
        self.uuid = "12345678-1234-5678-1234-56789abcdef1"
        self.service = service
        self.notifying = False
        self.value = [dbus.Byte(0)]
        dbus.service.Object.__init__(self, bus, self.path)
    
    def get_properties(self):
        return {
            'org.bluez.GattCharacteristic1': {
                'Service': self.service.get_path(),
                'UUID': self.uuid,
                'Flags': ['read', 'notify'],
                'Value': self.value,
            }
        }
    
    def get_path(self):
        return dbus.ObjectPath(self.path)
    
    @dbus.service.method('org.freedesktop.DBus.Properties',
                         in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != 'org.bluez.GattCharacteristic1':
            raise dbus.exceptions.DBusException(
                'org.freedesktop.DBus.Error.InvalidArgs',
                'Invalid interface')
        return self.get_properties()['org.bluez.GattCharacteristic1']
    
    @dbus.service.method('org.bluez.GattCharacteristic1',
                        in_signature='a{sv}', out_signature='ay')
    def ReadValue(self, options):
        return self.value
    
    @dbus.service.method('org.bluez.GattCharacteristic1')
    def StartNotify(self):
        if not self.notifying:
            self.notifying = True
            print('✅ Cliente BLE conectado')
    
    @dbus.service.method('org.bluez.GattCharacteristic1')
    def StopNotify(self):
        if self.notifying:
            self.notifying = False
            print('❌ Cliente BLE desconectado')
    
    @dbus.service.signal('org.freedesktop.DBus.Properties',
                         signature='sa{sv}as')
    def PropertiesChanged(self, interface, changed, invalidated):
        pass
    
    def send_emotion(self, emotion_idx, confidence):
        """Envía emoción: 1 byte índice + 1 byte confianza*100"""
        if not self.notifying:
            return False
        
        conf_byte = int(confidence * 100)
        self.value = [dbus.Byte(emotion_idx), dbus.Byte(conf_byte)]
        
        self.PropertiesChanged(
            'org.bluez.GattCharacteristic1',
            {'Value': self.value},
            []
        )
        print(f"📡 BLE: {EMOTION_LABELS[emotion_idx]} ({confidence:.2f})")
        return True


class BLEService(dbus.service.Object):
    """Servicio BLE"""
    
    def __init__(self, bus, index):
        self.path = '/org/bluez/example/service' + str(index)
        self.bus = bus
        self.uuid = "12345678-1234-5678-1234-56789abcdef0"
        self.primary = True
        self.characteristics = []
        dbus.service.Object.__init__(self, bus, self.path)
    
    def get_properties(self):
        return {
            'org.bluez.GattService1': {
                'UUID': self.uuid,
                'Primary': self.primary,
                'Characteristics': dbus.Array(
                    [c.get_path() for c in self.characteristics],
                    signature='o')
            }
        }
    
    def get_path(self):
        return dbus.ObjectPath(self.path)
    
    def add_characteristic(self, characteristic):
        self.characteristics.append(characteristic)
    
    @dbus.service.method('org.freedesktop.DBus.Properties',
                         in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != 'org.bluez.GattService1':
            raise dbus.exceptions.DBusException(
                'org.freedesktop.DBus.Error.InvalidArgs',
                'Invalid interface')
        return self.get_properties()['org.bluez.GattService1']


class BLEApplication(dbus.service.Object):
    """Aplicación GATT"""
    
    def __init__(self, bus):
        self.path = '/'
        self.services = []
        dbus.service.Object.__init__(self, bus, self.path)
    
    def get_path(self):
        return dbus.ObjectPath(self.path)
    
    def add_service(self, service):
        self.services.append(service)
    
    @dbus.service.method('org.freedesktop.DBus.ObjectManager',
                        out_signature='a{oa{sa{sv}}}')
    def GetManagedObjects(self):
        response = {}
        for service in self.services:
            response[service.get_path()] = service.get_properties()
            for char in service.characteristics:
                response[char.get_path()] = char.get_properties()
        return response


class BLEAdvertisement(dbus.service.Object):
    """Anuncio BLE"""
    
    def __init__(self, bus, index, ble_name):
        self.path = '/org/bluez/example/advertisement' + str(index)
        self.bus = bus
        self.ble_name = ble_name
        dbus.service.Object.__init__(self, bus, self.path)
    
    def get_properties(self):
        return {
            'org.bluez.LEAdvertisement1': {
                'Type': 'peripheral',
                'ServiceUUIDs': dbus.Array(["12345678-1234-5678-1234-56789abcdef0"], signature='s'),
                'LocalName': dbus.String(self.ble_name),
            }
        }
    
    def get_path(self):
        return dbus.ObjectPath(self.path)
    
    @dbus.service.method('org.freedesktop.DBus.Properties',
                         in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        if interface != 'org.bluez.LEAdvertisement1':
            raise dbus.exceptions.DBusException(
                'org.freedesktop.DBus.Error.InvalidArgs',
                'Invalid interface')
        return self.get_properties()['org.bluez.LEAdvertisement1']
    
    @dbus.service.method('org.bluez.LEAdvertisement1', out_signature='')
    def Release(self):
        pass


def find_adapter(bus):
    """Encuentra el adaptador Bluetooth"""
    remote_om = dbus.Interface(bus.get_object('org.bluez', '/'),
                                'org.freedesktop.DBus.ObjectManager')
    objects = remote_om.GetManagedObjects()
    for o, props in objects.items():
        if 'org.bluez.GattManager1' in props.keys():
            return o
    return None


def ble_server_thread(ble_name):
    """Process/thread que maneja el servidor BLE y la cola de emociones"""
    # Establecer el main context por defecto en este proceso/hilo para que
    # GLib/dbus operen en su propio contexto y no interfieran con la GUI.
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    
    adapter_path = find_adapter(bus)
    if not adapter_path:
        print('❌ ERROR: Adaptador Bluetooth no encontrado')
        return
    
    print(f'✅ Adaptador encontrado: {adapter_path}')
    
    # Crear aplicación, servicio y característica
    app = BLEApplication(bus)
    service = BLEService(bus, 0)
    app.add_service(service)
    
    char = BLECharacteristic(bus, 0, service)
    service.add_characteristic(char)
    
    # Registrar aplicación
    service_manager = dbus.Interface(
        bus.get_object('org.bluez', adapter_path),
        'org.bluez.GattManager1')
    
    service_manager.RegisterApplication(app.get_path(), {},
                                       reply_handler=lambda: print('✅ Aplicación BLE registrada'),
                                       error_handler=lambda e: print(f'❌ Error: {e}'))
    
    # Crear y registrar anuncio
    ad = BLEAdvertisement(bus, 0, ble_name)
    ad_manager = dbus.Interface(
        bus.get_object('org.bluez', adapter_path),
        'org.bluez.LEAdvertisingManager1')
    
    ad_manager.RegisterAdvertisement(ad.get_path(), {},
                                    reply_handler=lambda: print(f'✅ Anuncio BLE registrado: {ble_name}'),
                                    error_handler=lambda e: print(f'❌ Error: {e}'))
    
    print('📡 Servidor BLE activo')
    
    # Procesar cola de emociones
    def process_ble_queue():
        while not stop_event.is_set():
            try:
                emotion_idx, confidence = ble_queue.get(timeout=1)
                char.send_emotion(emotion_idx, confidence)
            except queue.Empty:
                pass
            except Exception as e:
                print(f'❌ Error BLE: {e}')
        return False
    
    GLib.timeout_add(100, process_ble_queue)
    
    mainloop = GLib.MainLoop()
    try:
        mainloop.run()
    except:
        mainloop.quit()

# ============================================================
# CARGA DEL MOTOR TRT
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
# INFERENCIA
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
    parser = argparse.ArgumentParser(description='FER Deployment con BLE')
    parser.add_argument('--engine', type=str, default='cbam4cnn_emotion_model.engine',
                        help='Ruta del modelo TensorRT')
    parser.add_argument('--camera', type=str, default='http://192.168.0.104:5000/video',
                        help='URL de la cámara IP')
    parser.add_argument('--headless', action='store_true',
                        help='Modo sin pantalla (headless)')
    parser.add_argument('--no-ble', action='store_true',
                        help='Desactivar servidor BLE')
    parser.add_argument('--ble-name', type=str, default='FER-Device',
                        help='Nombre del dispositivo BLE')
    parser.add_argument('--stats-interval', type=float, default=5.0,
                        help='Intervalo de estadísticas en segundos')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 FER DEPLOYMENT CON BLE")
    print("="*70)
    print(f"Modelo: {args.engine}")
    print(f"Cámara: {args.camera}")
    print(f"BLE: {'❌ Desactivado' if args.no_ble else f'✅ {args.ble_name}'}")
    print(f"Pantalla: {'Headless' if args.headless else 'Gráfica'}")
    print("="*70)
    
    # Iniciar servidor BLE en thread separado
    ble_thread = None
    if not args.no_ble:
        try:
            # Ejecutar servidor BLE en un proceso separado para evitar conflictos
            # entre GLib (DBus) y la GUI (cv2.imshow) en el hilo principal.
            ble_proc = mp.Process(target=ble_server_thread, args=(args.ble_name,), daemon=True)
            ble_proc.start()
            time.sleep(2)  # Esperar a que BLE se inicialice
        except Exception as e:
            print(f'⚠ Advertencia: No se pudo iniciar BLE: {e}')
    
    # Inicializar TensorRT
    print("🔄 Inicializando TensorRT...")
    try:
        engine = load_engine(args.engine)
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
    last_detection_time = 0.0
    last_vib_time = 0.0
    # Contadores para decidir la emoción mayoritaria en cada periodo de vibración
    mapped_counts = Counter()
    orig_counts = Counter()
    conf_sums = defaultdict(float)

    try:
        print("📷 Conectando a cámara...")
        
        while True:
            # Reconexión automática
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(args.camera)
                # Debug: información del backend y estado de apertura
                try:
                    backend_name = getattr(cap, 'getBackendName', None)
                    if callable(backend_name):
                        print(f"DEBUG: VideoCapture backend: {cap.getBackendName()}")
                except Exception:
                    pass
                print(f"DEBUG: cap.isOpened() -> {cap.isOpened()}")
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
            # (debug prints removed to reduce console spam)
            frame_time = time.time() - t_frame
            
            if not ret:
                stats.dropped_frames += 1
                cap.release()
                cap = None
                continue

            # Detección de rostros (solo cada DETECTION_INTERVAL segundos)
            current_time = time.time()
            if (current_time - last_detection_time) >= DETECTION_INTERVAL:
                t_detection = time.time()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
                detection_time = time.time() - t_detection

                stats.add_frame(frame_time, detection_time, len(faces))

                # Procesar rostros detectados: seleccionar la predicción con mayor confianza
                best_idx = None
                best_conf = 0.0
                best_face = None

                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    preds = classify_face(context, engine, input_name, output_name,
                                         face_img, d_input, d_output, stats)

                    if preds is None:
                        continue

                    idx = int(np.argmax(preds))
                    conf = float(preds[idx])

                    if conf > best_conf:
                        best_conf = conf
                        best_idx = idx
                        best_face = (x, y, w, h)

                if best_idx is not None:
                    emotion = EMOTION_LABELS[best_idx]
                    print(f"👤 {emotion:10s} ({best_conf:.2f}) | Frame: {stats.total_frames}")

                    # Acumular conteo para decidir la emoción mayoritaria en el periodo
                    mapped = EMOTION_TO_4.get(best_idx, None)
                    if mapped is not None:
                        mapped_counts[mapped] += 1
                        orig_counts[best_idx] += 1
                        conf_sums[best_idx] += best_conf

                    # Dibujar en frame la mejor predicción (visual only)
                    if not args.headless and best_face is not None:
                        x, y, w, h = best_face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                        cv2.putText(frame, f"{emotion} ({best_conf:.2f})", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                last_detection_time = current_time

                # Si ya pasó el intervalo de vibración, elegir la emoción mayoritaria y enviar UNA vez
                if (current_time - last_vib_time) >= VIBRATION_INTERVAL:
                    if mapped_counts:
                        most = mapped_counts.most_common(1)
                        if most:
                            best_mapped, cnt = most[0]
                            if cnt > 0:
                                # Elegir el índice original con más ocurrencias dentro de ese mapeo
                                candidates = [orig for orig in orig_counts.keys() if EMOTION_TO_4.get(orig) == best_mapped]
                                best_orig = None
                                if candidates:
                                    best_orig = max(candidates, key=lambda o: orig_counts[o])

                                # Calcular confianza promedio si disponible
                                avg_conf = 0.0
                                if best_orig is not None and orig_counts[best_orig] > 0:
                                    avg_conf = conf_sums[best_orig] / orig_counts[best_orig]

                                # Enviar UDP una sola vez para provocar vibración
                                try:
                                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                                    sock.settimeout(0.5)
                                    sock.sendto(str(best_mapped).encode('utf-8'), ('127.0.0.1', 5005))
                                    sock.close()
                                    print(f"→ VIBRAR: enviado etiqueta {best_mapped} (conteo={cnt})")
                                except Exception as e:
                                    print(f"⚠ Error enviando UDP de vibración: {e}")

                                # Enviar al servidor BLE interno (si activado) la emoción original y confianza promedio
                                if not args.no_ble and best_orig is not None:
                                    try:
                                        if not ble_queue.full():
                                            ble_queue.put_nowait((best_orig, avg_conf))
                                    except Exception:
                                        pass

                    # Reset counters y actualizar tiempo de última vibración
                    mapped_counts.clear()
                    orig_counts.clear()
                    conf_sums.clear()
                    last_vib_time = current_time

            # Mostrar estadísticas periódicamente
            if stats.should_print():
                stats.print_stats()

            # Visualización
            if not args.headless:
                try:
                    cv2.imshow("FER - Emotion Recognition", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        print("🛑 ESC presionado")
                        break
                except Exception as e:
                    print("ERROR en cv2.imshow:", e)
                    traceback.print_exc()
                    # No abortar inmediatamente; intentar continuar
                    time.sleep(0.5)
            else:
                time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por usuario")

    finally:
        print("\n🔻 Finalizando...")
        stop_event.set()
        stats.print_stats()
        
        if cap is not None:
            cap.release()
        if not args.headless:
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
