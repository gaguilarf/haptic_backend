# Configuración - FER Deployment con BLE

## Ejecución Básica

```bash
python main.py
```

## Opciones de Línea de Comandos

### BLE
- `--no-ble` → Desactiva el servidor BLE (solo inferencia)
- `--ble-name NAME` → Nombre del dispositivo BLE (default: `FER-Device`)

### Cámara
- `--camera URL` → URL de la cámara IP (default: `http://10.7.46.63:5000/video`)

### Modelo
- `--engine PATH` → Ruta del modelo TensorRT (default: `cbam4cnn_emotion_model.engine`)

### Display
- `--headless` → Modo sin pantalla (para Jetson sin monitor)

### Estadísticas
- `--stats-interval SECONDS` → Intervalo de estadísticas (default: 5.0)

## Ejemplos de Uso

### Ejecutar con BLE (default)
```bash
python main.py
```

### Ejecutar sin BLE (solo cámara + inferencia)
```bash
python main.py --no-ble
```

### Ejecutar en modo headless con BLE
```bash
python main.py --headless
```

### Ejecutar con URL de cámara personalizada
```bash
python main.py --camera http://192.168.1.50:8081/video
```

### Ejecutar con nombre BLE personalizado
```bash
python main.py --ble-name "EmotionDetector"
```

### Combinado: headless + BLE custom + cámara custom
```bash
python main.py --headless --ble-name "MyEmotionDevice" --camera http://192.168.1.100:8080/video
```

## Protocolo BLE

**Servicio UUID:** `12345678-1234-5678-1234-56789abcdef0`

**Característica UUID:** `12345678-1234-5678-1234-56789abcdef1`

**Formato de datos (2 bytes):**
- Byte 1: Índice de emoción (0-6)
  - 0 = Neutral
  - 1 = Happiness
  - 2 = Sadness
  - 3 = Surprise
  - 4 = Fear
  - 5 = Disgust
  - 6 = Anger
- Byte 2: Confianza × 100 (0-100)

## Ejemplo: Conectar desde Cliente BLE

```python
# Cliente Python con bleak
import asyncio
from bleak import BleakClient

async def main():
    device_address = "XX:XX:XX:XX:XX:XX"  # Dirección del Jetson
    char_uuid = "12345678-1234-5678-1234-56789abcdef1"
    
    async with BleakClient(device_address) as client:
        await client.start_notify(char_uuid, notification_handler)
        await asyncio.sleep(30)

def notification_handler(sender, data):
    emotion_idx = data[0]
    confidence = data[1] / 100.0
    emotions = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger"]
    print(f"Emoción: {emotions[emotion_idx]} ({confidence:.2f})")

asyncio.run(main())
```
