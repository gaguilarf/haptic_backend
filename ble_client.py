"""
BLE Inference Client for Jetson Nano
=====================================
Cliente BLE que se conecta a un dispositivo periférico y escribe
resultados de clasificación en tiempo real desde una cámara simulada.
"""

import asyncio
import logging
import signal
import random
import time
from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# UUIDs - deben coincidir con el servidor/dispositivo periférico
SERVICE_UUID = "19b10000-e8f2-537e-4f6c-d104768a1214"
CHAR_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214"

# Nombre del dispositivo a buscar
DEVICE_NAME = "SeedXIAO_Vib"

# Tasa de inferencia y envío
FPS = 0.2
FRAME_INTERVAL = 1.0 / FPS

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BLE_Client")

# =============================================================================
# MOTOR DE INFERENCIA SIMULADO
# =============================================================================

class UDPProtocol:
    """Protocolo UDP para recibir etiquetas desde el proceso de inferencia.

    Mensajes esperados: texto ASCII con un entero (p.e. "0" o "2\n").
    Coloca el entero en una cola asyncio para ser procesado por el cliente BLE.
    """

    def __init__(self, queue):
        self.queue = queue

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        try:
            text = data.decode('utf-8').strip()
            if not text:
                return
            val = int(text.split(',')[0])
            asyncio.create_task(self.queue.put(val))
            logger.info(f"UDP recibido {val} desde {addr}")
        except Exception as e:
            logger.warning(f"UDP parse error: {e} | raw: {data}")

# =============================================================================
# CLIENTE BLE
# =============================================================================

class BLEInferenceClient:
    """
    Cliente BLE que se conecta a un dispositivo periférico y escribe
    resultados de clasificación.
    """
    
    def __init__(self, device_name=DEVICE_NAME):
        self.device_name = device_name
        self.device_address = None
        self.client = None
        self.running = False
        self.connected = False
        self.received_queue = asyncio.Queue()
        self.udp_host = '127.0.0.1'
        self.udp_port = 5005
    
    async def scan_for_device(self, timeout=10.0):
        """
        Escanea dispositivos BLE y busca el dispositivo objetivo.
        
        Args:
            timeout: Tiempo máximo de escaneo en segundos
            
        Returns:
            Dirección del dispositivo si se encuentra, None si no
        """
        logger.info(f"Escaneando dispositivos BLE (timeout: {timeout}s)...")
        logger.info(f"Buscando: '{self.device_name}'")
        
        try:
            devices = await BleakScanner.discover(timeout=timeout)
            
            logger.info(f"Encontrados {len(devices)} dispositivos:")
            for device in devices:
                name = device.name or "Sin nombre"
                logger.info(f"  - {name} ({device.address})")
                
                if device.name == self.device_name:
                    logger.info(f"✓ Dispositivo objetivo encontrado: {device.address}")
                    return device.address
            
            logger.warning(f"✗ Dispositivo '{self.device_name}' no encontrado")
            return None
            
        except Exception as e:
            logger.error(f"Error durante escaneo: {e}")
            return None
    
    async def connect(self, max_retries=3):
        """
        Conecta al dispositivo BLE.
        
        Args:
            max_retries: Número máximo de intentos de conexión
            
        Returns:
            True si la conexión fue exitosa, False si no
        """
        if not self.device_address:
            logger.error("No hay dirección de dispositivo. Ejecuta scan_for_device() primero.")
            return False
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Intento de conexión {attempt}/{max_retries}...")
                
                self.client = BleakClient(self.device_address)
                await self.client.connect()
                
                if self.client.is_connected:
                    logger.info(f"✓ Conectado exitosamente a {self.device_address}")
                    self.connected = True
                    
                    # Verificar que el servicio y característica existen
                    if await self.verify_service():
                        return True
                    else:
                        await self.disconnect()
                        return False
                        
            except BleakError as e:
                logger.warning(f"Intento {attempt} fallido: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Error inesperado en conexión: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2)
        
        logger.error(f"✗ No se pudo conectar después de {max_retries} intentos")
        return False
    
    async def verify_service(self):
        """
        Verifica que el servicio y característica existen en el dispositivo.
        
        Returns:
            True si todo existe, False si no
        """
        try:
            logger.info("Verificando servicios y características...")
            
            # Obtener todos los servicios
            services = self.client.services
            
            # Buscar nuestro servicio
            service = services.get_service(SERVICE_UUID)
            if not service:
                logger.error(f"✗ Servicio {SERVICE_UUID} no encontrado")
                return False
            
            logger.info(f"✓ Servicio encontrado: {SERVICE_UUID}")
            
            # Buscar nuestra característica
            char = service.get_characteristic(CHAR_UUID)
            if not char:
                logger.error(f"✗ Característica {CHAR_UUID} no encontrada")
                return False
            
            logger.info(f"✓ Característica encontrada: {CHAR_UUID}")
            
            # Verificar que la característica soporta escritura
            if "write" not in char.properties and "write-without-response" not in char.properties:
                logger.error(f"✗ Característica no soporta escritura")
                logger.error(f"   Propiedades: {char.properties}")
                return False
            
            logger.info(f"✓ Característica soporta escritura")
            logger.info(f"   Propiedades: {char.properties}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error verificando servicio: {e}")
            return False
    
    async def write_prediction(self, value):
        """
        Escribe un valor de predicción a la característica BLE.
        
        Args:
            value: Valor entero de predicción (0-9)
            
        Returns:
            True si la escritura fue exitosa, False si no
        """
        if not self.connected or not self.client.is_connected:
            logger.error("No conectado al dispositivo")
            return False
        
        try:
            # Convertir entero a 4 bytes (little-endian)
            data = value.to_bytes(4, 'little', signed=True)
            
            # Escribir a la característica
            await self.client.write_gatt_char(CHAR_UUID, data)
            
            logger.info(
                f"ESCRIBIÓ | Valor: {value} | "
                f"Bytes: {' '.join(f'{b:02x}' for b in data)}"
            )
            return True
            
        except BleakError as e:
            logger.error(f"Error BLE al escribir: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inesperado al escribir: {e}")
            return False
    async def udp_server(self, host=None, port=None):
        host = host or self.udp_host
        port = port or self.udp_port
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: UDPProtocol(self.received_queue), local_addr=(host, port)
        )
        logger.info(f"UDP servidor escuchando en {host}:{port}")
        return transport

    async def write_loop(self):
        """Consume valores recibidos por UDP y los escribe por BLE."""
        logger.info("Iniciando write_loop (consumiendo cola UDP)")
        self.running = True
        while self.running:
            try:
                val = await self.received_queue.get()
                if not self.client or not self.client.is_connected:
                    logger.warning("No conectado al dispositivo; descartando valor")
                    continue
                await self.write_prediction(int(val))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error en write_loop: {e}")
                await asyncio.sleep(0.5)
    
    async def disconnect(self):
        """Desconecta del dispositivo BLE."""
        if self.client and self.connected:
            try:
                logger.info("Desconectando...")
                await self.client.disconnect()
                logger.info("✓ Desconectado exitosamente")
            except Exception as e:
                logger.error(f"Error al desconectar: {e}")
            finally:
                self.connected = False
    
    async def run(self):
        """
        Ejecuta el cliente completo: escanea, conecta y ejecuta inferencia.
        """
        logger.info("=" * 60)
        logger.info("Cliente BLE de Inferencia Iniciando")
        logger.info("=" * 60)
        
        try:
            # Paso 1: Escanear dispositivo
            self.device_address = await self.scan_for_device()
            if not self.device_address:
                logger.error("No se pudo encontrar el dispositivo. Abortando.")
                return
            
            # Paso 2: Conectar
            if not await self.connect():
                logger.error("No se pudo conectar al dispositivo. Abortando.")
                return
            
            # Paso 3: Levantar servidor UDP local y comenzar a consumir valores
            udp_transport = await self.udp_server()
            try:
                await self.write_loop()
            finally:
                try:
                    udp_transport.close()
                except Exception:
                    pass
            
        except KeyboardInterrupt:
            logger.info("\nInterrupción de teclado recibida")
        except Exception as e:
            logger.error(f"Error fatal: {e}", exc_info=True)
        finally:
            # Cleanup
            self.running = False
            await self.disconnect()
            logger.info("Cliente detenido limpiamente")

# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# =============================================================================

async def main():
    """Punto de entrada principal."""
    
    # Crear cliente
    client = BLEInferenceClient(device_name=DEVICE_NAME)
    
    # Manejador de señales para shutdown graceful
    def signal_handler():
        logger.info("\nSeñal de shutdown recibida...")
        client.running = False
    
    # Registrar manejadores de señales
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    # Ejecutar cliente
    await client.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nPrograma terminado por usuario")
    except Exception as e:
        logger.error(f"Error fatal: {e}", exc_info=True)