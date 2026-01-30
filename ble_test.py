"""
BLE Inference Server for Jetson Nano Orin
==========================================

This server implements a Bluetooth Low Energy (BLE) GATT peripheral that broadcasts
inference predictions at a fixed frame rate using the BlueZ Linux Bluetooth stack.

"""

import time
import random
import signal
import logging
import threading
from queue import Queue, Empty

import dbus
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib


# =============================================================================
# CONFIGURATION
# =============================================================================

# UUID for the custom GATT service
# This identifies our inference service to BLE clients
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"

# UUID for the characteristic that carries inference data
# Clients read from and subscribe to this characteristic
CHAR_UUID    = "12345678-1234-5678-1234-56789abcdef1"

# Inference and transmission rate
FPS = 5
FRAME_INTERVAL = 1.0 / FPS

# Device name visible during BLE scanning
DEVICE_NAME = "DL-Inference"

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BLE")


# =============================================================================
# INFERENCE ENGINE THREAD
# =============================================================================

class InferenceEngine(threading.Thread):
    """
    Simulates an inference engine running at a fixed frame rate.
    
    In a production system, this would:
    - Capture frames from a camera
    - Run a neural network (e.g., TensorRT on Jetson)
    - Output classification results
    
    This simulation generates random predictions (0-9) with realistic timing,
    including frame capture, preprocessing, and inference stages.
    """
    
    def __init__(self, queue, stop_event):
        """
        Initialize the inference engine.
        
        Args:
            queue: Thread-safe queue for passing predictions to BLE thread
            stop_event: Threading event to signal shutdown
        """
        super().__init__(daemon=True)
        self.queue = queue
        self.stop_event = stop_event
        self.frame_id = 0

    def run(self):
        """Main inference loop - runs at fixed FPS"""
        logger.info(f"Inference engine starting at {FPS} FPS")
        logger.info(f"   Frame interval: {FRAME_INTERVAL*1000:.1f}ms")
        
        # Wait for BLE server to fully initialize
        time.sleep(1)
        logger.info("Inference engine ready")

        next_time = time.time()
        while not self.stop_event.is_set():
            frame_start = time.time()
            
            # Stage 1: Capture frame from camera (5-8ms simulation)
            capture_start = time.time()
            time.sleep(random.uniform(0.005, 0.008))
            capture_time = (time.time() - capture_start) * 1000
            
            # Stage 2: Preprocessing (resize, normalize, convert format - 3-5ms)
            preprocess_start = time.time()
            time.sleep(random.uniform(0.003, 0.005))
            preprocess_time = (time.time() - preprocess_start) * 1000
            
            # Stage 3: Neural network inference (12-18ms)
            inference_start = time.time()
            time.sleep(random.uniform(0.012, 0.018))
            prediction = random.randint(0, 9)
            inference_time = (time.time() - inference_start) * 1000
            
            # Calculate total processing time
            processing_time = (time.time() - frame_start) * 1000
            
            # Clear queue if full to avoid backlog (keep only latest prediction)
            # This ensures we always send the most recent result
            try:
                old_value = self.queue.get_nowait()
                logger.debug(f"Dropped old prediction: {old_value}")
            except Empty:
                pass

            # Put new prediction in queue
            try:
                self.queue.put(prediction, block=False)
            except Exception as e:
                logger.warning(f"Failed to queue prediction: {e}")
                
            self.frame_id += 1

            # Maintain precise FPS timing
            next_time += FRAME_INTERVAL
            sleep_time = next_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
                sleep_ms = sleep_time * 1000
            else:
                sleep_ms = 0
                # Log if we're falling behind schedule
                logger.warning(f"Frame {self.frame_id} behind by {-sleep_time*1000:.1f}ms")
            
            # Calculate total frame time (processing + wait)
            total_frame_time = (time.time() - frame_start) * 1000
            
            logger.info(
                f"Frame {self.frame_id-1:05d} | Pred: {prediction} | "
                f"Cap: {capture_time:5.2f}ms + "
                f"Pre: {preprocess_time:5.2f}ms + "
                f"Inf: {inference_time:5.2f}ms = "
                f"Process: {processing_time:6.2f}ms + "
                f"Wait: {sleep_ms:5.2f}ms = "
                f"Frame: {total_frame_time:6.2f}ms"
            )


# =============================================================================
# GATT SERVER OBJECTS (BlueZ D-Bus Implementation)
# =============================================================================

class Application(dbus.service.Object):
    """
    Root GATT Application object.
    
    BlueZ uses the D-Bus ObjectManager interface to discover GATT services
    and characteristics. This class implements the ObjectManager that returns
    all GATT objects (services and characteristics) in our application.
    
    D-Bus Path: /
    Interface: org.freedesktop.DBus.ObjectManager
    """
    
    def __init__(self, bus):
        """
        Initialize the GATT application.
        
        Args:
            bus: D-Bus system bus connection
        """
        self.path = '/'
        self.services = []
        super().__init__(bus, self.path)
        logger.debug("GATT Application created")

    def add_service(self, service):
        """Add a GATT service to the application"""
        self.services.append(service)
        logger.debug(f"Service added: {service.uuid}")

    @dbus.service.method('org.freedesktop.DBus.ObjectManager',
                         out_signature='a{oa{sa{sv}}}')
    def GetManagedObjects(self):
        """
        Return all GATT objects (services and characteristics).
        
        BlueZ calls this method when our application is registered to discover
        the structure of our GATT server. The returned dictionary maps D-Bus
        object paths to their interfaces and properties.
        
        Returns:
            Dictionary of {path: {interface: {property: value}}}
        """
        objects = {}
        
        # Add all services
        for s in self.services:
            objects[s.get_path()] = s.get_properties()
            
            # Add all characteristics for each service
            for c in s.characteristics:
                objects[c.get_path()] = c.get_properties()
        
        logger.debug(f"GetManagedObjects returned {len(objects)} objects")
        return objects


class Service(dbus.service.Object):
    """
    GATT Service implementation.
    
    A GATT service is a collection of characteristics that provide related
    functionality. Our service contains a single characteristic for inference data.
    
    D-Bus Path: /org/bluez/example/service0
    Interface: org.bluez.GattService1
    """
    
    def __init__(self, bus, index):
        """
        Initialize a GATT service.
        
        Args:
            bus: D-Bus system bus connection
            index: Unique index for this service (for D-Bus path)
        """
        self.path = f'/org/bluez/example/service{index}'
        self.bus = bus
        self.uuid = SERVICE_UUID
        self.primary = True  # Primary service (vs. secondary/included service)
        self.characteristics = []
        super().__init__(bus, self.path)
        logger.debug(f"Service created: {self.uuid}")

    def get_path(self):
        """Return D-Bus object path"""
        return dbus.ObjectPath(self.path)

    def add_characteristic(self, characteristic):
        """Add a characteristic to this service"""
        self.characteristics.append(characteristic)
        logger.debug(f"Characteristic added to service: {characteristic.uuid}")

    def get_properties(self):
        """
        Return service properties for BlueZ.
        
        Properties:
            UUID: Service identifier (128-bit UUID)
            Primary: Whether this is a primary service
            Characteristics: Array of characteristic object paths
        """
        return {
            'org.bluez.GattService1': {
                'UUID': self.uuid,
                'Primary': self.primary,
                'Characteristics': dbus.Array(
                    [c.get_path() for c in self.characteristics], 
                    signature='o')
            }
        }

    @dbus.service.method('org.freedesktop.DBus.Properties',
                         in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        """
        D-Bus Properties interface method.
        
        BlueZ calls this to retrieve all properties of the service.
        """
        if interface != 'org.bluez.GattService1':
            raise dbus.exceptions.DBusException(
                'org.freedesktop.DBus.Error.InvalidArgs',
                'Invalid interface')
        return self.get_properties()['org.bluez.GattService1']


class Characteristic(dbus.service.Object):
    """
    GATT Characteristic implementation.
    
    A characteristic is a data value that can be read, written, or notified.
    Our characteristic supports:
        - Read: Client can read current value
        - Notify: Server can push updates to subscribed clients
    
    The characteristic holds a 4-byte integer representing the inference prediction.
    
    D-Bus Path: /org/bluez/example/service0/char0
    Interface: org.bluez.GattCharacteristic1
    """
    
    def __init__(self, bus, index, service):
        """
        Initialize a GATT characteristic.
        
        Args:
            bus: D-Bus system bus connection
            index: Unique index for this characteristic
            service: Parent service object
        """
        self.path = f'{service.path}/char{index}'
        self.bus = bus
        self.uuid = CHAR_UUID
        self.service = service
        
        # Flags define what operations are allowed on this characteristic
        # 'read': Client can read the value
        # 'notify': Client can subscribe to value change notifications
        self.flags = ['read', 'notify']
        
        self.notifying = False  # Whether any client has enabled notifications
        self.value = [dbus.Byte(0)] * 4  # 4-byte value buffer
        super().__init__(bus, self.path)
        logger.debug(f"Characteristic created: {self.uuid}")

    def get_path(self):
        """Return D-Bus object path"""
        return dbus.ObjectPath(self.path)

    def get_properties(self):
        """
        Return characteristic properties for BlueZ.
        
        Properties:
            Service: Parent service object path
            UUID: Characteristic identifier (128-bit UUID)
            Flags: Supported operations (read, write, notify, etc.)
            Value: Current data value
        """
        return {
            'org.bluez.GattCharacteristic1': {
                'Service': self.service.get_path(),
                'UUID': self.uuid,
                'Flags': self.flags,
                'Value': self.value
            }
        }

    @dbus.service.method('org.freedesktop.DBus.Properties',
                         in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        """
        D-Bus Properties interface method.
        
        BlueZ calls this to retrieve all properties of the characteristic.
        """
        if interface != 'org.bluez.GattCharacteristic1':
            raise dbus.exceptions.DBusException(
                'org.freedesktop.DBus.Error.InvalidArgs',
                'Invalid interface')
        return self.get_properties()['org.bluez.GattCharacteristic1']

    @dbus.service.method('org.bluez.GattCharacteristic1',
                        in_signature='a{sv}', out_signature='ay')
    def ReadValue(self, options):
        """
        Handle read request from BLE client.
        
        When a client reads this characteristic, this method is called.
        Returns the current value as a byte array.
        
        Args:
            options: Dictionary of options (e.g., offset, device)
            
        Returns:
            Byte array containing the current value
        """
        logger.info(f"Read request | Value: {int.from_bytes(bytes(self.value), 'little')}")
        return self.value

    @dbus.service.method('org.bluez.GattCharacteristic1')
    def StartNotify(self):
        """
        Enable notifications for the connected client.
        
        When a client subscribes to notifications (writes to Client Characteristic
        Configuration Descriptor), BlueZ calls this method. After this, we can
        send unsolicited updates to the client using PropertiesChanged signal.
        """
        if self.notifying:
            logger.debug('Already notifying')
            return
        self.notifying = True
        logger.info("Client subscribed to notifications")

    @dbus.service.method('org.bluez.GattCharacteristic1')
    def StopNotify(self):
        """
        Disable notifications for the connected client.
        
        Called when client unsubscribes from notifications or disconnects.
        """
        if not self.notifying:
            logger.debug('Not notifying')
            return
        self.notifying = False
        logger.info("Client unsubscribed from notifications")

    @dbus.service.signal('org.freedesktop.DBus.Properties',
                         signature='sa{sv}as')
    def PropertiesChanged(self, iface, changed, invalidated):
        """
        D-Bus signal for property changes.
        
        This signal is used to implement GATT notifications. When we emit this
        signal with a changed 'Value' property, BlueZ sends a GATT notification
        to all subscribed clients.
        
        Args:
            iface: Interface name (org.bluez.GattCharacteristic1)
            changed: Dictionary of changed properties {property: new_value}
            invalidated: List of invalidated property names
        """
        pass

    def notify(self, value):
        """
        Send a notification with a new prediction value.
        
        This is the main method for broadcasting inference results. It converts
        the integer prediction to a 4-byte array and emits a PropertiesChanged
        signal, which BlueZ translates into a BLE GATT notification.
        
        Args:
            value: Integer prediction value (0-9)
            
        Returns:
            True if notification was sent, False if no clients are subscribed
        """
        if not self.notifying:
            return False
        
        # Convert integer to 4 bytes (little-endian format)
        # Little-endian: least significant byte first (e.g., 256 = 0x00 0x01 0x00 0x00)
        data = value.to_bytes(4, 'little', signed=True)
        self.value = [dbus.Byte(b) for b in data]
        
        # Emit PropertiesChanged signal to trigger GATT notification
        self.PropertiesChanged(
            'org.bluez.GattCharacteristic1',
            {'Value': self.value},  # Changed properties
            []                       # Invalidated properties (none)
        )
        
        logger.info(f"SENT | Value: {value} | Bytes: {' '.join(f'{b:02x}' for b in data)}")
        return True


class Advertisement(dbus.service.Object):
    """
    BLE Advertisement implementation.
    
    Advertisements are broadcast packets that allow BLE clients to discover
    our device. This contains:
        - Device name (appears in scan results)
        - Service UUIDs (helps clients find relevant services)
        - Advertisement type (peripheral = connectable device)
    
    D-Bus Path: /org/bluez/example/advertisement0
    Interface: org.bluez.LEAdvertisement1
    """
    
    def __init__(self, bus, index):
        """
        Initialize a BLE advertisement.
        
        Args:
            bus: D-Bus system bus connection
            index: Unique index for this advertisement
        """
        self.path = f'/org/bluez/example/advertisement{index}'
        self.bus = bus
        self.ad_type = 'peripheral'  # Connectable peripheral device
        self.local_name = DEVICE_NAME
        self.service_uuids = [SERVICE_UUID]
        super().__init__(bus, self.path)
        logger.debug(f"Advertisement created: {self.local_name}")

    def get_path(self):
        """Return D-Bus object path"""
        return dbus.ObjectPath(self.path)

    def get_properties(self):
        """
        Return advertisement properties for BlueZ.
        
        Properties:
            Type: Advertisement type (peripheral, broadcast)
            LocalName: Device name shown in scan results
            ServiceUUIDs: List of service UUIDs to advertise
        """
        properties = {
            'Type': self.ad_type,
            'LocalName': dbus.String(self.local_name),
            'ServiceUUIDs': dbus.Array(self.service_uuids, signature='s')
        }
        return {
            'org.bluez.LEAdvertisement1': properties
        }

    @dbus.service.method('org.freedesktop.DBus.Properties',
                         in_signature='s', out_signature='a{sv}')
    def GetAll(self, interface):
        """
        D-Bus Properties interface method.
        
        BlueZ calls this to retrieve advertisement properties.
        """
        if interface != 'org.bluez.LEAdvertisement1':
            raise dbus.exceptions.DBusException(
                'org.freedesktop.DBus.Error.InvalidArgs',
                'Invalid interface')
        return self.get_properties()['org.bluez.LEAdvertisement1']

    @dbus.service.method('org.bluez.LEAdvertisement1', out_signature='')
    def Release(self):
        """
        Called by BlueZ when advertisement is unregistered.
        
        This happens during shutdown or when advertising is stopped.
        """
        logger.info("Advertisement released")


# =============================================================================
# BLE SERVER MANAGER
# =============================================================================

class BLEServer:
    """
    Main BLE server that orchestrates GATT service and advertising.
    
    This class:
        1. Initializes D-Bus connection to BlueZ daemon
        2. Creates GATT application with services and characteristics
        3. Registers GATT services with BlueZ
        4. Starts BLE advertising
        5. Processes inference queue and sends notifications
    
    BlueZ Architecture:
        Application (us) <--(D-Bus)--> BlueZ daemon <--> Bluetooth Hardware
        
        We communicate with the BlueZ daemon via D-Bus, and BlueZ handles
        all low-level BLE protocol details (connections, ATT, GATT, etc.)
    """
    
    def __init__(self, queue, stop_event):
        """
        Initialize the BLE server.
        
        Args:
            queue: Queue for receiving inference predictions
            stop_event: Event to signal shutdown
        """
        self.queue = queue
        self.stop_event = stop_event

        # Initialize D-Bus mainloop FIRST
        # This integrates D-Bus with GLib mainloop for event processing
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        
        # Connect to system D-Bus (where BlueZ lives)
        self.bus = dbus.SystemBus()
        logger.debug("Connected to D-Bus system bus")

        # Create GATT application structure
        self.app = Application(self.bus)
        
        # Create service
        self.service = Service(self.bus, 0)
        self.app.add_service(self.service)
        
        # Create characteristic
        self.char = Characteristic(self.bus, 0, self.service)
        self.service.add_characteristic(self.char)

        # Create GLib mainloop for event processing
        self.mainloop = GLib.MainLoop()
        logger.debug("GLib mainloop created")

    def find_adapter(self):
        """
        Find the BLE adapter (Bluetooth controller).
        
        Queries BlueZ via D-Bus to find available Bluetooth adapters.
        On Jetson Nano Orin, this is typically the built-in BLE controller.
        
        Returns:
            D-Bus object path of the adapter (e.g., /org/bluez/hci0)
            
        Raises:
            RuntimeError: If no adapter is found
        """
        logger.debug("Searching for BLE adapter...")
        
        # Get BlueZ ObjectManager to enumerate all objects
        remote_om = dbus.Interface(
            self.bus.get_object('org.bluez', '/'),
            'org.freedesktop.DBus.ObjectManager')
        
        # Get all managed objects (adapters, devices, etc.)
        objects = remote_om.GetManagedObjects()
        
        # Find object with GattManager1 interface (indicates BLE capability)
        for path, interfaces in objects.items():
            if 'org.bluez.GattManager1' in interfaces:
                logger.debug(f"Found adapter: {path}")
                return path
        
        raise RuntimeError("No BLE adapter found. Is Bluetooth enabled?")

    def register_app_cb(self):
        """Callback: GATT application registration succeeded"""
        logger.info("GATT application registered with BlueZ")

    def register_app_error_cb(self, error):
        """Callback: GATT application registration failed"""
        logger.error(f"Failed to register GATT application: {error}")
        self.mainloop.quit()

    def register_ad_cb(self):
        """Callback: Advertisement registration succeeded"""
        logger.info(f"Advertising started")
        logger.info(f"Device name: '{DEVICE_NAME}'")
        logger.info(f"Service UUID: {SERVICE_UUID}")

    def register_ad_error_cb(self, error):
        """Callback: Advertisement registration failed"""
        logger.error(f"Failed to register advertisement: {error}")
        self.mainloop.quit()

    def start(self):
        """
        Start the BLE server.
        
        This method:
            1. Finds the BLE adapter
            2. Registers the GATT application
            3. Starts advertising
            4. Runs the GLib mainloop (blocking)
        """
        logger.info("=" * 60)
        logger.info("BLE Inference Server Starting")
        logger.info("=" * 60)
        
        try:
            # Find BLE adapter
            adapter_path = self.find_adapter()
            logger.info(f"BLE adapter: {adapter_path}")
            
            # Get adapter D-Bus object
            adapter_obj = self.bus.get_object('org.bluez', adapter_path)
            
            # Get GATT Manager interface (for registering services)
            gatt_manager = dbus.Interface(
                adapter_obj, 
                'org.bluez.GattManager1')
            logger.debug("GattManager interface acquired")
            
            # Get LE Advertising Manager interface (for advertising)
            ad_manager = dbus.Interface(
                adapter_obj, 
                'org.bluez.LEAdvertisingManager1')
            logger.debug("LEAdvertisingManager interface acquired")
            
            # Register our GATT application with BlueZ
            # This exposes our services and characteristics to BLE clients
            logger.info("Registering GATT application...")
            gatt_manager.RegisterApplication(
                self.app.path, {},
                reply_handler=self.register_app_cb,
                error_handler=self.register_app_error_cb
            )
            
            # Register advertisement to make device discoverable
            logger.info("Registering advertisement...")
            self.ad = Advertisement(self.bus, 0)
            ad_manager.RegisterAdvertisement(
                self.ad.get_path(), {},
                reply_handler=self.register_ad_cb,
                error_handler=self.register_ad_error_cb
            )
            
            logger.info("=" * 60)
            logger.info(f"Server active and broadcasting at {FPS} FPS")
            logger.info(f"   Waiting for client connections...")
            logger.info("=" * 60)
            
            # Add periodic queue processor (runs every 50ms)
            GLib.timeout_add(50, self.process_queue)
            logger.debug("Queue processor scheduled (50ms interval)")
            
            # Add stop event checker (runs every 100ms)
            GLib.timeout_add(100, self.check_stop)
            logger.debug("Stop checker scheduled (100ms interval)")
            
            # Run GLib mainloop (blocks until quit)
            logger.info("Entering mainloop...")
            self.mainloop.run()
            
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            raise

    def process_queue(self):
        """
        Process inference queue and send BLE notifications.
        
        This function is called periodically by GLib mainloop (every 50ms).
        It checks for new predictions in the queue and sends them via
        GATT notifications to subscribed clients.
        
        Returns:
            True to continue periodic execution, False to stop
        """
        try:
            # Try to get a prediction from the queue (non-blocking)
            value = self.queue.get_nowait()
            
            # Send notification to subscribed clients
            if self.char.notify(value):
                logger.debug(f"Notification sent successfully")
            else:
                logger.debug(f"No clients subscribed, prediction dropped: {value}")
                
        except Empty:
            # Queue is empty, nothing to send
            pass
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
        
        # Continue processing if not stopped
        return not self.stop_event.is_set()

    def check_stop(self):
        """
        Check if shutdown has been requested.
        
        This function is called periodically by GLib mainloop (every 100ms).
        When stop_event is set, it quits the mainloop to shutdown gracefully.
        
        Returns:
            True to continue periodic execution, False to stop
        """
        if self.stop_event.is_set():
            logger.info("Stop event detected, shutting down mainloop...")
            self.mainloop.quit()
            return False
        return True


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for the BLE Inference Server.
    
    Sets up signal handlers, starts the inference engine thread,
    and runs the BLE server (blocking).
    """
    # Create thread coordination primitives
    stop_event = threading.Event()  # Signals shutdown to all threads
    queue = Queue(maxsize=1)        # Passes predictions from inference to BLE

    def signal_handler(signum, frame):
        """Handle shutdown signals (Ctrl+C, SIGTERM)"""
        logger.info(f"\nReceived signal {signum}, shutting down gracefully...")
        stop_event.set()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command
    logger.debug("Signal handlers registered")

    # Start inference engine thread
    logger.info("Starting inference engine thread...")
    engine = InferenceEngine(queue, stop_event)
    engine.start()

    # Start BLE server (blocks until shutdown)
    try:
        logger.info("Starting BLE server...")
        server = BLEServer(queue, stop_event)
        server.start()
    except KeyboardInterrupt:
        logger.info("\nKeyboard interrupt received")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        # Ensure clean shutdown
        stop_event.set()
        logger.info("Waiting for threads to finish...")
        time.sleep(0.5)
        logger.info("Server stopped cleanly")


if __name__ == "__main__":
    main()