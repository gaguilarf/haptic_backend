#!/usr/bin/env python3
"""
Servidor BLE mínimo - Envía números incrementales cada segundo

Este es el ejemplo más simple posible de un servidor BLE que envía datos.
Usa BlueZ D-Bus API (la ÚNICA forma de crear servidores BLE en Linux con Python).

NOTA: Bleak NO puede hacer esto. Bleak es solo cliente.
"""

import dbus
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib
import time
import threading

# UUIDs - cambialos si quieres
SERVICE_UUID = "12345678-1234-5678-1234-56789abcdef0"
CHAR_UUID = "12345678-1234-5678-1234-56789abcdef1"

# Estado global
counter = 0


class Characteristic(dbus.service.Object):
    """Característica BLE que envía datos"""
    
    def __init__(self, bus, index, service):
        self.path = service.path + '/char' + str(index)
        self.bus = bus
        self.uuid = CHAR_UUID
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
        print(f'📖 Cliente leyó: {self.value}')
        return self.value
    
    @dbus.service.method('org.bluez.GattCharacteristic1')
    def StartNotify(self):
        if self.notifying:
            return
        self.notifying = True
        print('✅ Cliente activó notificaciones')
    
    @dbus.service.method('org.bluez.GattCharacteristic1')
    def StopNotify(self):
        if not self.notifying:
            return
        self.notifying = False
        print('❌ Cliente desactivó notificaciones')
    
    @dbus.service.signal('org.freedesktop.DBus.Properties',
                         signature='sa{sv}as')
    def PropertiesChanged(self, interface, changed, invalidated):
        pass
    
    def send_value(self, value):
        """Envía un valor vía notificación"""
        if not self.notifying:
            return False
        
        # Convertir int a bytes
        self.value = [dbus.Byte(b) for b in value.to_bytes(4, 'little', signed=True)]
        
        # Enviar notificación
        self.PropertiesChanged(
            'org.bluez.GattCharacteristic1',
            {'Value': self.value},
            []
        )
        print(f'📤 Enviado: {value}')
        return True


class Service(dbus.service.Object):
    """Servicio BLE"""
    
    def __init__(self, bus, index):
        self.path = '/org/bluez/example/service' + str(index)
        self.bus = bus
        self.uuid = SERVICE_UUID
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


class Application(dbus.service.Object):
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


class Advertisement(dbus.service.Object):
    """Anuncio BLE"""
    
    def __init__(self, bus, index):
        self.path = '/org/bluez/example/advertisement' + str(index)
        self.bus = bus
        dbus.service.Object.__init__(self, bus, self.path)
    
    def get_properties(self):
        return {
            'org.bluez.LEAdvertisement1': {
                'Type': 'peripheral',
                'ServiceUUIDs': dbus.Array([SERVICE_UUID], signature='s'),
                'LocalName': dbus.String('SimpleServer'),
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
        print('Anuncio liberado')


def find_adapter(bus):
    """Encuentra el adaptador Bluetooth"""
    remote_om = dbus.Interface(bus.get_object('org.bluez', '/'),
                                'org.freedesktop.DBus.ObjectManager')
    objects = remote_om.GetManagedObjects()
    
    for o, props in objects.items():
        if 'org.bluez.GattManager1' in props.keys():
            return o
    return None


def producer_thread(characteristic):
    """Thread que genera y envía datos"""
    global counter
    
    print('🔄 Thread productor iniciado')
    time.sleep(2)  # Esperar a que todo esté listo
    
    while True:
        counter += 1
        characteristic.send_value(counter)
        time.sleep(1)  # Enviar cada segundo


def main():
    global counter
    
    print('=' * 60)
    print('🚀 SERVIDOR BLE MÍNIMO')
    print('=' * 60)
    print(f'Servicio UUID: {SERVICE_UUID}')
    print(f'Característica UUID: {CHAR_UUID}')
    print('=' * 60)
    
    # Inicializar D-Bus
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()
    
    # Encontrar adaptador
    adapter_path = find_adapter(bus)
    if not adapter_path:
        print('❌ ERROR: Adaptador Bluetooth no encontrado')
        return
    
    print(f'✅ Adaptador encontrado: {adapter_path}')
    
    # Crear aplicación
    app = Application(bus)
    
    # Crear servicio
    service = Service(bus, 0)
    app.add_service(service)
    
    # Crear característica
    char = Characteristic(bus, 0, service)
    service.add_characteristic(char)
    
    # Registrar aplicación
    service_manager = dbus.Interface(
        bus.get_object('org.bluez', adapter_path),
        'org.bluez.GattManager1')
    
    service_manager.RegisterApplication(app.get_path(), {},
                                       reply_handler=lambda: print('✅ Aplicación registrada'),
                                       error_handler=lambda e: print(f'❌ Error: {e}'))
    
    # Crear y registrar anuncio
    ad = Advertisement(bus, 0)
    ad_manager = dbus.Interface(
        bus.get_object('org.bluez', adapter_path),
        'org.bluez.LEAdvertisingManager1')
    
    ad_manager.RegisterAdvertisement(ad.get_path(), {},
                                    reply_handler=lambda: print('✅ Anuncio registrado'),
                                    error_handler=lambda e: print(f'❌ Error: {e}'))
    
    print('=' * 60)
    print('📡 Servidor BLE activo y anunciando como "SimpleServer"')
    print('🔢 Enviando contador cada 1 segundo...')
    print('=' * 60)
    
    # Iniciar thread productor
    producer = threading.Thread(target=producer_thread, args=(char,), daemon=True)
    producer.start()
    
    # Ejecutar mainloop
    mainloop = GLib.MainLoop()
    try:
        mainloop.run()
    except KeyboardInterrupt:
        print('\n👋 Deteniendo servidor...')


if __name__ == '__main__':
    main()
