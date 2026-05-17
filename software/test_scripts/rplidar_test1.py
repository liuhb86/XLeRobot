from os import path
from datetime import datetime
 
from rplidar import RPLidar
 
 
BAUD_RATE: int = 115200
TIMEOUT: int = 1
 
MAC_DEVICE_PATH: str = '/dev/cu.usbserial-0001'
LINUX_DEVICE_PATH: str = '/dev/ttyUSB0'
DEVICE_PATH: str = LINUX_DEVICE_PATH
 
 
if __name__ == '__main__':
 
    if path.exists(DEVICE_PATH):
 
        print(f'Found RPLidar on path: {DEVICE_PATH}')
 
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
 
        print(f'Date and time: {dt_string}')
 
        lidar = RPLidar(port=DEVICE_PATH, baudrate=BAUD_RATE, timeout=TIMEOUT)
 
        info = lidar.get_info()
        for key, value in info.items():
            print(f'{key.capitalize()}: {value}')
 
        health = lidar.get_health()
        print(f'Health: {health}')
 
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
    else:
        print(f'No device found for: {DEVICE_PATH}')
