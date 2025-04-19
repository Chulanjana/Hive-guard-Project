import adafruit_dht
import board
import time

dht_device_inside = adafruit_dht.DHT22(board.D4)
dht_device_outside = adafruit_dht.DHT11(board.D17)


def read_temp_hum_inside():
    for _ in range(3):
        try:
            temperature = dht_device_inside.temperature
            humidity = dht_device_inside.humidity
            if temperature is not None and humidity is not None:
                return round(temperature, 2), round(humidity, 2)
        except RuntimeError:
            time.sleep(1)
    return None, None


def read_temp_hum_outside():
    for _ in range(3):
        try:
            temperature = dht_device_outside.temperature
            humidity = dht_device_outside.humidity
            if temperature is not None and humidity is not None:
                return round(temperature, 2), round(humidity, 2)
        except RuntimeError:
            time.sleep(1)
    return None, None
