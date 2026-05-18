# turn off usb
uhubctl -l 2 -a 0
uhubctl -l 4 -a 0

# turn on usb
uhubctl -l 2 -a 1
uhubctl -l 4 -a 1
