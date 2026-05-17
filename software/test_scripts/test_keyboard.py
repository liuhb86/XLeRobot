import time
import traceback
import logging


def test1():
    from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
    from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig

    keyboard_config = KeyboardTeleopConfig()
    kb = KeyboardTeleop(keyboard_config)

    kb.connect()

    while True:
        try:
            # Get keyboard input
            keyboard_action = kb.get_action()
            print(keyboard_action)
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("User interrupted program")
            break
        except Exception as e:
            print(f"P control loop error: {e}")
            traceback.print_exc()
            break
    kb.disconnect()

def _on_press(key):
    print("press", key)

def _on_release(self, key):
    print("release", key)
    
def test2():
    from pynput import keyboard
    listener = keyboard.Listener(
                    on_press=_on_press,
                    on_release=_on_release,
                )
    listener.start()
    while True:
        time.sleep(1)
    listener.stop()

import asyncio

async def test3():
    from curtsies import Input
    print("Listening for keys... (ESC to quit)")
    with Input(keynames='curtsies') as input_generator:
        for key in input_generator:
            if key == '<ESC>':
                break
            print(f'Key pressed: {key}')
            # You can await other things here!

import sys
import tty
import termios
import select

def is_data():
    # Checks if there is a character waiting in the input buffer
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def test4():
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        print("Reading async... (Press 'q' to quit)")

        while True:
            if is_data():
                char = sys.stdin.read(1)
                if char == 'q':
                    break
                print(f"You pressed: {char}")
            
            # Do other async work here
            #print(".", end="", flush=True)
    finally:
        # Always restore terminal settings or your shell will act weird!
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

#asyncio.run(test3())
test4()