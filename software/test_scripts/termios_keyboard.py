import sys
import tty
import termios
import select

class TermiosKeyboard:

    def __init__(self):
        self.old_settings = None

    def connect(self):
        if not self.old_settings:
            self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def disconnect(self):
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            self.old_settings = None
    
    def get_action(self):
        items = {}
        while TermiosKeyboard._is_data():
            char = sys.stdin.read(1)
            items[char] = None
        return items

    @staticmethod
    def _is_data():
        # Checks if there is a character waiting in the input buffer
        return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

