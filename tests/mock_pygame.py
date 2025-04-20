"""
Mock pygame module for testing without creating actual windows.
"""
import logging

# Set up logging
logger = logging.getLogger("mock_pygame")

# Dummy event types
QUIT = 1
KEYDOWN = 2

# Dummy key constants
K_ESCAPE = 27
K_SPACE = 32
K_UP = 273
K_DOWN = 274
K_RIGHT = 275
K_LEFT = 276
K_e = 101
K_d = 100
K_p = 112
K_t = 116
K_h = 104

# Mock Surface class
class Surface:
    def __init__(self, size):
        self.size = size
        logger.debug(f"Created mock surface of size {size}")
    
    def fill(self, color):
        logger.debug(f"Surface filled with color {color}")
        return self
    
    def blit(self, source, dest, area=None, special_flags=0):
        logger.debug(f"Blit to position {dest}")
        return self

# Mock Rect class
class Rect:
    def __init__(self, left, top, width, height):
        self.left = left
        self.top = top
        self.width = width
        self.height = height
        self.center = (left + width//2, top + height//2)
    
    def get_rect(self, **kwargs):
        return self

# Mock Clock class
class Clock:
    def __init__(self):
        pass
    
    def tick(self, framerate):
        return 0

# Mock font
class Font:
    def __init__(self, name, size):
        self.name = name
        self.size = size
    
    def render(self, text, antialias, color):
        # Return a mock surface
        return Surface((len(text) * self.size // 2, self.size))

# Mock module functions
def init():
    logger.debug("Mock pygame initialized")
    return 1

def quit():
    logger.debug("Mock pygame quit")

def display_set_mode(size):
    logger.debug(f"Mock display set to {size}")
    return Surface(size)

def display_flip():
    logger.debug("Mock display flipped")

def display_update():
    logger.debug("Mock display updated")

def get_events():
    return []

def draw_rect(surface, color, rect, width=0):
    logger.debug(f"Drawing rect at {rect} with color {color}")
    return rect

def draw_circle(surface, color, center, radius, width=0):
    logger.debug(f"Drawing circle at {center} with radius {radius} color {color}")
    return Rect(center[0]-radius, center[1]-radius, radius*2, radius*2)

def draw_line(surface, color, start, end, width=1):
    logger.debug(f"Drawing line from {start} to {end} with color {color}")
    return Rect(min(start[0], end[0]), min(start[1], end[1]), 
               abs(end[0] - start[0]), abs(end[1] - start[1]))

# Mock modules
display = type('DisplayModule', (), {
    'set_caption': lambda caption: logger.debug(f"Set caption to {caption}"),
    'set_mode': display_set_mode,
    'flip': display_flip,
    'update': display_update
})()

font = type('FontModule', (), {
    'init': lambda: None,
    'SysFont': Font
})()

draw = type('DrawModule', (), {
    'rect': draw_rect,
    'circle': draw_circle,
    'line': draw_line
})()

class MockPygame:
    """A class to hold all pygame mock functionality"""
    QUIT = QUIT
    KEYDOWN = KEYDOWN
    K_ESCAPE = K_ESCAPE
    K_SPACE = K_SPACE
    K_UP = K_UP
    K_DOWN = K_DOWN
    K_RIGHT = K_RIGHT
    K_LEFT = K_LEFT
    K_e = K_e
    K_d = K_d
    K_p = K_p
    K_t = K_t
    K_h = K_h
    
    init = init
    quit = quit
    display = display
    font = font
    draw = draw
    Surface = Surface
    Rect = Rect
    Clock = Clock
    event = type('EventModule', (), {'get': get_events})()

# Create a global instance
pygame = MockPygame() 