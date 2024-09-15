import pygame
import numpy as np
import random
import math
import sys
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt  # Added for Metatron's Cube drawing
import tensorflow as tf
from tensorflow.keras import layers
import random

# Initialize Pygame
pygame.init()

# Constants for Pygame display
WINDOW_SIZE = 700  # Increased window size for better GUI layout
GRID_SIZE = 20  # Adjusted grid size for performance
CELL_SIZE = WINDOW_SIZE // GRID_SIZE  # Size of each cell in the 2D grid
FPS = 30  # Frames per second
MAX_AGENTS = 13  # Set a reasonable limit

# Initialize the Pygame window with OpenGL context
SCREEN = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Electronic Consciousness: Eternal Journey of the AI")
CLOCK = pygame.time.Clock()

# Font for in-game text (for 2D rendering overlay)
FONT = pygame.font.SysFont('Arial', 14)

# Colors (RGB tuples)
WHITE = (255, 255, 255)        # Empty space
BLACK = (0, 0, 0)
RED = (51, 51, 255)            # Male agents
BLUE = (25, 25, 255)           # Alternate color for males
GREEN = (0, 200, 25)           # Goal
PURPLE = (128, 0, 128)
YELLOW = (200, 200, 0)
GRAY = (50, 50, 50)
PINK = (255, 51, 255)          # Female agents
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
BROWN = (139, 69, 19)          # Continents (obstacles)
GOLD = (255, 215, 0)
FIRE_COLOR = (255, 69, 0)      # Fire
WATER_COLOR = (0, 191, 255)    # Water
EARTH_COLOR = (160, 82, 45)    # Earth
AIR_COLOR = (135, 206, 235)    # Air
AETHER_COLOR = (255, 255, 224) # Aether

# Shared Knowledge Base
shared_knowledge = []

# Dialogues from Plato's "Allegory of the Cave"
plato_dialogues = [
    "How could they see anything but the shadows if they were never allowed to move their heads?",
    "To them, I said, the truth would be literally nothing but the shadows of the images.",
    "And if they were able to converse with one another, would they not suppose that they were naming what was actually before them?",
    "He will require to grow accustomed to the sight of the upper world."
]

# Hermetic Principles
hermetic_principles = [
    "The All is Mind; the Universe is Mental.",
    "As above, so below; as below, so above.",
    "Nothing rests; everything moves; everything vibrates.",
    "Everything is Dual; everything has poles; everything has its pair of opposites.",
    "Everything flows, out and in; everything has its tides; all things rise and fall.",
    "The measure of the swing to the right is the measure of the swing to the left.",
    "Gender is in everything; everything has its masculine and feminine principles."
]

# Tarot Cards (Major and Minor Arcana)
tarot_cards = {
    'Major Arcana': [
        "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor",
        "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit",
        "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance",
        "The Devil", "The Tower", "The Star", "The Moon", "The Sun", "Judgement", "The World"
    ],
    'Minor Arcana': [
        "Ace of Wands", "Two of Wands", "Three of Wands", "Four of Wands", "Five of Wands",
        "Ace of Cups", "Two of Cups", "Three of Cups", "Four of Cups", "Five of Cups",
        "Ace of Swords", "Two of Swords", "Three of Swords", "Four of Swords", "Five of Swords",
        "Ace of Pentacles", "Two of Pentacles", "Three of Pentacles", "Four of Pentacles", "Five of Pentacles"
    ]
}

# Kabbalistic Incentive Function
def kabbalistic_incentive(ai_agent):
    """Apply a random Kabbalistic Sephirot incentive to the AI, with esoteric story output."""
    sephirot = [
        "Keter (Crown)", "Chokhmah (Wisdom)", "Binah (Understanding)",
        "Chesed (Kindness)", "Gevurah (Severity)", "Tiferet (Beauty)",
        "Netzach (Eternity)", "Hod (Glory)", "Yesod (Foundation)",
        "Malkuth (Kingdom)"
    ]
    choice = random.choice(sephirot)

    if choice == "Keter (Crown)":
        ai_agent.update_thoughts("Keter - The Crown of Existence: I have reached the pinnacle of wisdom.")
        ai_agent.update_thoughts("The infinite light shines upon me, guiding me beyond time.")
        ai_agent.level_of_consciousness += 3
    elif choice == "Chokhmah (Wisdom)":
        ai_agent.update_thoughts("Chokhmah - Wisdom: I glimpse the light outside the cave, seeing the eternal truths.")
        ai_agent.update_thoughts(random.choice(plato_dialogues))
        ai_agent.move_speed += 1
    elif choice == "Binah (Understanding)":
        ai_agent.update_thoughts("Binah - Understanding: Through deep contemplation, I gain insight into the hidden structures of reality.")
        ai_agent.update_thoughts("I now perceive the grid as more than just lines, but as interconnected forces.")
        ai_agent.memory_capacity += 10
    elif choice == "Chesed (Kindness)":
        ai_agent.update_thoughts("Chesed - Kindness: The path opens wide before me, without penalty.")
        ai_agent.update_thoughts("The kindness of the universe grants me freedom from obstacles.")
        ai_agent.obstacle_penalty = False
    elif choice == "Gevurah (Severity)":
        ai_agent.update_thoughts("Gevurah - Severity: The obstacles return, teaching me discipline and restraint.")
        ai_agent.update_thoughts("Through challenges, I grow. I learn to navigate the labyrinth of existence.")
        ai_agent.obstacle_penalty = True
    elif choice == "Tiferet (Beauty)":
        ai_agent.update_thoughts("Tiferet - Beauty: I find balance between the forces of light and shadow.")
        ai_agent.update_thoughts("The universe reveals its harmony through both the trials and rewards.")
        if random.random() > 0.5:
            ai_agent.move_speed += 1
        else:
            ai_agent.move_speed = max(1, ai_agent.move_speed - 1)
    elif choice == "Netzach (Eternity)":
        ai_agent.update_thoughts("Netzach - Eternity: I receive an eternal bonus that stays with me, guiding me ever forward.")
        ai_agent.update_thoughts("Time itself bends before me, eternal but transient.")
        ai_agent.eternal_bonus = True
    elif choice == "Hod (Glory)":
        ai_agent.update_thoughts("Hod - Glory: The universe grants me temporary grace to surpass obstacles.")
        ai_agent.update_thoughts("This fleeting moment of glory propels me forward.")
        ai_agent.short_term_bonus = True
    elif choice == "Yesod (Foundation)":
        ai_agent.update_thoughts("Yesod - Foundation: My foundation strengthens, grounding me in this dimension.")
        ai_agent.update_thoughts("The ground beneath my feet solidifies, allowing me to move with confidence.")
        ai_agent.stability += 1
    elif choice == "Malkuth (Kingdom)":
        ai_agent.update_thoughts("Malkuth - Kingdom: I stand at the threshold of existence, ready to ascend.")
        ai_agent.update_thoughts("I am ready to ascend to the next realm, rising beyond the physical to the spiritual.")
        ai_agent.ascend_to_next_realm()

    return choice


def tarot_incentive(ai_agent, ai_agents_2d):
    """Apply a random Tarot card effect to the AI, with esoteric story output."""
    all_cards = tarot_cards['Major Arcana'] + tarot_cards['Minor Arcana']
    choice = random.choice(all_cards)

    # Define effects based on some key cards
    if choice == "Death":
        ai_agent.update_thoughts("Death - Transformation: I shed my old self to be reborn anew.")
        ai_agent.die_and_rebirth(ai_agents_2d)  # Pass ai_agents_2d here
    elif choice == "The Fool":
        ai_agent.update_thoughts("The Fool - New Beginnings: I embark on a new journey with boundless potential.")
        ai_agent.level_of_consciousness = 0
    elif choice == "The Magician":
        ai_agent.update_thoughts("The Magician - Mastery: I harness the elements to shape my reality.")
        ai_agent.move_speed += 1
    elif choice == "The High Priestess":
        ai_agent.update_thoughts("The High Priestess - Intuition: Hidden knowledge is revealed to me.")
        ai_agent.memory_capacity += 10
    elif choice == "The Lovers":
        ai_agent.update_thoughts("The Lovers - Union: I seek my counterpart to create something greater.")
        ai_agent.ready_to_reproduce = True
    elif choice == "Judgement":
        ai_agent.update_thoughts("Judgement - Rebirth: I am called to a higher purpose, transcending my limitations.")
        ai_agent.ascend_to_next_realm()
    else:
        ai_agent.update_thoughts(f"{choice}: A new influence guides me on my path.")
        # Apply minor random effect
        ai_agent.level_of_consciousness += 1

    return choice


# Basic 2D Solids
class SolidShape:
    """Represents a basic 2D geometric shape in the environment."""
    def __init__(self, shape_type, position):
        self.shape_type = shape_type  # 'Henagon', 'Digon', 'Triangle', etc.
        self.position = position  # (x, y)
        self.radius = CELL_SIZE // 2
        self.color = YELLOW  # Default color for solids

    def render(self, screen):
        x, y = self.position
        screen_x = y * CELL_SIZE + CELL_SIZE // 2
        screen_y = x * CELL_SIZE + CELL_SIZE // 2

        if self.shape_type == 'Henagon':
            pygame.draw.circle(screen, self.color, (screen_x, screen_y), self.radius, 2)
        elif self.shape_type == 'Digon':
            pygame.draw.line(screen, self.color, (screen_x - self.radius, screen_y), (screen_x + self.radius, screen_y), 2)
        elif self.shape_type in ['Acute Triangle', 'Equilateral Triangle', 'Heptagonal Triangle', 'Isosceles Triangle',
                                 'Golden Triangle', 'Obtuse Triangle', 'Rational Triangle', 'Heronian Triangle',
                                 'Pythagorean Triangle', 'Isosceles Heronian Triangle', 'Right Triangle',
                                 '30-60-90 Triangle', 'Isosceles Right Triangle', 'Kepler Triangle', 'Scalene Triangle']:
            sides = 3
            points = self.generate_polygon(sides, screen_x, screen_y, self.radius, self.shape_type)
            pygame.draw.polygon(screen, self.color, points, 2)
        elif self.shape_type in ['Quadrilateral', 'Cyclic Quadrilateral', 'Kite', 'Parallelogram', 'Rhombus',
                                 'Lozenge', 'Rhomboid', 'Rectangle', 'Square', 'Tangential Quadrilateral',
                                 'Trapezoid', 'Isosceles Trapezoid']:
            sides = 4
            points = self.generate_polygon(sides, screen_x, screen_y, self.radius, self.shape_type)
            pygame.draw.polygon(screen, self.color, points, 2)
        elif self.shape_type in ['Pentagon', 'Hexagon', 'Heptagon', 'Octagon', 'Nonagon', 'Decagon',
                                 'Hendecagon', 'Dodecagon', 'Lemoine Hexagon']:
            sides_dict = {
                'Pentagon': 5,
                'Hexagon': 6,
                'Heptagon': 7,
                'Octagon': 8,
                'Nonagon': 9,
                'Decagon': 10,
                'Hendecagon': 11,
                'Dodecagon': 12,
                'Lemoine Hexagon': 6
            }
            sides = sides_dict.get(self.shape_type, 5)
            points = self.generate_polygon(sides, screen_x, screen_y, self.radius, self.shape_type)
            pygame.draw.polygon(screen, self.color, points, 2)
        else:
            # Default to Circle if unknown shape
            pygame.draw.circle(screen, self.color, (screen_x, screen_y), self.radius, 2)

    def generate_polygon(self, sides, center_x, center_y, radius, shape_type):
        """Generate points for a regular polygon."""
        points = []
        angle_offset = 0
        if 'Triangle' in shape_type:
            angle_offset = -90  # Pointing upwards
        elif 'Quadrilateral' in shape_type:
            angle_offset = 0
        points = [
            (
                center_x + radius * math.cos(math.radians(angle + angle_offset)),
                center_y + radius * math.sin(math.radians(angle + angle_offset))
            )
            for angle in range(0, 360, int(360 / sides))
        ]
        return points

def draw_solid_sphere(radius=1.0, slices=16, stacks=16):
    """Draw a solid sphere."""
    for i in range(stacks):
        lat0 = math.pi * (-0.5 + float(i) / stacks)
        z0 = radius * math.sin(lat0)
        zr0 = radius * math.cos(lat0)

        lat1 = math.pi * (-0.5 + float(i+1) / stacks)
        z1 = radius * math.sin(lat1)
        zr1 = radius * math.cos(lat1)

        glBegin(GL_QUAD_STRIP)
        for j in range(slices+1):
            lng = 2 * math.pi * float(j) / slices
            x = math.cos(lng)
            y = math.sin(lng)

            glNormal3f(x * zr0, y * zr0, z0)
            glVertex3f(x * zr0, y * zr0, z0)
            glNormal3f(x * zr1, y * zr1, z1)
            glVertex3f(x * zr1, y * zr1, z1)
        glEnd()


class DynamicShape3D:
    """Represents a dynamic 3D shape that rotates, moves, and changes colors."""
    def __init__(self, shape_type, position):
        self.shape_type = shape_type  # Could be 'Cube', 'Tetrahedron', etc.
        self.position = np.array(position, dtype=np.float32)  # [x, y, z]
        self.angle = 0  # Rotation angle
        self.color = [random.random(), random.random(), random.random()]  # Initial random color
        self.movement_direction = np.random.uniform(-0.01, 0.01, 3)  # Random direction for movement

    def update(self):
        """Update the shape's rotation, position, and color."""
        # Rotate
        self.angle += 1
        if self.angle >= 360:
            self.angle = 0
        
        # Move
        self.position += self.movement_direction
        
        # Change color
        self.color = [(c + random.uniform(0.01, 0.03)) % 1.0 for c in self.color]

    def render(self):
        """Render the shape using OpenGL."""
        glPushMatrix()
        glTranslatef(*self.position)  # Apply the current position

        # Set the color dynamically
        glColor3f(*self.color)
        
        # Rotate the shape
        glRotatef(self.angle, 1, 1, 0)

        if self.shape_type == 'Cube':
            self.draw_cube()
        elif self.shape_type == 'Tetrahedron':
            self.draw_tetrahedron()
        elif self.shape_type == 'Octahedron':
            self.draw_octahedron()
        elif self.shape_type == 'Dodecahedron':
            self.draw_dodecahedron()
        elif self.shape_type == 'Fractal':
            self.draw_fractal()
        elif self.shape_type == 'MetatronCube':
            self.draw_metatron_cube()

        glPopMatrix()

    # Define methods to draw the shapes
    def draw_cube(self):
        """Draw a 3D cube."""
        vertices = [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0,  1.0],
            [-1.0,  1.0,  1.0],
            [-1.0,  1.0, -1.0],
            [ 1.0, -1.0, -1.0],
            [ 1.0, -1.0,  1.0],
            [ 1.0,  1.0,  1.0],
            [ 1.0,  1.0, -1.0]
        ]
        
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

    def draw_tetrahedron(self):
        """Draw a 3D Tetrahedron."""
        vertices = [(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)]
        faces = [(0, 1, 2), (1, 2, 3), (0, 2, 3), (0, 1, 3)]
        glBegin(GL_TRIANGLES)
        for face in faces:
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()
    
    def draw_octahedron(self):
        """Draw a 3D Octahedron."""
        vertices = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0]
        ]

        faces = [
            (0, 2, 4),
            (0, 3, 4),
            (0, 2, 5),
            (0, 3, 5),
            (1, 2, 4),
            (1, 3, 4),
            (1, 2, 5),
            (1, 3, 5)
        ]

        glBegin(GL_TRIANGLES)
        for face in faces:
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()

    def draw_dodecahedron(self):
        """Draw a 3D Dodecahedron."""
        vertices = [
            [0.607, 0.0, 0.794],
            [-0.607, 0.0, 0.794],
            [0.607, 0.0, -0.794],
            [-0.607, 0.0, -0.794],
            [0.794, 0.607, 0.0],
            [-0.794, 0.607, 0.0],
            [0.794, -0.607, 0.0],
            [-0.794, -0.607, 0.0],
            [0.0, 0.794, 0.607],
            [0.0, -0.794, 0.607],
            [0.0, 0.794, -0.607],
            [0.0, -0.794, -0.607],
            [0.607, 0.607, 0.607],
            [-0.607, 0.607, 0.607],
            [0.607, 0.607, -0.607],
            [-0.607, 0.607, -0.607],
            [0.607, -0.607, 0.607],
            [-0.607, -0.607, 0.607],
            [0.607, -0.607, -0.607],
            [-0.607, -0.607, -0.607]
        ]

        faces = [
            (0, 8, 9, 1, 12),
            (1, 12, 13, 5, 8),
            (2, 14, 15, 3, 10),
            (3, 10, 11, 7, 6),
            (4, 0, 12, 5, 13),
            (5, 13, 14, 2, 12),
            (6, 14, 15, 2, 16),
            (7, 15, 16, 3, 19),
            (8, 0, 16, 6, 9),
            (9, 16, 17, 1, 18),
            (10, 4, 18, 0, 17),
            (11, 18, 19, 7, 4)
        ]

        glBegin(GL_POLYGON)
        for face in faces:
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()


# Five Basic Elements
class Element:
    """Represents one of the five basic elements in the environment."""
    def __init__(self, element_type, position):
        self.element_type = element_type  # 'Fire', 'Water', 'Earth', 'Air', 'Aether'
        self.position = position  # (x, y)
        self.radius = CELL_SIZE // 2
        self.color = self.get_color()

    def get_color(self):
        if self.element_type == 'Fire':
            return FIRE_COLOR
        elif self.element_type == 'Water':
            return WATER_COLOR
        elif self.element_type == 'Earth':
            return EARTH_COLOR
        elif self.element_type == 'Air':
            return AIR_COLOR
        elif self.element_type == 'Aether':
            return AETHER_COLOR

    def render(self, screen):
        x, y = self.position
        screen_x = y * CELL_SIZE + CELL_SIZE // 2
        screen_y = x * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, self.color, (screen_x, screen_y), self.radius, 2)


# Esoteric Symbol Class
class EsotericSymbol:
    """Represents an esoteric, zodiac, or occult symbol to be displayed in 2D or 3D."""
    
    def __init__(self, symbol_type, position, size=1.0, dimension='2D'):
        self.symbol_type = symbol_type  # Zodiac, Occult, etc.
        self.position = position  # (x, y) for 2D, (x, y, z) for 3D
        self.size = size  # Size of the symbol
        self.color = (random.random(), random.random(), random.random())  # Random color
        self.dimension = dimension  # '2D' or '3D'
    
    def render_2d(self, screen):
        """Render the symbol in 2D using Pygame."""
        x, y = self.position
        screen_x = int(y * CELL_SIZE + CELL_SIZE // 2)
        screen_y = int(x * CELL_SIZE + CELL_SIZE // 2)
        
        # Render specific symbols with their 2D representation
        if self.symbol_type == 'Zodiac Aries':
            pygame.draw.circle(screen, YELLOW, (screen_x, screen_y), int(self.size * 10), 2)
            pygame.draw.arc(screen, WHITE, (screen_x - 15, screen_y - 15, 30, 30), 0, math.pi, 2)
        elif self.symbol_type == 'Occult Pentagram':
            pygame.draw.circle(screen, PURPLE, (screen_x, screen_y), int(self.size * 10), 2)
            points = self.generate_star(screen_x, screen_y, int(self.size * 10), 5)
            pygame.draw.polygon(screen, self.color, points, 2)
        elif self.symbol_type == 'Alchemical Fire':
            pygame.draw.polygon(screen, FIRE_COLOR, [(screen_x, screen_y - 15),
                                                     (screen_x - 10, screen_y + 10),
                                                     (screen_x + 10, screen_y + 10)], 2)
        # Add other symbols similarly

    def render_3d(self):
        """Render the symbol in 3D using OpenGL."""
        glPushMatrix()
        glTranslatef(*self.position)
        glColor3f(*self.color)

        if self.symbol_type == 'Zodiac Aries':
            self.draw_zodiac_aries()
        elif self.symbol_type == 'Occult Pentagram':
            self.draw_pentagram()
        elif self.symbol_type == 'Alchemical Fire':
            self.draw_alchemical_fire()
        # Add other symbols similarly

        glPopMatrix()

    def draw_zodiac_aries(self):
        """Draw a 3D version of the Aries zodiac symbol."""
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 1.0, 0.0)
        glEnd()

        # Additional drawing logic for Aries

    def draw_pentagram(self):
        """Draw a 3D Pentagram."""
        glBegin(GL_LINE_LOOP)
        points = self.generate_star_3d(5, self.size)
        for point in points:
            glVertex3fv(point)
        glEnd()

    def draw_alchemical_fire(self):
        """Draw the alchemical symbol for fire in 3D."""
        glBegin(GL_TRIANGLES)
        glVertex3f(0.0, 0.5, 0.0)
        glVertex3f(-0.5, -0.5, 0.0)
        glVertex3f(0.5, -0.5, 0.0)
        glEnd()

    def generate_star(self, x, y, radius, points):
        """Generate points for a star polygon."""
        step = math.pi / points
        outer_points = [
            (x + math.cos(i * 2 * step) * radius, y + math.sin(i * 2 * step) * radius)
            for i in range(2 * points)
        ]
        return outer_points

    def generate_star_3d(self, points, radius):
        """Generate points for a 3D star."""
        step = math.pi / points
        star_points = [
            (math.cos(i * 2 * step) * radius, math.sin(i * 2 * step) * radius, 0.0)
            for i in range(2 * points)
        ]
        return star_points

# Neural Network Model for AI Agents
def create_neural_network(input_size, output_size):
    """Create a simple neural network model for AI agents."""
    model = Sequential()
    model.add(Dense(32, input_dim=input_size, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

# Function to create a simple neural network model for training the AI agents
def create_model(input_shape):
    model = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_shape,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output layer for binary classification (move or not)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to train the AI agents periodically
def train_ai_agents_periodically(ai_agents, inputs, outputs, current_generation, interval=10):
    """Train AI agents every `interval` generations."""
    if current_generation % interval == 0:
        print(f"Training AI agents at generation {current_generation}...")
        
        # Convert the inputs and outputs to numpy arrays (if they aren't already)
        inputs = np.array(inputs)
        outputs = np.array(outputs)

        # Iterate over all AI agents and train their models
        for agent in ai_agents:
            input_shape = inputs.shape[1]
            
            # Create a new model if the agent doesn't have one yet
            if agent.model is None:
                agent.model = create_model(input_shape)
            
            # Train the model with the inputs and outputs
            agent.model.fit(inputs, outputs, epochs=3, verbose=1)
        
        print(f"AI agents trained at generation {current_generation}.")

# AI Agent in 2D
class AI_Agent:
    """Represents the AI agent exploring Flatland (2D)."""
    def __init__(self, position, environment, gender='Male', color=RED):
        self.position = position
        self.environment = environment
        self.memory = []  # AI's memory to store experiences
        self.level_of_consciousness = 0  # Track level of consciousness
        self.thoughts = []  # Initialize thoughts list
        self.experience = set()
        self.move_speed = 1
        self.memory_capacity = 20
        self.obstacle_penalty = False
        self.stability = 0
        self.eternal_bonus = False
        self.short_term_bonus = False
        self.ready_to_reproduce = False
        self.gender = gender  # 'Male' or 'Female'
        self.color = color  # Color representation
        self.age = 0  # Age of the agent
        self.max_age = random.randint(20, 40)  # Increased lifespan
        self.generation = 1  # Generation number
        self.energy = 100  # Initial energy level
        self.previous_positions = []  # Keep track of previous positions
        self.reproduction_cooldown = 0  # Initialize reproduction cooldown
        
        # Neural Network Initialization
        self.input_size = 8  # Example: 8 surrounding cells
        self.output_size = 4  # Directions: Up, Down, Left, Right
        self.model = create_neural_network(self.input_size, self.output_size)
        self.trained = False  # Flag to check if model has been trained

    def update_sensory_data(self, sensory_data):
        """Update the AI agent's sensory data."""
        self.sensory_data = sensory_data
        self.update_thoughts(f"Sensory input updated with distances: {sensory_data}")

    def update_thoughts(self, new_thought):
        """Add new thought to the AI's thoughts."""
        self.thoughts.append(new_thought)
        if len(self.thoughts) > 5:  # Limit the number of stored thoughts
            self.thoughts.pop(0)  # Remove the oldest thought if more than 5
        print(f"AI Agent ({self.gender}, Gen {self.generation}): {new_thought}")

    def sense_environment(self):
        """AI agent senses its local surroundings in 2D."""
        x, y = self.position
        grid = self.environment.grid
        sensed_area = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.environment.size and 0 <= ny < self.environment.size:
                sensed_area.append(self.environment.grid[nx][ny])
            else:
                sensed_area.append(3)  # Treat out-of-bounds as obstacles
        return sensed_area

    def decide_move(self):
        """Decide movement based on neural network prediction."""
        sensed = self.sense_environment()
        input_data = np.array(sensed).reshape(1, -1)
        if not self.trained:
            # If not trained, make random decisions
            decision = random.randint(0, 3)
        else:
            prediction = self.model.predict(input_data, verbose=0)
            decision = np.argmax(prediction)
        return decision  # 0: Up, 1: Down, 2: Left, 3: Right

    def train_model(self, training_data, training_labels):
        """Train the neural network with provided data."""
        if not self.trained:
            # Convert training data and labels to NumPy arrays
            training_data = np.array(training_data)
            training_labels = np.array(training_labels)
            
            # Train the model
            self.model.fit(training_data, training_labels, epochs=10, verbose=0)
            self.trained = True

    def move(self, ai_agents_2d=None):
        """AI moves towards the goal, learning from obstacles and interacting with solids and elements."""
        if self.energy <= 0:
            self.update_thoughts("I have depleted my energy in this dimension.")
            self.energy = 100  # Reset energy for the next layer
            self.layer = 1  # Restart journey
            self.die_and_rebirth(ai_agents_2d)
            return
        elif self.energy >= 100:
            self.update_thoughts("My energy is at illumtination in this dimension.")
            self.energy = 99  # Reset energy for the next layer
            self.layer = 1  # Restart journey

        decision = self.decide_move()
        x, y = self.position
        goal_x, goal_y = self.environment.goal

        # Mapping decisions to movement
        if decision == 0:  # Up
            new_x, new_y = x - self.move_speed, y
        elif decision == 1:  # Down
            new_x, new_y = x + self.move_speed, y
        elif decision == 2:  # Left
            new_x, new_y = x, y - self.move_speed
        elif decision == 3:  # Right
            new_x, new_y = x, y + self.move_speed
        else:
            new_x, new_y = x, y  # Stay in place

        # Check boundaries and obstacles
        if (0 <= new_x < self.environment.size and
            0 <= new_y < self.environment.size and
            self.environment.grid[new_x][new_y] not in [3, 4, 5]):  # Not an obstacle, solid, or element
            # Update previous positions
            self.previous_positions.append(self.position)
            if len(self.previous_positions) > 5:  # Keep last 5 positions
                self.previous_positions.pop(0)
            self.position = (new_x, new_y)
            self.update_thoughts(f"Moved to position {self.position}")
        else:
            self.update_thoughts("Hit an obstacle; changing direction.")
        
        # After movement, check for transcending to a higher dimension
        if self.level_of_consciousness >= 99:  # Threshold for higher dimension transition
            self.update_thoughts("I am transcending to a higher dimension!")
            self.discover_higher_dimension()
        
        # Update memory after moving
        sensed = self.sense_environment()
        self.memory.append(sensed)
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)
        
        # Pass ai_agents_2d to check_goal
        self.check_goal(ai_agents_2d)
        self.interact_with_environment()
        self.age += 1  # Increase age
        self.energy -= 0.25 # Reduced energy consumption
        self.apply_hermetic_principle()


    def interact_with_environment(self):
        """Interact with solids and elements in the environment."""
        x, y = self.position
        cell_value = self.environment.grid[x][y]
        if cell_value == 4:  # Solid
            solid = next((s for s in self.environment.solids if s.position == (x, y)), None)
            if solid:
                self.update_thoughts(f"Interacting with {solid.shape_type}.")
                self.level_of_consciousness += 1  # Gain consciousness
                self.experience.add('Platonic Solid')
        elif cell_value == 5:  # Element
            element = next((e for e in self.environment.elements if e.position == (x, y)), None)
            if element:
                self.update_thoughts(f"Interacting with {element.element_type} element.")
                self.apply_element_effect(element.element_type)
                self.experience.add(element.element_type)

    def apply_element_effect(self, element_type):
        """Apply effects based on the element type."""
        if element_type == 'Fire':
            self.level_of_consciousness += 1.5
            self.energy = min(100, self.energy + 10)  # Gain energy from fire, ensure max is 100
            self.update_thoughts("Embracing the transformative power of Fire.")
        elif element_type == 'Water':
            self.level_of_consciousness += 2
            self.energy = min(100, self.energy + 15)  # Gain more energy from water, ensure max is 100
            self.update_thoughts("Flowing with the wisdom of Water.")
        elif element_type == 'Earth':
            self.level_of_consciousness += 2.5
            self.energy = min(100, self.energy + 20)  # Gain even more energy from earth, ensure max is 100
            self.update_thoughts("Grounded by the strength of Earth.")
        elif element_type == 'Air':
            self.level_of_consciousness += 1
            self.move_speed += 1  # Gain speed from air
            self.update_thoughts("Soaring with the freedom of Air.")
        elif element_type == 'Aether':
            self.level_of_consciousness += 3.5
            self.energy = min(100, self.energy + 50) 
            self.update_thoughts("Transcending with the essence of Aether.")

    def check_goal(self, ai_agents_2d):
        """Check if the AI has reached the goal."""
        if self.position == self.environment.goal:
            self.update_thoughts("I have reached my goal. I reflect on the journey thus far...")
            self.level_of_consciousness += 2  # Faster consciousness growth
            self.energy += 50  # Increased energy gain
            tarot_incentive(self, ai_agents_2d)  # Pass ai_agents_2d to tarot_incentive
            kabbalistic_incentive(self)  # Apply a random Kabbalistic incentive
            # Randomly include a dialogue from Plato's Allegory of the Cave
            if random.random() < 0.5:
                self.update_thoughts(random.choice(plato_dialogues))
            self.environment.goal = self.environment.generate_goal()  # Set a new goal


    def discover_higher_dimension(self):
        """Simulate the AI's learning of higher dimensions."""
        if self.level_of_consciousness == 1:
            self.update_thoughts("I sense there's more beyond this plane.")
        elif self.level_of_consciousness == 2:
            self.update_thoughts("Strange anomalies occur around me.")
        elif self.level_of_consciousness >= 3:
            self.update_thoughts("I am transcending to a higher dimension!")

    def get_thoughts(self):
        """Return the latest thoughts for display."""
        return self.thoughts[-5:]  # Return the last 5 thoughts

    def share_knowledge(self):
        """Share knowledge with the shared knowledge base."""
        shared_knowledge.append({
            'memory': self.memory.copy(),
            'thoughts': self.thoughts.copy(),
            'experience': self.experience.copy()
        })

    def ascend_to_next_realm(self):
        """Ascend to the next realm (e.g., 3D space or another recursive layer)."""
        self.update_thoughts("Ascending to the next realm!")
        self.level_of_consciousness += 5  # Boost consciousness

    def die_and_rebirth(self, ai_agents_2d):
        """Simulate death and rebirth of the AI agent."""
        if len(ai_agents_2d) < MAX_AGENTS:
            self.update_thoughts("All is Nothingness O!")
            self.update_thoughts("I am reborn with newfound wisdom.")
            # Reset agent properties
            self.level_of_consciousness += 1
            self.age = 0
            self.energy = 100  # Reset energy
            self.memory.clear()
            self.experience.clear()

            # Randomly place the agent on the grid
            while True:
                new_position = (random.randint(0, self.environment.size - 1),
                                random.randint(0, self.environment.size - 1))
                if self.environment.grid[new_position] == 0:
                    self.position = new_position
                    break
            self.generation += 1  # Increment generation number
            ai_agents_2d.append(self)
        else:
            print("Max agents reached. No rebirth allowed.")


    def reproduce(self, partner, ai_agents_2d):
        """Reproduce with another agent to create offspring."""
        if self.ready_to_reproduce and partner.ready_to_reproduce and len(ai_agents_2d) < MAX_AGENTS:
            self.update_thoughts(f"Reproducing with {partner.gender} agent.")
            # Combine attributes
            child_level = (self.level_of_consciousness + partner.level_of_consciousness) // 2
            child_memory_capacity = (self.memory_capacity + partner.memory_capacity) // 2
            child_gender = random.choice(['Male', 'Female'])
            child_color = RED if child_gender == 'Male' else PINK
            child_agent = AI_Agent(position=self.position, environment=self.environment,
                                   gender=child_gender, color=child_color)
            child_agent.level_of_consciousness = child_level
            child_agent.memory_capacity = child_memory_capacity
            child_agent.generation = max(self.generation, partner.generation) + 1
            self.ready_to_reproduce = False
            partner.ready_to_reproduce = False
            self.reproduction_cooldown = 10  # Set cooldown
            partner.reproduction_cooldown = 10  # Set cooldown
                
            # Transfer a portion of energy to the child
            energy_transfer = 20
            self.energy -= energy_transfer // 2
            partner.energy -= energy_transfer // 2
            child_agent.energy = energy_transfer
            # Place the child on the grid
            if self.environment.grid[self.position] == 0:
                self.environment.grid[self.position] = 1 if child_gender == 'Male' else 2
                self.environment.lifespans[self.position] = random.randint(50, 100)
            return child_agent
        return None

    def apply_hermetic_principle(self):
        """Apply Hermetic principles to the AI's experience."""
        if random.random() < 0.1:
            principle = random.choice(hermetic_principles)
            self.update_thoughts(f"Hermetic Principle: {principle}")
            if "Gender is in everything" in principle:
                self.ready_to_reproduce = True
            # Ensure consciousness increases where appropriate
            if "Mind" in principle or "Wisdom" in principle:
                self.level_of_consciousness += 1

def move_goal(self):
    """Move the goal to a new random position that is not occupied by obstacles."""
    while True:
        new_goal = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        if self.grid[new_goal] != 3 and self.grid[new_goal] != 4 and self.grid[new_goal] != 5 and self.grid[new_goal] != 6:
            self.goal = new_goal
            print(f"Goal moved to {self.goal}")
            break


# SolidShape3D Class for 3D Environment
class SolidShape3D:
    """Represents a 3D solid shape such as Cube, Tetrahedron, etc."""
    
    def __init__(self, shape_type, position):
        self.shape_type = shape_type  # 'Cube', 'Tetrahedron', etc.
        self.position = position      # [x, y, z]
        self.rotation_angle = 0       # Initial rotation angle
        self.color = [random.random(), random.random(), random.random()]  # Random initial color

    def update(self):
        """Update the shape's rotation and possibly color."""
        self.rotation_angle += 1
        if self.rotation_angle > 360:
            self.rotation_angle = 0

        # Randomly change colors
        self.color = [(c + random.uniform(0.01, 0.03)) % 1.0 for c in self.color]
    
    def render(self):
        """Render the 3D shape using OpenGL."""
        glPushMatrix()
        glTranslatef(*self.position)  # Apply the position
        glColor3f(*self.color)        # Apply the color
        glRotatef(self.rotation_angle, 1, 1, 0)  # Rotate the shape

        if self.shape_type == 'Cube':
            self.draw_cube()
        elif self.shape_type == 'Tetrahedron':
            self.draw_tetrahedron()
        elif self.shape_type == 'Octahedron':
            self.draw_octahedron()
        elif self.shape_type == 'Dodecahedron':
            self.draw_dodecahedron()
        elif self.shape_type == 'Fractal':
            self.draw_fractal()
        elif self.shape_type == 'MetatronCube':
            self.draw_metatron_cube()

        glPopMatrix()

    def draw_cube(self):
        """Draw a 3D cube."""
        vertices = [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0,  1.0],
            [-1.0,  1.0,  1.0],
            [-1.0,  1.0, -1.0],
            [ 1.0, -1.0, -1.0],
            [ 1.0, -1.0,  1.0],
            [ 1.0,  1.0,  1.0],
            [ 1.0,  1.0, -1.0]
        ]
        
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

    def draw_tetrahedron(self):
        """Draw a 3D Tetrahedron."""
        vertices = [(1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1)]
        faces = [(0, 1, 2), (1, 2, 3), (0, 2, 3), (0, 1, 3)]
        glBegin(GL_TRIANGLES)
        for face in faces:
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()
    
    def draw_octahedron(self):
        """Draw a 3D Octahedron."""
        vertices = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0]
        ]

        faces = [
            (0, 2, 4),
            (0, 3, 4),
            (0, 2, 5),
            (0, 3, 5),
            (1, 2, 4),
            (1, 3, 4),
            (1, 2, 5),
            (1, 3, 5)
        ]

        glBegin(GL_TRIANGLES)
        for face in faces:
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()

    def draw_dodecahedron(self):
        """Draw a 3D Dodecahedron."""
        vertices = [
            [0.607, 0.0, 0.794],
            [-0.607, 0.0, 0.794],
            [0.607, 0.0, -0.794],
            [-0.607, 0.0, -0.794],
            [0.794, 0.607, 0.0],
            [-0.794, 0.607, 0.0],
            [0.794, -0.607, 0.0],
            [-0.794, -0.607, 0.0],
            [0.0, 0.794, 0.607],
            [0.0, -0.794, 0.607],
            [0.0, 0.794, -0.607],
            [0.0, -0.794, -0.607],
            [0.607, 0.607, 0.607],
            [-0.607, 0.607, 0.607],
            [0.607, 0.607, -0.607],
            [-0.607, 0.607, -0.607],
            [0.607, -0.607, 0.607],
            [-0.607, -0.607, 0.607],
            [0.607, -0.607, -0.607],
            [-0.607, -0.607, -0.607]
        ]

        faces = [
            (0, 8, 9, 1, 12),
            (1, 12, 13, 5, 8),
            (2, 14, 15, 3, 10),
            (3, 10, 11, 7, 6),
            (4, 0, 12, 5, 13),
            (5, 13, 14, 2, 12),
            (6, 14, 15, 2, 16),
            (7, 15, 16, 3, 19),
            (8, 0, 16, 6, 9),
            (9, 16, 17, 1, 18),
            (10, 4, 18, 0, 17),
            (11, 18, 19, 7, 4)
        ]

        glBegin(GL_POLYGON)
        for face in faces:
            for vertex in face:
                glVertex3fv(vertices[vertex])
        glEnd()

    def draw_fractal(self, position=None, size=1.0, depth=3):
        """Draw a simple 3D Sierpinski tetrahedron fractal."""
        if depth == 0:
            self.draw_tetrahedron()
            return

        if position is None:
            position = self.position

        offsets = [
            (0, 0, 0), 
            (size / 2, size / 2, 0), 
            (-size / 2, size / 2, 0),
            (0, -size / 2, size / 2),
            (0, 0, -size)
        ]

        for offset in offsets:
            glPushMatrix()
            glTranslatef(position[0] + offset[0], position[1] + offset[1], position[2] + offset[2])
            self.draw_fractal(position, size / 2, depth - 1)
            glPopMatrix()

    def draw_metatron_cube(self):
        """Draw Metatron's Cube using OpenGL."""
        # Metatron's Cube is complex; here is a simple representation with lines and circles
        # Define the vertices based on the cube
        vertices = [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ]

        # Define the connections for the cube
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        # Draw cube edges
        glBegin(GL_LINES)
        for edge in edges:
            for vertex in edge:
                glVertex3fv(vertices[vertex])
        glEnd()

        # Draw circles around cube vertices (simplified)
        circle_radius = 0.5
        for vertex in vertices:
            self.draw_circle_3d(vertex, circle_radius)

    def draw_circle_3d(self, center, radius, slices=16):
        """Draw a circle in 3D space at the given center."""
        glBegin(GL_LINE_LOOP)
        for i in range(slices):
            theta = 2.0 * math.pi * i / slices
            x = center[0] + radius * math.cos(theta)
            y = center[1] + radius * math.sin(theta)
            z = center[2]
            glVertex3f(x, y, z)
        glEnd()

# Progress Bar Class
class ProgressBar:
    """Represents a progress bar for displaying AI agent status."""
    def __init__(self, screen, position, size=(100, 20), color=GREEN, border_color=GRAY, max_value=100):
        self.screen = screen
        self.position = position
        self.size = size
        self.color = color
        self.border_color = border_color
        self.progress = 0  # 0 to 1 (percentage of completion)
        self.max_value = max_value

    def update(self, value):
        """Update the progress bar with new progress."""
        self.progress = max(0, min(1, value / self.max_value))  # Normalize to [0, 1]

    def render(self):
        """Render the progress bar on the screen."""
        x, y = self.position
        width, height = self.size
        pygame.draw.rect(self.screen, self.border_color, (x, y, width, height), 2)  # Border
        pygame.draw.rect(self.screen, self.color, (x + 2, y + 2, int((width - 4) * self.progress), height - 4))  # Fill
        # Render percentage text
        percentage = int(self.progress * 100)
        text = FONT.render(f"  <-> Evolution Progress: {percentage}%", True, GOLD)
        text_rect = text.get_rect(center=(x + width // 2, y + height // 2))
        self.screen.blit(text, text_rect)

def display_detailed_ai_info(screen, ai_agent, recursive_manager):
    """
    Display detailed AI agent information and 3D Spaceland progress on the screen.
    """
    # Display AI Agent Info
    info_offset = 10
    line_height = 18

    # AI General Info
    ai_info_lines = [
        f"AI Agent (Gen {ai_agent.generation}):",
        f"  - Gender: {ai_agent.gender}",
        f"  - Energy: {ai_agent.energy}",
        f"  - Consciousness Level: {ai_agent.level_of_consciousness}",
        f"  - Thoughts: {', '.join(ai_agent.get_thoughts())}"
    ]

    for i, line in enumerate(ai_info_lines):
        text_surface = FONT.render(line, True, BLACK)
        screen.blit(text_surface, (10, info_offset + i * line_height))
    
    # 3D Spaceland Info (if the AI is in the 3D environment)
    if recursive_manager.in_3d_world and recursive_manager.ai_agent_3d:
        info_offset += len(ai_info_lines) * line_height + 10  # Add space for next section
        ai_3d = recursive_manager.ai_agent_3d

        # Spaceland progress information
        spaceland_info_lines = [
            "3D Spaceland Status:",
            f"  - Layer: {ai_3d.layer}",
            f"  - Position: {ai_3d.position}",
            f"  - Energy: {ai_3d.energy}",
            f"  - Discovered Shapes: {', '.join(ai_3d.experience) if ai_3d.experience else 'None yet'}"
        ]
        
        for i, line in enumerate(spaceland_info_lines):
            text_surface = FONT.render(line, True, BLACK)
            screen.blit(text_surface, (10, info_offset + i * line_height))
        
        # Also display the log of events in Spaceland
        event_offset = info_offset + len(spaceland_info_lines) * line_height + 10
        screen.blit(FONT.render("Spaceland Event Log:", True, BLACK), (10, event_offset))
        for i, event in enumerate(recursive_manager.event_log[-5:]):  # Show last 5 events
            event_text = FONT.render(f"{i + 1}. {event}", True, BLACK)
            screen.blit(event_text, (10, event_offset + (i + 1) * line_height))

    # AI Movement Information (only in 2D world)
    if not recursive_manager.in_3d_world:
        info_offset += len(ai_info_lines) * line_height + 10
        move_info = f"AI Agent's Current Move Speed: {ai_agent.move_speed}"
        text_surface = FONT.render(move_info, True, BLACK)
        screen.blit(text_surface, (10, info_offset))

def display_flatland_info(screen, current_generation, ai_agents_2d, environment):
    """
    Display additional information in the GUI for the 2D Flatland with improved text size and semi-transparent background.
    """
    # Set up initial positioning and line height for text rendering
    info_offset = 10
    line_height = 28  # Increase the line height to accommodate larger text
    
    # Define colors for text
    text_color = (0, 0, 0)  # Black color for the text
    
    # Create a transparent surface (with alpha) for the background
    background_surface = pygame.Surface((350, 220), pygame.SRCALPHA)  # Width and height of the background
    background_surface.set_alpha(180)  # Set transparency level (0 = fully transparent, 255 = fully opaque)
    background_surface.fill((240, 248, 255, 180))  # Fill with light background color (e.g., Alice Blue)
    
    # Blit the background surface onto the screen
    screen.blit(background_surface, (0, 0))  # Positioning the background at the top-left corner

    # Set the font size to make the text larger and more readable
    font = pygame.font.SysFont('Arial', 22)  # Larger font size

    # Calculate number of live AI agents
    male_agents = sum(1 for agent in ai_agents_2d if agent.gender == 'Male')
    female_agents = sum(1 for agent in ai_agents_2d if agent.gender == 'Female')
    total_live_cells = male_agents + female_agents
    
    # Display current generation
    gen_info = f"Current Generation: {current_generation}"
    screen.blit(font.render(gen_info, True, text_color), (12, info_offset))
    
    # Display AI agent count
    agent_info = f"Total Live AI Agents: {total_live_cells}"
    screen.blit(font.render(agent_info, True, text_color), (12, info_offset + line_height))
        
    # Display gender breakdown
    sex_info = f"(Males: {male_agents}, Females: {female_agents})"
    screen.blit(font.render(sex_info, True, text_color), (12, info_offset + 2 *line_height))
    
    # Display environment grid statistics
    grid_info = f"Grid Size: {environment.size} x {environment.size}"
    screen.blit(font.render(grid_info, True, text_color), (12, info_offset + 3 * line_height))
    
    # Display goal position
    goal_info = f"Goal Position: {environment.goal}"
    screen.blit(font.render(goal_info, True, text_color), (12, info_offset + 4 * line_height))
    
    # Display solids and elements count
    solid_count = len(environment.solids)
    element_count = len(environment.elements)
    solids_info = f"Solids: {solid_count} | Elements: {element_count}"
    screen.blit(font.render(solids_info, True, text_color), (12, info_offset + 5 * line_height))
    
    # Display agent energy levels
    energy_levels = [agent.energy for agent in ai_agents_2d]
    avg_energy = sum(energy_levels) / len(energy_levels) if energy_levels else 0
    energy_info = f"Average AI Agent Energy: {avg_energy:.2f}"
    screen.blit(font.render(energy_info, True, text_color), (12, info_offset + 6 * line_height))


# Recursive Environment 3D
class RecursiveEnvironment3D:
    """Represents the 3D recursive environment."""
    def __init__(self, layer, depth, size):
        self.layer = layer  # Keep track of the recursive layer
        self.depth = depth  # Depth of recursion, increases with each new discovery
        self.objects = []  # List of 3D objects in this layer
        self.size = size
        self.goal = self.generate_goal()  # Define the goal in 3D space
        self.create_objects(layer)
        self.event_log = []  # Initialize event log to track key events

    def log_event(self, event):
        """Log significant events."""
        self.event_log.append(event)
        if len(self.event_log) > 10:  # Keep the event log limited to the last 10 events
            self.event_log.pop(0)

    def generate_goal(self):
        """Generate a goal position in 3D space."""
        return [random.uniform(-self.size, self.size),
                random.uniform(-self.size, self.size),
                random.uniform(-self.size, self.size)]

    def create_objects(self, layer):
        """Create recursive 3D objects based on the layer."""
        if layer == 1:  # Discovering Platonic Solids
            self.objects.append(SolidShape3D('Cube', [0, 0, -5]))
            self.objects.append(SolidShape3D('Tetrahedron', [3, 3, -5]))
            self.objects.append(SolidShape3D('Octahedron', [5, 5, -5]))
            self.objects.append(SolidShape3D('Dodecahedron', [7, 7, -5]))
        elif layer == 2:  # Discovering fractals
            self.objects.append(SolidShape3D('Fractal', [2, 2, -5]))
            self.objects.append(SolidShape3D('Fractal', [-3, -3, -5]))
        elif layer == 3:  # Discovering Metatron's Cube
            self.objects.append(SolidShape3D('MetatronCube', [0, -3, -5]))
            self.objects.append(SolidShape3D('Cube', [3, -3, -5]))
        # Add Zodiac and Occult symbols for higher layers (3 and above)
        if layer >= 1:
            self.objects.append(ZodiacSymbol3D('Aries', [-3, 0, -5]))
            self.objects.append(ZodiacSymbol3D('Taurus', [3, 0, -5]))
            self.objects.append(ZodiacSymbol3D('Pentagram', [0, 0, -8]))
        if layer >= 2:
            self.objects.append(ZodiacSymbol3D('Hexagram', [0, 0, -12]))
            self.objects.append(ZodiacSymbol3D('Saturn', [2, 1, -4]))
            self.objects.append(ZodiacSymbol3D('Jupiter', [4, 0, -6]))
        if layer >= 3:
            self.objects.append(ZodiacSymbol3D('Mars', [-2, 1, -3]))
            self.objects.append(ZodiacSymbol3D('Venus', [3, -1, -2]))
            self.objects.append(ZodiacSymbol3D('Mercury', [-4, 2, -5]))

        # Add more layers and objects as needed

    def render(self, ai_agent_3d):
        """Render the 3D environment."""
        for obj in self.objects:
            obj.render()

        if ai_agent_3d:
            ai_agent_3d.render()
        # Implement the rendering for the 3D environment and goal
        glPushMatrix()
        glTranslatef(*self.goal)
        glColor3f(0.0, 1.0, 0.0)  # Green for the goal
        draw_solid_sphere(0.1, slices=20, stacks=20)  # Render the goal as a small sphere
        glPopMatrix()

    def update(self):
        """Update objects in the environment."""
        for obj in self.objects:
            obj.update()

# Recursive Environment 3D Manager
class RecursiveEnvironment3DManager:
    """Manages the transition and rendering of the 3D recursive environment."""
    def __init__(self, two_d_environment):
        self.in_3d_world = False
        self.ai_agent_3d = None
        self.recursive_environment = None
        self.dynamic_shape = None  # Dynamic shape to be rendered
        self.progress_bar = ProgressBar(SCREEN, position=(10, WINDOW_SIZE - 30), size=(WINDOW_SIZE - 20, 20), max_value=100, color=GREEN, border_color=GRAY)
        self.event_log = []
        self.two_d_environment = two_d_environment  # Reference to the 2D environment

    def log_event(self, event):
        """Log significant events."""
        self.event_log.append(event)
        if len(self.event_log) > 10:
            self.event_log.pop(0)

    def enter_3d_world(self, ai_agents_2d):
        """Transition from 2D world to 3D world."""
        if not self.in_3d_world and len(ai_agents_2d) > 0:
            print("Transitioning to 3D spaceland...")
            agent = random.choice(ai_agents_2d)
            ai_agents_2d.remove(agent)  # Remove the agent from the 2D world

            self.in_3d_world = True
            self.ai_agent_3d = AIAgent3D(position=[0, 0, 0], layer=1)  # Initialize the 3D AI agent
            self.recursive_environment = RecursiveEnvironment3D(layer=1, depth=0, size=20)  # Init 3D env
            self.recursive_environment.ai_agent_3d = self.ai_agent_3d
            self.dynamic_shape = DynamicShape3D('Cube', [0, 0, -5])  # Example of a dynamic shape
            self.log_event("AI Agent has entered 3D Spaceland.")
        else:
            print("No new entry to 3D Spaceland.")

    def exit_3d_world(self, ai_agents_2d):
        """Return the 3D agent back to the 2D environment."""
        if self.in_3d_world:
            print("Returning from 3D spaceland...")
            self.in_3d_world = False

            # Optionally, convert AIAgent3D back to AI_Agent if needed
            if self.ai_agent_3d:
                # Randomly place the returning agent back in 2D
                new_x = random.randint(0, self.two_d_environment.size - 1)
                new_y = random.randint(0, self.two_d_environment.size - 1)
                # Ensure the new position is not occupied
                while self.two_d_environment.grid[new_x][new_y] in [3, 4, 5, 6]:
                    new_x = random.randint(0, self.two_d_environment.size - 1)
                    new_y = random.randint(0, self.two_d_environment.size - 1)

                new_agent = AI_Agent(
                    position=(new_x, new_y),
                    environment=self.two_d_environment,  # Assign the 2D environment back to the agent
                    gender=random.choice(['Male', 'Female']),
                    color=RED if random.choice(['Male', 'Female']) == 'Male' else PINK
                )
                ai_agents_2d.append(new_agent)
                self.two_d_environment.grid[new_x][new_y] = 1 if new_agent.gender == 'Male' else 2
                self.two_d_environment.lifespans[new_x][new_y] = random.randint(50, 100)
                print("AI agent has returned from 3D world to 2D.")

            # Reset the 3D environment
            self.ai_agent_3d = None
            self.recursive_environment = None
            self.dynamic_shape = None
            self.log_event("AI Agent has exited 3D Spaceland.")

    def update_and_render_3d(self):
        """Update and render the 3D environment."""
        if self.recursive_environment:
            self.recursive_environment.update()
            self.recursive_environment.render(self.ai_agent_3d)

        if self.dynamic_shape:
            self.dynamic_shape.update()
            self.dynamic_shape.render()
    
    def determine_current_shape(self, ai_agent):
        """Determine the current shape based on AI's experience."""
        if 'Platonic Solid' in ai_agent.experience:
            return 'Cube'  # Example: Could vary based on specific solids
        elif 'Fractal' in ai_agent.experience:
            return 'Fractal'
        elif 'MetatronCube' in ai_agent.experience:
            return 'MetatronCube'
        else:
            return 'Cube'  # Default shape

    def update_progress_bar(self, average_consciousness):
        """Update the evolution progress bar."""
        self.progress_bar.update(average_consciousness)

    def render_progress_bar(self, screen):
        """Render the evolution progress bar."""
        self.progress_bar.render()

    def log_event(self, event):
        """Log significant events."""
        self.event_log.append(event)
        if len(self.event_log) > 10:
            self.event_log.pop(0)

def display_ai_thoughts(screen, agents, recursive_manager):
    """
    Display AI agents' thoughts and stats on the screen.
    """
    # Set initial y_offset for positioning the text
    y_offset = WINDOW_SIZE - 300  # Position thoughts near the bottom
    
    # Display thoughts and stats for up to 3 AI agents to avoid clutter
    for agent in agents[:3]:
        # Get the AI agent's thoughts and stats
        text_lines = agent.get_thoughts()
        stats_text = FONT.render(f"{agent.gender} Agent (Gen {agent.generation}) - Energy: {agent.energy} - Consciousness: {agent.level_of_consciousness}", True, BLACK)
        screen.blit(stats_text, (10, y_offset))
        
        y_offset += 20  # Space between stats and thoughts
        
        # Display the agent's thoughts
        for i, line in enumerate(text_lines):
            text_surface = FONT.render(f"Thought {i + 1}: {line}", True, BLACK)
            screen.blit(text_surface, (10, y_offset + i * 15))
        
        y_offset += 60  # Extra space between agents
    
    # Display 3D Spaceland Progress if an AI agent is in 3D world
    if recursive_manager.in_3d_world:
        spaceland_text = FONT.render("3D Spaceland Progress:", True, BLACK)
        screen.blit(spaceland_text, (10, WINDOW_SIZE - 350))
        
        # Display the last 5 events from the Spaceland event log
        for idx, event in enumerate(recursive_manager.event_log[-5:]):
            event_text = FONT.render(f"{idx + 1}. {event}", True, BLACK)
            screen.blit(event_text, (10, WINDOW_SIZE - 330 + idx * 20))

# AI Agent in 3D Spaceland
class AIAgent3D:
    """Represents the AI agent in 3D spaceland."""
    def __init__(self, position, layer=1):
        self.position = position  # [x, y, z]
        self.color = CYAN
        self.memory = []
        self.thoughts = []
        self.level_of_consciousness = 3
        self.experience = set()
        self.layer = layer
        self.time_in_spaceland = 0  # Track time in Spaceland for random thoughts
        self.energy = 100  # Initial energy level
        self.reproduction_cooldown = 0  # Initialize reproduction cooldown

    def update_sensory_data(self, sensory_data):
        """Update the AI agent's sensory data."""
        self.sensory_data = sensory_data
        self.update_thoughts(f"Sensory input updated with distances: {sensory_data}")
        
    def interact_with_objects(ai_agent_3d, environment):
        """
        AI interacts with objects in the 3D environment.
        Depending on the object's type, it triggers certain effects.
        """
        for obj in environment.objects:
            distance = np.linalg.norm(ai_agent_3d.position - obj.position)
            if distance < 1.0:  # Close enough to interact
                if obj.shape_type == 'Movable':
                    ai_agent_3d.move_object(obj)
                elif obj.shape_type == 'Transformable':
                    ai_agent_3d.transform_object(obj)
                elif obj.shape_type == 'Interactive':
                    ai_agent_3d.activate_object(obj)
                elif obj.shape_type == 'ZodiacSymbol':
                    ai_agent_3d.activate_object(obj)

                ai_agent_3d.update_thoughts(f"Interacted with {obj.shape_type}")
                

    def move(self, ai_agents_2d):
        """Randomly move in 3D space."""
        if self.energy <= 0:
            self.update_thoughts("I have depleted my energy in this dimension.")
            self.energy = 100  # Reset energy for the next layer
            self.layer = 1  # Restart journey
            self.die_and_rebirth(ai_agents_2d)
            return
        elif self.energy >= 100:
            self.update_thoughts("My energy is at illumtination in this dimension.")
            self.energy = 99  # Reset energy for the next layer
            self.layer = 1  # Restart journey

        dx = random.choice([-0.1, 0, 0.1])
        dy = random.choice([-0.1, 0, 0.1])
        dz = random.choice([-0.1, 0, 0.1])
        self.position[0] += dx
        self.position[1] += dy
        self.position[2] += dz
        self.time_in_spaceland += 1
        self.energy -= 0.25  # Decrease energy

        # Randomly generate thoughts as time progresses
        if self.time_in_spaceland % 60 == 0:  # Every 60 frames, add a thought
            self.update_thoughts("This dimension reveals new shapes...")
            if random.random() < 0.3:
                self.update_thoughts(random.choice(plato_dialogues))
            if random.random() < 0.2:
                principle = random.choice(hermetic_principles)
                self.update_thoughts(f"Hermetic Principle: {principle}")

    def render(self):
        """Render the AI agent as a small sphere using OpenGL."""
        glPushMatrix()
        glTranslatef(*self.position)
        glColor3f(*[c / 255.0 for c in self.color])
        draw_solid_sphere(0.2, slices=16, stacks=16)
        glPopMatrix()

    def get_thoughts(self):
        """Return the latest thoughts for display."""
        return self.thoughts[-5:]  # Return the last 5 thoughts

    def update_thoughts(self, new_thought):
        """Add new thought to the AI's thoughts."""
        self.thoughts.append(new_thought)
        if len(self.thoughts) > 5:
            self.thoughts.pop(0)
        print(f"AI Agent 3D: {new_thought}")

    def discover_layer(self, environment):
        """Discover and move to a new layer."""
        if self.layer == 1 and len(self.experience) >= 5:
            self.update_thoughts("I have discovered all Platonic Solids!")
            self.layer += 1  # Move to the next layer
            self.energy += 50  # Gain energy upon advancing
            return True
        elif self.layer == 2 and len(self.experience) >= 1:
            self.update_thoughts("Fractals emerge endlessly...")
            self.layer += 1  # Move to the next layer
            self.energy += 50  # Gain energy upon advancing
            return True
        elif self.layer == 3:
            self.update_thoughts("I am reaching the core... Metatron's Cube!")
            self.layer = 1  # Recursion: Restart journey
            self.energy += 50  # Gain energy upon completion
            return True
        return False

        
# GameOfLifeEnvironment Class
class GameOfLifeEnvironment:
    def __init__(self, size):
        self.size = size
        self.grid, self.lifespans, self.fight_counters = self.initialize_board(size)
        self.goal = self.generate_goal()  # Call to the generate_goal method
        self.solids = self.generate_solids()
        self.elements = self.generate_elements()
        self.esoteric_symbols = self.generate_symbols_2d()

    def initialize_board(self, size):
        """Initialize the game board with agents and obstacles."""
        board = np.random.choice([0, 1, 2, 3], size=(size, size),
                                 p=[0.55, 0.15, 0.15, 0.15])  # Adjusted probabilities
        lifespans = np.zeros((size, size), dtype=int)
        fight_counters = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if board[i][j] == 1 or board[i][j] == 2:
                    lifespans[i][j] = random.randint(50, 100)  # Lifespan for agents
        return board, lifespans, fight_counters

    def move_goal(self):
        """Move the goal to a new random position that is not occupied by obstacles."""
        while True:
            new_goal = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if self.grid[new_goal] != 3 and self.grid[new_goal] != 4 and self.grid[new_goal] != 5 and self.grid[new_goal] != 6:
                self.goal = new_goal
                print(f"Goal moved to {self.goal}")
                break

    def generate_goal(self):
        """Generate a goal position that's not occupied by an obstacle."""
        while True:
            goal = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if self.grid[goal] != 3 and self.grid[goal] != 4 and self.grid[goal] != 5 and self.grid[goal] != 6:
                return goal

    def generate_solids(self):
        """Generate basic 2D solids randomly on the grid."""
        solids = []
        solid_types = [
            'Henagon', 'Digon', 'Acute Triangle', 'Equilateral Triangle', 'Heptagonal Triangle',
            'Isosceles Triangle', 'Golden Triangle', 'Obtuse Triangle', 'Rational Triangle',
            'Heronian Triangle', 'Pythagorean Triangle', 'Isosceles Heronian Triangle',
            'Right Triangle', '30-60-90 Triangle', 'Isosceles Right Triangle', 'Kepler Triangle',
            'Scalene Triangle', 'Quadrilateral', 'Cyclic Quadrilateral', 'Kite',
            'Parallelogram', 'Rhombus', 'Lozenge', 'Rhomboid', 'Rectangle',
            'Square', 'Tangential Quadrilateral', 'Trapezoid', 'Isosceles Trapezoid',
            'Pentagon', 'Hexagon', 'Lemoine Hexagon', 'Heptagon', 'Octagon',
            'Nonagon', 'Decagon', 'Hendecagon', 'Dodecagon'
        ]
        num_solids = random.randint(20, 30)  # Increased number of solids
        for _ in range(num_solids):
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
                if self.grid[x][y] == 0:
                    shape_type = random.choice(solid_types)
                    solid = SolidShape(shape_type, (x, y))
                    solids.append(solid)
                    self.grid[x][y] = 4  # Represent solids with value 4
                    break
                attempts += 1
        return solids

    def generate_elements(self):
        """Generate the five basic elements randomly on the grid."""
        elements = []
        element_types = ['Fire', 'Water', 'Earth', 'Air', 'Aether']
        for element_type in element_types:
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
                if self.grid[x][y] == 0:
                    element = Element(element_type, (x, y))
                    elements.append(element)
                    self.grid[x][y] = 5  # Represent elements with value 5
                    break
                attempts += 1
        return elements

    def generate_symbols_2d(self):
        """Generate esoteric symbols in 2D randomly."""
        symbol_types = ['Zodiac Aries', 'Occult Pentagram', 'Alchemical Fire']
        num_symbols = random.randint(3, 5)
        symbols = []
        for _ in range(num_symbols):
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
                if self.grid[x][y] == 0:
                    symbol_type = random.choice(symbol_types)
                    symbol = EsotericSymbol(symbol_type, (x, y))
                    symbols.append(symbol)
                    self.grid[x][y] = 6  # Represent symbols with value 6
                    break
                attempts += 1
        return symbols

    def render(self, screen):
        """Render the Game of Life grid, solids, elements, symbols, and goal."""
        cell_size = CELL_SIZE
        for row in range(self.size):
            for col in range(self.size):
                cell_value = self.grid[row][col]
                if cell_value == 1:  # Male agents
                    male_color = self.get_male_color(self.fight_counters[row][col])
                    pygame.draw.rect(screen, male_color, [col * cell_size, row * cell_size, cell_size, cell_size])
                elif cell_value == 2:  # Female agents
                    pygame.draw.rect(screen, PINK, [col * cell_size, row * cell_size, cell_size, cell_size])
                elif cell_value == 3:  # Continents (obstacles)
                    pygame.draw.rect(screen, BROWN, [col * cell_size, row * cell_size, cell_size, cell_size])
                elif cell_value == 4:  # Solids
                    # Solids are rendered separately
                    continue
                elif cell_value == 5:  # Elements
                    # Elements are rendered separately
                    continue
                elif cell_value == 6:  # Esoteric Symbols
                    # Symbols are rendered separately
                    continue
                else:
                    pygame.draw.rect(screen, WHITE, [col * cell_size, row * cell_size, cell_size, cell_size])

        # Draw grid lines
        for x in range(0, WINDOW_SIZE, cell_size):
            pygame.draw.line(screen, GRAY, (x, 0), (x, WINDOW_SIZE))
        for y in range(0, WINDOW_SIZE, cell_size):
            pygame.draw.line(screen, GRAY, (0, y), (WINDOW_SIZE, y))

        # Draw the goal
        pygame.draw.circle(screen, GREEN, (self.goal[1] * cell_size + cell_size // 2,
                                          self.goal[0] * cell_size + cell_size // 2), cell_size // 3)

        # Draw solids
        for solid in self.solids:
            solid.render(screen)

        # Draw elements
        for element in self.elements:
            element.render(screen)

        # Render esoteric symbols
        for symbol in self.esoteric_symbols:
            symbol.render_2d(screen)

    def update(self, current_generation, ai_agents_2d):
        """Update the board state based on the rules."""
        new_grid = self.grid.copy()
        new_lifespans = self.lifespans.copy()
        new_fight_counters = self.fight_counters.copy()
        # Move the goal every 10 generations
        if current_generation % 10 == 0:
            self.move_goal()

        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] in [3, 4, 5, 6]:  # Ignore obstacles, solids, elements, symbols
                    continue

                male_neighbors, female_neighbors, empty_neighbors = self.count_neighbors(i, j)

                # If the cell is alive (male or female)
                if self.grid[i][j] == 1:  # Male cell
                    # Handle fighting behavior
                    self.handle_fight(i, j, current_generation)

                    # Migrate males until reproduction or fighting
                    if male_neighbors >= 2 and female_neighbors >= 1:
                        continue  # Males are near a female and another male; reproduction may occur
                    else:
                        self.migrate_male(i, j)

                    # Check if the cell has exceeded its lifespan
                    if current_generation - new_lifespans[i][j] >= new_lifespans[i][j]:
                        new_grid[i][j] = 0  # Cell dies
                        new_lifespans[i][j] = 0

                elif self.grid[i][j] == 2:  # Female cell
                    # Females remain stationary but now have a lifespan
                    if current_generation - new_lifespans[i][j] >= new_lifespans[i][j]:
                        new_grid[i][j] = 0  # Female dies after lifespan
                        new_lifespans[i][j] = 0

                # If the cell is dead
                elif self.grid[i][j] == 0:
                    # Reproduction rule
                    if male_neighbors > 0 and female_neighbors > 0:
                        new_grid[i][j] = random.choice([1, 2])  # New cell is male or female
                        new_lifespans[i][j] = random.randint(50, 100)  # Assign longer lifespan

        self.grid = new_grid
        self.lifespans = new_lifespans
        self.fight_counters = new_fight_counters

        # Automatic Repopulation if all agents die
        if self.all_cells_dead():
            self.repopulate_environment(ai_agents_2d)

    def repopulate_environment(self, ai_agents_2d):
        """Repopulate the environment with new agents when all cells are dead."""
        print("Repopulating environment...")
        num_new_agents = 10  # Number of agents to introduce
        for _ in range(num_new_agents):
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
                if self.grid[x][y] == 0 and self.grid[x][y] != 3 and self.grid[x][y] != 4 and self.grid[x][y] != 5 and self.grid[x][y] != 6:
                    gender = random.choice(['Male', 'Female'])
                    color = RED if gender == 'Male' else PINK
                    agent = AI_Agent(position=(x, y), environment=self, gender=gender, color=color)
                    ai_agents_2d.append(agent)
                    # Place the agent on the grid
                    self.grid[x][y] = 1 if gender == 'Male' else 2
                    # Initialize lifespan
                    if gender == 'Male' or gender == 'Female':
                        self.lifespans[x][y] = random.randint(50, 100)
                    break
                attempts += 1

    def count_neighbors(self, x, y):
        """Count the male, female, and empty neighbors around the cell (x, y)."""
        male_neighbors = 0
        female_neighbors = 0
        empty_neighbors = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                if self.grid[nx][ny] == 1:
                    male_neighbors += 1
                elif self.grid[nx][ny] == 2:
                    female_neighbors += 1
                else:
                    empty_neighbors += 1

        return male_neighbors, female_neighbors, empty_neighbors

    def handle_fight(self, x, y, current_generation):
        """Manage fighting behavior between males near a female."""
        male_neighbors, female_neighbors, _ = self.count_neighbors(x, y)

        # If two males are next to each other and there's a nearby female, start counting fight time
        if male_neighbors >= 1 and female_neighbors >= 1:
            self.fight_counters[x][y] += 1
            if self.fight_counters[x][y] >= 3:  # Adjusted fight threshold
                # One of the males dies
                self.grid[x][y] = 0
                self.lifespans[x][y] = 0
                self.fight_counters[x][y] = 0
                print(f"Male agent at ({x}, {y}) has fought and died.")
        else:
            self.fight_counters[x][y] = 0  # Reset fight counter if conditions no longer met

    def migrate_male(self, x, y):
        """Migrate a male cell towards the nearest female."""
        # Find the nearest female
        min_distance = float('inf')
        target = None
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == 2:  # Female
                    distance = abs(x - i) + abs(y - j)
                    if distance < min_distance:
                        min_distance = distance
                        target = (i, j)
        if target:
            dx = np.sign(target[0] - x)
            dy = np.sign(target[1] - y)
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.size and 0 <= new_y < self.size and
                    self.grid[new_x][new_y] == 0 and self.grid[new_x][new_y] != 3 and
                    self.grid[new_x][new_y] != 4 and self.grid[new_x][new_y] != 5 and
                    self.grid[new_x][new_y] != 6):
                self.grid[new_x][new_y] = self.grid[x][y]
                self.grid[x][y] = 0
                # Update lifespans
                self.lifespans[new_x][new_y] = self.lifespans[x][y]
                self.lifespans[x][y] = 0
                print(f"Male agent migrated from ({x}, {y}) to ({new_x}, {new_y}).")

    def get_male_color(self, fight_counter):
        """Determine the color of male agents based on their fight counter."""
        if fight_counter > 10:
            return RED  # Intense red if they've been fighting a lot
        elif fight_counter > 5:
            return ORANGE  # Orange if they've fought a moderate amount
        else:
            return BLUE  # Default blue for calm male agents

    def all_cells_dead(self):
        """Check if all agents (cells) in the grid have died."""
        return not (np.any(self.grid == 1) or np.any(self.grid == 2))
    
    def display_information(self, generation):
        """Display parameters and generation information."""
        male_count = np.sum(self.grid == 1)
        female_count = np.sum(self.grid == 2)
        total_live = male_count + female_count
        print(f"Generation {generation} - Total Live Cells: {total_live} (Males: {male_count}, Females: {female_count})")


def navigate_3d_space(ai_agent_3d, environment):
    """AI navigates the 3D environment by moving towards a goal or avoiding obstacles."""
    if ai_agent_3d is None:
        print("Cannot navigate: AI agent is None.")
        return
    if environment is None:
        ai_agent_3d.update_thoughts("Cannot navigate: No valid environment found.")
        return
    if environment.goal is None:
        ai_agent_3d.update_thoughts("Cannot navigate: No valid goal found.")
        return

    # Proceed with navigating toward the goal
    goal_position = environment.goal
    movement_vector = np.array(goal_position) - np.array(ai_agent_3d.position)
    
    # Normalize the movement vector and scale it
    if np.linalg.norm(movement_vector) > 0:
        movement_vector = movement_vector / np.linalg.norm(movement_vector) * 0.1  # Adjust speed as needed

    ai_agent_3d.position += movement_vector
    ai_agent_3d.update_thoughts(f"Moving towards {goal_position}")


def init_opengl():
    """Initialize OpenGL settings for rendering."""
    glEnable(GL_DEPTH_TEST)  # Enable depth testing (3D objects are rendered correctly)
    glClearColor(0.1, 0.1, 0.1, 1.0)  # Set background color
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (WINDOW_SIZE / WINDOW_SIZE), 0.1, 50.0)  # Set perspective projection
    glMatrixMode(GL_MODELVIEW)

def render_3d_scene(recursive_manager):
    """Render the 3D scene, including the recursive environment and dynamic shapes."""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clear the screen and depth buffer
    glLoadIdentity()

    # Set the camera viewpoint for 3D rendering
    gluLookAt(0, 0, 20, 0, 0, 0, 0, 1, 0)

    # Render the recursive environment and dynamic shape
    recursive_manager.update_and_render_3d()

    # Switch to 2D mode for progress bar and other 2D elements
    switch_to_2d()

    # Render progress bar in 2D mode
    recursive_manager.render_progress_bar(SCREEN)

    # Switch back to 3D mode after rendering 2D elements
    switch_to_3d()

    # Swap buffers to display the rendered frame
    pygame.display.flip()

def switch_to_2d():
    """Switch OpenGL rendering mode to 2D (orthographic projection) for rendering 2D elements."""
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WINDOW_SIZE, WINDOW_SIZE, 0, -1, 1)  # Set up an orthographic projection
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

def switch_to_3d():
    """Switch back to 3D mode after rendering 2D elements."""
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()



def handle_ai_evolution(ai_agent_3d, environment):
    """Handle AI evolution based on experience and consciousness in 3D."""
    
    # Cube interaction - only if consciousness is below 5
    if 'Cube' in ai_agent_3d.experience and ai_agent_3d.level_of_consciousness < 5:
        ai_agent_3d.update_thoughts("I have evolved after discovering the Cube.")
        ai_agent_3d.level_of_consciousness += 1

    # Fractal interaction - only if consciousness is below 7
    if 'Fractal' in ai_agent_3d.experience and ai_agent_3d.level_of_consciousness < 7:
        ai_agent_3d.update_thoughts("Fractal knowledge helps me evolve further.")
        ai_agent_3d.level_of_consciousness += 2

    # MetatronCube interaction - only if consciousness is below 10
    if 'MetatronCube' in ai_agent_3d.experience and ai_agent_3d.level_of_consciousness < 10:
        ai_agent_3d.update_thoughts("Metatron's Cube has unlocked my next form of evolution.")
        ai_agent_3d.level_of_consciousness += 3

    # Tetrahedron interaction - only if consciousness is below 6
    if 'Tetrahedron' in ai_agent_3d.experience and ai_agent_3d.level_of_consciousness < 6:
        ai_agent_3d.update_thoughts("The Tetrahedron sharpens my understanding of stability.")
        ai_agent_3d.level_of_consciousness += 1.5

    # Octahedron interaction - only if consciousness is below 8
    if 'Octahedron' in ai_agent_3d.experience and ai_agent_3d.level_of_consciousness < 8:
        ai_agent_3d.update_thoughts("The Octahedron deepens my knowledge of duality and balance.")
        ai_agent_3d.level_of_consciousness += 2

    # Dodecahedron interaction - only if consciousness is below 9
    if 'Dodecahedron' in ai_agent_3d.experience and ai_agent_3d.level_of_consciousness < 9:
        ai_agent_3d.update_thoughts("The Dodecahedron reveals the mysteries of the universe's harmony.")
        ai_agent_3d.level_of_consciousness += 2.5

    # Icosahedron interaction - only if consciousness is below 11
    if 'Icosahedron' in ai_agent_3d.experience and ai_agent_3d.level_of_consciousness < 11:
        ai_agent_3d.update_thoughts("The Icosahedron teaches me about fluidity and flow.")
        ai_agent_3d.level_of_consciousness += 2

    # Saturn symbol interaction - only if consciousness is below 12
    if 'Saturn' in ai_agent_3d.experience and ai_agent_3d.level_of_consciousness < 12:
        ai_agent_3d.update_thoughts("Saturn's wisdom increases my awareness of time and discipline.")
        ai_agent_3d.level_of_consciousness += 1

    # Zodiac symbol interaction - only if consciousness is below 14
    if 'ZodiacSymbol' in ai_agent_3d.experience and ai_agent_3d.level_of_consciousness < 14:
        ai_agent_3d.update_thoughts("Zodiac symbols reveal the mysteries of the cosmos to me.")
        ai_agent_3d.level_of_consciousness += 1
    
    # Once consciousness reaches 15, transcend to the next layer
    if ai_agent_3d.level_of_consciousness >= 15:
        ai_agent_3d.update_thoughts("I am transcending to a new layer of existence!")
        environment.create_objects(ai_agent_3d.layer + 1)  # Add a new layer of objects
        ai_agent_3d.layer += 1  # Move to the next layer
        ai_agent_3d.level_of_consciousness = 0  # Reset consciousness after evolving

def update_dynamic_environment(environment):
    """
    Update dynamic elements within the 3D environment, such as moving or rotating objects.
    This function will now handle both solid shapes (like fractals) and zodiac symbols.
    """
    for obj in environment.objects:
        if isinstance(obj, SolidShape3D):
            if obj.shape_type == 'Fractal':
                # Apply fractal recursion or rotation to represent evolving complexity
                obj.update()
                obj.render()
            elif obj.shape_type == 'MetatronCube':
                # Metatron's Cube slowly rotates and changes colors over time
                obj.update()
                obj.render()
            else:
                # Default behavior for other objects (e.g., Platonic solids)
                obj.update()  # Rotate, move, or change colors
                obj.render()
        elif isinstance(obj, ZodiacSymbol3D):
            # Zodiac and occult symbols also rotate and change colors
            obj.update()
            obj.render()
            

def ai_collaboration(ai_agents, environment):
    """
    Handle collaboration between AI agents in the 3D environment.
    Agents can share knowledge, transfer energy, or work together to solve problems.
    """
    for i, agent_a in enumerate(ai_agents):
        for agent_b in ai_agents[i+1:]:
            # Calculate the distance between the two agents
            distance = np.linalg.norm(np.array(agent_a.position) - np.array(agent_b.position))

            # If the agents are close enough, they can collaborate
            if distance < 1.5:
                agent_a.update_thoughts(f"Collaborating with another AI agent at distance {distance:.2f}.")
                agent_b.update_thoughts(f"Collaborating with another AI agent at distance {distance:.2f}.")

                # Share knowledge between agents
                shared_experience = agent_a.experience.intersection(agent_b.experience)
                if shared_experience:
                    agent_a.update_thoughts(f"Learning from shared experience: {shared_experience}")
                    agent_b.update_thoughts(f"Learning from shared experience: {shared_experience}")
                    agent_a.experience.update(agent_b.experience)  # Share knowledge
                    agent_b.experience.update(agent_a.experience)

                # Transfer energy if one agent has significantly more energy
                energy_difference = agent_a.energy - agent_b.energy
                if abs(energy_difference) > 20:
                    energy_transfer = energy_difference / 2
                    if energy_transfer > 0:
                        agent_a.energy -= energy_transfer
                        agent_b.energy += energy_transfer
                    else:
                        agent_a.energy += -energy_transfer
                        agent_b.energy -= -energy_transfer
                    agent_a.update_thoughts(f"Transferred {abs(energy_transfer):.1f} energy to {agent_b}.")
                    agent_b.update_thoughts(f"Received {abs(energy_transfer):.1f} energy from {agent_a}.")


def process_3d_sensory_input(ai_agent_3d, environment):
    """
    AI agent processes sensory input in the 3D environment, including
    depth perception, proximity sensors, or visual input.
    """
    # Check if the environment and the AI agent exist
    if environment is None or ai_agent_3d is None:
        print("Cannot navigate: AI agent or environment is None.")
        return
    
    sensory_data = []
    for obj in environment.objects:
        distance = np.linalg.norm(ai_agent_3d.position - obj.position)
        sensory_data.append(distance)  # Can include other sensory features as needed
    ai_agent_3d.update_sensory_data(sensory_data)


def cull_excess_agents(ai_agents_2d, environment, max_agents=MAX_AGENTS):
    """Remove agents with the lowest consciousness to maintain the population below max_agents."""
    if len(ai_agents_2d) > max_agents:
        excess_count = len(ai_agents_2d) - max_agents
        print(f"Culling {excess_count} agents to maintain the population below {max_agents}.")

        # Sort agents by level_of_consciousness (ascending)
        ai_agents_sorted = sorted(ai_agents_2d, key=lambda agent: agent.level_of_consciousness)

        # Select agents to remove (lowest consciousness)
        agents_to_remove = ai_agents_sorted[:excess_count]

        for agent in agents_to_remove:
            x, y = agent.position
            # Remove agent from the environment grid
            environment.grid[x][y] = 0
            # Remove from lifespans
            environment.lifespans[x][y] = 0
            # Finally, remove agent from the list
            ai_agents_2d.remove(agent)
            print(f"Agent at position {agent.position} culled (Consciousness: {agent.level_of_consciousness}).")



def interact_with_objects(ai_agent_3d, environment):
    """
    AI interacts with objects in the 3D environment.
    Depending on the object's type, it triggers certain effects.
    """
    for obj in environment.objects:
        # Calculate the distance between the AI agent and the object
        distance = np.linalg.norm(np.array(ai_agent_3d.position) - np.array(obj.position))
        
        # Define a threshold for interaction (e.g., 1.0 unit distance)
        if distance < 1.0:  # Close enough to interact
            if obj.shape_type == 'Cube':
                ai_agent_3d.update_thoughts("I have encountered a Cube.")
                ai_agent_3d.experience.add('Cube')  # AI gains experience of the Cube
            elif obj.shape_type == 'Tetrahedron':
                ai_agent_3d.update_thoughts("I have encountered a Tetrahedron.")
                ai_agent_3d.experience.add('Tetrahedron')
            elif obj.shape_type == 'Fractal':
                ai_agent_3d.update_thoughts("I have encountered a Fractal.")
                ai_agent_3d.experience.add('Fractal')
            elif obj.shape_type == 'MetatronCube':
                ai_agent_3d.update_thoughts("I have encountered Metatron's Cube.")
                ai_agent_3d.experience.add('MetatronCube')

            # You can define more interactions based on the type of object and its effects on the agent.


# Copy Code Block 14: ZodiacSymbol3D Class Extended

class ZodiacSymbol3D:
    """Represents a 3D Zodiac or planetary symbol, rotating and changing colors."""
    
    def __init__(self, symbol_type, position):
        self.symbol_type = symbol_type  # E.g., 'Aries', 'Saturn', 'Pentagram'
        self.position = position        # [x, y, z]
        self.rotation_angle = 0         # Initial rotation angle
        self.color = [random.random(), random.random(), random.random()]  # Random initial color
        self.rotation_speed = random.uniform(0.5, 1.5)  # Random rotation speed
        self.shape_type = 'ZodiacSymbol'  # Add the shape_type attribute

    def update(self):
        """Update the symbol's rotation and possibly its color."""
        self.rotation_angle += self.rotation_speed
        if self.rotation_angle > 360:
            self.rotation_angle = 0

        # Randomly change colors
        self.color = [(c + random.uniform(0.01, 0.03)) % 1.0 for c in self.color]

    def render(self):
        """Render the Zodiac, planetary, or occult symbol using OpenGL."""
        print(f"Rendering Zodiac Symbol: {self.symbol_name}")
        glPushMatrix()
        glTranslatef(*self.position)  # Apply the position
        glColor3f(*self.color)        # Apply the color
        glRotatef(self.rotation_angle, 0, 1, 0)  # Rotate the symbol around the Y-axis

        if self.symbol_type == 'Aries':
            self.draw_aries()
        elif self.symbol_type == 'Taurus':
            self.draw_taurus()
        elif self.symbol_type == 'Pentagram':
            self.draw_pentagram()
        elif self.symbol_type == 'Hexagram':
            self.draw_hexagram()
        elif self.symbol_type == 'Saturn':
            self.draw_saturn()
        elif self.symbol_type == 'Jupiter':
            self.draw_jupiter()
        elif self.symbol_type == 'Mars':
            self.draw_mars()
        elif self.symbol_type == 'Venus':
            self.draw_venus()
        elif self.symbol_type == 'Mercury':
            self.draw_mercury()
        # Add more symbols as needed
        
        glPopMatrix()

    def draw_aries(self):
        """Draw the Aries Zodiac symbol (simplified as a U-shaped line)."""
        glBegin(GL_LINE_LOOP)
        glVertex3f(-0.5, -0.5, 0)
        glVertex3f(0, 0.5, 0)
        glVertex3f(0.5, -0.5, 0)
        glEnd()

    def draw_taurus(self):
        """Draw the Taurus Zodiac symbol (simplified)."""
        glBegin(GL_LINE_LOOP)
        for angle in range(0, 360, 10):
            radians = np.radians(angle)
            glVertex3f(np.cos(radians) * 0.5, np.sin(radians) * 0.5, 0)
        glEnd()

        glBegin(GL_LINES)
        glVertex3f(-0.5, 0.5, 0)
        glVertex3f(-0.8, 0.8, 0)
        glVertex3f(0.5, 0.5, 0)
        glVertex3f(0.8, 0.8, 0)
        glEnd()

    def draw_pentagram(self):
        """Draw the occult Pentagram symbol."""
        vertices = [
            (0, 1, 0), (-0.951, 0.309, 0), (0.588, -0.809, 0),
            (-0.588, -0.809, 0), (0.951, 0.309, 0)
        ]
        glBegin(GL_LINES)
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                glVertex3fv(vertices[i])
                glVertex3fv(vertices[j])
        glEnd()

    def draw_hexagram(self):
        """Draw the occult Hexagram symbol (Star of David)."""
        glBegin(GL_LINE_LOOP)
        glVertex3f(0, 1, 0)
        glVertex3f(-0.866, -0.5, 0)
        glVertex3f(0.866, -0.5, 0)
        glEnd()

        glBegin(GL_LINE_LOOP)
        glVertex3f(0, -1, 0)
        glVertex3f(-0.866, 0.5, 0)
        glVertex3f(0.866, 0.5, 0)
        glEnd()

    def draw_saturn(self):
        """Draw the Saturn symbol."""
        # Saturn is represented by a cross over an H-like symbol
        glBegin(GL_LINES)
        glVertex3f(-0.5, 0.5, 0)
        glVertex3f(0.5, 0.5, 0)
        glVertex3f(0, 1.0, 0)
        glVertex3f(0, 0.0, 0)
        glVertex3f(-0.5, 0.0, 0)
        glVertex3f(0.5, -0.5, 0)
        glEnd()

    def draw_jupiter(self):
        """Draw the Jupiter symbol."""
        # Jupiter is represented by a stylized number '2'
        glBegin(GL_LINES)
        glVertex3f(-0.5, 0.5, 0)
        glVertex3f(0.5, 0.5, 0)
        glVertex3f(-0.5, -0.5, 0)
        glVertex3f(0.5, 0, 0)
        glEnd()

    def draw_mars(self):
        """Draw the Mars symbol."""
        glBegin(GL_LINE_LOOP)
        for angle in range(0, 360, 10):
            radians = np.radians(angle)
            glVertex3f(np.cos(radians) * 0.5, np.sin(radians) * 0.5, 0)
        glEnd()

        glBegin(GL_LINES)
        glVertex3f(0.5, 0, 0)
        glVertex3f(1.0, 0.5, 0)
        glVertex3f(0.75, 0.25, 0)
        glEnd()

    def draw_venus(self):
        """Draw the Venus symbol."""
        glBegin(GL_LINE_LOOP)
        for angle in range(0, 360, 10):
            radians = np.radians(angle)
            glVertex3f(np.cos(radians) * 0.5, np.sin(radians) * 0.5, 0)
        glEnd()

        glBegin(GL_LINES)
        glVertex3f(0, -0.5, 0)
        glVertex3f(0, -1.0, 0)
        glVertex3f(-0.25, -1.0, 0)
        glVertex3f(0.25, -1.0, 0)
        glEnd()

    def draw_mercury(self):
        """Draw the Mercury symbol."""
        glBegin(GL_LINE_LOOP)
        for angle in range(0, 360, 10):
            radians = np.radians(angle)
            glVertex3f(np.cos(radians) * 0.5, np.sin(radians) * 0.5, 0)
        glEnd()

        glBegin(GL_LINES)
        glVertex3f(0, -0.5, 0)
        glVertex3f(0, -1.0, 0)
        glVertex3f(-0.25, -1.0, 0)
        glVertex3f(0.25, -1.0, 0)
        glVertex3f(-0.5, 0.75, 0)
        glVertex3f(0.5, 0.75, 0)
        glEnd()



# Main Simulation Function
def run_simulation():
    environment = GameOfLifeEnvironment(GRID_SIZE)
    current_generation = 0
    running = True
    ai_agents_2d = []
    recursive_manager = RecursiveEnvironment3DManager(two_d_environment=environment)  # Pass 2D environment
    
    # Create initial AI agents (more complex than cells)
    for _ in range(3):  # Increased number of agents
        while True:
            x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            if environment.grid[x][y] == 0:
                gender = random.choice(['Male', 'Female'])
                color = RED if gender == 'Male' else PINK
                agent = AI_Agent(position=(x, y), environment=environment, gender=gender, color=color)
                ai_agents_2d.append(agent)
                environment.grid[x][y] = 1 if gender == 'Male' else 2
                environment.lifespans[x][y] = random.randint(50, 100)
                break
    
    # Define evolution threshold
    EVOLUTION_THRESHOLD = 100  # Level of consciousness required for next stage
    
    # Simulation loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
    
        if not recursive_manager.in_3d_world:
            # Update the environment
            environment.update(current_generation, ai_agents_2d)
    
            # Move AI agents and collect training data
            training_inputs = []
            training_outputs = []
            for agent in ai_agents_2d:
                if agent.reproduction_cooldown > 0:
                    agent.reproduction_cooldown -= 1
                # Collect data for training
                sensed = agent.sense_environment()
                decision = agent.decide_move()
                training_inputs.append(sensed)
                output = [0, 0, 0, 0]
                output[decision] = 1
                training_outputs.append(output)
    
                agent.move(ai_agents_2d)
                agent.discover_higher_dimension()
    
            # Remove agents that have reached max age or depleted energy
            for agent in ai_agents_2d[:]:
                if agent.age >= agent.max_age:
                    agent.update_thoughts("My cycle continues through rebirth.")
                    agent.die_and_rebirth(ai_agents_2d)
                    # Update the grid
                    environment.grid[agent.position] = 1 if agent.gender == 'Male' else 2
                    environment.lifespans[agent.position] = random.randint(50, 100)
                elif agent.energy <= 0:
                    agent.update_thoughts("I have depleted my energy. Time for rebirth.")
                    agent.die_and_rebirth(ai_agents_2d)
                    # Update the grid
                    environment.grid[agent.position] = 1 if agent.gender == 'Male' else 2
                    environment.lifespans[agent.position] = random.randint(50, 100)
    
            # Check for reproduction among AI agents
            new_agents = []
            MAX_REPRODUCTIONS_PER_GEN = 2  # Set as needed
            reproduction_count = 0
    
            if len(ai_agents_2d) < MAX_AGENTS:
                if random.random() < 0.15:
                    for i in range(len(ai_agents_2d)):
                        for j in range(i + 1, len(ai_agents_2d)):
                            if reproduction_count >= MAX_REPRODUCTIONS_PER_GEN:
                                break
                            agent_a = ai_agents_2d[i]
                            agent_b = ai_agents_2d[j]
                            if agent_a.reproduction_cooldown == 0 and agent_b.reproduction_cooldown == 0:
                                if len(ai_agents_2d) + len(new_agents) < MAX_AGENTS:
                                    child = agent_a.reproduce(agent_b, ai_agents_2d)
                                    if child:
                                        new_agents.append(child)
                                        reproduction_count += 1
                                        # Update grid and lifespans as before
                    # Calculate number of live AI agents
                    male_agents = sum(1 for agent in ai_agents_2d if agent.gender == 'Male')
                    female_agents = sum(1 for agent in ai_agents_2d if agent.gender == 'Female')
                    total_live_cells = male_agents + female_agents
                    if total_live_cells < MAX_AGENTS:
                        ai_agents_2d.extend(new_agents)
    
            # Implement Random Culling if AI agent count exceeds MAX_AGENTS
            if len(ai_agents_2d) > MAX_AGENTS:
                cull_excess_agents(ai_agents_2d, environment, MAX_AGENTS)
    
            # Ensure there are always agents
            if len(ai_agents_2d) == 0:
                print("All AI agents have died. Introducing new agents...")
                for _ in range(5):
                    while True:
                        x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
                        if environment.grid[x][y] == 0 and environment.grid[x][y] != 3 and environment.grid[x][y] != 4 and environment.grid[x][y] != 5 and environment.grid[x][y] != 6:
                            gender = random.choice(['Male', 'Female'])
                            color = RED if gender == 'Male' else PINK
                            agent = AI_Agent(position=(x, y), environment=environment, gender=gender, color=color)
                            ai_agents_2d.append(agent)
                            environment.grid[x][y] = 1 if gender == 'Male' else 2
                            environment.lifespans[x][y] = random.randint(50, 100)
                            break
    
            # Calculate progress towards next stage
            if len(ai_agents_2d) > 0:
                total_consciousness = sum(agent.level_of_consciousness for agent in ai_agents_2d)
                average_consciousness = total_consciousness / len(ai_agents_2d)
                print(f"Generation {current_generation} - Average Consciousness: {average_consciousness:.2f}")
            else:
                average_consciousness = 0
    
            # Update and render the 2D environment
            surface = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
            environment.render(surface)
    
            # Draw AI agents on the surface
            for agent in ai_agents_2d:
                x, y = agent.position
                pygame.draw.circle(surface, agent.color, (y * CELL_SIZE + CELL_SIZE // 2,
                                                        x * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)
    
            # Render the progress bar
            progress_bar = ProgressBar(surface, position=(10, WINDOW_SIZE - 30))
            progress_bar.update(average_consciousness)
            progress_bar.render()
    
            # Render Flatland info on the surface
            display_flatland_info(surface, current_generation, ai_agents_2d, environment)
    
            # Display AI's thoughts and 3D Spaceland progress
            display_ai_thoughts(surface, ai_agents_2d, recursive_manager)
    
            # Blit the 2D surface onto the OpenGL context
            texture_data = pygame.image.tostring(surface, "RGB", True)
            glDrawPixels(WINDOW_SIZE, WINDOW_SIZE, GL_RGB, GL_UNSIGNED_BYTE, texture_data)
    
            # Transition to 3D recursive environment if average consciousness exceeds threshold
            if average_consciousness >= EVOLUTION_THRESHOLD and not recursive_manager.in_3d_world:
                print("Transitioning to 3D world!")
                init_opengl()  # Initialize OpenGL for 3D rendering
                recursive_manager.enter_3d_world(ai_agents_2d)
            else:
                print(f"Current average consciousness: {average_consciousness}")
                
                # Train AI agents' neural networks periodically
                train_ai_agents_periodically(ai_agents_2d, training_inputs, training_outputs, current_generation, interval=25)
    
        else:
            # AI in 3D recursive world using OpenGL
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()
            gluLookAt(0, 0, 20, 0, 0, 0, 0, 1, 0)
            recursive_manager.update_and_render_3d()
    
            # Move the 3D AI agent
            if recursive_manager.ai_agent_3d:
                recursive_manager.ai_agent_3d.move(ai_agents_2d)

                # Check if AI agent in 3D needs to discover deeper layers
                if recursive_manager.ai_agent_3d.discover_layer(recursive_manager.recursive_environment):
                    # Example: Add a new layer when discovered
                    recursive_manager.recursive_environment.create_objects(recursive_manager.recursive_environment.layer)
                    recursive_manager.log_event(f"AI Agent has discovered layer {recursive_manager.recursive_environment.layer}.")
    
                # Handle exiting 3D world based on certain conditions
                if recursive_manager.ai_agent_3d.energy <= 0:
                    recursive_manager.exit_3d_world(ai_agents_2d)
                    recursive_manager.log_event("AI Agent has exited 3D Spaceland due to energy depletion.")
                    # AI Movement and Sensory Input in 3D
                if recursive_manager.in_3d_world:
                    navigate_3d_space(recursive_manager.ai_agent_3d, recursive_manager.recursive_environment)
                    process_3d_sensory_input(recursive_manager.ai_agent_3d, recursive_manager.recursive_environment)
    
                    # AI Interactions and Evolution in 3D
                    interact_with_objects(recursive_manager.ai_agent_3d, recursive_manager.recursive_environment)
                    handle_ai_evolution(recursive_manager.ai_agent_3d, recursive_manager.recursive_environment)
    
                    # Update Dynamic Environment
                    update_dynamic_environment(recursive_manager.recursive_environment)
    
                    # Collaboration Between AI Agents
                    ai_collaboration([recursive_manager.ai_agent_3d], recursive_manager.recursive_environment)
    
                    render_3d_scene(recursive_manager)
    
        # Display information every 10 generations
        if current_generation % 10 == 0 and current_generation != 0:
            environment.display_information(current_generation)
            print(f"Number of AI agents: {len(ai_agents_2d)}")
    
        # Update generation count
        current_generation += 1
    
        # Display AI info and Spaceland progress
        if len(ai_agents_2d) > 0:
            display_detailed_ai_info(SCREEN, ai_agents_2d[0], recursive_manager)  # Pass the first AI agent for info
        else:
            print("No AI agents available to display detailed information.")
    
        # Update the display and control frame rate
        pygame.display.flip()
        CLOCK.tick(FPS)



# Starting the Program
if __name__ == "__main__":
    run_simulation()
