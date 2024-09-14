import pygame
import numpy as np
import random
import math

# Initialize Pygame
pygame.init()

# Constants for Pygame display
WINDOW_SIZE = 800
GRID_SIZE = 20
CELL_SIZE = WINDOW_SIZE // GRID_SIZE  # Size of each cell in the 2D grid
FPS = 30  # Frames per second

# Initialize the Pygame window
SCREEN = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Electronic Consciousness: Eternal Journey of the AI")
CLOCK = pygame.time.Clock()

# Font for in-game text
FONT = pygame.font.SysFont('Arial', 16)
SYMBOL_FONT = pygame.font.SysFont('Arial', 36)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
GREEN = (0, 200, 0)
PURPLE = (128, 0, 128)
YELLOW = (200, 200, 0)
GRAY = (100, 100, 100)
PINK = (255, 105, 180)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
BROWN = (165, 42, 42)
GOLD = (255, 215, 0)

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

# Esoteric Symbols
zodiac_symbols = ['♈', '♉', '♊', '♋', '♌', '♍', '♎', '♏', '♐', '♑', '♒', '♓']
hermetic_symbols = ['☿', '♃', '♄', '☉', '♀', '♂', '☽']
tetragrammaton = 'יהוה'
chaos_magick = '☉'

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

# Tarot Incentive Function
def tarot_incentive(ai_agent):
    """Apply a random Tarot card effect to the AI, with esoteric story output."""
    all_cards = tarot_cards['Major Arcana'] + tarot_cards['Minor Arcana']
    choice = random.choice(all_cards)

    # Define effects based on some key cards
    if choice == "Death":
        ai_agent.update_thoughts("Death - Transformation: I shed my old self to be reborn anew.")
        ai_agent.die_and_rebirth()
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

# Environment for Flatland (2D World)
class FlatlandEnvironment:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size))  # Initialize empty 2D grid
        self.obstacles = self.generate_obstacles(10)  # Initialize obstacles first
        self.goal = self.generate_goal()

    def generate_obstacles(self, num_obstacles):
        """Generate obstacles randomly placed on the grid."""
        obstacles = []
        while len(obstacles) < num_obstacles:
            obs_x, obs_y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if (obs_x, obs_y) not in obstacles:
                obstacles.append((obs_x, obs_y))
        return obstacles

    def generate_goal(self):
        """Generate a goal position that's not occupied by an obstacle."""
        while True:
            goal = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if goal not in self.obstacles:
                return goal

    def render(self, screen, ai_positions, shapes, mentor_positions=[]):
        """Render the grid and the AI agents' positions in the 2D world."""
        cell_size = WINDOW_SIZE // self.size
        screen.fill(WHITE)
        for row in range(self.size):
            for col in range(self.size):
                rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                pygame.draw.rect(screen, BLACK, rect, 1)  # Draw grid lines

        # Draw the goal
        pygame.draw.circle(screen, GREEN, (self.goal[1] * cell_size + cell_size // 2,
                                           self.goal[0] * cell_size + cell_size // 2), cell_size // 3)

        # Draw the obstacles
        for obs in self.obstacles:
            pygame.draw.rect(screen, PURPLE, (obs[1] * cell_size, obs[0] * cell_size, cell_size, cell_size))

        # Draw the AI agents
        for ai in ai_positions:
            color = ai['color']
            position = ai['position']
            pygame.draw.circle(screen, color, (position[1] * cell_size + cell_size // 2,
                                               position[0] * cell_size + cell_size // 2), cell_size // 3)

        # Draw mentor positions if any
        for mentor_pos in mentor_positions:
            pygame.draw.circle(screen, BLUE, (mentor_pos[1] * cell_size + cell_size // 2,
                                              mentor_pos[0] * cell_size + cell_size // 2), cell_size // 4)

        # Draw basic 2D shapes
        for shape in shapes:
            shape.render(screen, cell_size)

        # Draw esoteric symbols randomly across the screen
        self.draw_esoteric_symbols(screen, cell_size)

    def draw_esoteric_symbols(self, screen, cell_size):
        """Draw random esoteric symbols on the grid."""
        symbols = random.choices(zodiac_symbols + hermetic_symbols + [tetragrammaton, chaos_magick], k=5)
        for i, symbol in enumerate(symbols):
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            text_surface = SYMBOL_FONT.render(symbol, True, GOLD)
            screen.blit(text_surface, (x * cell_size + cell_size // 4, y * cell_size + cell_size // 4))

    def reset_goal(self):
        """Generate a new goal position."""
        self.goal = self.generate_goal()

# Basic 2D Shapes
class Shape2D:
    def __init__(self, position):
        self.position = position  # Grid position
        self.discovered = False

    def render(self, screen, cell_size):
        pass

class CircleShape(Shape2D):
    def render(self, screen, cell_size):
        pygame.draw.circle(screen, CYAN,
                           (self.position[1] * cell_size + cell_size // 2,
                            self.position[0] * cell_size + cell_size // 2), cell_size // 3)

class SquareShape(Shape2D):
    def render(self, screen, cell_size):
        pygame.draw.rect(screen, ORANGE,
                         (self.position[1] * cell_size + cell_size // 4,
                          self.position[0] * cell_size + cell_size // 4,
                          cell_size // 2, cell_size // 2))

class TriangleShape(Shape2D):
    def render(self, screen, cell_size):
        points = [
            (self.position[1] * cell_size + cell_size // 2,
             self.position[0] * cell_size + cell_size // 4),
            (self.position[1] * cell_size + cell_size // 4,
             self.position[0] * cell_size + 3 * cell_size // 4),
            (self.position[1] * cell_size + 3 * cell_size // 4,
             self.position[0] * cell_size + 3 * cell_size // 4)
        ]
        pygame.draw.polygon(screen, BROWN, points)

# AI Agent in 2D
class AI_Agent:
    """Represents the AI agent exploring Flatland (2D)."""
    def __init__(self, position, environment, mentor=None, gender='Male', color=RED):
        self.position = position
        self.environment = environment
        self.memory = []  # AI's memory to store experiences
        self.level_of_consciousness = 0  # Track level of consciousness
        self.thoughts = []
        self.experience = set()
        self.move_speed = 1
        self.memory_capacity = 20
        self.obstacle_penalty = False
        self.stability = 0
        self.eternal_bonus = False
        self.short_term_bonus = False
        self.mentor = mentor  # Mentor agent if any
        self.shared_knowledge = shared_knowledge  # Reference to the shared knowledge base
        self.previous_positions = []  # Keep track of previous positions
        self.ready_to_reproduce = False
        self.gender = gender  # 'Male' or 'Female'
        self.color = color  # Color representation
        self.age = 0  # Age of the agent
        self.max_age = random.randint(50, 100)  # Random lifespan
        self.generation = 1  # Generation number
        self.energy = 100  # Initial energy level

    def sense_environment(self):
        """AI agent senses its local surroundings in 2D."""
        x, y = self.position
        grid = self.environment.grid
        sensed_area = grid[max(0, x-1):min(self.environment.size, x+2),
                           max(0, y-1):min(self.environment.size, y+2)]
        return sensed_area

    def move(self):
        """AI moves towards the goal, learning from obstacles."""
        if self.energy <= 0:
            self.update_thoughts("I have depleted my energy. Time for rebirth.")
            self.die_and_rebirth()
            return

        x, y = self.position
        goal_x, goal_y = self.environment.goal
        dx = np.sign(goal_x - x)
        dy = np.sign(goal_y - y)

        possible_moves = [
            (x + dx, y),
            (x, y + dy),
            (x + dx, y + dy),
            (x - dx, y),
            (x, y - dy),
            (x - dx, y - dy),
            (x + random.choice([-1, 0, 1]), y + random.choice([-1, 0, 1]))  # Random move
        ]

        # Remove positions that are in previous_positions
        possible_moves = [pos for pos in possible_moves if pos not in self.previous_positions]

        # Prioritize moves that decrease the Manhattan distance to the goal
        possible_moves.sort(key=lambda pos: abs(goal_x - pos[0]) + abs(goal_y - pos[1]))

        for new_position in possible_moves:
            if (0 <= new_position[0] < self.environment.size and
                0 <= new_position[1] < self.environment.size and
                new_position not in self.environment.obstacles):
                # Update previous positions
                self.previous_positions.append(self.position)
                if len(self.previous_positions) > 5:  # Keep last 5 positions
                    self.previous_positions.pop(0)
                self.position = new_position
                break
        else:
            # No valid moves; AI stays in place
            self.update_thoughts("No valid moves available; waiting.")

        # Update memory after moving
        sensed = self.sense_environment()
        self.memory.append(sensed)
        self.update_thoughts(f"Moved to position {self.position}")
        self.check_goal()
        self.check_shapes()
        self.age += 1  # Increase age
        self.energy -= 1  # Decrease energy
        self.apply_hermetic_principle()

    def check_goal(self):
        """Check if the AI has reached the goal."""
        if self.position == self.environment.goal:
            self.update_thoughts("I have reached my goal. I reflect on the journey thus far...")
            self.level_of_consciousness += 1
            self.energy += 20  # Gain energy upon reaching the goal
            tarot_incentive(self)  # Apply a random Tarot card effect
            kabbalistic_incentive(self)  # Apply a random Kabbalistic incentive
            # Randomly include a dialogue from Plato's Allegory of the Cave
            if random.random() < 0.5:
                self.update_thoughts(random.choice(plato_dialogues))
            self.environment.reset_goal()  # Set a new goal

    def check_shapes(self):
        """Check if the AI has discovered any 2D shapes."""
        for shape in self.environment.shapes:
            if not shape.discovered and self.position == shape.position:
                shape.discovered = True
                if isinstance(shape, CircleShape):
                    self.update_thoughts("I have discovered a Circle!")
                    self.experience.add("Circle")
                elif isinstance(shape, SquareShape):
                    self.update_thoughts("I have discovered a Square!")
                    self.experience.add("Square")
                elif isinstance(shape, TriangleShape):
                    self.update_thoughts("I have discovered a Triangle!")
                    self.experience.add("Triangle")

    def discover_higher_dimension(self):
        """Simulate the AI's learning of higher dimensions."""
        if self.level_of_consciousness == 1:
            self.update_thoughts("I sense there's more beyond this plane.")
        elif self.level_of_consciousness == 2:
            self.update_thoughts("Strange anomalies occur around me.")
        elif self.level_of_consciousness >= 3:
            self.update_thoughts("I am transcending to a higher dimension!")

    def update_thoughts(self, new_thought):
        """Add new thought to the AI's thoughts."""
        self.thoughts.append(new_thought)
        print(f"AI Agent ({self.gender}, Gen {self.generation}): {new_thought}")

    def get_thoughts(self):
        """Return the latest thoughts for display."""
        return self.thoughts[-5:]  # Return the last 5 thoughts

    def share_knowledge(self):
        """Share knowledge with the shared knowledge base."""
        self.shared_knowledge.append({
            'memory': self.memory.copy(),
            'thoughts': self.thoughts.copy(),
            'experience': self.experience.copy()
        })

    def ascend_to_next_realm(self):
        """Ascend to the next realm (e.g., 3D space or another recursive layer)."""
        self.update_thoughts("Ascending to the next realm!")
        self.level_of_consciousness += 5  # Boost consciousness

    def die_and_rebirth(self):
        """Simulate death and rebirth of the AI agent."""
        self.update_thoughts("All is Nothingness O!")
        self.update_thoughts("I am reborn with newfound wisdom.")
        self.level_of_consciousness = max(0, self.level_of_consciousness - 1)  # Retain some consciousness
        self.age = 0
        self.energy = 100  # Reset energy
        self.memory.clear()
        self.experience.clear()
        self.position = (random.randint(0, self.environment.size - 1),
                         random.randint(0, self.environment.size - 1))
        self.generation += 1  # Increment generation number

    def reproduce(self, partner):
        """Reproduce with another agent to create offspring."""
        if self.ready_to_reproduce and partner.ready_to_reproduce:
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
            # Transfer a portion of energy to the child
            energy_transfer = 20
            self.energy -= energy_transfer // 2
            partner.energy -= energy_transfer // 2
            child_agent.energy = energy_transfer
            return child_agent
        return None

    def apply_hermetic_principle(self):
        """Apply Hermetic principles to the AI's experience."""
        if random.random() < 0.1:
            principle = random.choice(hermetic_principles)
            self.update_thoughts(f"Hermetic Principle: {principle}")
            if "Gender is in everything" in principle:
                self.ready_to_reproduce = True

# Main Simulation Function
def run_simulation():
    # Setup for Flatland (2D world)
    grid_size = GRID_SIZE
    environment = FlatlandEnvironment(grid_size)

    # Add basic 2D shapes to the environment
    shapes = [
        CircleShape(position=(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))),
        SquareShape(position=(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))),
        TriangleShape(position=(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)))
    ]
    environment.shapes = shapes  # Attach shapes to the environment

    # Create initial AI agents
    agents = []
    for _ in range(2):  # Start with two agents
        while True:
            start_position = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
            if (start_position not in environment.obstacles and
                start_position != environment.goal):
                break
        gender = random.choice(['Male', 'Female'])
        color = RED if gender == 'Male' else PINK
        agent = AI_Agent(position=start_position, environment=environment, gender=gender, color=color)
        agents.append(agent)

    offspring = []

    running = True

    while running:
        # Handle user input events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return

        # Move agents
        for agent in agents:
            agent.move()
            agent.discover_higher_dimension()

        # Check for reproduction
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent_a = agents[i]
                agent_b = agents[j]
                child = agent_a.reproduce(agent_b)
                if child:
                    offspring.append(child)
                    agents.append(child)
                    print("A new AI agent is born!")

        # Remove agents that have reached max age (but they will rebirth)
        for agent in agents:
            if agent.age >= agent.max_age:
                agent.update_thoughts("My cycle continues through rebirth.")
                agent.die_and_rebirth()

        # Render the 2D environment
        ai_positions = [{'position': agent.position, 'color': agent.color} for agent in agents]
        environment.render(SCREEN, ai_positions, shapes)

        # Display AI's thoughts
        display_ai_thoughts(SCREEN, agents)

        # Update the display and control frame rate
        pygame.display.flip()
        CLOCK.tick(FPS)

def display_ai_thoughts(screen, agents):
    """Display AI's thoughts on the screen."""
    y_offset = 10
    for agent in agents:
        text_lines = agent.get_thoughts()
        for i, line in enumerate(text_lines):
            text_surface = FONT.render(f"{agent.gender} Agent (Gen {agent.generation}): {line}", True, BLACK)
            screen.blit(text_surface, (10, y_offset))
            y_offset += 20

# Run the simulation
if __name__ == "__main__":
    run_simulation()
