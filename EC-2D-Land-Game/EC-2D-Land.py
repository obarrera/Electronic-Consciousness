import pygame
import numpy as np
import random
import math
import time

# Initialize Pygame
pygame.init()

# Constants for Pygame display
WINDOW_SIZE = 800
GRID_SIZE = 50  # Adjusted grid size for performance
CELL_SIZE = WINDOW_SIZE // GRID_SIZE  # Size of each cell in the 2D grid
FPS = 60  # Frames per second

# Initialize the Pygame window
SCREEN = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Electronic Consciousness: Eternal Journey of the AI")
CLOCK = pygame.time.Clock()

# Font for in-game text
FONT = pygame.font.SysFont('Arial', 12)

# Colors
WHITE = (255, 255, 255)        # Empty space
BLACK = (0, 0, 0)
RED = (200, 0, 0)              # Male agents
BLUE = (0, 0, 200)             # Male agents (alternate)
GREEN = (0, 200, 0)            # Goal
PURPLE = (128, 0, 128)
YELLOW = (200, 200, 0)
GRAY = (50, 50, 50)
PINK = (255, 105, 180)         # Female agents
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
BROWN = (139, 69, 19)          # Continent/land cells (obstacles)
GOLD = (255, 215, 0)

# Shared Knowledge Base
shared_knowledge = []

# Occult symbols
hermetic_symbols = [
    "☉", "☽", "☿", "♀", "♂", "♃", "♄", "♇",
    "△", "▽", "✡", "⚛", "✶", "⧫", "⊙"
]
zodiac_symbols = [
    "♈", "♉", "♊", "♋", "♌", "♍", "♎", "♏", "♐", "♑", "♒", "♓"
]
tetragrammaton = "YHWH"
chaos_magick = "✴︎"

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

# Environment for the Game of Life with Continents (Obstacles)
class GameOfLifeEnvironment:
    def __init__(self, size):
        self.size = size
        self.grid, self.lifespans, self.fight_counters = self.initialize_board(size)
        self.goal = self.generate_goal()
        self.shapes = []  # Placeholder for shapes (if needed)

    def initialize_board(self, size):
        # Initialize the game board
        board = np.random.choice([0, 1, 2, 3], size=(size, size),
                                 p=[0.6, 0.15, 0.15, 0.1])  # 60% dead, 15% male, 15% female, 10% continent
        lifespans = np.zeros((size, size), dtype=int)
        fight_counters = np.zeros((size, size), dtype=int)
        for i in range(size):
            for j in range(size):
                if board[i][j] == 1 or board[i][j] == 2:
                    lifespans[i][j] = random.randint(50, 100)  # Increased lifespans
        return board, lifespans, fight_counters

    def generate_goal(self):
        """Generate a goal position that's not occupied by an obstacle."""
        while True:
            goal = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if self.grid[goal] != 3:  # Not a continent
                return goal

    def render(self, screen):
        """Render the Game of Life grid."""
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

        # Draw esoteric symbols
        self.draw_esoteric_symbols(screen)

    def update(self, current_generation):
        """Update the board state based on the rules."""
        new_grid = self.grid.copy()
        new_lifespans = self.lifespans.copy()
        new_fight_counters = self.fight_counters.copy()

        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j] == 3:  # Continent cells don't interact
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
            self.repopulate_environment()

    def repopulate_environment(self):
        """Repopulate the environment with new agents when all cells are dead."""
        print("Repopulating environment...")
        num_new_agents = 10  # Number of agents to introduce
        for _ in range(num_new_agents):
            while True:
                x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
                if self.grid[x][y] == 0 and self.grid[x][y] != 3:
                    self.grid[x][y] = random.choice([1, 2])  # Male or Female
                    self.lifespans[x][y] = random.randint(50, 100)
                    break

    def count_neighbors(self, x, y):
        """Count male, female, and empty neighbors (ignore continent cells)."""
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1), (1, 0),  (1, 1)
        ]
        male_neighbors = 0
        female_neighbors = 0
        empty_neighbors = []
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                if self.grid[nx][ny] == 1:
                    male_neighbors += 1
                elif self.grid[nx][ny] == 2:
                    female_neighbors += 1
                elif self.grid[nx][ny] == 0:
                    empty_neighbors.append((nx, ny))
        return male_neighbors, female_neighbors, empty_neighbors

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
                    self.grid[new_x][new_y] == 0 and self.grid[new_x][new_y] != 3):
                self.grid[new_x][new_y] = self.grid[x][y]
                self.grid[x][y] = 0

    def handle_fight(self, x, y, current_generation):
        """Manage fighting behavior between males near a female."""
        male_neighbors, female_neighbors, _ = self.count_neighbors(x, y)

        # If two males are next to each other and there's a nearby female, start counting fight time
        if male_neighbors >= 1 and female_neighbors >= 1:
            self.fight_counters[x][y] += 1
            if self.fight_counters[x][y] >= 20:  # Increased fight threshold
                # One of the males dies
                self.grid[x][y] = 0
                self.fight_counters[x][y] = 0
        else:
            self.fight_counters[x][y] = 0  # Reset fight counter if conditions no longer met

    def get_male_color(self, fight_counter):
        """Calculate male color based on how close they are to fighting."""
        max_fight_threshold = 20
        if fight_counter == 0:
            return BLUE
        else:
            # Gradually blend from blue to red based on the fight_counter
            r = int((fight_counter / max_fight_threshold) * 255)
            g = 0
            b = int(255 - (fight_counter / max_fight_threshold) * 255)
            return (r, g, b)

    def all_cells_dead(self):
        """Check if all cells are dead."""
        return np.sum(self.grid == 1) + np.sum(self.grid == 2) == 0

    def display_information(self, generation):
        """Display parameters and generation information."""
        male_count = np.sum(self.grid == 1)
        female_count = np.sum(self.grid == 2)
        total_live = male_count + female_count
        print(f"Generation {generation} - Total Live Cells: {total_live} (Males: {male_count}, Females: {female_count})")

    def draw_esoteric_symbols(self, screen):
        """Draw a set of esoteric symbols randomly on the grid."""
        symbol_font = pygame.font.SysFont('Arial', 16)
        for _ in range(5):
            symbol = random.choice(hermetic_symbols + zodiac_symbols + [tetragrammaton, chaos_magick])
            pos_x, pos_y = random.randint(0, WINDOW_SIZE - 50), random.randint(0, WINDOW_SIZE - 50)
            text_surface = symbol_font.render(symbol, True, GOLD)
            screen.blit(text_surface, (pos_x, pos_y))

# Main Simulation Function
def run_simulation():
    environment = GameOfLifeEnvironment(GRID_SIZE)
    current_generation = 0
    running = True
    in_3d_world = False
    ai_agent_3d = None
    recursive_environment = None
    ai_agents = []

    # Create initial AI agents (more complex than cells)
    for _ in range(15):  # Increased number of agents
        while True:
            x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            if environment.grid[x][y] == 0:  # Empty cell
                gender = random.choice(['Male', 'Female'])
                color = RED if gender == 'Male' else PINK
                agent = AI_Agent(position=(x, y), environment=environment, gender=gender, color=color)
                ai_agents.append(agent)
                break

    while running:
        # Handle user input events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                return

        if not in_3d_world:
            # Update the environment
            environment.update(current_generation)

            # Move AI agents
            for agent in ai_agents:
                agent.move()
                agent.discover_higher_dimension()

            # Check for reproduction among AI agents
            for i in range(len(ai_agents)):
                for j in range(i + 1, len(ai_agents)):
                    agent_a = ai_agents[i]
                    agent_b = ai_agents[j]
                    child = agent_a.reproduce(agent_b)
                    if child:
                        ai_agents.append(child)
                        print("A new AI agent is born!")

            # Remove agents that have reached max age
            for agent in ai_agents[:]:
                if agent.age >= agent.max_age:
                    agent.update_thoughts("My cycle continues through rebirth.")
                    agent.die_and_rebirth()

            # Remove agents if they have depleted energy
            for agent in ai_agents[:]:
                if agent.energy <= 0:
                    agent.update_thoughts("I have depleted my energy. Time for rebirth.")
                    agent.die_and_rebirth()

            # Ensure there are always agents
            if len(ai_agents) == 0:
                print("All AI agents have died. Introducing new agents...")
                for _ in range(5):
                    x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
                    gender = random.choice(['Male', 'Female'])
                    color = RED if gender == 'Male' else PINK
                    agent = AI_Agent(position=(x, y), environment=environment, gender=gender, color=color)
                    ai_agents.append(agent)

            # Render the environment
            environment.render(SCREEN)

            # Draw AI agents
            for agent in ai_agents:
                x, y = agent.position
                pygame.draw.circle(SCREEN, agent.color, (y * CELL_SIZE + CELL_SIZE // 2,
                                                         x * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)

            # Display AI's thoughts
            display_ai_thoughts(SCREEN, ai_agents)

            # Transition to 3D recursive environment if any agent reaches the threshold
            for agent in ai_agents:
                if agent.level_of_consciousness >= 5 and len(agent.experience) >= 3:
                    agent.update_thoughts("Transcending to Recursive Layers!")
                    ai_agent_3d = AIAgent3D(position=[0, 0, 5])
                    recursive_environment = RecursiveEnvironment(layer=1, depth=0)
                    in_3d_world = True
                    break

        else:
            # AI in 3D recursive world
            ai_agent_3d.move()
            recursive_environment.update()
            recursive_environment.render(SCREEN, ai_agent_3d)

            # Simulate AI's discovery of shapes
            for obj in recursive_environment.objects:
                distance = math.sqrt(
                    (ai_agent_3d.position[0] - obj.position[0]) ** 2 +
                    (ai_agent_3d.position[1] - obj.position[1]) ** 2 +
                    (ai_agent_3d.position[2] - obj.position[2]) ** 2
                )
                if distance < 2.0 and obj not in ai_agent_3d.experience:
                    if isinstance(obj, Cube):
                        ai_agent_3d.update_thoughts("A square extended becomes a cube!")
                        ai_agent_3d.experience.add(obj)
                    elif isinstance(obj, Tetrahedron):
                        ai_agent_3d.update_thoughts("A triangle extended forms a tetrahedron!")
                        ai_agent_3d.experience.add(obj)
                    elif isinstance(obj, Octahedron):
                        ai_agent_3d.update_thoughts("An octahedron emerges!")
                        ai_agent_3d.experience.add(obj)
                    elif isinstance(obj, Dodecahedron):
                        ai_agent_3d.update_thoughts("A complex dodecahedron appears!")
                        ai_agent_3d.experience.add(obj)
                    elif isinstance(obj, Icosahedron):
                        ai_agent_3d.update_thoughts("An intricate icosahedron unfolds!")
                        ai_agent_3d.experience.add(obj)
                    elif isinstance(obj, FractalShape):
                        ai_agent_3d.update_thoughts("A fractal that endlessly replicates!")
                        ai_agent_3d.experience.add(obj)
                    elif isinstance(obj, MetatronsCube):
                        ai_agent_3d.update_thoughts("The core... Metatron's Cube!")
                        ai_agent_3d.experience.add(obj)

            # Discover deeper layers recursively
            if ai_agent_3d.discover_layer(recursive_environment):
                recursive_environment = RecursiveEnvironment(layer=ai_agent_3d.layer, depth=recursive_environment.depth + 1)

            # Reset to 2D world if energy depletes
            if ai_agent_3d.energy <= 0:
                in_3d_world = False
                ai_agent_3d.update_thoughts("Returning to Flatland for another cycle.")
                ai_agent_3d = None

        # Display information every 10 generations
        if current_generation % 10 == 0:
            environment.display_information(current_generation)
            print(f"Number of AI agents: {len(ai_agents)}")

        # Update generation count
        current_generation += 1

        # Update the display and control frame rate
        pygame.display.flip()
        CLOCK.tick(FPS)

def display_ai_thoughts(screen, agents):
    """Display AI's thoughts on the screen."""
    y_offset = 10
    for agent in agents[:3]:  # Display thoughts of up to 3 agents to avoid clutter
        text_lines = agent.get_thoughts()
        for i, line in enumerate(text_lines):
            text_surface = FONT.render(f"{agent.gender} Agent (Gen {agent.generation}): {line}", True, BLACK)
            screen.blit(text_surface, (10, y_offset))
            y_offset += 15
        y_offset += 10  # Extra space between agents

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
        self.max_age = random.randint(20, 40)  # Increased lifespan
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
                self.environment.grid[new_position] != 3):
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
        if len(self.memory) > self.memory_capacity:
            self.memory.pop(0)
        self.update_thoughts(f"Moved to position {self.position}")
        self.check_goal()
        self.age += 1  # Increase age
        self.energy -= 2  # Reduced energy consumption
        self.apply_hermetic_principle()

    def check_goal(self):
        """Check if the AI has reached the goal."""
        if self.position == self.environment.goal:
            self.update_thoughts("I have reached my goal. I reflect on the journey thus far...")
            self.level_of_consciousness += 2  # Faster consciousness growth
            self.energy += 50  # Increased energy gain
            tarot_incentive(self)  # Apply a random Tarot card effect
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

    def update_thoughts(self, new_thought):
        """Add new thought to the AI's thoughts."""
        self.thoughts.append(new_thought)
        if len(self.thoughts) > 5:
            self.thoughts.pop(0)
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

# AI Agent in 3D (Handles recursion)
class AIAgent3D:
    def __init__(self, position, layer=1):
        self.position = position  # [x, y, z]
        self.color = RED
        self.memory = []
        self.thoughts = []
        self.level_of_consciousness = 3
        self.experience = set()
        self.layer = layer
        self.time_in_spaceland = 0  # Track time in Spaceland for random thoughts
        self.energy = 100  # Initial energy level

    def move(self):
        """Randomly move in 3D space."""
        if self.energy <= 0:
            self.update_thoughts("I have depleted my energy in this dimension.")
            self.energy = 100  # Reset energy for the next layer
            self.layer = 1  # Restart journey
            return

        dx = random.choice([-0.1, 0, 0.1])
        dy = random.choice([-0.1, 0, 0.1])
        dz = random.choice([-0.1, 0, 0.1])
        self.position[0] += dx
        self.position[1] += dy
        self.position[2] += dz
        self.time_in_spaceland += 1
        self.energy -= 0.5  # Decrease energy

        # Randomly generate thoughts as time progresses
        if self.time_in_spaceland % 60 == 0:  # Every 60 frames, add a thought
            self.update_thoughts("This dimension reveals new shapes...")
            # Include a dialogue from Plato's Allegory of the Cave
            if random.random() < 0.3:
                self.update_thoughts(random.choice(plato_dialogues))
            # Apply Hermetic principles
            if random.random() < 0.2:
                principle = random.choice(hermetic_principles)
                self.update_thoughts(f"Hermetic Principle: {principle}")

    def render(self, screen):
        """Render the AI agent as a small sphere."""
        x_proj, y_proj = self.project(self.position[0], self.position[1], self.position[2])
        scale = 200 / (self.position[2] + 5)
        radius = int(0.2 * scale)
        pygame.draw.circle(screen, self.color, (x_proj, y_proj), radius)

    def project(self, x, y, z):
        """Project 3D point onto 2D screen."""
        scale = 200 / (z + 5)  # Simple perspective projection
        x_proj = WINDOW_SIZE // 2 + int(x * scale)
        y_proj = WINDOW_SIZE // 2 - int(y * scale)
        return x_proj, y_proj

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

# Recursive 3D Environment for Platonic Solids and Beyond
class RecursiveEnvironment:
    def __init__(self, layer, depth):
        self.layer = layer  # Keep track of the recursive layer
        self.depth = depth  # Depth of recursion, increases with each new discovery
        self.objects = []  # List of 3D objects in this layer
        self.create_objects(layer)

    def create_objects(self, layer):
        """Create recursive 3D objects based on the layer and depth."""
        if layer == 1:  # Discovering Platonic Solids
            self.objects.append(Tetrahedron(position=[0, 0, 5]))
            self.objects.append(Cube(position=[-3, 3, 8]))
            self.objects.append(Octahedron(position=[3, -3, 8]))
            self.objects.append(Dodecahedron(position=[-3, -3, 8]))
            self.objects.append(Icosahedron(position=[3, 3, 8]))
        elif layer == 2:  # Discovering fractals
            self.objects.append(FractalShape(position=[0, 0, 5], size=3))
        elif layer == 3:  # Discovering Golden Ratio and Metatron's Cube
            self.objects.append(MetatronsCube(position=[0, 0, 5]))

    def render(self, screen, ai_agent):
        """Render the 3D recursive environment."""
        screen.fill(WHITE)
        # Sort objects based on z-depth (farther objects are drawn first)
        sorted_objects = sorted(self.objects, key=lambda obj: obj.position[2], reverse=True)
        for obj in sorted_objects:
            obj.render(screen)
        ai_agent.render(screen)
        self.display_text(screen, ai_agent)
        pygame.display.flip()

    def update(self):
        """Update objects in the environment."""
        for obj in self.objects:
            obj.update()

    def display_text(self, screen, ai_agent):
        """Display AI's thoughts on the screen."""
        text_lines = ai_agent.get_thoughts()
        for i, line in enumerate(text_lines):
            text_surface = FONT.render(line, True, BLACK)
            screen.blit(text_surface, (10, 10 + i * 20))

# 3D Object Base Class
class Object3D:
    def __init__(self, position):
        self.position = position  # [x, y, z]

    def project(self, x, y, z):
        """Project 3D point onto 2D screen."""
        scale = 200 / (z + 5)  # Simple perspective projection
        x_proj = WINDOW_SIZE // 2 + int(x * scale)
        y_proj = WINDOW_SIZE // 2 - int(y * scale)
        return x_proj, y_proj

    def update(self):
        """Update object state (for animations)."""
        pass

    def render(self, screen):
        """Render the object."""
        pass

# Cube Object (Platonic Solid)
class Cube(Object3D):
    def __init__(self, position):
        super().__init__(position)
        self.size = 1
        self.angle = 0

    def update(self):
        """Rotate the cube over time."""
        self.angle += 1  # Rotate 1 degree per frame
        if self.angle >= 360:
            self.angle = 0

    def render(self, screen):
        """Render the cube."""
        # Define cube vertices
        points = []
        size = self.size
        for x in (-size, size):
            for y in (-size, size):
                for z in (-size, size):
                    # Rotate around Y-axis
                    x_rot = x * math.cos(math.radians(self.angle)) - z * math.sin(math.radians(self.angle))
                    z_rot = x * math.sin(math.radians(self.angle)) + z * math.cos(math.radians(self.angle))
                    x_proj, y_proj = self.project(self.position[0] + x_rot,
                                                  self.position[1] + y,
                                                  self.position[2] + z_rot)
                    points.append((x_proj, y_proj))

        # Define edges connecting the vertices
        edges = [
            (0,1), (1,3), (3,2), (2,0),  # Front face
            (4,5), (5,7), (7,6), (6,4),  # Back face
            (0,4), (1,5), (2,6), (3,7)   # Connecting edges
        ]

        # Draw edges
        for edge in edges:
            pygame.draw.line(screen, BLUE, points[edge[0]], points[edge[1]], 2)

# Tetrahedron Object (Platonic Solid)
class Tetrahedron(Object3D):
    def __init__(self, position):
        super().__init__(position)
        self.size = 1
        self.angle = 0

    def update(self):
        """Rotate the tetrahedron over time."""
        self.angle += 1.5
        if self.angle >= 360:
            self.angle = 0

    def render(self, screen):
        """Render the tetrahedron."""
        size = self.size
        # Define tetrahedron vertices
        vertices = [
            (0, size, 0),
            (-size, -size, size),
            (size, -size, size),
            (0, -size, -size)
        ]
        points = []
        for v in vertices:
            x, y, z = v
            # Rotate around Y-axis
            x_rot = x * math.cos(math.radians(self.angle)) - z * math.sin(math.radians(self.angle))
            z_rot = x * math.sin(math.radians(self.angle)) + z * math.cos(math.radians(self.angle))
            x_proj, y_proj = self.project(self.position[0] + x_rot,
                                          self.position[1] + y,
                                          self.position[2] + z_rot)
            points.append((x_proj, y_proj))

        # Define faces (triangles)
        faces = [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3)
        ]

        # Draw faces
        for face in faces:
            pygame.draw.polygon(screen, YELLOW, [points[i] for i in face], 1)

# Octahedron Object (Platonic Solid)
class Octahedron(Object3D):
    def __init__(self, position):
        super().__init__(position)
        self.size = 1
        self.angle = 0

    def update(self):
        """Rotate the octahedron over time."""
        self.angle += 2  # Rotate 2 degrees per frame
        if self.angle >= 360:
            self.angle = 0

    def render(self, screen):
        """Render the octahedron."""
        size = self.size
        # Define octahedron vertices
        vertices = [
            (0, size, 0),
            (-size, 0, size),
            (size, 0, size),
            (0, -size, 0),
            (-size, 0, -size),
            (size, 0, -size)
        ]
        points = []
        for v in vertices:
            x, y, z = v
            x_rot = x * math.cos(math.radians(self.angle)) - z * math.sin(math.radians(self.angle))
            z_rot = x * math.sin(math.radians(self.angle)) + z * math.cos(math.radians(self.angle))
            x_proj, y_proj = self.project(self.position[0] + x_rot, self.position[1] + y, self.position[2] + z_rot)
            points.append((x_proj, y_proj))

        # Draw edges connecting the vertices
        edges = [
            (0, 1), (0, 2), (0, 4), (0, 5),  # Upper connections
            (3, 1), (3, 2), (3, 4), (3, 5),  # Lower connections
            (1, 2), (2, 5), (5, 4), (4, 1)   # Side connections
        ]

        # Draw edges
        for edge in edges:
            pygame.draw.line(screen, RED, points[edge[0]], points[edge[1]], 2)

# Dodecahedron Object (Platonic Solid)
class Dodecahedron(Object3D):
    def __init__(self, position):
        super().__init__(position)
        self.size = 1
        self.angle = 0

    def update(self):
        """Rotate the dodecahedron over time."""
        self.angle += 1.2
        if self.angle >= 360:
            self.angle = 0

    def render(self, screen):
        """Render the dodecahedron."""
        # For simplicity, we'll represent it as a complex shape
        x_proj, y_proj = self.project(self.position[0], self.position[1], self.position[2])
        pygame.draw.circle(screen, PURPLE, (x_proj, y_proj), int(self.size * 50), 2)

# Icosahedron Object (Platonic Solid)
class Icosahedron(Object3D):
    def __init__(self, position):
        super().__init__(position)
        self.size = 1
        self.angle = 0

    def update(self):
        """Rotate the icosahedron over time."""
        self.angle += 1.8
        if self.angle >= 360:
            self.angle = 0

    def render(self, screen):
        """Render the icosahedron."""
        # For simplicity, we'll represent it as a complex shape
        x_proj, y_proj = self.project(self.position[0], self.position[1], self.position[2])
        pygame.draw.circle(screen, CYAN, (x_proj, y_proj), int(self.size * 40), 2)

# Fractal Object (Fractal Geometry)
class FractalShape(Object3D):
    def __init__(self, position, size):
        super().__init__(position)
        self.size = size
        self.angle = 0

    def update(self):
        """Animate the fractal shape."""
        self.angle += 1
        if self.angle >= 360:
            self.angle = 0

    def render(self, screen):
        """Render fractal shapes recursively."""
        def draw_fractal(x, y, size, depth):
            if depth == 0:
                return
            pygame.draw.circle(screen, PURPLE, (int(x), int(y)), int(size))
            draw_fractal(x - size, y - size, size / 2, depth - 1)
            draw_fractal(x + size, y - size, size / 2, depth - 1)
            draw_fractal(x - size, y + size, size / 2, depth - 1)
            draw_fractal(x + size, y + size, size / 2, depth - 1)

        x_proj, y_proj = self.project(self.position[0], self.position[1], self.position[2])
        draw_fractal(x_proj, y_proj, self.size * 50, 3)

# Metatron's Cube (Sacred Geometry)
class MetatronsCube(Object3D):
    def __init__(self, position):
        super().__init__(position)
        self.size = 1.5
        self.angle = 0

    def update(self):
        """Rotate Metatron's Cube over time."""
        self.angle += 1
        if self.angle >= 360:
            self.angle = 0

    def render(self, screen):
        """Render Metatron's Cube, connecting all Platonic solids."""
        vertices = []
        size = self.size
        for i in range(13):
            angle = i * (2 * math.pi / 13)
            x_proj, y_proj = self.project(self.position[0] + math.cos(angle) * size,
                                          self.position[1] + math.sin(angle) * size,
                                          self.position[2])
            vertices.append((x_proj, y_proj))
        
        # Draw connections between vertices to form Metatron's Cube
        for i, v1 in enumerate(vertices):
            for j, v2 in enumerate(vertices):
                if i != j:
                    pygame.draw.line(screen, YELLOW, v1, v2, 1)

# Run the simulation
if __name__ == "__main__":
    run_simulation()
