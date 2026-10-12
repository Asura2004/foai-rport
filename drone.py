import pygame
import sys
import math
import heapq
import time
import random

# Initialize Pygame
pygame.init()

info_object = pygame.display.Info()
SCREEN_WIDTH = info_object.current_w
SCREEN_HEIGHT = info_object.current_h
PANEL_HEIGHT = 90

SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Autonomous Drone Exploration: A* / Dijkstra")

CELL_SIZE = 22
GRID_COLS = SCREEN_WIDTH // CELL_SIZE
GRID_ROWS = (SCREEN_HEIGHT - PANEL_HEIGHT) // CELL_SIZE

SQRT2 = math.sqrt(2)

# Palette
COLOR_BG = (24, 26, 27)
COLOR_PANEL = (33, 37, 41)
COLOR_EMPTY = (45, 52, 58)
COLOR_OBSTACLE = (15, 17, 19)
COLOR_START = (40, 167, 69)      # Green
COLOR_GOAL = (220, 53, 69)       # Red
COLOR_PATH = (255, 193, 7)       # Yellow
COLOR_CLOSEST = (255, 133, 27)   # Orange for partial/closest reachable point
COLOR_OPEN = (23, 162, 184)      # Cyan
COLOR_CLOSED = (108, 117, 125)   # Grey
COLOR_DRONE = (255, 87, 34)      # Orange pointer
COLOR_TEXT = (248, 249, 250)

FONT = pygame.font.SysFont("monospace", 15, bold=True)
HEADER_FONT = pygame.font.SysFont("monospace", 17, bold=True)
CLOCK = pygame.time.Clock()

class GridWorld:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.start = (2, 2)
        self.goal = (rows - 3, cols - 3)

    def is_valid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] == 0

    def clear_walls(self):
        self.grid = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

    def generate_random_walls(self, density=0.25):
        self.clear_walls()
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) != self.start and (r, c) != self.goal:
                    if random.random() < density:
                        self.grid[r][c] = 1

    def get_neighbors(self, node):
        r, c = node
        neighbors = []
        cardinals = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in cardinals:
            nr, nc = r + dr, c + dc
            if self.is_valid(nr, nc):
                neighbors.append(((nr, nc), 1.0))

        # Diagonal transitions (strictly preventing corner-cutting)
        diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in diagonals:
            nr, nc = r + dr, c + dc
            if self.is_valid(nr, nc):
                if self.is_valid(r + dr, c) and self.is_valid(r, c + dc):
                    neighbors.append(((nr, nc), SQRT2))
        return neighbors

def heuristic_octile(a, b):
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return (dr + dc) + (SQRT2 - 2.0) * min(dr, dc)

def draw_interface(screen, world, path, closed_set, open_set, telemetry, drone_pos, closest_pos):
    screen.fill(COLOR_BG)
    
    # 1. Dashboard
    pygame.draw.rect(screen, COLOR_PANEL, (0, 0, SCREEN_WIDTH, PANEL_HEIGHT))
    
    t1 = HEADER_FONT.render(
        f"Algo: {telemetry['mode']}  |  [SPACE: Run] [M: Toggle Algo] [C: Clear] [R: Random Maze] [ESC: Exit]",
        True, COLOR_TEXT
    )
    t2 = FONT.render(
        "Interactive: [Drag Left: Wall] [Drag Right: Erase] [Hold 'S': Move Start] [Hold 'E': Move Goal]",
        True, (173, 181, 189)
    )

    if "BLOCKED" in telemetry['status']:
        status_color = (255, 133, 27)
    elif "FOUND" in telemetry['status'] or "FLYING" in telemetry['status']:
        status_color = (40, 167, 69)
    else:
        status_color = (255, 193, 7)

    dist_str = f" | Remaining Dist: {telemetry['dist_to_goal']:.2f}" if telemetry.get('dist_to_goal', 0.0) > 0 else ""
    t3 = FONT.render(
        f"Status: {telemetry['status']} | Explored: {len(closed_set)} | Cost: {telemetry['cost']:.2f} | Steps: {len(path)}{dist_str}",
        True, status_color
    )
    
    screen.blit(t1, (16, 10))
    screen.blit(t2, (16, 36))
    screen.blit(t3, (16, 62))

    # 2. Grid Elements
    for r in range(world.rows):
        for c in range(world.cols):
            rect = (c * CELL_SIZE, (r * CELL_SIZE) + PANEL_HEIGHT, CELL_SIZE - 1, CELL_SIZE - 1)
            pos = (r, c)
            
            if world.grid[r][c] == 1:
                pygame.draw.rect(screen, COLOR_OBSTACLE, rect)
            elif pos == world.start:
                pygame.draw.rect(screen, COLOR_START, rect)
            elif pos == world.goal:
                pygame.draw.rect(screen, COLOR_GOAL, rect)
            elif pos == closest_pos:
                pygame.draw.rect(screen, COLOR_CLOSEST, rect)
            elif pos in path:
                pygame.draw.rect(screen, COLOR_PATH, rect)
            elif pos in closed_set:
                pygame.draw.rect(screen, COLOR_CLOSED, rect)
            elif pos in open_set:
                pygame.draw.rect(screen, COLOR_OPEN, rect)
            else:
                pygame.draw.rect(screen, COLOR_EMPTY, rect)

    # 3. Drone / Pointer Indicator
    if drone_pos is not None:
        r, c = drone_pos
        cx = int(c * CELL_SIZE + CELL_SIZE / 2)
        cy = int(r * CELL_SIZE + PANEL_HEIGHT + CELL_SIZE / 2)
        radius = max(4, CELL_SIZE // 3)
        pygame.draw.circle(screen, COLOR_DRONE, (cx, cy), radius)
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy), radius, 2)

    pygame.display.flip()

def main():
    world = GridWorld(GRID_ROWS, GRID_COLS)
    use_a_star = True

    path = []
    closed_set = set()
    open_set_tracker = set()
    closest_node = None
    telemetry = {
        "mode": "A* (Octile Heuristic)",
        "nodes": 0,
        "cost": 0.0,
        "dist_to_goal": 0.0,
        "status": "READY"
    }

    open_heap = []
    came_from = {}
    cost_so_far = {}
    tie_breaker = 0
    state = "IDLE"  # "IDLE", "SEARCHING", "FOLLOWING_PATH"

    drone_pos = world.start
    final_path_idx = 0

    drawing_wall = False
    erasing_wall = False

    while True:
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                elif event.key == pygame.K_SPACE and state == "IDLE":
                    open_heap = []
                    came_from = {}
                    cost_so_far = {world.start: 0.0}
                    closed_set = set()
                    open_set_tracker = {world.start}
                    path = []
                    closest_node = world.start
                    drone_pos = world.start
                    tie_breaker = 0

                    h_start = heuristic_octile(world.start, world.goal) * 1.001 if use_a_star else 0.0
                    heapq.heappush(open_heap, (h_start, h_start, tie_breaker, world.start))
                    
                    state = "SEARCHING"
                    telemetry["status"] = "SEARCHING..."
                    telemetry["dist_to_goal"] = 0.0

                elif event.key == pygame.K_m and state == "IDLE":
                    use_a_star = not use_a_star
                    telemetry["mode"] = "A* (Octile Heuristic)" if use_a_star else "Dijkstra Search"

                elif event.key == pygame.K_c and state == "IDLE":
                    world.clear_walls()
                    path, closed_set, open_set_tracker = [], set(), set()
                    closest_node = None
                    telemetry.update({"cost": 0.0, "dist_to_goal": 0.0, "status": "CLEARED"})
                    drone_pos = world.start

                elif event.key == pygame.K_r and state == "IDLE":
                    world.generate_random_walls(density=0.25)
                    path, closed_set, open_set_tracker = [], set(), set()
                    closest_node = None
                    telemetry.update({"cost": 0.0, "dist_to_goal": 0.0, "status": "READY"})
                    drone_pos = world.start

            # Mouse Interactions
            elif event.type == pygame.MOUSEBUTTONDOWN and state == "IDLE":
                mx, my = pygame.mouse.get_pos()
                if my >= PANEL_HEIGHT:
                    r = (my - PANEL_HEIGHT) // CELL_SIZE
                    c = mx // CELL_SIZE
                    if 0 <= r < world.rows and 0 <= c < world.cols:
                        if keys[pygame.K_s]:
                            if (r, c) != world.goal:
                                world.grid[r][c] = 0
                                world.start = (r, c)
                                drone_pos = (r, c)
                        elif keys[pygame.K_e]:
                            if (r, c) != world.start:
                                world.grid[r][c] = 0
                                world.goal = (r, c)
                        elif event.button == 1:
                            if (r, c) != world.start and (r, c) != world.goal:
                                drawing_wall = True
                                world.grid[r][c] = 1
                        elif event.button == 3:
                            erasing_wall = True
                            world.grid[r][c] = 0

            elif event.type == pygame.MOUSEBUTTONUP:
                drawing_wall = False
                erasing_wall = False

            elif event.type == pygame.MOUSEMOTION and state == "IDLE":
                mx, my = pygame.mouse.get_pos()
                if my >= PANEL_HEIGHT:
                    r = (my - PANEL_HEIGHT) // CELL_SIZE
                    c = mx // CELL_SIZE
                    if 0 <= r < world.rows and 0 <= c < world.cols:
                        if keys[pygame.K_s] and pygame.mouse.get_pressed()[0]:
                            if (r, c) != world.goal:
                                world.grid[r][c] = 0
                                world.start = (r, c)
                                drone_pos = (r, c)
                        elif keys[pygame.K_e] and pygame.mouse.get_pressed()[0]:
                            if (r, c) != world.start:
                                world.grid[r][c] = 0
                                world.goal = (r, c)
                        elif drawing_wall:
                            if (r, c) != world.start and (r, c) != world.goal:
                                world.grid[r][c] = 1
                        elif erasing_wall:
                            world.grid[r][c] = 0

        # State Machine Execution
        if state == "SEARCHING":
            steps_per_frame = 2
            found_goal = False

            for _ in range(steps_per_frame):
                if not open_heap:
                    break

                _, _, _, current = heapq.heappop(open_heap)
                if current in closed_set:
                    continue

                closed_set.add(current)
                open_set_tracker.discard(current)
                drone_pos = current

                # Track the node that got closest to the goal
                if closest_node is None or heuristic_octile(current, world.goal) < heuristic_octile(closest_node, world.goal):
                    closest_node = current

                if current == world.goal:
                    found_goal = True
                    break

                for neighbor, move_cost in world.get_neighbors(current):
                    if neighbor in closed_set:
                        continue

                    new_cost = cost_so_far[current] + move_cost
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        came_from[neighbor] = current

                        h = heuristic_octile(neighbor, world.goal) * 1.001 if use_a_star else 0.0
                        f = new_cost + h
                        tie_breaker += 1
                        heapq.heappush(open_heap, (f, h, tie_breaker, neighbor))
                        open_set_tracker.add(neighbor)

            if found_goal:
                # Goal reached: reconstruct full shortest path
                curr = world.goal
                reconstructed = []
                while curr in came_from:
                    reconstructed.append(curr)
                    curr = came_from[curr]
                reconstructed.append(world.start)
                reconstructed.reverse()

                path = reconstructed
                telemetry["cost"] = cost_so_far.get(world.goal, 0.0)
                telemetry["dist_to_goal"] = 0.0
                telemetry["status"] = "FLYING TO GOAL"
                final_path_idx = 0
                state = "FOLLOWING_PATH"

            elif not open_heap:
                # Goal unreachable: reconstruct path to closest reachable node
                target = closest_node if closest_node is not None else world.start
                curr = target
                reconstructed = []
                while curr in came_from:
                    reconstructed.append(curr)
                    curr = came_from[curr]
                if world.start not in reconstructed:
                    reconstructed.append(world.start)
                reconstructed.reverse()

                path = reconstructed
                rem_dist = heuristic_octile(target, world.goal)
                telemetry["cost"] = cost_so_far.get(target, 0.0)
                telemetry["dist_to_goal"] = rem_dist
                telemetry["status"] = "GOAL BLOCKED - FLYING TO CLOSEST POINT"
                final_path_idx = 0
                state = "FOLLOWING_PATH"

        elif state == "FOLLOWING_PATH":
            if final_path_idx < len(path):
                drone_pos = path[final_path_idx]
                final_path_idx += 1
            else:
                if closest_node == world.goal or (path and path[-1] == world.goal):
                    telemetry["status"] = "REACHED GOAL"
                else:
                    telemetry["status"] = "BLOCKED: STOPPED AT CLOSEST REACHABLE STEP"
                state = "IDLE"

        draw_interface(SCREEN, world, path, closed_set, open_set_tracker, telemetry, drone_pos, closest_node)
        CLOCK.tick(45)

if __name__ == "__main__":
    main()