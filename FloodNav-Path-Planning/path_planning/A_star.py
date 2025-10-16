import math
import heapq

class AStarPlanner:
    def __init__(self, grid):
        self.grid = grid
        self.x_width = grid.shape[1]
        self.y_width = grid.shape[0]
        self.obstacle_map = self.create_obstacle_map()
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, heuristic, parent=None):
            self.x = x
            self.y = y
            self.cost = cost
            self.heuristic = heuristic
            self.total_cost = cost + heuristic
            self.parent = parent

        def __lt__(self, other):
            return self.total_cost < other.total_cost

    def planning(self, sx, sy, gx, gy):
        start_node = self.Node(sx, sy, 0.0, self.calc_heuristic(sx, sy, gx, gy))
        goal_node = self.Node(gx, gy, 0.0, 0.0)

        open_set = []
        heapq.heappush(open_set, start_node)
        visited = set()

        while open_set:
            current = heapq.heappop(open_set)
            c_id = (current.x, current.y)

            if c_id in visited:
                continue
            visited.add(c_id)

            if c_id == (gx, gy):
                return self.calc_final_path(current)

            for motion in self.motion:
                node_x = current.x + motion[0]
                node_y = current.y + motion[1]
                new_cost = current.cost + motion[2]
                heuristic = self.calc_heuristic(node_x, node_y, gx, gy)
                node = self.Node(node_x, node_y, new_cost, heuristic, current)
                n_id = (node.x, node.y)

                if not self.is_valid(node) or n_id in visited:
                    continue

                heapq.heappush(open_set, node)

        print("Path Not Found")
        return [], []
    
    def planning_multi_goal(self, start, goals):
        sx, sy = start
        full_rx, full_ry = [], []
        total_distance = 0.0

        for gx, gy in goals:
            rx, ry = self.planning(sx, sy, gx, gy)

            if not rx:
                print(f"Path from ({sx}, {sy}) to ({gx}, {gy}) not found!")
                return [], [], 0.0

            if full_rx:
                rx, ry = rx[1:], ry[1:]

            full_rx.extend(rx)
            full_ry.extend(ry)
            total_distance += self.calculate_path_distance(rx, ry)

            sx, sy = gx, gy

        return full_rx, full_ry, total_distance

    def calc_final_path(self, goal_node):
        rx, ry = [], []
        node = goal_node  
        while node is not None:
            rx.append(node.x)
            ry.append(node.y)
            node = node.parent
        rx.reverse()
        ry.reverse()
        return rx, ry

    def calc_heuristic(self, x1, y1, x2, y2):
        return math.hypot(x1 - x2, y1 - y2)

    def is_valid(self, node):
        if node.x < 0 or node.y < 0 or node.x >= self.x_width or node.y >= self.y_width:
            return False
        return not self.obstacle_map[node.y, node.x]

    def create_obstacle_map(self):
        return self.grid == 0

    @staticmethod
    def get_motion_model():
        step = 1
        return [
            [step, 0, step], [0, step, step], [-step, 0, step], [0, -step, step],
            [-step, -step, math.sqrt(2) * step], [-step, step, math.sqrt(2) * step],
            [step, -step, math.sqrt(2) * step], [step, step, math.sqrt(2) * step]
        ]
    
    def calculate_path_distance(self, rx, ry):
        return sum(math.hypot(rx[i] - rx[i - 1], ry[i] - ry[i - 1]) for i in range(1, len(rx)))
