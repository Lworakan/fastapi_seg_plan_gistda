import math
import heapq

class BestFirstSearchPlanner:
    def __init__(self, grid):
        self.grid = grid
        self.x_width = grid.shape[1]
        self.y_width = grid.shape[0]
        self.obstacle_map = self.create_obstacle_map()
        self.motion = self.get_motion_model()
  
    class Node:
        def __init__(self, x, y, h_cost, parent=None):
            self.x = x
            self.y = y
            self.h_cost = h_cost  # Cost-to-go (heuristic)
            self.parent = parent

        def __lt__(self, other):
            return self.h_cost < other.h_cost  # Sort by heuristic only

    def heuristic(self, x, y, gx, gy):
        # Euclidean Distance heuristic function
        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)

    def planning(self, sx, sy, gx, gy):
        start_node = self.Node(sx, sy, self.heuristic(sx, sy, gx, gy))
        goal_node = self.Node(gx, gy, 0)

        open_list = []
        heapq.heappush(open_list, start_node)  # Priority queue sorted by cost
        visited = {}  # Dictionary to store best heuristic found
        visited[(sx, sy)] = start_node.h_cost

        while open_list:
            current = heapq.heappop(open_list)  # Get lowest-cost node

            if (current.x, current.y) == (gx, gy):
                path_x, path_y = self.calc_final_path(current)
                return path_x, path_y

            for motion in self.motion:
                node_x = current.x + motion[0]
                node_y = current.y + motion[1]
                node_h = self.heuristic(node_x, node_y, gx, gy)

                node = self.Node(node_x, node_y, node_h, current)

                if not self.is_valid(node):
                    continue

                if (node_x, node_y) in visited and visited[(node_x, node_y)] <= node_h:
                    continue
                
                visited[(node_x, node_y)] = node_h
                heapq.heappush(open_list, node)

        print("Path Not Found")
        return [], []
    
    def planning_multi_goal(self, start, goals):
        sx, sy = start
        full_rx, full_ry = [], []
        total_distance = 0.0

        for gx, gy in goals:
            rx, ry = self.planning(sx, sy, gx, gy) 
            
            if not rx: 
                print(f"Path Not Found")
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
        node = goal_node  # Start from the goal

        while node is not None:
            rx.append(node.x)
            ry.append(node.y)
            node = node.parent

        rx.reverse()
        ry.reverse()
        return rx, ry

    def is_valid(self, node):
        if node.x < 0 or node.y < 0 or node.x >= self.x_width or node.y >= self.y_width:
            return False
        return not self.obstacle_map[node.y, node.x]

    def create_obstacle_map(self):
        return self.grid == 0  # Assuming obstacles are nonzero values

    @staticmethod
    def get_motion_model():
        step = 1
        return [[step, 0, step], 
                [0, step, step], 
                [-step, 0, step], 
                [0, -step, step],
                [-step, -step, math.sqrt(2) * step], 
                [-step, step, math.sqrt(2) * step], 
                [step, -step, math.sqrt(2) * step], 
                [step, step, math.sqrt(2) * step]]
    
    def calculate_path_distance(self, rx, ry):
        return sum(math.hypot(rx[i] - rx[i - 1], ry[i] - ry[i - 1]) for i in range(1, len(rx)))