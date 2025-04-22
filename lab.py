
# 12. Write a program to implement Traveling salesman problem
# 13. Implementing a program to apply Deduction Theorem on Logic Expressions
# 14. Write a Program to Implement 8-Queens Problem





# Depth First Search (DFS) implementation using a loop

def dfs(graph, start):
    """
    Perform DFS on a graph from the start node using a loop.
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    visited = set()
    stack = [start]
    result = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            # Add neighbors in reverse order for correct traversal order
            for neighbor in reversed(graph.get(node, [])):
                if neighbor not in visited:
                    stack.append(neighbor)
    return result

# Example usage:
if __name__ == "__main__":
    # Example graph as adjacency list
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    print("DFS Traversal:", dfs(graph, 'A'))



# Breadth First Search (BFS) implementation using a loop

from collections import deque

def bfs(graph, start):
    """
    Perform BFS on a graph from the start node using a queue.
    Time Complexity: O(V + E)
    Space Complexity: O(V)
    """
    visited = set()
    queue = deque([start])
    result = []

    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            result.append(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
    return result

# Example usage:
if __name__ == "__main__":
    # Example graph as adjacency list
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    print("BFS Traversal:", bfs(graph, 'A'))






# Depth Limited Search (DLS) implementation using a loop

def dls(graph, start, limit):
    """
    Perform Depth Limited Search (DLS) on a graph from the start node up to a given depth limit.
    Time Complexity: O(V + E) (in the worst case, similar to DFS)
    Space Complexity: O(V)
    :param graph: dict, adjacency list of the graph
    :param start: starting node
    :param limit: maximum depth to search
    :return: list of nodes visited within the depth limit
    """
    visited = set()
    stack = [(start, 0)]  # (node, current_depth)
    result = []

    while stack:
        node, depth = stack.pop()
        if node not in visited and depth <= limit:
            visited.add(node)
            result.append(node)
            if depth < limit:
                for neighbor in reversed(graph.get(node, [])):
                    if neighbor not in visited:
                        stack.append((neighbor, depth + 1))
    return result

# Example usage:
if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    print("DLS Traversal (limit=2):", dls(graph, 'A', 2))





# Iterative Deepening Depth First Search (IDDFS) implementation

def iddfs(graph, start, max_depth):
    """
    Perform Iterative Deepening Depth First Search (IDDFS) on a graph.
    Time Complexity: O(b^d), where b is branching factor and d is depth
    Space Complexity: O(d), where d is the depth limit
    :param graph: dict, adjacency list of the graph
    :param start: starting node
    :param max_depth: maximum depth to search
    :return: list of nodes visited in order for the last depth
    """
    def dls(current, depth, visited, result):
        if depth == 0:
            if current not in visited:
                visited.add(current)
                result.append(current)
            return
        if current not in visited:
            visited.add(current)
            result.append(current)
            for neighbor in graph.get(current, []):
                dls(neighbor, depth - 1, visited, result)

    for depth in range(max_depth + 1):
        visited = set()
        result = []
        dls(start, depth, visited, result)
        print(f"IDDFS Traversal (depth={depth}):", result)
    return result

# Example usage:
if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': ['F'],
        'F': []
    }
    iddfs(graph, 'A', 3)




# 6. Uniform search

import heapq

def uniform_cost_search(graph, start, goal):
    """
    Uniform Cost Search (UCS) finds the least-cost path from start to goal.
    Time Complexity: O((V + E) * log V)
    Space Complexity: O(V)
    :param graph: dict, adjacency list with (neighbor, cost) pairs
    :param start: starting node
    :param goal: goal node
    :return: (total_cost, path) if found, else (None, [])
    """
    queue = [(0, start, [start])]  # (cost, node, path)
    visited = set()

    while queue:
        cost, node, path = heapq.heappop(queue)
        if node == goal:
            return cost, path
        if node not in visited:
            visited.add(node)
            for neighbor, edge_cost in graph.get(node, []):
                if neighbor not in visited:
                    heapq.heappush(queue, (cost + edge_cost, neighbor, path + [neighbor]))
    return None, []

# Example usage:
if __name__ == "__main__":
    # Graph as adjacency list with costs
    graph = {
        'A': [('B', 1), ('C', 5)],
        'B': [('D', 3), ('E', 1)],
        'C': [('F', 2)],
        'D': [],
        'E': [('F', 1)],
        'F': []
    }
    total_cost, path = uniform_cost_search(graph, 'A', 'F')
    print("Uniform Cost Search: Cost =", total_cost, ", Path =", path)



# 10. Best first search

import heapq

def best_first_search(graph, start, goal, heuristic):
    """
    Best First Search (Greedy) algorithm.
    Time Complexity: O((V + E) * log V)
    Space Complexity: O(V)
    :param graph: dict, adjacency list with (neighbor, cost) pairs
    :param start: starting node
    :param goal: goal node
    :param heuristic: function(node) -> estimated cost to goal
    :return: path from start to goal if found, else []
    """
    queue = [(heuristic(start), start, [start])]  # (heuristic, node, path)
    visited = set()

    while queue:
        h, node, path = heapq.heappop(queue)
        if node == goal:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor, _ in graph.get(node, []):
                if neighbor not in visited:
                    heapq.heappush(queue, (heuristic(neighbor), neighbor, path + [neighbor]))
    return []

# Example usage:
if __name__ == "__main__":
    # Example graph with costs (costs are ignored in Best First Search)
    graph = {
        'A': [('B', 1), ('C', 5)],
        'B': [('D', 3), ('E', 1)],
        'C': [('F', 2)],
        'D': [],
        'E': [('F', 1)],
        'F': []
    }
    # Example heuristic (straight-line distance or any estimate)
    heuristic = lambda node: {'A': 5, 'B': 3, 'C': 2, 'D': 6, 'E': 1, 'F': 0}[node]
    path = best_first_search(graph, 'A', 'F', heuristic)
    print("Best First Search Path:", path)



# 9. A* search

import heapq

def a_star_search(graph, start, goal, heuristic):
    """
    Simple A* Search algorithm.
    Time Complexity: O((V + E) * log V)
    Space Complexity: O(V)
    :param graph: dict, adjacency list with (neighbor, cost) pairs
    :param start: starting node
    :param goal: goal node
    :param heuristic: function(node) -> estimated cost to goal
    :return: (total_cost, path) if found, else (None, [])
    """
    queue = [(heuristic(start), 0, start, [start])]  # (f, g, node, path)
    visited = set()

    while queue:
        f, g, node, path = heapq.heappop(queue)
        if node == goal:
            return g, path
        if node not in visited:
            visited.add(node)
            for neighbor, cost in graph.get(node, []):
                if neighbor not in visited:
                    new_g = g + cost
                    new_f = new_g + heuristic(neighbor)
                    heapq.heappush(queue, (new_f, new_g, neighbor, path + [neighbor]))
    return None, []

# Example usage:
if __name__ == "__main__":
    graph = {
        'A': [('B', 1), ('C', 5)],
        'B': [('D', 3), ('E', 1)],
        'C': [('F', 2)],
        'D': [],
        'E': [('F', 1)],
        'F': []
    }
    heuristic = lambda node: {'A': 5, 'B': 3, 'C': 2, 'D': 6, 'E': 1, 'F': 0}[node]
    cost, path = a_star_search(graph, 'A', 'F', heuristic)
    print("A* Search: Cost =", cost, ", Path =", path)


# 7. Bidirectional Search

from collections import deque

def bidirectional_search(graph, start, goal):
    """
    Simple Bidirectional Search for unweighted graphs.
    Time Complexity: O(b^(d/2)), where b is branching factor, d is shortest path length
    Space Complexity: O(b^(d/2))
    :param graph: dict, adjacency list
    :param start: starting node
    :param goal: goal node
    :return: path from start to goal if found, else []
    """
    if start == goal:
        return [start]

    # Queues for forward and backward search
    queue_start = deque([[start]])
    queue_goal = deque([[goal]])

    # Visited sets and parent maps
    visited_start = {start: [start]}
    visited_goal = {goal: [goal]}

    while queue_start and queue_goal:
        # Expand forward
        path_start = queue_start.popleft()
        last_start = path_start[-1]
        for neighbor in graph.get(last_start, []):
            if neighbor not in visited_start:
                new_path = path_start + [neighbor]
                visited_start[neighbor] = new_path
                queue_start.append(new_path)
                if neighbor in visited_goal:
                    # Path found
                    return new_path + visited_goal[neighbor][-2::-1]

        # Expand backward
        path_goal = queue_goal.popleft()
        last_goal = path_goal[-1]
        for neighbor in graph.get(last_goal, []):
            if neighbor not in visited_goal:
                new_path = path_goal + [neighbor]
                visited_goal[neighbor] = new_path
                queue_goal.append(new_path)
                if neighbor in visited_start:
                    # Path found
                    return visited_start[neighbor] + new_path[-2::-1]
    return []

# Example usage:
if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    path = bidirectional_search(graph, 'A', 'F')
    print("Bidirectional Search Path:", path)



# 11. Hill Climbing Algorithm

def hill_climbing(start, get_neighbors, heuristic):
    """
    Simple Hill Climbing algorithm.
    Time Complexity: Depends on the number of steps and neighbors (often O(b^d))
    Space Complexity: O(1) if path not stored, O(d) if path is stored
    :param start: starting state
    :param get_neighbors: function(state) -> list of neighbor states
    :param heuristic: function(state) -> value (lower is better)
    :return: local optimum state and path
    """
    current = start
    path = [current]
    while True:
        neighbors = get_neighbors(current)
        if not neighbors:
            break
        next_state = min(neighbors, key=heuristic)
        if heuristic(next_state) >= heuristic(current):
            break
        current = next_state
        path.append(current)
    return current, path

# Example usage:
if __name__ == "__main__":
    # Example: Find minimum of a simple function f(x) = (x-3)^2
    def get_neighbors(x):
        return [x - 1, x + 1]

    def heuristic(x):
        return (x - 3) ** 2

    start = 10
    optimum, path = hill_climbing(start, get_neighbors, heuristic)
    print("Hill Climbing Optimum:", optimum)
    print("Path:", path)


# 1. 8 Puzzle Problem (Easy A* Search)

import heapq

def manhattan(state, goal):
    """Sum of Manhattan distances of tiles from their goal positions."""
    distance = 0
    for num in range(1, 9):
        for i in range(3):
            for j in range(3):
                if state[i][j] == num:
                    x1, y1 = i, j
                if goal[i][j] == num:
                    x2, y2 = i, j
        distance += abs(x1 - x2) + abs(y1 - y2)
    return distance

def get_neighbors(state):
    """Generate all possible moves from current state."""
    neighbors = []
    for i in range(3):
        for j in range(3):
            if state[i][j] == 0:
                x, y = i, j
    moves = [(-1,0), (1,0), (0,-1), (0,1)]  # Up, Down, Left, Right
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new_state = [list(row) for row in state]
            new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
            neighbors.append(tuple(tuple(row) for row in new_state))
    return neighbors

def a_star_8_puzzle(start, goal):
    """A* search for 8 puzzle (easy version)."""
    queue = [(manhattan(start, goal), 0, start, [])]  # (f, g, state, path)
    visited = set()
    while queue:
        f, g, state, path = heapq.heappop(queue)
        if state == goal:
            return path + [state]
        if state in visited:
            continue
        visited.add(state)
        for neighbor in get_neighbors(state):
            if neighbor not in visited:
                heapq.heappush(queue, (g + 1 + manhattan(neighbor, goal), g + 1, neighbor, path + [state]))
    return None

# Example usage:
if __name__ == "__main__":
    start = (
        (1, 2, 3),
        (4, 0, 6),
        (7, 5, 8)
    )
    goal = (
        (1, 2, 3),
        (4, 5, 6),
        (7, 8, 0)
    )
    solution = a_star_8_puzzle(start, goal)
    if solution:
        print("8 Puzzle Solution Steps:")
        for step in solution:
            for row in step:
                print(row)
            print("---")
    else:
        print("No solution found.")


# Robot Vacuum Cleaner Path Planning (Simple Grid BFS)

from collections import deque

def robot_vacuum_cleaner(grid, start):
    """
    Simple BFS to clean all reachable cells in a grid.
    :param grid: 2D list, 0 = empty, 1 = obstacle
    :param start: (x, y) starting position
    :return: list of cells visited in order
    """
    rows, cols = len(grid), len(grid[0])
    visited = set()
    queue = deque([start])
    path = []

    while queue:
        x, y = queue.popleft()
        if (x, y) in visited or grid[x][y] == 1:
            continue
        visited.add((x, y))
        path.append((x, y))
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                queue.append((nx, ny))
    return path

# Example usage:
if __name__ == "__main__":
    grid = [
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    start = (0, 0)
    cleaned = robot_vacuum_cleaner(grid, start)
    print("Cells cleaned in order:", cleaned)




# Traveling Salesman Problem (TSP) - Brute Force (Easy Version)

import itertools

def tsp_brute_force(graph, start):
    """
    Solves the Traveling Salesman Problem using brute force.
    Time Complexity: O(n!), where n is the number of cities.
    Space Complexity: O(n)
    :param graph: dict, adjacency matrix as dict of dicts {city1: {city2: cost, ...}, ...}
    :param start: starting city
    :return: (min_cost, best_path)
    """
    cities = list(graph.keys())
    cities.remove(start)
    min_cost = float('inf')
    best_path = []

    for perm in itertools.permutations(cities):
        path = [start] + list(perm) + [start]
        cost = 0
        for i in range(len(path) - 1):
            cost += graph[path[i]][path[i+1]]
        if cost < min_cost:
            min_cost = cost
            best_path = path
    return min_cost, best_path

# Example usage:
if __name__ == "__main__":
    graph = {
        'A': {'A': 0, 'B': 10, 'C': 15, 'D': 20},
        'B': {'A': 10, 'B': 0, 'C': 35, 'D': 25},
        'C': {'A': 15, 'B': 35, 'C': 0, 'D': 30},
        'D': {'A': 20, 'B': 25, 'C': 30, 'D': 0}
    }
    min_cost, best_path = tsp_brute_force(graph, 'A')
    print("TSP Minimum Cost:", min_cost)
    print("TSP Path:", best_path)



# 8-Queens Problem (Backtracking Solution)

def is_safe(board, row, col, n):
    # Check column
    for i in range(row):
        if board[i] == col:
            return False
        # Check diagonals
        if abs(board[i] - col) == abs(i - row):
            return False
    return True

def solve_n_queens_util(n, row, board, solutions):
    if row == n:
        solutions.append(board[:])
        return
    for col in range(n):
        if is_safe(board, row, col, n):
            board[row] = col
            solve_n_queens_util(n, row + 1, board, solutions)

def solve_n_queens(n=8):
    """
    Solves the N-Queens problem and returns all solutions.
    Time Complexity: O(N!), Space Complexity: O(N^2) for storing solutions.
    :param n: Number of queens (default 8)
    :return: List of solutions, each as a list of column positions.
    """
    solutions = []
    board = [-1] * n
    solve_n_queens_util(n, 0, board, solutions)
    return solutions

def print_board(solution):
    n = len(solution)
    for row in solution:
        print(" ".join("Q" if i == row else "." for i in range(n)))
    print()

# Example usage:
if __name__ == "__main__":
    solutions = solve_n_queens(8)
    print(f"Total solutions for 8-Queens: {len(solutions)}")
    print("First solution:")
    print_board(solutions[0])



# Deduction Theorem on Logic Expressions (Easy Version)

def implies(p, q):
    # Logical implication: p → q is True unless p is True and q is False
    return (not p) or q

def deduction_theorem(premise, assumption, conclusion):
    """
    Checks if (premise and assumption) => conclusion is equivalent to premise => (assumption -> conclusion)
    for all possible truth values.
    :param premise: function(p, q) -> bool
    :param assumption: function(p, q) -> bool
    :param conclusion: function(p, q) -> bool
    :return: True if deduction theorem holds, False otherwise
    """
    for p in [False, True]:
        for q in [False, True]:
            # (premise and assumption) => conclusion
            left = implies(premise(p, q) and assumption(p, q), conclusion(p, q))
            # premise => (assumption -> conclusion)
            right = implies(premise(p, q), implies(assumption(p, q), conclusion(p, q)))
            if left != right:
                return False
    return True

# Example usage:
if __name__ == "__main__":
    # Example: premise = p, assumption = q, conclusion = r
    # We'll use p and q as variables, and conclusion as q
    premise = lambda p, q: p
    assumption = lambda p, q: q
    conclusion = lambda p, q: q
    result = deduction_theorem(premise, assumption, conclusion)
    print("Deduction Theorem holds:" if result else "Deduction Theorem does not hold.")