import matplotlib.pyplot as plt
import collections
import time

class ClassicSolver:
    def __init__(self, maze, start, end):
        self.maze = maze
        self.start = start
        self.end = end
        self.height, self.width = maze.shape

    def visualize(self, path, title="Maze Solver"):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.maze, cmap='binary')
        
        # Vẽ điểm đầu và cuối
        plt.plot(self.start[1], self.start[0], 'go', markersize=10, label='Start') # Xanh lá
        plt.plot(self.end[1], self.end[0], 'ro', markersize=10, label='End')     # Đỏ
        
        # Vẽ đường đi tìm được
        if path:
            y_coords = [p[0] for p in path]
            x_coords = [p[1] for p in path]
            plt.plot(x_coords, y_coords, 'b-', linewidth=2, label='Path')  

        plt.title(title)
        plt.legend()
        plt.show()

    def solve_bfs(self):
    
        queue = collections.deque([self.start])
        came_from = {self.start: None}
        visited = set([self.start])

        while queue:
            current = queue.popleft()

            if current == self.end:
                break

            r, c = current
            # Duyệt 4 hướng: Lên, Xuống, Trái, Phải
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.height and 0 <= nc < self.width and 
                    self.maze[nr, nc] == 0 and (nr, nc) not in visited):
                    queue.append((nr, nc))
                    visited.add((nr, nc))
                    came_from[(nr, nc)] = current
        
        return self.reconstruct_path(came_from)

    def solve_dfs(self):
        stack = [self.start]
        came_from = {self.start: None}
        visited = set([self.start])

        while stack:
            current = stack.pop()

            if current == self.end:
                break

            r, c = current
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.height and 0 <= nc < self.width and 
                    self.maze[nr, nc] == 0 and (nr, nc) not in visited):
                    stack.append((nr, nc))
                    visited.add((nr, nc))
                    came_from[(nr, nc)] = current
        
        return self.reconstruct_path(came_from)

    def reconstruct_path(self, came_from):
        if self.end not in came_from:
            print("Không tìm thấy đường đi!")
            return []
        
        path = []
        current = self.end
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path