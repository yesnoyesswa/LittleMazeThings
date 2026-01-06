import numpy as np
import heapq
import matplotlib.pyplot as plt
from mazegenfromc import generate_maze

def get_manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def update_knowledge(current_pos, maze_np, knowledge_map, vision_range=1):
    size = maze_np.shape[0]
    y, x = current_pos
    for i in range(y - vision_range, y + vision_range + 1):
        for j in range(x - vision_range, x + vision_range + 1):
            if 0 <= i < size and 0 <= j < size:
                knowledge_map[i, j] = maze_np[i, j]

def a_star_algorithm(start, goal, knowledge_map):
    size = knowledge_map.shape[0]
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: get_manhattan_dist(start, goal)}
    while open_set:
        current = heapq.heappop(open_set)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dy, current[1] + dx)
            if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                if knowledge_map[neighbor[0], neighbor[1]] == 1: continue
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + get_manhattan_dist(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None

def animate_astar_solve(maze_np, vision_limit=1, sleep_time=0.01):
    size = maze_np.shape[0]
    knowledge_map = np.full((size, size), -1.0) 
    start_pos = tuple(np.argwhere(maze_np == 2)[0])
    goal_pos = tuple(np.argwhere(maze_np == 9)[0])
    curr_pos = start_pos
    
    path_history = [] # Lưu vết chân đã đi
    success = False
    final_steps = 0

    plt.ion() 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    for step in range(2000):
        path_history.append(curr_pos)
        if curr_pos == goal_pos:
            success = True
            final_steps = step
            break

        update_knowledge(curr_pos, maze_np, knowledge_map, vision_range=vision_limit)
        path = a_star_algorithm(curr_pos, goal_pos, knowledge_map)
        
        if not path: break

        ax1.clear()
        ax2.clear()
        
        # Thực tế
        ax1.imshow(maze_np, cmap='nipy_spectral')
        ax1.plot(curr_pos[1], curr_pos[0], 'ro', markersize=6)
        ax1.set_title(f"Thực tế (Bước {step})")
        
        # Trong bộ nhớ + tính toán đường tối ưu
        ax2.imshow(knowledge_map, cmap='nipy_spectral', vmin=-1, vmax=9)
        
        # Đường kế hoạch màu đỏ
        if path:
            py, px = zip(*path)
            ax2.plot(px, py, 'r--', linewidth=2)
            
        ax2.set_title(f"Bộ nhớ (Tầm nhìn {vision_limit*2+1}x{vision_limit*2+1})")
        
        plt.pause(sleep_time)
        curr_pos = path[0]

    plt.ioff()
    
    print(f"KẾT QUẢ A*")
    print(f"Kích thước mê cung: {size}x{size}")
    print(f"Tầm nhìn Agent   : {vision_limit*2+1}x{vision_limit*2+1}")
    print(f"Trạng thái       : {'THÀNH CÔNG ' if success else 'THẤT BẠI '}")
    print(f"Tổng số bước đi  : {final_steps}")
    
    plt.show()

if __name__ == "__main__":
    maze = generate_maze(23)
    animate_astar_solve(maze, vision_limit=1)