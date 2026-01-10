import numpy as np
import heapq
import matplotlib.pyplot as plt
from mazegenfromc import generate_maze



def dfs_algorithm(start, goal, knowledge_map, global_visited):
    size = knowledge_map.shape[0]
    stack = [start]
    came_from = {start: None}
    
    while stack:
        current = stack.pop()
        if current == goal: break
        
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        
        neighbors.sort(key=lambda n: (current[0]+n[0], current[1]+n[1]) in global_visited, reverse=True)
        
        for dy, dx in neighbors:
            neighbor = (current[0] + dy, current[1] + dx)
            if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                if knowledge_map[neighbor] != -1 and neighbor not in came_from:
                    came_from[neighbor] = current
                    stack.append(neighbor)
                    
    return reconstruct_path(came_from, start, goal)

def dijkstra_algorithm(start, goal, knowledge_map, _):
    size = knowledge_map.shape[0]
    open_set = [(0, start)]
    came_from = {start: None}
    g_score = {start: 0}

    while open_set:
        curr_g, current = heapq.heappop(open_set)
        if current == goal: break

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dy, current[1] + dx)
            if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                if knowledge_map[neighbor] == -1: continue
                
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (tentative_g, neighbor))
    return reconstruct_path(came_from, start, goal)

def a_star_algorithm(start, goal, knowledge_map, _):
    size = knowledge_map.shape[0]
    open_set = [(0, start)] # (f_score, position)
    came_from = {start: None}
    g_score = {start: 0}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal: break

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dy, current[1] + dx)
            if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                if knowledge_map[neighbor] == -1: continue
                
                tentative_g = g_score[current] + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    
                    h = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    f = tentative_g + h
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f, neighbor))
    return reconstruct_path(came_from, start, goal)

def reconstruct_path(came_from, start, goal):
    if goal not in came_from: return None
    path = []
    curr = goal
    while curr is not None:
        path.append(curr)
        curr = came_from[curr]
    return path[::-1][1:]



def animate_solve(maze_np, algo_name="DFS", vision_limit=1):
    size = maze_np.shape[0]
    knowledge_map = np.full((size, size), -5.0) 
    global_visited = set() 
    
    start_pos = tuple(np.argwhere(maze_np == 2)[0])
    goal_pos = tuple(np.argwhere(maze_np == 1)[0])
    curr_pos = start_pos

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    algorithms = {"DFS": dfs_algorithm, "Dijkstra": dijkstra_algorithm, "A*": a_star_algorithm}
    search_func = algorithms.get(algo_name, a_star_algorithm)

    step = 0
    current_plan = []

    while curr_pos != goal_pos and step < 2000:
        global_visited.add(curr_pos)
        
        y, x = curr_pos
        for i in range(y-vision_limit, y+vision_limit+1):
            for j in range(x-vision_limit, x+vision_limit+1):
                if 0 <= i < size and 0 <= j < size:
                    knowledge_map[i, j] = maze_np[i, j]

        if not current_plan or knowledge_map[current_plan[0]] == -1:
            current_plan = search_func(curr_pos, goal_pos, knowledge_map, global_visited)
        
        if not current_plan: break

    
        ax1.clear(); ax2.clear()
        ax1.imshow(maze_np, cmap='terrain')
        ax1.plot(curr_pos[1], curr_pos[0], 'ro', markersize=5)
        ax1.set_title(f"Thực tế - {algo_name}")
        
        ax2.imshow(knowledge_map, cmap='terrain', vmin=-5, vmax=2)
        if current_plan:
            py, px = zip(*current_plan)
            ax2.plot(px, py, 'r--', linewidth=1.5)
        ax2.set_title(f"Bộ nhớ Agent (Bước {step})")
        
        plt.pause(0.01)
        curr_pos = current_plan.pop(0)
        step += 1

    if curr_pos == goal_pos:
        y, x = curr_pos
        for i in range(y-vision_limit, y+vision_limit+1):
            for j in range(x-vision_limit, x+vision_limit+1):
                if 0 <= i < size and 0 <= j < size:
                    knowledge_map[i, j] = maze_np[i, j]

        ax1.clear(); ax2.clear()
        ax1.imshow(maze_np, cmap='terrain')

        ax1.plot(goal_pos[1], goal_pos[0], 'ro', markersize=5) 
        ax1.set_title(f"THÀNH CÔNG - {algo_name} (Tổng bước: {step})")
        
        ax2.imshow(knowledge_map, cmap='terrain', vmin=-5, vmax=2)
        ax2.set_title("Đã chạm đích!")
        
        plt.draw()
        plt.pause(1)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    maze = np.array(generate_maze(20))
    animate_solve(maze, algo_name="Dijsktra", vision_limit=1)