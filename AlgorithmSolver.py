import numpy as np
import heapq
import matplotlib.pyplot as plt
from mazegenfromc import generate_maze

def get_path_to_start(node, came_from):
    path = []
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    return path[::-1]

def get_physical_transition(curr_pos, next_node, came_from):
    path_to_curr = get_path_to_start(curr_pos, came_from)
    path_to_next = get_path_to_start(next_node, came_from)
    common_idx = 0
    for i in range(min(len(path_to_curr), len(path_to_next))):
        if path_to_curr[i] == path_to_next[i]: common_idx = i
        else: break
    transition = []
    for i in range(len(path_to_curr) - 1, common_idx - 1, -1): transition.append(path_to_curr[i])
    for i in range(common_idx + 1, len(path_to_next)): transition.append(path_to_next[i])
    return transition




def physical_dfs(start, goal, maze_map):
    size = maze_map.shape[0]
    stack, visited, history, path_stack = [start], {start}, [start], [start]
    while path_stack:
        curr = path_stack[-1]
        if curr == goal: break
        found = False
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nb = (curr[0] + dy, curr[1] + dx)
            if 0 <= nb[0] < size and 0 <= nb[1] < size and maze_map[nb] != 1 and nb not in visited:
                visited.add(nb); path_stack.append(nb); history.append(nb); found = True; break
        if not found:
            path_stack.pop()
            if path_stack: history.append(path_stack[-1])
    return history

def physical_dijkstra(start, goal, maze_map):
    size = maze_map.shape[0]
    open_set, came_from, g_score = [(0, start)], {start: None}, {start: 0}
    explored, history, curr_pos = set(), [], start
    while open_set:
        _, next_node = heapq.heappop(open_set)
        if next_node in explored: continue
        history.extend(get_physical_transition(curr_pos, next_node, came_from))
        curr_pos = next_node
        explored.add(next_node)
        if next_node == goal: break
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (next_node[0] + dy, next_node[1] + dx)
            if 0 <= nb[0] < size and 0 <= nb[1] < size and maze_map[nb] != 1 and nb not in explored:
                new_g = g_score[next_node] + 1
                if nb not in g_score or new_g < g_score[nb]:
                    g_score[nb] = new_g; came_from[nb] = next_node
                    heapq.heappush(open_set, (new_g, nb))
    return history

def a_star_plan(start, goal, knowledge):
    size = knowledge.shape[0]
    open_set, came_from, g_score = [(0, start)], {start: None}, {start: 0}
    while open_set:
        _, curr = heapq.heappop(open_set)
        if curr == goal:
            p = []
            while curr in came_from and curr is not None: p.append(curr); curr = came_from[curr]
            return p[::-1][1:]
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nb = (curr[0] + dy, curr[1] + dx)
            if 0 <= nb[0] < size and 0 <= nb[1] < size and knowledge[nb] != 1:
                new_g = g_score[curr] + 1
                if nb not in g_score or new_g < g_score[nb]:
                    g_score[nb] = new_g
                    h = abs(nb[0]-goal[0]) + abs(nb[1]-goal[1])
                    came_from[nb] = curr; heapq.heappush(open_set, (new_g + h, nb))
    return []


def animate_with_trail(maze_np, algo="A*", pause_time=0.01):
    size = maze_np.shape[0]
    
    start_pos = tuple(np.argwhere(maze_np == 2)[0])
    goal_pos = tuple(np.argwhere(maze_np == 9)[0])
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    trail_x, trail_y = [], []
    curr_pos = start_pos

    if algo in ["DFS", "Dijkstra"]:
        path = physical_dfs(start_pos, goal_pos, maze_np) if algo == "DFS" else physical_dijkstra(start_pos, goal_pos, maze_np)
        for i, pos in enumerate(path):
            trail_x.append(pos[1]); trail_y.append(pos[0])
            curr_pos = pos
            
            ax1.clear(); ax2.clear()
            ax1.imshow(maze_np, cmap='terrain', vmin=0, vmax=9)
            ax2.imshow(maze_np, cmap='terrain', vmin=0, vmax=9)
            ax1.plot(trail_x, trail_y, 'r-', alpha=0.3, linewidth=2)
            ax1.plot(pos[1], pos[0], 'ro', markersize=8)
            ax1.set_title(f"{algo} - Bước: {i}")
            plt.pause(pause_time)
    else:
        
        knowledge = np.full((size, size), -1.0) 
        step, plan = 0, []
        while curr_pos != goal_pos and step < 2000:
            trail_x.append(curr_pos[1]); trail_y.append(curr_pos[0])
            
            y, x = curr_pos
            for i in range(y-1, y+2):
                for j in range(x-1, x+2):
                    if 0 <= i < size and 0 <= j < size: knowledge[i, j] = maze_np[i, j]
            
            
            if not plan or knowledge[plan[0]] == 1: 
                plan = a_star_plan(curr_pos, goal_pos, knowledge)
            if not plan: break
            
            ax1.clear(); ax2.clear()
            ax1.imshow(maze_np, cmap='terrain', vmin=0, vmax=9)
            ax1.plot(trail_x, trail_y, 'r-', alpha=0.3)
            ax1.plot(curr_pos[1], curr_pos[0], 'ro', markersize=8)
            
            
            ax2.imshow(knowledge, cmap='terrain', vmin=-1, vmax=9)
            ax2.plot(trail_x, trail_y, 'y-', alpha=0.4)
            if plan: 
                py, px = zip(*plan); ax2.plot(px, py, 'c--')
            
            ax1.set_title(f"A* - Bước: {step}")
            plt.pause(pause_time)
            curr_pos = plan.pop(0); step += 1

    if curr_pos == goal_pos:
        trail_x.append(goal_pos[1]); trail_y.append(goal_pos[0])
        ax1.clear(); ax2.clear()
        
        ax1.imshow(maze_np, cmap='terrain', vmin=0, vmax=9)
        ax1.plot(trail_x, trail_y, 'r-', alpha=0.4, linewidth=2)
        ax1.plot(goal_pos[1], goal_pos[0], 'ro', markersize=5)
        ax1.set_title(f"THÀNH CÔNG! - {algo}")

        display_brain = maze_np if algo != "A*" else knowledge
        ax2.imshow(display_brain, cmap='terrain', vmin=-1, vmax=9)
        ax2.plot(trail_x, trail_y, 'y-', alpha=0.5, linewidth=2)
        ax2.set_title("Đã chạm mục tiêu")
        plt.draw()
        plt.pause(1.5)

    plt.ioff(); plt.show()

if __name__ == "__main__":
    maze = np.array(generate_maze(5)) 
    animate_with_trail(maze, algo="Dijkstra", pause_time=0.001)