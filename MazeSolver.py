import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mazegenfromc import generate_maze
import time

model_path = "maze_ai_checkpoint.pth" 
maze_size = 20                    
vision_range = 4                

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

conv_layers = nn.Sequential(
    # Lớp 1: 9*9 -> 7*7
    nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3), 
    nn.ReLU(),
    # Lớp 2: 7*7 -> 5*5
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
    nn.ReLU(),
    # Lớp 3: 5*5 -> 3*3
    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
    nn.ReLU(),
    # Lớp 4: 3*3 -> 1*1
    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
    nn.ReLU()
)
decision_layer = nn.Linear(66, 4)
conv_layers = conv_layers.to(device)
decision_layer = decision_layer.to(device)

def load_trained_model(path):
    checkpoint = torch.load(path, map_location=device)
    conv_layers.load_state_dict(checkpoint['conv_state'])
    decision_layer.load_state_dict(checkpoint['decision_state'])
    conv_layers.eval()
    decision_layer.eval()
    print(f"Đã load thành công trọng số từ {path}")

def get_local_view(maze_np, pos):
    y, x = pos
    view = np.ones((9, 9), dtype=np.float32)
    for i in range(9):
        for j in range(9):
            mi, mj = y + i - 4, x + j - 4
            if 0 <= mi < maze_np.shape[0] and 0 <= mj < maze_np.shape[1]:
                view[i, j] = maze_np[mi, mj]
    return view

def predict_move(maze_np, curr_pos, goal_pos):
    with torch.no_grad():
        # View Tensor
        view = get_local_view(maze_np, curr_pos)
        view_tensor = torch.FloatTensor(view).unsqueeze(0).unsqueeze(0).to(device)
        # Goal Vector
        goal_vec = np.array([goal_pos[0] - curr_pos[0], goal_pos[1] - curr_pos[1]], dtype=np.float32)
        goal_tensor = torch.FloatTensor(goal_vec).unsqueeze(0).to(device)
        
        # Forward
        features = conv_layers(view_tensor)
        combined = torch.cat((features, goal_tensor), dim=1)
        q_values = decision_layer(combined)
        
        return torch.argmax(q_values).item()

# --- VÒNG LẶP TRÌNH DIỄN ---
def run_demo():
    # 1. Khởi tạo
    maze = generate_maze(MAZE_SIZE)
    start_pos = tuple(np.argwhere(maze == 2)[0])
    goal_pos = tuple(np.argwhere(maze == 9)[0])
    curr_pos = start_pos
    
    load_trained_model(MODEL_PATH)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    steps = 0
    path_history = [start_pos]
    
    while curr_pos != goal_pos and steps < 1000:
        action = predict_move(maze, curr_pos, goal_pos)
        
        # Tính toán tọa độ mới
        move_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dy, dx = move_map[action]
        next_pos = (curr_pos[0] + dy, curr_pos[1] + dx)
        
        # Kiểm tra va chạm (Dành cho demo an toàn)
        if 0 <= next_pos[0] < MAZE_SIZE and 0 <= next_pos[1] < MAZE_SIZE:
            if maze[next_pos[0], next_pos[1]] != 1:
                curr_pos = next_pos
                path_history.append(curr_pos)
        
        # Vẽ minh họa
        ax.clear()
        ax.imshow(maze, cmap='nipy_spectral')
        py, px = zip(*path_history)
        ax.plot(px, py, 'y-', linewidth=2, alpha=0.7) # Vẽ đường đã đi
        ax.plot(curr_pos[1], curr_pos[0], 'ro', markersize=10) # Agent
        ax.set_title(f"AI Showcase - Size {MAZE_SIZE}x{MAZE_SIZE} - Bước: {steps}")
        
        plt.pause(0.05)
        steps += 1
        
    plt.ioff()
    if curr_pos == goal_pos:
        print(f"✨ AI đã phá đảo mê cung {MAZE_SIZE}x{MAZE_SIZE} trong {steps} bước!")
    plt.show()

if __name__ == "__main__":
    run_demo()