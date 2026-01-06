import numpy as np
import random as rand
import os

def generate_maze(size: int):

    row_base = [0, -1, 1]
    row_all2 = [2, 2, 2]
    top_grid = [row_base * (size - 1) + [0],
                row_all2 * (size - 1) + [-1],
                row_all2 * (size - 1) + [1]]
    bottom_grid = [row_base * (size - 1) + [0]]
    maze_grid = np.array(top_grid * (size - 1) + bottom_grid, dtype=int)

    H, W = maze_grid.shape

    y, x = H - 1, W - 1

    def can_go_north(y, x):  
        return y - 3 >= 0
    def can_go_south(y, x):
        return y + 3 < H
    def can_go_west(y, x):
        return x - 3 >= 0
    def can_go_east(y, x):
        return x + 3 < W

    def legal_moves(y, x):
        moves = []
        if can_go_north(y, x): moves.append("North")
        if can_go_west(y, x):  moves.append("West")
        if can_go_south(y, x): moves.append("South")
        if can_go_east(y, x):  moves.append("East")
        return moves

    def move_original_node(direction: str, y: int, x: int):
        if direction == "North":
            maze_grid[y-1, x] = -1
            maze_grid[y-2, x] =  1
            return y-3, x
        if direction == "West":
            maze_grid[y, x-1] = -1
            maze_grid[y, x-2] =  1
            return y, x-3
        if direction == "South":
            maze_grid[y+1, x] = -1
            maze_grid[y+2, x] =  1
            return y+3, x
        if direction == "East":
            maze_grid[y, x+1] = -1
            maze_grid[y, x+2] =  1
            return y, x+3
        return y, x 

    def direction_arrow_remover(y: int, x: int):
        if y + 2 < H and maze_grid[y+1, x] == -1:
            maze_grid[y+1, x] = 2
            maze_grid[y+2, x] = 2
        if y - 2 >= 0 and maze_grid[y-1, x] == -1:
            maze_grid[y-1, x] = 2
            maze_grid[y-2, x] = 2
        if x + 2 < W and maze_grid[y, x+1] == -1:
            maze_grid[y, x+1] = 2
            maze_grid[y, x+2] = 2
        if x - 2 >= 0 and maze_grid[y, x-1] == -1:
            maze_grid[y, x-1] = 2
            maze_grid[y, x-2] = 2

    for _ in range(size ** 5):
        options = legal_moves(y, x)
        if not options:
            break
        d = rand.choice(options)
        y, x = move_original_node(d, y, x) 
        direction_arrow_remover(y, x)

    pre_convert_maze = maze_grid
    pre_convert_size = 3*size-2

    pre_01_maze = np.full((pre_convert_size, pre_convert_size), 1)
    for i in range (pre_convert_size):
        for j in range (pre_convert_size):
            if (pre_convert_maze[i,j] == 0) or (pre_convert_maze[i,j] == 1) or (pre_convert_maze[i,j] == -1):
                pre_01_maze[i,j] = 0
    rows, cols = pre_01_maze.shape

    maze_dict = []
    for i in range(rows):    
        if i % 3 != 2: 
            temp_row = []
            for j in range(cols):    
                if j % 3 != 2:
                    temp_row.append(pre_01_maze[i, j])
            maze_dict.append(temp_row) 
    maze_final = np.array(maze_dict)
    maze_final[-1, -1] = 9 
    maze_final[0,0] = 2

    return maze_final

path = os.path.dirname(os.path.abspath(__file__))
mazenotconvert = os.path.join(path, "maze(notconverted).txt")
maze01 = os.path.join(path, "maze01.txt")
mazefinal = os.path.join(path, "mazefinal.txt")

if __name__ == "__main__":
    gen_size = 30    
    maze_final = generate_maze(gen_size)
    #np.savetxt(mazenotconvert, pre_convert_maze, fmt="%d", delimiter=" ")

    #pre_convert_size = 3*gen_size-2
    #pre_01_maze = np.full((pre_convert_size, pre_convert_size), 1)
    #for i in range (pre_convert_size):
        #for j in range (pre_convert_size):
            #if (pre_convert_maze[i,j] == 0) or (pre_convert_maze[i,j] == 1) or (pre_convert_maze[i,j] == -1):
                #pre_01_maze[i,j] = 0
    #np.savetxt(maze01, pre_01_maze, fmt="%d", delimiter=" ")
                
    #rows, cols = pre_01_maze.shape
    #maze_dict = []
    #for i in range(rows):    
        #if i % 3 != 2: 
            #temp_row = []
            #for j in range(cols):    
                #if j % 3 != 2:
                    #temp_row.append(pre_01_maze[i, j])
            #maze_dict.append(temp_row) 
    #maze_final = np.array(maze_dict)
    #maze_final[-1, -1] = 9 
    #maze_final[0,0] = 2
    np.savetxt(mazefinal, maze_final, fmt="%d", delimiter=" ")



