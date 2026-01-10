#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

using namespace std;

extern "C" {

    void generate_maze_cpp(int* output_maze, int size) {
        int pre_size = 3 * size - 2;
        vector<int> maze_grid(pre_size * pre_size);

        for (int i = 0; i < pre_size; ++i) {
            for (int j = 0; j < pre_size; ++j) {
                int row_type = i % 3;
                int col_type = j % 3;
                
                if (row_type == 0) { 
                    if (col_type == 0) maze_grid[i * pre_size + j] = 0;
                    else if (col_type == 1) maze_grid[i * pre_size + j] = -1;
                    else maze_grid[i * pre_size + j] = 1;
                } else { 
                    if (col_type == 2 && row_type == 1) maze_grid[i * pre_size + j] = -1;
                    else if (col_type == 2 && row_type == 2) maze_grid[i * pre_size + j] = 1;
                    else maze_grid[i * pre_size + j] = 2;
                }
            }
        }

        int y = pre_size - 1;
        int x = pre_size - 1;
        random_device rd;
        mt19937 gen(rd());
        
        long long iterations = (long long)size * size * size * size * size;
        for (long long i = 0; i < iterations; ++i) {
            vector<int> moves;
            if (y - 3 >= 0) moves.push_back(0); 
            if (x - 3 >= 0) moves.push_back(1); 
            if (y + 3 < pre_size) moves.push_back(2); 
            if (x + 3 < pre_size) moves.push_back(3); 

            if (moves.empty()) break;

            uniform_int_distribution<> dis(0, moves.size() - 1);
            int d = moves[dis(gen)];

            if (d == 0) { maze_grid[(y-1)*pre_size + x] = -1; maze_grid[(y-2)*pre_size + x] = 1; y -= 3; }
            else if (d == 1) { maze_grid[y*pre_size + (x-1)] = -1; maze_grid[y*pre_size + (x-2)] = 1; x -= 3; }
            else if (d == 2) { maze_grid[(y+1)*pre_size + x] = -1; maze_grid[(y+2)*pre_size + x] = 1; y += 3; }
            else if (d == 3) { maze_grid[y*pre_size + (x+1)] = -1; maze_grid[y*pre_size + (x+2)] = 1; x += 3; }

            for (int dy : {-1, 1}) {
                if (y+2*dy >= 0 && y+2*dy < pre_size && maze_grid[(y+dy)*pre_size + x] == -1) {
                    maze_grid[(y+dy)*pre_size + x] = 2; maze_grid[(y+2*dy)*pre_size + x] = 2;
                }
                if (x+2*dy >= 0 && x+2*dy < pre_size && maze_grid[y*pre_size + (x+dy)] == -1) {
                    maze_grid[y*pre_size + (x+dy)] = 2; maze_grid[y*pre_size + (x+2*dy)] = 2;
                }
            }
        }

        
        int final_size = 2 * size - 1;
        int out_r = 0;
        for (int i = 0; i < pre_size; ++i) {
            if (i % 3 == 2) continue;
            int out_c = 0;
            for (int j = 0; j < pre_size; ++j) {
                if (j % 3 == 2) continue;
                
                int val = maze_grid[i * pre_size + j];
            
                output_maze[out_r * final_size + out_c] = (val == 0 || val == 1 || val == -1) ? 0 : -1;
                out_c++;
            }
            out_r++;
        }
        
        output_maze[0] = 2;                                
        output_maze[final_size * final_size - 1] = 1;        
    } 
} 

//g++ -O3 -shared -static -static-libgcc -static-libstdc++ -o maze_genc.dll .\mazegenfaster.cpp