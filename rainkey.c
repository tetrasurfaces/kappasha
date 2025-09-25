// rainkey.c
// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2025 Anonymous
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/agpl-3.0.html>.
//
// Notes: Generates a spiral sequence on a QWERTY grid, calculates kangaroo hop distance,
//        generates a spectrum kappa, and computes Shannon entropy. Adapted from rainkey.py
//        for C, without visualization or wise_transforms integration.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// QWERTY layout (4x10 grid)
char qwerty[4][10] = {
    {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'},
    {'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'},
    {'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';'},
    {'Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/'}
};

// Hex mapping (0-9, a-f for all keys)
char hex_map[256] = {0};
void init_hex_map() {
    char hex_chars[] = "0123456789abcdef";
    int i, j;
    for (i = 0; i < 10; i++) hex_map[qwerty[0][i]] = hex_chars[i];
    for (i = 0; i < 10; i++) hex_map[qwerty[1][i]] = hex_chars[i + 10];
    for (i = 0; i < 9; i++) hex_map[qwerty[2][i]] = hex_chars[i + 20];
    hex_map[qwerty[2][9]] = 's';  // ';'
    for (i = 0; i < 7; i++) hex_map[qwerty[3][i]] = hex_chars[i + 27];
    hex_map[qwerty[3][7]] = 'c';  // ','
    hex_map[qwerty[3][8]] = 'd';  // '.'
    hex_map[qwerty[3][9]] = 'f';  // '/'
}

int key_pos[4][10][2] = {
    {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}, {0, 8}, {0, 9}},
    {{1, 0}, {1, 1}, {1, 2}, {1, 3}, {1, 4}, {1, 5}, {1, 6}, {1, 7}, {1, 8}, {1, 9}},
    {{2, 0}, {2, 1}, {2, 2}, {2, 3}, {2, 4}, {2, 5}, {2, 6}, {2, 7}, {2, 8}, {2, 9}},
    {{3, 0}, {3, 1}, {3, 2}, {3, 3}, {3, 4}, {3, 5}, {3, 6}, {3, 7}, {3, 8}, {3, 9}}
};

int find_position(char key, int *r, int *c) {
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 10; j++)
            if (qwerty[i][j] == key) {
                *r = i;
                *c = j;
                return 1;
            }
    return 0;
}

void generate_spiral_sequence(char start_key, int num_hops, float kappa, char *sequence) {
    if (num_hops > 20) num_hops = 20;  // Prevent buffer overflow
    int r, c;
    if (!find_position(start_key, &r, &c)) {
        strcpy(sequence, "Invalid start key");
        return;
    }
    sequence[0] = start_key;
    int visited[4][10] = {0};
    visited[r][c] = 1;
    int seq_len = 1;
    float theta = 0.0;
    srand(time(NULL) * clock());  // Unique seed per run
    float time_factor = (float)(rand() % 1000) / 1000.0 + 0.01;
    for (int hop = 1; hop < num_hops; hop++) {
        theta += (137.5 * M_PI / 180.0) / (hop * kappa) + time_factor;
        float distance = hop / kappa;
        float dx = cos(theta) * distance;
        float dy = sin(theta) * distance;
        int new_r = (int)((r + dy) + 0.5) % 4;
        int new_c = (int)((c + dx) + 0.5) % 10;
        char new_key = qwerty[new_r][new_c];
        if (!visited[new_r][new_c] && new_key != start_key) {
            sequence[seq_len++] = new_key;
            visited[new_r][new_c] = 1;
            r = new_r;
            c = new_c;
        } else if (seq_len < 40) {
            int dirs[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
            int dr = dirs[rand() % 4][0];
            int dc = dirs[rand() % 4][1];
            int adj_r = (r + dr + 4) % 4;
            int adj_c = (c + dc + 10) % 10;
            char adj_key = qwerty[adj_r][adj_c];
            if (!visited[adj_r][adj_c] && adj_key != start_key) {
                sequence[seq_len++] = adj_key;
                visited[adj_r][adj_c] = 1;
                r = adj_r;
                c = adj_c;
            }
        }
        if (seq_len >= num_hops) break;
    }
    while (seq_len < num_hops) {
        int r_new, c_new;
        do {
            r_new = rand() % 4;
            c_new = rand() % 10;
        } while (visited[r_new][c_new] || qwerty[r_new][c_new] == start_key);
        sequence[seq_len++] = qwerty[r_new][c_new];
        visited[r_new][c_new] = 1;
    }
    sequence[seq_len] = '\0';
}

int pollard_kangaroo_on_grid(int start_r, int start_c, int target_r, int target_c, int steps) {
    int grid_rows = 4, grid_cols = 10;
    int m = (int)sqrt(grid_rows * grid_cols) + 15;
    int jumps[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    int tame_map[4][10] = {-1};
    int x_t = start_r, y_t = start_c;
    for (int i = 0; i < m; i++) {
        int pos_r = x_t % grid_rows;
        int pos_c = y_t % grid_cols;
        if (tame_map[pos_r][pos_c] == -1) tame_map[pos_r][pos_c] = i;
        int j = rand() % 8;
        x_t = (x_t + jumps[j][0]) % grid_rows;
        y_t = (y_t + jumps[j][1]) % grid_cols;
    }
    int x_w = target_r, y_w = target_c;
    for (int i = 0; i < m * 4; i++) {
        int pos_r = x_w % grid_rows;
        int pos_c = y_w % grid_cols;
        if (tame_map[pos_r][pos_c] != -1) return tame_map[pos_r][pos_c] + i;
        int j = rand() % 8;
        x_w = (x_w + jumps[j][0]) % grid_rows;
        y_w = (y_w + jumps[j][1]) % grid_cols;
    }
    // Fallback to Manhattan distance
    return abs(start_r - target_r) + abs(start_c - target_c);
}

void generate_spectrum_kappa(char *sequence, char *kappa, int len) {
    int i = 0;
    for (; sequence[i] && i < len; i++) {
        kappa[i] = hex_map[(unsigned char)sequence[i]];
    }
    while (i < 16) kappa[i++] = '0';
    kappa[16] = '\0';
    for (i = 0; i < 16; i++) kappa[i] = tolower(kappa[i]);
    sprintf(kappa, "0x%s", kappa + (kappa[0] == '0' && kappa[1] == 'x' ? 2 : 0));
}

double calculate_shannon_entropy(char *sequence) {
    int freq[256] = {0};
    int total = strlen(sequence);
    for (int i = 0; i < total; i++) freq[(unsigned char)sequence[i]]++;
    double entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (freq[i] > 0) {
            double p = (double)freq[i] / total;
            entropy -= p * log2(p);
        }
    }
    return entropy;
}

int main(int argc, char *argv[]) {
    char start_key = (argc > 1) ? argv[1][0] : 'Q';
    int num_hops = (argc > 2) ? atoi(argv[2]) : 20;
    if (num_hops > 20) num_hops = 20;  // Prevent buffer overflow
    float kappa = (argc > 3) ? atof(argv[3]) : 1.0;

    init_hex_map();
    char sequence[21];  // Max 20 chars + null
    generate_spiral_sequence(start_key, num_hops, kappa, sequence);
    printf("Generated Sequence: %s\n", sequence);

    int start_r, start_c, target_r, target_c;
    find_position(sequence[0], &start_r, &start_c);
    find_position(sequence[num_hops - 1], &target_r, &target_c);
    int hop_distance = pollard_kangaroo_on_grid(start_r, start_c, target_r, target_c, 800);
    printf("Kangaroo hop distance to end key: %d\n", hop_distance);

    char spectrum_kappa[18];  // 0x + 16 chars + null
    generate_spectrum_kappa(sequence, spectrum_kappa, num_hops);
    printf("Spectrum Kappa: %s\n", spectrum_kappa);

    double entropy = calculate_shannon_entropy(sequence);
    printf("Shannon Entropy: %.4f bits\n", entropy);

    return 0;
}
