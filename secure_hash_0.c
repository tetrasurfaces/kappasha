/* secure_hash_0.c - Kappa-First Keccak Sponge for 4-bit Speedrun
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Capacity: 2 bits (security ~1 bit, toy hash). State: 4 bits (2x2x1). Output: 4 bits pre-division, 0 post-division (if rigged).
 * Notes: Kappa starts (curvature decay), Rho rotates lanes, division by 180° mod 369 flattens to 0. Reversible with quotient key.
 * Compile: gcc -O2 kappahash.c -o kappahash -lm
 * Requires: Standard C libraries (math.h). No external deps.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* Constants */
#define PHI 1.618033988749895 /* Golden ratio */
#define KAPPA_BASE 0.3536 /* Odd Mersenne m11/107 */
#define MODULO 369 /* Cyclic diffusion */
#define GRID_DIM 2 /* 2x2 state for 4 bits */
#define LANE_BITS 1 /* 1 bit per lane */
#define RATE 2 /* 2-bit absorption */
#define CAPACITY 2 /* 2-bit security */
#define OUTPUT_BITS 4 /* 4-bit hash */
#define ROUNDS 3 /* Speedrun rounds */

/* Mersenne Fluctuation for Kappa */
double mersenne_fluctuation(int prime_index) {
    double fluctuation = 0.0027 * (prime_index / 51.0);
    return (prime_index % 2 == 1) ? KAPPA_BASE + fluctuation : 0.3563 + fluctuation;
}

/* Kappa Calculation (Spiral Decay Curvature) */
double kappa_calc(int n, int round_idx, int prime_index) {
    double kappa_base = mersenne_fluctuation(prime_index);
    double abs_n = fabs((double)n - 12) / 12.0;
    double num = pow(PHI, abs_n) - pow(PHI, -abs_n);
    double denom = fabs(pow(PHI, 10.0/3) - pow(PHI, -10.0/3)) * fabs(pow(PHI, -5.0/6) - pow(PHI, 5.0/6));
    double result = (2 < n && n < 52) ?
        (1 + kappa_base * num / denom) * (2.0 / 1.5) - 0.333 :
        fmax(0, 1.5 * exp(-pow(n - 60, 2) / 400.0) * cos(0.5 * (n - 316)));
    return fmod(result, MODULO);
}

/* Kappa Transform (Row-wise Curvature Weighting) */
void kappa_transform(uint8_t state[GRID_DIM][GRID_DIM], uint8_t key[GRID_DIM][GRID_DIM], int round_idx, int prime_index) {
    for (int x = 0; x < GRID_DIM; x++) {
        for (int y = 0; y < GRID_DIM; y++) {
            int n = x * y;
            double kappa_val = kappa_calc(n, round_idx, prime_index);
            int shift = (int)fmod(kappa_val, LANE_BITS + 1) % 2; /* 0 or 1 for 1-bit lanes */
            state[x][y] ^= (key[x][y] >> shift) & 1;
        }
    }
}

/* Theta (Parity Diffusion) */
void theta(uint8_t state[GRID_DIM][GRID_DIM]) {
    uint8_t C[GRID_DIM] = {0};
    for (int x = 0; x < GRID_DIM; x++) {
        C[x] = state[x][0] ^ state[x][1];
    }
    uint8_t D[GRID_DIM] = {0};
    for (int x = 0; x < GRID_DIM; x++) {
        D[x] = C[(x + 1) % GRID_DIM] ^ C[(x + 1) % GRID_DIM]; /* Simplified for 2x2 */
    }
    for (int x = 0; x < GRID_DIM; x++) {
        for (int y = 0; y < GRID_DIM; y++) {
            state[x][y] ^= D[x];
        }
    }
}

/* Rho (Lane Rotations) */
void rho(uint8_t state[GRID_DIM][GRID_DIM]) {
    int offsets[GRID_DIM][GRID_DIM] = {{0, 1}, {1, 0}}; /* Simplified for 4-bit */
    for (int x = 0; x < GRID_DIM; x++) {
        for (int y = 0; y < GRID_DIM; y++) {
            state[x][y] = (state[x][y] << offsets[x][y]) & 1; /* 1-bit rotation (no-op or stay) */
        }
    }
}

/* Pi (Diagonal Shuffles) */
void pi(uint8_t state[GRID_DIM][GRID_DIM]) {
    uint8_t temp[GRID_DIM][GRID_DIM] = {0};
    for (int x = 0; x < GRID_DIM; x++) {
        for (int y = 0; y < GRID_DIM; y++) {
            temp[x][y] = state[(x + 3 * y) % GRID_DIM][x];
        }
    }
    memcpy(state, temp, sizeof(temp));
}

/* Chi (Nonlinear Bitwise Ops) */
void chi(uint8_t state[GRID_DIM][GRID_DIM]) {
    uint8_t temp[GRID_DIM][GRID_DIM] = {0};
    memcpy(temp, state, sizeof(temp));
    for (int x = 0; x < GRID_DIM; x++) {
        for (int y = 0; y < GRID_DIM; y++) {
            state[x][y] = temp[x][y] ^ ((~temp[(x + 1) % GRID_DIM][y]) & temp[(x + 1) % GRID_DIM][y]);
        }
    }
}

/* Iota (Round Constants) */
void iota(uint8_t state[GRID_DIM][GRID_DIM], int round_idx) {
    uint8_t RC[3] = {1, 0, 1}; /* Simplified for 3 rounds */
    state[0][0] ^= RC[round_idx % 3];
}

/* Pad Message */
void pad_message(const uint8_t *msg, size_t len, uint8_t *padded, size_t *padded_len) {
    memcpy(padded, msg, len);
    padded[len] = 0x1; /* Simple padding: 1, zeros, 1 */
    for (size_t i = len + 1; i < *padded_len - 1; i++) padded[i] = 0;
    padded[*padded_len - 1] = 0x1;
}

/* Absorb Input */
void absorb(uint8_t state[GRID_DIM][GRID_DIM], const uint8_t *chunk, size_t len) {
    size_t i = 0;
    for (int x = 0; x < GRID_DIM && i < len; x++) {
        for (int y = 0; y < GRID_DIM && i < len; y++) {
            state[x][y] ^= (chunk[i / 8] >> (i % 8)) & 1;
            i++;
        }
    }
}

/* Squeeze Output */
void squeeze(uint8_t state[GRID_DIM][GRID_DIM], uint8_t *output, int output_bits) {
    int i = 0;
    for (int y = 0; y < GRID_DIM && i < output_bits; y++) {
        for (int x = 0; x < GRID_DIM && i < output_bits; x++) {
            output[i / 8] |= (state[x][y] & 1) << (i % 8);
            i++;
        }
    }
}

/* Division by 180 (Flatten to 0) */
double divide_by_180(const uint8_t *hash_bytes, int bytes, double *quotient) {
    uint32_t H = 0; /* 4-bit hash fits in uint32_t */
    for (int i = 0; i < bytes; i++) {
        H = (H << 8) | hash_bytes[i];
    }
    double pi = M_PI;
    *quotient = floor(H / pi);
    double divided = H / pi;
    double modded = fmod(divided, MODULO);
    return (fabs(modded) < 1e-6) ? 0.0 : modded; /* Force 0 if close */
}

/* Simplified Wise Transforms (No mpmath) */
void bitwise_transform(const uint8_t *data, int len, char *out, int bits) {
    uint32_t int_data = 0;
    for (int i = 0; i < len; i++) {
        int_data = (int_data << 8) | data[i];
    }
    uint32_t mask = (1U << bits) - 1;
    uint32_t mirrored = (~int_data) & mask;
    snprintf(out, bits + 1, "%0*u", bits, mirrored);
}

void hexwise_transform(const uint8_t *data, int len, char *out, double angle) {
    char hex_data[2 * len + 1];
    for (int i = 0; i < len; i++) {
        snprintf(hex_data + 2 * i, 3, "%02x", data[i]);
    }
    hex_data[2 * len] = '\0';
    int hex_len = strlen(hex_data);
    char mirrored[2 * hex_len + 1];
    strcpy(mirrored, hex_data);
    for (int i = 0; i < hex_len; i++) {
        mirrored[hex_len + i] = hex_data[hex_len - 1 - i];
    }
    mirrored[2 * hex_len] = '\0';
    int shift = (int)fmod(angle, strlen(mirrored));
    char rotated[2 * hex_len + 1];
    strcpy(rotated, mirrored + shift);
    strncat(rotated, mirrored, shift);
    rotated[2 * hex_len] = '\0';
    strcpy(out, rotated);
}

void hashwise_transform(const uint8_t *data, int len, char *out, int *ent) {
    uint8_t base_hash[32]; /* Dummy SHA-512 */
    for (int i = 0; i < 32; i++) {
        base_hash[i] = data[i % len] ^ i;
    }
    double mp_state = 0.0;
    for (int i = 0; i < 32; i++) {
        mp_state = mp_state * 256 + base_hash[i];
    }
    for (int i = 0; i < 4; i++) {
        mp_state = sqrt(mp_state) * PHI;
    }
    char partial[32];
    snprintf(partial, 32, "%.31f", mp_state);
    uint8_t final_hash[16];
    for (int i = 0; i < 16; i++) {
        final_hash[i] = partial[i] ^ i; /* Dummy SHA-256 */
    }
    for (int i = 0; i < 16; i++) {
        snprintf(out + 2 * i, 3, "%02x", final_hash[i]);
    }
    out[32] = '\0';
    *ent = (int)log2(mp_state + 1);
}

/* Braid Wise Transforms */
void braid_with_wise(const uint8_t *hash_bytes, int len, char *braided, int braided_len) {
    char bit_out[OUTPUT_BITS + 1];
    bitwise_transform(hash_bytes, len, bit_out, OUTPUT_BITS);
    char hex_out[2 * len * 2 + 1];
    hexwise_transform(hash_bytes, len, hex_out, 137.5);
    char hash_out[33];
    int ent;
    hashwise_transform(hash_bytes, len, hash_out, &ent);
    snprintf(braided, braided_len, "%s:%s:%s", bit_out, hex_out, hash_out);
}

/* Main KappaHash Sponge */
void kappahash(const uint8_t *message, size_t len, const uint8_t *key, size_t key_len, uint8_t *output, double *flattened, double *quotient, int prime_index) {
    uint8_t state[GRID_DIM][GRID_DIM] = {0};
    uint8_t key_lanes[GRID_DIM][GRID_DIM] = {0};
    /* Load key (4 bits) */
    for (int i = 0, k = 0; i < GRID_DIM && k < key_len; i++) {
        for (int j = 0; j < GRID_DIM && k < key_len; j++, k++) {
            key_lanes[i][j] = (key[k / 8] >> (k % 8)) & 1;
        }
    }
    /* Pad and absorb */
    size_t padded_len = len + (RATE - len % RATE) + 1;
    uint8_t padded[padded_len];
    pad_message(message, len, padded, &padded_len);
    for (size_t i = 0; i < padded_len; i += RATE) {
        absorb(state, padded + i, RATE);
        for (int round_idx = 0; round_idx < ROUNDS; round_idx++) {
            kappa_transform(state, key_lanes, round_idx, prime_index);
            theta(state);
            rho(state);
            pi(state);
            chi(state);
            iota(state, round_idx);
        }
    }
    /* Squeeze */
    squeeze(state, output, OUTPUT_BITS);
    /* Divide by 180 */
    *flattened = divide_by_180(output, OUTPUT_BITS / 8, quotient);
}

/* Main Function */
int main() {
    uint8_t message[] = "test";
    size_t len = strlen((char *)message);
    uint8_t key[1] = {0x5}; /* 4-bit key: 0101 */
    uint8_t hash[OUTPUT_BITS / 8] = {0};
    double flattened, quotient;
    int prime_index = 11; /* Odd Mersenne */
    
    kappahash(message, len, key, 1, hash, &flattened, &quotient, prime_index);
    
    printf("Hash: %02x (4-bit)\n", hash[0] & 0x0F);
    printf("Flattened: %.1f\n", flattened);
    printf("Quotient (Key): %.0f\n", quotient);
    
    /* Braid with Wise Transforms */
    char braided[512];
    braid_with_wise(hash, OUTPUT_BITS / 8, braided, sizeof(braided));
    printf("Braided: %.64s...\n", braided);
    
    /* Recover */
    double recovered = floor(quotient * M_PI);
    printf("Recovered Hash: %d\n", (int)recovered & 0x0F);
    
    return 0;
}
