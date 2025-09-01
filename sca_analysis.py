import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os

def ge_confidence_from_cpa(cpa, beta=1.0, standardize=True):
    s = np.max(np.abs(cpa), axis=1)
    if standardize:
        s = (s - s.mean()) / (s.std(ddof=0) + 1e-20)
    p = np.exp(beta * s); p /= p.sum()

    order = np.argsort(-s); sorted_s = s[order]
    K = len(s)

    confidence = sorted_s[0] - sorted_s[1] if K > 1 else sorted_s[0]
    return {'confidence': confidence}

# AES S-box and inverse S-box
AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B,
    0xFE, 0xD7, 0xAB, 0x76, 0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0,
    0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0, 0xB7, 0xFD, 0x93, 0x26,
    0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2,
    0xEB, 0x27, 0xB2, 0x75, 0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0,
    0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84, 0x53, 0xD1, 0x00, 0xED,
    0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F,
    0x50, 0x3C, 0x9F, 0xA8, 0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5,
    0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2, 0xCD, 0x0C, 0x13, 0xEC,
    0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14,
    0xDE, 0x5E, 0x0B, 0xDB, 0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C,
    0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79, 0xE7, 0xC8, 0x37, 0x6D,
    0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F,
    0x4B, 0xBD, 0x8B, 0x8A, 0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E,
    0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E, 0xE1, 0xF8, 0x98, 0x11,
    0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F,
    0xB0, 0x54, 0xBB, 0x16
])

AES_InvSbox = np.zeros(256, dtype=np.uint8)
for i in range(256):
    AES_InvSbox[AES_Sbox[i]] = i

def compute_hw_table():
    return np.array([bin(x).count("1") for x in range(256)], dtype=np.uint8)

def generate_hypothesis(ciphertext_column, key_guesses, inv_sbox, hw_table):
    d = np.bitwise_xor(key_guesses, ciphertext_column).astype(np.uint8)
    e = inv_sbox[d]
    hypothesis = hw_table[e]
    return hypothesis, e

def normalize(matrix):
    centered = matrix - np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    std[std == 0] = 1e-9
    return centered / std

def plot_cpa_results(cpa_results, byte_idx, save_path):
    plt.figure(figsize=(12, 6))
    for i in range(cpa_results.shape[1]):
        plt.plot(np.arange(256), cpa_results[:, i], color='red', alpha=0.2)
    plt.title(f'CPA Correlation Traces for Byte {byte_idx+1}')
    plt.xlabel('Key Guess')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


# # **Correlation Power Analysis (CPA) — Ciphertext Known Attack**


def compute_correlation(X, Y):
    m = X.shape[0]
    sum_xy = np.matmul(X.T, Y)
    mean_x, mean_y = np.sum(X, axis=0)[:, None]/m, np.sum(Y, axis=0)[:, None]/m
    numerator = sum_xy.T + m * np.matmul(mean_y, mean_x.T) - np.matmul(np.sum(Y, axis=0)[:, None], mean_x.T) - np.matmul(mean_y, np.sum(X, axis=0)[:, None].T)

    var_x = np.sum(X**2, axis=0)[:, None] + m*mean_x**2 - 2*mean_x*np.sum(X, axis=0)[:, None]
    var_y = np.sum(Y**2, axis=0)[:, None] + m*mean_y**2 - 2*mean_y*np.sum(Y, axis=0)[:, None]
    var_x[var_x <= 0] = np.inf
    var_y[var_y <= 0] = np.inf

    return numerator / np.sqrt(np.matmul(var_y, var_x.T))


# --- CPA Attack per byte ---
def cpa_attack_byte(traces, ciphertexts, attack_byte, hw_table, coord_pairs):
    m = len(traces)
    c_idx, b_idx = coord_pairs[attack_byte]
    c = np.repeat(ciphertexts[:, [c_idx]], 256, axis=1)
    b = np.repeat(ciphertexts[:, [b_idx]], 256, axis=1)
    key_guesses = np.tile(np.arange(256), (m,1)).astype(np.uint8)

    hypothesis, _ = generate_hypothesis(b, key_guesses, AES_InvSbox, hw_table)
    normalized_traces = normalize(traces)
    normalized_hyp = normalize(hypothesis)

    cpa_results = compute_correlation(normalized_traces, normalized_hyp)
    max_corr_per_key = np.max(np.abs(cpa_results), axis=1)
    best_key_guess = np.argmax(max_corr_per_key)

    return best_key_guess, cpa_results

# --- Main Attack Loop ---
def run_cpa_attack(traces, ciphertexts, save_path):
    os.makedirs(save_path, exist_ok=True)
    predicted_key = np.zeros(16, dtype=np.uint8)
    hw_table = compute_hw_table()

    # Attack order → AES canonical column order
    coord_pairs = np.array([
        [1,1],[5,5],[9,9],[13,13],
        [6,2],[10,6],[14,10],[2,14],
        [11,3],[15,7],[3,11],[7,15],
        [16,4],[4,8],[8,12],[12,16]
    ]) - 1

    # Temporary array to store results in attack order
    predicted_key = np.zeros(16, dtype=np.uint8)
    byte_num = 0
    for col in range(4):
        for row in range(4):
            attack_byte = row * 4 + col
            best_key, cpa_results = cpa_attack_byte(traces, ciphertexts, attack_byte, hw_table, coord_pairs)
            predicted_key[byte_num] = best_key
            byte_num = byte_num + 1

            metrics = ge_confidence_from_cpa(cpa_results, beta=1.0, standardize=True)
            print("*"*100)
            print(f"Attack byte {attack_byte+1} predicted key: {best_key}")
            print(f"Confidence (top-2 difference): {metrics['confidence']:.4f}")

            plot_cpa_results(cpa_results, attack_byte, f'{save_path}/cpa_correlation_byte_{attack_byte}.png')
            np.save(f'{save_path}/cpa_result_byte_{attack_byte}.npy', cpa_results)

    print("\nPredicted full AES last round key (hex):", predicted_key.tobytes().hex())
    return predicted_key

# round constants
Rcon = [0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1B,0x36]

def sub_word(word):
    return [AES_Sbox[b] for b in word]

def rot_word(word):
    return word[1:] + word[:1]

def inv_key_schedule(last_round_key):
    # total words for AES-128 = 44 (11 round keys × 4 words)
    words = [last_round_key[i:i+4] for i in range(0, 16, 4)]
    all_words = [None]*44
    all_words[40:44] = words

    for i in range(43, 3, -1):
        temp = all_words[i-1].copy()
        if i % 4 == 0:
            temp = sub_word(rot_word(temp))
            temp[0] ^= Rcon[(i//4)-1]
        all_words[i-4] = [a ^ b for a,b in zip(all_words[i], temp)]

    # first 4 words = original key
    original_key = sum(all_words[0:4], [])
    return original_key


# # **CPA on Simulated Side-Channel Power Traces**

import kagglehub

# Download latest version
path = kagglehub.dataset_download("pepelord2233/bit-by-bit-dataset")

print("Path to dataset files:", path)

import pandas as pd
import numpy as np

# load without header so first row is preserved
data = pd.read_csv(f"{path}/simulated_power_trace.csv", header=None)

# first column is ciphertext
ciphertexts_hex = data.iloc[:, 0].astype(str).values  
ciphertexts = np.array([list(bytes.fromhex(ct)) for ct in ciphertexts_hex])

# rest are traces
traces = data.iloc[:, 1:].to_numpy(dtype=np.int8)  

print(f"Loaded traces shape: {traces.shape} and dtye: {type(traces[0][0])}")
print(f"Loaded plaintexts shape: {ciphertexts.shape}")

predicted_key = run_cpa_attack(traces, ciphertexts, "simulation_power_results")
print(predicted_key)

# run recovery
original_key = np.array (inv_key_schedule(predicted_key))
print("Recovered AES-128 master key (bytes):", original_key[:])
print("Hex:", ''.join(f"{b:02x}" for b in original_key))


import numpy as np
import os
import json

base_path = "simulation_power_results"
results = {}

for byte_index in range(16):
    file_path = os.path.join(base_path, f"cpa_result_byte_{byte_index}.npy")

    # Load CPA result matrix
    # shape assumption: (num_key_guesses, num_samples)
    cpa_matrix = np.load(file_path)

    # For each key guess, take maximum correlation over all samples
    max_corr_per_key = np.max(np.abs(cpa_matrix), axis=1)  # shape (256,)

    # Store in dictionary
    results[byte_index] = max_corr_per_key.tolist()

# Save dictionary to JSON
with open(os.path.join(base_path, "cpa_summary.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Saved summary to cpa_summary.json")


import json, os

out_folder = 'simulated_trace_key_bytes'
os.makedirs(out_folder, exist_ok=True)

data = json.load(open('cpa_summary.json'))

for i in range(16):
    with open(f'{out_folder}/byte_{i:02}.txt', 'w') as f:
        f.writelines(f"{s}\n" for s in data[str(i)])


import numpy as np

# Suppose `arr` is your array
arr = original_key

# Save as space-separated text file
np.savetxt("simulated_power_trace_masterkey.txt", arr, fmt="%d")  


# # **CPA on Real Side-Channel Power Traces**

import pandas as pd
import numpy as np

# load without header so first row is preserved
data = pd.read_csv(f"{path}/real_power_trace.csv", header=None)

# first column is plaintext
plaintexts_hex = data.iloc[:, 0].astype(str).values  
plaintexts = np.array([list(bytes.fromhex(ct)) for ct in plaintexts_hex])

# second column is ciphertext
ciphertexts_hex = data.iloc[:, 1].astype(str).values  
ciphertexts = np.array([list(bytes.fromhex(ct)) for ct in ciphertexts_hex])

# rest are traces
traces = data.iloc[:, 2:].to_numpy(dtype=np.int8)  

print(f"Loaded traces shape: {traces.shape} and dtye: {type(traces[0][0])}")
print(f"Loaded plaintexts shape: {plaintexts.shape}")
print(f"Loaded ciphertexts shape: {ciphertexts.shape}")


import matplotlib.pyplot as plt

# plot first 5 traces
plt.figure(figsize=(20,6))
for i in range(10):
    plt.plot(traces[i], label=f"Trace {i}")

plt.xlabel("Sample index")
plt.ylabel("Power value")
plt.title("First 10 Traces")
plt.legend()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# region of interest
roi = traces[:, 700:750]

# min value in that region for each trace
peak_vals = np.min(roi, axis=1)

# keep only traces that dip below a chosen threshold
threshold = 80   # adjust depending on dataset
mask = peak_vals < threshold

filtered_traces = traces[mask]
filtered_ciphertexts = np.array(ciphertexts)[mask]
filtered_plaintexts = np.array(plaintexts)[mask]

print("Original traces:", traces.shape[0])
print("Filtered traces:", filtered_traces.shape[0])

# optional: plot before and after
plt.figure(figsize=(12,4))
for i in range(10):
    plt.plot(traces[i], alpha=0.6)
plt.title("Before filtering")
plt.show()

plt.figure(figsize=(12,4))
for i in range(10):
    plt.plot(filtered_traces[i], alpha=0.6)
plt.title("After filtering")
plt.show()


traces = filtered_traces 
ciphertexts = filtered_ciphertexts
plaintexts = filtered_plaintexts


predicted_key = run_cpa_attack(traces, ciphertexts, "real_power_results")
print(predicted_key)

# run recovery
original_key = np.array (inv_key_schedule(predicted_key))
print("Recovered AES-128 master key (bytes):", original_key[:])
print("Hex:", ''.join(f"{b:02x}" for b in original_key))


import numpy as np
import os
import json

base_path = "real_power_results"
results = {}

for byte_index in range(16):
    file_path = os.path.join(base_path, f"cpa_result_byte_{byte_index}.npy")

    # Load CPA result matrix
    # shape assumption: (num_key_guesses, num_samples)
    cpa_matrix = np.load(file_path)

    # For each key guess, take maximum correlation over all samples
    max_corr_per_key = np.max(np.abs(cpa_matrix), axis=1)  # shape (256,)

    # Store in dictionary
    results[byte_index] = max_corr_per_key.tolist()

# Save dictionary to JSON
with open(os.path.join(base_path, "cpa_summary.json"), "w") as f:
    json.dump(results, f, indent=2)

print("Saved summary to cpa_summary.json")


import json, os

out_folder = 'real_trace_key_bytes'
os.makedirs(out_folder, exist_ok=True)

data = json.load(open('cpa_summary.json'))

for i in range(16):
    with open(f'{out_folder}/byte_{i:02}.txt', 'w') as f:
        f.writelines(f"{s}\n" for s in data[str(i)])

import numpy as np

# Suppose `arr` is your array
arr = original_key

# Save as space-separated text file
np.savetxt("real_power_trace_masterkey.txt", arr, fmt="%d")  
