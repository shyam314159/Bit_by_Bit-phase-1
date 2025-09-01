```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
```

# **Side-Channel Attack Metrics**

We analyze CPA results to estimate **Confidence** for key recovery.  

- ## **Confidence:**  
  Difference between the top two key scores:

  $\text{Confidence} = s_{\text{top}} - s_{\text{second}}$

  - High → top candidate clearly stands out, attack is reliable.  
  - Low → ambiguity between top candidates.



```python
def ge_confidence_from_cpa(cpa, beta=1.0, standardize=True):
    s = np.max(np.abs(cpa), axis=1)
    if standardize:
        s = (s - s.mean()) / (s.std(ddof=0) + 1e-20)
    p = np.exp(beta * s); p /= p.sum()
    
    order = np.argsort(-s); sorted_s = s[order]
    K = len(s)

    confidence = sorted_s[0] - sorted_s[1] if K > 1 else sorted_s[0]
    return {'confidence': confidence}
```

# **Helper Functions**


```python
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

```

# **Correlation Power Analysis (CPA) — Ciphertext Known Attack**

## Idea
Devices leak power ≈ function of data.  
We model leakage with **Hamming Weight**:

$$
HW(x) = number \: of \: 1 \: bits \: in \: x
$$

---

## Steps
1. **Hypothesis:**  
   For each key guess \(k \in [0,255]\):

   - If leakage comes from **last round SBox input**:

   $$
   z_i(k) = InvSBox(c_i \oplus k), \quad h_i(k) = HW(z_i(k))
   $$


   where \(c_i\) is ciphertext byte before shift row operation of corresponding trace \(i\).

2. **Correlation:**  
   For traces \(t_i\) and hypothesis \(h_i(k)\):

   $$
   \rho(k) = \frac{\text{Cov}(t,h(k))}{\sigma_t \, \sigma_{h(k)}}
   $$

3. **Decision:**  
   Key guess with max \(|\rho(k)|\) is the recovered key byte.

---

**Result:** Repeat for 16 bytes → AES last round key.



```python
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


```

# **CPA on Simulated Side-Channel Power Traces**


```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("pepelord2233/bit-by-bit-dataset")

print("Path to dataset files:", path)
```

    Path to dataset files: /kaggle/input/bit-by-bit-dataset



```python
import pandas as pd
import numpy as np

# load without header so first row is preserved
data = pd.read_csv("/kaggle/input/bit-by-bit-dataset/simulated_power_trace.csv", header=None)

# first column is ciphertext
ciphertexts_hex = data.iloc[:, 0].astype(str).values  
ciphertexts = np.array([list(bytes.fromhex(ct)) for ct in ciphertexts_hex])

# rest are traces
traces = data.iloc[:, 1:].to_numpy(dtype=np.int8)  

print(f"Loaded traces shape: {traces.shape} and dtye: {type(traces[0][0])}")
print(f"Loaded plaintexts shape: {ciphertexts.shape}")
```

    Loaded traces shape: (5000, 12) and dtye: <class 'numpy.int8'>
    Loaded plaintexts shape: (5000, 16)



```python
predicted_key = run_cpa_attack(traces, ciphertexts, "simulation_power_results")
```

    ****************************************************************************************************
    Attack byte 1 predicted key: 208
    Confidence (top-2 difference): 11.1334



    
![png](output_10_1.png)
    


    ****************************************************************************************************
    Attack byte 5 predicted key: 20
    Confidence (top-2 difference): 9.1605



    
![png](output_10_3.png)
    


    ****************************************************************************************************
    Attack byte 9 predicted key: 249
    Confidence (top-2 difference): 10.5383



    
![png](output_10_5.png)
    


    ****************************************************************************************************
    Attack byte 13 predicted key: 168
    Confidence (top-2 difference): 9.5574



    
![png](output_10_7.png)
    


    ****************************************************************************************************
    Attack byte 2 predicted key: 201
    Confidence (top-2 difference): 10.0299



    
![png](output_10_9.png)
    


    ****************************************************************************************************
    Attack byte 6 predicted key: 238
    Confidence (top-2 difference): 11.0491



    
![png](output_10_11.png)
    


    ****************************************************************************************************
    Attack byte 10 predicted key: 37
    Confidence (top-2 difference): 11.1120



    
![png](output_10_13.png)
    


    ****************************************************************************************************
    Attack byte 14 predicted key: 137
    Confidence (top-2 difference): 10.4165



    
![png](output_10_15.png)
    


    ****************************************************************************************************
    Attack byte 3 predicted key: 225
    Confidence (top-2 difference): 9.9341



    
![png](output_10_17.png)
    


    ****************************************************************************************************
    Attack byte 7 predicted key: 63
    Confidence (top-2 difference): 11.0717



    
![png](output_10_19.png)
    


    ****************************************************************************************************
    Attack byte 11 predicted key: 12
    Confidence (top-2 difference): 9.9246



    
![png](output_10_21.png)
    


    ****************************************************************************************************
    Attack byte 15 predicted key: 200
    Confidence (top-2 difference): 11.5052



    
![png](output_10_23.png)
    


    ****************************************************************************************************
    Attack byte 4 predicted key: 182
    Confidence (top-2 difference): 10.9076



    
![png](output_10_25.png)
    


    ****************************************************************************************************
    Attack byte 8 predicted key: 99
    Confidence (top-2 difference): 10.1633



    
![png](output_10_27.png)
    


    ****************************************************************************************************
    Attack byte 12 predicted key: 12
    Confidence (top-2 difference): 10.3940



    
![png](output_10_29.png)
    


    ****************************************************************************************************
    Attack byte 16 predicted key: 166
    Confidence (top-2 difference): 10.9005



    
![png](output_10_31.png)
    


    
    Predicted full AES last round key (hex): d014f9a8c9ee2589e13f0cc8b6630ca6


>**Actual 128 bit key value**


```python
predicted_key
```




    array([208,  20, 249, 168, 201, 238,  37, 137, 225,  63,  12, 200, 182,
            99,  12, 166], dtype=uint8)




```python
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

# run recovery
original_key = np.array (inv_key_schedule(predicted_key))
print("Recovered AES-128 master key (bytes):", original_key[:])
print("Hex:", ''.join(f"{b:02x}" for b in original_key))

```

    Recovered AES-128 master key (bytes): [ 43 126  21  22  40 174 210 166 171 247  21 136   9 207  79  60]
    Hex: 2b7e151628aed2a6abf7158809cf4f3c


# Structure of `cpa_summary.json`

The file contains a dictionary where:  

- **Keys** = AES byte index (0–15)  
- **Values** = list of 256 correlation scores, one per possible key guess  

---

## Example

```json
{
  "0": [0.0123, 0.0345, ..., 0.8765], 
  "1": [0.0456, 0.0234, ..., 0.9123],
  ...
  "15": [0.0789, 0.0567, ..., 0.8342]
}


The analysis produced sixteen files containing the correlation scores for each AES key byte:

- `byte_**.txt`    

Each file lists **256 possible values of the corresponding key byte**, one per line, and is saved in the folder:  `/kaggle/working/simulated_trace_key_bytes`


Additionally, the final AES key (derived from the highest correlation values) is saved in a separate file: simulated_power_trace_masterkey.txt


```python
import numpy as np
import os
import json

base_path = "/kaggle/working/simulation_power_results"
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

```

    Saved summary to cpa_summary.json



```python
import json, os

out_folder = '/kaggle/working/simulated_trace_key_bytes'
os.makedirs(out_folder, exist_ok=True)

data = json.load(open('/kaggle/working/simulation_power_results/cpa_summary.json'))

for i in range(16):
    with open(f'{out_folder}/byte_{i:02}.txt', 'w') as f:
        f.writelines(f"{s}\n" for s in data[str(i)])

```


```python
import numpy as np

# Suppose `arr` is your array
arr = original_key

# Save as space-separated text file
np.savetxt("/kaggle/working/simulated_power_trace_masterkey.txt", arr, fmt="%d")  

```

# **CPA on Real Side-Channel Power Traces**


```python
import pandas as pd
import numpy as np

# load without header so first row is preserved
data = pd.read_csv("/kaggle/input/bit-by-bit-dataset/real_power_trace.csv", header=None)

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
```

    Loaded traces shape: (8939, 2001) and dtye: <class 'numpy.int8'>
    Loaded plaintexts shape: (8939, 16)
    Loaded ciphertexts shape: (8939, 16)


    /usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py:1753: RuntimeWarning: invalid value encountered in cast
      result[rl.indexer] = arr



```python
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
```


    
![png](output_20_0.png)
    



```python
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


traces = filtered_traces # [:, 500:1000]
ciphertexts = filtered_ciphertexts
plaintexts = filtered_plaintexts
```

    Original traces: 8939
    Filtered traces: 8938



    
![png](output_21_1.png)
    



    
![png](output_21_2.png)
    



```python
predicted_key = run_cpa_attack(traces, ciphertexts, "real_power_results")
```

    ****************************************************************************************************
    Attack byte 1 predicted key: 153
    Confidence (top-2 difference): 0.5998



    
![png](output_22_1.png)
    


    ****************************************************************************************************
    Attack byte 5 predicted key: 152
    Confidence (top-2 difference): 0.7384



    
![png](output_22_3.png)
    


    ****************************************************************************************************
    Attack byte 9 predicted key: 238
    Confidence (top-2 difference): 0.3105



    
![png](output_22_5.png)
    


    ****************************************************************************************************
    Attack byte 13 predicted key: 147
    Confidence (top-2 difference): 0.0088



    
![png](output_22_7.png)
    


    ****************************************************************************************************
    Attack byte 2 predicted key: 30
    Confidence (top-2 difference): 0.9798



    
![png](output_22_9.png)
    


    ****************************************************************************************************
    Attack byte 6 predicted key: 151
    Confidence (top-2 difference): 0.2593



    
![png](output_22_11.png)
    


    ****************************************************************************************************
    Attack byte 10 predicted key: 203
    Confidence (top-2 difference): 0.4402



    
![png](output_22_13.png)
    


    ****************************************************************************************************
    Attack byte 14 predicted key: 209
    Confidence (top-2 difference): 0.8274



    
![png](output_22_15.png)
    


    ****************************************************************************************************
    Attack byte 3 predicted key: 165
    Confidence (top-2 difference): 1.0289



    
![png](output_22_17.png)
    


    ****************************************************************************************************
    Attack byte 7 predicted key: 20
    Confidence (top-2 difference): 0.8035



    
![png](output_22_19.png)
    


    ****************************************************************************************************
    Attack byte 11 predicted key: 30
    Confidence (top-2 difference): 0.2416



    
![png](output_22_21.png)
    


    ****************************************************************************************************
    Attack byte 15 predicted key: 244
    Confidence (top-2 difference): 0.7307



    
![png](output_22_23.png)
    


    ****************************************************************************************************
    Attack byte 4 predicted key: 41
    Confidence (top-2 difference): 0.1079



    
![png](output_22_25.png)
    


    ****************************************************************************************************
    Attack byte 8 predicted key: 36
    Confidence (top-2 difference): 0.6120



    
![png](output_22_27.png)
    


    ****************************************************************************************************
    Attack byte 12 predicted key: 132
    Confidence (top-2 difference): 0.0813



    
![png](output_22_29.png)
    


    ****************************************************************************************************
    Attack byte 16 predicted key: 86
    Confidence (top-2 difference): 0.1717



    
![png](output_22_31.png)
    


    
    Predicted full AES last round key (hex): 9998ee931e97cbd1a5141ef429248456


>**Actual 128 bit key value**


```python
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

# run recovery
original_key = np.array (inv_key_schedule(predicted_key))
print("Recovered AES-128 master key (bytes):", original_key[:])
print("Hex:", ''.join(f"{b:02x}" for b in original_key))

```

    Recovered AES-128 master key (bytes): [  9  74  51  85 172  56  55 250 162 162 118 135  75 101  42 134]
    Hex: 094a3355ac3837faa2a276874b652a86


# Structure of `cpa_summary.json`

The file contains a dictionary where:  

- **Keys** = AES byte index (0–15)  
- **Values** = list of 256 correlation scores, one per possible key guess  

---

## Example

```json
{
  "0": [0.0123, 0.0345, ..., 0.8765], 
  "1": [0.0456, 0.0234, ..., 0.9123],
  ...
  "15": [0.0789, 0.0567, ..., 0.8342]
}


The analysis produced sixteen files containing the correlation scores for each AES key byte:

- `byte_**.txt`    

Each file lists **256 possible values of the corresponding key byte**, one per line, and is saved in the folder:  `/kaggle/working/real_trace_key_bytes`


Additionally, the final AES key (derived from the highest correlation values) is saved in a separate file: real_power_trace_masterkey.txt




```python
import numpy as np
import os
import json

base_path = "/kaggle/working/real_power_results"
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

```

    Saved summary to cpa_summary.json



```python
import json, os

out_folder = '/kaggle/working/real_trace_key_bytes'
os.makedirs(out_folder, exist_ok=True)

data = json.load(open('/kaggle/working/real_power_results/cpa_summary.json'))

for i in range(16):
    with open(f'{out_folder}/byte_{i:02}.txt', 'w') as f:
        f.writelines(f"{s}\n" for s in data[str(i)])

```


```python
import numpy as np

# Suppose `arr` is your array
arr = original_key

# Save as space-separated text file
np.savetxt("/kaggle/working/real_power_trace_masterkey.txt", arr, fmt="%d")  


```

# Problems Faced During CPA Attack on Real Traces

We attempted a **Correlation Power Analysis (CPA) attack** on the real power traces using a **ciphertext-known attack on the last AES round**. However, the results did not match expectations:  

- The **correlation peaks are not evident**, unlike what we observe in simulation traces.  
- This makes it difficult to confidently recover key bytes using the standard CPA methodology.  

Additionally, **leakage analysis methods** such as **TVLA (Test Vector Leakage Assessment)** cannot be applied in this case because we **do not have access to the secret key or intermediate values**. This limitation prevents verification of potential leakage in the collected traces.  

Overall, the attack on real traces is inconclusive, and further investigation or more advanced techniques may be required to extract the key successfully.



```python
!zip -r real_trace_key_bytes.zip /kaggle/working/real_trace_key_bytes
```

      adding: kaggle/working/real_trace_key_bytes/ (stored 0%)
      adding: kaggle/working/real_trace_key_bytes/byte_14.txt (deflated 54%)
      adding: kaggle/working/real_trace_key_bytes/byte_12.txt (deflated 54%)
      adding: kaggle/working/real_trace_key_bytes/byte_08.txt (deflated 54%)
      adding: kaggle/working/real_trace_key_bytes/byte_11.txt (deflated 54%)
      adding: kaggle/working/real_trace_key_bytes/byte_06.txt (deflated 54%)
      adding: kaggle/working/real_trace_key_bytes/byte_03.txt (deflated 54%)
      adding: kaggle/working/real_trace_key_bytes/byte_04.txt (deflated 54%)
      adding: kaggle/working/real_trace_key_bytes/byte_09.txt (deflated 54%)
      adding: kaggle/working/real_trace_key_bytes/byte_02.txt (deflated 54%)
      adding: kaggle/working/real_trace_key_bytes/byte_13.txt (deflated 54%)
      adding: kaggle/working/real_trace_key_bytes/byte_01.txt (deflated 54%)
      adding: kaggle/working/real_trace_key_bytes/byte_10.txt (deflated 54%)
      adding: kaggle/working/real_trace_key_bytes/byte_05.txt (deflated 54%)
      adding: kaggle/working/real_trace_key_bytes/byte_15.txt (deflated 54%)
      adding: kaggle/working/real_trace_key_bytes/byte_00.txt (deflated 54%)
      adding: kaggle/working/real_trace_key_bytes/byte_07.txt (deflated 54%)



```python
!zip -r simulated_trace_key_bytes.zip /kaggle/working/simulated_trace_key_bytes
```

      adding: kaggle/working/simulated_trace_key_bytes/ (stored 0%)
      adding: kaggle/working/simulated_trace_key_bytes/byte_14.txt (deflated 54%)
      adding: kaggle/working/simulated_trace_key_bytes/byte_12.txt (deflated 54%)
      adding: kaggle/working/simulated_trace_key_bytes/byte_08.txt (deflated 54%)
      adding: kaggle/working/simulated_trace_key_bytes/byte_11.txt (deflated 54%)
      adding: kaggle/working/simulated_trace_key_bytes/byte_06.txt (deflated 53%)
      adding: kaggle/working/simulated_trace_key_bytes/byte_03.txt (deflated 54%)
      adding: kaggle/working/simulated_trace_key_bytes/byte_04.txt (deflated 54%)
      adding: kaggle/working/simulated_trace_key_bytes/byte_09.txt (deflated 54%)
      adding: kaggle/working/simulated_trace_key_bytes/byte_02.txt (deflated 54%)
      adding: kaggle/working/simulated_trace_key_bytes/byte_13.txt (deflated 53%)
      adding: kaggle/working/simulated_trace_key_bytes/byte_01.txt (deflated 53%)
      adding: kaggle/working/simulated_trace_key_bytes/byte_10.txt (deflated 54%)
      adding: kaggle/working/simulated_trace_key_bytes/byte_05.txt (deflated 53%)
      adding: kaggle/working/simulated_trace_key_bytes/byte_15.txt (deflated 54%)
      adding: kaggle/working/simulated_trace_key_bytes/byte_00.txt (deflated 53%)
      adding: kaggle/working/simulated_trace_key_bytes/byte_07.txt (deflated 54%)



```python

```
