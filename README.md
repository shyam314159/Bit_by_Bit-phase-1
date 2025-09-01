# Bit by Bit Hackathon 2025 — Phase 1 Submission

This repository contains our submission for **Phase 1 of the Bit by Bit Hackathon 2025**.  
It showcases our work on **side-channel cryptanalysis of AES-128**, specifically applying **Correlation Power Analysis (CPA)** on simulated and real power traces.  
The goal is to demonstrate practical key recovery techniques and document challenges faced when moving from simulated to hardware-collected traces.


# **Side-Channel Attack Metrics**

We analyze CPA results to estimate **Confidence** for key recovery.  

- ## **Confidence:**  
  Difference between the top two key scores:

  $\text{Confidence} = s_{\text{top}} - s_{\text{second}}$

  - High → top candidate clearly stands out, attack is reliable.  
  - Low → ambiguity between top candidates.


# **Correlation Power Analysis (CPA) — Ciphertext Known Attack**

## Idea
Devices leak power ≈ function of data.  
We model leakage with **Hamming Weight**:

$$HW(x) = number \ of \ 1 \ bits \ in \ x$$

---

## Steps
1. **Hypothesis:**  
   For each key guess (k in [0,255]):

   - If leakage comes from **last round SBox input**:

   $$z_i(k) = InvSBox(c_i \oplus k), \quad h_i(k) = HW(z_i(k))$$


   where \(c_i\) is ciphertext byte before shift row operation of corresponding trace \(i\).

2. **Correlation:**  
   For traces \(t_i\) and hypothesis \(h_i(k)\):

   $$\rho(k) = \frac{\text{Cov}(t,h(k))}{\sigma_t \, \sigma_{h(k)}}$$

3. **Decision:**  
   Key guess with max $$\rho(k)$$ is the recovered key byte.

---

**Result:** Repeat for 16 bytes → AES last round key.

# **CPA on Real Side-Channel Power Traces**
    
    
    Predicted full AES last round key (bytes): [208,  20, 249, 168, 201, 238,  37, 137, 225,  63,  12, 200, 182, 99,  12, 166]
    Predicted full AES last round key (hex): d014f9a8c9ee2589e13f0cc8b6630ca6


    Recovered AES-128 master key (bytes): [ 43, 126,  21,  22,  40, 174, 210, 166, 171, 247,  21, 136,   9, 207,  79,  60]
    Recovered AES-128 master key (Hex): 2b7e151628aed2a6abf7158809cf4f3c


The analysis produced sixteen files containing the correlation scores for each AES key byte:

- `byte_**.txt`    

Each file lists **256 possible values of the corresponding key byte**, one per line, and is saved in the folder:  `simulated_trace_key_bytes`


Additionally, the final AES key (derived from the highest correlation values) is saved in a separate file: simulated_power_trace_masterkey.txt


# **CPA on Real Side-Channel Power Traces**

    
    Predicted full AES last round key (hex): 9998ee931e97cbd1a5141ef429248456
    Predicted full AES last round key (bytes): [153, 152, 238, 147, 30, 151, 203, 209, 165, 20, 30, 244, 41, 36, 132, 86]


    Recovered AES-128 master key (bytes): [  9,  74,  51,  85, 172,  56,  55, 250, 162, 162, 118, 135,  75, 101,  42, 134]
    Recovered AES-128 master key (Hex): 094a3355ac3837faa2a276874b652a86


The analysis produced sixteen files containing the correlation scores for each AES key byte:

- `byte_**.txt`    

Each file lists **256 possible values of the corresponding key byte**, one per line, and is saved in the folder:  `real_trace_key_bytes`


Additionally, the final AES key (derived from the highest correlation values) is saved in a separate file: real_power_trace_masterkey.txt



# Problems Faced During CPA Attack on Real Traces

We attempted a **Correlation Power Analysis (CPA) attack** on the real power traces using a **ciphertext-known attack on the last AES round**. However, the results did not match expectations:  

- The **correlation peaks are not evident**, unlike what we observe in simulation traces.  
- This makes it difficult to confidently recover key bytes using the standard CPA methodology.  

Additionally, **leakage analysis methods** such as **TVLA (Test Vector Leakage Assessment)** cannot be applied in this case because we **do not have access to the secret key or intermediate values**. This limitation prevents verification of potential leakage in the collected traces.  

Overall, the attack on real traces is inconclusive, and further investigation or more advanced techniques may be required to extract the key successfully.

