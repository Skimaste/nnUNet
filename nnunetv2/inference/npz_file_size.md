An `.npz` file growing from 300 MB to 3 GB when stacking five files usually points to **unexpected data duplication, dtype inflation, or poor compression**. Here are the most likely causes:

---

### 🔍 1. **Loss of Compression Benefit**
- `.npz` is a zipped archive of `.npy` files (one per array).
- If your original arrays had **similar structure or values**, they might compress well individually.
- When you stack them, the resulting array might be **less compressible**, especially if:
  - The values are now more varied,
  - The layout disrupts compressibility,
  - Or compression is disabled (`np.savez` vs `np.savez_compressed`).

**Fix**: Use `np.savez_compressed` instead of `np.savez`.

---

### 🔍 2. **Data Type Blow-Up**
- Check your array `dtype` before and after stacking.
  - E.g., if your original arrays were `float16` or `float32`, and stacking coerced them to `float64`, that **doubles the memory usage** or more.

**Fix**: Explicitly set `dtype` when stacking:
```python
stacked = np.stack([a.astype(np.float32) for a in arrays])
```

---

### 🔍 3. **Unintended Copying or Metadata**
- Check that you're not stacking additional metadata or padding zeros.
- If arrays were of different shapes, NumPy may have silently expanded them with broadcasting rules.

**Fix**: Ensure all arrays are the same shape before stacking, and inspect their shape and size:
```python
for a in arrays:
    print(a.shape, a.dtype, a.nbytes)
```

---

### 🔍 4. **You’re Storing Intermediates**
- Sometimes people save the original arrays plus the stacked result in the same `.npz` — leading to a 6× file size increase.

**Fix**: Save only what’s needed:
```python
np.savez_compressed("stacked.npz", stacked=stacked)
```

---

Would you like help inspecting the file sizes and dtypes in your `.npz` file directly?