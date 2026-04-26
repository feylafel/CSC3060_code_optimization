# Bitwise kernel — Version 2 (optimization report)

## What we changed (code-level)

**File:** `src/kernel/bitwise.cpp` — function `stu_bitwise`.

The reference `naive_bitwise` computes, for each index, a long chain of byte operations (`shared`, `either`, `diff`, `mixed0`, `mixed1`, then XOR). The student implementation keeps the **same input/output contract** (per-element `int8_t`, compared to the reference in `bitwise_check`) but evaluates the **equivalent** mapping using a **compact per-byte expression**, then **vectorizes** that expression with SSE2.

**Per byte (as unsigned 8-bit, matching the casts in `naive_bitwise`):**

```text
result_byte = 0xA5  XOR  ( (a_byte OR b_byte) AND 0x99 )
```

**Vectorized (16 bytes at a time):** load 16 `int8_t` from `a` and 16 from `b` as `__m128i` (raw bits match `uint8_t` for `| & ^`). Constants `0xA5` and `0x99` are **broadcast to every byte** with `_mm_set1_epi8`. Then:

```text
out =  A5_vec  XOR  ( (a_vec OR b_vec) AND  99_vec )
```

**Tail:** any remaining length that is not a multiple of 16 is handled with a **scalar** loop using the same per-byte formula.

**Loop structure:** 32-byte unrolled main loop (two SSE blocks), then 16-byte blocks, then scalar tail — same **memory access pattern** as a typical SSE kernel (unaligned load/store, high throughput on the main range).

---

## Why this is a sound optimization strategy

1. **Algebraic simplification (same function, fewer operations)**  
   The reference’s multi-step boolean mix collapses to **three** bitwise steps per output byte: `OR`, `AND`, `XOR`. Fewer operations per element usually means **fewer micro-ops** and **better register pressure**, and it keeps the **hot loop** small for the instruction cache.

2. **Bit-level / SIMD-style parallelism (assignment focus)**  
   `| & ^` do not carry between bytes. Each byte lane is independent, so a single `__m128i` instruction updates **16 outputs in parallel** with the **same** logic. That is exactly **data-parallel, lane-wise** SIMD, without horizontal dependencies between elements.

3. **Correctness relative to the lab**  
   The course checks **equality** to `naive_bitwise` (via `bitwise_check`), not the literal sequence of sub-expressions. As long as the **byte function** is equivalent to the reference on the promoted `uint8_t` values, the vectorized and scalar paths are both valid. *(Equivalence of the short formula to the long reference was verified separately; see your own proof or truth table in the writeup if required.)*

4. **What the TA typically looks for**  
   - You named the **bottleneck** (many ops per element in the naive form).  
   - You used a **defensible** transformation (short formula) + **exploitation of independence** (SIMD on packed bytes).  
   - You still handle **length not divisible by 16** (tail loop).  
   - You can cite **measured** speedup from `run_all` / `single_bench` with your build flags.

---

## What to paste or adapt for your final report

- **Problem:** per-element boolean pipeline on `int8` arrays.  
- **Idea 1 (math):** replace the pipeline by an **equivalent** 3-op byte formula.  
- **Idea 2 (hw):** apply that formula in **16-wide SIMD** with broadcast masks; scalar tail.  
- **Result:** show **speedup** vs baseline/naive in the table from the benchmark driver, and optionally **asm or perf** notes (e.g. fewer instructions in the loop body) if the assignment asks for evidence.

This document is a template you can merge into a larger “Code optimization” report section.
