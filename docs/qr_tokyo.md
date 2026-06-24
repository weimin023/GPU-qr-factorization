# Parallel QR Factorization of Block Low-Rank Matrices 筆記

Paper: **Parallel QR Factorization of Block Low-Rank Matrices**

Authors: M. Ridwan Apriansyah, Rio Yokota, Tokyo Institute of Technology

Links:

- arXiv: <https://arxiv.org/abs/2208.06194>
- ACM TOMS DOI: <https://doi.org/10.1145/3538647>

## 1. 論文主旨

這篇論文研究 **Block Low-Rank, BLR** 矩陣上的 QR factorization。BLR 矩陣把大矩陣切成平坦的二維 blocks，部分 off-diagonal blocks 以 low-rank 形式近似：

```text
A_ij ~= U_ij V_ij^T
```

論文提出兩個 Householder BLR-QR 方法：

1. **Block-column-wise Householder BLR-QR**
   - 類似傳統 blocked Householder QR。
   - 一次處理一整個 block column。
   - 利用 BLR 結構降低 arithmetic complexity。

2. **Tiled Householder BLR-QR**
   - 把 block-column QR 再拆成更小的 tile operations。
   - 單次 operation 粒度更細，更適合 task-based parallel execution。
   - 代價是需要儲存更多 intermediate `T` matrices。

論文也比較了既有的 **Blocked Modified Gram-Schmidt, MGS** BLR-QR。MGS 平行直覺較好，但數值穩定性不如 Householder。

## 2. 和我們問題的差異

我們的 submission 目標是 dense batched square QR：

```text
input:  A.shape = (batch, n, n), float32 CUDA tensor
output: (H, tau), same compact Householder format as torch.geqrf
```

重要差異：

- 論文主體假設 BLR 或低秩 off-diagonal blocks。
- 我們的測資是 dense matrix，不能任意 low-rank approximate。
- checker 會用 `H` 和 `tau` reconstruct `Q`，因此輸出必須維持 compact Householder convention。
- 論文 parallelism 主要面向 CPU shared-memory task runtime；我們需要把概念映射成 CUDA/CuTe DSL tile kernels。

因此，**BLR compression 不適合直接用作 submission 主路徑**。可用的是 tiled Householder 的 operation decomposition 與 `Y/T` block reflector 思路。

## 3. Dense QR 背景

### 3.1 Blocked MGS QR

論文先整理 blocked MGS：

```text
Algorithm 1: Blocked MGS QR factorization

Input:
  A with p x q blocks

Output:
  Q with p x q blocks
  R with q x q blocks
  A = Q R

for j = 1..q:
    [Q_j, R_jj] = QR(A_j)
    for k = j+1..q:
        R_jk = Q_j^T A_k
        A_k = A_k - Q_j R_jk
```

判斷：

- 優點：block-column level 有明顯 parallelism。
- 缺點：繼承 Gram-Schmidt 的數值風險。
- 對我們不適合：remote cases 包含 rank-def、near-rank、clustered、nearcollinear 等病態輸入。

### 3.2 Blocked Householder Dense QR

Blocked Householder 一次 factor 一個 panel 或 block column，並使用 compact WY 表示：

```text
Q_panel = I - Y T Y^T
```

其中：

- `Y` 儲存 Householder vectors。
- `T` 是小型 triangular matrix。
- trailing update 變成 Level-3-ish operation：

```text
C = C - Y T (Y^T C)
```

論文中的 blocked Householder QR 可抽象成：

```text
Algorithm 2: Blocked Householder QR factorization

Input:
  A with p x q blocks

Output:
  Y, R with p x q blocks
  T with 1 x q blocks
  R is upper triangular
  Y and T contain intermediate orthogonal factors

for k = 1..q:
    QR([A_kk, A_k+1,k, ..., A_pk]^T)
        produces R_kk, zeros below panel, and Qhat_k = I - Y_k T_k Y_k^T

    for j = k+1..q:
        [R_kj, A_k+1,j, ..., A_pj]^T =
            Qhat_k^T [A_kj, A_k+1,j, ..., A_pj]^T
```

這和目前 `submission_stream_unsafe.py` 的主架構高度接近。

## 4. Tiled Householder Dense QR

Tiled Householder 把整個 block-column triangularization 拆成多個 two-block operations。論文描述的 dense tiled QR 可整理成四個 operation 類型：

| Operation | 作用 | LAPACK/PLASMA 常見名稱 |
| --- | --- | --- |
| diagonal tile QR | 對 `A_kk` 做 QR，得到 `R_kk` 與 local reflector | `GEQRT` |
| apply diagonal reflector | 用 `Q_kk^T` 更新同一 tile row 的 trailing tiles | `ORMQR` |
| triangularize below tile | 對 `[R_kk; A_ik]` 做 QR，消去 `A_ik` | `TSQRT` |
| apply trapezoidal reflector | 用 `Q_ik^T` 更新 `[R_kj; A_ij]` | `TSMQR` |

論文中的 tiled Householder QR 可抽象成：

```text
Algorithm 4: Tiled Householder QR factorization

Input:
  A with p x q blocks

Output:
  Y, T, R with p x q blocks
  R is upper triangular
  Y and T contain intermediate orthogonal factors

for k = 1..q:
    QR(A_kk) = Qhat_kk R_kk
        Qhat_kk = I - Y_kk T_kk Y_kk^T

    for j = k+1..q:
        R_kj = Qhat_kk^T A_kj

    for i = k+1..p:
        QR([R_kk; A_ik]) = Qhat_ik [R'_kk; 0]
            Qhat_ik = I - [I; Y_ik] T_ik [I; Y_ik]^T

        for j = k+1..q:
            [R_kj; A_ij] = Qhat_ik^T [R_kj; A_ij]
```

核心價值：

- 每個 operation 只碰一個或兩個 tiles。
- task granularity 比 block-column QR 更細。
- task dependency 是 wavefront/DAG，容易有更多 parallel work。
- 代價是需要更多 `Y/T` intermediate storage。

## 5. BLR 版本的額外內容

BLR QR 會根據 block 類型採用不同操作：

- dense block：直接使用 dense Householder QR 或 dense update。
- low-rank block：利用 `U V^T` 結構降低 update 成本。
- low-rank addition/update 後可能需要重新 compression 或 rank revealing QR。

論文的 complexity improvement 主要來自這裡。

對我們的 dense submission，這部分不建議直接使用：

- 近似會破壞 exact QR checker。
- 需要額外 rank threshold，threshold 選擇會影響 correctness。
- remote cases 中 rank-def 與 near-rank 並不代表 off-diagonal blocks 可安全低秩近似。

## 6. 對目前 CuTe DSL submission 的映射

目前 `submission_stream_unsafe.py` 的 pipeline：

```text
for each panel:
    _panel_factor_apply_cutedsl_mvp(...)
        -> part2_3_factor_apply_panel_cuda
        -> factor panel and apply reflectors inside panel

    _build_panel_v_torch(...)
    _build_compact_wy_t_torch(...)
        -> build compact WY V/T

    part5_apply_panel_wy_gemv_cuda(...)
        -> apply C = C - V T (V^T C) to trailing matrix
```

對應關係：

| 論文概念 | 目前程式碼 | 狀態 |
| --- | --- | --- |
| Blocked Householder panel QR | `_panel_factor_apply_cutedsl_mvp` | 已有 correctness kernel，但 grid 只有 `(batch, 1, 1)` |
| Compact WY `Y/T` | `_build_panel_v_torch`, `_build_compact_wy_t_torch` | 數學相容，但仍用 Torch 建構 |
| Apply block reflector | `part5_apply_panel_wy_gemv_cuda` | 已有 CuTe DSL kernel，但仍是 per trailing column CTA |
| Tiled Householder `GEQRT/ORMQR/TSQRT/TSMQR` | 尚未實作 | 可作為下一代 parallelization direction |
| BLR low-rank update | 無 | 不建議用於主路徑 |

## 7. 能否用上

### 可直接吸收的想法

1. **保留 Householder，不切到 MGS**
   - 論文也指出 Householder 比 MGS 更穩定。
   - 符合我們的 compact `(H, tau)` output。

2. **把 trailing update 做成 tiled WY**
   - 目前 Part 5 是 `(batch, trailing_col)` 一個 CTA。
   - 應改成 `(batch, row_tile, col_tile)`，更接近論文 tiled operation 的 locality 與 granularity。

3. **把 panel factorization 往 tiled Householder 拆**
   - 目前 panel factor 一個 batch 只有一個 CTA。
   - Tokyo tiled QR 的 `GEQRT/TSQRT` 結構可以啟發我們把 panel 拆成 diagonal tile QR 加 below-tile eliminations。

4. **用更多小 `T` matrices 表示 intermediate reflectors**
   - Tiled QR 會產生更多 `T_kk`、`T_ik`。
   - 這可以提高 parallelism，但會增加 storage 和 output packing 複雜度。

### 不建議直接使用的部分

1. **BLR compression**
   - 不適合 exact dense QR checker。
   - 可能破壞 reconstruction residual。

2. **MGS BLR-QR**
   - 數值風險高。
   - output format 也不是 natural compact Householder `(H, tau)`。

3. **完整 CPU task runtime model**
   - 我們需要 CUDA/CuTe grid-level parallelism，不是 CPU task scheduler。

## 8. 建議落地順序

### Step 1: Tiled WY trailing update

先改 `part5_apply_panel_wy_gemv_cuda`：

```text
current:
  grid = (batch, trailing_col, 1)
  one CTA computes full-row V^T C for one trailing column

target:
  grid = (batch, row_tile, col_tile)
  one CTA handles a C tile
  reuse V/T inside tile
```

這是最接近論文精神、且最不破壞 output format 的改法。

### Step 2: Move compact WY T build into CuTe DSL

把 `_build_panel_v_torch` 和 `_build_compact_wy_t_torch` 移出 Torch runtime：

```text
input:
  compact Householder vectors stored in H
  tau[:, j_start:j_end]

output:
  T for the current panel
```

`nb <= 16/32` 時這是小矩陣工作，但可以減少 Python/Torch launch overhead。

### Step 3: Panel row-tiled reduction

把 `_part2_3_factor_apply_panel_kernel` 的 sigma reduction 拆成 row tiles：

```text
partial_sigma[b, panel_col, row_tile] = sum(x_tail_tile^2)
sigma = reduce(partial_sigma)
```

這會讓 panel factorization 不再只有 batch 個 CTAs。

### Step 4: Full tiled Householder QR experiment

最後才考慮完整 `GEQRT/ORMQR/TSQRT/TSMQR` 路線。

風險：

- 需要把多個 tile reflectors pack 回 checker 期待的 `H/tau`。
- 需要確認 reflector order 和 `torch.linalg.householder_product` convention 一致。
- 需要更多 `T` storage 或 temporary buffers。

## 9. 結論

這篇論文對我們最有用的不是 BLR 低秩近似，而是 **tiled Householder 的分解方式**：

- 用 `Y/T` block reflectors。
- 把 monolithic block-column work 拆成 tile operations。
- 用更細粒度的 dependencies 換取 parallelism。

短期最值得做：

```text
part5_apply_panel_wy_gemv_cuda
    -> tiled WY trailing update
```

中期再做：

```text
_build_compact_wy_t_torch
    -> CuTe DSL T build

_part2_3_factor_apply_panel_kernel
    -> row-tiled panel reduction
```

長期才考慮：

```text
full tiled Householder QR, GEQRT/ORMQR/TSQRT/TSMQR style
```
