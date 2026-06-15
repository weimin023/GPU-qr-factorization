# CuTe DSL QR 分解計畫

## 目前環境

CuTe DSL 已安裝在 workspace-local target 目錄：

```bash
/workspace/.cutedsl
```

在目前 shell 啟用：

```bash
source /workspace/cutedsl_env.sh
```

等價的手動設定：

```bash
export PYTHONPATH="/workspace/.cutedsl:/workspace/.cutedsl/nvidia_cutlass_dsl/python_packages:${PYTHONPATH}"
```

Smoke test：

```bash
python -c "import cutlass; import cutlass.cute as cute; print('cute import ok')"
```

目前觀察到的機器狀態：

- Python: 3.12.3
- CuTe DSL package: `nvidia-cutlass-dsl[cu13] == 4.5.2`
- NVIDIA driver: 590.48.01
- `nvidia-smi` 回報的 driver CUDA version: 13.1
- `nvcc`: CUDA 13.0
- PyTorch: 2.11.0+cu130
- 執行 `source /workspace/cutedsl_env.sh` 後的 PyTorch CUDA visibility: `torch.cuda.is_available() == True`, `torch.cuda.device_count() == 4`

CuTe DSL Python package 可以正確 import，且 source `cutedsl_env.sh` 後 PyTorch 可以看到 CUDA devices。原本的問題是 `CUDA_VISIBLE_DEVICES=all`：這對部分 container runtime 是合法值，但對 PyTorch/CUDA device enumeration 不是合法 GPU list。

## 比賽公告萃取

公告中對實作有直接影響的資訊：

- 問題是 real square matrix QR decomposition，不是一般 rectangular QR。搭配 `qr_official.py` 可確認 input shape 是 `(batch, n, n)`。
- 目標分解：

$$
A = QR,
\qquad
Q^T = Q^{-1},
\qquad
R = \operatorname{triu}(H).
$$

- Reference implementation 是 `torch.geqrf`，輸出 compact Householder factors `(H, tau)`。Evaluator 會 materialize `Q`，再從 `H` 取出 `R`。
- 評估 properties：

$$
\begin{aligned}
\text{factorization:}\quad &R \approx Q^TA, \\
\text{orthogonality:}\quad &Q^TQ \approx I, \\
\text{reconstruction:}\quad &QR \approx A, \\
\text{triangularity:}\quad &\operatorname{lower}(Q^TA)\approx 0 .
\end{aligned}
$$

- Tolerance 會用 relative tolerance 並按 $n\epsilon_{\text{fp32}}$ scale。這表示可以探索較低 bit width，但 rank-deficient、near-collinear、row-scaled cases 會放大數值不穩定。
- Benchmark 主要會測 dense random square matrices，但 correctness tests 包含：
  - rank-deficient
  - near-rank-deficient
  - banded
  - row-scaled
  - near-collinear
  - upper-triangular
  - clustered-scale
- 公告明確指出 naive Householder 每個 column 依賴前一個 column，因此 GPU-unfriendly；它也指出 blocked Householder 的方向是把 reflectors 累積成 compact form，再用大矩陣乘法套用。

對目前計畫的結論：

- 保持 compact Householder output `(H, tau)`，不要改成顯式輸出 `(Q, R)`。
- 優先優化 blocked Householder / compact WY：

$$
C \leftarrow C - VT(V^TC).
$$

- 不應改走 classical Gram-Schmidt 作為主要方案；它雖然直覺上 column parallel，但在公告要求的病態 cases 下數值風險較高。
- 官方 shape 是 square，所以目前優化可以專注在 batched `(n,n)`；一般 `(m,n)` 支援不是優先項目。

## Householder 數學表示慣例

這份實作使用的 Householder QR 數學，和手寫推導是一樣的；差別是每個 reflector 的儲存方式採用 LAPACK/`torch.geqrf` compact form，而不是顯式 materialize 每個 Householder matrix。

手寫推導常見的 normalized-vector 形式：

$$
\begin{aligned}
\alpha &= -\operatorname{sign}_{0}(b_0)\lVert b\rVert_2, \\
u &= \frac{b - \alpha e_1}{\lVert b - \alpha e_1\rVert_2}, \\
H &= I - 2uu^T, \\
A &\leftarrow HA .
\end{aligned}
$$

目前實作使用的 compact form：

$$
\begin{aligned}
\sigma &= \sum_{i>0} b_i^2, \\
\beta &= -\operatorname{sign}_{0}(\alpha_0)\sqrt{\alpha_0^2+\sigma}, \\
\tau &= \frac{\beta-\alpha_0}{\beta}, \\
v &= \left[
1,\frac{b_1}{\alpha_0-\beta},\ldots,
\frac{b_{m-1}}{\alpha_0-\beta}
\right]^T, \\
H &= I-\tau vv^T, \\
A &\leftarrow HA .
\end{aligned}
$$

這兩種形式在數學上等價：

$$
\tau vv^T = 2uu^T,
\qquad
I-\tau vv^T = I-2uu^T .
$$

以筆記中的第一個 column 為例：

$$
\begin{aligned}
b &= [3,6,6]^T, \\
\beta &= -\lVert b\rVert_2 = -9, \\
\tau &= \frac{-9-3}{-9} = \frac{4}{3}, \\
v &= \left[1,\frac{6}{3-(-9)},\frac{6}{3-(-9)}\right]^T
   = \left[1,\frac{1}{2},\frac{1}{2}\right]^T, \\
I-\tau vv^T &= I-\frac{4}{3}vv^T = I-2uu^T .
\end{aligned}
$$

所以 QR 分解和手寫的顯式流程相同：

$$
\begin{aligned}
A_1 &= H_1A, \\
A_2 &= H_2A_1, \\
R &= \operatorname{upper}(A_k), \\
Q &= H_1H_2\cdots H_k .
\end{aligned}
$$

差別在儲存格式：

| 推導寫法 | 目前實作 |
| --- | --- |
| 概念上建立或寫出完整 `H_j` matrices | 不 materialize 完整 `H_j` |
| 儲存 normalized `u_j`，係數固定為 `2` | 儲存 unnormalized `v_j`，係數為 scalar `tau_j` |
| 適合手算與閱讀 | 符合 LAPACK/`torch.geqrf` 與 `torch.linalg.householder_product` |
| `Q` 由顯式 reflector 相乘形成 | `Q` 只在 validation 時由 compact `(H, tau)` 重建 |

在 `qr_impl.py` 中，輸出的 `H` 不是完整 Householder matrix，而是 compact storage：

$$
\begin{aligned}
\operatorname{upper}(H) &= R, \\
\operatorname{strictLower}(H) &= \{v_j[1:]\}_j, \\
\tau_j &\text{ is the scalar in } H_j = I-\tau_j v_jv_j^T .
\end{aligned}
$$

## QR 拆分

CUDA/CuTe DSL 版本的 compute pipeline 拆成 5 個部分。這樣可以把 Householder 的 serial dependency 控制在 panel 內，同時把高流量的計算搬進 fused GPU kernels。Validation、benchmarking、fallback 另外列成工程檢查項目，不算 QR compute part。

## Backend 對照表

| Part | 元件 | 主要 backend | 目前 MVP backend | 備註 |
| --- | --- | --- | --- | --- |
| 1 | Python orchestration and dispatch | Python + Torch CUDA allocation | Python + Torch CUDA | 除 fallback/debug 外不使用 Torch CPU。Panel loop 與 kernel launch 保留在 Python 外層。 |
| 2 | Panel column factor kernel | CuTe DSL/CUDA | Fused CuTe DSL panel kernel | 第一個 custom-kernel 目標。處理 reductions、beta/tau、以及 compact Householder vector storage。 |
| 3 | Panel internal apply kernel | CuTe DSL/CUDA | Fused CuTe DSL panel kernel | 已和 Part 2 fuse 成目前第一版 panel kernel。 |
| 4 | Build compact WY `T` | CuTe DSL/CUDA | 先用 Torch CUDA，之後移到 CuTe DSL/CUDA | 小矩陣且 latency-sensitive。先用 Torch CUDA 隔離 correctness。 |
| 5 | Trailing matrix update | CuTe DSL/CUDA 或 CUTLASS GEMM-style kernels | CuTe DSL WY 或 raw CUDA tiled WY | 目前主線是 compact WY；raw CUDA tiled WY 已作為效能對照 backend。 |

## Part 1: Python Orchestration And Dispatch

Backend: Python orchestration 搭配 Torch CUDA allocations。Torch CPU 只用於 fallback 或 debugging。

數學函式：

$$
\begin{aligned}
(H,\tau) &= \operatorname{QR}(A), \\
\text{for each panel } p:\quad
(H,\tau) &\leftarrow \operatorname{panelFactorApply}(H,\tau,p), \\
H &\leftarrow \operatorname{trailingUpdate}(H,\tau,p).
\end{aligned}
$$

職責：

- 接收 shape 為 `(batch, n, n)`、dtype 為 `float32` 的 input tensor `A`。
- 配置 output `H = A.clone()` 和 `tau = zeros(batch, n)`。
- 依 panel loop：`j_start = 0, nb, 2*nb, ...`。
- Launch CuTe DSL kernels 做 panel factorization、`T` construction、trailing update。
- 回傳與 `torch.geqrf` 相同 compact Householder 格式的 `(H, tau)`。

這部分保留在 Python，因為 panel progression 本身是 sequential，launch orchestration 放在 DSL 外面比較單純。

## Part 2: Panel Column Factor Kernel

Backend: CuTe DSL/CUDA。目前實作用 fused CuTe DSL panel kernel，包含 parallel sigma reduction 與 panel-internal reflector application。

數學函式：

$$
\begin{aligned}
x &= H_{j:m,\;j}, \\
\alpha &= x_0, \\
\sigma &= \sum_{i=1}^{m-j-1} x_i^2, \\
\beta &= -\operatorname{sign}_{0}(\alpha)\sqrt{\alpha^2+\sigma}, \\
\tau_j &= \frac{\beta-\alpha}{\beta}, \\
v_j &= \left[
1,\frac{x_1}{\alpha-\beta},\ldots,
\frac{x_{m-j-1}}{\alpha-\beta}
\right]^T, \\
H_{j,j} &\leftarrow \beta, \\
H_{j+1:m,\;j} &\leftarrow v_j[1:] .
\end{aligned}
$$

對應目前程式碼區域：`qr_impl.py` 的 panel reflector computation，以及 `cutedsl_kernels.py` 的 `part2_3_factor_apply_panel_cuda`。

每個 panel column `j` 的職責：

- 對每個 batch item 計算 $\sigma=\sum_i H_{i,j}^2$。
- 計算穩定的 Householder values：
  - $\alpha = H_{j,j}$
  - $\beta = -\operatorname{sign}_{0}(\alpha)\sqrt{\alpha^2+\sigma}$
  - $\tau = (\beta-\alpha)/\beta$
  - $v_{\text{tail}} = H_{j+1:m,j}/(\alpha-\beta)$
- 儲存：
  - $H_{j,j} \leftarrow \beta$
  - $H_{j+1:m,j} \leftarrow v_{\text{tail}}$
  - $\tau_j \leftarrow \tau$

重要修正：不要直接用 `torch.sign(alpha)` 來決定 beta 的 sign。當 `alpha == 0` 時會得到 `0`，使 nonzero reflector 被錯誤地關掉。應使用 `alpha >= 0 ? +1 : -1`。

平行化方式：

- 每個 batch item 對應一個 CTA。
- 128 threads 對 column tail 做 shared-memory parallel sigma reduction。
- thread 0 計算 `beta/tau/scale`。
- 所有 threads 平行寫回 `v_tail`。
- 同一個 fused panel kernel 內接著做 panel-internal apply。

## Part 3: Panel Internal Apply Kernel

Backend: CuTe DSL/CUDA。目前實作用 fused CuTe DSL panel kernel，包含 parallel sigma reduction 與 panel-internal reflector application。

數學函式：

$$
\begin{aligned}
C_{\text{panel}} &= H_{j:m,\;j+1:j_{\text{end}}}, \\
w &= \tau_j v_j^T C_{\text{panel}}, \\
C_{\text{panel}} &\leftarrow C_{\text{panel}} - v_jw .
\end{aligned}
$$

對應目前程式碼區域：在 current panel 內，把每個 reflector 套用到剩餘 columns。

職責：

- 對目前 reflector `v`，更新 columns `j+1 : j_end`。
- 計算 $w=\tau_j v_j^T H_{j:m,\;j+1:j_{\text{end}}}$。
- 套用 $H_{j:m,\;j+1:j_{\text{end}}}\leftarrow H_{j:m,\;j+1:j_{\text{end}}}-v_jw$。

平行化方式：

- Batch items 互相獨立。
- Remaining panel width 通常很小，例如 `nb <= 16` 或 `nb <= 32`。
- 目前已和 Part 2 fuse 成 `part2_3_factor_apply_panel_cuda`，避免每個 panel column 都回到 Python launch 多個小 kernels。

## Part 4: Build Compact WY `T` Kernel

Backend: 目前 MVP 用 Torch CUDA，優化版再移到 CuTe DSL/CUDA。

數學函式：

$$
\begin{aligned}
V &= [v_0,v_1,\ldots,v_{b-1}], \\
T_{0,0} &= \tau_0, \\
T_{j,j} &= \tau_j, \\
T_{j,0:j} &=
-\tau_j\left(v_j^T V_{:,0:j}\right)T_{0:j,0:j},
\qquad j=1,\ldots,b-1, \\
Q_{\text{panel}} &= I - VTV^T .
\end{aligned}
$$

對應目前程式碼區域：reverse-order lower-triangular `T` 的建構。

職責：

- 從已存在 `H` 中的 compact Householder vectors 邏輯建出 `V`；優化版應避免 materialize full `V`。
- 建構 panel 的 reverse-order lower-triangular `T`：
  - $T_{j,j}=\tau_j$
  - $T_{j,0:j}=-\tau_j(v_j^T V_{:,0:j})T_{0:j,0:j}$

平行性：

- Batch dimension 互相獨立。
- `nb` 很小，所以這部分偏 latency-sensitive，而不是 FLOP-heavy。
- 優化版可考慮 shared memory 儲存 `T`，並直接從 compact `H` stream reads。

第一版允許 materialize `V` 和 `T` temporary tensors。優化版應盡量避免 full `V` materialization。

## Part 5 GEMV Fusion 歷史紀錄

以下三點是先前 GEMV-style reflector kernel 的實驗結論。主流程目前已移除這條路徑，只保留 compact WY；這段保留作為為什麼轉向 WY tiled kernel 的背景。

1. `v` 可以直接 stream into shared memory

   Householder vector `v = [1, v_tail]` 已經以 compact form 存在 `H` 的 column 中。做 trailing update 時，不一定要先 materialize full `V` 到 global memory。更好的方式是每個 CTA 直接從 compact `H[:, j:, j]` streaming load 需要的 `v` segment 到 shared memory 或 registers。

   $$
   H_{j+1:m,j}\rightarrow v_{\text{tile}},
   \qquad
   v_0 = 1 \text{ is implicit}.
   $$

2. `tau` 是 scalar，適合 fuse

   每個 reflector 只有一個 `tau_j` scalar。這個值可以由 thread 0 或 CTA leader load 一次，放在 register 或 shared scalar，後面 dot/update 都重用，不需要反覆從 global memory 讀。

   $$
   \tau_j = \tau_{\text{batch},j},
   \qquad
   w = \tau_j(v^Tx).
   $$

3. $Hx = x-\tau v(v^Tx)$ 可以完全 GEMV 化

   對單一 reflector 來說，每個 target column `x` 都是同一個 GEMV-like operation：

   $$
   \begin{aligned}
   d_c &= v^T x_c, \\
   x_c &\leftarrow x_c-\tau v d_c .
   \end{aligned}
   $$

   對一組 trailing columns `C`，可視為：

   $$
   \begin{aligned}
   w &= \tau v^T C, \\
   C &\leftarrow C - vw .
   \end{aligned}
   $$

   這代表 Part 5 不必先形成 full Householder matrix，也不必顯式形成 `H_j`。可以用 tile-based GEMV/rank-1 update：CTA 處理一批 columns，先對每個 column 做 reduction 得到 `dot_col`，再用 shared `v_tile` 更新 column tile。

Fusion 目標：

$$
\begin{aligned}
d_{\text{tile}} &= v_j^T C_{\text{tile}}, \\
C_{\text{tile}} &\leftarrow
C_{\text{tile}}-\tau_jv_jd_{\text{tile}} .
\end{aligned}
$$

其中 $v_j$ 從 compact storage stream 到 shared memory/registers，$\tau_j$ 只載入一次並在 tile 內重用。

先前 `apply_reflector_cols_gemv_cuda` 使用這個設計：每個 `(batch, target column)` 對應一個 CTA，128 threads 對 `v^T x` 做 parallel reduction，並平行更新 column rows。profiling 後確認它仍被 per-reflector launch 架構拖垮，因此主流程已改成 panel-level WY。

## Part 5: Trailing Matrix Update Kernels

Backend: 目前主流程使用 CuTe DSL compact WY kernel。下一步要把目前 per-column CTA WY kernel 改成 multi-row/multi-column tiled WY kernel。

數學函式：

$$
C \leftarrow (I - VTV^T)C = C - VT(V^TC).
$$

對應目前程式碼中的 WY target：

$$
\begin{aligned}
W &\leftarrow V^TC, \\
W &\leftarrow TW, \\
C &\leftarrow C - VW .
\end{aligned}
$$

職責：

- 把整個 panel 套用到 `H[:, j_start:, j_end:]`。
- 使用 compact WY 形式：$C \leftarrow C - VT(V^TC)$。

實作狀態：

- 目前主流程：用 compact WY update：
  - $W \leftarrow V^TC$
  - $W \leftarrow TW$
  - $C \leftarrow C - VW$
- 目前 kernel 粒度：每個 `(batch, trailing column)` 一個 CTA。
- 下一步 kernel 粒度：切成 `(batch, row_tile, col_tile)`，讓 tile 內重用 `V/T`。
- 小/中型 `n`：可以 fuse 部分 update，降低 global memory traffic。
- 大型 `n`：讓 GEMM-like tiling 主導；這是 CuTe layout 和 MMA support 最值得發揮的地方。

這是整個演算法 arithmetic intensity 最高的區域，但 WY order 必須完全符合 compact Householder convention。目前已驗證 reverse-order lower-triangular `T` 的 WY order。

## Validation, Benchmarking, And Fallback

Backend: GPU validation/benchmarking 用 Torch CUDA；Torch CPU 只做 no-GPU fallback 或 debug path。

數學函式：

$$
\begin{aligned}
Q &= \operatorname{householderProduct}(H,\tau), \\
R &= \operatorname{triu}(H), \\
r_{\text{factor}} &= \lVert R-Q^TA\rVert_1, \\
r_{\text{orth}} &= \lVert Q^TQ-I\rVert_1, \\
r_{\text{recon}} &= \lVert QR-A\rVert_1, \\
r_{\text{tri}} &= \lVert \operatorname{lower}(Q^TA)\rVert_1 .
\end{aligned}
$$

接受條件：

$$
r_{\text{factor}}\le \epsilon_{\text{factor}},
\qquad
r_{\text{orth}}\le \epsilon_{\text{orth}} .
$$

`qr_official.py` 主要用 factorization 與 orthogonality 作 hard failure gate，並回報 reconstruction 與 triangularity scaled residual 作診斷。

職責：

- 用 `qr_official.py` 的 official checker 驗證。
- 覆蓋所有 official input cases：
  - `dense`
  - `upper`
  - `diagonal`
  - `rankdef`
  - `nearrank`
  - `clustered`
  - `band`
  - `nearcollinear`
  - `rowscale`
- 依 component benchmark：
  - panel factorization
  - panel internal update
  - `T` build
  - trailing update
  - total QR
- 對 unsupported device、dtype、shape，或 CuTe DSL import failure，保留 `torch.geqrf` fallback。

## 建議建置順序

1. 修正目前 PyTorch 實作中的 Householder sign bug。
2. 加上 official-style correctness test wrapper，呼叫 `check_implementation`。
3. 已完成：Part 2 sigma reduction 與 Part 3 panel-internal reflector application 已 fuse 成一個真正的 CuTe DSL panel kernel。
4. Part 4 先保留 Torch CUDA；Part 5 目前只保留 compact WY CuTe DSL kernel。
5. 已完成：Part 2 使用 parallel shared-memory sigma reduction，並和 Part 3 fuse 到 `part2_3_factor_apply_panel_cuda`。
6. 下一步：把 Part 4 移到 CuTe DSL。
7. 下一步優化：根據 ncu 結果，把 Part 5 WY 從 per-column CTA 改成真正 tiled WY kernel。
8. 依 `n`、batch size、GPU architecture 調整 `nb`。

## 實際 Kernel 邊界

目前最務實的 MVP 是每個 panel 有 3 個 runtime stages：

1. `part2_3_factor_apply_panel_cuda`: fused CuTe DSL panel kernel，包含 parallel sigma reduction 與 panel-internal apply。
2. `build_compact_wy_t_torch`: 先用 Torch CUDA 建立 reverse-order lower-triangular `T`。
3. `part5_apply_panel_wy_gemv_cuda`: 用 CuTe DSL compact WY kernel 做 trailing update：
   $C \leftarrow C - VT(V^TC)$。
4. Python loop dispatch: 前進到下一個 panel。

目前已移除主流程中的 `gemv`、`gemv4`、`single_thread`、`wy_cutedsl4` variants；保留單一路徑可以讓 profiling 結果更乾淨。

## Profiling: Part 5 三種做法歷史紀錄

Profiling script: `profile_qr_variants.py`

執行方式：

```bash
source /workspace/cutedsl_env.sh
python -u profile_qr_variants.py
```

測試環境：早期同一份 `householder_qr_blocked(..., backend="cutedsl")`，只切換已移除的 `trailing_variant`。時間是 ms per QR，已做少量 warmup。`wy_cutedsl` 已使用修正後的 reverse-order lower-triangular `T`，並通過 correctness。

| batch | n | nb | 方法 | correctness | 時間 |
| --- | --- | --- | --- | --- | --- |
| 2 | 32 | 8 | `torch.geqrf` baseline | verified | 2.549 ms |
| 2 | 32 | 8 | `gemv`: per-column CTA, 128-thread GEMV | rel=3.488e-07 | 2002.744 ms |
| 2 | 32 | 8 | `gemv4`: 4-column tiled CTA, 128-thread GEMV | rel=3.488e-07 | 5457.196 ms |
| 2 | 32 | 8 | `wy_cutedsl`: verified WY-GEMV CuTe DSL kernel | rel=3.628e-07 | 597.486 ms |
| 4 | 64 | 16 | `torch.geqrf` baseline | verified | 3.504 ms |
| 4 | 64 | 16 | `gemv`: per-column CTA, 128-thread GEMV | rel=4.845e-07 | 4105.647 ms |
| 4 | 64 | 16 | `gemv4`: 4-column tiled CTA, 128-thread GEMV | rel=4.845e-07 | 11862.999 ms |
| 4 | 64 | 16 | `wy_cutedsl`: verified WY-GEMV CuTe DSL kernel | rel=5.133e-07 | 688.772 ms |

解讀：

- `gemv` 和 `gemv4` 都通過 correctness，但仍被 per-reflector launch 架構拖垮。
- `gemv4` 嘗試一個 CTA 處理 4 個 columns，但 4 組 partial reductions 帶來更多 shared memory 與 barrier 成本，因此比 `gemv` 更慢。
- `wy_cutedsl` 是目前三者中最快的 custom path，因為它使用 panel-level WY：$C \leftarrow C - VT(V^TC)$，把 trailing update 從每個 reflector 多次 launch 壓成每個 panel 一次 WY application。
- `wy_cutedsl` 仍不是最終高效 tiled GEMM kernel。它目前是每個 `(batch, trailing column)` 一個 CTA 的 WY-GEMV correctness kernel。
- 下一步真正值得做的是把 `wy_cutedsl` 從 per-column CTA 擴成 multi-column tiled WY kernel，讓一個 CTA/CTA group 同時處理多個 columns，並重用 shared `V/T` tile。

## Profiling: WY-GEMV4 後續測試

新增 variant:

- `wy_cutedsl4`: 4-column tiled verified WY-GEMV CuTe DSL kernel

這個版本把 `wy_cutedsl` 的 per-column CTA 擴成一個 CTA 同時處理 4 個 trailing columns。Correctness 通過，但目前效能比 `wy_cutedsl` 慢。

| batch | n | nb | 方法 | correctness | 時間 |
| --- | --- | --- | --- | --- | --- |
| 2 | 32 | 8 | `torch.geqrf` baseline | verified | 2.574 ms |
| 2 | 32 | 8 | `gemv`: per-column CTA, 128-thread GEMV | rel=3.488e-07 | 1998.277 ms |
| 2 | 32 | 8 | `gemv4`: 4-column tiled CTA, 128-thread GEMV | rel=3.488e-07 | 5479.305 ms |
| 2 | 32 | 8 | `wy_cutedsl`: verified WY-GEMV CuTe DSL kernel | rel=3.628e-07 | 600.925 ms |
| 2 | 32 | 8 | `wy_cutedsl4`: 4-column tiled verified WY-GEMV | rel=3.628e-07 | 1122.378 ms |
| 4 | 64 | 16 | `torch.geqrf` baseline | verified | 3.509 ms |
| 4 | 64 | 16 | `gemv`: per-column CTA, 128-thread GEMV | rel=4.845e-07 | 4157.937 ms |
| 4 | 64 | 16 | `gemv4`: 4-column tiled CTA, 128-thread GEMV | rel=4.845e-07 | 11897.860 ms |
| 4 | 64 | 16 | `wy_cutedsl`: verified WY-GEMV CuTe DSL kernel | rel=5.133e-07 | 691.318 ms |
| 4 | 64 | 16 | `wy_cutedsl4`: 4-column tiled verified WY-GEMV | rel=5.133e-07 | 1258.434 ms |

解讀：

- `wy_cutedsl4` correctness 通過，但不是 performance win。
- 直接把 columns unroll 成 4 會減少 CTA 數，但每個 CTA 需要 4 套 `y/z/partial` buffers 和更多 shared-memory traffic/barriers。
- 目前瓶頸不是單純 CTA 數，而是 reduction/shared-memory/barrier 結構。
- 下一步應該改成更接近 GEMM 的 tiled WY：一個 CTA/warp group 處理 `(rows tile, cols tile)`，把 `V/T` tile 重用在多個 columns 上，而不是在單 CTA 裡複製 4 套 scalar GEMV pipeline。

## Profiling: WY-only + ncu

目前主流程只保留 WY 做法：

- Panel: `part2_3_factor_apply_panel_cuda`
- Part 4: `build_compact_wy_t_torch`
- Trailing update: `part5_apply_panel_wy_gemv_cuda`

一般 timing：

```bash
source /workspace/cutedsl_env.sh
python -u profile_qr_variants.py
```

| batch | n | nb | 方法 | correctness | 時間 |
| --- | --- | --- | --- | --- | --- |
| 2 | 32 | 8 | `torch.geqrf` baseline | verified | 2.540 ms |
| 2 | 32 | 8 | `wy_cutedsl` only | rel=3.628e-07 | 559.005 ms |
| 4 | 64 | 16 | `torch.geqrf` baseline | verified | 1.402 ms |
| 4 | 64 | 16 | `wy_cutedsl` only | rel=5.133e-07 | 560.280 ms |

ncu 指令：

```bash
source /workspace/cutedsl_env.sh
ncu --target-processes all --set roofline \
  --kernel-name 'regex:.*(part2_3_factor_apply_panel|part5_apply_panel_wy).*' \
  python -u profile_qr_variants.py --ncu --batch 4 --n 64 --nb 16

ncu --target-processes all \
  --section LaunchStats --section Occupancy --section SchedulerStats --section WarpStateStats \
  --kernel-name 'regex:.*part5_apply_panel_wy.*' \
  python -u profile_qr_variants.py --ncu --batch 4 --n 64 --nb 16
```

ncu 重點結果，`batch=4, n=64, nb=16`：

| kernel | launch grid | duration | achieved occupancy | main observation |
| --- | --- | --- | --- | --- |
| `part2_3_factor_apply_panel` | `(4, 1, 1)` blocks | 25.5-58.0 us | 8.33% | grid 遠小於 188 SM，barrier stall 約 57-66% |
| `part5_apply_panel_wy_gemv` first trailing panel | `(4, 48, 1)` blocks | 22.2 us | 8.27% | only 0.10 waves/SM，barrier stall 約 50% |
| `part5_apply_panel_wy_gemv` second trailing panel | `(4, 32, 1)` blocks | 21.8 us | 8.15% | grid 128 blocks，小於 188 SM |
| `part5_apply_panel_wy_gemv` third trailing panel | `(4, 16, 1)` blocks | 21.3 us | 8.08% | grid 64 blocks，小於 188 SM |

解讀：

- 目前不是 DRAM bandwidth-bound，也不是 FP32 throughput-bound；roofline 顯示 FP32/DRAM 都遠低於峰值。
- 真正瓶頸是 launch shape 太小與 CTA 內 barrier-heavy reduction。WY kernel 每個 `(batch, trailing column)` 一個 CTA，對 RTX PRO 6000 Blackwell 的 188 SM 來說平行度不夠。
- `part5_apply_panel_wy_gemv` 的 scheduler 指標顯示 `No Eligible` 約 95%，`Issued Warp Per Scheduler` 約 0.04-0.05，代表大量時間沒有 ready warp 可發射。
- `part2_3_factor_apply_panel` 更嚴重：每個 panel 只有 `batch` 個 blocks，目前 batch=4 時幾乎無法填滿 GPU。

接下來的優化方向：

1. 優先把 WY trailing update 改成真正 tiled WY：grid 應該切成 `(batch, row_tile, col_tile)`，不是只切 `(batch, col)`。
2. 在 tile 內把 `V` 與 `T` 載入 shared memory，讓多個 columns 共用同一份 panel data。
3. 把 `V^T C`、`T y`、`V z` 拆成 warp-level 或 CTA tile-level reductions，減少每個 panel column 都做全 CTA `sync_threads()` reduction。
4. Panel kernel 也需要重新切 row/block parallelism；目前一個 batch 一個 CTA 太小，適合正確性驗證，不適合作為最終高效 kernel。
5. 在做 tiled WY 前，不建議再做 `wy_cutedsl8` 這類 scalar GEMV unroll，因為 ncu 已經指出主要問題是 eligible warp/occupancy/barrier，而不是單純 CTA 數。

## CUDA Tiled WY 評估

新增 raw CUDA backend：

- `backend="cutedsl"`：Panel 用 `part2_3_factor_apply_panel_cuda`，Part 5 用 CuTe DSL `part5_apply_panel_wy_gemv_cuda`。
- `backend="cuda"`：Panel 仍用同一個 `part2_3_factor_apply_panel_cuda`，Part 5 改用 raw CUDA tiled WY。

raw CUDA tiled WY 位於 `cuda_wy_kernels.py`，分成三個 kernels：

1. `compute_y_partial_kernel`
   - grid: `(trailing_col, batch, panel_col * row_tile)`
   - 數學：分 row tile 計算 $Y_{\text{partial}} = V_{\text{tile}}^T C_{\text{tile}}$
   - 目的：把原本每個 column 一個 CTA 掃完整 rows，改成 row-tiled parallel reduction。
2. `reduce_y_compute_z_kernel`
   - grid: `(trailing_col, batch)`
   - 數學：$Y=\operatorname{reduce}(Y_{\text{partial}})$，接著 $Z=TY$
   - 目前瓶頸：grid 仍偏小，而且每個 CTA 內仍有 shared-memory reduction。
3. `update_c_tiled_kernel`
   - grid: `(col_tile, row_tile, batch)`
   - 數學：$C_{\text{tile}}\leftarrow C_{\text{tile}}-V_{\text{tile}}Z_{\text{tile}}$
   - 這是目前真正 row/column tiled 的 update stage。

另有一個 fused shared-memory fast path：

- `apply_panel_wy_fused_tile_kernel`
- 啟用條件：`active_rows <= 128`
- grid: `(trailing_col_tile, batch)`
- 做法：

$$
\begin{aligned}
C_{\text{tile}},V_{\text{tile}},T
&\rightarrow \text{shared memory}, \\
Y_{\text{tile}} &= V_{\text{tile}}^T C_{\text{tile}}, \\
Z_{\text{tile}} &= TY_{\text{tile}}, \\
C_{\text{tile}} &\leftarrow C_{\text{tile}}-V_{\text{tile}}Z_{\text{tile}} .
\end{aligned}
$$

這個版本符合「一次把 tile 搬進 shared memory，做更多計算再寫回」的方向，並且避免 `Y_partial` / `Z` global-memory intermediate。限制是 grid 會變得更小，而且 shared-memory footprint 較大。

一般 timing：

```bash
source /workspace/cutedsl_env.sh
python -u profile_qr_variants.py
```

| batch | n | nb | backend | correctness | 時間 |
| --- | --- | --- | --- | --- | --- |
| 2 | 32 | 8 | `torch.geqrf` baseline | verified | 2.546 ms |
| 2 | 32 | 8 | `cutedsl`: CuTe DSL per-column WY-GEMV | rel=3.628e-07 | 562.272 ms |
| 2 | 32 | 8 | `cuda`: raw CUDA row/column tiled WY | rel=3.628e-07 | 319.255 ms |
| 4 | 64 | 16 | `torch.geqrf` baseline | verified | 3.492 ms |
| 4 | 64 | 16 | `cutedsl`: CuTe DSL per-column WY-GEMV | rel=5.133e-07 | 563.245 ms |
| 4 | 64 | 16 | `cuda`: raw CUDA row/column tiled WY | rel=5.133e-07 | 320.278 ms |

ncu 指令：

```bash
source /workspace/cutedsl_env.sh
ncu --target-processes all \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --kernel-name 'regex:.*(compute_y_partial|reduce_y_compute_z|update_c_tiled).*' \
  python -u profile_qr_variants.py --ncu --backend cuda --batch 4 --n 64 --nb 16
```

ncu 重點結果，`batch=4, n=64, nb=16`：

| stage | first trailing panel grid | duration | achieved occupancy | observation |
| --- | --- | --- | --- | --- |
| `compute_y_partial_kernel` | `(48, 4, 16)` blocks | 6.62 us | 78.21% | 平行度明顯比 CuTe per-column WY 好，是目前最健康的一段 |
| `reduce_y_compute_z_kernel` | `(48, 4, 1)` blocks | 24.64 us | 5.81% | grid 仍小，且 reduction/barrier 成本高 |
| `update_c_tiled_kernel` | `(3, 4, 4)` blocks | 7.81 us | 16.55% | 已 row/col tiled，但小矩陣下 blocks 仍少 |
| `apply_panel_wy_fused_tile_kernel` | `(3, 4, 1)` blocks | 10.05 us | 16.56% | shared-memory fused fast path，少 global intermediate，但 grid 太小且 shared memory 限制 occupancy |

評估：

- raw CUDA backend 的總 QR wall time 比 CuTe DSL path 短，`n=64` 約從 `563 ms` 降到 `320 ms`。
- 但若只看 Part 5 的 kernel elapsed time，raw CUDA 三個 kernels 加總不一定比 CuTe DSL 單一 `part5_apply_panel_wy_gemv` kernel 更短。改善主要來自：
  - raw CUDA extension 一次 Python call 內 launch 三個 kernels，host/runtime overhead 比多次 CuTe DSL dispatch 更低。
  - `compute_y_partial_kernel` 的 grid 大很多，occupancy 從 CuTe WY 的約 8% 提高到約 78%。
- 目前新的瓶頸移到 `reduce_y_compute_z_kernel`，它仍是 per `(batch, trailing_col)` CTA，occupancy 約 5-6%。
- fused shared-memory fast path correctness 通過，但總時間沒有明顯優於三階段 CUDA path。ncu 顯示它的 first trailing panel 只有 12 blocks，遠小於 188 SM；同時 shared memory 約 32 KiB/block，使 theoretical occupancy 限制在 50%。結論是：搬進 shared memory 本身是對的，但 tile decomposition 必須保留足夠 grid parallelism。

接下來的 CUDA 優化方向：

1. 優先 fuse `reduce_y_compute_z_kernel` 和 `update_c_tiled_kernel`，或讓 update kernel 直接讀 `Y_partial` 做 tile-local reduction，降低一次 kernel launch 和 global `Z` traffic。
2. 把 `reduce_y_compute_z_kernel` 從 per-column CTA 改成 column-tile CTA，讓一個 CTA 同時處理多個 columns 的 `Y/Z`。
3. 對 `nb <= 16/32` 使用 warp-level reduction (`__shfl_down_sync`) 取代 shared-memory tree reduction，減少 `__syncthreads()`。
4. 對 `n=64` 這種小矩陣，update stage blocks 仍少；可以把 row tile 切更細，或讓一個 CTA 只處理較小 `(row, col)` tile 以增加 grid。
5. Panel kernel 仍是最大架構限制之一：`part2_3_factor_apply_panel_cuda` 每個 batch 一個 CTA，對 188 SM GPU 太小。Part 5 改善後，下一步要把 panel factor/apply 也拆出更多 row/block parallelism。
