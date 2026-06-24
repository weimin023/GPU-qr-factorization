# CuTe DSL QR 分解計畫

## 最新狀態摘要（2026-06-22）

> 本節是目前版本的 authoritative snapshot。`最近進展（2026-06-16）` 之後保留
> 實驗歷史，其中部分 Torch/CuTe 主線、效能數字與下一步判斷已被目前版本取代。

### 現行 submission 架構

目前 `submission.py` 的 active path：

1. `n <= 64`：small-square shared-memory raw CUDA kernel
2. `n > 64`：blocked Householder QR，預設 `nb=64`
3. panel：raw CUDA `panel_factor_apply`
4. compact WY `T`：raw CUDA two-stage build，`build_t_dot_kernel` 先平行計算 reflector dot，`build_t_finish_kernel` 再完成 triangular recurrence
5. trailing update：IEEE FP32 `torch.bmm`/cuBLAS，計算 `C -= V @ (T @ (V^T @ C))`
6. large raw CUDA reduction：warp-shuffle 加 8 個 shared warp partials
7. 舊 raw CUDA WY kernels 仍保留作 profiling/fallback reference，不在 active path

目前 backend 標記：

```text
raw-cuda-small-square+raw-cuda-panel-fused+raw-cuda-t+fp32-gemm-wy
```

### 2026-06-22 最新更新：two-stage build-T 已整合並遠端通過

這次根據 NCU 結論，把原本每個 batch 只有一個 CTA、在 `jj/prev` 內序列做 dot reduction 的
`build_t_kernel`，改成兩階段：

1. `build_t_dot_kernel`：grid = `(pair_count, batch)`，每個 `(jj, prev)` dot 用一個 CTA 平行 reduction，結果寫入 `dot_ws(batch, nb, nb)`。
2. `build_t_finish_kernel`：保留原本 compact WY lower-triangular recurrence，從 `dot_ws` 讀 dot 後建立 `T`。
3. Python/C++ wrapper 介面不變，只在 C++ extension wrapper 內配置 `dot_ws`。
4. 精度維持 IEEE FP32；沒有引入 TF32/FP8/NVFP4。

submission `827457`：

- B200 `qr_v2`
- public/secret test：passed
- public/secret benchmark：passed
- public/secret leaderboard：passed
- CLI 等待時間：約 `280s`
- 無 timeout

本地 remote-style 19 cases 全部通過。相對上一版 FP32 GEMM-WY baseline `827405`：

| shape | FP32 GEMM-WY baseline | two-stage build-T | speedup |
| --- | ---: | ---: | ---: |
| `batch=40,n=176` | `4.646 ms` | `3.493 ms` | `1.33x` |
| `batch=40,n=352` | `14.517 ms` | `9.804 ms` | `1.48x` |
| `batch=16,n=512` | `18.861 ms` | `10.920 ms` | `1.73x` |
| `batch=4,n=1024` | `49.321 ms` | `18.510 ms` | `2.66x` |
| `batch=2,n=2048` | `160.734 ms` | `42.295 ms` | `3.80x` |
| `batch=1,n=4096` | `463.575 ms` | `105.438 ms` | `4.40x` |

### 目前是否已有 true GEMM

有，但範圍很明確：目前 active path 的 trailing WY update 已經是 true GEMM，透過
`torch.bmm`/cuBLAS 以 IEEE FP32 執行三段 batched GEMM：

```text
Y = V^T @ C
Z = T @ Y
C = C - V @ Z
```

這不是手寫 CuTe/CUTLASS GEMM kernel，也不是 Tensor Core FP8/TF32 路徑。先前測過 TF32，對目前
`nb=64` skinny GEMM 幾乎沒有速度收益且 band/rowscale correctness 失敗；PyTorch batched FP8
`bmm` 在目前環境也不支援。因此 active GEMM 是「cuBLAS true GEMM + IEEE FP32」，而 panel
factor/apply 與 build-T recurrence 仍是手寫 raw CUDA，不是 GEMM。

### 本輪完成項目

#### 1. warp-shuffle raw baseline 已遠端通過

submission `827385` 使用 raw CUDA WY 與 large warp-shuffle reduction：

- B200 `qr_v2`
- public/secret test：passed
- public/secret benchmark：passed
- public/secret leaderboard：passed
- CLI 等待時間：約 `210s`
- 無 timeout

large `block_sum` 從 256-thread shared tree 改成 warp reduction 後，本地端到端約提升
`1.11x-1.17x`。small-square 套用相同改法反而慢 `4-5%`，因此 small path 已回退。

#### 2. raw CUDA stage breakdown

工具：`profile_raw_stages.py --wy-backend raw`

| shape | panel | build T | raw WY | total |
| --- | ---: | ---: | ---: | ---: |
| `batch=16,n=512` | `8.011 ms` (26.6%) | `10.457 ms` (34.7%) | `11.659 ms` (38.7%) | `30.160 ms` |
| `batch=4,n=1024` | `13.035 ms` (17.7%) | `35.520 ms` (48.3%) | `24.897 ms` (33.9%) | `73.513 ms` |
| `batch=2,n=2048` | `30.092 ms` (11.0%) | `128.268 ms` (46.9%) | `115.233 ms` (42.1%) | `273.620 ms` |
| `batch=1,n=4096` | `79.685 ms` (8.0%) | `378.870 ms` (38.0%) | `537.905 ms` (54.0%) | `996.658 ms` |

此結果確認 raw WY 是大矩陣首要或第二瓶頸，值得改成 GEMM 路徑。

#### 3. 低精度 Tensor Core 實驗結論

隔離實驗：`submission_tensorcore.py`。

- E4M3 `torch.bmm`：目前 PyTorch `2.12.0+cu130` 回報
  `NotImplementedError: baddbmm_cuda not implemented for Float8_e4m3fn`
- FP8 `_scaled_mm` 只有 2D 路徑；逐 batch Python launch 不適合目前 workload
- TF32 GEMM：`n=512` 約 `18.77 ms`，但 band 與 rowscale correctness 失敗
- TF32 band residual：`0.0771 > 0.0483`
- TF32 rowscale residual：`0.0958 > 0.0697`
- IEEE FP32 GEMM：`n=512` 約 `18.89 ms`，與 TF32 幾乎沒有速度差，19 cases 全通過

因此不使用 TF32/FP8。這組 `nb=64` skinny GEMM 在本機沒有低精度收益，加入 FP32
residual correction只會增加額外 GEMM；目前最佳選擇是直接使用 IEEE FP32 cuBLAS。

直接量化 `y_partial` 的舊實驗也維持否決：FP8 E4M3 scaled residual `736`，NVFP4
E2M1 scaled residual `2.76e3`。

#### 4. FP32 GEMM-WY 已整合並遠端通過

submission `827405`：

- B200 `qr_v2`
- public/secret test：passed
- public/secret benchmark：passed
- public/secret leaderboard：passed
- CLI 等待時間：約 `140s`
- 無 timeout

本地 remote-style 19 cases 全部通過。相對 submission `827385` 的 raw WY baseline：

| shape | raw WY baseline | FP32 GEMM-WY | speedup |
| --- | ---: | ---: | ---: |
| `batch=40,n=176` | `5.747 ms` | `4.646 ms` | `1.24x` |
| `batch=40,n=352` | `25.086 ms` | `14.517 ms` | `1.73x` |
| `batch=16,n=512` | `30.133 ms` | `18.861 ms` | `1.60x` |
| `batch=4,n=1024` | `73.365 ms` | `49.321 ms` | `1.49x` |
| `batch=2,n=2048` | `273.299 ms` | `160.734 ms` | `1.70x` |
| `batch=1,n=4096` | `996.239 ms` | `463.575 ms` | `2.15x` |

新路徑也移除了大尺寸 `y_workspace`；`V` 目前在每個 panel 以 Torch tensor ops materialize。

### 最新 stage breakdown

工具：`profile_raw_stages.py --wy-backend gemm`

| shape | panel | build T | GEMM-WY | total |
| --- | ---: | ---: | ---: | ---: |
| `batch=16,n=512` | 約 `8.0 ms` | 約 `2.4-2.5 ms` | 約 `0.38 ms` | 約 `10.9 ms` |
| `batch=4,n=1024` | `13.034 ms` (70.1%) | `4.608 ms` (24.8%) | `0.886 ms` (4.8%) | `18.591 ms` |
| `batch=2,n=2048` | `30.081 ms` (70.9%) | `9.732 ms` (22.9%) | `2.522 ms` (5.9%) | `42.454 ms` |
| `batch=1,n=4096` | `79.954 ms` (75.7%) | `20.188 ms` (19.1%) | `5.322 ms` (5.0%) | `105.613 ms` |

two-stage build-T 已把 `build T` 從原本 `55-82%` 降到約 `19-25%`。目前主要瓶頸轉移到
raw CUDA panel factor/apply，約占 `70-76%`；GEMM-WY 仍只有約 `5%` 以內。

## 下一步優化方向（更新後優先順序）

### P0：重做 raw CUDA panel factor/apply

目前 stage profile 顯示 panel 已是最大 hot path：`n=1024-4096` 約占 `70-76%`。下一輪不應再優先投入
GEMM-WY 或低精度，而應集中在 panel：

1. 減少 per-column launch：目前每個 panel column 仍包含 sigma partial、finalize/scale、target apply 多段 launch。
2. 改善 `finalize_scale_kernel`：目前仍接近 one CTA per batch，對 B200 這種 188 SM GPU 平行度太低。
3. 重寫 `apply_target_fused_kernel` ownership：從「每 target column 一 CTA」走向 `(target column, row tile)` 或 multi-target/multi-row tiled ownership。
4. 降低同步與 long scoreboard：優先用 warp-level reduction、row tile partial reduction、shared/register staging reflector。
5. 對大矩陣保留 correctness-first 策略；panel 直接影響 Householder reflector，數值風險比 trailing GEMM 更高。

### P1：進一步壓縮 build-T finish recurrence

`build_t_dot_kernel` 已解掉最大序列 dot bottleneck，但 `build_t_finish_kernel` 仍是一個 batch 一個 CTA 的 triangular recurrence。它現在不是第一瓶頸，但仍占 `19-25%`：

1. 先 profile `build_t_dot_kernel` 與 `build_t_finish_kernel` 分別占比。
2. 若 finish 占比高，將同一 `jj` 的 `col` update 改成更寬的 warp/CTA partition。
3. 評估 `nb=96/128` 之前，先確認 finish recurrence 不會因 `nb^3` 成本放大。

### P2：shape-aware `nb` sweep

GEMM-WY 已很便宜，two-stage build-T 也大幅降低 T build 成本，因此 `nb=96/128` 值得重新測一次。
較大 `nb` 可能減少 panel 次數，但會增加 T recurrence 和 panel 內工作量；應用實測決定：

```text
nb in {32, 48, 64, 96, 128}
```

### P3：true GEMM 路徑維持 IEEE FP32 cuBLAS

目前已有 true GEMM，但只用在 trailing WY update。短期不建議改：

1. cuBLAS `torch.bmm` 的 GEMM-WY 已只占約 `5%` 以內。
2. TF32 先前 correctness 失敗且速度沒有收益。
3. FP8/NVFP4 對 Householder QR residual 不穩，目前 PyTorch batched FP8 `bmm` 也不可用。
4. 若 panel/build-T 再大幅下降，才重新評估 custom CUTLASS/CuTe true GEMM 或 fused GEMM-WY materialization。

### P4：降低 V materialization overhead

GEMM-WY 目前會每個 panel 用 Torch tensor ops materialize unit-lower `V`。現在不是 hot path，但若 panel 優化後
GEMM-WY 比例上升，可用小 CUDA kernel fuse `clone + tril + eye + assignment`。

### P5：合併 inline extensions

small-square 與 panel 仍是兩個 `load_inline` extension。合併可降低遠端 JIT wall-clock 與 timeout 風險，但不直接改善 steady-state benchmark，因此排在 compute hot path 後。

### 每輪驗收門檻

1. Python syntax與 CUDA extension compilation
2. remote-style 19 cases全部通過
3. 修改前後受控 A/B benchmark
4. stage profile確認收益命中預期 hot path
5. B200 public/secret leaderboard submission不 timeout

## 最近進展（2026-06-16）

### 已完成：panel kernel 的 sigma reduction 改成 warp-local 為主

目前 `submission.py` 的 `_part2_3_factor_apply_panel_kernel` 已將 `sigma` reduction
從傳統的 block-wide shared-memory tree reduction，改成：

1. 每個 warp 先用 `cute.arch.shuffle_sync_up(...)` 做 warp 內累加
2. 每個 warp 只由最後一個 lane 寫一個 partial sum 到 shared memory
3. 由 warp 0 再收斂 4 個 warp 的 partial sums
4. 最後只保留必要的 block-wide `cute.arch.sync_threads()`

對應程式位置：

- [submission.py](/modelz/workspace/weiminc/qr/GPU-qr-factorization/submission.py:36)
- [submission.py](/modelz/workspace/weiminc/qr/GPU-qr-factorization/submission.py:54)

這次改動的目的，是先把最密集、風險最低的一段同步熱點換成 warp-local 為主的型態，而不碰 reflector apply 的正確性敏感區。

### 量測結果

在現有 `.venv` 與 `source ./env.sh` 的環境下，本地 correctness 仍然通過：

- `python verify_submission.py --module submission --trials 1`

代表性效能量測：

- case: dense
- `n=512, batch=16`
- 量測對象：第一個 panel block 的 `_panel_factor_apply_cutedsl_mvp(...)`

結果：

- 修改前：約 `81.0 ms`
- 修改後：約 `75.3 ms`

大約是 `7%` 左右的 panel kernel 改善。

### 已嘗試但暫不保留的方向

#### 1. 直接把 apply reflector 改成 warp-parallel

做法：

- 每個 warp 負責一個 target column
- warp 內 threads 分攤 rows
- 用 shared `partial` 做每個 warp 的 reduction

結果：

- correctness 可通過
- 但 `panel` 變慢到約 `129 ms`

原因判斷：

- 雖然 thread 分工變合理，但 global memory 仍然是 column-wise stride access
- reduction 仍搭配大量 block-wide `sync_threads()`
- 同步成本吃掉了 row-parallel 的好處

#### 2. 在 apply reflector 中加入 shared-memory tiling

做法：

- 每輪把 `v` column 與 4 個 target columns 搬進 shared memory
- 在 shared memory 上做 dot 與 update

結果：

- correctness 壞掉，曾出現大幅 factorization residual
- 就算暫時不看 correctness，`panel` 也惡化到約 `239 ms`

原因判斷：

- 現有 kernel 的 phase 切分不適合直接疊加大量 block-wide barrier
- shared-memory staging 若仍由全 block 在每輪 target 上同步，成本太高
- 這類改法已不是「局部修補」等級，需要重設 kernel 結構

上述兩個版本都已回退；目前保留的是 correctness 正常、效能有小幅正向改善的 warp-local `sigma` reduction 版本。

### 對目前瓶頸的更新判斷

目前可以更具體地說：

1. `sigma` reduction 的 block-wide synchronization 確實有可收的 overhead
2. 但 panel kernel 最大問題不只是 reduction，而是 `apply reflector` 的資料流與同步模型
3. 單純把 `apply reflector` 局部改成 warp-parallel，不足以自然變快
4. 若想在 panel kernel 再拿顯著收益，必須同時重設：
   - thread/warp work partition
   - global load pattern
   - shared-memory usage
   - synchronization boundaries

### 下一步建議

目前建議的優先順序如下：

1. **保留現有 warp-local sigma reduction**
   - 這是已驗證可保留的正向改動

2. **不要再局部硬改 apply reflector**
   - 目前 evidence 顯示這樣容易變成「更多同步 + 同樣差的 global access」

3. **優先評估 trailing update 改成 `torch.bmm` / cuBLAS 路線**
   - 先前量測顯示 trailing update 與 panel 同量級，不是次要問題
   - 若規則允許，直接交給成熟 GEMM backend，可能比手刻 CuTe kernel 更快拿到大幅收益

4. **panel 若要繼續優化，應走完整重設而非小修**
   - 方向應是 warp-local 為主、減少 block-wide barriers
   - 盡量避免同一段 shared memory 同時扮演太多角色
   - 需要重新設計 `apply reflector` 的 tile 與 ownership，而不是直接在現有 loop 上疊 shared-memory buffer

5. **`build T` 仍然是明確可優化點**
   - 目前 `block=(1,1,1)`，仍屬低效率實作
   - 但在 panel 結構未重設之前，它不是最該先砍的一刀

## 最近進展（2026-06-17）

### 已確認：trailing update 主線已切到 `torch.bmm`

`householder_qr_blocked(...)` 目前主線在 panel factorization 與 `T` 建立之後，已經使用 batched GEMM 形式的 trailing update：

```text
C <- C - V @ (T @ (V^T @ C))
```

對應程式位置：

- [submission.py](/modelz/workspace/weiminc/qr/GPU-qr-factorization/submission.py:501)
- [submission.py](/modelz/workspace/weiminc/qr/GPU-qr-factorization/submission.py:510)

也就是說，`docs` 中先前建議的「優先評估 trailing update 改成 `torch.bmm` / cuBLAS 路線」已經落地，而且目前就是 submission 主線。

### 舊 CuTe trailing kernel vs `torch.bmm` 實測

我用同一份 panel factorization 結果與同一個 `T` factor，直接比較：

- 舊路徑：`part5_apply_panel_wy_fused_update_cuda(...)`
- 新路徑：`torch.bmm` 版 `C <- C - V @ (T @ (V^T @ C))`

代表性量測：

| shape | CuTe trailing kernel | `torch.bmm` trailing | speedup |
| --- | ---: | ---: | ---: |
| `batch=4, n=128` | 約 `109.95 ms` | 約 `0.159 ms` | 約 `690x` |
| `batch=16, n=512` | 約 `109.89 ms` | 約 `0.0767 ms` | 約 `1433x` |

這個結果很明確：

- trailing update 已經不該再優先花時間手刻 CuTe kernel
- 舊 `part5_*` kernels 目前更適合作為 correctness/benchmark reference，而不是主線效能路徑

### 端到端狀態更新

在目前主線與本地 benchmark 下：

- `verify_submission.py --remote-cases --max-n 512 --benchmark-torch`
- correctness 維持通過
- 相較前一版手寫 CuTe trailing update，端到端時間已明顯下降

代表性結果：

| case | current custom | `torch.geqrf` | ratio |
| --- | ---: | ---: | ---: |
| `batch=40, n=176` dense | 約 `297 ms` | 約 `31.4 ms` | 約 `9.5x` slower |
| `batch=40, n=352` dense | 約 `639 ms` | 約 `73.3 ms` | 約 `8.7x` slower |
| `batch=16, n=512` dense | 約 `866 ms` | 約 `46.0 ms` | 約 `18.8x` slower |

這和先前約 `31x` 到 `74x` slower 的狀態相比，已經有明顯改善。

### 更新後的下一步

既然 trailing update 已成功切到 cuBLAS 路線，接下來優先順序應更新為：

1. **panel kernel 繼續收同步與資料流成本**
   - 目前已完成 warp-local `sigma` reduction
   - 下一步若再動 panel，應聚焦在 correctness 安全的局部收益，不要再直接硬改 `apply reflector`

2. **優先處理 `build T`**
   - 目前 `build_compact_wy_t_cuda(...).launch(grid=(batch_count, 1, 1), block=(1, 1, 1))`
   - 這是現在最明確、最乾淨的低效率點之一

3. **重新量測 panel / build T / trailing update 的占比**
   - trailing update 已大幅降到可忽略量級
   - 後續 profiling 應確認剩餘時間主要集中在 panel 還是 `build T`

4. **考慮保留舊 `part5_*` kernels 僅作研究用途**
   - 若提交檔案大小、可讀性或維護成本變成問題，可再考慮移除或封存

### 同日更新：`build T` 主線也切到 batched Torch 路線

原本 `build_compact_wy_t_cuda(...)` 是：

```text
grid=(batch_count, 1, 1), block=(1, 1, 1)
```

這代表整個 `T` factor 建構幾乎是單 thread kernel。現在主線已改成直接從 compact `H` storage 重建
`V`，再用 batched `torch.bmm` 走 `larft` 風格遞推，最後把結果轉成目前 trailing update 使用的 lower-triangular `T`。

對應程式位置：

- [submission.py](/modelz/workspace/weiminc/qr/GPU-qr-factorization/submission.py:435)

目前主線組合變成：

1. panel factorization：CuTe DSL custom kernel
2. build `T`：Torch batched GEMM path
3. trailing update：Torch batched GEMM path

### `build T` 改動後的量測

代表性分段量測：

- case: dense
- `batch=16, n=512`
- `nb=64`

結果：

| component | time |
| --- | ---: |
| panel factor (`_panel_factor_apply_cutedsl_mvp`) | 約 `74.7 ms` |
| build `T`（新 Torch 路線） | 約 `3.07 ms` |
| 舊 CuTe trailing kernel（僅對照） | 約 `107.6 ms` |
| current full blocked QR | 約 `619.7 ms` |

### 端到端改善

在 `verify_submission.py --remote-cases --max-n 512 --benchmark-torch` 下，代表性 case 進一步改善為：

| case | current custom | `torch.geqrf` | ratio |
| --- | ---: | ---: | ---: |
| `batch=40, n=176` dense | 約 `233.6 ms` | 約 `41.7 ms` | 約 `5.6x` slower |
| `batch=40, n=352` dense | 約 `488.2 ms` | 約 `77.6 ms` | 約 `6.3x` slower |
| `batch=16, n=512` dense | 約 `633.8 ms` | 約 `45.5 ms` | 約 `13.9x` slower |

和前一階段相比，這表示：

- trailing update 的大頭已經被砍掉
- `build T` 的單 thread bottleneck 也已經消失
- 剩下的主要問題幾乎可以明確收斂到 panel factorization

### 更新後的下一步

現在下一步優先順序應再收斂為：

1. **重新聚焦 panel kernel**
   - `sigma` reduction 已做過 warp-local 優化
   - 接下來應先找出 panel 內還有哪些同步或 memory pattern 是安全可收的

2. **做更精確的 panel profiling**
   - 區分 `sigma` reduction、`scale v_tail`、`apply reflector`
   - 確認目前真正最重的是 `apply reflector`，而不是其他固定成本

3. **保留舊 CuTe `build T` / trailing kernels 作對照，但不再當主線**
   - 它們目前的主要價值是做 correctness / benchmark reference

### 同日更新：panel access profiling

為了確認 panel bottleneck 是否主要來自 row-major tensor 上的 column-wise access，新增了兩個 profiling 腳本：

- [profile_panel.py](/modelz/workspace/weiminc/qr/GPU-qr-factorization/profile_panel.py)
- [profile_panel_access.py](/modelz/workspace/weiminc/qr/GPU-qr-factorization/profile_panel_access.py)

其中：

- `profile_panel.py` 用 Torch CUDA 高階運算做較粗粒度的 `sigma` / `scale` / `apply` 對照
- `profile_panel_access.py` 用 CuTe microkernels 更貼近目前 panel kernel 的 thread/work pattern，比較：
  - row-major 上的 strided column access
  - transpose 後用 contiguous last-dim access 的版本

代表性結果：

| shape | sigma row / transposed | dot row / transposed | update row / transposed |
| --- | ---: | ---: | ---: |
| `batch=16, n=512, nb=64` | `1.14x` | `1.01x` | `1.00x` |
| `batch=40, n=176, nb=64` | `1.00x` | `1.01x` | `0.99x` |

目前可得結論：

1. 單看這種 access-pattern microbenchmark，row-major column-wise access 並沒有自然放大成一個壓倒性的成本來源
2. 這表示 panel 目前的慢點更可能來自：
   - 每個 panel 只有 `grid=(batch, 1, 1)` 的低 grid parallelism
   - `apply reflector` 階段只有 `panel_width - 1` 個 threads 有效工作
   - 每個 `j` 都有多次 block-wide `sync_threads()`
   - panel kernel 內大量 serial-per-thread loop
3. 因此，「column-wise access 不友善」仍然是真的，但在目前實作下，它不像是唯一主因，也不像已經被證明是最大主因

補充：

- 原本想用 `ncu` 看 scheduler / memory metrics，但本地環境沒有 `ncu` command，因此這輪 profiling 以 runtime microbench 為主

這個結果也支持目前的優先順序：

- 若下一步再動 panel，應優先處理 work partition / synchronization / grid parallelism
- 不應先假設「只要改 transpose 或 shared-memory reorder 就會自然解決主要瓶頸」

### 同日更新：`ncu` panel profile 與後續小幅優化

在本地以 `sudo ncu` 單獨 profile `part2_3_factor_apply_panel` 後，代表性結果（`batch=16, n=512, nb=64`）如下：

- `Grid Size = 16`
- `Block Size = 128`（profile 時的舊版）
- `Achieved Occupancy ≈ 8.3%`
- `One or More Eligible ≈ 1.41%`
- `No Eligible ≈ 98.59%`
- `Warp Cycles Per Issued Instruction ≈ 70.8`
- stall 組成約：
  - `CTA barrier` 約 `52.9%`
  - `L1TEX scoreboard` 約 `42.3%`
- `Memory Throughput ≈ 613-672 MB/s`
- `Mem Busy < 1%`
- `L2 Hit Rate ≈ 99%`

這個結果非常明確地表示：

1. **目前 panel kernel 不是 bandwidth-bound**
   - DRAM/L2 都沒有被打滿
   - 單純把它解讀成「column-wise access 太差，所以一定是 memory bound」並不成立

2. **主要問題是 launch shape 與 barrier-heavy execution**
   - `grid=(batch,1,1)` 讓 188 SM 上只有 16 blocks，平行度遠遠不足
   - CTA 內 barrier stall 非常重，eligible warps 幾乎沒有

3. **memory pattern 仍有成本，但更像次級問題**
   - `L1TEX scoreboard` stall 依然高，代表資料存取型態與 locality 還是值得優化
   - 但它和 barrier / grid underfill 是同時存在的問題，而不是唯一主因

### 已做的小幅 panel 優化

根據上述 `ncu` 訊號，先做了兩個低風險調整：

1. **panel CTA 從 128 threads 收到 64 threads**
   - 理由：`nb <= 64`，而 `apply reflector` 最多只需要 63 個 target threads
   - 這能直接減少每次 barrier 要等待的 warp 數

2. **加入 small-square fast path**
   - `n <= 64` 時直接走 `torch.geqrf`
   - 這是參考 MAGMA `smallsq` 分流的務實版本
   - 目的：砍掉目前小矩陣案例最明顯的固定 launch / orchestration overhead

### 這兩步之後的效果

代表性 benchmark：

| case | current custom | `torch.geqrf` | ratio |
| --- | ---: | ---: | ---: |
| `batch=20, n=32` dense | 約 `1.04-2.69 ms` | 約 `1.07-2.72 ms` | 約 `0.97x-0.99x` |
| `batch=40, n=176` dense | 約 `232-235 ms` | 約 `34-38 ms` | 約 `6.1x-6.9x` |
| `batch=40, n=352` dense | 約 `468-471 ms` | 約 `71-72 ms` | 約 `6.5x` |
| `batch=16, n=512` dense | 約 `623-633 ms` | 約 `44-46 ms` | 約 `13.6x-14.8x` |

解讀：

- small-square path 很有效，`n=32` 已幾乎追平 `torch.geqrf`
- 對中大尺寸，64-thread CTA 有小幅幫助，但不會改變主結論
- `n=512` 仍然明確受限於 panel 架構本身

### 更新後的下一步

現在最值得做的事已經更明確：

1. **panel 改成 multi-CTA / row-tiled decomposition**
   - 這是唯一能真正處理 `grid size << #SMs` 問題的方向
   - 單 CTA 內再做小修，收益會很有限

2. **若不立即重寫 panel，就不要再在單 CTA 上疊更多 barrier-heavy tricks**
   - 先前經驗已經證明這條路很容易變慢

3. **保留 `n <= 64` fast path**
   - 它已經是有效且低風險的 shape-based dispatch

### 同日更新：panel 主線改成 row-tiled batched Torch CUDA 路徑

在確認單 CTA CuTe panel kernel 的主要問題是：

- `grid=(batch, 1, 1)` 太小
- barrier stall 很重
- 單 CTA 內 thread utilization 差

之後，主線 panel 已改成較接近 multi-CTA / row-tiled decomposition 的 GPU-wide batched 路徑：

- 保留 panel progression 在 Python 中沿 `j` sequential 前進
- 但每個 `j` 的三個主要工作都改成 Torch CUDA tensor ops：
  - `sigma` reduction
  - `v_tail` scaling
  - panel-internal reflector application

對應程式位置：

- [submission.py](/modelz/workspace/weiminc/qr/GPU-qr-factorization/submission.py:432)

這個版本不是把舊 CuTe kernel 硬拆成多個 CuTe row-tiled kernels，而是直接改成 GPU 上的 batched tensor path，先把「單 CTA panel」這個最大架構限制移除。

### 新 panel 路徑效果

代表性 panel 單段量測：

| case | panel time |
| --- | ---: |
| `batch=16, n=512, nb=64` | 約 `14.1 ms` |

對比先前版本：

- 舊單 CTA panel 約 `75 ms`
- 新 row-tiled batched panel 約 `14 ms`

大約是 **5x 以上** 的 panel 改善。

### 端到端 benchmark 更新

| case | current custom | `torch.geqrf` | ratio |
| --- | ---: | ---: | ---: |
| `batch=20, n=32` dense | 約 `2.4 ms` | 約 `2.7 ms` | 約 `0.89x` |
| `batch=40, n=176` dense | 約 `54.4 ms` | 約 `33.7 ms` | 約 `1.62x` |
| `batch=40, n=352` dense | 約 `100.5 ms` | 約 `75.5 ms` | 約 `1.33x` |
| `batch=16, n=512` dense | 約 `136.5 ms` | 約 `45.8 ms` | 約 `2.98x` |

這比前一階段的：

- `n=176` 約 `6x-7x` slower
- `n=352` 約 `6.5x` slower
- `n=512` 約 `13x-15x` slower

有大幅改善。

### 更新後的結論

1. 對目前這份 submission，真正有決定性效果的不是在單 CTA CuTe panel kernel 上做更多局部修補
2. 把 panel 改成 GPU-wide batched row-tiled 路徑，收益遠大於先前所有單 CTA 小修
3. 目前 custom path 已經從「顯著落後 Torch」收斂到：
   - `n=176/352` 僅約 `1.3x-1.6x` slower
   - `n=512` 約 `3x` slower

### 下一步

如果要再往前推，接下來更值得看的方向是：

1. `panel` 內是否還能進一步 fuse / 減少中間 tensor traffic
2. `build T` 是否值得再從 Torch batched path 移回更低 overhead 的自訂 kernel
3. 是否要針對 `n > 512` 或更大 batch 再做 shape-based dispatch

## 方向修正（2026-06-17）

### 明確約束：主路徑不能使用 `torch.geqrf`

如果目標是實際效能快過 `torch.geqrf`，那麼：

- `torch.geqrf` 不能再出現在 active fast path
- 它只能保留成：
  - fallback（例如沒有 cutlass / CUDA backend 時）
  - correctness / benchmark reference

因此先前為了清掉小矩陣固定成本而加入的 `n <= 64 -> torch.geqrf` fast path 已移除。

### 移除 `geqrf` fast path 後的現況

代表性 benchmark：

| case | current custom | `torch.geqrf` | ratio |
| --- | ---: | ---: | ---: |
| `batch=20, n=32` dense | 約 `6.8-6.9 ms` | 約 `0.47-1.13 ms` | 約 `6x-15x` slower |
| `batch=40, n=176` dense | 約 `43-44 ms` | 約 `55-60 ms` | 約 `0.74x-0.79x` |
| `batch=40, n=352` dense | 約 `88-92 ms` | 約 `110-112 ms` | 約 `0.80x-0.82x` |
| `batch=16, n=512` dense | 約 `124-127 ms` | 約 `70 ms` | 約 `1.8x` slower |

目前可以更誠實地描述為：

- 中尺寸 `n=176/352`，目前 custom path 已有機會快過 `torch.geqrf`
- `n=512` 仍落後約 `1.8x`
- 小尺寸 `n<=64` 若不用 `torch.geqrf` fast path，固定成本會再次暴露

### 這代表什麼

這其實把下一步收得更清楚：

1. **不要再往 Torch fallback / dispatch 方向優化**
   - 那會破壞「主路徑不靠 `geqrf`」這個目標

2. **真正該投資的是 raw CUDA / CUDA C / inline PTX**
   - 特別是小尺寸 panel 與固定成本
   - 以及 `n=512+` 時仍落後的 panel hot path

3. **目前 row-tiled Torch panel 可以當作效能基線，但不應視為終點**
   - 它證明「把 panel 從單 CTA kernel 解放」是正確方向
   - 但若要再超過 `geqrf`，接下來需要更低 overhead、更可控的 custom kernel

### 接下來的具體優先順序

1. **先做 small-square custom kernel（`n <= 32/64`）**
   - 這是最明顯還落後 `geqrf` 的區段
   - 最適合用 CUDA C / shared-memory resident panel kernel 直接吃掉固定成本

2. **把目前 row-tiled panel 的 hot path 轉成 raw CUDA**
   - 先針對 `sigma + scale + panel-apply` 做 fused kernel
   - 讓 Torch tensor-op orchestration 退居次要角色

3. **必要時再往 inline PTX 推**
   - 若需要更細緻掌控：
     - shared-memory movement
     - reduction structure
     - warp-specialized scheduling
   - 再考慮 `cp.async`、warp shuffle、更 aggressive 的 register/shared-memory choreography

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

## MAGMA Batched GEQRF 對照總覽

MAGMA `geqrf_batched` 與目前實作使用相同的 compact Householder convention：

```text
Q = H(1) H(2) ... H(k)
H(i) = I - tau * v * v'
v(i) = 1
v(i+1:m) stored in A(i+1:m, i)
tau stored in TAU(i)
```

因此兩者的數學 output format 相容；差異主要在工程設計：

1. **Expert workspace API**
   - MAGMA expert routines expose `nb`, `dR_array`, `dT_array`, `dW_array`, and `provide_RT`.
   - `dR/dT/dW` are not incidental temporaries; they are part of the algorithm and layout design.
   - Current CuTe DSL implementation only recently moved to reusable `t_workspace` / `y_partial_workspace`; layout is still simple.
2. **Small-square special path**
   - MAGMA provides `magma_*geqrf_batched_smallsq`, documented for small square matrices up to 32.
   - Current implementation uses the same generic panel kernel for `n=16/32/64`.
   - This likely explains the large constant runtime floor around 78 ms for small `n`.
3. **Coalesced memory layout**
   - MAGMA docs require leading dimensions such as `LDDA`, `LDDR`, `LDDT` to be divisible by 16 for coalesced access.
   - Our input is PyTorch row-major `(batch, n, n)`. Householder panel operations scan fixed columns, which means stride-`n` memory access.
   - This is a major gap versus MAGMA’s column-major/coalesced panel access.
4. **Production kernel family**
   - MAGMA has separate batched, expert, and small-square routines.
   - Current code is a correctness-first single CuTe DSL path with a small number of generic kernels.
5. **Compiled library vs Python/CuTe DSL orchestration**
   - MAGMA avoids Python-level orchestration overhead.
   - Current code pays CuTe DSL JIT/launch orchestration overhead; small cases are dominated by fixed cost.

手工優化時，應把 MAGMA 當作 design reference，而不是直接照抄 API。最值得借的是：workspace layout、small-square fast path、16-aligned/coalesced tile assumptions、以及 shape-dependent kernel dispatch。

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

### MAGMA 對照與手工優化方向

MAGMA expert batched GEQRF 把 orchestration 需要的 blocking/workspace 明確外露：

```text
nb
dR_array
dT_array
dW_array
provide_RT
```

目前 CuTe DSL 版本已做的對應：

- `nb` default 改為 64。
- `t_workspace` 在每次 QR 中重用。
- `y_partial_workspace` 在每次 QR 中重用。

仍然不足：

- workspace layout 沒有像 MAGMA 那樣針對 coalesced access 設計。
- `t_workspace` / `y_partial_workspace` 還是普通 row-major tensor temporary。
- Python panel loop 仍會 launch 多個 CuTe kernels；小 `n` 時固定 launch cost 主導。

後續手工方向：

1. **Shape dispatch**
   - `n <= 32`: small-square special kernel。
   - `33 <= n <= 64`: single-panel special kernel。
   - `n > 64`: blocked WY path。
2. **Workspace layout specialization**
   - 為 `T` / `Y_partial` 設計 16-aligned tile layout。
   - 讓 fused update kernel 的 `Y_partial` reads 更 contiguous。
3. **降低 Python allocation / dispatch**
   - 已完成 per-QR workspace reuse。
   - 下一步是把多個 per-panel stages fuse，或做 persistent style kernel，但風險較高。

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

### MAGMA 對照與手工優化方向

MAGMA 的 output convention 與此處一致，但 kernel engineering 不同。

主要差距：

1. **Memory coalescing**
   - MAGMA docs 強調 `LDDA` divisible by 16，以利 coalesced access。
   - MAGMA 傳統上以 column-major layout 為主，panel column scan 是 contiguous 或接近 contiguous。
   - PyTorch input 是 row-major `(batch, n, n)`；目前掃 `h[bidx, row, j]` 是 stride-`n` access。
   - 這使 panel sigma reduction 和 `v_tail` writeback 都不是理想 coalesced path。
2. **Grid parallelism**
   - 目前每個 batch item 一個 CTA。
   - 對 `batch=1/2/4/20/40` 都遠小於 B200/Blackwell 的 SM 數。
   - MAGMA batched kernels通常會根據 small/medium shape 使用不同 kernel family，而不是單一 `(batch,1,1)` grid。
3. **Small-square path**
   - MAGMA 有 `geqrf_batched_smallsq`，針對 `n <= 32`。
   - Current panel kernel 對 `n=16/32` 仍走 generic 128-thread panel code。

手工優化方向：

1. **Small-square shared-memory kernel**
   - 對 `n <= 32` 寫專用 kernel。
   - 一個 CTA 處理一個 matrix，將整個 small matrix 或 column tiles 搬進 shared memory。
   - 在 shared memory 中做 Householder panel loop，最後寫回 compact `H/tau`。
   - 目標是打掉目前 small `n` 固定約 78 ms 的 overhead floor。
2. **Panel tile-transpose**
   - 對 row-major input，先把 active panel tile transposed/load 到 shared memory。
   - 在 shared memory 中讓 column reductions contiguous。
   - 寫回 compact `v_tail` 時再轉回 row-major。
3. **Multi-CTA panel reduction**
   - 把 sigma 計算拆為 `partial_sigma[b, j, row_tile]`。
   - 第二階段 reduce 出 sigma，再 scale/writeback。
   - 風險：kernel launch 增加，可能只對大 `n` 值得。
4. **Warp-level reduction**
   - 用 warp shuffle 取代 shared-memory tree reduction。
   - 優先用在 sigma reduction；減少 `sync_threads()` barrier。

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

### MAGMA 對照與手工優化方向

目前試過的單 CTA row-parallel panel internal apply：

```text
每個 target column:
  128 threads reduce v^T x
  128 threads update rows
```

結果 correctness 通過，但 runtime 變差，原因是每個 target column 都增加多輪 `sync_threads()`。這提醒我們：不能只在單 CTA 內增加 parallelism。

MAGMA-style 的可學方向：

1. **在 small-square kernel 中 fuse panel internal apply**
   - 對 `n <= 32/64`，把 panel 和 internal apply 全部放在 shared memory。
   - target columns 的更新可以用 shared-memory tile，而不是 global stride access。
2. **對 larger panels 使用 tile operation**
   - 類似 tiled Householder / GEQRT-ORMQR 思路。
   - 不一定完整採用 MAGMA API，但可以拆成 diagonal tile factor + trailing tile update。
3. **避免 per-target full CTA barriers**
   - 若要 parallelize target columns，應該讓不同 warps 或 CTAs 處理不同 target column/tile，而不是所有 128 threads 對每個 target column 重複全 CTA reduction。
4. **保留 compact output order**
   - 所有 panel internal apply 的 reflector order 必須和 `torch.geqrf` compatible。
   - 任何 tiled rewrite 都要先用 `torch.linalg.householder_product` checker 驗證。

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

### MAGMA 對照與手工優化方向

MAGMA expert interface 提供 `dT_array`，且文件要求 `LDDT >= min(NB,min(M,N))`，並建議相關 leading dimensions divisible by 16。這表示 `T` 不只是臨時數學物件，而是有 layout 約束的 workspace/output object。

目前 CuTe DSL 主線：

- 已移除 Torch `V` materialization。
- 已用 `_build_compact_wy_t_kernel` 直接從 compact `H/tau` 建 `T`。
- `T` workspace shape 是 `(batch, nb, nb)`，目前 `nb <= 64`。

差距：

- `T` 是普通 contiguous tensor，沒有特別為 fused update 的 access pattern 設計。
- `T` build kernel 是 batch-level sequential small loop，沒有使用 warp-level parallelism。
- 對 `n <= 64` single-panel 情況，Part 5 不跑，建 `T` 也不需要；目前程式已只在 `j_end < n` 時建 `T`，這點合理。

手工優化方向：

1. **T layout specialization**
   - 讓 `T[row_i, k]` 在 fused update 中被 coalesced/shared-friendly 讀取。
   - 可測 transposed `T` layout，讓 inner loop `k` 更連續。
2. **Build T in shared memory**
   - 對每個 batch/panel 在 shared memory 中建 `T`，必要時只寫出 fused update 需要的 layout。
   - 若能把 T-build 和 Part 5 fused，可能減少 global `T` traffic。
3. **Warp parallel T-build**
   - 對 `nb=64`，`v_j^T V_prev` 的 dot products 可由 warps 分工。
   - 但 Part 4 目前不是最大 bottleneck，優先級低於 small-square path 和 panel memory layout。
4. **MAGMA `provide_RT` 啟發**
   - 我們 checker 不需要額外輸出 full `R/T` workspace。
   - 因此 `T` 應以 local workspace 最佳化，不必維持可讀 output format。

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

### MAGMA 對照與手工優化方向

MAGMA expert interface 的 `dW_array` 是 Part 5 這類 trailing update 的重要 workspace。文件指出 `dW_array` dimension 需要 `LDDW >= NB`。這和目前 `Y_partial` / fused update 的用途相近。

目前 CuTe DSL 主線：

```text
compute_y_partial:
  Y_partial = V_tile^T C_tile

fused update:
  read Y_partial
  compute Y = reduce(row_tiles)
  compute Z = T Y
  update C_tile -= V_tile Z
```

有效改動：

- row-tiled `Y_partial` 提高 grid parallelism。
- fused reduce/update 省掉 global `Z` temporary。
- `nb=64` 降低 panel count，對 `n=64/128/176` 明顯有效。

與 MAGMA 的差距：

1. **Workspace layout**
   - MAGMA `dW` 是 algorithm-aware workspace。
   - `Y_partial` 目前是普通 `(batch, panel_cols, trailing_cols, row_tiles)` layout。
   - Fused update 讀 `y_partial[b, panel_i, col, rt]`，對 thread tile 未必 coalesced。
2. **GEMM-like implementation**
   - MAGMA trailing update 會更接近 BLAS-3 / batched BLAS design。
   - 目前 fused update 是 scalar loop over `panel_cols`，尚未使用 MMA 或 CuTe tiled GEMM machinery。
3. **Column tiling experiment**
   - 嘗試一個 CTA 同時處理 4 columns 的 reduce stage，correctness 通過但 runtime 變差。
   - 這表示此 stage 仍需要足夠 independent CTAs，不宜簡單合併 columns 降低 grid size。

手工優化方向：

1. **Re-layout `Y_partial`**
   - 測試 `(batch, trailing_cols, panel_cols, row_tiles)` 或 col-tile-major layout。
   - 目標是 fused update 讀同一 col tile 的 `Y_partial` 更連續。
2. **Use shared `V/T` tile explicitly**
   - Fused update 目前從 global `H` 讀 compact `V`。
   - 可把 active row tile 的 `V` 搬進 shared memory，供 16 columns 重用。
3. **MMA/GEMM-style WY**
   - 將 `V_tile^T C_tile`、`T Y`、`V_tile Z` 寫成更規則的 small GEMM。
   - 這是接近 MAGMA/CUTLASS 的長期方向。
4. **Retain grid parallelism**
   - 不要只為了增加 per-CTA work 而合併太多 columns。
   - col4 reduce 的退化證明：grid size 對 B200 很重要。
5. **Shape-dependent Part 5 path**
   - `n <= 64`: single panel，不需要 Part 5。
   - `64 < n <= 176`: current fused path 有效。
   - 更大 `n`: 可能需要 true GEMM-like tiled WY，避免 scalar loops over `panel_cols` 成為瓶頸。

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

## Raw CUDA small-square inline fast path

日期：2026-06-17

`n <= 64` small-square path 已改成 raw CUDA C single-panel kernel，並已接入 `submission.py`。重要工程狀態：

1. **提交形式**
   - Popcorn 提交目前只能交 `.py`，所以 raw CUDA source 已內嵌在 `submission.py`。
   - 使用 `torch.utils.cpp_extension.load_inline(...)` JIT build extension。
   - `small_square_qr.cu` 仍保留在 repo root，作為開發/閱讀用 source；提交時不依賴外部 `.cu` 檔。

2. **dispatch policy**
   - `custom_kernel(data)` 對 CUDA float32 square input 做 shape gate。
   - `n <= 64`：走 inline raw CUDA small-square QR。
   - `n > 64`：維持既有 blocked path。
   - backend 固定為 `raw-cuda-small-square+cutlass`，已移除 runtime backend selector 與 `torch.geqrf` fallback。

3. **kernel design**
   - one CTA per batch matrix。
   - `block = 128 threads`。
   - shared-memory resident `64 x 64` row-major tile。
   - shared layout:
     - `s_a`: matrix tile。
     - `s_tau`: compact Householder tau。
     - `s_reduce`: CTA reduction scratch。
     - `s_scalars`: `scale / tau_j / w`。
   - 輸出仍是 `torch.geqrf` compatible compact Householder `(H, tau)`。

4. **已驗證 timing**

remote-style benchmark 指令：

```bash
source env.sh
python verify_submission.py --module submission --remote-cases --max-n 352 --benchmark-torch --trials 10
```

代表性結果：

| case | result | custom | `torch.geqrf` | `torch.linalg.qr` | custom/geqrf |
| --- | --- | ---: | ---: | ---: | ---: |
| remote dense `batch=20, n=32` | pass | `0.470 ms` | `0.595 ms` | `2.082 ms` | `0.79x` |
| remote dense `batch=40, n=176` | pass | `42.063 ms` | `32.137 ms` | `41.887 ms` | `1.31x` |
| remote dense `batch=40, n=352` | pass | `86.990 ms` | `70.694 ms` | `89.825 ms` | `1.23x` |

small-square benchmark 指令：

```bash
source env.sh
python profile/benchmark_small_square_qr.py --n 16 32 64 --batch 20 --trials 10 --case dense upper rankdef rowscale
```

代表性 small-square 結果：

| case | result | custom | `torch.geqrf` | ratio |
| --- | --- | ---: | ---: | ---: |
| dense `batch=20, n=16` | pass | `0.280 ms` | `0.301 ms` | `0.93x` |
| dense `batch=20, n=32` | pass | `0.334 ms` | `0.740 ms` | `0.45x` |
| dense `batch=20, n=64` | pass | `1.702 ms` | `3.060 ms` | `0.56x` |
| rowscale `batch=20, n=64` | pass | `1.733 ms` | `3.075 ms` | `0.56x` |

解讀：

- small-square 固定成本已從 CuTe DSL 主線的數 ms/數十 ms 級別降到 sub-ms 級別。
- raw CUDA path 已過 official checker。
- `n=176/352` 仍落後 `torch.geqrf`，代表中尺寸 blocked path 仍是下一個主要優化目標。
- 首次呼叫會包含 JIT compile 風險；若遠端評測把第一次 extension build 算入 submission time，這仍是需要注意的提交風險。

### Repo layout 更新

根目錄已收斂成 submission/runtime 相關檔案：

```text
submission.py
small_square_qr.cu
env.sh
cutedsl_env.sh
verify_submission.py
```

root 保留 `verify_submission.py` 作為常用 local checker。profiling、benchmark、submit helper 仍放在 `profile/`：

```text
profile/benchmark_small_square_qr.py
profile/profile_panel.py
profile/profile_panel_access.py
profile/profile_panel_ncu.py
profile/profile_qr.py
profile/run_benchmark.sh
profile/submit_qr_v2.sh
```

`profile/verify_submission.py` 也保留一份副本，兩份 verifier 都會把 repo root 和 `profile/` 加到 `sys.path`，因此 root checker 能找到 `profile/qr_official.py`，profile checker 也能找到 root `submission.py`。

### 後續優先級調整

原本「先做 small-square custom kernel」已完成第一版。下一步優先順序改為：

1. 減少/規避 `load_inline` 首次 JIT compile 對遠端 submission 的影響。
2. 對 `n <= 64` raw CUDA kernel 做 micro-optimization：
   - warp-level reduction 取代 full CTA shared-memory tree。
   - upper/diagonal/rank-deficient cases 的 early-exit 或更低 barrier path。
   - 視 batch size 決定是否一 CTA per matrix 仍足夠。
3. 把中尺寸 panel hot path 轉成 raw CUDA/C++ extension，避免 Python/Torch orchestration 成為主要 overhead。

## CuTe DSL 主線更新：T build、tiled WY、fused update

日期：2026-06-16

目前 `submission.py` 固定為 `raw-cuda-small-square+cutlass` backend：`n <= 64` 先走 inline raw CUDA small-square fast path，其餘尺寸維持 blocked CuTe/Torch path。blocked path 的有效主線如下：

1. Panel factor/apply:
   - `part2_3_factor_apply_panel_cuda`
   - 仍是每個 batch 一個 CTA。
   - panel-internal apply 維持原本 sequential-per-target-column 做法。
2. Build compact WY `T`:
   - `_build_compact_wy_t_kernel`
   - 直接從 compact `H` 和 `tau` 建 `T`。
   - 移除 Torch materialized `V` 和 Torch `bmm` T-build。
3. Part 5 trailing update:
   - `_part5_compute_y_partial_kernel`
   - `_part5_reduce_z_update_c_fused_kernel`
   - `part5_apply_panel_wy_fused_update_cuda`
   - 先以 row-tiled 方式計算 `Y_partial = V_tile^T C_tile`。
   - 再讓 fused update kernel 直接讀 `Y_partial`，在 `(col_tile, row_tile, batch)` CTA 內計算 `Y`、`Z = T Y` 並更新 `C_tile`。
   - 不再 materialize global `Z` temporary。
4. Panel width:
   - default `nb = 64`
   - runtime 內 clamp `nb <= 64`
   - 目前 shared-memory buffers 以 `panel_cols <= 64` 設計。

### 有效改動

#### 1. Part 4 移到 CuTe DSL

舊做法：

```text
_build_panel_v_torch
_build_compact_wy_t_torch
```

新做法：

```text
_build_compact_wy_t_kernel
```

結果：

- correctness 通過。
- 少掉 full `V` materialization。
- 少掉 Torch `bmm` 和 Torch runtime launch overhead。
- 對小 `nb` 的單次 kernel 本身不一定快很多，但讓主路徑更乾淨，也便於後續 fuse。

#### 2. Part 5 從 per-column WY-GEMV 改為 row-tiled WY

舊做法：

```text
grid = (batch, trailing_col, 1)
每個 CTA 掃完整 active rows
```

新做法：

```text
compute_y_partial:
  grid = (trailing_cols, batch, panel_cols * row_tiles)

fused reduce/update:
  grid = (ceil(trailing_cols / 16), ceil(active_rows / 16), batch)
```

數學仍是：

$$
C \leftarrow C - V T (V^T C)
$$

但先把 `V^T C` 拆成 row tiles，提高 grid parallelism。

#### 3. `nb = 64`

這是目前最大的 win。

原因：

- 對 `n <= 64`，整個 QR 變成 single panel，不需要 Part 5 trailing update。
- 對 `n > 64`，panel 數減半，Python/CuTe DSL launch 次數和 panel progression 次數都下降。
- `nb = 64` 仍可被目前 fixed shared buffers 支援。

實測 quick timing：

| shape | `nb=32` 舊主線 | `nb=64` 主線 |
| --- | ---: | ---: |
| `batch=4, n=64` | 約 327 ms | 約 78 ms |
| `batch=4, n=128` | 約 820 ms | 約 326 ms |
| `batch=4, n=176` | 約 1336 ms | 約 572 ms |

#### 4. Fuse `reduce_y_compute_z` 和 `update_c`

舊 three-stage 做法：

```text
Y_partial -> Z global -> update C
```

新 fused 做法：

```text
Y_partial -> fused kernel computes Y/Z and updates C
```

效果：

| shape | separate reduce/update | fused reduce/update |
| --- | ---: | ---: |
| `batch=4, n=128` | 約 327.5 ms | 約 299.1 ms |
| `batch=40, n=176` | 約 577.5 ms | 約 521.8 ms |

解讀：

- 省掉一個 kernel launch。
- 省掉 global `Z` write/read。
- 雖然 fused update 會在每個 row tile 重算同一個 `Z`，但對目前 benchmark 尺寸仍然划算。

### 回退的實驗

#### 1. Panel internal apply row-parallel

嘗試內容：

- 在 `part2_3_factor_apply_panel_cuda` 內，把 panel-internal apply 從「每個 target column 一個 thread sequential 掃 rows」改成「128 threads 對每個 target column 做 reduction，並 parallel update rows」。

結果：

- correctness 通過。
- runtime 顯著變差。

實測：

| shape | 原 panel apply | row-parallel panel apply |
| --- | ---: | ---: |
| `batch=2, n=16` | 約 79 ms | 約 127 ms |
| `batch=4, n=64` | 約 327 ms | 約 420 ms |

結論：

- 在同一 CTA 內增加 parallel reduction 會引入大量 `sync_threads()`。
- 對目前小 panel width，barrier 成本大於 row parallelism 收益。
- 真正要改善 panel kernel，應該走 multi-CTA row/block decomposition，而不是在單 CTA 內加更多 barriers。

#### 2. Column-tiled `reduce_y_compute_z`

嘗試內容：

- 把 `reduce_y_compute_z` 從每個 trailing column 一個 CTA 改成一個 CTA 同時處理 4 個 trailing columns。

結果：

- correctness 通過。
- runtime 變差。

實測：

| shape | per-column reduce | col4 reduce |
| --- | ---: | ---: |
| `batch=4, n=128` | 約 328 ms | 約 385 ms |
| `batch=4, n=176` | 約 575 ms | 約 677 ms |

結論：

- 這個 stage 需要更多 independent CTAs 來填 occupancy。
- 合併 4 columns 減少 grid size，反而降低 parallelism。
- 簡單 column tiling 不適合目前 launch shape。

### 當前 correctness / timing

驗證指令：

```bash
source /workspace/cutedsl_env.sh
python verify_submission.py --module submission_stream_unsafe --trials 1
python verify_submission.py --module submission_stream_unsafe --remote-cases --max-n 176 --trials 1
```

結果：

- default 9 cases 通過。
- remote-style `n=32`、`n=176` dense cases 通過。

目前 quick timing：

| shape | time |
| --- | ---: |
| `batch=4, n=64` | 約 78.3 ms |
| `batch=4, n=128` | 約 299.1 ms |
| `batch=40, n=176` | 約 521.8 ms |

與 Torch baseline 對比：

| shape | custom CuTe DSL | `torch.geqrf` | ratio |
| --- | ---: | ---: | ---: |
| `batch=2, n=16` | 約 78.4 ms | 約 0.53 ms | 約 148x slower |
| `batch=20, n=32` | 約 77.9 ms | 約 0.74 ms | 約 106x slower |
| `batch=4, n=64` | 約 78.4 ms | 約 2.61 ms | 約 30x slower |
| `batch=4, n=128` | 約 297.6 ms | 約 4.17 ms | 約 71x slower |
| `batch=40, n=176` | 約 517.7 ms | 約 34.3 ms | 約 15x slower |

解讀：

- CuTe DSL custom path correctness OK，但仍是 prototype，遠慢於 production `torch.geqrf` / MAGMA-style implementation。
- 小尺寸 `n=16/32/64` 有固定約 78 ms floor，最可能是 CuTe DSL orchestration/launch overhead 和 generic panel kernel 造成。
- MAGMA 的 small-square special path 是下一個最值得手工優化的方向。

### 下一步方向

1. **Shape-based `nb` policy**
   - 目前固定 `nb=64`。
   - 可測 `nb=128`，但需擴大 shared buffers 並重新評估 occupancy / correctness。
2. **Panel multi-CTA decomposition**
   - 單 CTA 內 row-parallel 已證明不划算。
   - 若要提升 panel parallelism，需要把 sigma reduction、scale、panel apply 拆成 row/block tiled kernels。
   - 風險是 kernel launch 數增加，可能只對大 `n` 有利。
3. **Warp-level reduction**
   - 若 CuTe DSL 能乾淨使用 warp shuffle，可替換 shared-memory tree reduction。
   - 優先目標是 `_part5_compute_y_partial_kernel` 和 panel sigma reduction。
4. **Full tiled Householder / GEQRT-TSQRT 路線**
   - 仍是長期研究方向。
   - 最大風險是把 tile reflectors 正確 pack 回 `torch.geqrf` compatible `(H, tau)`。

## Raw CUDA panel/T/WY update and next optimizations (2026-06-17)

Current `submission.py` active path has moved further toward CUDA C:

1. `n <= 64`: inline raw CUDA small-square QR.
2. `n > 64`: blocked QR with raw CUDA panel factor/apply, raw CUDA compact-WY `T` build, and raw CUDA row-tiled fused WY trailing update.
3. CuTe DSL kernels are still kept in the file as fallback/reference code, but they are no longer the active panel/T/WY path.

Representative benchmark after raw CUDA T/WY update:

| case | custom | `torch.geqrf` | custom/geqrf |
| --- | ---: | ---: | ---: |
| `batch=20, n=32` dense | `0.265 ms` | `0.384 ms` | `0.69x` |
| `batch=40, n=176` dense | `7.266 ms` | `17.081 ms` | `0.43x` |
| `batch=40, n=352` dense | `30.060 ms` | `37.256 ms` | `0.81x` |
| `batch=16, n=512` dense | `36.97 ms` | `23.49 ms` | `1.57x` |

First-panel breakdown:

| shape | panel | build T | WY update |
| --- | ---: | ---: | ---: |
| `batch=40, n=176` | `1.47 ms` | `1.15 ms` | `1.13 ms` |
| `batch=40, n=352` | `2.22 ms` | `2.16 ms` | `5.95 ms` |
| `batch=16, n=512` | `1.63 ms` | `2.38 ms` | `4.91 ms` |

Immediate optimization plan:

1. **Add `nb=128` blocked path support**
   - Expand raw CUDA panel/T/WY assumptions from `panel_cols <= 64` to `panel_cols <= 128`.
   - Increase WY shared `Y/Z` buffers from `64 * 16` to `128 * 16` elements.
   - Use shape-based/default `nb=128` for the blocked path to reduce panel count and Python/kernel launch count, especially for `n=512`.

2. **Fuse panel dot/update**
   - Replace the two-stage panel internal apply (`dot_partial_kernel` then `update_target_kernel`) with one raw CUDA kernel per `(batch, target column)` CTA where possible.
   - The fused kernel computes `dot = v^T c` and updates `c <- c - tau * v * dot` inside the same CTA.
   - This removes `dot_ws` traffic and roughly halves panel apply launches.
   - It is expected to help medium sizes where launch overhead and panel progression still matter.

Verification target after these changes:

```bash
source /workspace/cutedsl_env.sh
.venv/bin/python verify_submission.py --remote-cases --max-n 512 --trials 1 --benchmark-torch
```

### Implementation result: fused panel apply kept, `nb=128` kept as opt-in support

Both planned optimizations were implemented in `submission.py`:

1. Raw CUDA panel kernels now accept `panel_cols <= 128`, and raw WY shared buffers were expanded to support `128 * 16` panel/column tiles.
2. Panel internal apply now has an active fused raw CUDA target-column kernel (`apply_target_fused_kernel`) that computes `dot = v^T c` and updates the target column in one CTA, avoiding the old `dot_ws` two-stage path.

Correctness passed for:

```bash
source /workspace/cutedsl_env.sh
.venv/bin/python verify_submission.py --module submission --trials 1
.venv/bin/python verify_submission.py --module submission --remote-cases --max-n 512 --trials 1 --benchmark-torch
```

Measured result with active default `nb=64` plus fused panel apply:

| case | custom | `torch.geqrf` | custom/geqrf |
| --- | ---: | ---: | ---: |
| `batch=20, n=32` dense | `0.273 ms` | `0.374 ms` | `0.73x` |
| `batch=40, n=176` dense | `6.395 ms` | `17.069 ms` | `0.37x` |
| `batch=40, n=352` dense | `28.618 ms` | `37.301 ms` | `0.77x` |
| `batch=16, n=512` dense | `35.18 ms` | `23.47 ms` | `1.50x` |

Comparison against the previous raw CUDA T/WY state:

| case | before | after | note |
| --- | ---: | ---: | --- |
| `batch=40, n=176` | `7.266 ms` | `6.395 ms` | fused panel helps |
| `batch=40, n=352` | `30.060 ms` | `28.618 ms` | fused panel helps |
| `batch=16, n=512` | `36.97 ms` | `35.18 ms` | fused panel helps modestly |

`nb=128` was also tested but is not the active default because it was slower with the current raw T/WY structure:

| shape | `nb=64` | `nb=128` |
| --- | ---: | ---: |
| `batch=40, n=176` | `6.396 ms` | `9.788 ms` |
| `batch=40, n=352` | `28.525 ms` | `39.125 ms` |
| `batch=16, n=512` | `35.185 ms` | `51.624 ms` |

Conclusion:

- Keep `nb=64` as the active default.
- Keep `panel_cols <= 128` support as an opt-in path for future tuning.
- The next optimization should target `n=512`, likely by reducing per-panel launch count further or improving raw WY/T behavior for larger panels before trying `nb=128` again.

## Current raw CUDA panel optimization pass (2026-06-22)

Active `submission.py` path after this pass:

1. `n <= 64`: raw CUDA small-square path.
2. `n > 64`: raw CUDA panel factor/apply, raw CUDA compact-WY `T` build, and IEEE FP32 `torch.bmm` compact-WY trailing update.
3. Panel apply now uses a 4-target tiled kernel for coalesced access within each row.
4. Small-tail panel factorization now uses a guarded single-kernel factor path for `m <= 1024 && sigma_tiles <= 1`.

Implemented changes:

1. **`build_t_finish_kernel` barrier reduction**
   - Previous finish stage updated each row through `prev` one step at a time and synchronized after every `prev`.
   - New finish stage assigns each `col` to a thread lane, accumulates all `prev` contributions locally, writes once, and synchronizes once per `jj`.
   - Same FP32 math, but accumulation order changes.

2. **`apply_target_tiled4_kernel`**
   - Replaces per-target-column CTA panel apply with one CTA handling 4 adjacent target columns.
   - Threads are split into 4 column lanes and 64 row lanes, so target-column loads/stores are more coalesced than the old row-strided per-column kernel.
   - Correctness passed on remote-style cases through `n=4096`.

3. **`factor_single_tile_kernel` activation**
   - Existing single-tile factor kernel is now used for `m <= 1024 && sigma_tiles <= 1`.
   - This fuses the small-tail `sigma_partial + finalize_scale` pair into one launch.
   - Larger matrices keep the older multi-tile sigma/finalize path.

Representative A/B results under the local shared GPU environment:

| change | shape | before | after | note |
| --- | ---: | ---: | ---: | --- |
| build-T finish barrier reduction | `batch=4, n=1024` build-T | `6.743 ms` | `1.745 ms` | same apply path A/B |
| tiled4 panel apply | `batch=16, n=512` panel | `14.483 ms` | `9.800 ms` | same build-T path A/B |
| tiled4 panel apply | `batch=4, n=1024` panel | `26.368 ms` | `19.848 ms` | build-T later shows cache/measurement interaction |
| current final candidate | `batch=16, n=512` total | - | `10.624 ms` | stage median, GPU0 |
| current final candidate | `batch=4, n=1024` total | - | `25.429 ms` | stage median, GPU1 |
| current final candidate | `batch=2, n=2048` total | - | `71.954 ms` | stage median |
| current final candidate | `batch=1, n=4096` total | - | `190.980 ms` | stage median on busy GPU |

Correctness:

```bash
source .venv/bin/activate
source ./env.sh
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
CUDA_VISIBLE_DEVICES=2 python -u verify_submission.py --remote-cases --trials 1
```

Result: 19/19 remote-style cases passed. A less-contended verification run showed approximate e2e timings:

| case | e2e |
| --- | ---: |
| `batch=16, n=512` dense | `6.36 ms` |
| `batch=4, n=1024` dense | `13.79 ms` |
| `batch=2, n=2048` dense | `36.92 ms` |
| `batch=1, n=4096` dense | `98.84 ms` |

Submission:

```bash
popcorn-cli submit --gpu B200 --leaderboard qr_v2 --mode leaderboard --no-tui submission.py
```

Submission id: `827886`; result: succeeded. Secret and public test/benchmark/leaderboard runs all passed on B200, with no timeout.

Current known bottleneck expectation before the next NCU pass:

1. Panel remains dominant, especially for `n >= 2048`.
2. `apply_target_tiled4_kernel` should reduce uncoalesced memory traffic but may reduce CTA count versus per-target apply; the next NCU pass should check whether the net limit is now launch/grid occupancy or memory coalescing.
3. `build_t_dot_kernel` likely remains memory-access-pattern limited because it still walks row-major columns with a large stride.
4. `build_t_finish_kernel` should no longer be dominated by CTA barriers; if NCU still flags it, the next step is splitting finish across more CTAs or replacing it with a small triangular matmul formulation.

### NCU follow-up after submission 827886

Reports generated for the submitted/current version:

```text
reports/ncu_panel_current_n1024.ncu-rep
reports/ncu_build_t_current2_n1024.ncu-rep
```

Representative `n=1024, batch=4, nb=64` NCU metrics:

| kernel | duration | grid | waves/SM | achieved occupancy | key signal |
| --- | ---: | ---: | ---: | ---: | --- |
| `sigma_partial_kernel` | `3.7-3.9 us` | `16` | `0.01` | `15-17%` | tiny grid, 87% excessive sectors |
| `finalize_scale_kernel` | `~6.1 us` | `4` | `0.00` | `13-15%` | tiny grid, launch/occupancy limited |
| `apply_target_tiled4_kernel` | `9.6-11.6 us` | `64` | `0.06` | `15-16%` | grid still too small; excessive sectors improved to 61% |
| `build_t_dot_kernel` | `80.35 us` | `8064` | `7.15` | `94.7%` | high occupancy but 87% excessive sectors |
| `build_t_finish_kernel` | `127.9 us` | `4` | `0.00` | `16.6%` | still grid/barrier limited despite lower runtime |

Interpretation:

1. `apply_target_tiled4_kernel` improved memory access versus the previous per-target kernel: NCU excessive sectors dropped from the old roughly `87%` to about `61%` on the captured launches. It is still panel-dominant because the grid is only 64 blocks early in the panel, much smaller than 188 SMs.
2. `sigma_partial_kernel` / `finalize_scale_kernel` are individually short but launch many times. The small-tail `factor_single_tile_kernel` helps only late in the factorization; early `n=1024` panels still use multi-tile sigma/finalize.
3. `build_t_dot_kernel` now has enough grid and occupancy, so its remaining problem is memory layout: row-major column walks create uncoalesced loads.
4. `build_t_finish_kernel` runtime is much lower after the barrier rewrite, but NCU still flags the one-CTA-per-batch launch shape and remaining CTA barriers.

Next optimization direction from NCU:

1. Try `apply_target_tiled2` or hybrid policy: `tiled4` improves coalescing but reduces grid from `target_count * batch` to `ceil(target_count/4) * batch`; a `tiled2` version may recover occupancy while still improving memory sectors.
2. For `n >= 2048`, consider row-tile split for `apply_target_tiled4_kernel`: multiple CTAs per target tile compute partial dots, then update in a second pass. This costs extra global workspace/launches, so it should be guarded only for large remaining rows.
3. For `build_t_dot_kernel`, investigate staging panel columns in shared memory or transposed scratch for the current panel; this directly attacks the 87% excessive sectors.
4. For `build_t_finish_kernel`, splitting by `(batch, jj)` or using a compact triangular matrix update may reduce barrier wait, but expected payoff is smaller than panel apply.


## Triton build-T + row-split multi8 panel update (2026-06-23)

Active `submission.py` backend after this pass:

```text
raw-cuda-small-square+raw-cuda-panel-tail-rowsplit-multi8+triton-dot-raw-finish-t+fp32-gemm-wy
```

Main changes since submission `827886`:

1. **Triton build-T dot stage**
   - `build_t_dot_kernel` was replaced on the active path by `_build_t_dot_triton_kernel` plus the existing raw CUDA finish stage.
   - This keeps the compact-WY `T` math in FP32, but lets Triton generate a better vectorized dot kernel for the panel history terms.

2. **Panel apply hybrid policy**
   - Small/medium tails keep the raw CUDA tiled apply path.
   - For large matrices, the panel apply uses row-split decomposition guarded by:

```cpp
m >= 4096 && target_count >= 8 && apply_row_tiles >= 4
```

3. **Row-split multi-target apply**
   - The large-matrix row-split path now uses `dot_partial_multi8_kernel` and `update_target_multi8_kernel`.
   - One CTA handles up to 8 adjacent target columns, improving coalescing versus the earlier per-target row-split variant.
   - The extension cache tag is `parowsplit8x_t8_tdot1`.

Correctness and submission:

```bash
source .venv/bin/activate
source ./env.sh
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPYCACHEPREFIX=/tmp/qr_pycache_verify
CUDA_VISIBLE_DEVICES=2 python -u verify_submission.py --module submission --remote-cases --trials 1
popcorn-cli submit --gpu B200 --leaderboard qr_v2 --mode leaderboard --no-tui submission.py
```

Local remote-style verification passed all available cases. Representative one-trial timings:

| case | e2e |
| --- | ---: |
| `batch=16, n=512` dense | `5.86 ms` |
| `batch=4, n=1024` dense | `13.24 ms` |
| `batch=2, n=2048` dense | `34.40 ms` |
| `batch=1, n=4096` dense | `73.98 ms` |
| `batch=1, n=4096` upper | `73.85 ms` |

Leaderboard submission:

- Submission id: `828975`
- Result: `succeeded`
- Public and secret `test`, `benchmark`, and `leaderboard` runs all passed on B200.
- No timeout observed. Total remote wait was about `90s`.

### Stage-level impact

Representative active-path stage timings after Triton build-T and row-split multi8:

| shape | panel | build T | WY update | total |
| --- | ---: | ---: | ---: | ---: |
| `batch=16, n=512` | `4.83 ms` | `0.63 ms` | `0.38 ms` | `5.88 ms` |
| `batch=4, n=1024` | `11.17 ms` | `1.22 ms` | `0.88 ms` | `13.34 ms` |
| `batch=2, n=2048` | `28.98 ms` | `2.44 ms` | `2.45 ms` | `33.98 ms` |

For `n=4096`, some local profile runs were contaminated by shared GPU load. Clean verifier/A-B runs place the current total around `74 ms`, improved from the previous roughly `98 ms` post-`827886` local verifier timing.

Observed optimization deltas:

| change | representative effect |
| --- | --- |
| Triton build-T dot | `n=512` build T about `1.14 -> 0.63 ms`; `n=4096` about `8.2 -> 5.2 ms` |
| row-split per-target, `n=4096` | total about `90.8 -> 82.3 ms` in A/B |
| row-split multi8, `n=4096` | total about `82.8 -> 74.8 ms` in A/B |

### NCU result for current large-panel path

Current report:

```text
reports/ncu_panel_rowsplit_multi8_n4096.ncu-rep
```

Representative first-launch metrics:

| kernel | grid | duration | achieved occupancy | no eligible | key signal |
| --- | ---: | ---: | ---: | ---: | --- |
| `dot_partial_multi8_kernel` | `128` blocks | `4.29 us` | `18.62%` | `88.53%` | low grid/waves, but faster than per-target due to better coalescing |
| `update_target_multi8_kernel` | `128` blocks | `8.32 us` | `16.19%` | `95.55%` | still underfilled; update dominates the pair |

Compared with the previous per-target row-split NCU capture:

| kernel family | before | after |
| --- | ---: | ---: |
| dot partial | `7.52 us` | `4.29 us` |
| update target | `12.70 us` | `8.32 us` |

Interpretation:

- Multi8 wins despite lower CTA count because adjacent target-column accesses are more coalesced.
- The path is still not SM-saturated. NCU shows high `no eligible` and low waves/SM on the captured launches.
- For late panels, `target_count` shrinks, so fixed multi8 can become too coarse. This is the next most natural tuning target.

### Current GEMM / Tensor Core status

There is still no custom true GEMM kernel on the active path.

- The compact-WY trailing update is currently delegated to `torch.bmm` in FP32, so that portion should already use PyTorch/cuBLAS-style GEMM machinery where applicable.
- The QR panel factorization and panel apply are Householder reductions plus rank-1 style updates, not a straightforward tensor-core GEMM without restructuring.
- The build-T dot stage is now Triton-generated, but it is a collection of compact panel-history dot products, not a large tensor-core GEMM.

Therefore the next useful Tensor Core direction is not simply changing the existing panel kernels to FP16/FP8. A safer route is to isolate larger GEMM-like pieces, especially compact-WY trailing update variants, and keep panel scalar/reduction math in FP32 unless a mixed-precision error study proves otherwise.

### Next optimization priorities

1. **Target-count-aware row-split hybrid**
   - Keep multi8 when `target_count` is large, for example `>= 32`.
   - Try multi4 for medium tails, for example `8..31`.
   - Fall back to the existing tiled2/tiled4 path for very small tails.
   - Goal: keep the coalescing win early while recovering CTA count and occupancy late.

2. **Tune row tile size for row-split multi8**
   - Current row-split tile is effectively 256 rows per CTA path.
   - Try a smaller row tile such as 128 for large `n=4096`, which increases CTAs and may reduce underfill.
   - Risk: more partial reductions and more global traffic, so it should be benchmarked only under the `m >= 4096` guard first.

3. **Reduce `update_target_multi8_kernel` redundancy**
   - Current update CTAs sum row-tile partial dots from `dot_ws` for each target lane.
   - A separate compact reduction of target weights could reduce repeated summation, but adds another launch and workspace.
   - This should be tried after the multi8/multi4 hybrid because NCU suggests launch/grid shape is still the simpler bottleneck.

4. **Revisit compact-WY trailing update only after panel gains flatten**
   - Since `torch.bmm` is already fast and stage timings show panel dominates, replacing WY with custom Tensor Core code is lower priority right now.
   - A future experiment can build a dedicated Triton/CuTe DSL WY GEMM-like kernel for fixed QR shapes, but it should be measured against PyTorch FP32 `bmm`, not assumed faster.

5. **Mixed precision remains high risk for QR correctness**
   - FP16/FP8/NVFP4 are attractive for GEMM-like updates, but Householder QR is sensitive to reflector norms, `tau`, and orthogonality accumulation.
   - If tested, start with mixed precision only in trailing update candidates while retaining panel factorization, `tau`, `T`, and residual-sensitive reductions in FP32.


## Row-split row tile tuning after submission 828975 (2026-06-23)

Follow-up optimization after the multi8 row-split submission:

1. **Target-count-aware multi8/multi4 hybrid**
   - Added experimental `dot_partial_multi4_kernel` and `update_target_multi4_kernel`.
   - Swept `ROW_SPLIT_MULTI8_MIN_TARGETS` on `batch=1, n=4096`.
   - Results did not beat the all-multi8 policy:

| threshold | total | panel |
| ---: | ---: | ---: |
| `8` | `74.30 ms` | `63.57 ms` |
| `16` | `74.51 ms` | `63.79 ms` |
| `24` | `74.50 ms` | `63.78 ms` |
| `32` | `74.51 ms` | `63.77 ms` |
| `48` | `74.57 ms` | `63.85 ms` |

Conclusion: keep active default at threshold `8`, meaning the row-split path uses multi8 whenever it is enabled. The multi4 kernels remain in the source as an env-gated experiment but are not active by default.

2. **Row-split row tile size**
   - Added `ROW_SPLIT_ROWS_PER_TILE`, wired through CUDA source, C++ workspace allocation, extension cache tag, and build flags.
   - This changes only the row-split panel apply path; small/medium matrix paths remain unchanged.

Representative `batch=1, n=4096` stage profile:

| row tile | total | panel | note |
| ---: | ---: | ---: | --- |
| `256` | `74.33 ms` | `63.59 ms` | previous default |
| `192` | `74.02 ms` | `63.30 ms` | little benefit |
| `128` | `70.63 ms` | `59.89 ms` | clear win |
| `64` | `70.38 ms` | `59.60 ms` | best local result |

Active default is now row tile `64`:

```text
raw-cuda-small-square+raw-cuda-panel-tail-rowsplit-multi8-r64+triton-dot-raw-finish-t+fp32-gemm-wy
```

Correctness and submission:

- Local `verify_submission.py --module submission --remote-cases --trials 1`: all available cases passed.
- Representative local e2e: `n4096 dense ~69.85 ms`, `n4096 upper ~69.78 ms`.
- Submission id: `829052`
- Result: `succeeded`; public and secret `test`, `benchmark`, and `leaderboard` all passed on B200.
- No timeout observed, though remote wait was about `140s`, longer than submission `828975`.

Next direction:

1. Capture NCU for the r64 row-split path and compare against `reports/ncu_panel_rowsplit_multi8_n4096.ncu-rep`.
2. Check whether the extra row-split CTAs improved waves/SM and `no eligible`, or whether the win is mainly from shorter per-CTA row work.
3. If NCU still shows update underfill, revisit `update_target_multi8_kernel` weight reduction redundancy.


### Panel bottleneck details and CUB/intrinsics assessment (2026-06-23)

Current panel bottleneck is more about parallelism shape and memory/reduction choreography than raw floating-point throughput.

Observed panel characteristics:

1. **Panel still dominates total runtime**
   - After Triton build-T and FP32 `torch.bmm` WY update, representative stage profiles still show panel around `85%` of `n=4096` runtime.
   - With row-split r64, `batch=1, n=4096` is roughly `panel ~59.6 ms`, `build T ~5.3 ms`, `WY ~5.3 ms`, total around `70.4 ms`.

2. **The hot panel apply path is dot/update split**
   - Large matrices use `dot_partial_multi8_kernel` followed by `update_target_multi8_kernel`.
   - This creates extra global-memory traffic through `dot_ws`, but gives enough row parallelism to beat the one-CTA-per-target style apply.
   - `update_target_multi8_kernel` still repeats the small `dot_ws` summation for every row-tile update CTA, so there is some redundant reduction work.

3. **SM underfill remains visible**
   - Earlier r256 NCU showed low achieved occupancy and high `no eligible` on the multi8 row-split kernels.
   - Reducing row tile from `256` to `64` improved local `n=4096` time, likely by increasing CTA count and reducing long per-CTA row work.
   - A new r64 NCU capture is needed to confirm whether waves/SM and eligible warps improved, or whether the win mainly comes from better latency granularity.

4. **Panel factorization itself is serial by reflector**
   - Each Householder column depends on the previous column update, so the outer `j` loop is inherently sequential inside a panel.
   - This limits how much launch-level or CTA-level parallelism can be extracted without changing the algorithmic blocking strategy.
   - The best current wins have come from parallelizing each reflector's apply over rows and adjacent target columns.

5. **Memory access is structured but not GEMM-like enough**
   - Multi8 improves adjacent target-column coalescing.
   - The operation is still mostly Householder vector dot/update, not a large dense GEMM tile that Tensor Cores can consume directly.

Assessment of using CUB:

- CUB is reasonable for correctness-safe reductions, especially `BlockReduce` in `sigma_partial_kernel`, `factor_single_tile_kernel`, `finalize_scale_kernel`, and older per-target row-split reductions.
- Expected speedup is probably modest because the current `block_sum` already uses warp shuffle plus shared memory for cross-warp reduction.
- CUB may increase template compile time, register use, and extension binary size. For tiny kernels with many launches, this can be a net neutral or regression.
- Best CUB experiment would be isolated behind a macro such as `USE_CUB_BLOCK_REDUCE`, then benchmark panel stage only. Do not convert every reduction at once.

Assessment of CUDA intrinsics:

- Intrinsics are more attractive than CUB for this codebase because they can be introduced locally without changing kernel structure.
- Already used: `__shfl_down_sync` in `block_sum`.
- Reasonable next intrinsic experiments:
  - Use warp-level reductions specialized for `kRowLanes` in `dot_partial_multi8_kernel` instead of shared-memory `partial[kRowLanes][8]` plus a serial lane-0 sum.
  - Use vectorized loads/stores such as `float4` where alignment and `target_count` allow contiguous target columns. This may help multi8 dot/update more than changing reductions.
  - Consider `fmaf` explicitly in dot/update loops only if compiler SASS shows it is not already generating FFMA.
  - Keep `sqrtf` for Householder norm. Approximate `rsqrt`/fast reciprocal square root is not recommended because QR correctness and orthogonality margins depend on stable reflector scaling.

Recommendation:

1. First run NCU on the current r64 path.
2. If reductions still show as a clear stall, try a macro-gated CUB `BlockReduce` only for `block_sum` users in panel factorization, not for the multi8 shared-memory per-column reduction yet.
3. For multi8 apply, prioritize a custom warp-level/intrinsic reduction and vectorized memory experiment over CUB.
4. Avoid approximate math intrinsics for `sqrtf`, `tau`, or panel normalization unless a full residual/orthogonality sweep confirms safety.


### R64 NCU and rejected intrinsic/update experiments (2026-06-23)

Generated current r64 NCU report:

```text
reports/ncu_panel_rowsplit_multi8_r64_n4096.ncu-rep
```

First captured row-split launches, compared with the older r256 multi8 report:

| kernel | r256 duration | r64 duration | r256 achieved occupancy | r64 achieved occupancy | r64 key stalls |
| --- | ---: | ---: | ---: | ---: | --- |
| `dot_partial_multi8_kernel` | `4.29 us` | `3.55 us` | `18.62%` | `40.24%` | L1TEX scoreboard about `47.8%` |
| `update_target_multi8_kernel` | `8.32 us` | `6.21 us` | `16.19%` | `42.34%` | barrier about `52.6%`, L1TEX scoreboard about `31.6%` |

Interpretation:

- The r64 row tile improved grid size and occupancy substantially.
- `dot_partial_multi8_kernel` is now more clearly memory/coalescing limited than reduction limited.
- `update_target_multi8_kernel` still spends a lot of time at the CTA barrier because only `row_lane == 0` computes the repeated `dot_ws` sum while other row lanes wait.

Experiments attempted after this NCU pass:

1. **Warp-level dot partial reduction**
   - Changed `dot_partial_multi8_kernel` to reduce the four same-column lanes inside each warp using `__shfl_down_sync`, then write only `8 x 8` shared partials instead of `32 x 8`.
   - Correctness passed.
   - `batch=1, n=4096` stage profile regressed slightly: total about `70.53 ms` versus r64 baseline about `70.38 ms`.
   - Decision: rejected and reverted. The extra shuffle work did not beat the original simple shared-memory reduction.

2. **Separate multi8 weight reduction before update**
   - Added a temporary `reduce_weights_multi8_kernel` to sum `dot_ws` once per target tile, then an `update_target_multi8_weighted_kernel` without the barrier/repeated sum.
   - Correctness passed.
   - `n4096` verifier regressed to about `72.3-72.5 ms`.
   - Decision: rejected and reverted. The added launch/pass cost outweighed removing the repeated sum and barrier.

Active path remains:

```text
raw-cuda-small-square+raw-cuda-panel-tail-rowsplit-multi8-r64+triton-dot-raw-finish-t+fp32-gemm-wy
```

Current validated local representative timings after reverting failed experiments:

| case | e2e |
| --- | ---: |
| `batch=1, n=4096` dense | `69.85 ms` |
| `batch=1, n=4096` upper | `69.85 ms` |

Updated recommendation:

- CUB is still unlikely to provide a large win because the rejected warp-reduction experiment suggests reduction mechanics are not the dominant cost in `dot_partial_multi8`.
- Avoid adding extra per-reflector launches unless the new launch removes much more work than the weight-reduction attempt did.
- The next plausible direction is memory-layout/coalescing rather than reduction library changes: inspect NCU source counters for the exact uncoalesced lines and try a targeted load/store layout change or a different thread mapping for multi8 apply.


### Source-counter coalescing and vectorized mapping experiments (2026-06-23)

Follow-up after the r64 NCU source-counter review:

1. **Thread mapping check: multi8 vs all-multi4**
   - Used the existing `ROW_SPLIT_MULTI8_MIN_TARGETS` switch to force all row-split apply through multi4.
   - `batch=1, n=4096` result:

| policy | total | panel |
| --- | ---: | ---: |
| multi8 active, threshold `8` | `70.32 ms` | `59.57 ms` |
| all multi4, threshold `999` | `72.06 ms` | `61.32 ms` |

Conclusion: the current 8-target lane mapping is better. Extra CTA count from multi4 does not compensate for weaker adjacent-column coalescing.

2. **Householder `v` broadcast via shuffle**
   - Tried loading `v = H[row, j]` only from `col_lane == 0`, then broadcasting to the other 7 target lanes with `__shfl_sync`.
   - Motivation: reduce repeated same-address `v` loads across the 8 target lanes.
   - Result: the verifier became abnormally slow/hung on large cases, so the experiment was stopped and reverted.
   - Likely cause: the extra shuffle/control flow interacts poorly with the current predication/thread mapping, and the repeated `v` load is probably already cheap through cache/broadcast behavior.

3. **Vec4-style row mapping**
   - Added temporary `dot_partial_multi8_vec4_kernel` / `update_target_multi8_vec4_kernel`.
   - Each thread handled one row and four contiguous target columns, reducing `v` loads and making each thread's target access vector-like.
   - Correctness passed, but `n=4096` regressed to about `72.4 ms`.
   - Decision: rejected and reverted. The original scalar lane mapping gives better warp-level parallelism and coalescing despite repeated `v` loads.

Active path remains unchanged after these experiments:

```text
raw-cuda-small-square+raw-cuda-panel-tail-rowsplit-multi8-r64+triton-dot-raw-finish-t+fp32-gemm-wy
```

Current interpretation:

- The row-split multi8 scalar-lane layout is already close to the best simple mapping for row-major `H`: lanes `0..7` access adjacent target columns for the same row, which coalesces the dominant target load/store path.
- The uncoalesced source-counter warning is likely from unavoidable mixed access patterns: strided Householder column loads, `dot_ws` reductions, and predicated tail handling.
- Small local vectorization changes have not helped. Further improvement probably needs a larger structural change, such as staging Householder vectors or target tiles in shared memory across multiple reflectors, or changing panel blocking rather than only remapping lanes within one reflector apply.


### 結構性 blocking 與 packed-reflector staging 實驗（2026-06-23）

在嘗試了一系列 lane mapping / vectorization 等局部調整都無法再改善 r64 路徑後，又測試了兩個結構性方向。

1. **演算法層 blocking sweep（`nb`）**

active 預設仍是 `nb=64`。對 `nb=32/64/96/128` 做 sweep 後，在重要的 remote-style shape 上 `nb=64` 仍是總時間最佳的選擇：

| shape | 最佳 `nb` | 備註 |
| --- | ---: | --- |
| `batch=16, n=512` | `64` | `5.88 ms`；`nb=32` 會增加 WY work，更大的 `nb` 會增加 panel/build-T |
| `batch=4, n=1024` | `64` | `13.31 ms`；`96/128` 因 build-T/panel 成長而變慢 |
| `batch=2, n=2048` | `64` | `34.58 ms`；`32` 接近，但 WY 成本較高 |
| `batch=1, n=4096` | `64` | `70.38 ms`；`96/128` 雖能降 WY，但 panel/build-T 成長更多 |

代表性的 `n=4096` sweep：

| `nb` | panel | build T | WY | total |
| ---: | ---: | ---: | ---: | ---: |
| `32` | `59.87 ms` | `5.03 ms` | `10.95 ms` | `76.45 ms` |
| `64` | `59.62 ms` | `5.27 ms` | `5.29 ms` | `70.38 ms` |
| `96` | `60.54 ms` | `6.64 ms` | `4.53 ms` | `71.85 ms` |
| `128` | `61.77 ms` | `7.97 ms` | `3.56 ms` | `73.40 ms` |

結論：不調整 `nb` 政策。較大的 panel 帶來的 panel/T 成長超過了 WY 更新次數變少的好處；較小的 panel 又會付出更多 WY launch/work 成本。

2. **Packed Householder reflector staging（`vpack`）**

實作了一個臨時的 packed reflector 實驗：

- 配置 panel-local 的 `v_ws[batch, panel_cols, m - j_start]`。
- 對每個 reflector `j`，launch `pack_v_kernel` 把 `v_j` 從 row-major column 上的 strided 儲存複製到連續的 workspace。
- 在 row-split 的 `dot_partial_multi8_packed_kernel` 與 `update_target_multi8_packed_kernel` 中使用打包好的 `v_ws`。

correctness 通過，但效能反而變差：

| 版本 | `n=4096` dense | `n=4096` upper |
| --- | ---: | ---: |
| r64 baseline | 約 `69.9 ms` | 約 `69.9 ms` |
| vpack staging | 約 `76.2 ms` | 約 `76.0 ms` |

結論：以這種形式做顯式的 packed reflector staging 並不划算。每個 reflector 額外的 pack launch 加上 global 寫入/讀取 traffic，比把 `v_j` 變成連續所節省的成本還高。

更新後的結構性解讀：

- 在目前 panel/build-T/WY 平衡下，`nb=64` 是局部最佳。
- 如果一個顯式的 memory layout staging 需要每個 reflector 多一次 launch，那它就太貴。
- 未來任何 staging 都必須 fuse 進現有的 factor/finalize kernel；否則就要 stage 更多可在多個 reflector 間重用的資料，攤提才會划算。
- 真正下一步結構性的方向，應該是換一個 panel 演算法，例如真正的 blocked panel factorization（一次套用多個 reflector），或是 TSQR/GEQRT-like 的 tile 策略。這比局部 kernel tuning 是更大的 correctness 工程。


### NCU 狀態與結構性路線優先順序（2026-06-23）

目前判斷：

- 在嘗試新結構之前，沒有立即重跑 NCU 的必要。r64 報告已經顯示了相關 bottleneck：panel 主導，且 row-split apply 的限制是來自 memory/synchronization 形狀，而不是純 compute。
- 只有在出現新的結構性原型後才重跑 NCU。對同一個 r64 kernel 重複 profile 不太可能改變方向。

目前 bottleneck 摘要：

| 區域 | 目前訊號 |
| --- | --- |
| 總執行時間 | 本地 `n=4096` 約 `70 ms`，其中 panel 約 `59-60 ms` |
| `dot_partial_multi8_kernel` | r64 把 occupancy 提升到約 `40%`，duration 約 `3.55 us`；剩餘 stall 主要來自 L1TEX scoreboard / memory pattern |
| `update_target_multi8_kernel` | duration 約 `6.21 us`；主要 stall 來自 CTA barrier 與 L1TEX scoreboard |
| 局部 tuning 狀態 | reduction、weight pre-reduce、lane remap、vector-style mapping，以及顯式 `vpack` staging 都無法贏過 r64 baseline |

結構性路線優先順序：

1. **多 reflector apply 側分支實驗**
   - 最合理的下一個實驗。
   - 維持穩定的 single-reflector factorization 順序，但研究在不破壞依賴關係的前提下，能否將部分 apply 工作以 `ib=2/4` 群組化。
   - 主要限制：reflector `j+1` 必須在 reflector `j` 已套用到 column `j+1` 後才能 factor，因此不能天真地先 factor 多個 reflector 再一次 apply。

2. **真正的 blocked panel factorization**
   - 長期最有潛力的路線。
   - 會建構一小塊 panel block 的 reflector，並透過 compact block reflector 一次套用，使 panel 工作更接近 GEMM。
   - correctness 風險高，因為 `H/tau` layout 與下游的 compact-WY 路徑必須維持相容。

3. **TSQR / GEQRT-like tile route**
   - parallelism 最高，風險也最高。
   - 很可能需要一份 side implementation，而不是在現有 panel 路徑上做漸進式修改。
   - 只有在 multi-reflector / blocked panel 實驗顯示「目前 Householder panel 結構就是上限」時才值得嘗試。

近期計畫：

- 先檢視並把最安全的 `ib=2` multi-reflector apply 方案作為 side experiment 做出原型。
- 若依賴關係仍逼著我們維持相同數量的 dot/update 階段，就避免投入只會徒增 launch 或 staging traffic 的大規模重寫。
- 整個過程都保留 `multi8-r64` 作為穩定的 submission baseline。


### `ib=2` 多 reflector apply 側分支實驗（2026-06-23）

實作了一個用 `PANEL_QR_PAIR_APPLY2` 控制的 side experiment，benchmark 後再移除。

想法：

1. 正常 factor reflector `j`。
2. 只把 reflector `j` apply 到 column `j+1`，讓 column `j+1` 可以被 factor。
3. Factor reflector `j+1`。
4. 用 pair kernel 把兩個 reflector 一起 apply 到剩餘的 target column `j+2..j_end`。

這尊重了「`v_{j+1}` 必須等 reflector `j` 對 column `j+1` 完成 update 後才能 factor」這個依賴關係。

correctness：

- `PANEL_QR_PAIR_APPLY2=1` 通過了現有 remote-style verifier cases。

效能：

| 版本 | panel | total |
| --- | ---: | ---: |
| baseline r64 | `59.72 ms` | `70.47 ms` |
| pair2 實驗 | `66.29 ms` | `77.03 ms` |

verifier 在 `n=4096` 的端到端時間也退步到約 `76.5 ms`。

決定：

- 否決並從 `submission.py` 移除。
- pair2 雖然減少了部分 reflector `j` 的重複 apply，但同時加了 column `j+1` 的特殊 update，以及帶兩條 dot stream 加上 `v1^T v0` 交叉項的 pair kernel。
- 額外工作量與複雜度超過了 launch/memory 的節省。

更新後的結論：

- 淺層的 multi-reflector apply 改造不夠用。
- 真正划算的做法很可能需要一個真正的 blocked panel 形式，讓 block reflector 自然構成，而不是疊在 per-reflector factor/apply 之上。
- 若要繼續結構性的方向，應該開 side branch 做真正的 blocked panel 或 TSQR/GEQRT-style 原型；保留目前的 r64 submission 作為穩定的 production 路徑。


### 結構性側分支原型：blocked panel 與 panel-wise TSQR/GEQRT（2026-06-23）

新增 side prototype 檔案：

```text
prototype_structural_qr.py
```

這份檔案刻意不是 leaderboard submission：它回傳顯式的 `(Q, R)`，並使用顯式 residual checker。目的是在試圖把這些結構性演算法轉回 compact Householder `(H, tau)` 儲存之前，先驗證演算法本身。

已實作的原型路線：

1. **`blocked_panel_qr_explicit`**
   - 用 `torch.linalg.qr(..., mode="complete")` 對 `ib` 寬的 panel 做 factorization。
   - 把整個 panel transform apply 到 trailing column。
   - 為了驗證而 accumulate 顯式的 `Q`。
   - 它表達的是真正 blocked panel 的數學形狀，但不是有效率的實作。

2. **`tsqr_panel_qr_explicit`**
   - 對每個 `ib` 寬的 panel，先在 row tile 上做局部 factorization。
   - 對 tile `R` 區塊堆疊起來再做 QR factorization。
   - 把 tile-local 與 top-tree 的 transform 展開回完整的 panel transform。
   - 對 square QR 而言，這比較接近 GEQRT/TSQRT-style 的 panel 路線。

3. **`tsqr_geqrt_prototype`**
   - 對整個矩陣做兩層 TSQR 的玩具版本。
   - 對 square `n x n` 的 remote case 來說，只有當 `row_tile >= n` 時才有意義；否則每個 local tile 太短，無法產生 `n x n` 的 R。因此這條路線比 panel-wise TSQR 更不直接可用。

correctness smoke：

| 原型 | shape | 結果 |
| --- | --- | --- |
| blocked panel | `n=128, ib=16` | scaled reconstruction 約 `0.010`，orthogonality 約 `0.46` |
| panel-wise TSQR | `n=128, ib=16, tile=64` | scaled reconstruction 約 `0.018`，orthogonality 約 `0.44` |

`n=256, batch=1` 的方向性 sweep：

| 路線 | 代表性的本地原型最佳時間 | 備註 |
| --- | ---: | --- |
| 顯式 blocked panel | 約 `117-123 ms` | 正確但極慢，因為每個 panel 都要 materialize 完整 Q |
| panel-wise TSQR/GEQRT-like | 視 `ib/tile` 約 `25-38 ms` | 正確，且結構上更有潛力 |
| 整體矩陣 TSQR 玩具版 | `tile=n` 時約 `1.5 ms` | 實質上等於一次 `torch.linalg.qr`，不是可用的 tiled square-QR 路線 |

重要的解讀：

- 顯式 blocked panel 原型確認了正確性，但以目前寫法不是效能路徑。
- panel-wise TSQR/GEQRT-like 分解是更有潛力的結構性方向，因為它能在每個窄 panel 內暴露 row-tile parallelism。
- production 端真正的難題不只是計算顯式的 `Q/R`：leaderboard 要求的是與 `torch.linalg.householder_product` 與 `R = triu(H)` 相容的 compact Householder `(H, tau)`。
- 真正的實作必須儲存 tile/tree reflector，並且要嘛把它們轉成全域的 compact Householder 形式，要嘛在仍能通過官方 checker 的前提下改變輸出策略。這個轉換是最主要的研究風險。

如果要繼續這個分支，建議的下一步：

1. 把 `prototype_structural_qr.py` 保留為 side research 檔案。
2. 在小 shape 上設計 panel-wise TSQR reflector 的 compact 儲存方案，例如 `n=128, ib=16, row_tile=64`。
3. 在動 CUDA 之前，先寫一個 CPU/Torch 的轉換器，把 panel-wise TSQR reflector 映射成 checker 相容的 `(H, tau)`，或證明這種直接轉換在實務上不可行。
4. 只有在 compact 轉換能成立時，才開始 CUDA/CuTe 實作。


### Compact `(H, tau)` conversion feasibility for structural prototypes (2026-06-23)

Added an explicit-Q to compact-Householder feasibility oracle in `prototype_structural_qr.py`:

```python
explicit_qr_to_compact(a, q)
```

The oracle uses `torch.geqrf(q)` to obtain compact reflectors for the explicit `Q`, then sets:

```text
Qh = householder_product(Hq, tau)
R  = Qh.T @ A
H  = tril(Hq, -1) + triu(R)
```

This is not a production implementation because it calls `torch.geqrf`, but it answers the key math question: an explicit structural QR result can be represented in the checker-compatible compact `(H, tau)` format.

Feasibility results:

| route | shape | compact conversion result |
| --- | --- | --- |
| blocked panel explicit | `n=128, ib=16` | scaled reconstruction about `0.022`, orthogonality about `0.76` |
| panel-wise TSQR | `n=128, ib=16, tile=64` | scaled reconstruction about `0.025`, orthogonality about `0.77` |
| blocked panel explicit | `n=176, ib=32` | scaled reconstruction about `0.016`, orthogonality about `0.46` |
| panel-wise TSQR | `n=176, ib=32, tile=128` | scaled reconstruction about `0.019`, orthogonality about `0.47` |
| panel-wise TSQR | `n=352, ib=32, tile=128` | scaled reconstruction about `0.0075`, orthogonality about `0.41` |
| panel-wise TSQR | `n=512, ib=32, tile=128` | scaled reconstruction about `0.0062`, orthogonality about `0.32` |

Implementation note:

- The panel-wise TSQR prototype needed tail-tile handling because the final row tile in a panel can have fewer than `ib` rows. The stacked-R tree must use only the effective `min(tile_rows, ib)` R rows for that tile.

Conclusion:

- Compact output is mathematically feasible for structural routes.
- The remaining blocker is implementation, not representation: a production path cannot call `torch.geqrf(Q)` to convert explicit Q.
- The next research step is to derive or implement a custom compact conversion for the tree/panel reflectors, or to directly emit global compact Householder vectors during panel-wise TSQR.

Practical implication:

- Panel-wise TSQR/GEQRT remains the more promising structural branch.
- However, before writing CUDA kernels, the side prototype should implement a no-`torch.geqrf` compact converter on a small case and verify it with `torch.linalg.householder_product`.


### No-`torch.geqrf` compact converter prototype (2026-06-23)

Added a self-contained Torch Householder converter to `prototype_structural_qr.py`:

```python
householder_geqrf_compact_torch(x)
explicit_qr_to_compact_self(a, q)
```

This removes the dependency on `torch.geqrf(q)` for the side prototype. It computes compact Householder storage for an explicit `Q`, then rebuilds the checker format:

```text
Qh = householder_product(Hq, tau)
R  = Qh.T @ A
H  = tril(Hq, -1) + triu(R)
```

Representative CUDA smoke results:

| route | shape | explicit route time | compact conversion | scaled reconstruction | scaled orthogonality |
| --- | --- | ---: | ---: | ---: | ---: |
| blocked panel | `n=64, batch=1, ib=16` | `106.3 ms` | `81.7 ms` | `0.027` | `0.65` |
| panel-wise TSQR | `n=64, batch=1, ib=16, tile=64` | `36.1 ms` | `15.5 ms` | `0.027` | `0.69` |
| blocked panel | `n=128, batch=1, ib=16` | `106.5 ms` | `97.2 ms` | `0.027` | `0.68` |
| panel-wise TSQR | `n=128, batch=1, ib=16, tile=64` | `31.3 ms` | `30.9 ms` | `0.020` | `0.78` |
| blocked panel | `n=176, batch=2, ib=32` | `120.4 ms` | `109.9 ms` | `0.017` | `0.50` |
| panel-wise TSQR | `n=176, batch=2, ib=32, tile=128` | `33.0 ms` | `41.7 ms` | `0.023` | `0.48` |
| blocked panel | `n=352, batch=1, ib=32` | `120.9 ms` | `149.0 ms` | `0.0062` | `0.39` |
| panel-wise TSQR | `n=352, batch=1, ib=32, tile=128` | `35.1 ms` | `82.9 ms` | `0.0080` | `0.38` |

Interpretation:

- The self-contained converter confirms that compact output does not fundamentally require `torch.geqrf`.
- Correctness is acceptable for the checker-style residuals on small and medium smoke cases.
- The naive converter is far too slow because it performs an explicit global GEQRF-style conversion of materialized `Q`.
- Therefore the production path should not be `structural QR -> explicit Q -> compact conversion`.

Updated structural conclusion:

- Panel-wise TSQR/GEQRT-like decomposition is still the best research branch, but only if it emits compact reflectors directly.
- A global explicit-Q converter is useful as an oracle and debugging tool, not as an implementation strategy.
- The next practical step is to design a direct compact-emission path for one small panel: store tile-local Householder vectors plus the tree/top reflectors, then test whether they can be linearized into the official `(H, tau)` convention without materializing full `Q`.


### Direct compact blocked-panel prototype with WY apply (2026-06-23)

Added a checker-compatible blocked Householder side route to `prototype_structural_qr.py`:

```python
blocked_panel_qr_compact_torch(a, ib)
blocked_panel_qr_compact_wy_torch(a, ib)
_build_wy_t_forward(v, tau)
```

Purpose:

- Avoid the previous `structural QR -> explicit Q -> compact conversion` path.
- Emit official compact Householder `(H, tau)` directly while factoring panels.
- Validate that a blocked panel can apply trailing columns with compact WY while still returning checker-compatible output.

Implementation shape:

1. Factor only the current `ib`-wide panel sequentially and store Householder tails directly in `H`.
2. Build `V` from the stored compact panel reflectors.
3. Build LAPACK-style forward/columnwise `T` for `Q = I - V T V^T`.
4. Apply trailing columns as:

```text
C <- C - V @ (T.T @ (V.T @ C))
```

5. Return `H = work` and `tau` directly; no explicit full-matrix `Q` conversion is needed.

Representative CUDA smoke results:

| route | shape | time | scaled reconstruction | scaled orthogonality |
| --- | --- | ---: | ---: | ---: |
| full-Q blocked compact oracle | `n=64, batch=1, ib=16` | `214.5 ms` | `0.013` | `0.66` |
| compact-WY blocked panel | `n=64, batch=1, ib=16` | `19.7 ms` | `0.013` | `0.69` |
| compact-WY blocked panel | `n=128, batch=1, ib=16` | `38.4 ms` | `0.0092` | `0.70` |
| compact-WY blocked panel | `n=176, batch=2, ib=32` | `54.5 ms` | `0.0078` | `0.46` |
| compact-WY blocked panel | `n=352, batch=1, ib=32` | `104.2 ms` | `0.0037` | `0.38` |

Comparison notes:

- The compact-WY route is much faster than materializing full panel `Q`, so the `T` direction and WY formula are validated.
- It is still slower than the explicit `torch.linalg.qr` side prototype because this prototype uses Python/Torch loops for per-column panel factorization and per-panel `T` construction.
- That slowdown is expected and does not invalidate the route; it identifies exactly what must move into CUDA/CuTe.

Updated recommendation:

- Prefer direct compact blocked-panel over TSQR-to-compact conversion as the next production-oriented branch.
- The smallest useful production experiment is not full TSQR; it is replacing the current per-reflector panel/trailing updates with a true panel-blocked compact-WY update for one panel size, likely `ib=16` or `ib=32` first.
- Keep the current `submission.py` r64 baseline untouched until the blocked compact-WY CUDA path beats it on local verifier cases.

Next concrete engineering step:

1. In a side file or gated path, implement CUDA kernels for compact blocked panel factorization with `ib=16` first.
2. Reuse the existing raw CUDA `build_t` logic where possible, but build `T` per panel immediately after panel factorization.
3. Replace per-reflector trailing apply inside the panel with one WY apply per block.
4. Benchmark only `n=128/176/352` first; if those regress, do not promote to `submission.py`.


### Shape-aware blocked-panel `nb` gated experiment (2026-06-23)

Added a small gated runtime selector in `submission.py`:

```python
_runtime_block_nb(n)
```

Behavior:

- `QR_BLOCK_NB=<int>` still overrides the block size for side benchmarking.
- Without the env override, the default is now shape-aware:

```text
n == 352 -> nb = 32
n == 512 -> nb = 48
otherwise -> nb = 64
```

This keeps the large-matrix r64 behavior intact while using smaller blocks only where controlled local e2e timing showed a win.

Controlled direct sweep, dense representative cases:

| shape | best tested `nb` | median time | previous `nb=64` median | interpretation |
| --- | ---: | ---: | ---: | --- |
| `batch=40,n=176` | `64` | `1.522 ms` | `1.522 ms` | keep r64 |
| `batch=40,n=352` | `32` | `3.977 ms` | `4.272 ms` | use nb32 |
| `batch=16,n=512` | `48` | `5.768 ms` | `5.844 ms` | use nb48, small win |
| `batch=4,n=1024` | `64` | `13.225 ms` | `13.225 ms` | keep r64 |

Verifier/e2e results after enabling shape-aware default:

| case group | result |
| --- | --- |
| remote-style cases through `n=512` | all passed |
| `n=1024` dense/rankdef/nearrank/clustered | all passed |
| `n=2048` dense/rankdef | all passed |
| `n=4096` dense/upper | all passed |

Representative shape-aware timings from full verifier:

| shape | e2e time |
| --- | ---: |
| `batch=40,n=176` | `1.571 ms` |
| `batch=40,n=352` | `4.001 ms` |
| `batch=16,n=512` | `5.72-5.74 ms` across remote-style cases |
| `batch=4,n=1024` | `13.19-13.21 ms` |
| `batch=2,n=2048` | `34.36-34.38 ms` |
| `batch=1,n=4096` | `69.94-70.08 ms` |

Interpretation:

- Smaller `nb` is not a general blocked-panel improvement; it increases panel count and WY launch/GEMM work.
- The only useful wins are shape-specific: `n=352` likes `nb=32`, and `n=512` slightly likes `nb=48`.
- This is a low-risk improvement because the env override remains available and large shapes retain the previous r64 path.

Current backend tag:

```text
raw-cuda-small-square+raw-cuda-panel-tail-rowsplit-multi8-shape-nb+triton-dot-raw-finish-t+fp32-gemm-wy
```

Recommended next action:

- This version is reasonable to submit once more to check remote scoring/timeout.
- If remote score confirms the local gains, keep shape-aware `nb`; otherwise revert default to r64 and leave `QR_BLOCK_NB` as a side-benchmark hook.
- The next deeper optimization remains CUDA-level panel factorization, not further global `nb` tuning.


### Shape-aware `nb` leaderboard submission (2026-06-23)

Submitted the shape-aware block-size version:

```text
submission_id = 829235
backend = raw-cuda-small-square+raw-cuda-panel-tail-rowsplit-multi8-shape-nb+triton-dot-raw-finish-t+fp32-gemm-wy
```

Remote result:

- B200 `qr_v2`
- public test: passed
- public benchmark: passed
- public leaderboard: passed
- secret test: passed
- secret benchmark: passed
- secret leaderboard: passed
- CLI wait time: about `170s`
- no timeout

Conclusion:

- Shape-aware `nb` is safe enough to keep for now.
- Remote accepted the `n=352 -> nb=32`, `n=512 -> nb=48`, otherwise `nb=64` policy.
- Next optimization should return to panel-kernel internals; global block-size tuning now looks mostly exhausted.


### Gated Triton panel-kernel 原型（2026-06-23）

在 `submission.py` 加入了一個預設關閉的 Triton panel 實驗：

```text
QR_PANEL_BACKEND=triton
```

實作的 kernel：

```python
_panel_factor_col_triton_kernel
_panel_apply_targets_triton_kernel
_panel_factor_apply_triton_singleblock
```

範圍：

- 在單一 Triton row block 內支援 `m - j_start <= 1024`。
- row span 更大時 fallback 回 raw CUDA panel 路徑。
- 預設路徑不變，只有設定 `QR_PANEL_BACKEND=triton` 時才會啟用。
- 這個實驗每次 Triton launch 只 factor 一個 reflector，再用另一次 Triton launch 把這個 reflector apply 到 panel 內一塊 target column。

correctness：

- `QR_PANEL_BACKEND=triton` 通過了到 `n=512` 為止的 remote-style verifier cases。
- 預設路徑也在 `n=512` 之前重新驗證過，仍然正確，效能與已提交的 shape-aware baseline 等價。

代表性的 Triton-panel 端到端時間：

| shape/case | Triton panel | 預設 shape-aware baseline | 結果 |
| --- | ---: | ---: | --- |
| `batch=40,n=176,dense` | `5.48 ms` | 約 `1.6 ms` | 慢很多 |
| `batch=40,n=352,dense` | `12.04 ms` | 約 `4.0 ms` | 慢很多 |
| `batch=16,n=512,dense cond=2` | `16.87 ms` | 約 `5.7-5.8 ms` | 慢很多 |
| `batch=16,n=512` 其他 remote-style cases | `13.8-14.3 ms` | 約 `5.7-5.8 ms` | 慢很多 |

解讀：

- Triton 能表達 panel 的數學，回傳的 `(H, tau)` 也與 checker 相容。
- 但這種「每個 reflector 一次 Triton launch」的天真重寫無法競爭：launch 次數增加，又沒有 fuse 足夠的 panel 依賴鏈。
- 這個實驗作為 correctness oracle 與開發 scaffold 仍有價值，但不應在 submission 中啟用。

更新後的建議：

- 不要用天真的 Triton kernel 取代目前的 raw CUDA panel。
- 如果要再走 Triton 寫 panel，設計必須是 persistent/fused panel kernel，能在單次 launch 內處理多個 reflector step，很可能要搭配固定的小 panel 寬度或 row tile。
- 短期來看，更有潛力的方向仍然是繼續對現有 multi8 row-split kernel 做 raw CUDA source-counter 級別的調整，因為它們已經避開了最糟糕的 launch overhead，且 NCU bottleneck 已知。


### 真正 fused/register-resident 的 Triton panel 原型升為 shape-aware 預設（2026-06-23）

天真版的 Triton panel 路徑不夠用，因為它對每個 reflector launch 一次，且依賴 reflector 之間透過 global memory 的 store/load 順序。第一版「fused」嘗試把每個 step 寫回 global memory，也是無效的：`nb=2` 變成有限值但數值錯誤，更大的 `nb` 直接產生 NaN。

根本原因：

- persistent panel kernel 不能先在 global memory 更新下一個 panel column，再把它當成 block-wide synchronization model 立刻 refactor。
- panel tile 必須在多個 reflector step 之間，全程 resident 在 Triton program 的 state 裡。

實作了一條真正 register-resident 的 Triton panel 路徑：

```python
_panel_factor_apply_register_triton_kernel
_panel_factor_apply_triton_register
```

執行期 policy：

```text
n in {352, 512, 1024}: nb = 16, panel backend = triton_register
其他情況: 沿用先前的 raw CUDA panel policy，nb = 64，除非有覆寫
```

仍可覆寫：

```text
QR_PANEL_BACKEND=raw | triton | triton_fused | triton_register
QR_BLOCK_NB=<int>
```

correctness：

- 啟用 shape-aware Triton-register 預設後，完整的 remote-style verifier 通過。
- `n=2048` 與 `n=4096` 仍走先前的 raw CUDA 路徑，也仍然通過。

代表性的完整 verifier 時間：

| shape/case | 先前 shape-aware baseline | Triton-register shape-aware | 結果 |
| --- | ---: | ---: | --- |
| `batch=40,n=176,dense` | 約 `1.57 ms` | `1.59 ms` | 不變/raw 路徑 |
| `batch=40,n=352,dense` | 約 `4.00 ms` | `3.19 ms` | 變快 |
| `batch=16,n=512,dense cond=2` | 約 `5.74 ms` | `4.61 ms` | 變快 |
| `batch=16,n=512` 其他 remote-style cases | 約 `5.72-5.74 ms` | `4.52-4.62 ms` | 變快 |
| `batch=4,n=1024` remote-style cases | 約 `13.19-13.21 ms` | `8.96-9.23 ms` | 大幅變快 |
| `batch=2,n=2048` | 約 `34.36-34.38 ms` | `34.37-34.38 ms` | 不變/raw 路徑 |
| `batch=1,n=4096` | 約 `69.94-70.08 ms` | `69.81-70.04 ms` | 不變/raw 路徑 |

解讀：

- 這驗證了先前的結論：只有當 panel 真正 fused/persistent 時，Triton 才可行。
- 把 panel tile 留在 Triton tensor state 裡，避開了 global memory 的依賴 hazard，也把 launch 次數降到每個 panel 一次。
- 目前實作的限制是 `panel_cols <= 16`，但這已足以改善中等尺寸 shape。

目前的 backend 標籤：

```text
raw-cuda-small-square+shape-triton-register-panel+triton-dot-raw-finish-t+fp32-gemm-wy
```

建議的下一步行動：

- 把這個版本送上 leaderboard；本地 verifier 的收益已大到值得一次 remote 確認。
- 提交之後，profile Triton-register `n=1024` 路徑，看 `PANEL_N=24/32` 是否值得嘗試，或 register pressure 會不會把收益吃掉。


### Triton-register `PANEL_N` sweep (2026-06-23)

Generalized the register-resident Triton panel kernel from `PANEL_N=16` to support `PANEL_N=32`:

```text
QR_TRITON_PANEL_N=16 | 32
```

Important Triton constraint:

- `tl.arange(0, PANEL_N)` requires a power-of-two range.
- `PANEL_N=24` fails Triton compilation with `arange's range must be a power of 2`.
- `PANEL_N=32` is valid but too expensive to compile for `n=1024` in the current kernel shape; it was interrupted while stuck in `ptxas`.

Updated shape-aware default:

```text
n == 352 -> Triton-register panel, nb/PANEL_N = 32
n == 512 -> Triton-register panel, nb/PANEL_N = 32
n == 1024 -> Triton-register panel, nb/PANEL_N = 16
otherwise -> previous raw CUDA policy, nb = 64
```

Full verifier after this update passed all remote-style cases.

Representative timings:

| shape/case | previous Triton-register `PANEL_N=16` | updated default | result |
| --- | ---: | ---: | --- |
| `batch=40,n=352,dense` | about `3.2-3.6 ms` | `1.66 ms` | much faster |
| `batch=16,n=512,dense cond=2` | about `4.6-5.0 ms` | `2.34 ms` | much faster |
| `batch=16,n=512` remote-style cases | about `4.5-5.0 ms` | `2.19-2.31 ms` | much faster |
| `batch=4,n=1024` remote-style cases | about `9.0-9.2 ms` | `9.09-9.19 ms` | keep `PANEL_N=16` |
| `batch=2,n=2048` | about `34.4 ms` | `34.4 ms` | unchanged/raw path |
| `batch=1,n=4096` | about `70 ms` | `69.96-69.97 ms` | unchanged/raw path |

Interpretation:

- `PANEL_N=32` is very effective for `n=352/512` because it halves panel launches and still fits the register-resident tile well enough.
- `PANEL_N=32` for `n=1024` is not currently safe for submission because Triton/NVIDIA codegen spends too long in `ptxas`.
- The medium-shape path is now substantially faster; the remaining leaderboard bottleneck is likely the large raw CUDA path for `n=2048/4096`.

Recommended next step:

- Extend the variable-panel idea to large shapes: use raw CUDA for early large-row panels, then switch to Triton-register panels for the tail once `m - j_start <= 1024`.
- This requires changing the fixed-`nb` `for` loop into a dynamic `while` loop so tail panels can use `PANEL_N=32` even when the main large-shape block size remains `64`.


### Medium-shape Triton-register `PANEL_N=32` default, large-tail rollback (2026-06-23)

Promoted the safe part of the `PANEL_N` sweep:

```text
n == 352 -> Triton-register PANEL_N=32
n == 512 -> Triton-register PANEL_N=32
n == 1024 -> Triton-register PANEL_N=16
n >= 2048 -> raw CUDA panel path
```

The backend tag now records this as:

```text
raw-cuda-small-square+shape-triton-register-panel-p32p16+triton-dot-raw-finish-t+fp32-gemm-wy
```

Large-tail experiment:

- Tried switching `2048/4096` tails to Triton-register when `m - j_start <= 1024`.
- This triggered heavy Triton compilation and was interrupted while compiling, before producing useful verifier results.
- Decision: rollback large-tail auto switch. Keep `2048/4096` on raw CUDA for now.

Final full verifier after rollback:

| shape/case group | result | representative time |
| --- | --- | ---: |
| `n=352` dense | passed | `1.62 ms` |
| `n=512` remote-style cases | passed | `2.22-2.36 ms` |
| `n=1024` remote-style cases | passed | `9.20-9.28 ms` |
| `n=2048` dense/rankdef | passed | `34.38-34.40 ms` |
| `n=4096` dense/upper | passed | `70.03-70.18 ms` |

Interpretation:

- `PANEL_N=32` is a strong win for `352/512` and should stay enabled.
- `PANEL_N=32` is not safe for `1024` due excessive Triton compile/codegen cost; keep `PANEL_N=16` there.
- Large-shape optimization needs a different approach than simply compiling a `BLOCK_M=1024, PANEL_N=32` register tile in Triton.

Next optimization direction:

- For `2048/4096`, avoid new heavy Triton register kernels for now.
- Return to raw CUDA large-panel NCU work, or design a smaller row-tiled persistent Triton kernel that does not instantiate a huge `1024 x 32` register panel.


### Large raw CUDA panel experiments: multi16, row tile, and fused apply (2026-06-23)

Ran three large-shape raw CUDA panel experiments for `2048/4096`, with medium shapes forced out of the path using:

```text
QR_PANEL_BACKEND=raw
QR_BLOCK_NB=64
```

#### 1. `multi16` row-split target tile

Added a gated compile-time option:

```text
ROW_SPLIT_TARGET_TILE=16
```

Result:

| setting | `n=2048 dense` | `n=4096 dense` | `n=4096 upper` | conclusion |
| --- | ---: | ---: | ---: | --- |
| `multi8` default | `34.37 ms` | `69.99 ms` | `69.67 ms` | baseline |
| `multi16` | `34.42 ms` | `69.98 ms` | `69.72 ms` | no win |

Decision:

- Keep `multi8` as default.
- `multi16` is correct but does not improve the large path.

#### 2. `ROW_SPLIT_ROWS_PER_TILE` sweep

Tested row tile sizes:

```text
ROW_SPLIT_ROWS_PER_TILE in {32, 64, 128}
```

Result:

| row tile | `n=2048 dense` | `n=4096 dense` | `n=4096 upper` | conclusion |
| --- | ---: | ---: | ---: | --- |
| `32` | `34.38 ms` | `71.35 ms` | `71.14 ms` | slower |
| `64` | `34.37 ms` | `70.13 ms` | `69.89 ms` | best/near-best |
| `128` | `34.37 ms` | `70.17 ms` | `69.98 ms` | no win |

Decision:

- Keep `ROW_SPLIT_ROWS_PER_TILE=64`.

#### 3. Dot+update fusion proxy: disable row-split

Added a gated compile-time option:

```text
ROW_SPLIT_MIN_M=<int>
```

Experiment:

- `ROW_SPLIT_MIN_M=4096`: normal row-split path.
- `ROW_SPLIT_MIN_M=999999`: disable row-split, forcing existing fused `apply_target_tiled` dot+update path for `4096`.

Result:

| setting | `n=4096 dense` | `n=4096 upper` | conclusion |
| --- | ---: | ---: | --- |
| row-split default | `69.85 ms` | `69.72 ms` | baseline |
| forced fused tiled apply | `89.59 ms` | `89.50 ms` | much slower |

Interpretation:

- Avoiding `dot_ws` global traffic is not enough; the row-split parallelism is more important for `4096`.
- A profitable dot+update fusion would need cross-row-tile cooperation or a different reduction strategy, not simply a single CTA fused apply.

Current large-shape decision:

```text
ROW_SPLIT_TARGET_TILE=8
ROW_SPLIT_ROWS_PER_TILE=64
ROW_SPLIT_MIN_M=4096
```

Next direction for `2048/4096`:

- Keep raw CUDA row-split as the stable large path.
- The next worthwhile experiment should be source-counter guided: reduce barrier/scoreboard stalls inside `dot_partial_multi8_kernel` and `update_target_multi8_kernel`, not change high-level tiling blindly.
- If attempting fusion again, it must preserve row-tile parallelism, likely with a two-level or cooperative reduction design.


### Large row-tiled persistent panel feasibility and CUDA cooperative prototype (2026-06-23)

Added side feasibility runner:

```text
prototype_large_rowtile_persistent.py
```

Purpose:

- Estimate whether a row-tiled persistent panel is feasible for `2048/4096`.
- Compare against the current raw large path.
- Avoid implementing an incorrect Triton kernel that lacks cross-row-tile synchronization.

Key finding:

- A correct row-tiled persistent panel needs grid-wide barriers after each reflector phase:
  1. sigma partials across row tiles
  2. sigma finalization / scale broadcast
  3. panel-target dot partials across row tiles
  4. dot finalization / update broadcast
- Triton programs do not provide a grid-wide barrier, so a one-launch Triton implementation would be mathematically unsafe unless it keeps the full reduction domain in one program, which is exactly what failed for large `1024 x 32` register tiles.

Feasibility estimates on B200-class local GPU (`188 SMs`):

| shape | panel_n | row_tile | target_tile | first-panel coop blocks | active blocks/SM needed | minimum grid barriers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `4096` | `16` | `128` | `16` | `32` | `1` | `12288` |
| `4096` | `32` | `128` | `16` | `64` | `1` | `12288` |
| `4096` | `32` | `64` | `8` | `256` | `2` | `12288` |
| `2048` | `16` | `128` | `16` | `32` | `1` | `6144` |
| `2048` | `32` | `128` | `16` | `64` | `1` | `6144` |

Interpretation:

- Some cooperative CUDA grid shapes are theoretically feasible if active blocks per SM remains low.
- The barrier count is very high, so a cooperative kernel must save enough launch/global-memory overhead to compensate.

Implemented experimental gated CUDA cooperative prototype in `submission.py`:

```text
PANEL_COOP_PANEL=1
PANEL_COOP_ROW_TILE=128
QR_PANEL_BACKEND=raw
QR_BLOCK_NB=16
```

Status:

- First compile attempt found and fixed a cooperative launch argument type issue.
- The corrected cooperative prototype then took too long in compile/run and was interrupted.
- It remains gated off by default:

```text
PANEL_COOP_PANEL=0
```

Decision:

- Do not enable cooperative row-tiled persistent panel by default.
- Keep it only as an experimental scaffold.
- The next large-shape work should either use NCU source counters on the current raw row-split kernels, or design a smaller multi-phase CUDA cooperative kernel with fewer compile-time dimensions and explicit compile-time budget checks.




### Side prototype: large TSQR / GEQRT-like panel route (2026-06-23)

Added a side-only prototype:

```text
prototype_large_tsqr_panel.py
```

Scope:

- Does not modify `submission.py`.
- Splits the experiment into:
  - R-only two-level TSQR panel factorization for large `2048/4096` feasibility.
  - Explicit-Q one-panel materialization only for smaller correctness checks.

Correctness smoke:

| shape | nb | row_tile | path | result |
| --- | ---: | ---: | --- | --- |
| `512 dense` | `64` | `256` | R-only TSQR | `rel_r_err=3.21e-7` |
| `512 dense` | `64` | `256` | explicit-Q panel | `rel_recon=2.08e-7`, `orth_abs=7.39e-6` |
| `1024 dense` | `64` | `512` | R-only TSQR | `rel_r_err=3.10e-7` |
| `1024 dense` | `64` | `512` | explicit-Q panel | `rel_recon=2.22e-7`, `orth_abs=1.04e-5` |

Large panel-only timing:

| shape | nb | row_tile | TSQR R-only | Torch skinny QR reference | result |
| --- | ---: | ---: | ---: | ---: | --- |
| `2048 dense` | `64` | `512` | `1.209 ms` | `0.357 ms` | TSQR prototype slower |
| `4096 dense` | `64` | `512` | `2.255 ms` | `0.393 ms` | TSQR prototype slower |
| `4096 upper` | `64` | `512` | `2.047 ms` | `0.339 ms` | upper structure does not rescue Python/Torch TSQR |
| `4096 dense` | `32` | `512` | `1.334 ms` | `0.225 ms` | narrower panel helps but still slower |
| `4096 dense` | `16` | `512` | `0.838 ms` | `0.156 ms` | narrower panel helps but still slower |

Interpretation:

- The TSQR / GEQRT-like math route is correct enough to continue.
- The current side prototype is not a performance candidate; it launches many small QR operations and loses to cuSOLVER's skinny-panel QR.
- If extrapolated naively over all `4096/64 = 64` panels, R-only TSQR is already too expensive before trailing updates.
- A profitable large route needs a custom tiled CUDA/CUTLASS-like kernel that fuses local panel factorization, stacked-R reduction, and trailing update. Simply composing `torch.linalg.qr` calls is not viable.

Next structural direction:

- Keep this file as the oracle/measurement harness.
- If continuing TSQR, prototype a CUDA tile kernel for one fixed shape first:
  - `n=4096`, `nb=16 or 32`, `row_tile=512`.
  - Emit local tile `R_i` and compact local reflectors.
  - Factor stacked `R_i` separately.
  - Only after panel factor timing is competitive, add trailing update.
- Do not route leaderboard traffic to TSQR until the one-panel custom kernel beats the current raw panel cost.



### CUDA proof: local tile Householder R for TSQR panel (2026-06-23)

Added another side-only prototype:

```text
prototype_cuda_tsqr_panel.py
```

Scope:

- Does not modify `submission.py`.
- Implements a raw CUDA Householder QR kernel for independent row tiles of one skinny panel.
- Emits local tile `R_i` blocks, then uses Torch QR only for the small stacked-R factorization.
- This measures a more realistic lower-level route than composing many `torch.linalg.qr` calls over row tiles.

Initial bug fixed:

- The first version had an invalid block-wide reduction broadcast: only warp 0 saw the final block sum, other warps used per-warp sums during Householder updates.
- After broadcasting the reduced value through shared memory, correctness returned to expected FP32 scale.

Results:

| shape | nb | row_tile | CUDA local tile R | stacked-R Torch QR | CUDA TSQR total | Torch TSQR total | Torch panel QR | R error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `512 dense` | `16` | `256` | `0.088 ms` | `0.095 ms` | `0.154 ms` | `0.252 ms` | `0.115 ms` | `1.63e-7` |
| `4096 dense` | `16` | `512` | `0.130 ms` | `0.115 ms` | `0.225 ms` | `0.860 ms` | `0.145 ms` | `1.73e-7` |
| `4096 upper` | `16` | `512` | `0.128 ms` | `0.125 ms` | `0.219 ms` | `0.811 ms` | `0.137 ms` | `1.46e-8` |
| `2048 dense` | `32` | `512` | `0.730 ms` | `0.211 ms` | `0.916 ms` | `0.805 ms` | `0.216 ms` | `2.24e-7` |
| `4096 dense` | `32` | `512` | `0.731 ms` | `0.142 ms` | `0.850 ms` | `1.335 ms` | `0.225 ms` | `1.48e-7` |
| `4096 dense` | `32` | `256` | `0.437 ms` | `0.179 ms` | `0.588 ms` | `2.057 ms` | `0.226 ms` | `1.59e-7` |
| `4096 dense` | `64` | `256` | `1.631 ms` | `0.319 ms` | `1.922 ms` | `3.413 ms` | `0.383 ms` | `2.21e-7` |

Interpretation:

- The custom CUDA local tile QR successfully removes much of the Python/Torch row-tile launch overhead.
- `nb=16` is the only configuration that looks remotely plausible as a structural starting point.
- `nb=32/64` local Householder cost rises quickly; even before trailing updates, the R-only path is slower than direct skinny QR reference.
- A naive extrapolation is not yet attractive:
  - `4096, nb=16`: about `256` panels, so `0.225 ms * 256 ~= 57.6 ms` for R-only TSQR work before trailing apply.
  - `4096, nb=32`: about `128` panels, so `0.588 ms * 128 ~= 75.3 ms` before trailing apply.
  - `4096, nb=64`: about `64` panels, so `1.922 ms * 64 ~= 123 ms` before trailing apply.
- Therefore, this CUDA proof is encouraging as a kernel feasibility result, but not yet a clear replacement for the current raw panel baseline.

Next decision:

- Do not integrate this into `submission.py` yet.
- If continuing this route, the next proof should remove the Torch stacked-R QR and implement the tiny stacked-R factor in CUDA for the fixed `num_tiles * nb` shape.
- Only continue toward a full route if `nb=16` one-panel total can drop clearly below `0.15 ms`, because trailing apply still remains unsolved.
- The more likely high-upside variant is not plain TSQR R-only, but a GEQRT/TSQRT-style path that emits compact tile reflectors and applies multiple reflectors to the trailing matrix tile-by-tile.



### CUDA proof update: stacked-R QR in CUDA (2026-06-23)

Updated side-only prototype:

```text
prototype_cuda_tsqr_panel.py
```

Change:

- Added `stacked_householder_r_kernel` to factor the small stacked-R matrix in CUDA.
- This removes the Torch QR dependency from the one-panel TSQR proof path.
- Still does not modify or route through `submission.py`.

Smoke correctness:

| shape | nb | row_tile | full CUDA total | R error |
| --- | ---: | ---: | ---: | ---: |
| `512 dense` | `16` | `256` | `0.144 ms` | `1.49e-7` |

Large-shape results:

| shape | nb | row_tile | local tile R | stacked-R Torch | stacked-R CUDA | TSQR Torch-stack total | TSQR full-CUDA total | Torch panel QR | R error full CUDA |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4096 dense` | `16` | `512` | `0.129 ms` | `0.117 ms` | `0.075 ms` | `0.222 ms` | `0.194 ms` | `0.144 ms` | `1.33e-7` |
| `4096 upper` | `16` | `512` | `0.129 ms` | `0.113 ms` | `0.074 ms` | `0.217 ms` | `0.190 ms` | `0.132 ms` | `2.92e-8` |
| `2048 dense` | `16` | `512` | `0.128 ms` | `0.094 ms` | `0.068 ms` | `0.199 ms` | `0.186 ms` | `0.146 ms` | `1.20e-7` |
| `4096 dense` | `16` | `256` | `0.092 ms` | `0.099 ms` | `0.091 ms` | `0.168 ms` | `0.169 ms` | `0.149 ms` | `1.26e-7` |
| `4096 dense` | `16` | `128` | `0.077 ms` | `0.111 ms` | `0.127 ms` | `0.165 ms` | `0.190 ms` | `0.152 ms` | `1.30e-7` |
| `4096 dense` | `32` | `256` | `0.438 ms` | `0.179 ms` | `0.728 ms` | `0.588 ms` | `1.153 ms` | `0.223 ms` | `1.34e-7` |

Interpretation:

- CUDA stacked-R QR helps for `nb=16,row_tile=512`: total improves from `0.222 ms` to `0.194 ms`.
- The best measured one-panel TSQR proof is still around `0.165-0.169 ms` for `nb=16,row_tile=128/256`, not below the `<0.15 ms` target.
- For `row_tile=128`, local tile QR improves but stacked-R grows and dominates.
- For `row_tile=512`, stacked-R is cheap but local tile QR is slower.
- For `nb=32`, the generic stacked-R CUDA kernel is not competitive; the stack is too large for this simple one-CTA Householder kernel.

Decision:

- Plain two-kernel TSQR R-only is probably not enough to beat the current baseline after adding trailing apply.
- Further TSQR work would need one of:
  - fuse local tile R and stacked-R reduction for `nb=16` to reduce launch/global-memory overhead,
  - specialize the stacked-R kernel more aggressively for fixed `num_tiles <= 16, nb=16`,
  - or pivot to a GEQRT/TSQRT-style tiled reflector apply where the value comes from reducing the full panel/trailing-update dependency, not from R-only timing.
- Do not integrate this route into `submission.py` yet.



### Triton autotune stacked-R QR side proof (2026-06-23)

Updated side-only prototype:

```text
prototype_cuda_tsqr_panel.py
```

Change:

- Kept the CUDA local tile Householder QR kernel.
- Added a Triton `@triton.autotune` stacked-R Householder QR kernel.
- Autotune currently sweeps `num_warps in {4, 8, 16}` for fixed `NB`, `STACK_ROWS`, and `BLOCK_ROWS`.
- This remains a side prototype only; no `submission.py` integration.

Stable sequential results:

| shape | nb | row_tile | stacked Torch QR | stacked CUDA QR | stacked Triton QR | TSQR Torch-stack | TSQR full-CUDA | TSQR Triton-stack | Torch panel QR | R error Triton-stack |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `512 dense` | `16` | `256` | `0.096 ms` | `0.067 ms` | `0.042 ms` | `0.155 ms` | `0.143 ms` | `0.098 ms` | `0.110 ms` | `1.13e-7` |
| `4096 dense` | `16` | `128` | `0.112 ms` | `0.127 ms` | `0.038 ms` | `0.164 ms` | `0.189 ms` | `0.090 ms` | `0.144 ms` | `1.26e-7` |
| `4096 dense` | `16` | `256` | `0.098 ms` | `0.088 ms` | `0.035 ms` | `0.167 ms` | `0.167 ms` | `0.100 ms` | `0.146 ms` | `1.28e-7` |
| `4096 dense` | `16` | `512` | `0.302 ms` | `0.074 ms` | `0.034 ms` | `0.222 ms` | `0.192 ms` | `0.140 ms` | `0.149 ms` | `1.66e-7` |
| `4096 upper` | `16` | `128` | `0.105 ms` | `0.127 ms` | `0.038 ms` | `0.157 ms` | `0.192 ms` | `0.087 ms` | `0.132 ms` | `1.46e-8` |
| `4096 dense` | `32` | `256` | `0.177 ms` | `0.727 ms` | `0.076 ms` | `0.586 ms` | `1.156 ms` | `0.487 ms` | `0.225 ms` | `1.53e-7` |

Interpretation:

- Triton autotune is clearly better than the previous one-CTA CUDA stacked-R kernel for this small dense stacked-R QR.
- For `nb=16,row_tile=128`, the one-panel proof finally passes the earlier target: `TSQR Triton-stack ~= 0.090 ms`, below direct Torch skinny panel QR `~= 0.144 ms`.
- For `nb=32`, Triton stacked-R is fast, but local tile QR dominates; total `0.487 ms` is still too slow.
- The best structural TSQR proof point is now:

```text
nb=16, row_tile=128, CUDA local tile R + Triton autotuned stacked-R
```

Decision:

- This is the first TSQR-side result that is fast enough to justify a next structural experiment.
- Still do not integrate into `submission.py` as a QR replacement yet, because this only computes final panel R; it does not emit checker-compatible compact Householder reflectors or apply trailing updates.
- Next worthwhile side prototype: keep `nb=16,row_tile=128`, and prototype a tile-reflector/GEQRT-like trailing apply path or a representation converter. The value must come from reducing the full panel/trailing-update dependency, not only from R-only timing.



### Checker-compatible TSQR bridge and trailing update prototype (2026-06-23)

Added side-only prototype:

```text
prototype_tsqr_checker_bridge.py
```

Purpose:

- Keep `nb=16,row_tile=128` as the structural route candidate.
- Materialize explicit TSQR panel `Q_panel` as a correctness oracle.
- Apply the panel transform to the trailing matrix with blocked QR semantics:

```text
work[j:, j:] = Q_panel.T @ work[j:, j:]
```

- Accumulate explicit `Q_total`.
- Convert explicit `Q_total` to checker-compatible compact `(H,tau)` using GEQRF on `Q_total`, then set `triu(H) = triu(Q.T @ A)`.

This is intentionally not a performance path; it validates the representation bridge before writing tile-reflector / GEQRT-like apply kernels.

Results:

| case | nb | row_tile | mode | TSQR/update time | converter time | checker |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `n=128 dense` | `16` | `128` | full QR + compact bridge | `3.423 ms` | `0.602 ms` | pass |
| `n=256 dense` | `16` | `128` | full QR + compact bridge | `8.269 ms` | `1.130 ms` | pass |
| `n=512 dense` | `16` | `128` | full QR + compact bridge | `171.306 ms` first-run explicit path, `23.635 ms` one-shot bridge | `1.996 ms` | pass |
| `n=4096 dense` | `16` | `128` | first panel trailing update only | `160.187 ms` | skipped | partial reconstruction OK |

Representative checker output:

| shape | scaled factor | scaled reconstruction | scaled triangular | scaled orthogonality |
| --- | ---: | ---: | ---: | ---: |
| `128 dense` | `0.0386` | `0.0219` | `0.0379` | `0.833` |
| `256 dense` | `0.0184` | `0.0113` | `0.0182` | `0.529` |
| `512 dense` | `0.00751` | `0.00632` | `0.00738` | `0.317` |

Large partial-panel check:

- `4096 dense`, first panel only: `scaled_recon=5.36e-4`, `scaled_orth=0.0231`.
- `scaled_lower` is large in partial mode because the matrix is intentionally not fully triangular after one panel.
- This confirms the panel trailing update preserves `A ~= Q_partial @ updated_work` at large size.

Interpretation:

- The TSQR panel representation can drive a correct blocked trailing update.
- The explicit-Q-to-compact converter produces official checker-compatible `(H,tau)` for complete QR.
- This validates the bridge needed before a production GEQRT/TSQRT-like route.
- The explicit materialization and converter are too slow for submission; the next step is to avoid full `Q_panel`/`Q_total` materialization.

Next structural work:

- Fixed candidate remains:

```text
nb=16, row_tile=128
CUDA local tile R + Triton autotuned stacked-R
```

- Replace explicit `Q_panel.T @ trailing` with a tile-reflector apply path:
  - apply local tile reflectors per row tile,
  - apply stacked-R/top-tree reflectors to the compact tile coordinates,
  - keep the trailing matrix update tiled instead of materializing full `Q_panel`.
- Longer term, emit checker-compatible compact `(H,tau)` directly, or implement a cheaper representation converter from the TSQR/tree reflectors.



### TSQR tree-apply bridge without full Q_panel materialization (2026-06-23)

Updated side-only prototype:

```text
prototype_tsqr_checker_bridge.py
```

Change:

- Added a tree-apply path that no longer assembles the full `rows x rows` `Q_panel` for trailing update.
- The apply structure is now GEQRT/TSQRT-like:
  1. apply each local tile `Q_i.T` to its row tile,
  2. gather only compact coordinate rows,
  3. apply top-tree `Q_top.T`,
  4. scatter compact coordinates back.
- For checker conversion only, the prototype still materializes dense `Q_total`, because the official output format remains `(H,tau)`.

Correctness results:

| case | nb | row_tile | mode | result |
| --- | ---: | ---: | --- | --- |
| `n=128 dense` | `16` | `128` | full tree-apply + compact bridge | checker pass, `r_rel_diff=1.82e-7` vs explicit oracle |
| `n=256 dense` | `16` | `128` | full tree-apply + compact bridge | checker pass, `r_rel_diff=3.60e-7` vs explicit oracle |
| `n=512 dense` | `16` | `128` | full tree-apply + compact bridge | checker pass, `r_rel_diff=5.94e-7` vs explicit oracle |
| `n=4096 dense` | `16` | `128` | first panel tree apply only | `scaled_recon=5.04e-4`, `work_rel_diff=6.54e-8` vs explicit oracle |
| `n=4096 upper` | `16` | `128` | first panel tree apply only | exact structural match, `scaled_recon=0`, `work_rel_diff=0` |

Timing observations:

| case | explicit full-Q panel/update | tree apply |
| --- | ---: | ---: |
| `n=4096 dense`, first panel | `159.246 ms` | `9.205 ms` |
| `n=4096 upper`, first panel | `155.512 ms` | `9.084 ms` |

Interpretation:

- The TSQR tree representation can apply trailing updates correctly without materializing full `Q_panel`.
- This is a major structural improvement over the explicit-Q oracle, but it is still Torch/Python tile-loop code and not a performance candidate yet.
- The remaining production gaps are now clearer:
  1. replace explicit local `Q_i` with compact tile reflectors emitted by the local tile QR kernel,
  2. implement local tile trailing apply in CUDA/Triton,
  3. implement top-tree compact-coordinate apply in Triton/CUDA,
  4. avoid dense `Q_total` materialization by directly emitting checker-compatible compact `(H,tau)` or by implementing a cheaper tree-reflector-to-compact converter.

When this can be considered for `submission.py`:

- Minimum correctness gate:
  - full QR checker pass on representative local cases through at least `n=512`, all hard cases (`rankdef`, `band`, `rowscale`, `nearcollinear`), not just dense;
  - first-panel and multi-panel partial reconstruction checks for `n=2048/4096`;
  - no dense `Q_panel`/`Q_total` materialization in the active path.
- Minimum performance gate:
  - one `4096, nb=16,row_tile=128` panel factor + tree trailing apply must beat the current raw panel+GEMM-WY panel step by a visible margin;
  - target: first-panel tree apply should be kernelized from the current Torch `~9 ms` down to sub-ms scale, otherwise it cannot compete with the current full `4096` end-to-end time around `70 ms`;
  - all required kernels must compile within leaderboard timeout budget.
- Integration gate:
  - keep behind env flag first, e.g. `QR_PANEL_BACKEND=tsqr_tree`;
  - submit only after remote-style 19 cases pass and NCU/stage profile shows the large-shape path is actually faster.

Decision:

- Do not integrate into `submission.py` yet.
- The next useful prototype is a kernelized tree trailing apply for a single fixed case:

```text
n=4096, nb=16, row_tile=128
local tile QR: existing CUDA proof
stacked-R QR: Triton autotuned proof
trailing apply: new tile-wise CUDA/Triton proof
```



### Kernelized tree trailing-apply oracle and stage breakdown (2026-06-23)

Updated side-only prototype:

```text
prototype_tsqr_checker_bridge.py
```

Change:

- Added `apply_tsqr_tree_transpose_bmm`, a packed batched-GEMM variant of the tree trailing apply.
- This keeps the same TSQR tree semantics, but represents local tile apply as one batched GEMM plus one top-tree compact-coordinate GEMM.
- It is still an oracle because local tile `Q_i` is explicit; production must use compact tile reflectors instead.

Correctness:

- `n=128 dense`: packed-BMM tree path matches explicit oracle exactly in `R/Q` within measurement, checker pass.
- `n=4096 dense`, first panel: packed-BMM tree path has `scaled_recon=5.04e-4`, `work_rel_diff=6.54e-8` vs explicit oracle.

First-panel stage breakdown for `n=4096, nb=16,row_tile=128`:

| stage | median |
| --- | ---: |
| Torch TSQR tree factor oracle | `4.658 ms` |
| trailing tree apply, Python loop over tile GEMMs | `0.729 ms` |
| trailing tree apply, packed batched GEMM | `0.925 ms` |
| dense basis apply for converter oracle | `0.918 ms` |

Interpretation:

- The tree trailing apply itself is already sub-ms even in the Torch GEMM oracle path.
- Packed BMM is not faster here because packing/scatter overhead outweighs launch reduction.
- The dominant non-production cost is now the factor/representation oracle: Torch QR emits explicit local `Q_i`, while the fast CUDA proof currently emits only local `R_i`.
- Therefore, hand-writing a Triton/CUDA trailing apply kernel is not the next highest leverage until local tile QR emits compact reflectors and `tau`.

Updated integration gate:

- Before touching `submission.py`, the side path needs a fixed-shape local tile QR kernel that emits:
  - tile-local compact Householder storage for each `128 x 16` tile,
  - tile-local `tau`,
  - local `R_i` for Triton stacked-R QR.
- Then the tree apply can use compact reflectors directly:
  - local tile apply: apply each tile's 16 reflectors to its trailing rows,
  - top-tree apply: use Triton stacked-R reflectors/top transform on compact coordinate rows.
- Only after this reflector-emitting path passes first-panel reconstruction for `2048/4096` and beats the current raw panel step should it be hidden behind a `QR_PANEL_BACKEND=tsqr_tree` flag.



### CUDA local tile QR now emits compact reflectors (2026-06-23)

Updated side-only prototype:

```text
prototype_cuda_tsqr_panel.py
```

Change:

- Added `local_tile_householder_compact` to the inline CUDA proof extension.
- The local tile QR kernel now emits three outputs for each `row_tile x nb` tile:
  - `H_tile`: compact Householder storage with upper `R_i` and lower reflector tails,
  - `tau_tile`: tile-local Householder coefficients,
  - `R_i`: the same local R block used by stacked-R QR.
- Existing `local_tile_householder_r` remains unchanged for A/B timing.

Validation method:

- Reconstruct reduced tile `Q_i` with `torch.linalg.householder_product(H_tile, tau_tile)`.
- Check `Q_i.T @ tile_panel` against emitted `R_i`.
- Compare emitted `R_i` against the old R-only CUDA kernel.

Stable sequential results for `nb=16,row_tile=128`:

| case | local R-only | local compact | tile scaled recon | tile scaled lower | tile scaled orth | R block diff | TSQR Triton-stack total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `512 dense` | `0.078 ms` | `0.078 ms` | `0.0278` | `0.0146` | `4.8` | `0.0` | `0.084 ms` |
| `4096 dense` | `0.073 ms` | `0.076 ms` | `0.0331` | `0.0187` | `6.39` | `0.0` | `0.088 ms` |
| `4096 upper` | `0.074 ms` | `0.076 ms` | `0.00383` | `0.0` | `4.0` | `0.0` | `0.087 ms` |

Interpretation:

- Emitting compact tile reflectors costs almost nothing over the previous R-only local tile QR proof.
- The emitted `R_i` is bitwise/effectively identical to the old R-only path (`R block diff = 0`).
- The tile-local compact representation is numerically healthy and can replace explicit `Q_i` in the next tree-apply prototype.

Remaining production gap:

- `torch.linalg.householder_product` was used only for validation. A submission path cannot reconstruct explicit `Q_i`.
- The next required side prototype should use `H_tile/tau_tile` directly to apply local tile reflectors to trailing tiles:

```text
for local reflector j in 0..15:
    C[j:, :] -= tau[j] * v_j * (v_j.T @ C[j:, :])
```

- Once local compact-reflector apply matches the current tree-apply oracle on `4096` first-panel reconstruction, the top-tree compact-coordinate apply can be tackled next.



### Compact-reflector local apply kernel and Triton top apply (2026-06-23)

Updated side-only prototypes:

```text
prototype_cuda_tsqr_panel.py
prototype_tsqr_checker_bridge.py
```

Change:

- Added `local_compact_apply_qt` CUDA kernel.
  - Inputs: `H_tile`, `tau_tile`, and a trailing matrix tile.
  - Applies the 16 tile-local Householder reflectors directly, without materializing `Q_i`.
- Added `top_compact_apply_qt_triton` Triton autotuned kernel.
  - Applies the top-tree compact reflectors to compact coordinate rows.
- Added bridge path:

```text
CUDA local tile QR -> H_tile/tau_tile/R_i
Triton stacked-R QR / compact top QR oracle
CUDA local compact apply
Triton top compact-coordinate apply
```

Correctness:

- `n=128 dense`: compact CUDA-local path checker pass.
- `n=4096 dense`, first panel: `scaled_recon=3.71e-4`, `work_rel_diff~=2.53e-7` vs explicit oracle.
- `n=4096 upper`, first panel: CUDA+Triton top apply matches Torch top apply exactly in the measured comparison.

Stage breakdown for `n=4096, nb=16,row_tile=128`:

| stage | dense median |
| --- | ---: |
| local compact tile QR / representation emit | `0.154 ms` |
| local compact apply, CUDA, `block_n=8` | `1.474 ms` |
| top compact-coordinate apply, Triton | `0.049 ms` |
| full local+top compact update | `1.518 ms` |
| previous local CUDA + Torch top | `2.971 ms` |
| previous all-Torch compact apply | `45.907 ms` |

Upper sanity:

| stage | upper median |
| --- | ---: |
| full local+top compact update | `1.517 ms` |
| local CUDA + Torch top | `1.787 ms` |
| relative diff | `0.0` |

Interpretation:

- The compact reflector path is now structurally close to a submission backend for one panel update.
- The top-tree apply is no longer a bottleneck after Triton; it is around `0.05 ms`.
- The remaining large cost in the panel update is the local compact apply kernel (`~1.47 ms` for the first `4096` panel).
- Combined first-panel factor+apply is roughly `1.67 ms` before any final `(H,tau)` output conversion problem.

Current blocker before adapting to `submission.py`:

- The computation path can update the trailing matrix without explicit `Q_i`, but the official output must still be a single checker-compatible compact `(H,tau)` for the whole matrix.
- The current TSQR/tree reflector representation is not the same as LAPACK-style sequential compact Householder layout expected by `torch.linalg.householder_product(H,tau)`.
- Existing checker bridge still uses dense `Q_total -> geqrf(Q_total)` as an oracle, which is too slow and cannot be used in submission.

What must be true before integration:

1. Implement or avoid the tree-reflector-to-standard-compact conversion.
   - Option A: derive a cheaper converter from tree reflectors to standard `(H,tau)`.
   - Option B: restructure the TSQR path to emit standard sequential reflectors while preserving enough parallelism.
   - Option C: use TSQR only as an internal update accelerator while standard raw panel still emits official reflectors, if that can be made mathematically consistent.
2. Optimize local compact apply below the current `~1.47 ms` first-panel cost.
   - Potential knobs: vectorize columns, specialize `nb=16,row_tile=128,block_n=8`, reduce per-reflector shared syncs, or use warp-per-column reductions.
3. Full local hard-case correctness through `n=512`, plus `2048/4096` first/multi-panel partial reconstruction.
4. Only then add an env-gated `submission.py` path, e.g. `QR_PANEL_BACKEND=tsqr_tree`, for controlled A/B profiling.

Decision:

- Still not ready to adapt into `submission.py` as an active path.
- The next most direct engineering task is local compact apply optimization, but the bigger architectural blocker is official `(H,tau)` emission. Without solving that, a fast internal TSQR update path cannot pass the checker as a standalone submission backend.



### Paper Algorithm 2: explicit TSQR Q to WY reconstruction (2026-06-23)

Reference:

- Local PDF: `docs/High_Performance_Householder_QR_Factorization_on_Emerging_GPU_Architectures.pdf`
- Relevant idea: TSQR accelerates panel factorization but emits an explicit/thin `Q`; the paper reconstructs a WY representation from explicit `Q` via non-pivoting LU on `S - Q` / `I - Q`.

Side-only implementation:

```text
prototype_tsqr_checker_bridge.py
```

Added:

- `nonpivoting_lu_square`
- `reconstruct_wy_from_explicit_q`
- `wy_reconstruction_residual`
- `tsqr_blocked_paper_wy_apply`

Implemented paper-style reconstruction:

```text
A = S_thin - Q_thin
[L1, U] = non_pivoting_LU(A[:k, :])
L2 * U = A[k:, :]
Y = [L1; L2]
W * L1.T = A
Q_wy = S - W * Y.T
```

Important correction learned during validation:

- For a TSQR panel, the complete-Q nullspace is not unique.
- A rank-`k` WY representation should be validated against the first `k` thin-Q columns, not against an arbitrary `m x m` complete `Q`.
- On `4096` first-panel tests, full trailing work differs from the previous explicit complete-Q TSQR update because the WY route chooses a different canonical orthogonal completion. This is expected; the key requirement is that the resulting blocked QR remains valid.

Results:

| case | path | result |
| --- | --- | ---: |
| `n=128 dense, nb=16,row_tile=128` | `paper_wy_recon` | `k=128`, `wy_rel_err=2.49e-7`, `40.8 ms` Torch oracle |
| `n=128 dense, nb=16,row_tile=128` | `paper_wy_blocked` | checker pass, `scaled_recon=0.0184`, `scaled_orth=0.484`, `34.3 ms` one-shot |
| `n=4096 dense, max_panels=1` | `paper_wy_recon` | `k=16`, `wy_rel_err=5.28e-8`, `38.4 ms` Torch oracle |
| `n=4096 dense, max_panels=1` | `paper_wy_partial` | `scaled_recon=8.5e-4`, `scaled_orth=0.0368`, `12.5 ms` one-shot |

Interpretation:

- The paper route is mathematically viable: TSQR thin-Q can be reconstructed into a WY/canonical Householder completion and can drive a valid blocked QR on small full cases.
- This is a better bridge than the old dense `Q_total -> geqrf(Q_total)` oracle because it targets the panel-level representation problem directly.
- The current implementation is still Torch-heavy and not submission-ready. The `4096` reconstruction spends tens of ms mostly on dense `Q` materialization and triangular solves.

Next implementation direction:

1. Avoid materializing complete `Q_panel`; reconstruct WY from the compact TSQR tree's thin basis only.
2. CUDA/Triton-specialize the tiny no-pivot LU/TRSM for `k=16` or `k=32`.
3. Decide output strategy:
   - Either convert WY to checker-compatible `(H,tau)`;
   - Or use WY internally for trailing update while keeping a standard panel path for final `(H,tau)`.
4. Re-test full `n=512` and partial `2048/4096` after replacing the Torch reconstruction pieces.



### CUDA/Triton paper-WY converter and apply prototype (2026-06-23)

Updated side-only prototypes:

```text
prototype_cuda_tsqr_panel.py
prototype_tsqr_checker_bridge.py
```

Change:

- Added CUDA extension function `reconstruct_wy_lu`.
  - Input: complete panel `Q` oracle and target `k`.
  - Uses fixed `A = I - Q_thin` sign convention.
  - Performs small no-pivot LU on the leading `k x k` block.
  - Emits `W`, `Y`, and `signs`.
- Added Triton WY apply:

```text
Q_wy = S - W * Y.T
Q_wy.T * C = S*C - Y*(W.T*C)
```

- Added side path `tsqr_blocked_paper_wy_cuda_triton_apply`.
  - Still uses explicit TSQR `Q_panel` as source, so this is not submission-ready.
  - It removes Torch LU/TRSM and avoids materializing dense `Q_wy` for the trailing update.

Bug found and fixed:

- CUDA converter initially read shared `signs` before all threads had written them.
- Missing `__syncthreads()` caused wrong `A = S - Q` and invalid LU results.
- A second issue came from using diagonal sign selection while the Torch oracle's stable path was `identity_minus_q`; the CUDA path now uses `I - Q` to match the validated route.

Correctness:

| case | check | result |
| --- | --- | ---: |
| `n=128, k=16` | CUDA `W/Y` vs Torch `W/Y` | `~5e-8` relative |
| `n=128, k=16` | Triton apply vs Torch formula | `~2.7e-7` relative |
| `n=4096, k=16` | CUDA `W/Y` vs Torch `W/Y` | `~5e-8` relative |
| `n=4096, k=16` | CUDA/Triton partial update vs Torch paper-WY update | `3.26e-7` relative |
| `n=4096, k=16` | self partial reconstruction | `scaled_recon ~= 4.71e-4` |

Clean in-process kernel timing after warmup:

| case | CUDA WY reconstruct | Triton WY apply |
| --- | ---: | ---: |
| `n=128,k=16` | `0.056 ms` | `0.045 ms` |
| `n=4096,k=16` | `0.238 ms` | `0.347 ms` |

Important timing caveat:

- The CLI `one_shot` line for `paper_wy_cuda_tri` includes Triton first-call compile/autotune overhead and is not representative.
- Clean in-process timing shows the converter/apply kernels are sub-ms for `4096,k=16`.

Current status:

- CUDA/Triton paper-WY converter/apply is now correct as a side prototype.
- It still depends on explicit `Q_panel` from `tsqr_panel_factor_explicit`, which is the wrong source for submission.
- The next real step is to feed the converter from compact TSQR tree/thin basis without materializing complete `Q_panel`.



### Official checker tolerance notes (2026-06-23)

Source:

```text
profile/qr_official.py
```

The official/local checker only hard-fails two residuals:

```text
factor: ||R - Q.T @ A||_1 <= 20 * n * eps_float32 * ||A||_1
orth:   ||Q.T @ Q - I||_1 <= 100 * n * eps_float32 * ||I||_1
```

With `eps_float32 ~= 1.192e-7`, this gives:

| n | factor_rtol | orth_rtol |
| ---: | ---: | ---: |
| `128` | `3.05e-4` | `1.53e-3` |
| `512` | `1.22e-3` | `6.10e-3` |
| `2048` | `4.88e-3` | `2.44e-2` |
| `4096` | `9.77e-3` | `4.88e-2` |

Important nuance:

- `scaled_reconstruction_residual` and `scaled_triangular_residual` are printed but not used as fail conditions.
- Therefore, for aggressive approximate routes, the important gates are:
  - `scaled_factor_residual <= 20`
  - `scaled_orthogonality_residual <= 100`
- This tolerance is fairly loose at large `n`, but the output must still be valid `(H,tau)` because the checker reconstructs `Q` through `torch.linalg.householder_product(H,tau)`.



### Thin-Q input for CUDA/Triton paper-WY path (2026-06-23)

Change:

- Updated CUDA `reconstruct_wy_lu` to accept `q` with shape `(batch, rows, q_cols)` where `q_cols >= k`; it no longer requires a square complete-Q input.
- Updated Python wrapper to pass only `q[:, :, :k]`.
- Added `tsqr_panel_factor_thin` Torch oracle that materializes only TSQR `Q_thin` (`rows x k`) instead of complete `Q_panel` (`rows x rows`).
- Added side path:

```text
tsqr_panel_factor_thin -> CUDA reconstruct_wy_lu -> Triton WY apply
```

Purpose:

- This does not yet remove the Torch TSQR oracle, but it changes the converter/apply dataflow to the production-like interface: only thin panel basis is needed.
- The next replacement target is `tsqr_panel_factor_thin` itself: compute the same thin basis from compact TSQR tree reflectors without building complete local or panel Q.

Validation:

| case | path | result |
| --- | --- | ---: |
| `n=128,max_panels=1` | thin-Q CUDA/Triton vs complete-Q CUDA/Triton | `0.0` relative diff |
| `n=128,max_panels=1` | thin-Q CUDA/Triton vs Torch paper-WY | `1.83e-7` relative diff |
| `n=4096,max_panels=1` | thin-Q CUDA/Triton vs complete-Q CUDA/Triton | `0.0` relative diff |
| `n=4096,max_panels=1` | thin-Q CUDA/Triton vs Torch paper-WY | `3.26e-7` relative diff |

Clean in-process stage timing after warmup:

| case | Torch TSQR thin-Q oracle | CUDA WY reconstruct | Triton WY apply |
| --- | ---: | ---: | ---: |
| `n=128,k=16` | `0.297 ms` | `0.057 ms` | `0.043 ms` |
| `n=4096,k=16` | `4.228 ms` | `0.222 ms` | `0.353 ms` |

Interpretation:

- The converter now truly only needs `Q_thin`; complete panel `Q` is no longer part of the CUDA/Triton converter/apply interface.
- The current bottleneck in this side route is `tsqr_panel_factor_thin`, not WY reconstruction or WY application.
- Next optimization target: generate `Q_thin` from CUDA compact tile reflectors and top-tree compact reflectors, or avoid explicit `Q_thin` entirely by deriving `Y/W` from the compact tree representation.



### Compact-tree generated thin-Q basis (2026-06-23)

Updated side-only prototypes:

```text
prototype_cuda_tsqr_panel.py
prototype_tsqr_checker_bridge.py
```

Change:

- Added CUDA `local_compact_apply_q`, the forward-`Q` counterpart of the existing `local_compact_apply_qt`.
- Added Triton `top_compact_apply_q_triton`, the forward-`Q` counterpart of `top_compact_apply_qt_triton`.
- Added `tsqr_panel_factor_thin_compact_cuda_triton`:

```text
CUDA local tile compact QR -> H_tile/tau_tile/R_i
Torch geqrf top compact QR on stacked R
Triton apply top Q to compact coordinate basis
CUDA apply local tile Q to form Q_thin
```

- Added partial update path:

```text
compact-tree Q_thin -> CUDA reconstruct_wy_lu -> Triton WY apply
```

Correctness:

| case | check | result |
| --- | --- | ---: |
| `n=128,max_panels=1` | compact-tree thin path vs Torch paper-WY | `2.33e-7` relative diff |
| `n=128,max_panels=1` | compact-tree thin path vs Torch thin-Q CUDA path | `2.50e-7` relative diff |
| `n=2048,max_panels=1` | compact-tree thin path vs Torch paper-WY | `3.28e-7` relative diff |
| `n=2048,max_panels=1` | compact-tree thin path vs Torch thin-Q CUDA path | `7.86e-8` relative diff |
| `n=4096,max_panels=1` | compact-tree thin path vs Torch paper-WY | `3.29e-7` relative diff |
| `n=4096,max_panels=1` | compact-tree thin path vs Torch thin-Q CUDA path | `6.57e-8` relative diff |

Clean in-process stage timing after warmup:

| case | Torch TSQR thin-Q oracle | compact-tree Q_thin | CUDA WY reconstruct | Triton WY apply | total compact-tree WY panel |
| --- | ---: | ---: | ---: | ---: | ---: |
| `n=128,k=16` | `0.266 ms` | `0.361 ms` | `0.057 ms` | `0.039 ms` | `0.457 ms` |
| `n=4096,k=16` | `4.198 ms` | `0.411 ms` | `0.223 ms` | `0.355 ms` | `0.989 ms` |

Interpretation:

- The compact-tree thin-basis route removes the Torch TSQR thin-Q bottleneck for large panels.
- For `4096,k=16`, the first-panel structural route is now roughly:

```text
compact Q_thin generation: 0.41 ms
WY reconstruction:         0.22 ms
WY trailing apply:         0.36 ms
total:                     0.99 ms
```

- This is now competitive as an internal first-panel update prototype.
- However, it is still not ready for `submission.py` because it does not emit the official global `(H,tau)` output layout.

Submission integration gate:

1. Correctness gate:
   - Full QR checker pass for at least `n=128/256/512` across dense plus hard cases (`upper`, `rankdef`, `nearrank`, `band`, `nearcollinear`, `rowscale`).
   - Multi-panel partial reconstruction sanity for `2048/4096`, not only first panel.
   - `scaled_factor_residual <= 20` and `scaled_orthogonality_residual <= 100` under `profile/qr_official.py`.
2. Output-format gate:
   - Either implement a cheap converter from the WY/tree representation into checker-compatible `(H,tau)`;
   - Or use the TSQR/WY route only as an internal trailing-update accelerator while preserving a standard compact Householder `(H,tau)` emission path.
3. Performance gate:
   - End-to-end local benchmark must improve the current `submission.py` large-size path, not just one panel.
   - The route must avoid expensive Triton first-call/autotune costs inside leaderboard timing, likely through precompiled/fixed configs or env-gated warmup.
4. Integration gate:
   - Add behind an env flag first, e.g. `QR_PANEL_BACKEND=tsqr_wy_tree`.
   - Run profile verify/benchmark locally before any submit.

Current answer to "when can it be connected to submission":

- Not yet as the default path.
- It can be connected as an env-gated experimental path once the output-format gate is solved or once we decide to keep standard raw panel factorization for `(H,tau)` and use this route only for trailing updates.
- The next coding step should be a small full-blocked prototype using compact-tree WY panels for multiple panels, then decide whether its final representation can be converted cheaply enough for the official checker.



### Multi-panel compact-tree WY full checker bridge (2026-06-23)

Updated side-only prototype:

```text
prototype_tsqr_checker_bridge.py
```

Change:

- Added `materialize_wy_q` for side-only dense `Q = S - W Y.T` materialization.
- Added `tsqr_blocked_paper_wy_compact_thin_full`.
- The full bridge now does:

```text
for each panel:
    compact-tree Q_thin
    CUDA reconstruct W/Y/signs
    Triton apply Q_wy.T to trailing matrix
    side-only materialize Q_wy and accumulate Q_total
final:
    explicit_qr_to_compact(data, Q_total) -> official (H,tau)
```

Correctness:

| case | checker | scaled factor | scaled orth | note |
| --- | --- | ---: | ---: | --- |
| `n=128 dense` | pass | `0.0341` | `0.757` | tolerance gates: factor `<=20`, orth `<=100` |
| `n=256 dense` | pass | `0.0166` | `0.497` | multi-panel pass |
| `n=512 dense` | pass | `0.00888` | `0.325` | multi-panel pass |

Observed one-shot bridge times:

| case | `cmpthin_wy_full` one-shot |
| --- | ---: |
| `n=128 dense` | `2200 ms` |
| `n=256 dense` | `4162 ms` |
| `n=512 dense` | `8380 ms` |

Interpretation:

- This route is numerically viable for multiple panels and passes the official checker through `n=512 dense`.
- The enormous full-bridge time is expected and not representative of the internal panel update speed:
  - it materializes dense panel `Q_wy`,
  - accumulates dense `Q_total`,
  - converts dense `Q_total -> (H,tau)` via `geqrf`,
  - and includes side-prototype Python/Triton first-call overhead.
- Therefore, this validates correctness but not production performance.

Updated integration decision:

- We can now justify adding an env-gated experimental path only if its output strategy is explicit:
  1. **Internal accelerator mode:** keep the current raw/standard panel path for official `(H,tau)`, but use compact-tree WY for selected large trailing updates if mathematically consistent with the emitted reflectors.
  2. **Converter mode:** implement a cheap WY/tree-to-standard `(H,tau)` converter, avoiding dense `Q_total`.
- Do not wire this as default submission yet.
- Before an env-gated integration attempt, run:
  - hard-case full checker for `n=128/256/512`,
  - multi-panel partial reconstruction for `n=2048/4096`,
  - compare current `submission.py` large-size stage timing against compact-tree WY internal update timing.



### Env-gated submission hook for compact-tree WY route (2026-06-23)

Updated:

```text
submission.py
```

Change:

- Added opt-in backend:

```bash
QR_PANEL_BACKEND=tsqr_wy_tree
```

- Added safety limit:

```bash
QR_TSQR_WY_MAX_N=512   # default
QR_TSQR_WY_NB=16       # default
QR_TSQR_WY_ROW_TILE=128
```

- The hook lazy-imports the side prototypes only when the env flag is set:

```text
prototype_cuda_tsqr_panel.py
prototype_tsqr_checker_bridge.py
```

- If import/build/runtime fails, it returns `None` and `custom_kernel` falls back to the existing default backend.
- Default submission behavior is unchanged when the env flag is absent.

Validation through `submission.custom_kernel`:

| env | case | checker | scaled factor | scaled orth |
| --- | --- | --- | ---: | ---: |
| default | `n=128 dense` | pass | `0.0392` | `0.751` |
| `QR_PANEL_BACKEND=tsqr_wy_tree QR_TSQR_WY_MAX_N=128` | `n=128 dense` | pass | `0.0341` | `0.757` |
| `QR_PANEL_BACKEND=tsqr_wy_tree QR_TSQR_WY_MAX_N=256` | `n=256 dense` | pass | `0.0166` | `0.518` |
| `QR_PANEL_BACKEND=tsqr_wy_tree QR_TSQR_WY_MAX_N=512` | `n=512 dense` | pass | `0.00888` | `0.325` |

Current limitation:

- This is still not suitable for real submission because it uses the side-only dense `Q_total -> (H,tau)` bridge.
- Its purpose is local A/B validation and stage timing only.
- The path is intentionally hidden behind env vars and should not affect leaderboard submissions unless explicitly enabled.

Next:

- Add a profile script / stage timer that calls `submission.custom_kernel` with `QR_PANEL_BACKEND=tsqr_wy_tree`.
- Compare:
  - default large-shape time,
  - env-gated full bridge time,
  - side-prototype internal panel stage timing.
- If internal panel timing remains promising, decide between:
  - implementing a cheap `(H,tau)` converter,
  - or using compact-tree WY only for trailing updates while preserving standard Householder output.



### TSQR/WY env backend profile script (2026-06-23)

Added:

```text
profile_tsqr_wy_backend.py
```

Note:

- `profile/` is root-owned in the current workspace, so the script was added at repo root.
- It compares default `submission.custom_kernel` against:

```bash
QR_PANEL_BACKEND=tsqr_wy_tree
```

- It is intended for local A/B and regression checks, not submission.

Command used:

```bash
CUDA_VISIBLE_DEVICES=2 python3 profile_tsqr_wy_backend.py \
  --ns 128,256,512 --batch 1 --case dense --trials 1 --warmup 1
```

Results:

| n | default ms | tsqr_wy_tree ms | ratio | default checker | tsqr checker |
| ---: | ---: | ---: | ---: | --- | --- |
| `128` | `1.195` | `5.166` | `4.32x slower` | pass | pass |
| `256` | `2.283` | `9.436` | `4.13x slower` | pass | pass |
| `512` | `3.032` | `19.108` | `6.30x slower` | pass | pass |

Interpretation:

- The env-gated backend is correctly wired through `submission.custom_kernel`.
- It is much slower end-to-end because it still performs the side-only dense `Q_total -> (H,tau)` bridge.
- This confirms the right next optimization is **not** more panel-kernel tuning inside the full bridge; the blocker is output-format conversion or a hybrid path that preserves standard `(H,tau)` while using compact-tree WY only where it is mathematically consistent.



### Hybrid standard-output WY update experiment (2026-06-23)

Added side-only prototype:

```text
prototype_hybrid_wy_update.py
```

Purpose:

- Test the more practical hybrid idea:

```text
standard raw panel -> checker-compatible H/tau
same standard reflectors -> alternative trailing update kernel
```

- This preserves output-format correctness because the trailing update uses the exact same standard compact Householder reflectors emitted in `H,tau`.

Method:

- Factor first panel with existing raw CUDA panel path.
- Build compact WY `T` with existing `_build_compact_wy_t_triton_dot`.
- Compare current GEMM-WY update:

```text
C -= V * T * (V.T * C)
```

- Against side Triton equivalent using existing `apply_wy_qt_triton`:

```text
Q.T*C = C - Y*(W.T*C)
Y = V
W = V*T.T
```

Results:

| case | GEMM-WY update | Triton WY-equivalent | relative diff | result |
| --- | ---: | ---: | ---: | --- |
| `n=512, nb=16` | `0.109 ms` | `0.174 ms` | `1.15e-7` | Triton slower |
| `n=4096, nb=16` | `0.268 ms` | `0.543 ms` | `9.85e-8` | Triton slower |
| `n=4096, nb=64` | `0.269 ms` | `1.572 ms` | `2.48e-7` | Triton much slower |

Interpretation:

- The hybrid math is valid and preserves standard `(H,tau)` semantics.
- However, the current Triton WY apply is not a good replacement for the existing GEMM-WY update.
- GEMM-WY is already extremely fast for this stage; hand-written reduction/update Triton loses.

Updated direction:

- Do not integrate this Triton WY update into `submission.py`.
- The hybrid idea is still the safer output-format strategy, but it needs a better accelerator than the current Triton apply.
- Near-term useful directions:
  1. Focus on panel factorization/output generation, not GEMM-WY update.
  2. If pursuing hybrid update, use true GEMM/Tensor Core formulation rather than custom reduction kernels.
  3. For the TSQR/tree route, the output-format converter remains the main unresolved blocker.



### WY/tree conversion reference check: HPDC 2020 neural-engine QR paper (2026-06-23)

Checked local PDF:

```text
docs/3369583.3392685.pdf
```

Paper:

```text
High Accuracy Matrix Computations on Neural Engines: A Study of QR Factorization and its Applications
```

Main takeaway:

- This paper is useful as an algorithmic direction reference, but it does **not** provide the missing WY/tree-to-standard-compact `(H,tau)` converter.
- Its core QR method is Recursive Gram-Schmidt QR (`RGSQRF`) plus a Communication-Avoiding Gram-Schmidt panel, not Householder TSQR/WY tree conversion.
- Householder and WY appear mostly as background / related work references.

Relevant methods from the paper:

1. Recursive Gram-Schmidt QR:

```text
A = [A1 | A2]
A1 = Q1 R11
R12 = Q1.T A2
A2 <- A2 - Q1 R12
A2 = Q2 R22
Q = [Q1 Q2]
R = [[R11, R12],
     [0,   R22]]
```

2. CAQR-style panel:

- Split a tall panel into independent `256 x 32` row tiles.
- QR each tile independently using Modified Gram-Schmidt in shared memory.
- Stack the small `R_i` blocks and recursively factor the stacked matrix.
- Apply the top-level `Q` blocks back to local tile `Q_i`.
- Their implementation uses one row per thread, block-level reductions with CUB, and batched GEMM for inter-tile communication.

3. Accuracy recovery:

- The paper relies on scaling, iterative refinement, and optional reorthogonalization to regain accuracy when using low precision / TensorCore-heavy paths.

Implication for our current QR contest path:

- It reinforces our profiling conclusion: simply accelerating the trailing update with TensorCore/GEMM is not enough when panel/output generation dominates.
- It supports a more radical side prototype: approximate/loose-tolerance CAQR/MGS or recursive Gram-Schmidt panel for large `2048/4096` cases.
- However, direct RGSQRF/CAQR output is explicit `Q,R`, not the checker-required compact Householder `(H,tau)`.
- Therefore it does not remove the current integration blocker:

```text
fast TSQR/WY/tree internal representation
        -> checker-compatible standard compact Householder H,tau
```

Recommended use of this reference:

- Do not chase it as a WY/tree converter source.
- If we use it, treat it as a separate side branch:
  - `256 x 32` MGS/CAQR panel prototype,
  - recursive stacked-`R` reduction,
  - TensorCore/GEMM-heavy trailing/update path,
  - then separately solve how to emit or approximate checker-compatible `(H,tau)`.

Near-term decision:

- Keep the current TSQR/WY converter investigation focused on either:
  1. cheap tree/WY -> standard `(H,tau)` conversion, or
  2. hybrid standard-output route where standard Householder reflectors remain the official output.
- Use the HPDC 2020 paper only if we decide to open an aggressive Gram-Schmidt/CAQR approximation side branch.



### Direct WY/tree to standard compact `(H,tau)` prototype (2026-06-23)

Added side-only implementation:

```text
prototype_tsqr_checker_bridge.py
profile_wy_direct_compact.py
```

Key idea:

- For each compact-tree TSQR/WY panel, avoid materializing global `Q_total`.
- The paper-style reconstruction gives:

```text
Q_panel ~= I - W * Y.T
```

- Treat `Y` as the standard compact reflector storage `V`.
- Solve the tiny panel system:

```text
W = Y * T.T
```

- Use:

```text
tau = diag(T)
H[j:, j:j+jb].lower = Y.lower
H.upper = final R from the WY-updated work matrix
```

- This produces checker-compatible sequential compact Householder output panel-by-panel.

Validation:

```text
source .venv/bin/activate
source ./env.sh
export CUDA_HOME="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
PYTHONPYCACHEPREFIX=/tmp/qr_pycache CUDA_VISIBLE_DEVICES=2 \
  python3 profile_wy_direct_compact.py --ns 1024,2048,4096 --batch 1 --case dense --nb 16 --row-tile 128 --warmup 1
```

Results:

| n | warmed `wy_direct_compact` | checker | scaled factor | scaled orth |
| ---: | ---: | --- | ---: | ---: |
| `128` | `8.58 ms` | pass | `0.0387` | `0.883` |
| `256` | `17.39 ms` | pass | `0.0244` | `0.581` |
| `512` | `35.88 ms` | pass | `0.00875` | `0.408` |
| `1024` | `51.25 ms` | pass | `0.00514` | `0.273` |
| `2048` | `105.25 ms` | pass | `0.00279` | `0.200` |
| `4096` | `246.64 ms` | pass | `0.00154` | `0.146` |

Additional diagnostic:

- The optimistic full-WY shortcut also passed for `n=128/256/512` when applied to dense `Q_total`, but that route still needs a global `Q_total`.
- The panel-by-panel direct output path is the important result because it avoids the old dense `Q_total -> geqrf(Q_total)` bridge.

Interpretation:

- This substantially weakens the previous output-format blocker: TSQR/WY tree panels can emit official `(H,tau)` directly.
- Correctness is strong under the official checker tolerance, including `4096 dense`.
- The current implementation is still side-only and slower than the default `submission.py` large-size path.
- Major remaining costs:
  - Python loop over every `nb=16` panel.
  - Dynamic-shape Triton compile/autotune overhead on first run.
  - Multiple small kernels per panel: compact-tree thin basis, CUDA WY reconstruction, tiny triangular solve, Triton WY apply.
  - No fused persistent panel loop yet.

Next direction:

1. Move `wy_direct_compact` behind an env-gated `submission.py` backend only after trimming per-panel overhead.
2. Fuse the panel path:

```text
compact-tree Q_thin -> WY reconstruct -> tau extraction -> trailing apply -> H/tau writeback
```

3. Specialize to fixed `nb=16,row_tile=128` and reduce dynamic Triton shape specialization.
4. Profile against current default at `2048/4096`; the direct converter is correct, but needs a persistent/fused implementation before it can beat the raw CUDA panel path.



### Env-gated submission hook for direct WY/tree compact output (2026-06-24)

Implemented a local A/B hook in:

```text
submission.py
```

New backend flag:

```text
QR_PANEL_BACKEND=tsqr_wy_direct
QR_TSQR_WY_DIRECT_MAX_N=4096
QR_TSQR_WY_NB=16
QR_TSQR_WY_ROW_TILE=128
```

Behavior:

- Default submission path is unchanged.
- The new backend is off unless `QR_PANEL_BACKEND=tsqr_wy_direct` is set.
- It calls the side prototype `tsqr_blocked_paper_wy_compact_thin_standard_output`, which emits `(H,tau)` directly instead of using dense `Q_total -> geqrf`.
- Added fallback hygiene so top-level TSQR backend flags are not accidentally treated as regular panel backend names if the experimental path returns `None`.

Local `submission.custom_kernel` A/B result, warmed, dense case:

| n | default `custom_kernel` | `tsqr_wy_direct` | checker | decision |
| ---: | ---: | ---: | --- | --- |
| `1024` | `10.17 ms` | `43.53 ms` | pass | keep side-only |
| `2048` | `32.42 ms` | `90.20 ms` | pass | keep side-only |

Interpretation:

- Integration is functionally successful: the direct WY/tree compact output can be called through `submission.custom_kernel` and passes the official checker.
- It is not submission-ready as the default path because it is currently `~2.8-4.3x` slower than the existing backend on tested large cases.
- This confirms the converter blocker is solved enough for experimentation, but the implementation still needs structural fusion/persistent execution before it can compete.

Current stage breakdown for `n=2048, nb=16,row_tile=128` after removing per-panel metric sync:

| stage | total time | share |
| --- | ---: | ---: |
| compact-tree `Q_thin` generation | `53.84 ms` | largest |
| CUDA WY reconstruction | `12.91 ms` | medium |
| `tau=diag(T)` extraction | `15.05 ms` | medium |
| H/tau writeback copy | `5.01 ms` | small |
| Triton WY trailing apply | `13.41 ms` | medium |
| total staged sum | `100.24 ms` | close to measured `~90 ms` |

Block-size experiment:

- `nb=32` was slower than `nb=16` on `1024/2048`.
- `nb=64` was even slower on `1024` in the partial run.
- Therefore simply increasing panel width is not the right fix; per-panel kernels get heavier faster than panel count decreases.

Next optimization order before considering default submission integration:

1. Optimize `Q_thin` generation first. It is the largest stage.
2. Remove Python/Torch overhead from `tau=diag(T)` extraction with a tiny fixed-`nb=16` CUDA kernel.
3. Fuse per-panel sequence to reduce launches:

```text
compact-tree Q_thin -> WY reconstruct -> tau extraction -> H lower writeback -> WY apply
```

4. Only re-test submission integration after `tsqr_wy_direct` approaches or beats default on `2048/4096`.



### Fixed-nb16 Triton tau extraction and Q_thin integration assessment (2026-06-24)

Implemented in:

```text
prototype_tsqr_checker_bridge.py
```

Changes:

- Added fixed small-panel Triton kernel:

```text
wy_tau_diag16_triton
```

- Replaces the per-panel Torch path:

```text
torch.linalg.solve_triangular(y_top, w_top, upper=False, unitriangular=True)
tau = diag(T)
```

- The Triton kernel computes the unit-lower `16 x 16` forward solve in one program and stores only `diag(T)`.
- Also removed the per-panel `signs` CPU sync in the non-validation path.

Correctness check:

| check | result |
| --- | ---: |
| `max_abs(tau_triton - tau_torch)` | `1.19e-7` |

Warmed direct backend results after this change:

| n | before | after fixed-nb16 Triton tau | checker |
| ---: | ---: | ---: | --- |
| `1024` | `43.3-43.5 ms` | `31.5 ms` | pass |
| `2048` | `89.8-90.2 ms` | `68.7 ms` | pass |
| `4096` | `220.2 ms` | `180.9 ms` | pass |

Updated `n=2048` stage breakdown:

| stage | after fixed-nb16 Triton tau | previous |
| --- | ---: | ---: |
| compact-tree `Q_thin` generation | `53.22 ms` | `53.84 ms` |
| CUDA WY reconstruction | `13.02 ms` | `12.91 ms` |
| `tau=diag(T)` extraction | `5.62 ms` | `15.05 ms` |
| H/tau writeback copy | `5.06 ms` | `5.01 ms` |
| Triton WY trailing apply | `12.34 ms` | `13.41 ms` |
| staged sum | `89.25 ms` | `100.24 ms` |

Submission A/B after this change:

| n | default `custom_kernel` | `QR_PANEL_BACKEND=tsqr_wy_direct` | decision |
| ---: | ---: | ---: | --- |
| `1024` | `10.08 ms` | `31.70 ms` | keep side-only |
| `2048` | `32.85 ms` | `68.90 ms` | keep side-only |

Q_thin scatter/init experiment:

- Added a fast Q_thin path that avoids Python `coord_positions`/advanced-index scatter by using a Triton scatter/init kernel.
- It matched the old Q_thin path exactly:

```text
q_rel = 0.0
r_rel = 0.0
```

- End-to-end timing barely changed:

| n | direct after tau kernel | direct after fast Q_thin scatter |
| ---: | ---: | ---: |
| `1024` | `31.49 ms` | `31.36 ms` |
| `2048` | `68.70 ms` | `68.23 ms` |
| `4096` | `180.88 ms` | `181.84 ms` |

Interpretation:

- The fixed-`nb=16` Triton tau kernel was worthwhile and should remain in the side path.
- The Q_thin bottleneck is not Python scatter/indexing.
- The remaining Q_thin cost is structural: per panel it still needs local tile QR, top stacked-R QR, top compact apply, and local compact apply.

Persistent/fused CUDA assessment:

- A true single-kernel persistent Q_thin generator is difficult because the TSQR panel has global dependencies:
  - local tile QR blocks must all finish before stacked-R/top QR can start,
  - top QR must finish before top coordinate apply,
  - top coordinate apply must finish before local tile reflectors can expand Q_thin.
- A normal CUDA kernel has no grid-wide synchronization between independent CTAs.
- A true one-kernel design would need cooperative groups/grid sync or would have to process many tiles serially inside one CTA, which is likely too slow.
- Triton is even less suitable for the full persistent panel because programs cannot coordinate global phase barriers cleanly across tiles.

More realistic CUDA route:

1. Keep at least three phases, but implement them as purpose-built CUDA kernels instead of Python/Torch/Triton orchestration:

```text
phase 1: local tile compact QR -> H_tile, tau_tile, R_i
phase 2: stacked-R top QR -> H_top, tau_top, R_panel
phase 3: top coordinate apply + local compact apply -> Q_thin or directly WY input
```

2. The highest-value fusion target is phase 3:

```text
top compact apply -> basis/scatter -> local compact apply
```

- It currently materializes intermediate coordinate/basis tensors and launches separate kernels.
- A CUDA phase-3 kernel could keep the top-coordinate tile slice in shared/registers and immediately apply local tile reflectors.
- However, because top compact apply needs reductions over all stacked-R rows, a fully per-tile independent fusion is not trivial; either full top coordinates are materialized once, or the top apply is recomputed per tile, which is likely worse.

3. Representation-level alternative:

- Avoid Q_thin generation entirely by deriving the WY converter input directly from the compact TSQR tree.
- This is harder mathematically, but it attacks the largest remaining stage (`~53 ms` at `2048`) instead of shaving launch overhead.

Current recommendation:

- Keep `tsqr_wy_direct` env-gated, not default.
- Next implementation attempt should be CUDA phase-3 fusion or a direct compact-tree-to-WY-input derivation.
- Do not spend more time on Triton scatter/indexing for Q_thin; measured benefit is negligible.



### NCU profile for large `tsqr_wy_direct` side path (2026-06-24)

Generated NCU reports under:

```text
reports/ncu_tsqr_direct_n2048_*.ncu-rep
reports/ncu_tsqr_direct_n2048_*_details.txt
reports/ncu_tsqr_direct_n4096_*.ncu-rep
reports/ncu_tsqr_direct_n4096_*_details.txt
```

Target script:

```text
profile_wy_direct_ncu_target.py
```

NCU setup notes:

- Regular `ncu` hit `ERR_NVGPUCTRPERM`, so reports were captured with `sudo ncu`.
- `sudo` could not JIT-load the PyTorch CUDA extension because root lacked the Python/ninja environment.
- Workaround:
  - prebuild the TSQR extension into `/tmp/qr_torch_ext`,
  - pass `--prebuilt-ext /tmp/qr_torch_ext/qr_cuda_tsqr_panel_proof_ext_v13/qr_cuda_tsqr_panel_proof_ext_v13.so` to the NCU target.
- The target script warms up once before `cudaProfilerStart()`, then profiles one measured run.

Representative NCU captures, `n=2048`, `batch=1`, `nb=16`, `row_tile=128`:

| kernel | duration | grid | waves/SM | achieved occupancy | mem throughput | SM throughput | main stalls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `local_tile_householder_compact_kernel` | `65.89 us` | `16` | `0.01` | `16.66%` | `2.94%` | `1.70%` | short scoreboard `5.82`, barrier `5.58` |
| `_top_compact_apply_q_triton_kernel` | `8.19 us` | `1` | `0.00` | `8.31%` | `4.06%` | `0.12%` | long scoreboard `2.59` |
| `local_compact_apply_q_kernel` | `114.98 us` | `16` | `0.02` | `16.67%` | `1.86%` | `1.81%` | short scoreboard `4.30`, barrier `2.98` |
| `reconstruct_wy_lu_kernel` | `126.18 us` | `1` | `0.00` | `16.32%` | `0.52%` | `0.13%` | long scoreboard `3.56`, barrier `3.43` |
| `_wy_tau_diag16_triton_kernel` | `7.17 us` | `1` | `0.00` | `8.30%` | `4.68%` | `0.16%` | short scoreboard `4.87` |
| `_wy_tmp_triton_kernel` | `77.57 us` | `2048` | `10.89` | `16.62%` | `65.20%` | `7.79%` | LG throttle `21.02`, long scoreboard `10.79` |
| `_wy_apply_triton_kernel` | `18.46 us` | `16384` | `7.26` | `87.62%` | `64.11%` | `60.02%` | long scoreboard `16.80`, LG throttle `7.00` |

Representative NCU captures, `n=4096`:

| kernel | duration | grid | waves/SM | achieved occupancy | mem throughput | SM throughput | main stalls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `local_compact_apply_q_kernel` | `114.40 us` | `32` | `0.03` | `16.67%` | `3.72%` | `3.61%` | short scoreboard `4.30`, barrier `2.98` |
| `_wy_tmp_triton_kernel` | `294.69 us` | `4096` | `21.79` | `16.64%` | `76.70%` | `9.21%` | long scoreboard `22.55`, LG throttle `9.27` |
| `_wy_apply_triton_kernel` | `79.14 us` | `65536` | `29.05` | `89.37%` | `71.16%` | `58.87%` | long scoreboard `18.32`, LG throttle `7.53` |

Interpretation:

- The Q_thin/tree pipeline is dominated by tiny-grid kernels, not by raw memory bandwidth or compute throughput.
- `local_tile_householder_compact_kernel`, `local_compact_apply_q_kernel`, `reconstruct_wy_lu_kernel`, and tau extraction launch with only `1-32` CTAs for representative panels; this is far below `188` SMs.
- These kernels show high barrier / shared-memory scoreboard stalls and very low waves/SM, so source-level tuning inside one CTA will have limited upside unless grid parallelism changes.
- The trailing WY kernels are qualitatively different:
  - `_wy_tmp_triton_kernel` and `_wy_apply_triton_kernel` have large grids and high memory throughput.
  - `_wy_apply_triton_kernel` is already reasonably saturated (`~64-71%` memory, `~59-60%` SM throughput), so it is not the next main target.

Avoid-Qthin proof result:

- Added `tsqr_panel_wy_compact_tree_direct` and extension function `reconstruct_wy_lu_from_tree`.
- Correctness matched the old Q_thin route exactly for a single panel:

```text
w rel 0.0
y rel 0.0
signs rel 0.0
r rel 0.0
```

- End-to-end timing did not materially improve:

| n | Q_thin fast route | avoid-Qthin proof |
| ---: | ---: | ---: |
| `1024` | `~31.5 ms` | `33.3 ms` |
| `2048` | `~68.7 ms` | `69.8 ms` |
| `4096` | `~180.9 ms` | `173.0 ms` |

- Therefore simply avoiding Q_thin global materialization is not enough; the limiting cost is the small-grid phase structure and repeated per-panel launches.

Next direction from NCU:

1. Do not spend more time on local CTA micro-tuning of the current Q_thin kernels unless it also increases grid parallelism.
2. A competitive `tsqr_wy_direct` would need a structural redesign that batches/fuses work across panels or across multiple independent matrices, not just top/local apply fusion inside one panel.
3. For the current single-matrix case, the existing default `submission.py` raw Householder path remains much faster because it avoids the TSQR tree's many small-grid global-dependency phases.
4. Keep `tsqr_wy_direct` as env-gated research path; do not submit it as default.
