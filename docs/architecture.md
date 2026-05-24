# Mini NPU Accelerator — Micro-Architecture Specification (MAS)

> **Project:** INT8 Systolic Array Neural Processing Unit for Conv2D/GEMM  
> **Target Workload:** MobileNet Inference (Depthwise Separable Convolution)  
> **RTL Language:** SystemVerilog | **Verification:** Cocotb + Python/NumPy  
> **Interface:** AMBA AXI4-Lite Slave

---

## Table of Contents

1. [Introduction & Project Scope](#1-introduction--project-scope)
2. [Top-level Block Diagram](#2-top-level-block-diagram)
3. [System Interface](#3-system-interface)
4. [Memory Map & Register Map](#4-memory-map--register-map)
5. [Micro-Architecture Details](#5-micro-architecture-details)
6. [Dataflow Execution](#6-dataflow-execution)
7. [Verification Plan](#7-verification-plan)
8. [Resource Estimate & Target Platform](#8-resource-estimate--target-platform)

---

## 1. Introduction & Project Scope

Dự án **Mini NPU Accelerator** là một bộ tăng tốc phần cứng (Hardware Accelerator) được thiết kế để thực thi các phép toán **Convolution 2D (Conv2D)** và **General Matrix Multiply (GEMM)**, nhắm đến việc hỗ trợ inference mạng nơ-ron MobileNet trên thiết bị biên (Edge AI).

### Design Goals

| Mục tiêu | Giá trị |
|---|---|
| Compute precision | INT8 (weights & activations) |
| Accumulator width | INT32 (tránh overflow) |
| Array size | 8×8 = 64 PEs đồng thời |
| Peak throughput | 6.4 GOPS @ 100 MHz |
| Host interface | AMBA AXI4-Lite Slave |
| Clock domains | 1 (single-clock, no CDC) |

### Design Constraints

- **AXI4-Lite only** (không dùng AXI4-Full): loại bỏ Burst transfer, Out-of-order execution, Cache coherence — giữ logic điều khiển bus tối giản.
- **INT8 compute**: giảm hàng chục lần diện tích bộ nhân so với FP32/FP16.
- **Software-managed tiling**: CPU Host chịu trách nhiệm cắt nhỏ layer mạng thành tiles vừa SRAM, NPU hardware ở mức tối giản.
- **im2col preprocessing**: Conv2D được CPU chuyển thành GEMM trước khi nạp vào NPU, tránh phải implement sliding-window memory controller trên hardware.

---

## 2. Top-level Block Diagram

```
CPU Host (ARM / RISC-V)
        │
        │ AXI4-Lite (32-bit addr, 32-bit data)
        ▼
┌─────────────────────────────────────────────┐
│              Mini NPU Accelerator            │
│                                             │
│  ┌──────────────┐    ┌────────────────────┐ │
│  │ AXI4-Lite    │    │   Register File    │ │
│  │ Slave IF     │───▶│ CTRL/STATUS/DIM_*  │ │
│  │              │    │ QUANT_SCALE/SHIFT  │ │
│  └──────┬───────┘    └────────┬───────────┘ │
│         │                    │ start/config  │
│         │ write data         ▼               │
│         ▼            ┌──────────────┐        │
│  ┌──────────────┐    │ Controller   │        │
│  │  SRAM_W 4KB  │◀───│    FSM       │        │
│  │  SRAM_X 4KB  │◀───│              │        │
│  │  SRAM_Y 4KB  │───▶│ (DMA-like)  │        │
│  └──────────────┘    └──────┬───────┘        │
│                             │                 │
│                    ┌────────▼────────┐        │
│                    │  Skewing FIFOs  │        │
│                    │  (8 shift regs) │        │
│                    └────────┬────────┘        │
│                             │                 │
│                    ┌────────▼────────┐        │
│                    │ Systolic Array  │        │
│                    │    8 × 8 PEs   │        │
│                    │ (Weight-Station)│        │
│                    └────────┬────────┘        │
│                             │                 │
│                    ┌────────▼────────┐        │
│                    │ Deskewing FIFOs │        │
│                    └────────┬────────┘        │
│                             │                 │
│                    ┌────────▼────────┐        │
│                    │ Activation Unit │        │
│                    │ ReLU→Scale→Clamp│        │
│                    └────────┬────────┘        │
│                             │ INT8 output     │
│                             ▼                 │
│                         SRAM_Y                │
└─────────────────────────────────────────────┘
        │ irq (DONE interrupt)
        ▼
   CPU Host
```

### Bốn phân hệ logic chính

| # | Module | Vai trò |
|---|---|---|
| 1 | **AXI4-Lite Slave IF** | Cửa ngõ duy nhất, decode địa chỉ → SRAM / Register File |
| 2 | **Internal SRAMs** | SRAM_W / SRAM_X / SRAM_Y — memory banking tránh port contention |
| 3 | **Controller FSM** | Điều phối địa chỉ đọc, nạp weight, stream activation, đếm M/K/N |
| 4 | **Compute Engine + Activation Unit** | 64 MAC/cycle + ReLU + Quantization 3-stage pipeline |

---

## 3. System Interface

NPU hoạt động hoàn toàn như một **AXI4-Lite Slave** — không có Master interface, không tự truy cập DDR. CPU phải sao chép dữ liệu vào SRAM nội bộ trước khi trigger computation.

### Signal Table

| Signal | Direction | Width | Description |
|---|---|---|---|
| `clk` | Input | 1 | System clock. Toàn bộ logic đồng bộ với rising edge. |
| `rst_n` | Input | 1 | Active-Low synchronous reset. Reset FSM, registers, pointers. **Không** clear SRAM content. |
| **Write Address Channel** | | | |
| `s_axi_awvalid` | Input | 1 | Master: địa chỉ ghi hợp lệ |
| `s_axi_awready` | Output | 1 | NPU: sẵn sàng nhận địa chỉ ghi |
| `s_axi_awaddr` | Input | 32 | Địa chỉ ghi (decode → SRAM hoặc Register File) |
| **Write Data Channel** | | | |
| `s_axi_wvalid` | Input | 1 | Master: data ghi hợp lệ |
| `s_axi_wready` | Output | 1 | NPU: sẵn sàng nhận data |
| `s_axi_wdata` | Input | 32 | Data ghi. Đóng gói 4× INT8 trong 1 word 32-bit. |
| `s_axi_wstrb` | Input | 4 | Byte enable. Khuyến nghị `4'b1111` khi đổ SRAM. |
| **Write Response Channel** | | | |
| `s_axi_bvalid` | Output | 1 | NPU: phản hồi ghi hợp lệ |
| `s_axi_bready` | Input | 1 | Master: sẵn sàng nhận phản hồi |
| `s_axi_bresp` | Output | 2 | `2'b00` = OKAY, `2'b10` = SLVERR (địa chỉ không hợp lệ) |
| **Read Address Channel** | | | |
| `s_axi_arvalid` | Input | 1 | Master: địa chỉ đọc hợp lệ |
| `s_axi_arready` | Output | 1 | NPU: sẵn sàng nhận địa chỉ đọc |
| `s_axi_araddr` | Input | 32 | Địa chỉ đọc (SRAM_Y hoặc STATUS register) |
| **Read Data Channel** | | | |
| `s_axi_rvalid` | Output | 1 | NPU: data đọc hợp lệ (latency = 1 cycle sau arvalid∧arready) |
| `s_axi_rready` | Input | 1 | Master: sẵn sàng nhận data |
| `s_axi_rdata` | Output | 32 | Data đọc. 4× INT8 đóng gói khi đọc SRAM_Y. |
| `s_axi_rresp` | Output | 2 | `2'b00` = OKAY, `2'b10` = SLVERR |
| **`irq`** | Output | 1 | ⚡ Interrupt Request. Bật khi `DONE=1`, tự hạ khi CPU ghi `START` mới. Kết nối GIC/NVIC của SoC. |

### AXI4-Lite Transaction Timing

- **Write:** Nếu CPU raise `awvalid` và `wvalid` đồng thời → NPU raise `awready` và `wready` cùng lúc → zero wait-state.
- **Read:** Latency = 1 cycle tính từ khi `arvalid ∧ arready`. `rvalid` bật đúng 1 cycle sau.

---

## 4. Memory Map & Register Map

### 4.1 Address Space Overview

NPU được cấp phát **64 KB** địa chỉ. Base address do hệ thống quy định (ví dụ `0x4000_0000` trong Vivado). Bảng dưới đây dùng địa chỉ **offset**.

| Region | Offset Start | Size | Description |
|---|---|---|---|
| **Control & Config Registers** | `0x0000_0000` | 256 B | Thanh ghi điều khiển M/K/N, quantization, status |
| **Weight SRAM (SRAM_W)** | `0x0000_1000` | 4 KB | Ma trận Trọng số B — 4096 tham số INT8 (64 tile 8×8) |
| **Input SRAM (SRAM_X)** | `0x0000_2000` | 4 KB | Ma trận Đầu vào A — Activation data |
| **Output SRAM (SRAM_Y)** | `0x0000_3000` | 4 KB | Ma trận Kết quả C — INT8 output sau ReLU + Quant |

> **Lưu ý quan trọng về SOFT_RST:** `SOFT_RST` **KHÔNG** xóa nội dung SRAM (SRAM không có synchronous clear). Chỉ FSM state và address pointers được reset. CPU phải ghi lại dữ liệu vào SRAM trước mỗi lần trigger `START` mới sau khi reset.

### 4.2 K-Tiling & Accumulation Strategy ⚠️

> **Vấn đề:** Khi `K > 8` (ví dụ MobileNet layer với K=512), phép nhân phải được tile theo chiều K:
> `C = A[:,0:8]×B[0:8,:] + A[:,8:16]×B[8:16,:] + ...`

Kiến trúc tích hợp một **Accumulation Buffer INT32 nội bộ** (256 bytes = 64× INT32) để xử lý cross-tile accumulation mà không cần CPU can thiệp:

- **Intermediate tiles** (`ACCUM_EN=1`): Kết quả INT32 từ Systolic Array bypass Activation Unit, cộng dồn vào Accumulation Buffer.
- **Final tile** (`ACCUM_EN=0`): FSM gộp buffer với kết quả tile cuối, chạy qua ReLU + Quantization → ghi INT8 vào SRAM_Y.

### 4.3 Register File Detail

| Offset | Name | Access | Reset | Description |
|---|---|---|---|---|
| `0x0000` | `CTRL` | R/W | `0x0` | **Master Control Register**<br>• Bit 0 `START`: Auto-clearing, kích hoạt FSM<br>• Bit 1 `SOFT_RST`: Auto-clearing, reset FSM+pointers (SRAM preserved)<br>• Bit 2 `ACCUM_EN`: `1` = K-tile accumulation mode, `0` = final tile với ReLU+Quant<br>• Bit [31:3]: Reserved |
| `0x0004` | `STATUS` | RO | `0x1` | **System Status Register**<br>• Bit 0 `IDLE`: NPU nhàn rỗi, sẵn sàng nhận config<br>• Bit 1 `BUSY`: FSM đang trong pha Pre-load hoặc Computing<br>• Bit 2 `DONE`: Kết quả hợp lệ trong SRAM_Y. Tự xóa khi CPU ghi `START`<br>• Bit [31:3]: Reserved |
| `0x0008` | `DIM_M` | R/W | `0x8` | Số hàng ma trận A. Max = 255. |
| `0x000C` | `DIM_K` | R/W | `0x8` | Inner dimension (số cột A = số hàng B). Max = 255. |
| `0x0010` | `DIM_N` | R/W | `0x8` | Số cột ma trận B = số Output Feature Maps. Max = 255. |
| `0x0014` | `QUANT_SCALE` | R/W | `0x1` | Quantization scale factor. **Chỉ bits [15:0] có hiệu lực** (UINT16). Phép nhân INT32×UINT16 → 48-bit; giữ lại bits [47:16]. |
| `0x0018` | `QUANT_SHIFT` | R/W | `0x0` | Số bit dịch phải (arithmetic right-shift) sau Scale stage. |

### 4.4 CPU Software Flow

```
1. Ghi SRAM_W (offset 0x1000): ma trận Weight B
2. Ghi SRAM_X (offset 0x2000): ma trận Activation A  
3. Ghi DIM_M, DIM_K, DIM_N, QUANT_SCALE, QUANT_SHIFT
4. (Nếu K-tiling: ghi ACCUM_EN=1 cho intermediate tiles)
5. Ghi CTRL = 0x1 (START pulse)
6. Poll STATUS tại 0x0004 cho đến khi DONE=1 (hoặc chờ irq)
7. Đọc SRAM_Y (offset 0x3000): kết quả INT8
```

---

## 5. Micro-Architecture Details

### 5.1 Processing Element (PE)

PE là đơn vị tính toán nguyên thủy, thiết kế theo **Weight-Stationary dataflow**.

```
        weight_load_en
              │
  w_in ──────▼──── [Weight Reg 8-bit] ──── w_out (xuống PE dưới)
                            │
  act_in ──────────────────▶[MUL 8×8]──▶[32-bit signed result]
                                              │
  psum_in ─────────────────────────────────▶[ADD 32-bit]
                                              │
                                        [Psum_Out_Reg]──▶ psum_out (xuống PE dưới)
              │
         [Act_Out_Reg]──▶ act_out (sang PE bên phải)
```

**Phương trình tại PE(row, col) tại cycle t:**
```
psum_out[row][col][t+1] = weight[row][col] × act_in[row][col][t]
                        + psum_in[row][col][t]
act_out[row][col][t+1]  = act_in[row][col][t]
```

**Pipeline latency:** 1 cycle/PE → critical path qua chuỗi adder trong một column.

### 5.2 Systolic Array 8×8

64 PEs kết nối theo mạng lưới 2D, **nearest-neighbor only** (không có cross-routing hay global shared bus).

**Data flow:**
- `act_in` vào từ cột trái (Col 0), lan truyền từ trái → phải
- `psum_in` vào từ hàng trên (Row 0, giá trị 0), lan truyền từ trên → dưới
- Kết quả `psum_out` thoát từ đáy (Row 7)

**⚠️ Zero-Padding khi M/K/N không chia hết cho 8:**

Khi `K mod 8 ≠ 0` (ví dụ K=10, tile thứ 2 chỉ có 2 phần tử hợp lệ), FSM phải áp dụng **Input Masking**: chèn `0x00` vào 6 slot còn lại. Logic này nằm trong Address Generation Unit của FSM (trạng thái `ST_STREAM_ACT`), sử dụng bộ đếm `valid_count` để phân biệt dữ liệu thực và padding.

### 5.3 Controller FSM & Skewing/Deskewing Logic

#### Skewing Logic (trước khi vào array)

Dữ liệu phải vào mảng theo dạng **diagonal wavefront** để đảm bảo tính đúng của accumulation:

```
Row 0: độ trễ 0 cycles  (direct wire)
Row 1: độ trễ 1 cycle   (1-deep shift register)
Row 2: độ trễ 2 cycles  (2-deep shift register)
...
Row 7: độ trễ 7 cycles  (7-deep shift register)
```

**Deskewing Logic** (sau khi ra khỏi array) thực hiện ngược lại để đồng bộ kết quả:
```
Row 0: độ trễ 7 cycles
...
Row 7: độ trễ 0 cycles
```

#### FSM State Machine

```
          rst_n=0
             │
             ▼
         ┌────────┐
    ┌───▶│ST_IDLE │◀─────────────────────┐
    │    └────┬───┘                       │
    │         │ START=1                   │
    │         ▼                           │
    │    ┌──────────────┐                 │
    │    │ST_PRELOAD_W  │ 8 cycles        │
    │    │(nạp weight   │                 │
    │    │ vào 64 PEs)  │                 │
    │    └──────┬───────┘                 │
    │           │                         │
    │           ▼                         │
    │    ┌──────────────┐                 │
    │    │ST_STREAM_ACT │ M×K cycles      │
    │    │(stream act   │                 │
    │    │ qua skewing) │                 │
    │    └──────┬───────┘                 │
    │           │                         │
    │           ▼                         │
    │    ┌──────────────┐                 │
    │    │ST_DRAIN_OUT  │ ⚠️ 17 cycles    │
    │    │7+7+3 = 17    │ (xem ghi chú)  │
    │    └──────┬───────┘                 │
    │           │                         │
    │    ┌──────▼───────┐                 │
    └────│  ST_DONE     │─────────────────┘
         │(set DONE bit,│
         │ raise irq)   │
         └──────────────┘
```

> **⚠️ ST_DRAIN_OUT Cycle Budget (đã sửa):**
> ```
> Total drain cycles = (MAC_DIM - 1)     ← Partial Sum thoát khỏi array bottom
>                    + (MAC_DIM - 1)     ← Deskewing FIFO căn chỉnh
>                    + ACTUNIT_PIPE_DEPTH ← Pipeline 3-stage Activation Unit
>                    = 7 + 7 + 3 = 17 cycles
> ```
> Nếu chỉ đếm 14 cycles (bỏ qua Activation Unit pipeline), bit `DONE` bật sớm 3 cycles → **3 kết quả cuối trong SRAM_Y sẽ là garbage data**.

### 5.4 Activation Unit (3-stage Pipeline)

```
INT32 input (từ Deskewing FIFOs)
      │
      ▼
[Stage 1: Scale Multiplier]
  INT32 × QUANT_SCALE[15:0] → 48-bit result → giữ bits [47:16]
      │
      ▼ (1 pipeline register)
[Stage 2: Barrel Shifter]
  >> QUANT_SHIFT (arithmetic right shift)
      │
      ▼ (1 pipeline register)
[Stage 3: ReLU + Clamp]
  if MSB=1 → 0 (ReLU)
  if value > 127 → 127 (clamp)
  → INT8 output
      │
      ▼
Pack 4× INT8 → 32-bit word → ghi vào SRAM_Y
```

**Bit-width clarification cho QUANT_SCALE:**
- Register `0x0014` là 32-bit nhưng **chỉ bits [15:0] có hiệu lực**
- Multiplier thực tế: INT32 × UINT16 = 48-bit intermediate
- Giữ lại bits [47:16] (32 MSBs) → đưa vào Stage 2
- **Golden Model Python phải implement đúng:**

```python
import numpy as np

# Stage 1: Scale (chú ý: chỉ dùng 16-bit thấp của QUANT_SCALE)
scale = np.int64(QUANT_SCALE & 0xFFFF)
scaled = (C_int32.astype(np.int64) * scale) >> 16  # giữ bits [47:16]

# Stage 2: Shift
shifted = np.right_shift(scaled.astype(np.int32), QUANT_SHIFT)

# Stage 3: ReLU + Clamp
relu = np.maximum(0, shifted)
output = np.clip(relu, 0, 127).astype(np.int8)
```

---

## 6. Dataflow Execution

Ví dụ minh hoạ cho phép nhân `C[8×8] = A[8×8] × B[8×8]`:

| Phase | Cycles | Mô tả |
|---|---|---|
| **Configuration & Setup** | T₀ → T | CPU ghi SRAM_W, SRAM_X, DIM_*, QUANT_* qua AXI4-Lite |
| **Trigger** | T+1 | CPU ghi `CTRL=0x1`, FSM bừng tỉnh |
| **Weight Pre-load** | T+2 → T+9 | 8 cycles: nạp ma trận B[8×8] vào 64 PEs, chốt Weight-Stationary |
| **Diagonal Wavefront Compute** | T+10 → T+X | Stream activation qua Skewing FIFOs, sóng chéo quét qua array |
| **Drain + Deskew + Quantize** | X+1 → X+17 | **17 cycles**: xả pipeline, đồng bộ, ReLU, scale, clamp, ghi SRAM_Y |
| **Done** | X+18 | `DONE=1`, `irq` bật, CPU bắt đầu đọc SRAM_Y từ offset `0x3000` |

---

## 7. Verification Plan

### 7.1 Testbench Architecture (Cocotb + Python)

```
┌─────────────────────────────────────────────┐
│              Cocotb Test Environment         │
│                                             │
│  ┌──────────────┐    ┌────────────────────┐ │
│  │  Clock/Reset │    │  NumPy Golden      │ │
│  │  Generator   │    │  Reference Model   │ │
│  └──────────────┘    └────────────────────┘ │
│                                             │
│  ┌──────────────────────────────────────┐   │
│  │  AXI4-Lite Master VIP               │   │
│  │  (cocotbext-axi: AxiLiteMaster)     │   │
│  │                                     │   │
│  │  await axi.write(0x2000, data)      │   │
│  │  await axi.read(0x0004)  # STATUS   │   │
│  └─────────────────┬────────────────────┘   │
│                    │ VPI/VHPI               │
│                    ▼                        │
│            ┌───────────────┐               │
│            │  DUT (RTL)    │               │
│            │  mini_npu_top │               │
│            └───────┬───────┘               │
│                    │                        │
│  ┌─────────────────▼────────────────────┐   │
│  │  Scoreboard / Checker                │   │
│  │  np.testing.assert_array_equal(      │   │
│  │      dut_output, golden_output)      │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

**Dependencies:**
```bash
pip install cocotb cocotbext-axi numpy
# Simulator: Icarus Verilog (iverilog) hoặc Verilator
```

### 7.2 Test Cases

| Test | Mô tả | Điều kiện Pass |
|---|---|---|
| **Test 1: Basic Sanity** | Ma trận 2×2 hoặc 4×4, giá trị ±1 và 0. Kiểm tra AXI4-Lite handshake, FSM toggle, DONE signal. | `np.array_equal(dut_out, golden_out)` |
| **Test 2: Full Array Stress** | Ma trận 8×8 random INT8. Kiểm tra tất cả 64 PEs và Skewing/Deskewing correctness. | Bit-exact match với Golden Model |
| **Test 3: Saturation & Overflow** | Ma trận 8×8 toàn giá trị `±127/±128`. Xác minh INT32 không wrap-around và Clamp hoạt động. | Output clamp đúng tại 127 |
| **Test 4: ReLU Cut-off** | Dữ liệu thiết kế để sinh Partial Sum âm. Xác minh ReLU ép đúng về 0. | Không có giá trị âm trong SRAM_Y |
| **Test 5: FSM Illegal State** | Ghi `START=1` khi NPU đang `BUSY`. FSM phải bỏ qua. | Kết quả sau khi done khớp Golden Model; SVA assertion không fire |

### 7.3 Coverage & Assertions Plan

**Ngôn ngữ RTL: SystemVerilog** (không dùng Verilog-2001)
- Dùng `always_ff`, `always_comb`, `logic` type, `interface`
- Đây là chuẩn công nghiệp, điểm cộng mạnh trên CV

**Bước 1: Linting trước khi sim**
```bash
verilator --lint-only -Wall rtl/*.sv
```

**Bước 2: Coverage với Verilator**
```bash
verilator --coverage --cc rtl/*.sv --exe tb/sim_main.cpp
```
Mục tiêu: Line ≥ 95%, Branch ≥ 90%

**Bước 3: SVA Assertions (nhúng trong RTL)**
```systemverilog
`ifdef SIMULATION
// 1. Không trigger START khi đang BUSY
assert property (@(posedge clk) disable iff (!rst_n)
    busy |-> !$rose(start))
    else $error("START triggered while BUSY");

// 2. Sau DONE phải về IDLE
assert property (@(posedge clk) disable iff (!rst_n)
    done |=> idle)
    else $error("FSM did not return to IDLE after DONE");

// 3. Weight không thay đổi trong pha computing
assert property (@(posedge clk) disable iff (!rst_n)
    $stable(weight_reg) throughout compute_phase)
    else $error("Weight register changed during compute phase");

// 4. Accumulator không wrap-around
assert property (@(posedge clk) disable iff (!rst_n)
    (acc_out[31:8] == 24'b0) || acc_out[31])
    else $error("INT32 accumulator overflow detected");
`endif
```

**Chạy toàn bộ regression:**
```bash
make test        # chạy tất cả 5 test cases
make coverage    # generate HTML coverage report
```

---

## 8. Resource Estimate & Target Platform

### FPGA Resource Estimate

> Ước tính dựa trên phân tích kiến trúc, trước synthesis thực tế.

| Resource | Estimate | Justification |
|---|---|---|
| DSP48E1 | ~32–64 | 1 DSP/PE (INT8×INT8 fit trong 1 DSP18); có thể pack 2 phép nhân INT8 vào 1 DSP48 → 32 DSPs |
| BRAM36 | 3 | SRAM_W + SRAM_X + SRAM_Y (3 × 4KB) |
| LUT | ~2,000–4,000 | FSM + Skewing/Deskewing + AXI Slave + Activation Unit + Accumulation Buffer |
| Flip-Flops | ~1,500–3,000 | Pipeline registers trong PE và Activation Unit |
| **Peak Throughput** | **6.4 GOPS @ 100 MHz** | 64 MACs/cycle × 2 Ops/MAC × 100M cycles/s (effective với overhead) |
| **Target Frequency** | 100–200 MHz | Critical path: adder chain trong PE column |

### Recommended Target Platform

**Xilinx Zynq-7020 (PYNQ-Z2)**
- ARM Cortex-A9 dual-core đóng vai trò CPU Host
- FPGA fabric: 85K LUT, 220 DSP48, 140 BRAM36 — dư thoải mái
- PYNQ framework: viết software driver bằng Python, demo inference trực tiếp
- Toolchain: Vivado — chuẩn công nghiệp, phổ biến với nhà tuyển dụng

### Project Timeline

| Giai đoạn | Tuần | Nội dung |
|---|---|---|
| RTL Implementation | 1–6 | PE (1w) → Systolic Array (1w) → FSM + Skewing (2w) → AXI Slave + ActUnit + Integration (2w) |
| Verification | 7–10 | Cocotb setup + 5 test cases + Coverage report + SVA (4w) |
| FPGA (optional) | 11–13 | Vivado synthesis + timing closure + PYNQ demo (3w) |
| **Total** | **10–13 tuần** | Part-time ~15–20 giờ/tuần |

> **Điểm tốn thời gian nhất thực tế:** Debug Skewing/Deskewing FIFO alignment (~2–3 ngày waveform tracing) và AXI4-Lite handshake timing.

---

## References

| # | Paper / Resource |
|---|---|
| [1] | [Adaptive Tiling for Fixed-size Systolic Arrays — IEEE Xplore](https://ieeexplore.ieee.org/document/8545462/) |
| [2] | [Understanding Matrix Multiplication on Weight-Stationary Systolic Architecture — Telesens](https://telesens.co/2018/07/30/systolic-architectures/) |
| [3] | [Hardware-level AI Matrix Multiplication Accelerator in SystemVerilog — GitHub](https://github.com/mikolajnawr/AI-matrix-multiplication-accelerator) |
| [4] | [alexforencich/cocotbext-axi — GitHub](https://github.com/alexforencich/cocotbext-axi) |
| [5] | [Cocotb Documentation](https://docs.cocotb.org/en/latest/) |
| [6] | [Gemmini: Enabling Systematic Deep-Learning Architecture Evaluation — DAC 2021](https://people.eecs.berkeley.edu/~ysshao/assets/papers/genc2021-dac.pdf) |
| [7] | [Systolic Array — Wikipedia](https://en.wikipedia.org/wiki/Systolic_array) |

---

*Micro-Architecture Specification v2.0 — Mini NPU Accelerator*  
*Last updated: May 2026*
