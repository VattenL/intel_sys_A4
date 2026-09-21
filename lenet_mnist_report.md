# LeNet-5 trên MNIST

**Assignment 4 · Báo cáo kiến trúc kinh điển · Scratch NumPy · TensorFlow/Keras · PyTorch**

> Báo cáo cho notebook `06_mnist_lenet.ipynb`. Notebook `02_mnist.ipynb` với kiến trúc
> `CNN 2conv+fc` **giữ nguyên, không sửa** — đây là lần chạy song song để so sánh.
>
> Mọi con số lấy từ `results/06_mnist_lenet.json` và `results/02_mnist.json`.

---

# 1. LeNet-5 là gì

## 1.1 Bối cảnh

LeNet-5 được Yann LeCun và cộng sự công bố năm 1998 trong bài báo
*Gradient-Based Learning Applied to Document Recognition*. Đây là mạng tích chập
đầu tiên chạy được trên dữ liệu thật ở quy mô sản xuất — nó từng được dùng để đọc
số viết tay trên séc ngân hàng tại Mỹ.

Điểm quan trọng với bài này: **LeNet-5 sinh ra đúng cho bài toán nhận dạng chữ số
viết tay**. MNIST là sân nhà của nó, không phải một phép thử khiên cưỡng.

## 1.2 Kiến trúc

```text
   input 1×28×28
        ↓
   C1   Conv 6 @ 5×5, pad=2    →  6×28×28     feature maps
        ↓ ReLU
   S2   MaxPool 2×2            →  6×14×14     subsample
        ↓
   C3   Conv 16 @ 5×5, pad=0   → 16×10×10     feature maps
        ↓ ReLU
   S4   MaxPool 2×2            → 16×5×5       subsample
        ↓ Flatten              → 400
   F5   Dense 400 → 120        ↓ ReLU
   F6   Dense 120 → 84         ↓ ReLU
   out  Dense  84 → 10         (logits)
```

Tên lớp theo đúng ký hiệu bài báo gốc: `C` cho convolution, `S` cho subsample,
`F` cho fully-connected.

## 1.3 Vì sao `pad=2` ở C1

Bài báo gốc nhận ảnh **32×32** rồi dùng convolution *valid* 5×5 để ra 28×28.
MNIST chỉ có 28×28. Có hai cách xử lý:

| Cách | Thao tác | Kết quả |
|---|---|---|
| Pad ảnh | Đệm 28×28 → 32×32 rồi conv valid | 28×28 |
| **Pad convolution** (dùng ở đây) | Conv 5×5 với `pad=2` thẳng trên 28×28 | 28×28 |

Hai cách cho cùng kích thước feature map. Cách thứ hai bỏ được một bước tiền xử lý
và giữ pipeline dữ liệu **giống hệt** notebook baseline — điều kiện bắt buộc để so sánh
công bằng.

## 1.4 Phân bổ tham số — điểm đáng chú ý nhất

| Lớp | Tham số | Tỷ lệ |
|---|---:|---:|
| C1 Conv 1→6, 5×5 | 156 | 0.3% |
| C3 Conv 6→16, 5×5 | 2,416 | 3.9% |
| F5 Linear 400→120 | 48,120 | 78.0% |
| F6 Linear 120→84 | 10,164 | 16.5% |
| out Linear 84→10 | 850 | 1.4% |
| **Tổng** | **61,706** | 100% |

**Bộ trích xuất đặc trưng chỉ chiếm 4.2% tham số** (2,572 trên 61,706). Gần 80% nằm
ở một lớp Dense duy nhất. Đây là đặc điểm chung của CNN đời đầu: phần tích chập làm
việc nặng về mặt tính toán nhưng rất nhẹ về tham số, còn phần phân loại thì ngược lại.

---

# 2. "Modernized" nghĩa là gì

Bản LeNet-5 dùng trong bài **không phải bản 1998 nguyên văn**. Khác biệt:

| Thành phần | LeNet-5 gốc (1998) | Bản dùng ở đây |
|---|---|---|
| Hàm kích hoạt | `tanh` | **ReLU** |
| Lớp subsample | Average pooling | **Max pooling** |
| Topology | 6→16 maps, 5×5, head 120→84→10 | **giữ nguyên** |
| Optimizer | SGD | Adam (khớp baseline) |

Đây là bản mà hầu hết implementation hiện đại dùng, và có lý do kỹ thuật cụ thể trong
repo này: **`scratch_nn.py` chỉ hiện thực `ReLU` và `MaxPool2D`** — không có `Tanh`,
không có `AvgPool2D`. Chọn bản modernized cho phép cả ba framework chạy **cùng một mạng**
mà không phải viết thêm đạo hàm ngược cho hai loại lớp mới.

Nói rõ điều này là cần thiết: gọi kiến trúc là "LeNet-5" mà im lặng về việc đã đổi
activation và pooling sẽ là mô tả thiếu trung thực.

---

# 3. Thiết lập thí nghiệm

Điều kiện so sánh tuân theo quy tắc công bằng đã dùng xuyên suốt bài:

> Cùng dataset + cùng split + cùng hyperparameters — chỉ đổi kiến trúc.

| Thiết lập | Giá trị |
|---|---|
| Subset so sánh 3 framework | 10,000 train / 2,000 test |
| Tham chiếu full-data | 60,000 train / 10,000 test |
| Epochs | 5 |
| Batch size | 64 |
| Optimizer | Adam, lr = 1e-3 |
| Seed | 42 (`U.set_seed` gọi lại trước mỗi framework) |
| Chuẩn hoá | Per-channel, thống kê tính từ tập train |

Subset 10k tồn tại vì bản NumPy thuần quá chậm cho 60k ảnh. PyTorch và Keras không
vướng giới hạn đó nên được chạy thêm trên toàn bộ dữ liệu.

## 3.1 Lưu ý bắt buộc khi đọc cột thời gian

**PyTorch chạy GPU, TensorFlow chạy CPU.** TensorFlow từ bản 2.11 trở đi không còn
build GPU cho Windows native; máy chạy bài này có RTX 3060 Laptop nhưng chỉ PyTorch
dùng được.

Do đó cột `train_seconds` **so sánh hai loại phần cứng khác nhau**, không phản ánh
chất lượng framework. Cột `n_params` và `test_accuracy` thì so sánh được trực tiếp.

## 3.2 Ba cách hiện thực cùng một mạng

| Thành phần | Scratch (NumPy) | TensorFlow/Keras | PyTorch |
|---|---|---|---|
| Định nghĩa mạng | `S.Sequential([...])` | `keras.Sequential([...])` | `nn.Module` (`LeNet5`) |
| C1 / C3 | `S.Conv2D(k=5)` im2col + matmul | `layers.Conv2D(5)` | `nn.Conv2d(5)` |
| S2 / S4 | `S.MaxPool2D` định tuyến argmax | `layers.MaxPooling2D` | `nn.MaxPool2d` |
| F5 / F6 / out | `S.Dense` (`x@W+b`) | `layers.Dense` | `nn.Linear` |
| Gradient | **đạo hàm viết tay + col2im** | tự động (GradientTape) | tự động (autograd) |
| Vòng huấn luyện | for-loop tường minh | `model.fit()` | for-loop tường minh |
| Phần cứng | CPU | CPU | GPU (cuDNN) |

Bản scratch là bản duy nhất không có autograd — mọi đạo hàm đều được suy ra bằng tay.
Notebook chạy kiểm tra sai phân hữu hạn (`S.gradient_check`) trên kernel 5×5 trước khi
huấn luyện, vì các notebook trước chỉ mới kiểm chứng kernel 3×3.

---

# 4. Kết quả

## 4.1 Ba framework trên subset 10k

| Framework | Tham số | Accuracy | Macro-F1 | Train (s) | Phần cứng |
|---|---:|---:|---:|---:|---|
| Scratch (NumPy) | 61,706 | 97.75% | 0.9772 | 32.9 | CPU |
| TensorFlow/Keras | 61,706 | 97.60% | 0.9762 | 6.6 | CPU |
| PyTorch | 61,706 | 97.55% | 0.9754 | 4.2 | GPU |

Ba framework ra **đúng 61,706 tham số** — con số khớp với tính tay ở mục 1.4.
Đây là kiểm chứng mạnh nhất cho việc kiến trúc được dịch đúng sang cả ba nơi:
ba hiện thực độc lập, cùng một con số.

Chênh lệch accuracy giữa ba bản là **0.20 điểm** (97.55 → 97.75). Đó là nhiễu do
thứ tự khởi tạo và shuffle khác nhau, không phải bằng chứng bản nào tốt hơn.

## 4.2 Full-data (60k)

| Framework | Accuracy | Macro-F1 | Train (s) |
|---|---:|---:|---:|
| PyTorch | 98.70% | 0.9868 | 23.0 |
| TensorFlow/Keras | 98.59% | 0.9858 | 29.9 |

Gấp 6 lần dữ liệu đổi lấy khoảng **+1 điểm** accuracy (97.6 → 98.7).

---

# 5. LeNet-5 so với kiến trúc baseline

`02_mnist.ipynb` dùng `CNN 2conv+fc`: hai lớp conv 3×3 với 16 rồi 32 kênh, nối thẳng
vào một `Linear(1568 → 10)`. Tổng 20,490 tham số.

## 5.1 Bảng đối đầu

| Dataset | Framework | `CNN 2conv+fc` | LeNet-5 | Chênh |
|---|---|---:|---:|---:|
| MNIST (10k subset) | Scratch (NumPy) | 98.10% | 97.75% | −0.35 |
| MNIST (10k subset) | TensorFlow/Keras | 97.55% | 97.60% | **+0.05** |
| MNIST (10k subset) | PyTorch | 98.35% | 97.55% | −0.80 |
| MNIST (full 60k) | TensorFlow/Keras | 98.55% | 98.59% | **+0.04** |
| MNIST (full 60k) | PyTorch | 98.74% | 98.70% | −0.04 |

**Chênh lệch trung bình: −0.22 điểm.**

## 5.2 Đọc bảng này thế nào

**1. LeNet-5 không thắng, dù có nhiều tham số gấp 3.**
61,706 so với 20,490 — gấp 3.01 lần — mà accuracy vẫn hoà hoặc kém đi chút ít.
Đây không phải thất bại của LeNet. Đây là bằng chứng rằng **trên một dataset đủ dễ,
kiến trúc gần như không còn là biến quyết định.**

**2. Trên full-data, hai kiến trúc không phân biệt được.**
Chênh 0.04 điểm ở cả hai framework — tức khoảng 4 ảnh trên 10,000. Con số đó nhỏ hơn
dao động giữa các lần chạy. Kết luận trung thực là **hoà**, không phải "baseline thắng".

**3. Chênh lệch lớn nhất (−0.80) nằm ở subset, không phải full-data.**
Dữ liệu càng ít thì khác biệt kiến trúc càng lộ, và càng nhiều nhiễu. Đây là lý do
không nên kết luận từ một dòng đơn lẻ trong bảng.

## 5.3 Phát hiện đáng chú ý nhất — tham số không phải chi phí tính toán

| | Tham số | Scratch train (s) |
|---|---:|---:|
| `CNN 2conv+fc` | 20,490 | 75.1 |
| LeNet-5 | 61,706 | **32.9** |

**LeNet-5 có tham số gấp 3 nhưng huấn luyện nhanh hơn 2.28 lần.**

Nghe mâu thuẫn, nhưng tính ra thì hợp lý. Đếm số phép nhân-cộng (MAC) của hai lớp conv:

```text
CNN 2conv+fc:
  C1  28×28×16×1×3×3  =   112,896
  C2  14×14×32×16×3×3 =   903,168
  tổng                ≈ 1,016,064 MAC

LeNet-5:
  C1  28×28×6×1×5×5   =   117,600
  C3  10×10×16×6×5×5  =   240,000
  tổng                ≈   357,600 MAC

→ LeNet-5 rẻ hơn 2.84 lần về tính toán conv
```

Tỷ lệ 2.84× này khớp rất sát với tốc độ đo được 2.28×.

Nguyên nhân: **tham số của LeNet nằm ở lớp Dense, còn chi phí tính toán nằm ở lớp Conv.**
Một trọng số Dense được dùng đúng một lần cho mỗi mẫu. Một trọng số Conv được dùng lại
ở *mọi vị trí không gian* — với feature map 28×28 là 784 lần. Baseline có ít tham số hơn
nhưng dùng tới 16 và 32 kênh, nên khối lượng tính toán conv lớn hơn hẳn.

> **Rút ra:** đếm tham số là thước đo dung lượng bộ nhớ, **không phải** thước đo tốc độ.
> Hai chỉ số này có thể đi ngược chiều nhau, và ở đây chúng đi ngược nhau rõ rệt.

---

# 6. Kết luận

**LeNet-5 hoạt động tốt trên MNIST — đúng như kỳ vọng, vì đây là bài toán nó được
thiết kế cho.** 97.55–97.75% trên subset 10k, 98.70% trên full-data.

Nhưng nó **không tốt hơn** kiến trúc baseline tự đặt, dù nặng gấp 3 lần về tham số.
Chênh lệch trung bình −0.22 điểm, và trên full-data thì hai bên không phân biệt được.

Ba điều đáng mang đi:

1. **Trên dataset dễ, kiến trúc không phải biến quyết định.** MNIST đã bão hoà ở mức
   ~98–99% với gần như mọi CNN hợp lý. Muốn thấy kiến trúc thực sự tạo khác biệt thì
   phải đưa lên dữ liệu khó hơn — đó là nội dung của `09_lenet_cifar10_report.pdf`.

2. **Tham số ≠ tốc độ.** LeNet nhiều tham số gấp 3 mà chạy nhanh hơn 2.3 lần, vì chi phí
   thật nằm ở số kênh convolution chứ không ở kích thước lớp Dense.

3. **Ba framework ra cùng 61,706 tham số và accuracy lệch nhau 0.2 điểm.** Đó là xác nhận
   kiến trúc được hiện thực đúng ở cả ba nơi, kể cả bản NumPy với đạo hàm viết tay.

## Hạn chế

Kết quả này dựa trên **một seed, 5 epochs, không tinh chỉnh hyperparameter**. Các chênh
lệch dưới ~0.5 điểm nằm trong vùng nhiễu và không nên diễn giải quá mức. Muốn kết luận
chắc chắn hơn thì cần chạy nhiều seed và báo cáo trung bình ± độ lệch chuẩn.
