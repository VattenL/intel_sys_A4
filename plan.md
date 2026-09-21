# Kế hoạch viết báo cáo Assignment 04

## 1. Bối cảnh

Đề bài (ảnh `img/req.jpg`) yêu cầu năm việc:

1. Ba dataset: một bộ dữ liệu bảng cỡ lớn, hai bộ ảnh
2. CNN viết từ đầu (scratch), có mô hình lưu lại
3. CNN bằng TensorFlow/Keras, có mô hình lưu lại
4. CNN bằng PyTorch, có mô hình lưu lại
5. Trực quan hoá và so sánh ba bản cài đặt

Phần thuyết trình còn hỏi thêm: hiểu deep learning, hiểu CNN, cải tiến kiến trúc CNN, và demo.

Repo đã chạy xong toàn bộ. Báo cáo là bước cuối: gom lại thành một tài liệu.

## 2. Quyết định đã chốt

| Mục | Chốt |
|---|---|
| Phạm vi dữ liệu | Đủ ba bộ: Diabetes 130-US, MNIST, CIFAR-10 |
| Phụ lục | Nhúng nguyên sáu notebook, giữ cả markdown, code và output đã chạy |
| Số bản | Chỉ một bản đầy đủ, không làm bản rút gọn |
| Tác giả | Nhóm 1, ghi theo vai trò, không ghi tên và mã sinh viên |
| Ngôn ngữ | Tiếng Việt |
| Người đọc | Sinh viên chuyên ngành CNPM, không chuyên AI |

## 3. Nguyên tắc viết

**Số liệu không gõ tay.** Mọi con số trong báo cáo đọc từ `results/*.json` và `results/all_runs.csv`
do notebook ghi ra. Báo cáo và notebook vì vậy không thể lệch nhau.

**Thuật ngữ AI phải được giải thích tại chỗ.** Mỗi thuật ngữ xuất hiện lần đầu mang một số chú thích,
nội dung giải thích in cỡ nhỏ ngay chân trang đó, một hoặc hai câu, không công thức. Toàn bộ gom lại
ở phụ lục thuật ngữ để tra cứu.

**Không hứa suông.** Kết quả nào không đẹp thì viết đúng như nó có, kèm lý do. Phần hạn chế là một
chương thật sự, không phải một câu cho có.

**Thời gian chạy không so trực tiếp được.** Keras chạy CPU, PyTorch chạy GPU. Điều này phải nhắc ở
chương môi trường và nhắc lại ở mọi bảng có cột thời gian.

## 4. Cấu trúc báo cáo

### Phần đầu
Bìa, tóm tắt, mục lục, và một trang quy ước đọc báo cáo giải thích cách chú thích thuật ngữ.

### Chương 1. Đề bài và phạm vi
Chép lại yêu cầu, liệt kê những gì nhóm đã nộp, nêu đối tượng người đọc, phát biểu quy tắc công bằng
khi so sánh (cùng dữ liệu, cùng cách chia, cùng kiến trúc, siêu tham số tương đương), mô tả môi trường
chạy thật kèm cảnh báo về thời gian.

### Chương 2. Nền tảng tối thiểu
Chương dành riêng cho người mới. Bài toán phân loại, vòng lặp học bốn bước, vì sao ảnh cần CNN, các
lớp mạng dùng trong bài, công thức tính kích thước đầu ra và đếm tham số, các chỉ số đánh giá, và các
khái niệm epoch, batch, learning rate, seed.

### Chương 3. Ba bộ dữ liệu
Mô tả từng bộ, cách tiền xử lý, cách chia tập và chuẩn hoá qua một loader duy nhất, và lý giải vì sao
dữ liệu bảng không dùng convolution.

### Chương 4. Ba môi trường cài đặt
Từng môi trường một: scratch bằng NumPy với đạo hàm tự viết và kiểm tra gradient bằng sai phân hữu hạn,
TensorFlow/Keras với `fit()`, PyTorch với vòng lặp tường minh và autograd. Kết bằng bảng đối chiếu API
ba bên và phân tích mỗi mức trừu tượng giấu đi cái gì.

### Chương 5, 6, 7. Kết quả theo từng bộ dữ liệu
Diabetes (MLP), MNIST (CNN), CIFAR-10 (CNN). Mỗi chương theo cùng một khuôn: cấu hình, số tham số đối
chiếu tay với máy, bảng kết quả ba môi trường, đường loss và độ chính xác theo epoch, ma trận nhầm lẫn,
và phần đọc kết quả bằng lời.

### Chương 8. LeNet-5 trên hai bộ ảnh
Kiến trúc gốc 1998 và phần hiện đại hoá, kết quả trên từng bộ, và phân tích vì sao cùng một kiến trúc
lại cho hai kết quả trái ngược trên hai bộ ảnh cùng kích thước.

### Chương 9. So sánh ba bản cài đặt
Đây là mục 5 của đề bài. Số tham số trùng khớp tuyệt đối, độ chính xác nằm trong dải nhiễu (có đo sàn
nhiễu bằng nhiều seed để chứng minh), thời gian chạy và lý do GPU không phải lúc nào cũng nhanh hơn,
bảng thành phần theo slide bài giảng, và khuyến nghị chọn môi trường nào cho việc gì.

### Chương 10. Cải tiến kiến trúc CNN
Nguyên tắc kiến trúc mới bằng kiến trúc cũ cộng một cơ chế xử lý một hạn chế. Bốn mô hình M1 đến M4,
kết quả thật, phân tích vì sao attention lại làm tệ đi, và bài học rút ra.

### Chương 11. Mô hình đã lưu và kịch bản demo
Ba định dạng lưu trữ và vì sao chúng khác nhau, cách nạp lại, và kịch bản demo khi bảo vệ.

### Chương 12. Hạn chế và kết luận
Hạn chế trung thực, kết luận trả lời ba câu hỏi của phần thuyết trình, và bảng phân công bốn thành viên
theo vai trò.

### Phụ lục
Sáu notebook nhúng nguyên vẹn, mã nguồn dùng chung, và bảng thuật ngữ Việt Anh.

## 5. Cách dựng tài liệu

Dùng lại pipeline của `assignment03/reports/build`: nội dung sinh bằng Python ra HTML, tạo dáng bằng
`report.css` theo phong cách LaTeX, phân trang bằng Paged.js, in PDF bằng Chrome headless. Pipeline này
tự đánh số bảng và hình, tự sinh mục lục có số trang, tự nhúng ảnh thành data URI, và tự chuyển notebook
thành HTML.

Thư mục dự kiến đặt tại `reports/` trong repo này, gồm: kịch bản dựng, các tệp sinh nội dung theo chương,
tệp CSS, và tệp font.

## 6. Thứ tự thực hiện

1. Dựng khung pipeline và xác nhận in được một PDF rỗng đúng phong cách
2. Viết phần đầu và các chương 1 đến 4 (phần không phụ thuộc số liệu)
3. Viết các chương 5 đến 8 (sinh bảng và hình từ tệp kết quả)
4. Viết các chương 9 đến 12
5. Ghép phụ lục notebook và mã nguồn
6. Rà soát chú thích thuật ngữ, đối chiếu bảng thuật ngữ, in bản cuối

## 7. Rủi ro

**PDF sẽ rất nặng.** Sáu notebook có nhiều hình. Bản assignment03 chỉ có một notebook chính đã ra hơn
5 MB. Nếu quá nặng sẽ cần giảm độ phân giải ảnh nhúng.

**Chú thích thuật ngữ dễ bị lặp hoặc bị sót.** Cần một danh sách thuật ngữ tập trung, mỗi thuật ngữ chỉ
chú thích ở lần xuất hiện đầu, và bảng thuật ngữ ở phụ lục sinh từ chính danh sách đó.

**Font tiếng Việt.** Pipeline mẫu nhúng sẵn font trong `fonts.css`, cần sao chép sang để dấu tiếng Việt
không vỡ.
