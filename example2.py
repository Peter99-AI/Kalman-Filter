"""
    Dưới đây là một bài mô phỏng đơn giản trên phân bố chuẩn 1 chiều dựa trên ví dụ của Greg Welch và 
    Gary Bishop, giả sử có một nguồn điện với hiệu điện thế không thay đổi theo thời gian (cho lý tưởng 
    là vậy nhé, nghĩa là x= 0.28 volts mãi luôn). Bạn thực sự không biết giá trị hiệu điện thế của 
    nguồn điện này đâu, bạn quyết định đi tìm giá trị này như một thú vui lúc rảnh rỗi. May mắn thay, 
    bạn tìm được một cái vôn kế đời cũ không chính xác. Rất may là nhà sản xuất có để lại thông tin, 
    vôn kế này bị nhiễu theo phân bố  N(0,0.01). Biết rằng mỗi lần đo kế tiếp, hiệu điện thế không đổi
    gì nhiều (thiết lập ma trận chuyển trạng thái A), mô hình quan sát của được đo từ vôn kế bằng 
    giá trị thực tế cộng thêm nhiễu (giả định ma trận G). Không rõ độ lệch chuẩn của nguồn điện phóng 
    ra khi sử dụng bình thường là bao nhiêu, bạn tạm đoán là 1 volt (thiết lập ma trận Σ), tại thời điểm 
    ban đầu chưa biết bạn đoán nguồn điện 0.5 volt (thiết lập ^x).
"""



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('bmh')

def simulate_volts(x, G, R):
    """
    Hàm lấy giá trị quan sát y từ mô hình quan sát
    y = Gx + v
    (với v là vector ngẫu nhiên v ~ N(0,R))

    :param x: numpy.array - vector thực tế x
    :param G: ma trận mô hình quan sát
    :param R: ma trận hiệp phương sai nhiễu quan sát

    :return y: numpy.array - vector ngẫu nhiên quan sát được
    """
    return G @ x + np.random.multivariate_normal(np.zeros(shape=R.shape[0]), R)

# Số lượng vòng lặp
n_iteration = 100

# Thật sự là -0.28 vols
x = np.array([[-0.28]])

# Ma trận chuyển đổi trạng thái
A = np.array([[1]])
# Ma trận hiệp phương sai nhiễu hệ thống
Q = np.array([[1e-6]]) 
# Ma trận mô hình quan sát
G = np.array([[1]])
# Ma trận hiệp phương sai nhiễu quan sát
R = np.array([[0.01]])

list_of_x = []
list_of_y = []

# Thiết lập phân bố ban đầu
x_hat = np.array([[0.5]])
sigma = np.array([[1.0]])

for i in range(n_iteration):
    # Bước dự đoán (Prediction)
    x_hat = A @ x_hat
    sigma = A @ sigma @ A.T + Q
    print("Sigma: %r " % sigma)
    list_of_x.append(float(x_hat))
    
    # Bước đo lường (Measurement)
    y = simulate_volts(x, G, R)
    kalman_gain = sigma @ G.T @ np.linalg.inv(G @ sigma @ G.T + R)
    x_hat = x_hat + kalman_gain @ (y - G @ x_hat)
    sigma = (np.eye(len(sigma)) - kalman_gain @ G) @ sigma
    list_of_y.append(float(y))

# Vẽ nào
plt.plot(list_of_y,'k+',label='dữ liệu quan sát')
plt.axhline(x, color='b',label='giá trị thực tế')
plt.plot(list_of_x,'r-',label='giá trị dự đoán')
plt.legend()

plt.xlabel('Thời gian $t$')
plt.ylabel('Hiệu điện thế (volts)')
plt.title('Kalman Filter')

plt.show()
