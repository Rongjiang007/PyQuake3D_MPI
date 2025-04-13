
import numpy as np
import cupy as cp
import time
from scipy.sparse.linalg import svds
'''
# 矩阵大小
matrix_size = 1000

# 使用 NumPy 在 CPU 上进行矩阵相乘
np_matrix_a = np.random.rand(matrix_size, matrix_size)
np_matrix_b = np.random.rand(matrix_size, matrix_size)

start_time = time.time()
#np_result = np.dot(np_matrix_a, np_matrix_b)
np_result=np_matrix_a+np_matrix_b
np_time = time.time() - start_time

print(f"NumPy (CPU) matrix multiplication time: {np_time:.5f} seconds")

# 使用 CuPy 在 GPU 上进行矩阵相乘
#cp_matrix_a = cp.random.rand(matrix_size, matrix_size)
#cp_matrix_b = cp.random.rand(matrix_size, matrix_size)

start_time = time.time()
#cp_result = cp.dot(np_matrix_a, np_matrix_b)
cp_result=np_matrix_a+np_matrix_b
# 确保计算完成
cp.cuda.Stream.null.synchronize()
cp_time = time.time() - start_time

print(f"CuPy (GPU) matrix multiplication time: {cp_time:.5f} seconds")

# 验证结果是否一致（将结果从 GPU 拷贝到 CPU）
#np_cp_result = cp.asnumpy(cp_result)
#assert np.allclose(np_result, np_cp_result), "Results do not match!"

print("The results from NumPy and CuPy are consistent.")
'''
# import numpy as np
# import cupy as cp
# import time

# # Matrix size
# matrix_size = 10000

# # NumPy (CPU) matrix multiplication
# np_matrix_a = np.random.rand(matrix_size, matrix_size)
# np_matrix_b = np.random.rand(matrix_size, matrix_size)

# # start_time = time.time()
# # np_result = np.dot(np_matrix_a, np_matrix_b)
# # np_time = time.time() - start_time
# # print(f"NumPy (CPU) matrix multiplication time: {np_time:.5f} seconds")

# # CuPy (GPU) matrix multiplication
# cp_matrix_a = cp.asarray(np_matrix_a)  # Transfer data to GPU
# cp_matrix_b = cp.asarray(np_matrix_b)

# start_time = time.time()
# cp_result = cp.dot(cp_matrix_a, cp_matrix_b)
# cp.cuda.Stream.null.synchronize()  # Ensure completion
# cp_time = time.time() - start_time
# print(f"CuPy (GPU) matrix multiplication time: {cp_time:.5f} seconds")

# # Verify results
# np_cp_result = cp.asnumpy(cp_result)  # Transfer back to CPU
# #assert np.allclose(np_result, np_cp_result), "Results do not match!"

# print("The results from NumPy and CuPy are consistent.")



# import numpy as np
# import cupy as cp
# from concurrent.futures import ThreadPoolExecutor
# import time

# cuda_available = cp.cuda.runtime.getDeviceCount()
# print(f"Number of CUDA devices detected: {cuda_available}")

# # 定义任务
# def cpu_task(a, b):
#     start_time = time.time()  # 开始计时
#     #result = a + b
#     result = np.dot(a,b)
#     end_time = time.time()  # 结束计时
#     print(f"CPU task time: {end_time - start_time:.5f} seconds")
#     return result

# def gpu_task(a, b):
#     a_gpu = cp.array(a)
#     b_gpu = cp.array(b)
#     start_time = time.time()  # 开始计时
#     c_gpu = cp.dot(a_gpu,b_gpu)
#     end_time = time.time()  # 结束计时
#     result = cp.asnumpy(c_gpu)  # 从 GPU 拷贝结果到 CPU
#     print(f"GPU task time: {end_time - start_time:.5f} seconds")
#     return result

# # 初始化数据
# n = 100000
# a = np.random.rand(n,n).astype(np.float32)
# b = np.random.rand(n).astype(np.float32)

# # 并行执行任务
# with ThreadPoolExecutor() as executor:
#     cpu_future = executor.submit(cpu_task, a, b)  # 提交 CPU 任务
#     gpu_future = executor.submit(gpu_task, a, b)  # 提交 GPU 任务

#     cpu_result = cpu_future.result()  # 获取 CPU 结果
#     gpu_result = gpu_future.result()  # 获取 GPU 结果

# # 验证结果
# assert np.allclose(cpu_result, gpu_result)
# print("Successfully ran CPU and GPU tasks in parallel!")



# import numpy as np

# a = 800
# b = 800

# # 直接计算 e^a + e^b 可能会溢出
# result_direct = np.exp(a) + np.exp(b)
# from math import *
# # 使用 logaddexp 进行稳定计算
# result_stable = np.exp(np.logaddexp(a, b))
thread_svd=30
# print(f"Direct result: {result_direct}")
# print(f"Stable result using np.logaddexp: {result_stable}")
from joblib import dump, load
import joblib
import numpy as np

# data1=np.load('A1s.npy')

# block = joblib.load("bp5t_core/blocks_to_process.joblib")

# def blocks_process_MVM(xvector,blocks_to_process):
#     yvector=np.zeros(len(xvector))
#     #start_time = time.time()  # 记录起始时间
#     #for k in range(100):
#     for i in range(len(blocks_to_process)):
#         x_=xvector[blocks_to_process[i].row_cluster]
#         if(len(blocks_to_process[i].row_cluster)>=thread_svd):
#             Ax_rsvd = blocks_to_process[i].U_1s @ (blocks_to_process[i].S_1s * (blocks_to_process[i].Vt_1s @ x_))
#         else:
#             #print(blocks_to_process[i].Mf_A1s.shape,len(blocks_to_process[i].row_cluster),len(blocks_to_process[i].col_cluster),x_.shape)
#             Ax_rsvd=blocks_to_process[i].Mf_A1s @ x_
#         yvector[blocks_to_process[i].col_cluster]=yvector[blocks_to_process[i].col_cluster]+Ax_rsvd
#     #end_time = time.time()  # 记录结束时间
#     #elapsed_time = end_time - start_time  # 计算时间差
#     return yvector

# print(len(block))

# xvector=np.ones(data1.shape[0])
# yvector1=blocks_process_MVM(xvector,block)
# print(yvector1)

# yvector2=np.dot(data1,xvector)
# print(yvector2)


# from scipy.sparse.linalg import svds
# import numpy as np
# from cupy.linalg import svd
# # 生成一个 9000 × 8000 的随机矩阵
# A = cp.random.rand(16000, 8000)

# # 计算前 k=100 个奇异值及其对应的 U、V 矩阵
# k = 100
# U, S, Vt = svd(A, full_matrices=False)

# # 还原矩阵 A（近似）
# A_reconstructed = U @ np.diag(S) @ Vt

# # 计算误差
# error = np.linalg.norm(A - A_reconstructed) / np.linalg.norm(A)
# print("重构误差:", error)


from joblib import dump, load

blocks_to_process = joblib.load('core_turkey/blocks_to_process.joblib')
for i in range(100):
    arr=blocks_to_process[i].U_1s
    if np.any(arr > np.iinfo(np.int32).max):  
        print("数组包含超大整数！")


