import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.sparse import coo_matrix
from scipy.interpolate import lagrange
from scipy.interpolate import RegularGridInterpolator
import time
import SH_greenfunction
import DH_greenfunction
from joblib import dump, load
from mpi_config import comm, rank, size
#A1=np.load('bp5t_core/A1d.npy')
#A1s=np.load('A1s.npy')
from scipy.sparse.linalg import svds
import sys
from mpi4py import MPI

TASK_TAG = 1
RESULT_TAG = 2
STOP_TAG = 3  # 结束信号


def bounding_box(cluster, points):
    """
    计算给定点簇的 bounding box 范围
    :param cluster: 点索引列表
    :param points: 所有点坐标的矩阵，形状 (n, d)
    :return: (a, b) 分别是 bounding box 的最小和最大坐标
    """
    #print(points,cluster)
    a = np.min(points[cluster], axis=0)  # 计算每个维度上的最小值
    b = np.max(points[cluster], axis=0)  # 计算每个维度上的最大值
    return a, b

def diameter(a, b):
    """
    计算 bounding box 的直径
    :param a: bounding box 的最小坐标
    :param b: bounding box 的最大坐标
    :return: 直径
    """
    return np.sqrt(np.sum((b - a) ** 2))

def distance(a_tau, b_tau, a_sigma, b_sigma):
    """
    计算两个 bounding box 之间的距离
    :param a_tau, b_tau: 第一个 bounding box 的最小和最大坐标
    :param a_sigma, b_sigma: 第二个 bounding box 的最小和最大坐标
    :return: 两个 bounding box 之间的距离
    """
    return np.sqrt(np.sum((np.maximum(0, a_sigma - b_tau) ** 2) + 
                          (np.maximum(0, a_tau - b_sigma) ** 2)))


def is_admissible(cluster_tau,cluster_sigma,points, eta=2.0):
    """
    判断两个 bounding box 是否满足可接受性条件
    :param a_tau, b_tau: 第一个 bounding box 的最小和最大坐标
    :param a_sigma, b_sigma: 第二个 bounding box 的最小和最大坐标
    :param eta: 可接受性条件参数 (通常 0 < η < 1)
    :return: 是否满足 admissibility condition
    """
    a_tau, b_tau = bounding_box(cluster_tau, points)
    a_sigma, b_sigma = bounding_box(cluster_sigma, points)

    d_tau = diameter(a_tau, b_tau)
    d_sigma = diameter(a_sigma, b_sigma)
    d_Q = distance(a_tau, b_tau, a_sigma, b_sigma)
    
    return min(d_tau, d_sigma) <= eta * d_Q






def SVD_recompress(U_ACA, V_ACA, eps):
    """
    Recompress an ACA matrix approximation via SVD.

    :param U_ACA: The left-hand approximation matrix.
    :param V_ACA: The right-hand approximation matrix.
    :param eps: The tolerance of the approximation. The convergence condition is
        in terms of the difference in Frobenius norm between the target matrix
        and the approximation.

    :return U_SVD: The SVD recompressed left-hand approximation matrix.
    :return V_SVD: The SVD recompressed right-hand approximation matrix.
    """
    UQ, UR = np.linalg.qr(U_ACA)
    VQ, VR = np.linalg.qr(V_ACA.T)
    W, SIG, Z = np.linalg.svd(UR.dot(VR.T))

    frob_K = np.sqrt(np.cumsum(SIG[::-1] ** 2))[::-1]
    r = np.argmax(frob_K < eps)

    U = UQ.dot(W[:, :r] * SIG[:r])
    V = Z[:r, :].dot(VQ.T)
    return U, V





class TreeNode:
    """ 二叉树节点，存储 H 矩阵的 BlockTree 结构 """
    def __init__(self, indices, split_dim=None, split_value=None,level=0):
        self.indices = indices  # 当前节点存储的点索引
        self.split_dim = split_dim  # 划分维度
        self.split_value = split_value  # 划分阈值
        self.level=level
        self.left = None  # 左子树
        self.right = None  # 右子树

def build_block_tree(cluster, points,min_size=16,depth=0):
    """
    递归构造 BlockTree（二叉树）
    :param cluster: 需要划分的索引集合
    :param points: 所有点的坐标矩阵，形状 (n, d)
    :param min_size: 簇最小大小，达到该值不再分裂
    :return: 根节点
    """
    if len(cluster) <= min_size:
        return TreeNode(cluster,level=depth)  # 叶子节点，不再划分

    d = points.shape[1]  # 维度
    alpha = np.min(points, axis=0)  # 计算各维度上的最小值
    beta = np.max(points, axis=0)   # 计算各维度上的最大值
    #print(alpha,beta)
    # 选择分裂方向，使得 beta_j - alpha_j 最大
    j_max = np.argmax(beta - alpha)
    # 计算分裂阈值
    gamma = (alpha[j_max] + beta[j_max]) / 2

    #index1=np.where(points[:, j_max] <= gamma)[0]
    #index2=np.where(points[:, j_max] > gamma)[0]
    #print(len(index1),len(cluster))
    #tau_1, tau_2 = np.zeros(len(index1)), np.zeros(len(index2))
    index1=points[:, j_max] <= gamma
    index2=points[:, j_max] > gamma
    tau_1=cluster[index1]
    tau_2=cluster[index2]
    #if(depth<=3):
    #    print(depth,len(tau_1),len(tau_2))

    #if ((len(tau_1)< min_size) or (len(tau_2)< min_size)):
    if ((len(tau_1)==0) or (len(tau_2)==0)):
        return TreeNode(cluster,level=depth) 
    
    node = TreeNode(cluster, split_dim=j_max, split_value=gamma,level=depth)
    node.left = build_block_tree(tau_1, points[index1], min_size,depth+1)
    node.right = build_block_tree(tau_2, points[index2], min_size,depth+1)
    
    return node

def print_tree(node, depth=0):
    """ 递归打印二叉树结构 """
    if node is None:
        return
    print(" " * (depth * 2), f"Node: {node.indices}, split_dim={node.split_dim}, split_value={node.split_value}")
    print_tree(node.left, depth + 1)
    print_tree(node.right, depth + 1)



class Block:
    def __init__(self, row_cluster, col_cluster,row_index,col_index, children=None, level=0):
        """
        HMatrix中的Block,支持多层递归分块。
        :param row_cluster: 行索引
        :param col_cluster: 列索引
        :param data: 叶子块存储的矩阵数据
        :param children: 若是父块，则包含子块列表
        :param level: 记录当前块的层级
        """
        self.row_cluster = row_cluster
        self.col_cluster = col_cluster
        self.row_index=row_index
        self.col_index=col_index
        #self.jud_svd=True
        self.U_1s=[]
        self.S_1s=[]
        self.Vt_1s=[]
        self.U_2s=[]
        self.S_2s=[]
        self.Vt_2s=[]
        self.U_Bs=[]
        self.S_Bs=[]
        self.Vt_Bs=[]

        self.U_1d=[]
        self.S_1d=[]
        self.Vt_1d=[]
        self.U_2d=[]
        self.S_2d=[]
        self.Vt_2d=[]
        self.U_Bd=[]
        self.S_Bd=[]
        self.Vt_Bd=[]

        self.ACA_dict={}



        self.Mf_A1s=[]
        self.Mf_A2s=[]
        self.Mf_Bs=[]
        self.Mf_A1d=[]
        self.Mf_A2d=[]
        self.Mf_Bd=[]

        self.U_ACA=[]
        self.V_ACA=[]

        #self.data = data  # 仅叶子节点存储数据
        self.children = children if children is not None else []
        self.level = level  # 记录层级
        
        self.judproc=False #true means already calculate core funcs
        self.judsvd=False #ture means it is admissible

    def setdata(self,data1):
        self.data=data1

    def is_leaf(self):
        """判断是否为叶子块"""
        return len(self.children) == 0

    def apply_low_rank_approximation(self, rank=10):
        """
        对叶子块进行低秩近似（SVD 分解）
        :param rank: 低秩近似的阶数
        """
        if self.is_leaf() and self.data is not None:
            U, S, Vt = np.linalg.svd(self.data, full_matrices=False)
            k = min(rank, len(S))  # 取最大可用秩
            self.data = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
            return self.data
        return None


class BlockTree:
    def __init__(self, root_block,nodelst,elelst,eleVec,mu_,lambda_, xg,halfspace_jud, mini_leaf):
        """HMatrix的BlockTree结构,支持多层分块"""
        self.root_block = root_block
        self.nodelst=nodelst
        self.elelst=elelst
        self.eleVec=eleVec
        self.xg=xg
        self.mu_=mu_
        self.lambda_=lambda_
        self.halfspace_jud=halfspace_jud
        #self.Intep_rowindex=[]
        #self.Intep_colindex=[]
        self.count=0
        self.size=0
        self.mini_leaf=mini_leaf
        self.thread_svd=mini_leaf
        self.thread_comm=5000
        self.sparse_csr=[]
        self.yvector=np.zeros(len(xg))
        self.maxdepth=0
        self.progress_tra=0
        self.blocks_to_process=[]
        self.size_local_blocks=0


        
    
    def GetTtstress(self,Stress,col_cluster):
        Tra=[]
        #print(self.eleVec.shape)
        for k in range(col_cluster.shape[0]):
            i=col_cluster[k]
            ev11,ev12,ev13=self.eleVec[i,0],self.eleVec[i,1],self.eleVec[i,2]
            ev21,ev22,ev23=self.eleVec[i,3],self.eleVec[i,4],self.eleVec[i,5]
            ev31,ev32,ev33=self.eleVec[i,6],self.eleVec[i,7],self.eleVec[i,8]

            Tr1=Stress[0,k]*ev31+Stress[3,k]*ev32+Stress[4,k]*ev33
            Tr2=Stress[3,k]*ev31+Stress[1,k]*ev32+Stress[5,k]*ev33
            Trn=Stress[4,k]*ev31+Stress[5,k]*ev32+Stress[2,k]*ev33

            Tt1=Tr1*ev11+Tr2*ev12+Trn*ev13
            Tt2=Tr1*ev21+Tr2*ev22+Trn*ev23
            Tn=Tr1*ev31+Tr2*ev32+Trn*ev33
            Tra.append([Tt1,Tt2,Tn])
        Tra=np.array(Tra)
        return Tra
    

    def ACA_plus_full_trace(self,n_rows, n_cols, calc_rows, calc_cols, row_cluster,col_cluster,eps, max_iter=None, verbose=False):
        us_A1s,vs_A1s=[],[]
        us_A2s,vs_A2s=[],[]
        us_Bs,vs_Bs=[],[]

        us_A1d,vs_A1d=[],[]
        us_A2d,vs_A2d=[],[]
        us_Bd,vs_Bd=[],[]


        prevIstar, prevJstar = [], []


        def argmax_not_in_list(arr, disallowed):
            arg_sorted = arr.argsort()
            max_idx = arg_sorted.shape[0] - 1
            while max_idx >= 0:
                if arg_sorted[max_idx] not in disallowed:
                    return arg_sorted[max_idx]
                max_idx -= 1
            raise RuntimeError("All entries disallowed.")

        def calc_residual_rows(Istart, Iend):
            A1s_out,A2s_out,Bs_out,A1d_out,A2d_out,Bd_out = calc_rows(Istart, Iend,row_cluster,col_cluster)
            #print(A1s_out.shape,row_cluster.shape,col_cluster.shape)
            for i in range(len(us_A1s)):
                A1s_out -= us_A1s[i][Istart:Iend][:, None] * vs_A1s[i][None, :]
                A2s_out -= us_A2s[i][Istart:Iend][:, None] * vs_A2s[i][None, :]
                Bs_out -= us_Bs[i][Istart:Iend][:, None] * vs_Bs[i][None, :]
                A1d_out -= us_A1d[i][Istart:Iend][:, None] * vs_A1d[i][None, :]
                A2d_out -= us_A2d[i][Istart:Iend][:, None] * vs_A2d[i][None, :]
                Bd_out -= us_Bd[i][Istart:Iend][:, None] * vs_Bd[i][None, :]
            return A1s_out,A2s_out,Bs_out,A1d_out,A2d_out,Bd_out

        def calc_residual_cols(Jstart, Jend):
            A1s_out,A2s_out,Bs_out,A1d_out,A2d_out,Bd_out = calc_cols(Jstart, Jend, row_cluster,col_cluster)
            for i in range(len(us_A1s)):
                A1s_out -= vs_A1s[i][Jstart:Jend][None,:] * us_A1s[i][:, None]
                A2s_out -= vs_A2s[i][Jstart:Jend][None,:] * us_A2s[i][:, None]
                Bs_out -= vs_Bs[i][Jstart:Jend][None,:] * us_Bs[i][:, None]
                A1d_out -= vs_A1d[i][Jstart:Jend][None,:] * us_A1d[i][:, None]
                A2d_out -= vs_A2d[i][Jstart:Jend][None,:] * us_A2d[i][:, None]
                Bd_out -= vs_Bd[i][Jstart:Jend][None,:] * us_Bd[i][:, None]
            return A1s_out,A2s_out,Bs_out,A1d_out,A2d_out,Bd_out


        def reset_reference_row(Iref):
            while True:
                Iref = (Iref + 3) % n_rows
                Iref -= Iref % 3
                if Iref not in prevIstar:
                    break
            out,out1,out2,out3,out4,out5=calc_residual_rows(Iref, Iref + 3)
            return out,out1,out2,out3,out4,out5, Iref

        def reset_reference_col(Jref):
            while True:
                Jref = (Jref + 3) % n_cols
                Jref -= Jref % 3
                if Jref not in prevJstar:
                    break
            out,out1,out2,out3,out4,out5=calc_residual_cols(Jref, Jref + 3)
            return out,out1,out2,out3,out4,out5, Jref

        if max_iter is None:
            max_iter = min(n_rows, n_cols)

        RIstar = np.zeros(n_cols)
        RJstar = np.zeros(n_rows)
        RIstar1 = np.zeros(n_cols)
        RJstar1 = np.zeros(n_rows)
        RIstar2 = np.zeros(n_cols)
        RJstar2 = np.zeros(n_rows)
        RIstar3 = np.zeros(n_cols)
        RJstar3 = np.zeros(n_rows)
        RIstar4 = np.zeros(n_cols)
        RJstar4 = np.zeros(n_rows)
        RIstar5 = np.zeros(n_cols)
        RJstar5 = np.zeros(n_rows)

        Iref = np.random.randint(n_rows) - 3
        Jref = np.random.randint(n_cols) - 3
        RIref,RIref1,RIref2,RIref3,RIref4,RIref5, Iref = reset_reference_row(Iref)
        RJref,RJref1,RJref2,RJref3,RJref4,RJref5, Jref = reset_reference_col(Jref)
        
        for k in range(max_iter):
            print(k,'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',flush=True)
            maxabsRIref = np.max(np.abs(RIref), axis=0)
            Jstar = argmax_not_in_list(maxabsRIref, prevJstar)

            maxabsRJref = np.max(np.abs(RJref), axis=1)
            Istar = argmax_not_in_list(maxabsRJref, prevIstar)

            Jstar_val = maxabsRIref[Jstar]
            Istar_val = maxabsRJref[Istar]

            if Istar_val > Jstar_val:
                out,out1,out2,out3,out4,out5 = calc_residual_rows(Istar, Istar + 1)
                RIstar[:],RIstar1[:],RIstar2[:],RIstar3[:],RIstar4[:],RIstar5[:]=out[0],out1[0],out2[0],out3[0],out4[0],out5[0]
                Jstar = argmax_not_in_list(np.abs(RIstar), prevJstar)
                out,out1,out2,out3,out4,out5 = calc_residual_cols(Jstar, Jstar + 1)
                RJstar[:],RJstar1[:],RJstar2[:],RJstar3[:],RJstar4[:],RJstar5[:]=out[:, 0],out1[:, 0],out2[:, 0],out3[:, 0],out4[:, 0],out5[:, 0]
            else:
                out,out1,out2,out3,out4,out5 = calc_residual_cols(Jstar, Jstar + 1)
                RJstar[:],RJstar1[:],RJstar2[:],RJstar3[:],RJstar4[:],RJstar5[:]=out[:, 0],out1[:, 0],out2[:, 0],out3[:, 0],out4[:, 0],out5[:, 0]
                Istar = argmax_not_in_list(np.abs(RJstar), prevIstar)
                out,out1,out2,out3,out4,out5 = calc_residual_rows(Istar, Istar + 1)
                RIstar[:],RIstar1[:],RIstar2[:],RIstar3[:],RIstar4[:],RIstar5[:]=out[0],out1[0],out2[0],out3[0],out4[0],out5[0]

            alpha = RIstar[Jstar]
            alpha1= RIstar1[Jstar]
            alpha2 = RIstar2[Jstar]
            alpha3= RIstar3[Jstar]
            alpha4 = RIstar4[Jstar]
            alpha5= RIstar5[Jstar]
            if np.abs(alpha) < 1e-14:
                if verbose:
                    print(f"Terminated at k={k} due to small pivot.")
                break

            prevIstar.append(Istar)
            prevJstar.append(Jstar)
            #Istar_list.append(Istar)
            #Jstar_list.append(Jstar)

            u_k = RJstar.copy()
            v_k = RIstar.copy()
            u_k1 = RJstar1.copy()
            v_k1 = RIstar1.copy()
            u_k2 = RJstar2.copy()
            v_k2 = RIstar2.copy()
            u_k3 = RJstar3.copy()
            v_k3 = RIstar3.copy()
            u_k4 = RJstar4.copy()
            v_k4 = RIstar4.copy()
            u_k5 = RJstar5.copy()
            v_k5 = RIstar5.copy()

            us_A1s.append(u_k)
            vs_A1s.append(v_k/alpha)
            #print('alpha',alpha)
            us_A2s.append(u_k1)
            vs_A2s.append(v_k1/alpha1)
            us_Bs.append(u_k2)
            if(abs(alpha2)<1e-10):
                vs_Bs.append(np.zeros(len(v_k2)))
            else:
                vs_Bs.append(v_k2/alpha2)
            us_A1d.append(u_k3)
            vs_A1d.append(v_k3/alpha3)
            us_A2d.append(u_k4)
            vs_A2d.append(v_k4/alpha4)
            us_Bd.append(u_k5)
            if(abs(alpha5)<1e-10):
                vs_Bd.append(np.zeros(len(v_k5)))
            else:
                vs_Bd.append(v_k5/alpha5)

            step_size = np.sqrt(np.sum(u_k ** 2) * np.sum((v_k / alpha) ** 2))
            if verbose:
                print(
                    f"iteration:{k},pivot row={Istar:4d}, pivot col={Jstar:4d}, "
                    f"step size={step_size:1.3e}, "
                    f"tolerance={eps:1.3e}", flush=True
                )

            if step_size < eps:
                break

            if k == max_iter - 1:
                break

            if Iref <= Istar < Iref + 3:
                RIref,RIref1,RIref2,RIref3,RIref4,RIref5, Iref = reset_reference_row(Iref)
            else:
                RIref -= u_k[Iref:Iref+3][:, None] * (v_k / alpha)[None, :]

            if Jref <= Jstar < Jref + 3:
                RJref,RJref1,RJref2,RJref3,RJref4,RJref5, Jref = reset_reference_col(Jref)
            else:
                RJref -= (v_k / alpha)[Jref:Jref+3][None, :] * u_k[:, None]

        U_ACA_A1s = np.array(us_A1s).T
        V_ACA_A1s = np.array(vs_A1s)

        U_ACA_A2s = np.array(us_A2s).T
        V_ACA_A2s = np.array(vs_A2s)

        U_ACA_Bs = np.array(us_Bs).T
        V_ACA_Bs = np.array(vs_Bs)

        U_ACA_A1d = np.array(us_A1d).T
        V_ACA_A1d = np.array(vs_A1d)

        U_ACA_A2d = np.array(us_A2d).T
        V_ACA_A2d = np.array(vs_A2d)

        U_ACA_Bd = np.array(us_Bd).T
        V_ACA_Bd = np.array(vs_Bd)
        
       

        trace_data = {
            "U_ACA_A1s": U_ACA_A1s,
            "V_ACA_A1s": V_ACA_A1s,
            "U_ACA_A2s": U_ACA_A2s,
            "V_ACA_A2s": V_ACA_A2s,
            "U_ACA_Bs": U_ACA_Bs,
            "V_ACA_Bs": V_ACA_Bs,
            "U_ACA_A1d": U_ACA_A1d,
            "V_ACA_A1d": V_ACA_A1d,
            "U_ACA_A2d": U_ACA_A2d,
            "V_ACA_A2d": V_ACA_A2d,
            "U_ACA_Bd": U_ACA_Bd,
            "V_ACA_Bd": V_ACA_Bd
        }

        return trace_data

    #使用 A(i, j) 构造行提取函数
    def calc_rows(self,Istart, Iend,row_cluster,col_cluster):
        #A1=A1s[np.ix_(row_cluster[Istart: Iend],col_cluster)]
        A1s,A2s,Bs,A1d,A2d,Bd=self.calc_stressfunc_trans(row_cluster[Istart: Iend],col_cluster)
        return A1s,A2s,Bs,A1d,A2d,Bd

    def calc_cols(self,Jstart, Jend,row_cluster,col_cluster):
        #A1=A1s[np.ix_(row_cluster[Istart: Iend],col_cluster)]
        A1s,A2s,Bs,A1d,A2d,Bd=self.calc_stressfunc_trans(row_cluster,col_cluster[Jstart: Jend])
        return A1s,A2s,Bs,A1d,A2d,Bd


        

    def calc_stressfunc(self,row_cluster,col_cluster):
        A1s=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        A2s=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        Bs=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        A1d=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        A2d=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        Bd=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        X,Y,Z=self.xg[col_cluster, 0], self.xg[col_cluster, 1], self.xg[col_cluster, 2]

        Ts, Ss, Ds = 0, 1, 0
        for i in range(len(row_cluster)):
            #print('row_cluster',row_cluster[i])
            P1=np.copy(self.nodelst[self.elelst[row_cluster[i],0]-1])
            P2=np.copy(self.nodelst[self.elelst[row_cluster[i],1]-1])
            P3=np.copy(self.nodelst[self.elelst[row_cluster[i],2]-1])
            if(self.halfspace_jud==True):
                Stress,_=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            else:
                Stress,_=DH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            Tra = self.GetTtstress(Stress,col_cluster)
            A1s[i]=Tra[:, 0]
            A2s[i]=Tra[:, 1]
            Bs[i]=Tra[:, 2]

        Ts, Ss, Ds = 0, 0, 1
        for i in range(len(row_cluster)):
            #print('row_cluster',row_cluster[i])
            P1=np.copy(self.nodelst[self.elelst[row_cluster[i],0]-1])
            P2=np.copy(self.nodelst[self.elelst[row_cluster[i],1]-1])
            P3=np.copy(self.nodelst[self.elelst[row_cluster[i],2]-1])
            if(self.halfspace_jud==True):
                Stress,_=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            else:
                Stress,_=DH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            Tra = self.GetTtstress(Stress,col_cluster)
            A1d[i]=Tra[:, 0]
            A2d[i]=Tra[:, 1]
            Bd[i]=Tra[:, 2]
        return A1s.transpose(),A2s.transpose(),Bs.transpose(),A1d.transpose(),A2d.transpose(),Bd.transpose()


    def calc_stressfunc_trans(self,row_cluster,col_cluster):
        A1s=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        A2s=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        Bs=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        A1d=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        A2d=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        Bd=np.zeros([len(row_cluster),len(col_cluster)], dtype=np.float32)
        X,Y,Z=self.xg[row_cluster, 0], self.xg[row_cluster, 1], self.xg[row_cluster, 2]

        Ts, Ss, Ds = 0, 1, 0
        for i in range(len(col_cluster)):
            #print('row_cluster',row_cluster[i])
            P1=np.copy(self.nodelst[self.elelst[col_cluster[i],0]-1])
            P2=np.copy(self.nodelst[self.elelst[col_cluster[i],1]-1])
            P3=np.copy(self.nodelst[self.elelst[col_cluster[i],2]-1])
            if(self.halfspace_jud==True):
                Stress,_=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            else:
                Stress,_=DH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            Tra = self.GetTtstress(Stress,row_cluster)
            A1s[:,i]=Tra[:, 0]
            A2s[:,i]=Tra[:, 1]
            Bs[:,i]=Tra[:, 2]

        Ts, Ss, Ds = 0, 0, 1
        for i in range(len(col_cluster)):
            #print('row_cluster',row_cluster[i])
            P1=np.copy(self.nodelst[self.elelst[col_cluster[i],0]-1])
            P2=np.copy(self.nodelst[self.elelst[col_cluster[i],1]-1])
            P3=np.copy(self.nodelst[self.elelst[col_cluster[i],2]-1])
            if(self.halfspace_jud==True):
                Stress,_=SH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            else:
                Stress,_=DH_greenfunction.TDstressHS(X,Y,Z,P1,P2,P3,Ss,Ds,Ts,self.mu_,self.lambda_)
            Tra = self.GetTtstress(Stress,row_cluster)
            A1d[:,i]=Tra[:, 0]
            A2d[:,i]=Tra[:, 1]
            Bd[:,i]=Tra[:, 2]
        return A1s,A2s,Bs,A1d,A2d,Bd



    def master(self,dir,blocks_to_process,num_workers):
        task_id = 1
        active_workers = num_workers
        if(len(blocks_to_process)==0):
            self.collect_blocks(self.root_block)
            self.blocks_to_process.sort(key=lambda block: len(block.row_cluster), reverse=True)
        else:
            self.blocks_to_process=blocks_to_process
        
        tasks_total = len(self.blocks_to_process)  # 假设总共有 10 个任务
        k=0
        for i in range(tasks_total):
            block=self.blocks_to_process[i]
            
            if(len(block.row_cluster)>500 or len(block.col_cluster)>500):
                k=k+1
                #print(len(block.row_cluster))
            
        print('tasks_total',k,tasks_total)

        # **初始任务分配**
        for worker_rank in range(1, num_workers + 1):
            MPI.COMM_WORLD.send({'task':self.blocks_to_process[task_id-1],'task_id':task_id}, dest=worker_rank, tag=TASK_TAG)
            print(f"Master: assign task {task_id} to Worker {worker_rank}, size: {len(self.blocks_to_process[task_id-1].row_cluster),len(self.blocks_to_process[task_id-1].col_cluster)}", flush=True)
            task_id += 1
        
        finish_task=0
        # **循环等待 worker 返回结果，并分配新任务**
        while active_workers > 0:
            result = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=RESULT_TAG)
            #print(result)
            jud_already_calsvd=result['jud_already_calsvd']
            worker_rank = result['worker']
            block=result['result']
            #print(block.Mf_A1s)
            rece_id=result['task']
            if(jud_already_calsvd==True):
                print(f"Master: Worker {worker_rank} for task {result['task']} already done before, size = {len(result['result'].row_cluster),len(result['result'].col_cluster)}", flush=True)
                finish_task=finish_task+1
            else:
                #worker_rank = result['worker']
                self.blocks_to_process[rece_id-1]=block
                if(rece_id-1<len(self.blocks_to_process)):
                    if(len(self.blocks_to_process[rece_id-1].row_cluster)>300):
                        dump(self.blocks_to_process,dir+ "/blocks_to_process.joblib")
                finish_task=finish_task+1
                print(f"Master: Worker {worker_rank} finish task {result['task']} , size = {len(result['result'].row_cluster),len(result['result'].col_cluster)}", flush=True)
                #print(f"{finish_task}/ {tasks_total}", flush=True)
            print(f"Master: complete {finish_task}/ {tasks_total}", flush=True)
            # **如果还有任务，继续分配**
            if task_id <= tasks_total:
                MPI.COMM_WORLD.send({'task':self.blocks_to_process[task_id-1],'task_id':task_id}, dest=worker_rank, tag=TASK_TAG)
                print(f"Master: assign task {task_id} to Worker {worker_rank}, size = {len(self.blocks_to_process[task_id-1].row_cluster),len(self.blocks_to_process[task_id-1].col_cluster)}", flush=True)
                task_id += 1
            else:
                active_workers -= 1
                print(f"active_workers remain {active_workers}", flush=True)
                # **没有任务了，发送结束信号**
                MPI.COMM_WORLD.send(None, dest=worker_rank, tag=STOP_TAG)
                
            
            # if(task_id>20):
            #     for i in range(3):
            #         print(i,self.blocks_to_process[i].Vt_1s)
            #     break
        

        dump(self.blocks_to_process,dir+ "/blocks_to_process.joblib")
        print("Master: all tasks completed.", flush=True)


    def worker(self):
        while True:
            status = MPI.Status()  # 创建 MPI 状态对象
            rectask = MPI.COMM_WORLD.recv(source=0, tag=MPI.ANY_TAG, status=status)

            # 通过 status.tag 检查是否是终止信号
            if status.tag == STOP_TAG:
                print(f"Worker {MPI.COMM_WORLD.Get_rank()}: end receive message, exit.", flush=True)
                #MPI.COMM_WORLD.Abort(1)
                break
                
            jud_already_calsvd=False
            # **执行任务**
            if(status.tag == TASK_TAG):
                #print('task',task, flush=True)
                block=rectask['task']
                if hasattr(block, 'judproc') and block.judproc:
                    jud_already_calsvd=True
                    print(f"Worker {MPI.COMM_WORLD.Get_rank()}: skipping SVD for task {rectask['task_id']} (judproc=True)", flush=True)
                    
                else:
                    try:
                        block=self.svd_worker(block)
                        #block=self.ACA_worker(block)
                        
                    except:
                        #print(f"Worker: svd failed for worker {rectask['task_id']}", flush=True)
                        print(f"Worker: svd of task {rectask['task_id']} failed for worker {MPI.COMM_WORLD.Get_rank()} ", flush=True)
                        break
                
                #print(block.Mf_A1s)
                #result = task * task  # 假设任务是计算平方
                print(f"Worker {MPI.COMM_WORLD.Get_rank()}: task {rectask['task_id']}, size = {len(block.row_cluster),len(block.col_cluster)}", flush=True)

            # **返回结果**
            MPI.COMM_WORLD.send({'worker': MPI.COMM_WORLD.Get_rank(), 'task': rectask['task_id'], 'result': block,'jud_already_calsvd':jud_already_calsvd}, dest=0, tag=RESULT_TAG)




    def collect_blocks(self, block):
        """ 递归遍历所有叶子节点，收集需要计算 SVD 的块 """
        if block.is_leaf():
            self.blocks_to_process.append(block)
        else:
            for child in block.children:
                self.collect_blocks(child)


    

    def ACA_worker(self,block):
        if(len(block.row_cluster) > self.thread_svd and len(block.col_cluster) > self.thread_svd):
            block.judsvd=True
            eps=1.0
            ACA_dict=self.ACA_plus_full_trace(
                len(block.row_cluster), len(block.col_cluster), 
                self.calc_rows, 
                self.calc_cols, 
                block.row_cluster,
                block.col_cluster,
                eps/50.0, 
                max_iter=500, 
                verbose=True)
            #print('ACA_dict:',ACA_dict, flush=True)
            if(len(block.row_cluster)>1000 or len(block.col_cluster)>1000):
                U_A1s, V_A1s = SVD_recompress(ACA_dict['U_ACA_A1s'], ACA_dict['V_ACA_A1s'], eps)
                U_A2s, V_A2s = SVD_recompress(ACA_dict['U_ACA_A2s'], ACA_dict['V_ACA_A2s'], eps)
                if not np.all(U_Bs == 0):
                    U_Bs, V_Bs = SVD_recompress(ACA_dict['U_ACA_Bs'], ACA_dict['V_ACA_Bs'], eps)
                else:
                    U_Bs, V_Bs=[],[]
                U_A1d, V_A1d = SVD_recompress(ACA_dict['U_ACA_A1d'], ACA_dict['V_ACA_A1d'], eps)
                U_A2d, V_A2d = SVD_recompress(ACA_dict['U_ACA_A2d'], ACA_dict['V_ACA_A2d'], eps)
                if not np.all(U_Bd == 0):
                    U_Bd, V_Bd = SVD_recompress(ACA_dict['U_ACA_Bd'], ACA_dict['V_ACA_Bd'], eps)
                else:
                    U_Bd, V_Bd=[],[]
                ACA_dict = {
                    "U_ACA_A1s": U_A1s,
                    "V_ACA_A1s": V_A1s,
                    "U_ACA_A2s": U_A2s,
                    "V_ACA_A2s": V_A2s,
                    "U_ACA_Bs": U_Bs,
                    "V_ACA_Bs": V_Bs,
                    "U_ACA_A1d": U_A1d,
                    "V_ACA_A1d": V_A1d,
                    "U_ACA_A2d": U_A2d,
                    "V_ACA_A2d": V_A2d,
                    "U_ACA_Bd": U_Bd,
                    "V_ACA_Bd": V_Bd
            }
            block.ACA_dict=ACA_dict
            print(f"ACA of task success", flush=True)
            
            self.size += (block.ACA_dict['U_ACA_A1s'].nbytes + block.ACA_dict['V_ACA_A1s'].nbytes)*6.0/(1024*1024)
        else:
            A1s,A2s,Bs,A1d,A2d,Bd=self.calc_stressfunc_trans(block.row_cluster,block.col_cluster)
            block.Mf_A1s=A1s
            block.Mf_A2s=A2s
            block.Mf_Bs=Bs
            block.Mf_A1d=A1d
            block.Mf_A2d=A2d
            block.Mf_Bd=Bd
            self.size +=A1s.nbytes*6/(1024*1024)
        block.judproc=True
        return block


    def svd_worker(self, block):
        """ 计算单个 block 的 SVD """
        if(len(block.row_cluster) > self.thread_svd and len(block.col_cluster) > self.thread_svd):
            block.judsvd=True
            #if(rank==0):
            #print('rank:',rank,'  Progress of stress function calculation:',self.progress_tra,' / ',self.size_local_blocks,' size:',len(block.row_cluster),len(block.col_cluster), flush=True)
            A1s,A2s,Bs,A1d,A2d,Bd=self.calc_stressfunc(block.row_cluster,block.col_cluster)
            self.progress_tra=self.progress_tra+1.0
            k=max(int(len(block.row_cluster)/20),self.mini_leaf)
            k1=min(len(block.row_cluster),len(block.col_cluster))
            if(k>=k1):
                k=k1-2
            #A1s = np.random.rand(9000, 4200)
            #print('!!!!!!!!!!!!!!!!!!!!!',A1s.shape)
            #U, S, Vt = np.linalg.svd(A1s, full_matrices=False)
            U, S, Vt = svds(A1s,k=k)
            block.U_1s=U
            block.S_1s=S
            block.Vt_1s=Vt
            

            U, S, Vt = svds(A2s, k=k)
            block.U_2s=U
            block.S_2s=S
            block.Vt_2s=Vt

            if not np.all(Bs == 0):
                U, S, Vt = svds(Bs, k=k)
                block.U_Bs=U
                block.S_Bs=S
                block.Vt_Bs=Vt

            U, S, Vt = svds(A1d, k=k)
            block.U_1d=U
            block.S_1d=S
            block.Vt_1d=Vt

            U, S, Vt = svds(A2d, k=k)
            block.U_2d=U
            block.S_2d=S
            block.Vt_2d=Vt

            if not np.all(Bd == 0):
                U, S, Vt = svds(Bd, k=k)
                block.U_Bd=U
                block.S_Bd=S
                block.Vt_Bd=Vt
                
            #else:
            #    print(f"Bd zero !!!! ", flush=True)
            
            if not np.all(Bd == 0):
                self.size += (U.nbytes + S.nbytes + Vt.nbytes)*6.0/(1024*1024)
            else:
                self.size += (U.nbytes + S.nbytes + Vt.nbytes)*4.0/(1024*1024)
            
        else:
            #if(rank==0):
            #print('rank:',rank,'  Progress of stress function calculation:',rank,self.progress_tra/self.size_local_blocks, flush=True)
            #print('rank:',rank,'  Progress of stress function calculation:',self.progress_tra,' / ',self.size_local_blocks,' size:',len(block.row_cluster),len(block.col_cluster), flush=True)
            A1s,A2s,Bs,A1d,A2d,Bd=self.calc_stressfunc(block.row_cluster,block.col_cluster)
            self.progress_tra=self.progress_tra+1.0
            block.Mf_A1s=A1s
            block.Mf_A2s=A2s
            block.Mf_Bs=Bs
            block.Mf_A1d=A1d
            block.Mf_A2d=A2d
            block.Mf_Bd=Bd
            self.size +=A1s.nbytes*6/(1024*1024)
            
        block.judproc=True
        return block
    

    

    def parallel_traverse_SVD(self, dir,plotHmatrix=False):
        if rank == 0:
            # **主进程：收集任务**
            self.collect_blocks(self.root_block)
            num_blocks = len(self.blocks_to_process)
            print('num_blocks:',num_blocks)
            #self.calc_large_svd()
            # **均匀划分任务**
            #blocks_index0=np.arange(0,num_blocks,1)
            chunk_size = (num_blocks+size) // size  # 计算每个进程平均任务量
            task_chunks = [self.blocks_to_process[i * chunk_size: (i + 1) * chunk_size] for i in range(size)]
            #self.blocks_index0_chunks = [blocks_index0[i * chunk_size: (i + 1) * chunk_size] for i in range(size)]
            N1=0
            for i in range(size):
                N1=N1+len(task_chunks[i])
            if(N1<num_blocks):
                for i in range(num_blocks-N1):
                    task_chunks[-1].append(self.blocks_to_process[N1+i])
                    #self.blocks_index0_chunks[-1].append(i)

            #rank_chunks = [[i] * len(chunk) for i, chunk in enumerate(task_chunks)]
            #print(rank_chunks)
        
        else:
            task_chunks = None  # 非主进程不需要初始化任务

        # **任务分发：主进程将任务分给各进程**
        local_blocks = comm.scatter(task_chunks, root=0)
        #local_index0= comm.scatter(self.blocks_index0_chunks, root=0)
        self.size_local_blocks=len(local_blocks)

        #indexed_blocks = list(enumerate(local_blocks)) 
        local_blocks.sort(key=lambda block: len(block.row_cluster), reverse=True)
        #indexed_blocks.sort(key=lambda x: len(x[1].row_cluster) * len(x[1].col_cluster), reverse=True)
        # 获取排序后的索引
        #sorted_indices = [index for index, _ in indexed_blocks]
        #local_blocks = [block for _, block in indexed_blocks]

        # **各进程计算 SVD**
        local_results = []
        for block in local_blocks:
            if(len(block.row_cluster)<=self.thread_comm and len(block.col_cluster)<=self.thread_comm):
                self.svd_worker(block)
            local_results.append(block)  # 计算后的 block
        #print('local_results:',len(local_results))

        #comm.Barrier()
        # **收集所有计算结果到 rank=0**
        gathered_results = comm.gather(local_results, root=0)
        #gathered_size = comm.gather(self.size, root=0)
        #total_sum = comm.reduce(local_sum, op=MPI.SUM, root=0)
        if(rank==0):
            self.blocks_to_process = [block for sublist in gathered_results for block in sublist]
            dump(self.blocks_to_process,dir+ "/blocks_to_process.joblib")
            if(plotHmatrix==True):
                self.blocks_plot_mpi(gathered_results)

        return local_results


    def parallel_block_scatter(self,blocks_to_process,plotHmatrix=False):
        if rank == 0:
            # **主进程：收集任务**
            #self.collect_blocks(self.root_block)
            num_blocks = len(blocks_to_process)
            print('num_blocks:',num_blocks)
            
            # **均匀划分任务**
            chunk_size = num_blocks // size  # 计算每个进程平均任务量
            task_chunks = [blocks_to_process[i * chunk_size: (i + 1) * chunk_size] for i in range(size)]
            #self.rank_chunks = [[i] * len(chunk) for i, chunk in enumerate(task_chunks)]
            N1=0
            for i in range(size):
                N1=N1+len(task_chunks[i])
            if(N1<num_blocks):
                for i in range(num_blocks-N1):
                    task_chunks[-1].append(blocks_to_process[N1+i])
                    #self.rank_chunks[-1].append(size-1)
            print('task_chunks',len(task_chunks))
        else:
            task_chunks = None  # 非主进程不需要初始化任务
        # **任务分发：主进程将任务分给各进程**
        
        local_blocks = comm.scatter(task_chunks, root=0)
        if(plotHmatrix==True):
            gathered_blocks = comm.gather(local_blocks, root=0)
        # **收集所有计算结果到 rank=0**
        if(rank==0):
            if(plotHmatrix==True):
                self.blocks_plot_mpi(gathered_blocks)
        return local_blocks

    



    def parallel_block_scatter_send(self, blocks_to_process, plotHmatrix=False):

    
        local_blocks = None
        n_task_per_proc=10
        if rank == 0:
            num_blocks = len(blocks_to_process)
            print('num_blocks:', num_blocks)
    
            # 手动平均分配任务
            counts = [num_blocks // size] * size
            for i in range(num_blocks % size):
                counts[i] += 1
    
            task_chunks = []
            start = 0
            for c in counts:
                task_chunks.append(blocks_to_process[start:start+c])
                start += c
    
            print('task_chunks length per process:', [len(c) for c in task_chunks])
    
            # 发送给其他进程
            for i in range(1, size):
                print(i,len(task_chunks[i]), flush=True)
                batch=int(len(task_chunks[i])/n_task_per_proc)
                start = 0
                for j in range(n_task_per_proc):
                    #task_dict=self.trans_class_to_dict(task_chunks[i])
                    if(j<n_task_per_proc-1):
                        comm.send(task_chunks[i][start:start+batch], dest=i, tag=77+j)
                    else:
                        comm.send(task_chunks[i][start:], dest=i, tag=77+j)
                    start += batch
                    #comm.send(task_chunks[i], dest=i, tag=77)
            local_blocks = task_chunks[0]  # rank=0 自己的任务

        else:
            # 非0进程接收任务
            #print('rank',rank,len(task_chunks[i]))
            #local_blocks = comm.recv(source=0, tag=77)
            #print('rank',rank,len(local_blocks))

            local_blocks = []
            for j in range(n_task_per_proc):
                task = comm.recv(source=0, tag=77 + j)
                for k in range(len(task)):
                    local_blocks.append(task[k])
           
    
        # 可选绘图处理
        if plotHmatrix:
            # 分步接收每个进程的数据
            #if(rank==0):
            # gathered_blocks = []
            # for i in range(size):
            #     if rank == i:
            #         gathered_blocks.append(local_blocks)
            #     comm.barrier()  # 确保所有进程都已完成处理
            if rank == 0:
                #print('gathered_blocks:',len(gathered_blocks))
                self.blocks_plot_mpi(task_chunks)
    
        return local_blocks
    


    def blocks_process_MVM(self,xvector,blocks_to_process,type):
        yvector=np.zeros(len(xvector))
        #start_time = time.time()  # 记录起始时间
        #for k in range(100):
        #A1s=np.load('WMF_core/A1s.npy')
        if(type=='A1s'):
            for i in range(len(blocks_to_process)):
                x_=xvector[blocks_to_process[i].row_cluster]
                if(blocks_to_process[i].judsvd==True):
                    Ax_rsvd = blocks_to_process[i].U_1s @ (blocks_to_process[i].S_1s * (blocks_to_process[i].Vt_1s @ x_))
                else:
                    #print(blocks_to_process[i].Mf_A1s.shape,len(blocks_to_process[i].row_cluster),len(blocks_to_process[i].col_cluster),x_.shape)
                    Ax_rsvd=blocks_to_process[i].Mf_A1s @ x_
                # #print(A1s[blocks_to_process[i].col_cluster,blocks_to_process[i].row_cluster].shape)
                # A1s_=A1s[np.ix_(blocks_to_process[i].col_cluster,blocks_to_process[i].row_cluster)]
                # Ax1=np.dot(A1s_,x_)
                # error=np.linalg.norm(Ax1 - Ax_rsvd)
                # if(error>100):

                #     print(i,rank,'error',error,len(blocks_to_process[i].row_cluster),len(blocks_to_process[i].col_cluster),blocks_to_process[i].is_leaf())
                #     U, S, Vt = svds(A1s_,k=len(blocks_to_process[i].col_cluster)-1)
                #     blocks_to_process[i].U_1s=U
                #     blocks_to_process[i].S_1s=S
                #     blocks_to_process[i].Vt_1s=Vt
                #     plt.pcolor(A1s_)
                #     plt.show()
                #     #print(U.shape,S.shape,Vt.shape,x_.shape)
                #     Ax_rsvd = U @ (S * (Vt @ x_))
                #     error=np.linalg.norm(Ax1 - Ax_rsvd)
                #     print(i,rank,'error',error)
                
                yvector[blocks_to_process[i].col_cluster]=yvector[blocks_to_process[i].col_cluster]+Ax_rsvd
        if(type=='A2s'):
            for i in range(len(blocks_to_process)):
                x_=xvector[blocks_to_process[i].row_cluster]
                if(blocks_to_process[i].judsvd==True):
                    Ax_rsvd = blocks_to_process[i].U_2s @ (blocks_to_process[i].S_2s * (blocks_to_process[i].Vt_2s @ x_))
                else:
                    #print(blocks_to_process[i].Mf_A1s.shape,len(blocks_to_process[i].row_cluster),len(blocks_to_process[i].col_cluster),x_.shape)
                    Ax_rsvd=blocks_to_process[i].Mf_A2s @ x_
                yvector[blocks_to_process[i].col_cluster]=yvector[blocks_to_process[i].col_cluster]+Ax_rsvd

        if(type=='Bs'):
            for i in range(len(blocks_to_process)):
                
                x_=xvector[blocks_to_process[i].row_cluster]
                if(blocks_to_process[i].judsvd==True):
                    if(len(blocks_to_process[i].Vt_Bs)>0):
                        Ax_rsvd = blocks_to_process[i].U_Bs @ (blocks_to_process[i].S_Bs * (blocks_to_process[i].Vt_Bs @ x_))
                    else:
                        Ax_rsvd=np.zeros(len(blocks_to_process[i].col_cluster))
                else:
                    #print(blocks_to_process[i].Mf_A1s.shape,len(blocks_to_process[i].row_cluster),len(blocks_to_process[i].col_cluster),x_.shape)
                    Ax_rsvd=blocks_to_process[i].Mf_Bs @ x_
                yvector[blocks_to_process[i].col_cluster]=yvector[blocks_to_process[i].col_cluster]+Ax_rsvd

        if(type=='A1d'):
            for i in range(len(blocks_to_process)):
                x_=xvector[blocks_to_process[i].row_cluster]
                if(blocks_to_process[i].judsvd==True):
                    Ax_rsvd = blocks_to_process[i].U_1d @ (blocks_to_process[i].S_1d * (blocks_to_process[i].Vt_1d @ x_))
                else:
                    #print(blocks_to_process[i].Mf_A1s.shape,len(blocks_to_process[i].row_cluster),len(blocks_to_process[i].col_cluster),x_.shape)
                    Ax_rsvd=blocks_to_process[i].Mf_A1d @ x_
                yvector[blocks_to_process[i].col_cluster]=yvector[blocks_to_process[i].col_cluster]+Ax_rsvd

        if(type=='A2d'):
            for i in range(len(blocks_to_process)):
                x_=xvector[blocks_to_process[i].row_cluster]
                if(blocks_to_process[i].judsvd==True):
                    Ax_rsvd = blocks_to_process[i].U_2d @ (blocks_to_process[i].S_2d * (blocks_to_process[i].Vt_2d @ x_))
                else:
                    #print(blocks_to_process[i].Mf_A1s.shape,len(blocks_to_process[i].row_cluster),len(blocks_to_process[i].col_cluster),x_.shape)
                    Ax_rsvd=blocks_to_process[i].Mf_A2d @ x_
                yvector[blocks_to_process[i].col_cluster]=yvector[blocks_to_process[i].col_cluster]+Ax_rsvd

        if(type=='Bd'):
            for i in range(len(blocks_to_process)):
                
                x_=xvector[blocks_to_process[i].row_cluster]
                if(blocks_to_process[i].judsvd==True):
                    if(len(blocks_to_process[i].Vt_Bd)>0):
                        Ax_rsvd = blocks_to_process[i].U_Bd @ (blocks_to_process[i].S_Bd * (blocks_to_process[i].Vt_Bd @ x_))
                    else:
                        Ax_rsvd=np.zeros(len(blocks_to_process[i].col_cluster))
                else:
                    #print(blocks_to_process[i].Mf_A1s.shape,len(blocks_to_process[i].row_cluster),len(blocks_to_process[i].col_cluster),x_.shape)
                    Ax_rsvd=blocks_to_process[i].Mf_Bd @ x_
                yvector[blocks_to_process[i].col_cluster]=yvector[blocks_to_process[i].col_cluster]+Ax_rsvd
        #end_time = time.time()  # 记录结束时间
        #elapsed_time = end_time - start_time  # 计算时间差
        return yvector

    def blocks_plot_mpi(self,gathered_results):
        color1=['darkred','lime','blue','y','cyan','darkgreen','steelblue','tomato','chocolate','slateblue']*10
        plt.figure(figsize=(10,10))
        #print('gathered_results',len(gathered_results))
        for i in range(len(gathered_results)):
        #for i in range(4):
            for j in range(len(gathered_results[i])):
                block=gathered_results[i][j]
                rowleftsize=len(block.row_cluster)
                colleftsize=len(block.col_cluster)
                
                midr=block.row_index[0]+rowleftsize
                midc=block.col_index[0]+colleftsize
                plt.plot([block.col_index[0],block.col_index[-1]],[midr,midr],c=color1[i])
                plt.plot([midc,midc],[block.row_index[0],block.row_index[-1]],c=color1[i])
        
        plt.xlim(0,len(self.xg)-1)
        plt.ylim(0,len(self.xg)-1)
        plt.savefig('HmatrixStru_mpi.png',dpi=500)
        #plt.show()




    def apply_spr_values(self,sparse_csr):
        rows, cols = sparse_csr.nonzero()
        print('Store leaf stress data index in sparse matrix ...')
        #self.A1s_spr,self.A2s_spr,self.Bs_spr,self.A1d_spr,self.A2d_spr,self.Bd_spr=self.calc_stressfunc(rows, cols)
        
        # for i in range(sparse_csr.shape[0]):
        #     index1=np.where(rows==i)[0]
        #     tar_cols=cols[index1]

        #     #clac stress function
        #     self.A1s_spr,self.A2s_spr,self.Bs_spr,self.A1d_spr,self.A2d_spr,self.Bd_spr=self.calc_stressfunc([i], tar_cols)
        #     values=A1[i,tar_cols]
        #     sparse_csr[i,tar_cols]=values
        print('sparse matrix store completed')


    def traverse_and_apply(self, block=None, depth=0, max_level=2):
        """
        递归遍历 BlockTree 并在满足条件时进行低秩近似。
        :param block: 当前遍历的块
        :param depth: 当前层级
        :param max_level: 低秩近似的最大应用层级
        """
        if block is None:
            block = self.root_block

        indent = "    " * depth  # 视觉缩进
        if block.is_leaf():
            print(f"{indent}- Leaf Block: Level {block.level}, Rows {block.row_cluster}, Cols {block.col_cluster}, Data Shape: {block.data.shape}")
            if block.level <= max_level:  # 仅对 max_level 层以下的叶子节点做低秩近似
                block.apply_low_rank_approximation()
                print(f"{indent}  (Applied Low-Rank Approximation)")
        else:
            print(f"{indent}- Composite Block: Level {block.level}, Rows {block.row_cluster}, Cols {block.col_cluster}, {len(block.children)} sub-blocks")
            for child in block.children:
                self.traverse_and_apply(child, depth + 1, max_level)


# --------------------------
# 递归构造 BlockTree
# --------------------------

def create_recursive_blocks(row_cluster,col_cluster,row_index,col_index,points,plotHmatrix,mini_leaf=16, depth=0):
    """
    递归创建BlockTree
    :param matrix: 目标矩阵
    :param row_range: 当前块的行索引范围
    :param col_range: 当前块的列索引范围
    :param max_depth: 递归的最大深度
    :param depth: 当前深度
    :return: Block 对象
    """

    
    
    jud_admis=is_admissible(row_cluster.indices,col_cluster.indices,points, eta=2.0)
    if(len(row_cluster.indices)>5000 or len(col_cluster.indices)>5000):
        jud_admis=False
    if jud_admis == True or (len(row_cluster.indices) <= mini_leaf or len(col_cluster.indices) <= mini_leaf):
        # 终止条件：达到最大深度 or 矩阵块小到不能继续分块
        return Block(row_cluster.indices,col_cluster.indices,row_index,col_index, level=depth)

    row_cluster_left=row_cluster.left
    row_cluster_right=row_cluster.right
    col_cluster_left=col_cluster.left
    col_cluster_right=col_cluster.right

    if((row_cluster_left==None) or (row_cluster_right==None) or (col_cluster_left==None) or (col_cluster_right==None)):
        return Block(row_cluster.indices,col_cluster.indices,row_index,col_index, level=depth)

    rowleftsize=len(row_cluster_left.indices)
    #rowrightsize=len(row_cluster_right.indices)
    colleftsize=len(col_cluster_left.indices)
    #colrightsize=len(row_cluster_left.indices)
    #print('depth',depth)
    if(plotHmatrix==True):
        #print('!!!!!!!!!!!!!!')
        # midr=(row_index[0]+row_index[-1])/2.0
        # midc=(col_index[0]+col_index[-1])/2.0
        midr=row_index[0]+rowleftsize
        midc=col_index[0]+colleftsize
        plt.plot([col_index[0],col_index[-1]],[midr,midr],c='red')
        plt.plot([midc,midc],[row_index[0],row_index[-1]],c='red')
        #plt.show()

    # 计算中心索引
    #mid_row = len(row_range) // 2
    #mid_col = len(col_range) // 2

    

    

    # 创建四个子块
    children = [
        create_recursive_blocks(row_cluster_left, col_cluster_left,
                        row_index[:rowleftsize],col_index[:colleftsize],points, plotHmatrix,mini_leaf,depth + 1),
        create_recursive_blocks(row_cluster_left, col_cluster_right,
                        row_index[:rowleftsize],col_index[colleftsize:],points, plotHmatrix,mini_leaf,depth + 1),
        create_recursive_blocks(row_cluster_right, col_cluster_left, 
                        row_index[rowleftsize:],col_index[:colleftsize],points,plotHmatrix,mini_leaf,depth + 1),
        create_recursive_blocks(row_cluster_right, col_cluster_right, 
                        row_index[rowleftsize:],col_index[colleftsize:],points,plotHmatrix,mini_leaf,depth + 1),
    ]

    return Block(row_cluster.indices,col_cluster.indices,row_index,col_index,children=children, level=depth)



def createHmatrix(xg,nodelst,elelst,eleVec,mu_,lambda_,halfspace_jud,mini_leaf=32,plotHmatrix=False):
    cluster = np.arange(len(xg))
    cluster_raw =build_block_tree(cluster, xg)
    cluster_col =build_block_tree(cluster, xg)

    print(len(cluster_raw.left.indices))
    #testblock.print_tree(tree_root)
    if(plotHmatrix==True):
        plt.figure(figsize=(10,10))
    
    root_block = create_recursive_blocks(cluster_raw, cluster_col,cluster,cluster,xg,plotHmatrix,mini_leaf)
    print('Recursively traverse create the BlockTree.')
    tree_block = BlockTree(root_block,nodelst,elelst,eleVec,mu_,lambda_, xg,halfspace_jud,mini_leaf)
    print('Recursively traverse create the BlockTree completed.')
    #print('Recursively traverse the BlockTree to obtain the interpolation index positions.')
    #tree_block.traverse()
    #print('Recursively traverse completed.')
    #tree_block.Intep_rowindex=np.concatenate(tree_block.Intep_rowindex)
    #tree_block.Intep_colindex=np.concatenate(tree_block.Intep_colindex)
    #print(root_block.children[0].children[3].col_index)

    #sparse_matrix_coo = coo_matrix((np.ones(len(tree_block.Intep_rowindex)), 
    #        (tree_block.Intep_rowindex, tree_block.Intep_colindex)), shape=(len(xg), len(xg)))
    #sparse_csr = sparse_matrix_coo.tocsr()
    #sparse_csr=calc_Sgreenfunc_spr(sparse_csr)
    #tree_block.sparse_csr=sparse_csr
    #tree_block.apply_spr_values(sparse_csr)

    #print(root_block.row_cluster)
    # print('Recursively traverse the BlockTree to apply SVD...')
    # tree_block.traverse_SVD()
    # print('Apply SVD complete.')

    if(plotHmatrix==True):
        plt.xlim(cluster[0],cluster[-1])
        plt.ylim(cluster[0],cluster[-1])
        plt.savefig('HmatrixStru.png',dpi=500)
        #plt.show()
    return tree_block





# 生成测试矩阵
# n = 16  # 矩阵大小
# matrix = np.random.rand(n, n)

# # 创建递归 BlockTree（最大递归深度 4）
# root_block = create_recursive_blocks(matrix, list(range(n)), list(range(n)), max_depth=4)
# tree = BlockTree(root_block)

# # 仅对最大层级 2 以下的块进行低秩近似
# tree.traverse_and_apply(max_level=3)

# 遍历并打印结构
#tree.traverse()
