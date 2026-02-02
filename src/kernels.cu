#include <ctime>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <linux/limits.h>
#include <vector>
#include <cuda_fp16.h>

#include "../tester/utils.h"

template <typename T>
__global__ void trace_kernel(const T* d_input, T* d_result, size_t diagonal_len, size_t cols) {
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 取对角元素
  T diagonal_val = T(0);
  if (idx < diagonal_len) {
    size_t pos = idx * cols + idx;
    diagonal_val = d_input[pos];
  }

  // warp 内规约
  for (int offset = 16; offset > 0; offset >>= 1) {
    diagonal_val += __shfl_down_sync(0xffffffff, diagonal_val, offset);
  }

  __shared__ T warp_sums[32]; // 最多 32 个 warp (1024 / 32)
  int lane = tid % 32;        // 线程在 warp 内的编号(0~31)
  int warp_id = tid / 32;     // 线程属于第几个 warp (0~7)

  // 每个 warp 的 lane 0 把结果写入共享内存
  if (lane == 0) {
    warp_sums[warp_id] = diagonal_val;
  }
  __syncthreads();

  // 用第一个 warp 再对所有的 warp 的结果规约
  int num_warps = (blockDim.x + 31) / 32;
  if (tid < num_warps) {
    diagonal_val = warp_sums[tid];
  } else {
    diagonal_val = T(0);
  }

  // 第一个 warp 内再做一次规约
  if (warp_id == 0) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      diagonal_val += __shfl_down_sync(0xffffffff, diagonal_val, offset);
    }
  }

  // 只有线程 0 做 atomicAdd
  if (tid == 0) {
    atomicAdd(d_result, diagonal_val);
  }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  // TODO: Implement the trace function

  size_t diagonal_len = rows < cols ? rows : cols;
  if (diagonal_len == 0)
    return T(0);

  T *d_input, *d_result;
  cudaMalloc(&d_input, h_input.size() * sizeof(T));
  cudaMalloc(&d_result, sizeof(T));

  cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemset(d_result, 0, sizeof(T));

  int block_size = 256;
  int grid_size = (diagonal_len + block_size - 1) / block_size;
  trace_kernel<<<grid_size, block_size>>>(d_input, d_result, diagonal_len, cols);
  
  T h_result;
  cudaMemcpy(&h_result, d_result, sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_result);

  return h_result;
}

// float 和 half 类型的转换辅助函数
__device__ __forceinline__ float toFloat(float x) { return x; } 
__device__ __forceinline__ float toFloat(half x) { return __half2float(x); }

template<typename T>
__device__ __forceinline__ T fromFloat(float x);
template<>
__device__ __forceinline__ float fromFloat<float>(float x) { return x; }
template<>
__device__ __forceinline__ half fromFloat<half>(float x) { return __float2half(x); }

template <typename T>
__global__ void attention_kernel(
  const T* Q, const T* K, const T* V, T* O,
  int tgt_len, int src_len, int head_dim,
  int query_heads, int kv_heads, bool is_causal, float scale) {
  
  // 线程索引 -> (batch, head, tgt_pos)
  int batch_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int tgt_pos = blockIdx.z * blockDim.x + threadIdx.x;

  if (tgt_pos >= tgt_len)
    return;

  // GQA: 多个 query head 共享一个 kv head
  int heads_per_group = query_heads / kv_heads;
  int kv_head_idx = head_idx / heads_per_group;

  // 数据布局: [batch, seq_len, heads, head_dim]
  int q_offset = batch_idx * (tgt_len * query_heads * head_dim)
               + tgt_pos * (query_heads * head_dim) + head_idx * head_dim;
  
  int kv_batch_offset = batch_idx * (src_len * kv_heads * head_dim);
  int kv_seq_stride = kv_heads * head_dim;
  int kv_head_offset = kv_head_idx * head_dim;

  int o_offset = q_offset;

  // 计算 score 并找最大值
  float max_score = -1e20f;
  for (int src_pos = 0; src_pos < src_len; src_pos++) {
    if(is_causal && src_pos > tgt_pos)
      break;

    int k_offset = kv_batch_offset + src_pos * kv_seq_stride + kv_head_offset;

    // 计算 Q · K^T
    float score = 0.0f;
    for (int d = 0; d < head_dim; d++) {
      score += toFloat(Q[q_offset + d]) * toFloat(K[k_offset + d]);
    }
    score *= scale;
    max_score = fmax(max_score, score);
  }

  // 计算 softmax 分母
  float sum_exp = 0.0f;
  for (int src_pos = 0; src_pos < src_len; src_pos++) {
    if (is_causal && src_pos > tgt_pos)
      break;

    int k_offset = kv_batch_offset + src_pos * kv_seq_stride + kv_head_offset;

    float score = 0.0f;
    for (int d = 0; d < head_dim; d++) {
      score += toFloat(Q[q_offset + d]) * toFloat(K[k_offset + d]);
    }
    score *= scale;

    sum_exp += expf(score - max_score);
  }

  // 计算输出 O = softmax(score) · V
  for (int out_d = 0; out_d < head_dim; out_d++) {
    float out_val = 0.0f;

    for (int src_pos = 0; src_pos < src_len; src_pos++) {
      if (is_causal && src_pos > tgt_pos)
        break;

      int k_offset = kv_batch_offset + src_pos * kv_seq_stride + kv_head_offset;
      int v_offset = k_offset;

      float score = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        score += toFloat(Q[q_offset + d]) * toFloat(K[k_offset + d]);
      }
      score *= scale;

      float attention_weight = expf(score - max_score) / sum_exp;
      out_val += attention_weight * toFloat(V[v_offset + out_d]);
    }

    O[o_offset + out_d] = fromFloat<T>(out_val);
  }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // TODO: Implement the flash attention function
  T *d_q, *d_k, *d_v, *d_o;
  cudaMalloc(&d_q, h_q.size() * sizeof(T));
  cudaMalloc(&d_k, h_k.size() * sizeof(T));
  cudaMalloc(&d_v, h_v.size() * sizeof(T));
  cudaMalloc(&d_o, h_o.size() * sizeof(T));
  
  cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k.data(), h_k.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(T), cudaMemcpyHostToDevice);
  
  float scale = 1.0f / sqrt((float)head_dim);
  int block_size = 256;
  int grid_size = (target_seq_len + block_size - 1) / block_size;
  
  dim3 grid(batch_size, query_heads, grid_size);
  dim3 block(block_size);

  attention_kernel<<<grid, block>>>(
    d_q, d_k, d_v, d_o,
    target_seq_len, src_seq_len, head_dim,
    query_heads, kv_heads, is_causal, scale);

  cudaMemcpy(h_o.data(), d_o, h_o.size() * sizeof(T), cudaMemcpyDeviceToHost);

  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
