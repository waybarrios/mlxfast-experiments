// Copyright © 2023-2024 Apple Inc.

#include <cstdio>
#include <cstdlib>
#include <mutex>

#include "mlx/backend/common/broadcasting.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/reduce.h"
#include "mlx/backend/metal/unary.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

template <typename... Args>
auto get_quantized_kernel_wrapped(
    metal::Device& d,
    const std::string& name,
    const std::string& func,
    const std::string& mode,
    const std::string& type,
    int group_size,
    int bits,
    Args... args) {
  std::string template_def;
  std::string fname = ((mode == "affine") ? "affine_" : "fp_") + func;
  template_def = get_template_definition(
      name, fname, type, group_size, bits, std::forward<Args>(args)...);
  return get_quantized_kernel(d, name, template_def, mode);
}

template <typename... Args>
auto get_qmm_nax_kernel_wrapped(
    metal::Device& d,
    const std::string& name,
    const std::string& func,
    const std::string& mode,
    const std::string& type,
    int group_size,
    int bits,
    Args... args) {
  std::string template_def;
  std::string fname = ((mode == "affine") ? "affine_" : "fp_") + func;
  template_def = get_template_definition(
      name, fname, type, group_size, bits, std::forward<Args>(args)...);
  return get_qmm_nax_kernel(d, name, template_def, mode);
}

inline array
ensure_row_contiguous(const array& x, metal::Device& d, const Stream& s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    metal::get_command_encoder(s).add_temporary(x_copy);
    return x_copy;
  } else {
    return x;
  }
}

inline array ensure_row_contiguous_matrix(
    const array& x,
    metal::Device& d,
    const Stream& s) {
  if (x.ndim() < 2) {
    if (x.strides()[0] == 1) {
      return x;
    }
  } else {
    auto stride_0 = x.strides()[x.ndim() - 2];
    auto stride_1 = x.strides()[x.ndim() - 1];
    if (stride_0 == x.shape(-1) && stride_1 == 1) {
      return x;
    }
  }
  array x_copy = contiguous_copy_gpu(x, s);
  metal::get_command_encoder(s).add_temporary(x_copy);
  return x_copy;
}

inline int get_qmv_batch_limit(int D, int O, metal::Device& d) {
  auto arch_size = d.get_architecture().back();
  auto arch_gen = d.get_architecture_gen();
  if (arch_gen == 13 || arch_gen == 14) {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 14;
        } else if (D <= 4096 && O <= 4096) {
          return 10;
        } else {
          return 6;
        }
    }
  } else {
    switch (arch_size) {
      case 'd':
        if (D <= 2048 && O <= 2048) {
          return 32;
        } else if (D <= 4096 && O <= 4096) {
          return 18;
        } else {
          return 12;
        }
      default:
        if (D <= 2048 && O <= 2048) {
          return 18;
        } else if (D <= 4096 && O <= 4096) {
          return 12;
        } else {
          return 10;
        }
    }
  }
}

inline int add_strides_and_shapes(
    CommandEncoder& compute_encoder,
    bool skip,
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    int offset) {
  if (skip) {
    return 0;
  }

  // TODO: Collapse batch dimensions

  int x_batch_ndims = x.ndim() - 2;
  int w_batch_ndims = w.ndim() - 2;
  compute_encoder.set_bytes(x_batch_ndims, offset++);
  compute_encoder.set_vector_bytes(x.shape(), offset++);
  compute_encoder.set_vector_bytes(x.strides(), offset++);
  compute_encoder.set_bytes(w_batch_ndims, offset++);
  compute_encoder.set_vector_bytes(w.shape(), offset++);
  compute_encoder.set_vector_bytes(w.strides(), offset++);
  compute_encoder.set_vector_bytes(scales.strides(), offset++);
  if (biases) {
    compute_encoder.set_vector_bytes(biases->strides(), offset++);
  }

  return offset;
}

inline int add_gather_strides_and_shapes(
    CommandEncoder& compute_encoder,
    const array& lhs_indices,
    const array& rhs_indices,
    int offset) {
  auto [shape, strides] = collapse_contiguous_dims(
      lhs_indices.shape(), {lhs_indices.strides(), rhs_indices.strides()});
  int ndims = shape.size();

  compute_encoder.set_bytes(ndims, offset++);
  compute_encoder.set_vector_bytes(shape, offset++);
  compute_encoder.set_vector_bytes(strides[0], offset++);
  compute_encoder.set_vector_bytes(strides[1], offset++);

  return offset;
}

} // namespace

void qmv_quad(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  constexpr int quads_per_simd = 8;
  constexpr int results_per_quadgroup = 8;
  int bn = quads_per_simd * results_per_quadgroup;
  int simdgroup_size = 32;
  MTL::Size group_dims(simdgroup_size, 1, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());

  concatenate(
      kname,
      mode + "_qmv_quad_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_d_",
      K,
      B > 1 ? "_batch_1" : "_batch_0");
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qmv_quad", mode, type_string, group_size, bits, K, B > 1);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int bn = 8;
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  bool fast = N % bn == 0 && K % 512 == 0;

  concatenate(
      kname,
      mode + (fast ? "_qmv_fast_" : "_qmv_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      B > 1 ? "_batch_1" : "_batch_0");
  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      (fast ? "qmv_fast" : "qmv"),
      mode,
      type_string,
      group_size,
      bits,
      B > 1);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qvm_split_k(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  auto& compute_encoder = metal::get_command_encoder(s);

  int split_k = K > 8192 ? 32 : 8;
  int split_D = (K + split_k - 1) / split_k;
  int B = out.size() / M / N;
  B *= split_k;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims = MTL::Size(bk, num_simdgroups, 1);
  MTL::Size grid_dims = MTL::Size(M, N / bn, B);

  auto x_shape = x.shape();
  auto x_strides = x.strides();
  if (x_shape.size() == 1) {
    x_shape.insert(x_shape.begin(), 1);
    x_strides.insert(x_strides.begin(), 0);
  }

  int x_ndim = x_shape.size();
  int x_batch_ndims = x_ndim - 2;
  int w_batch_ndims = w.ndim() - 2;
  auto w_shape = w.shape();
  auto w_strides = w.strides();
  auto s_strides = scales.strides();

  // Add split_k dim with reshapes
  x_shape.insert(x_shape.end() - 2, split_k);
  x_shape.back() /= split_k;
  x_strides.insert(x_strides.end() - 2, split_D);
  x_strides[x_ndim - 1] = split_D;
  x_batch_ndims += 1;

  w_shape.insert(w_shape.end() - 2, split_k);
  w_shape[w.ndim() - 1] /= split_k;
  w_strides.insert(w_strides.end() - 2, split_D * w.shape(-1));
  w_batch_ndims += 1;
  s_strides.insert(s_strides.end() - 2, split_D * scales.shape(-1));

  int final_block_size = K - (split_k - 1) * split_D;

  auto temp_shape = out.shape();
  if (temp_shape.size() == 1) {
    temp_shape.insert(temp_shape.begin(), 1);
  }
  temp_shape.insert(temp_shape.end() - 2, split_k);
  array intermediate(temp_shape, x.dtype(), nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  compute_encoder.add_temporary(intermediate);

  std::string type_string = get_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      mode + "_qvm_split_k_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_spk_",
      split_k);

  // Encode and dispatch kernel
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qvm_split_k", mode, type_string, group_size, bits, split_k);

  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(intermediate, c++);
  compute_encoder.set_bytes(split_D, c++);
  compute_encoder.set_bytes(N, c++);

  compute_encoder.set_bytes(x_batch_ndims, c++);
  compute_encoder.set_vector_bytes(x_shape, c++);
  compute_encoder.set_vector_bytes(x_strides, c++);
  compute_encoder.set_bytes(w_batch_ndims, c++);
  compute_encoder.set_vector_bytes(w_shape, c++);
  compute_encoder.set_vector_bytes(w_strides, c++);
  compute_encoder.set_vector_bytes(s_strides, c++);
  if (biases) {
    auto b_strides = biases->strides();
    b_strides.insert(b_strides.end() - 2, split_D * biases->shape(-1));
    compute_encoder.set_vector_bytes(b_strides, c++);
  }
  compute_encoder.set_bytes(final_block_size, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  int axis = intermediate.ndim() - 3;
  ReductionPlan plan(
      ReductionOpType::ContiguousStridedReduce,
      {intermediate.shape(axis)},
      {intermediate.strides(axis)});
  strided_reduce_general_dispatch(
      intermediate, out, "sum", plan, {axis}, compute_encoder, d, s);
}

void qvm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims(bk, num_simdgroups, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + "_qvm_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      B > 1 ? "_batch_1" : "_batch_0");
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qvm", mode, type_string, group_size, bits, B > 1);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm_nax(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 64;
  int bn = 64;
  int bk = 64;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 64 == 0;
  bool aligned_M = M % 64 == 0;
  bool batched = B > 1;
  std::string type_string = get_type_string(x.dtype());
  static const bool static_laguna_shapes =
      env::get_var("DARKBLOOM_STATIC_NVFP4_SHAPES", "") != "0";
  const bool use_static_laguna_shape =
      static_laguna_shapes && transpose && aligned && !batched &&
      mode == "nvfp4" && type_string == "bfloat16_t" &&
      group_size == 16 && bits == 4 && !biases.has_value() &&
      ((K == 2048 && N == 1024) || (K == 512 && N == 2048));
  concatenate(
      kname,
      mode +
          (use_static_laguna_shape
               ? "_qmm_t_nax_static_"
               : (transpose ? "_qmm_t_nax_" : "_qmm_n_nax_")),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn,
      use_static_laguna_shape
          ? ("_k" + std::to_string(K) + "_n" + std::to_string(N) +
             "_alM_" + (aligned_M ? "true" : "false"))
          : "",
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      batched ? "_batch_1" : "_batch_0");
  std::string template_def;
  MTL::ComputePipelineState* kernel;
  if (use_static_laguna_shape) {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "qmm_t_nax_static",
        mode,
        type_string,
        group_size,
        bits,
        K,
        N,
        aligned_M,
        bm,
        bk,
        bn,
        wm,
        wn);
  } else if (transpose) {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "qmm_t_nax",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        batched,
        bm,
        bk,
        bn,
        wm,
        wn);
  } else {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "qmm_n_nax",
        mode,
        type_string,
        group_size,
        bits,
        batched,
        bm,
        bk,
        bn,
        wm,
        wn);
  }
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmm_nax(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 64;
  int bn = 64;
  int bk = 64;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 64 == 0;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_gather_qmm_t_nax_" : "_gather_qmm_n_nax_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm",
      bm,
      "_bn",
      bn,
      "_bk",
      bk,
      "_wm",
      wm,
      "_wn",
      wn,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "");
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "gather_qmm_t_nax",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        bm,
        bk,
        bn,
        wm,
        wn);
  } else {
    kernel = get_qmm_nax_kernel_wrapped(
        d,
        kname,
        "gather_qmm_n_nax",
        mode,
        type_string,
        group_size,
        bits,
        bm,
        bk,
        bn,
        wm,
        wn);
  }

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  if (metal::is_nax_available() && transpose && (K % 64 == 0) &&
      (env::enable_tf32() || x.dtype() != float32)) {
    return qmm_nax(
  }

  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 32 == 0;
  bool batched = B > 1;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_qmm_t_" : "_qmm_n_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "",
      batched ? "_batch_1" : "_batch_0");
  std::string template_def;
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_quantized_kernel_wrapped(
        d,
        kname,
        "qmm_t",
        mode,
        type_string,
        group_size,
        bits,
        aligned,
        batched);
  } else {
    kernel = get_quantized_kernel_wrapped(
        d, kname, "qmm_n", mode, type_string, group_size, bits, batched);
  }
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  add_strides_and_shapes(compute_encoder, B <= 1, x, w, scales, biases, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void qmm_splitk(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  // Choose split_k to target ~512 threadgroups
  int bm = 32, bn = 32;
  int n_tiles = (N + bn - 1) / bn;
  int m_tiles = (M + bm - 1) / bm;
  int current_tgs = n_tiles * m_tiles;
  int split_k = std::max(1, 512 / current_tgs);

  // Each K partition must be a whole number of BK-wide (32) K-tiles as well as
  int k_align = std::max(group_size, 32);
  split_k = std::min(split_k, K / k_align);

  // Ensure K divides evenly by split_k * k_align
  while (split_k > 1 && (K % (split_k * k_align) != 0)) {
    split_k--;
  }
  if (split_k <= 1) {
    return qmm(
        x, w, scales, biases, out, true, group_size, bits, M, N, K, d, s, mode);
  }

  int k_partition_size = K / split_k;
  int split_k_partition_stride = M * N;

  // Fused split-K replay (DARKBLOOM_QMM_SPLITK_FUSED=0 restores the shipped
  static const bool splitk_fused_enabled = [] {
    const char* raw = getenv("DARKBLOOM_QMM_SPLITK_FUSED");
    return !(raw && raw[0] == '0');
  }();
  if (splitk_fused_enabled && !biases && mode != "affine" &&
      M % 32 == 0 && N % 32 == 0) {
    auto& compute_encoder = metal::get_command_encoder(s);
    MTL::Size group_dims(32, 2, 2);
    MTL::Size grid_dims(n_tiles, m_tiles, 1);
    bool aligned = N % 32 == 0;
    std::string type_string = get_type_string(x.dtype());
    std::string kname;
    kname.reserve(64);
    concatenate(
        kname,
        mode + "_qmm_t_splitk_fused_",
        type_string,
        "_gs_",
        group_size,
        "_b_",
        bits,
        aligned ? "_alN_true" : "_alN_false");
    auto kernel = get_quantized_kernel_wrapped(
        d, kname, "qmm_t_splitk_fused", mode, type_string, group_size, bits, aligned);
    compute_encoder.set_compute_pipeline_state(kernel);
    int c = 0;
    compute_encoder.set_input_array(w, c++);
    compute_encoder.set_input_array(scales, c++);
    compute_encoder.set_input_array(x, c++);
    compute_encoder.set_output_array(out, c++);
    compute_encoder.set_bytes(K, c++);
    compute_encoder.set_bytes(N, c++);
    compute_encoder.set_bytes(M, c++);
    compute_encoder.set_bytes(k_partition_size, c++);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    return;
  }

  auto& compute_encoder = metal::get_command_encoder(s);
  auto temp_shape = out.shape();
  if (temp_shape.size() == 1) {
    temp_shape.insert(temp_shape.begin(), 1);
  }
  temp_shape.insert(temp_shape.begin(), split_k);
  array intermediate(temp_shape, x.dtype(), nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  compute_encoder.add_temporary(intermediate);

  // Grid: (N_tiles, M_tiles, split_k)
  MTL::Size group_dims(32, 2, 2);
  MTL::Size grid_dims(n_tiles, m_tiles, split_k);

  bool aligned = N % 32 == 0;
  std::string type_string = get_type_string(x.dtype());
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      mode + "_qmm_t_splitk_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      aligned ? "_alN_true" : "_alN_false");
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "qmm_t_splitk", mode, type_string, group_size, bits, aligned);

  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_output_array(intermediate, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.set_bytes(k_partition_size, c++);
  compute_encoder.set_bytes(split_k_partition_stride, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  // Sum across split_k dimension (axis 0)
  ReductionPlan plan(
      ReductionOpType::ContiguousStridedReduce,
      {intermediate.shape(0)},
      {intermediate.strides(0)});
  strided_reduce_general_dispatch(
      intermediate, out, "sum", plan, {0}, compute_encoder, d, s);
}

void gather_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  if (metal::is_nax_available() && transpose && (K % 64 == 0) &&
      (env::enable_tf32() || x.dtype() != float32)) {
    return gather_qmm_nax(
  }

  int B = out.size() / M / N;

  int wm = 2;
  int wn = 2;
  int bm = 32;
  int bn = 32;
  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

  std::string kname;
  kname.reserve(64);
  bool aligned = N % 32 == 0;
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_gather_qmm_t_" : "_gather_qmm_n_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      transpose ? (aligned ? "_alN_true" : "_alN_false") : "");
  MTL::ComputePipelineState* kernel;
  if (transpose) {
    kernel = get_quantized_kernel_wrapped(
        d, kname, "gather_qmm_t", mode, type_string, group_size, bits, aligned);
  } else {
    kernel = get_quantized_kernel_wrapped(
        d, kname, "gather_qmm_n", mode, type_string, group_size, bits);
  }

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(M, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  int bn = 8;
  int bk = 32;
  MTL::Size group_dims(bk, 2, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  bool fast = N % bn == 0 && K % 512 == 0;
  concatenate(
      kname,
      mode + (fast ? "_gather_qmv_fast_" : "_gather_qmv_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits);

  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      (fast ? "gather_qmv_fast" : "gather_qmv"),
      mode,
      type_string,
      group_size,
      bits);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qvm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  int B = out.size() / M / N;

  constexpr int num_simdgroups = 2;
  constexpr int bk = 32;
  int bn = std::min(group_size, 32) * num_simdgroups;
  MTL::Size group_dims(bk, num_simdgroups, 1);
  MTL::Size grid_dims(M, (N + bn - 1) / bn, B);

  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + "_gather_qvm_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits);
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "gather_qvm", mode, type_string, group_size, bits);
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  int c = 0;
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases) {
    compute_encoder.set_input_array(*biases, c++);
  }
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(lhs_indices, c++);
  compute_encoder.set_input_array(rhs_indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(N, c++);
  c = add_strides_and_shapes(compute_encoder, false, x, w, scales, biases, c++);
  add_gather_strides_and_shapes(compute_encoder, lhs_indices, rhs_indices, c);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

// DARKBLOOM_PREFILL_GATHER_RUNSKIP: elide gather-GEMM run iterations whose
namespace {

// THE SHIPPED DEFAULT STRENGTH. This is the single line to change to resize the
constexpr int kDarkbloomDefaultRunSkipPct = 100;

bool darkbloom_gather_run_skip_enabled() {
  static const bool enabled = [] {
    return env::get_var("DARKBLOOM_PREFILL_GATHER_RUNSKIP", "") != "0";
  }();
  return enabled;
}

// Magnitude in percent of output row-tiles, 1..100.
int darkbloom_gather_run_skip_pct() {
  static const int pct = [] {
    auto v = env::get_var("DARKBLOOM_PREFILL_GATHER_RUNSKIP", "");
    if (v.empty()) {
      return kDarkbloomDefaultRunSkipPct;
    }
    if (v == "1") {
      return 100;
    }
    if (v.rfind("1:", 0) != 0) {
      return kDarkbloomDefaultRunSkipPct;
    }
    int p = std::atoi(v.c_str() + 2);
    return (p < 1) ? 1 : ((p > 100) ? 100 : p);
  }();
  return pct;
}

// DARKBLOOM_STAGE_*: attack the per-run staging cost in

bool darkbloom_stage_flag(const char* name) {
  auto v = env::get_var(name, "");
  return v == "1";
}

bool darkbloom_stage_widest() {
  static const bool v = darkbloom_stage_flag("DARKBLOOM_STAGE_WIDEST");
  return v;
}

bool darkbloom_stage_wideld() {
  static const bool v = darkbloom_stage_flag("DARKBLOOM_STAGE_WIDELD");
  return v;
}

bool darkbloom_stage_runbar() {
  static const bool v = darkbloom_stage_flag("DARKBLOOM_STAGE_RUNBAR");
  return v;
}

bool darkbloom_stage_novol() {
  static const bool v = darkbloom_stage_flag("DARKBLOOM_STAGE_NOVOL");
  return v;
}

bool darkbloom_expert_aligned_gather() {
  static const bool v =
      env::get_var("DARKBLOOM_EXPERT_ALIGNED_GATHER", "") != "0";
  return v;
}

// DARKBLOOM_EXPERT_STAGE_WIDEST (default ON; "0" restores scalar staging as
bool darkbloom_expert_stage_widest() {
  static const bool v =
      env::get_var("DARKBLOOM_EXPERT_STAGE_WIDEST", "") != "0";
  return v;
}

// DARKBLOOM_EXPERT_STAGE_WIDELD (default ON; "0" restores scalar loads as
bool darkbloom_expert_stage_wideld() {
  static const bool v =
      env::get_var("DARKBLOOM_EXPERT_STAGE_WIDELD", "") != "0";
  return v;
}

// DARKBLOOM_EXPERT_GATHER_GROUPS (default 128; "64" restores the promoted
int darkbloom_expert_gather_groups() {
  static const int v = [] {
    auto s = env::get_var("DARKBLOOM_EXPERT_GATHER_GROUPS", "");
    if (s.empty()) {
      return 256;
    }
    int n = std::atoi(s.c_str());
    return (n > 0 && (256 % n) == 0) ? n : 256;
  }();
  return v;
}

// DARKBLOOM_STAGE_BM128: select the gather-GEMM m-tiling. NOT a function
int darkbloom_stage_bm128_variant() {
  static const int v = [] {
    auto s = env::get_var("DARKBLOOM_STAGE_BM128", "");
    if (s.empty()) {
      // Default 5 (2026-08-01, final): API absolutes across our four scored
      return 5;
    }
    if (s == "1") {
      return 1;
    }
    if (s == "2") {
      return 2;
    }
    if (s == "3") {
      return 3;
    }
    if (s == "4") {
      return 4;
    }
    if (s == "5") {
      return 5;
    }
    return 0;
  }();
  return v;
}

// Host half of the wide-access alignment contract. The kernel checks each
bool darkbloom_stage_wide_load_ok(
    const array& w,
    bool transpose,
    int bits,
    int N,
    int K,
    int bn) {
  if (bits != 4) {
    return false;
  }
  // Packed 4-bit: two weights per byte.
  if ((K % 2) != 0 || (N % 2) != 0) {
    return false;
  }
  // (1) buffer offset of the weight array itself.
  if ((w.offset() % 16) != 0) {
    return false;
  }
  // (2) per-expert stride, in bytes, exactly as the kernel computes it.
  const int64_t stride_w =
      transpose ? int64_t(N) * (K / 2) : int64_t(K) * (N / 2);
  if ((stride_w % 16) != 0) {
    return false;
  }
  // (3) tile column base, in bytes, for every tile the grid can produce.
  const int64_t col_step = transpose ? int64_t(bn) * (K / 2) : int64_t(bn) / 2;
  if ((col_step % 16) != 0) {
    return false;
  }
  return true;
}

} // namespace

// DARKBLOOM_GATHER_XMAJOR: fold this many ADJACENT BN-wide column tiles of
int darkbloom_gather_xmajor_ct() {
  static const int v = [] {
    const std::string s = env::get_var("DARKBLOOM_GATHER_XMAJOR", "");
    if (s.empty() || s == "0") {
      return 0;
    }
    if (s == "1") {
      return 4; // tuned default fold
    }
    const int ct = atoi(s.c_str());
    return (ct == 2 || ct == 4 || ct == 8 || ct == 16) ? ct : 0;
  }();
  return v;
}

// DARKBLOOM_SWIGLU_REGLOCAL: register-local swiglu epilogue in the
bool darkbloom_swiglu_reglocal() {
  static const bool v =
      env::get_var("DARKBLOOM_SWIGLU_REGLOCAL", "") != "0";
  return v;
}

bool darkbloom_bsearch_hoist() {
  static const bool v =
      env::get_var("DARKBLOOM_BSEARCH_HOIST", "") != "0";
  return v;
}

void gather_qmm_rhs_nax(
    const array& x_,
    const array& w_,
    const array& scales_,
    const std::optional<array>& biases_,
    const array& indices_,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string mode) {
  // Start by normalizing the indices
  array indices = ensure_row_contiguous(indices_, d, s);

  // Broadcast x with indices. If we are here that means lhs_indices were not
  auto broadcast_with_indices = [&d, &s, &indices](const array& x) {
    if (x.size() / x.shape(-2) / x.shape(-1) == indices.size()) {
      return ensure_row_contiguous(x, d, s);
    }

    auto x_shape = indices.shape();
    x_shape.push_back(x.shape(-2));
    x_shape.push_back(x.shape(-1));
    array new_x(std::move(x_shape), x.dtype(), nullptr, {});
    broadcast(x, new_x);
    return ensure_row_contiguous(new_x, d, s);
  };

  // Normalize the input arrays
  array x = broadcast_with_indices(x_);
  array w = ensure_row_contiguous(w_, d, s);
  array scales = ensure_row_contiguous(scales_, d, s);

  // TODO: Tune the block sizes
  int bm = 64, bn = 64, bk = 64;
  int wm = 2, wn = 2;
  const int bm128 = darkbloom_stage_bm128_variant();
  switch (bm128) {
    case 1: bm = 128; wm = 4; break;         // SM=32, less re-staging
    case 2: bm = 128; wm = 2; break;         // SM=64, predicted regression
    case 3: bm = 128; wm = 8; break;         // SM=16, both mechanisms
    case 4: bm = 64;  wm = 4; break;         // SM=16, 256 thr/TG, SHIPPED DEFAULT
    case 5: bm = 64;  wm = 4; wn = 1; break; // SM=16, 128 thr/TG, TN 2 -> 4
    default: break;                          // upstream: bm=64, wm=2, wn=2
  }

  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;
  const bool laguna_moe_shape =
      (K == 2048 && N == 1024) || (K == 512 && N == 2048);
  // wn == 1 admitted 2026-07-31 (GatherX): DARKBLOOM_STAGE_BM128=5's
  const bool expert_aligned =
      darkbloom_expert_aligned_gather() && mode != "affine" && transpose &&
      group_size == 16 && bits == 4 && laguna_moe_shape && M >= 64 &&
      align_N && align_K && bm == 64 && wm == 4 && (wn == 2 || wn == 1);
  std::string type_string = get_type_string(x.dtype());
  static const bool static_laguna_shapes =
      env::get_var("DARKBLOOM_STATIC_NVFP4_SHAPES", "") != "0";
  const bool static_expert_shape =
      expert_aligned && static_laguna_shapes && mode == "nvfp4" &&
      type_string == "bfloat16_t" && !biases_.has_value();
  // How many threadgroups the expert path spreads the 256 experts over; the
  const int egroups = darkbloom_expert_gather_groups();
  const bool expert_widest = expert_aligned && darkbloom_expert_stage_widest();
  // Certified per weight bank; the banks are prepared once at init and
  const bool expert_wideld = expert_aligned &&
      darkbloom_expert_stage_wideld() &&
      darkbloom_stage_wide_load_ok(w, transpose, bits, N, K, bn);

  // DARKBLOOM_STAGE2_GATHER ground truth at the DISPATCH site. The define
  {
    static const bool stage2_flag =
        env::get_var("DARKBLOOM_STAGE2_GATHER", "") == "1";
    static const bool trace_fusion =
        env::get_var("DARKBLOOM_TRACE_FUSION", "") == "1";
    if (stage2_flag || trace_fusion) {
      static std::once_flag stage2_once;
      std::call_once(stage2_once, [&]() {
        fprintf(
            stderr,
            "mlxfast: fusion %s: stage2_gather "
            "(dispatch expert=%d egroups=%d N=%d K=%d M=%d)\n",
            (stage2_flag && expert_aligned) ? "active" : "inactive",
            int(expert_aligned),
            egroups,
            N,
            K,
            M);
      });
    }
  }

  // DARKBLOOM_GATHER_XMAJOR ground truth at the DISPATCH site, same
  {
    static const int xmajor_trace_ct = darkbloom_gather_xmajor_ct();
    static const bool trace_fusion =
        env::get_var("DARKBLOOM_TRACE_FUSION", "") == "1";
    if (xmajor_trace_ct > 1 || trace_fusion) {
      static std::once_flag xmajor_once;
      std::call_once(xmajor_once, [&]() {
        fprintf(
            stderr,
            "mlxfast: fusion %s: gatherx "
            "(dispatch expert=%d ct=%d grid_x=%d N=%d K=%d M=%d)\n",
            (xmajor_trace_ct > 1 && expert_aligned) ? "active" : "inactive",
            int(expert_aligned),
            xmajor_trace_ct,
            (xmajor_trace_ct > 1 && expert_aligned)
                ? (N / bn) / xmajor_trace_ct
                : (N + bn - 1) / bn,
            N,
            K,
            M);
      });
    }
  }

  // Make the kernel name
  std::string kname;
  kname.reserve(64);
  concatenate(
      kname,
      mode +
          (static_expert_shape
               ? "_gather_qmm_rhs_expert_static_nax_nt_"
               : (expert_aligned
                      ? "_gather_qmm_rhs_expert_nax_nt_"
               : (transpose ? "_gather_qmm_rhs_nax_nt_"
                            : "_gather_qmm_rhs_nax_nn_"))),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm_",
      bm,
      "_bn_",
      bn,
      "_bk_",
      bk,
      "_wm_",
      wm,
      "_wn_",
      wn,
      static_expert_shape
          ? ("_k_" + std::to_string(K) + "_n_" + std::to_string(N))
          : "",
      expert_aligned
          ? ("_eg_" + std::to_string(egroups) + (expert_widest ? "_ws_1" : "_ws_0") +
             (expert_wideld ? "_wl_1" : "_wl_0"))
          : "");

  // Skipping dead runs is a pure work elision (see function constant 203 in
  const bool run_skip = darkbloom_gather_run_skip_enabled();
  const int run_skip_pct = run_skip ? darkbloom_gather_run_skip_pct() : 100;
  // DARKBLOOM_STAGE_*. Widening is additionally gated on the host-side
  const bool wide_ok =
      darkbloom_stage_wide_load_ok(w, transpose, bits, N, K, bn);
  const bool stage_widest = darkbloom_stage_widest();
  const bool stage_wideld = darkbloom_stage_wideld() && wide_ok;
  const bool stage_runbar = darkbloom_stage_runbar();
  const bool stage_novol = darkbloom_stage_novol();

  // Ground truth for the A/B harness. `stage_wideld` is silently downgraded
  if (darkbloom_stage_flag("DARKBLOOM_STAGE_TRACE")) {
    static std::once_flag once;
    std::call_once(once, [&]() {
      fprintf(
          stderr,
          "mlxfast: stage active: widest=%d wideld=%d(req=%d wide_ok=%d) "
          "runbar=%d novol=%d expert=%d expert_ws=%d expert_wl=%d bm128=%d "
          "bm=%d wm=%d wn=%d w.offset=%zu transpose=%d bits=%d N=%d K=%d "
          "bn=%d\n",
          int(stage_widest),
          int(stage_wideld),
          int(darkbloom_stage_wideld()),
          int(wide_ok),
          int(stage_runbar),
          int(stage_novol),
          int(expert_aligned),
          int(expert_widest),
          int(expert_wideld),
          bm128,
          bm,
          wm,
          wn,
          size_t(w.offset()),
          int(transpose),
          bits,
          N,
          K,
          bn);
    });
  }

  metal::MTLFCList func_consts;
  if (!expert_aligned) {
    func_consts = {
        {&align_M, MTL::DataType::DataTypeBool, 200},
        {&align_N, MTL::DataType::DataTypeBool, 201},
        {&align_K, MTL::DataType::DataTypeBool, 202},
        {&run_skip, MTL::DataType::DataTypeBool, 203},
        {&stage_widest, MTL::DataType::DataTypeBool, 204},
        {&stage_wideld, MTL::DataType::DataTypeBool, 205},
        {&stage_runbar, MTL::DataType::DataTypeBool, 206},
        {&stage_novol, MTL::DataType::DataTypeBool, 207},
    };
  }

  // And the kernel hash that includes the function constants
  std::string hash_name;
  hash_name.reserve(128);
  concatenate(
      hash_name,
      kname,
      "_align_M_",
      align_M ? 't' : 'n',
      "_align_N_",
      align_N ? 't' : 'n',
      "_align_K_",
      align_K ? 't' : 'n',
      "_rs_",
      run_skip ? 't' : 'n',
      "_stg_",
      stage_widest ? 'W' : 'n',
      stage_wideld ? 'L' : 'n',
      stage_runbar ? 'B' : 'n',
      stage_novol ? 'V' : 'n');

  // Get and set the kernel. Every expert-aligned instantiation (static and
  auto& compute_encoder = metal::get_command_encoder(s);
  MTL::ComputePipelineState* kernel;
  if (expert_aligned) {
    auto template_def = get_template_definition(
        kname,
        "fp_gather_qmm_rhs_expert_nax",
        get_type_string(x.dtype()),
        group_size,
        bits,
        bm,
        bn,
        bk,
        wm,
        wn,
        transpose,
        static_expert_shape ? K : 0,
        static_expert_shape ? N : 0,
        "bfloat",
        egroups,
        expert_widest,
        expert_wideld);
    kernel = get_qmm_nax_kernel(d, kname, template_def, mode);
  } else {
    kernel = get_gather_qmm_nax_kernel(
        d,
        kname,
        hash_name,
        func_consts,
        x,
        group_size,
        bits,
        mode,
        bm,
        bn,
        bk,
        wm,
        wn,
        transpose);
  }
  compute_encoder.set_compute_pipeline_state(kernel);

  MTL::Size group_dims(32, wn, wm);
  // DARKBLOOM_GATHER_XMAJOR: the expert kernel was compiled to walk
  const int xmajor_ct = expert_aligned ? darkbloom_gather_xmajor_ct() : 0;
  MTL::Size grid_dims(
      xmajor_ct > 1 ? (N / bn) / xmajor_ct : ((N + bn - 1) / bn),
      expert_aligned ? egroups : (M + bm - 1) / bm,
      1);

  int c = 0;
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases_) {
    array biases = ensure_row_contiguous(*biases_, d, s);
    compute_encoder.set_input_array(biases, c++);
  }
  compute_encoder.set_input_array(indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(K, c++);
  compute_encoder.set_bytes(run_skip_pct, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gather_qmm_rhs(
    const array& x_,
    const array& w_,
    const array& scales_,
    const std::optional<array>& biases_,
    const array& indices_,
    array& out,
    bool transpose,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string mode) {
  if (metal::is_nax_available() && transpose &&
      (env::enable_tf32() || x_.dtype() != float32)) {
    return gather_qmm_rhs_nax(
  }

  // Start by normalizing the indices
  array indices = ensure_row_contiguous(indices_, d, s);

  // Broadcast x with indices. If we are here that means lhs_indices were not
  auto broadcast_with_indices = [&d, &s, &indices](const array& x) {
    if (x.size() / x.shape(-2) / x.shape(-1) == indices.size()) {
      return ensure_row_contiguous(x, d, s);
    }

    auto x_shape = indices.shape();
    x_shape.push_back(x.shape(-2));
    x_shape.push_back(x.shape(-1));
    array new_x(std::move(x_shape), x.dtype(), nullptr, {});
    broadcast(x, new_x);
    return ensure_row_contiguous(new_x, d, s);
  };

  // Normalize the input arrays
  array x = broadcast_with_indices(x_);
  array w = ensure_row_contiguous(w_, d, s);
  array scales = ensure_row_contiguous(scales_, d, s);

  // TODO: Tune the block sizes
  int bm = 16, bn = 32, bk = 32;
  int wm = 1, wn = 2;

  const bool align_M = (M % bm) == 0;
  const bool align_N = (N % bn) == 0;
  const bool align_K = (K % bk) == 0;

  // Make the kernel name
  std::string kname;
  kname.reserve(64);
  std::string type_string = get_type_string(x.dtype());
  concatenate(
      kname,
      mode + (transpose ? "_gather_qmm_rhs_nt_" : "_gather_qmm_rhs_nn_"),
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits,
      "_bm_",
      bm,
      "_bn_",
      bn,
      "_bk_",
      bk,
      "_wm_",
      wm,
      "_wn_",
      wn);

  metal::MTLFCList func_consts = {
      {&align_M, MTL::DataType::DataTypeBool, 200},
      {&align_N, MTL::DataType::DataTypeBool, 201},
      {&align_K, MTL::DataType::DataTypeBool, 202},
  };

  // And the kernel hash that includes the function constants
  std::string hash_name;
  hash_name.reserve(128);
  concatenate(
      hash_name,
      kname,
      "_align_M_",
      align_M ? 't' : 'n',
      "_align_N_",
      align_N ? 't' : 'n',
      "_align_K_",
      align_K ? 't' : 'n');

  // Get and set the kernel
  auto& compute_encoder = metal::get_command_encoder(s);
  auto kernel = get_gather_qmm_kernel(
      d,
      kname,
      hash_name,
      func_consts,
      x,
      group_size,
      bits,
      mode,
      bm,
      bn,
      bk,
      wm,
      wn,
      transpose);
  compute_encoder.set_compute_pipeline_state(kernel);

  MTL::Size group_dims(32, wn, wm);
  MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, 1);

  int c = 0;
  compute_encoder.set_input_array(x, c++);
  compute_encoder.set_input_array(w, c++);
  compute_encoder.set_input_array(scales, c++);
  if (biases_) {
    array biases = ensure_row_contiguous(*biases_, d, s);
    compute_encoder.set_input_array(biases, c++);
  }
  compute_encoder.set_input_array(indices, c++);
  compute_encoder.set_output_array(out, c++);
  compute_encoder.set_bytes(M, c++);
  compute_encoder.set_bytes(N, c++);
  compute_encoder.set_bytes(K, c++);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void dispatch_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int group_size,
    int bits,
    int M,
    int N,
    int K,
    metal::Device& d,
    const Stream& s,
    const std::string& mode) {
  // It is a qmv with a small inner dimension so route to qmv_quad kernel
  if ((K == 128 || K == 64) && is_power_of_2(bits)) {
    qmv_quad(x, w, scales, biases, out, group_size, bits, M, N, K, d, s, mode);
    return;
  }
  qmv(x, w, scales, biases, out, group_size, bits, M, N, K, d, s, mode);
}

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  array x = ensure_row_contiguous_matrix(inputs[0], d, s);
  array w = ensure_row_contiguous_matrix(inputs[1], d, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], d, s);
  std::optional<array> biases = std::nullopt;
  if (inputs.size() == 4) {
    biases = ensure_row_contiguous_matrix(inputs[3], d, s);
  }

  // Extract the matmul shapes
  bool non_batched = w.ndim() == 2 && x.flags().row_contiguous;
  int K = x.shape(-1);
  int M = non_batched ? x.size() / K : x.shape(-2);
  int N = out.shape(-1);

  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
  auto mode = quantization_mode_to_string(mode_);
  // It is a matrix matrix product.
  if (M >= vector_limit) {
    // Use split-K qmm for small M with transposed weights (non-batched only)
    int B = out.size() / M / N;
    if (transpose_ && B == 1) {
      qmm_splitk(
          x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
      return;
    }
    qmm(x,
        w,
        scales,
        biases,
        out,
        transpose_,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  // Run of the mill qmv
  if (transpose_) {
    dispatch_qmv(
        x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
    return;
  }

  // Run of the mill qvm
  if (K < 1024) {
    qvm(x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
    return;
  }

  // Qvm with large dimension so route to a split K kernel for more parallelism
  qvm_split_k(
      x, w, scales, biases, out, group_size_, bits_, M, N, K, d, s, mode);
  return;
}

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  array x = ensure_row_contiguous_matrix(inputs[0], d, s);
  array w = ensure_row_contiguous_matrix(inputs[1], d, s);
  array scales = ensure_row_contiguous_matrix(inputs[2], d, s);
  std::optional<array> biases = std::nullopt;
  if (inputs.size() == 6) {
    biases = ensure_row_contiguous_matrix(inputs[3], d, s);
  }
  const array& lhs_indices = inputs[inputs.size() - 2];
  const array& rhs_indices = inputs[inputs.size() - 1];

  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);
  int B = out.size() / M / N;
  int E = w.size() / w.shape(-1) / w.shape(-2);
  int vector_limit = transpose_ ? get_qmv_batch_limit(K, N, d) : 4;
  auto mode = quantization_mode_to_string(mode_);

  // We are walking x in order and w is also in order so we can batch up the
  if (M == 1 && B >= 16 && right_sorted_ == true && B / E >= 4) {
    gather_qmm_rhs(
        x,
        w,
        scales,
        biases,
        rhs_indices,
        out,
        transpose_,
        group_size_,
        bits_,
        x.size() / K,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  // It is a matrix matrix product
  if (M >= vector_limit) {
    gather_qmm(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        out,
        transpose_,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  if (transpose_) {
    gather_qmv(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        out,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  }

  gather_qvm(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      out,
      group_size_,
      bits_,
      M,
      N,
      K,
      d,
      s,
      mode);
}

void quantize_dequantize(
    const array& in,
    array& out,
    std::string mode,
    int group_size,
    int bits,
    metal::Device& d,
    const Stream& s) {
  auto& compute_encoder = metal::get_command_encoder(s);

  auto w = ensure_row_contiguous(in, d, s);
  compute_encoder.set_input_array(w, 0);
  compute_encoder.set_output_array(out, 1);
  auto type_string = get_type_string(in.dtype());
  std::string kname;
  concatenate(
      kname,
      mode + "_quantize_dequantize_",
      type_string,
      "_gs_",
      group_size,
      "_b_",
      bits);
  auto kernel = get_quantized_kernel_wrapped(
      d, kname, "quantize_dequantize", mode, type_string, group_size, bits);

  compute_encoder.set_compute_pipeline_state(kernel);

  constexpr int uint8_per_uint32 = 4;
  constexpr int simd_size = 32;
  int packs_per_int = (bits == 3 || bits == 5) ? 8 : bits == 6 ? 4 : 8 / bits;
  int per_thread = std::max(group_size / simd_size, 1);
  size_t nthreads = w.size() / per_thread;

  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  auto group_dims = MTL::Size(thread_group_size, 1, 1);
  bool use_2d = nthreads > UINT_MAX;
  auto grid_shape = w.shape();
  grid_shape.back() /= per_thread;
  MTL::Size grid_dims = use_2d ? get_2d_grid_dims(grid_shape, w.strides())
                               : MTL::Size(nthreads, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto mode = quantization_mode_to_string(mode_);
  bool w_quantized = (inputs[1].dtype() == uint32);
  // Tensor-scale nvfp4 (global_scale_x / global_scale_w) is packed into
  int base_size = w_quantized ? 3 : 2;
  if (mode_ == QuantizationMode::Nvfp4 &&
      static_cast<int>(inputs.size()) > base_size) {
    throw std::runtime_error(
        "[QQMatmul] Global scale (tensor-scale nvfp4) is not supported "
        "on the Metal backend.");
  }
  if (w_quantized && inputs[0].shape(-2) == 1) {
    out.set_data(allocator::malloc(out.nbytes()));

    bool donate_x = inputs[0].is_donatable();
    array x = ensure_row_contiguous(inputs[0], d, s);
    // If x is a copy it should be donatable
    donate_x |= x.is_donatable();
    auto xhat = donate_x
        ? x
        : array(allocator::malloc(x.nbytes()), x.shape(), x.dtype());
    quantize_dequantize(x, xhat, mode, group_size_, bits_, d, s);

    // Make sure the last two dims of w and s are contiguous
    array w = ensure_row_contiguous_matrix(inputs[1], d, s);
    array scales = ensure_row_contiguous_matrix(inputs[2], d, s);

    bool non_batched = w.ndim() == 2;
    int K = x.shape(-1);
    int M = non_batched ? x.size() / K : x.shape(-2);
    int N = out.shape(-1);
    dispatch_qmv(
        xhat,
        w,
        scales,
        std::nullopt,
        out,
        group_size_,
        bits_,
        M,
        N,
        K,
        d,
        s,
        mode);
    return;
  } else {
    throw std::runtime_error("[QQMatmul] NYI for the general case");
  }
}

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& w_pre = inputs[0];
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  auto w = ensure_row_contiguous(w_pre, d, s);
  if (dequantize_) {
    auto scales = ensure_row_contiguous(inputs[1], d, s);
    if (mode_ == QuantizationMode::Affine) {
      auto biases = ensure_row_contiguous(inputs[2], d, s);
      compute_encoder.set_input_array(biases, 2);
    }
    compute_encoder.set_input_array(w, 0);
    compute_encoder.set_input_array(scales, 1);
    compute_encoder.set_output_array(out, 3);
  } else {
    auto& scales = outputs[1];
    scales.set_data(allocator::malloc(scales.nbytes()));
    if (mode_ == QuantizationMode::Affine) {
      auto& biases = outputs[2];
      biases.set_data(allocator::malloc(biases.nbytes()));
      compute_encoder.set_output_array(biases, 3);
    }
    compute_encoder.set_input_array(w, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_output_array(scales, 2);
  }

  auto type_string = dequantize_ ? get_type_string(out.dtype())
                                 : get_type_string(w_pre.dtype());
  auto mode = quantization_mode_to_string(mode_);
  std::string kname;
  concatenate(
      kname,
      mode + (dequantize_ ? "_dequantize" : "_quantize"),
      "_",
      type_string,
      "_gs_",
      group_size_,
      "_b_",
      bits_);
  auto kernel = get_quantized_kernel_wrapped(
      d,
      kname,
      dequantize_ ? "dequantize" : "quantize",
      mode,
      type_string,
      group_size_,
      bits_);

  compute_encoder.set_compute_pipeline_state(kernel);

  // Treat uint32 as uint8 in kernel
  constexpr int uint8_per_uint32 = 4;
  constexpr int simd_size = 32;
  int packs_per_int = (bits_ == 3 || bits_ == 5) ? 8
      : bits_ == 6                               ? 4
                                                 : 8 / bits_;
  int per_thread =
      dequantize_ ? packs_per_int : std::max(group_size_ / simd_size, 1);
  size_t nthreads =
      dequantize_ ? out.size() / packs_per_int : w.size() / per_thread;

  NS::UInteger thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  if (thread_group_size > nthreads) {
    thread_group_size = nthreads;
  }
  auto group_dims = MTL::Size(thread_group_size, 1, 1);
  bool use_2d = nthreads > UINT_MAX;
  auto grid_shape = w.shape();
  if (dequantize_) {
    grid_shape.back() *= uint8_per_uint32;
  } else {
    grid_shape.back() /= per_thread;
  }
  MTL::Size grid_dims = use_2d ? get_2d_grid_dims(grid_shape, w.strides())
                               : MTL::Size(nthreads, 1, 1);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

void fast::ConvertFP8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& in = inputs[0];
  auto& out = outputs[0];
  unary_op_gpu(inputs, out, name(), stream());
}

} // namespace mlx::core
