//===- Utils.h -  Utilities to support the Tensor dialect -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TENSOR_UTILS_UTILS_H_
#define MLIR_DIALECT_TENSOR_UTILS_UTILS_H_

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"

namespace mlir {
namespace tensor {

// Return a PadOp that pads `source` to `type` size where the static
// sizes are assumed to be greater than the dynamic sizes. If `type` has dynamic
// dimensions the padding width is set to zero. The op performs "high" padding
// (i.e. it adds trailing padding values until the desired size is met).
PadOp createPadHighOp(RankedTensorType type, Value source, Value pad,
                      bool nofold, Location loc, OpBuilder &builder);

// Creates dim ops for each dynamic dimension of the ranked tensor argument and
// returns these as values.
SmallVector<Value> createDynamicDimValues(OpBuilder &b, Location loc,
                                          Value rankedTensor);

/// Returns the transposed `rankedTensorType` if `transposeVector` is non-empty.
/// Fail if `transposeVector` is not a permutation matching the tensor rank.
FailureOr<RankedTensorType>
computeTransposedType(RankedTensorType rankedTensorType,
                      ArrayRef<int64_t> transposeVector);

/// Shell function to compute the Destination Permutation of PackOp
/// This function uses the helper function `computePackUnPackPerm` to get
/// the permutation vector. Only major difference between UnPack and Pack is
/// that packOp uses destination rank whereas unpack Uses source rank.
SmallVector<int64_t> getPackInverseDestPerm(tensor::PackOp packOp);

/// Shell function to compute the Source Permutation of unPackOp.
/// This function, like the getPackInverseDestPerm uses the helper function
/// computePackUnPackPerm` to get the permutation vector.
/// Only major difference between UnPack and Pack is that packOp uses
/// destination rank whereas unpack Uses source rank.
SmallVector<int64_t> getUnPackInverseSrcPerm(tensor::UnPackOp unpackOp);

/// Shell function to compute the Source rank permutation for unpackOp
/// Unpack requires some packing metadata data information, so created
/// another function where this value is passed by reference.
SmallVector<int64_t> getUnPackInverseSrcPerm(tensor::UnPackOp,
                                             PackingMetadata &metadata);

/// A tensor.insert_slice is a cast-like operation if it merely rank-extends the
/// source tensor or inserts the source tensor into a destination tensor with
/// the same shape.
bool isCastLikeInsertSliceOp(InsertSliceOp op);

/// A tensor.extract_slice is a cast-like operation if it merely rank-reduces
/// unit dimensions of the source tensor or extracts the entire source tensor.
bool isCastLikeExtractSliceOp(ExtractSliceOp op);

class TensorDimTrackingRewriter : public IRRewriter,
                                  public IRRewriter::Listener {
public:
  /// Create a new rewriter: Scan the given op for tensor::DimOps.
  TensorDimTrackingRewriter(Operation *op);
  /// Return all tracked tensor::DimOps.
  SmallVector<tensor::DimOp> getTensorDimOps() const;
  /// Return the result of a `tensor.dim` ops that computes has has the same
  /// `source` and `dim`.
  FailureOr<tensor::DimOp> queryCachedDimOps(Value source, int64_t dim) const;

protected:
  void notifyOperationErased(Operation *op) override;
  void notifyOperationInserted(Operation *op, InsertPoint previous) override;

private:
  llvm::SmallDenseSet<tensor::DimOp, 16> dimOps;
  llvm::DenseMap<Value, llvm::SmallDenseMap<uint64_t, tensor::DimOp>> cachedOps;
};

} // namespace tensor
} // namespace mlir

#endif // MLIR_DIALECT_TENSOR_UTILS_UTILS_H_
