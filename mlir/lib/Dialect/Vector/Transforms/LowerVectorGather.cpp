//===- LowerVectorGather.cpp - Lower 'vector.gather' operation ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.gather' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "vector-broadcast-lowering"

using namespace mlir;
using namespace mlir::vector;

namespace {
/// Unrolls 2 or more dimensional `vector.gather` ops by unrolling the
/// outermost dimension. For example:
/// ```
/// %g = vector.gather %base[%c0][%v], %mask, %pass_thru :
///        ... into vector<2x3xf32>
///
/// ==>
///
/// %0   = arith.constant dense<0.0> : vector<2x3xf32>
/// %g0  = vector.gather %base[%c0][%v0], %mask0, %pass_thru0 : ...
/// %1   = vector.insert %g0, %0 [0] : vector<3xf32> into vector<2x3xf32>
/// %g1  = vector.gather %base[%c0][%v1], %mask1, %pass_thru1 : ...
/// %g   = vector.insert %g1, %1 [1] : vector<3xf32> into vector<2x3xf32>
/// ```
///
/// When applied exhaustively, this will produce a sequence of 1-d gather ops.
///
/// Supports vector types with a fixed leading dimension.
struct UnrollGather : OpRewritePattern<vector::GatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::GatherOp op,
                                PatternRewriter &rewriter) const override {
    Value indexVec = op.getIndices();
    Value maskVec = op.getMask();
    Value passThruVec = op.getPassThru();

    auto unrollGatherFn = [&](PatternRewriter &rewriter, Location loc,
                              VectorType subTy, int64_t index) {
      int64_t thisIdx[1] = {index};

      Value indexSubVec =
          vector::ExtractOp::create(rewriter, loc, indexVec, thisIdx);
      Value maskSubVec =
          vector::ExtractOp::create(rewriter, loc, maskVec, thisIdx);
      Value passThruSubVec =
          vector::ExtractOp::create(rewriter, loc, passThruVec, thisIdx);
      return vector::GatherOp::create(rewriter, loc, subTy, op.getBase(),
                                      op.getOffsets(), indexSubVec, maskSubVec,
                                      passThruSubVec, op.getAlignmentAttr());
    };

    return unrollVectorOp(op, rewriter, unrollGatherFn);
  }
};

/// Rewrites a vector.gather of a strided MemRef as a gather of a non-strided
/// MemRef with updated indices that model the strided access.
///
/// Gather indices are flat offsets assuming contiguous (row-major) layout.
/// When the base memref comes from a subview with non-contiguous strides, we
/// decompose the flat index into per-dimension indices, recompose using the
/// physical strides of the subview, and gather from the collapsed (flat)
/// parent memref instead.
///
/// Example (2-D):
/// ```mlir
///   %subview = memref.subview %M (...)
///     : memref<100x3xf32> to memref<100xf32, strided<[3]>>
///   %gather = vector.gather %subview[%idxs] (...)
///     : memref<100xf32, strided<[3]>>
/// ```
/// ==>
/// ```mlir
///   %collapse_shape = memref.collapse_shape %M (...)
///     : memref<100x3xf32> into memref<300xf32>
///   %new_idxs = arith.muli %idxs, %c3 : vector<4xindex>
///   %gather = vector.gather %collapse_shape[%new_idxs] (...)
///     : memref<300xf32> (...)
/// ```
///
/// Example (N-D, e.g. 3-D with strides [15, 3, 1] and offset 9):
/// ```mlir
///   %subview = memref.subview %M[0, 3, 0] [2, 2, 3] [1, 1, 1]
///     : memref<2x5x3xf32>
///     to memref<2x2x3xf32, strided<[15, 3, 1], offset: 9>>
///   %gather = vector.gather %subview[%c0, %c0, %c0] [%flat_idx] (...)
/// ```
/// ==>
/// ```mlir
///   %flat_parent = memref.collapse_shape %M [[0, 1, 2]]
///     : memref<2x5x3xf32> into memref<30xf32>
///   // Decompose flat_idx (contiguous strides [6,3,1]) into per-dim indices,
///   // recompose with physical strides [15,3,1] + offset 9:
///   //   i0 = flat_idx / 6,  i1 = (flat_idx / 3) % 2,  i2 = flat_idx % 3
///   //   new_idx = 9 + i0*15 + i1*3 + i2*1
///   %gather = vector.gather %flat_parent[%c0] [%new_idx] (...)
/// ```
struct RemoveStrideFromGatherSource : OpRewritePattern<vector::GatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::GatherOp op,
                                PatternRewriter &rewriter) const override {
    Value base = op.getBase();

    // TODO: Strided accesses might be coming from other ops as well.
    auto subview = base.getDefiningOp<memref::SubViewOp>();
    if (!subview)
      return failure();

    auto sourceType = subview.getSource().getType();
    auto subviewType = cast<MemRefType>(subview.getResult().getType());

    // Subview must have non-identity layout (otherwise no stride fix needed).
    if (subviewType.getLayout().isIdentity())
      return failure();

    // Source (parent) memref must have identity layout so we can collapse it.
    if (!sourceType.getLayout().isIdentity())
      return rewriter.notifyMatchFailure(
          op, "source memref must have identity layout");

    // Only handle unit-stride subviews (no strided slicing like [::2]).
    for (int64_t stride : subview.getStaticStrides()) {
      if (stride != 1)
        return rewriter.notifyMatchFailure(
            op, "non-unit subview strides not supported");
    }

    // Get physical strides and offset of the subview result.
    SmallVector<int64_t> physStrides;
    int64_t offset;
    if (failed(subviewType.getStridesAndOffset(physStrides, offset)))
      return failure();

    // Require all static values.
    if (ShapedType::isDynamic(offset))
      return rewriter.notifyMatchFailure(op, "dynamic offset not supported");
    for (int64_t s : physStrides)
      if (ShapedType::isDynamic(s))
        return rewriter.notifyMatchFailure(op,
                                           "dynamic strides not supported");

    ArrayRef<int64_t> subviewShape = subviewType.getShape();
    for (int64_t d : subviewShape)
      if (ShapedType::isDynamic(d))
        return rewriter.notifyMatchFailure(op, "dynamic shape not supported");

    Location loc = op.getLoc();
    int64_t rank = subviewType.getRank();
    VectorType vType = op.getIndices().getType();
    Value indices = op.getIndices();

    // Compute logical (contiguous) strides for the subview shape.
    SmallVector<int64_t> logicalStrides(rank);
    logicalStrides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; --i)
      logicalStrides[i] = logicalStrides[i + 1] * subviewShape[i + 1];

    // If strides already match contiguous layout with zero offset, no
    // transformation needed.
    if (logicalStrides == ArrayRef<int64_t>(physStrides) && offset == 0)
      return failure();

    // Helper to create a splat vector constant.
    auto makeConstVec = [&](int64_t val) -> Value {
      return arith::ConstantOp::create(
          rewriter, loc, vType,
          DenseElementsAttr::get(vType, rewriter.getIndexAttr(val)));
    };

    // Decompose flat logical index into per-dimension indices and recompose
    // with physical strides:
    //   new_idx = offset + sum_k( ((flat / logicalStride_k) % dim_k) *
    //                             physStride_k )
    Value newIdx = makeConstVec(offset);

    for (int64_t i = 0; i < rank; ++i) {
      // dimIdx = (flat / logicalStrides[i]) % subviewShape[i]
      Value dimIdx = indices;
      if (logicalStrides[i] != 1)
        dimIdx = arith::DivUIOp::create(rewriter, loc, dimIdx,
                                        makeConstVec(logicalStrides[i]));
      // For the outermost dimension, no mod is needed (it's implicitly bounded
      // by the shape).
      if (i > 0)
        dimIdx = arith::RemUIOp::create(rewriter, loc, dimIdx,
                                        makeConstVec(subviewShape[i]));

      // Accumulate: newIdx += dimIdx * physStride
      if (physStrides[i] == 1) {
        newIdx = arith::AddIOp::create(rewriter, loc, newIdx, dimIdx);
      } else if (physStrides[i] != 0) {
        Value contrib = arith::MulIOp::create(rewriter, loc, dimIdx,
                                              makeConstVec(physStrides[i]));
        newIdx = arith::AddIOp::create(rewriter, loc, newIdx, contrib);
      }
    }

    // Collapse the source (parent) memref to 1-D.
    SmallVector<ReassociationIndices> reassoc;
    ReassociationIndices allDims;
    for (int64_t i = 0; i < sourceType.getRank(); ++i)
      allDims.push_back(i);
    reassoc.push_back(allDims);
    Value collapsed = memref::CollapseShapeOp::create(
        rewriter, loc, subview.getSource(), reassoc);

    // Create updated gather on the flat memref. The offset is baked into
    // newIdx, so the base offset is just [0].
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value newGather = vector::GatherOp::create(
        rewriter, loc, op.getResult().getType(), collapsed,
        SmallVector<Value>{zero}, newIdx, op.getMask(), op.getPassThru(),
        op.getAlignmentAttr());
    rewriter.replaceOp(op, newGather);

    return success();
  }
};

/// Turns 1-d `vector.gather` into a scalarized sequence of `vector.loads` or
/// `tensor.extract`s. To avoid out-of-bounds memory accesses, these
/// loads/extracts are made conditional using `scf.if` ops.
struct Gather1DToConditionalLoads : OpRewritePattern<vector::GatherOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(vector::GatherOp op,
                                PatternRewriter &rewriter) const override {
    VectorType resultTy = op.getType();
    if (resultTy.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "unsupported rank");

    if (resultTy.isScalable())
      return rewriter.notifyMatchFailure(op, "not a fixed-width vector");

    Location loc = op.getLoc();
    Type elemTy = resultTy.getElementType();
    // Vector type with a single element. Used to generate `vector.loads`.
    VectorType elemVecTy = VectorType::get({1}, elemTy);

    Value condMask = op.getMask();
    Value base = op.getBase();

    // Gather indices are computed as flat offsets assuming a contiguous
    // (row-major) layout. If the memref has a non-identity layout (e.g. a
    // strided subview), scalarizing the gather by adding the flat index to
    // the innermost base offset produces incorrect addresses. Bail out and
    // let RemoveStrideFromGatherSource handle the stride adjustment first.
    if (auto memType = dyn_cast<MemRefType>(base.getType())) {
      if (!memType.getLayout().isIdentity()) {
        return rewriter.notifyMatchFailure(
            op, "memref has non-identity layout; gather indices assume "
                "contiguous layout");
      }
    }

    Value indexVec = rewriter.createOrFold<arith::IndexCastOp>(
        loc, op.getIndexVectorType().clone(rewriter.getIndexType()),
        op.getIndices());
    auto baseOffsets = llvm::to_vector(op.getOffsets());
    Value lastBaseOffset = baseOffsets.back();

    Value result = op.getPassThru();
    BoolAttr nontemporalAttr = nullptr;
    IntegerAttr alignmentAttr = op.getAlignmentAttr();

    // Emit a conditional access for each vector element.
    for (int64_t i = 0, e = resultTy.getNumElements(); i < e; ++i) {
      int64_t thisIdx[1] = {i};
      Value condition =
          vector::ExtractOp::create(rewriter, loc, condMask, thisIdx);
      Value index = vector::ExtractOp::create(rewriter, loc, indexVec, thisIdx);
      baseOffsets.back() =
          rewriter.createOrFold<arith::AddIOp>(loc, lastBaseOffset, index);

      auto loadBuilder = [&](OpBuilder &b, Location loc) {
        Value extracted;
        if (isa<MemRefType>(base.getType())) {
          // `vector.load` does not support scalar result; emit a vector load
          // and extract the single result instead.
          Value load =
              vector::LoadOp::create(b, loc, elemVecTy, base, baseOffsets,
                                     nontemporalAttr, alignmentAttr);
          int64_t zeroIdx[1] = {0};
          extracted = vector::ExtractOp::create(b, loc, load, zeroIdx);
        } else {
          extracted = tensor::ExtractOp::create(b, loc, base, baseOffsets);
        }

        Value newResult =
            vector::InsertOp::create(b, loc, extracted, result, thisIdx);
        scf::YieldOp::create(b, loc, newResult);
      };
      auto passThruBuilder = [result](OpBuilder &b, Location loc) {
        scf::YieldOp::create(b, loc, result);
      };

      result = scf::IfOp::create(rewriter, loc, condition,
                                 /*thenBuilder=*/loadBuilder,
                                 /*elseBuilder=*/passThruBuilder)
                   .getResult(0);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

void mlir::vector::populateVectorGatherLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<UnrollGather>(patterns.getContext(), benefit);
}

void mlir::vector::populateVectorGatherToConditionalLoadPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<RemoveStrideFromGatherSource, Gather1DToConditionalLoads>(
      patterns.getContext(), benefit);
}
