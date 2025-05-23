; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_20 -verify-machineinstrs | FileCheck %s
; RUN: %if ptxas %{ llc < %s -mtriple=nvptx64 -mcpu=sm_20 -verify-machineinstrs | %ptxas-verify %}

target triple = "nvptx64-nvidia-cuda"
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"

define float @fcopysign_f_f(float %a, float %b) {
; CHECK-LABEL: fcopysign_f_f(
; CHECK:       {
; CHECK-NEXT:    .reg .f32 %f<4>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.f32 %f1, [fcopysign_f_f_param_0];
; CHECK-NEXT:    ld.param.f32 %f2, [fcopysign_f_f_param_1];
; CHECK-NEXT:    copysign.f32 %f3, %f2, %f1;
; CHECK-NEXT:    st.param.f32 [func_retval0], %f3;
; CHECK-NEXT:    ret;
  %val = call float @llvm.copysign.f32(float %a, float %b)
  ret float %val
}

define double @fcopysign_d_d(double %a, double %b) {
; CHECK-LABEL: fcopysign_d_d(
; CHECK:       {
; CHECK-NEXT:    .reg .f64 %fd<4>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.f64 %fd1, [fcopysign_d_d_param_0];
; CHECK-NEXT:    ld.param.f64 %fd2, [fcopysign_d_d_param_1];
; CHECK-NEXT:    copysign.f64 %fd3, %fd2, %fd1;
; CHECK-NEXT:    st.param.f64 [func_retval0], %fd3;
; CHECK-NEXT:    ret;
  %val = call double @llvm.copysign.f64(double %a, double %b)
  ret double %val
}

define float @fcopysign_f_d(float %a, double %b) {
; CHECK-LABEL: fcopysign_f_d(
; CHECK:       {
; CHECK-NEXT:    .reg .pred %p<2>;
; CHECK-NEXT:    .reg .f32 %f<5>;
; CHECK-NEXT:    .reg .b64 %rd<4>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.f32 %f1, [fcopysign_f_d_param_0];
; CHECK-NEXT:    abs.f32 %f2, %f1;
; CHECK-NEXT:    neg.f32 %f3, %f2;
; CHECK-NEXT:    ld.param.u64 %rd1, [fcopysign_f_d_param_1];
; CHECK-NEXT:    shr.u64 %rd2, %rd1, 63;
; CHECK-NEXT:    and.b64 %rd3, %rd2, 1;
; CHECK-NEXT:    setp.ne.b64 %p1, %rd3, 0;
; CHECK-NEXT:    selp.f32 %f4, %f3, %f2, %p1;
; CHECK-NEXT:    st.param.f32 [func_retval0], %f4;
; CHECK-NEXT:    ret;
  %c = fptrunc double %b to float
  %val = call float @llvm.copysign.f32(float %a, float %c)
  ret float %val
}

define float @fcopysign_f_h(float %a, half %b) {
; CHECK-LABEL: fcopysign_f_h(
; CHECK:       {
; CHECK-NEXT:    .reg .pred %p<2>;
; CHECK-NEXT:    .reg .b16 %rs<4>;
; CHECK-NEXT:    .reg .f32 %f<5>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.f32 %f1, [fcopysign_f_h_param_0];
; CHECK-NEXT:    abs.f32 %f2, %f1;
; CHECK-NEXT:    neg.f32 %f3, %f2;
; CHECK-NEXT:    ld.param.u16 %rs1, [fcopysign_f_h_param_1];
; CHECK-NEXT:    shr.u16 %rs2, %rs1, 15;
; CHECK-NEXT:    and.b16 %rs3, %rs2, 1;
; CHECK-NEXT:    setp.ne.b16 %p1, %rs3, 0;
; CHECK-NEXT:    selp.f32 %f4, %f3, %f2, %p1;
; CHECK-NEXT:    st.param.f32 [func_retval0], %f4;
; CHECK-NEXT:    ret;
  %c = fpext half %b to float
  %val = call float @llvm.copysign.f32(float %a, float %c)
  ret float %val
}

define double @fcopysign_d_f(double %a, float %b) {
; CHECK-LABEL: fcopysign_d_f(
; CHECK:       {
; CHECK-NEXT:    .reg .pred %p<2>;
; CHECK-NEXT:    .reg .b32 %r<4>;
; CHECK-NEXT:    .reg .f64 %fd<5>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.f64 %fd1, [fcopysign_d_f_param_0];
; CHECK-NEXT:    abs.f64 %fd2, %fd1;
; CHECK-NEXT:    neg.f64 %fd3, %fd2;
; CHECK-NEXT:    ld.param.u32 %r1, [fcopysign_d_f_param_1];
; CHECK-NEXT:    shr.u32 %r2, %r1, 31;
; CHECK-NEXT:    and.b32 %r3, %r2, 1;
; CHECK-NEXT:    setp.ne.b32 %p1, %r3, 0;
; CHECK-NEXT:    selp.f64 %fd4, %fd3, %fd2, %p1;
; CHECK-NEXT:    st.param.f64 [func_retval0], %fd4;
; CHECK-NEXT:    ret;
  %c = fpext float %b to double
  %val = call double @llvm.copysign.f64(double %a, double %c)
  ret double %val
}

define double @fcopysign_d_h(double %a, half %b) {
; CHECK-LABEL: fcopysign_d_h(
; CHECK:       {
; CHECK-NEXT:    .reg .pred %p<2>;
; CHECK-NEXT:    .reg .b16 %rs<4>;
; CHECK-NEXT:    .reg .f64 %fd<5>;
; CHECK-EMPTY:
; CHECK-NEXT:  // %bb.0:
; CHECK-NEXT:    ld.param.f64 %fd1, [fcopysign_d_h_param_0];
; CHECK-NEXT:    abs.f64 %fd2, %fd1;
; CHECK-NEXT:    neg.f64 %fd3, %fd2;
; CHECK-NEXT:    ld.param.u16 %rs1, [fcopysign_d_h_param_1];
; CHECK-NEXT:    shr.u16 %rs2, %rs1, 15;
; CHECK-NEXT:    and.b16 %rs3, %rs2, 1;
; CHECK-NEXT:    setp.ne.b16 %p1, %rs3, 0;
; CHECK-NEXT:    selp.f64 %fd4, %fd3, %fd2, %p1;
; CHECK-NEXT:    st.param.f64 [func_retval0], %fd4;
; CHECK-NEXT:    ret;
  %c = fpext half %b to double
  %val = call double @llvm.copysign.f64(double %a, double %c)
  ret double %val
}


declare float @llvm.copysign.f32(float, float)
declare double @llvm.copysign.f64(double, double)
