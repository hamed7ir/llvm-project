; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Exercise NonSemantic DebugNoLine emission.
;
; DebugNoLine (opcode 104) has no operands. Per the NonSemantic.Shader.DebugInfo
; spec it "discontinues any source-level line and column information specified by
; any previous DebugLine instruction" and must appear within a block.
;
; The SPIRV-LLVM-Translator forward path (LLVMToSPIRVDbgTran::transLocationInfo)
; does not itself emit DebugNoLine; it only consumes it on read
; (SPIRVFunction.cpp resets the current line when it sees DebugNoLine). The
; translator's analogous forward behaviour for a *missing* location is to emit
; DebugNoScope. This test pins the symmetric backend strategy: when an
; instruction in the middle of a block has no DebugLoc while a DebugLine region
; is active, emit DebugNoLine to end that region.
;
; Like DebugLine, DebugNoLine is an in-block instruction placed inline before the
; semantic instruction it applies to. The whole-function CHECK block below shows
; the semantic instructions (OpIAdd / OpIMul / OpReturnValue) interleaved with
; the non-semantic ones (DebugFunctionDefinition, DebugLine, DebugNoLine):
;
;   %t0 = add ...     !dbg line 3   -> DebugLine(3)
;   %t1 = mul ...     (no !dbg)     -> DebugNoLine (region ended)
;   ret  ...          !dbg line 5   -> DebugLine(5)

; --- Module-scope prerequisites (order-independent). ---
; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}debug-no-line.c"
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction {{.*}}
; Each value is emitted once and reused (dedup); V3 is both line 3 and the
; column-start of line 5. Trailing {{$}} anchors avoid "10" matching "100".
; CHECK-DAG: [[V3:%[0-9]+]] = OpConstant [[I32]] 3{{$}}
; CHECK-DAG: [[V5:%[0-9]+]] = OpConstant [[I32]] 5{{$}}
; CHECK-DAG: [[V10:%[0-9]+]] = OpConstant [[I32]] 10{{$}}
; CHECK-DAG: [[V11:%[0-9]+]] = OpConstant [[I32]] 11{{$}}
; CHECK-DAG: [[V4:%[0-9]+]] = OpConstant [[I32]] 4{{$}}

; --- The whole function body, in order. ---
; CHECK:      [[FN:%[0-9]+]] = OpFunction
; CHECK-NEXT: [[A:%[0-9]+]] = OpFunctionParameter
; CHECK-NEXT: [[B:%[0-9]+]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF]] [[FN]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V3]] [[V3]] [[V10]] [[V11]]
; CHECK-NEXT: [[T0:%[0-9]+]] = OpIAdd [[I32]] [[A]] [[B]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugNoLine
; CHECK-NEXT: [[T1:%[0-9]+]] = OpIMul [[I32]] [[T0]] [[A]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V5]] [[V5]] [[V3]] [[V4]]
; CHECK-NEXT: OpReturnValue [[T1]]
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

define spir_func i32 @maybe_line(i32 %a, i32 %b) !dbg !5 {
entry:
  %t0 = add i32 %a, %b, !dbg !8   ; line 3, col 10
  %t1 = mul i32 %t0, %a           ; no debug location
  ret i32 %t1, !dbg !10           ; line 5, col 3
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-no-line.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "maybe_line", linkageName: "maybe_line", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocation(line: 3, column: 10, scope: !5)
!10 = !DILocation(line: 5, column: 3, scope: !5)
