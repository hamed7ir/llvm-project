//===- LoadStoreVec.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A pass that vectorizes short store-load chains.
// Unlike generic bundle vectorization, this pass can vectorize instructions
// of different types.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_LOADSTOREVEC_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_LOADSTOREVEC_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/SandboxIR/Pass.h"

namespace llvm {

class DataLayout;

namespace sandboxir {

class Context;
class Value;
class Instruction;
class Scheduler;
class Type;

class LLVM_ABI LoadStoreVec final : public RegionPass {
  const DataLayout *DL = nullptr;
  /// Checks legality of vectorization and \returns the vector type on success,
  /// nullopt otherwise.
  std::optional<Type *> canVectorize(ArrayRef<Instruction *> Bndl,
                                     Scheduler &Sched);

  void tryEraseDeadInstrs(ArrayRef<Instruction *> Stores,
                          ArrayRef<Value *> Operands);

  /// Vectorizes the store chain \p Bndl by packing its stored values into a
  /// single vector value and storing that instead. Direction-agnostic: it
  /// doesn't matter whether a stored value is a load, a constant, or an
  /// arbitrary SSA value, and mixing kinds within one chain is fine. \Returns
  /// whether it succeeded.
  bool vectorizeStores(ArrayRef<Instruction *> Bndl, Region &Rgn,
                       Scheduler &Sched, const Analyses &A);

  /// \Returns true if \p Run is a consecutive store chain that \p Sched can
  /// schedule together. packOperands() can turn any operand kind into a
  /// vector value, so this is purely an address/scheduling legality check,
  /// not an eligibility one.
  bool isLegalStoreRun(ArrayRef<Instruction *> Run, Scheduler &Sched,
                       const Analyses &A);

  /// \Returns the longest run of stores in \p Bndl starting at \p Start that
  /// isLegalStoreRun() accepts, or an empty range if no run of length >= 2
  /// qualifies.
  /// NOTE: This only ever shrinks the candidate from the high end. A
  /// successful multi-instruction Scheduler::trySchedule() call permanently
  /// commits that exact set of instructions as one scheduled bundle, so
  /// growing a bundle that already succeeded at a shorter length is not
  /// something the scheduler supports.
  ArrayRef<Instruction *> findLegalStoreRun(ArrayRef<Instruction *> Bndl,
                                            unsigned Start, Scheduler &Sched,
                                            const Analyses &A);

public:
  LoadStoreVec(StringRef AuxArg) : RegionPass("load-store-vec") {
    assert(AuxArg.empty() && "This pass ignores aux arg!");
  }
  bool runOnRegion(Region &Rgn, const Analyses &A) final;
};

} // namespace sandboxir

} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_STRUCTINITVEC_H
