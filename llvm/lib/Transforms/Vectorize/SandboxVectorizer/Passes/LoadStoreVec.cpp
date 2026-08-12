//===- LoadStoreVec.cpp - Vectorizer pass short load-store chains ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/Passes/LoadStoreVec.h"
#include "llvm/SandboxIR/Module.h"
#include "llvm/SandboxIR/Region.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InstructionCost.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Debug.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/RegionWithScore.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Scheduler.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"

namespace llvm {

extern cl::opt<int> CostThreshold; // Defined in TransactionAcceptOrRevert.cpp

namespace sandboxir {

#define DEBUG_PREFIX_LOCAL DEBUG_PREFIX "LoadStoreVec: "

std::optional<Type *> LoadStoreVec::canVectorize(ArrayRef<Instruction *> Bndl,
                                                 Scheduler &Sched) {
  // Check if in the same BB.
  if (LegalityAnalysis::differentBlock(Bndl))
    return std::nullopt;

  // Check if instructions repeat.
  if (!LegalityAnalysis::areUnique(Bndl))
    return std::nullopt;

  // Check scheduling.
  if (!Sched.trySchedule(Bndl))
    return std::nullopt;

  return VecUtils::getCombinedVectorTypeFor(Bndl, *DL);
}

void LoadStoreVec::tryEraseDeadInstrs(ArrayRef<Instruction *> Stores,
                                      ArrayRef<Value *> Operands) {
  SmallPtrSet<Instruction *, 8> DeadCandidates;
  for (auto *SI : Stores) {
    if (auto *PtrI =
            dyn_cast<Instruction>(cast<StoreInst>(SI)->getPointerOperand()))
      DeadCandidates.insert(PtrI);
    SI->eraseFromParent();
  }
  for (auto *Op : Operands) {
    // Unlike a load that only ever fed this store, a packed operand (see
    // packOperands()) isn't necessarily single-use, so only erase it once
    // its last use (the store we just erased above) is actually gone.
    auto *LI = dyn_cast<LoadInst>(Op);
    if (LI == nullptr || !LI->hasNUses(0))
      continue;
    if (auto *PtrI = dyn_cast<Instruction>(LI->getPointerOperand()))
      DeadCandidates.insert(PtrI);
    LI->eraseFromParent();
  }
  for (auto *PtrI : DeadCandidates)
    if (!PtrI->hasNUsesOrMore(1))
      PtrI->eraseFromParent();
}

/// \Returns the narrowest scalar element type across \p Operands, mirroring
/// VecUtils::getCombinedVectorTypeFor() but for arbitrary Values rather than
/// only Instructions: a store's operands can be Constants, Arguments, or any
/// other Value, not just Instructions.
static Type *getCombinedElementType(ArrayRef<Value *> Operands,
                                    const DataLayout &DL) {
  Type *MinElmTy = nullptr;
  unsigned MinElmBits = std::numeric_limits<unsigned>::max();
  for (Value *V : Operands) {
    Type *ElmTy = VecUtils::getElementType(Utils::getExpectedType(V));
    unsigned ElmBits = Utils::getNumBits(ElmTy, DL);
    if (ElmBits < MinElmBits) {
      MinElmBits = ElmBits;
      MinElmTy = ElmTy;
    }
  }
  return MinElmTy;
}

/// \Returns \p V (of type \p FromTy) reinterpreted as \p ToTy, which must be
/// the same bit width. Chooses bitcast, ptrtoint, or inttoptr as needed: a
/// plain bitcast can't convert between pointer and non-pointer types, and
/// inttoptr requires an integer source, so a non-integer, non-pointer type
/// (e.g. double) headed for a pointer-typed granularity goes through an
/// intermediate same-width integer bitcast first.
static Value *reinterpretSameWidth(Value *V, Type *FromTy, Type *ToTy,
                                   const DataLayout &DL,
                                   BasicBlock::iterator &WhereIt,
                                   Context &Ctx) {
  auto Track = [&](Value *NewV) {
    if (!isa<Constant>(NewV))
      WhereIt = std::next(cast<Instruction>(NewV)->getIterator());
    return NewV;
  };
  if (FromTy->isPointerTy()) {
    Type *IntTy = IntegerType::get(Ctx, Utils::getNumBits(FromTy, DL));
    V = Track(PtrToIntInst::create(V, IntTy, WhereIt, Ctx, "PackP2I"));
    FromTy = IntTy;
  }
  if (ToTy->isPointerTy()) {
    if (!FromTy->isIntegerTy()) {
      Type *IntTy = IntegerType::get(Ctx, Utils::getNumBits(FromTy, DL));
      V = Track(BitCastInst::create(V, IntTy, WhereIt, Ctx, "PackToInt"));
      FromTy = IntTy;
    }
    return Track(IntToPtrInst::create(V, ToTy, WhereIt, Ctx, "PackI2P"));
  }
  if (FromTy == ToTy)
    return V;
  return Track(BitCastInst::create(V, ToTy, WhereIt, Ctx, "PackCast"));
}

/// Packs \p Operands into a single vector value. Direction-agnostic: it
/// doesn't matter whether an operand is a load, a constant, or an arbitrary
/// SSA value, and mixing kinds is fine -- unlike an extractelement/
/// insertelement pair, this never fails, so there's no legality check here.
///
/// Operands are combined at the granularity of their narrowest common scalar
/// element type (the same granularity getCombinedVectorTypeFor() computes for
/// a mixed-type chain); an operand wider than that granularity is first split
/// into multiple lanes via a bitcast, matching how a vector-typed element is
/// split into per-lane extracts below.
static Value *packOperands(ArrayRef<Value *> Operands, const DataLayout &DL,
                           BasicBlock *BB) {
  Type *ElemTy = getCombinedElementType(Operands, DL);
  unsigned ElemBits = Utils::getNumBits(ElemTy, DL);
  Context &Ctx = Operands[0]->getContext();
  BasicBlock::iterator WhereIt =
      VecUtils::getInsertPointAfterInstrs(Operands, BB);

  unsigned NumLanes = 0;
  for (Value *Op : Operands)
    NumLanes += Utils::getNumBits(Op, DL) / ElemBits;
  Value *LastInsert = PoisonValue::get(VecUtils::getWideType(ElemTy, NumLanes));

  unsigned InsertIdx = 0;
  for (Value *Op : Operands) {
    Value *Elm = Op;
    Type *OpElemTy = VecUtils::getElementType(Utils::getExpectedType(Op));
    // If Op's own granularity is coarser than ElemTy, split it into a vector
    // of ElemTy-sized lanes first, so the logic below can extract them.
    if (OpElemTy != ElemTy) {
      unsigned OpBits = Utils::getNumBits(Op, DL);
      if (OpBits == ElemBits) {
        Elm = reinterpretSameWidth(Elm, OpElemTy, ElemTy, DL, WhereIt, Ctx);
      } else {
        // A pointer can't be bitcast to a narrower vector directly, but this
        // is otherwise the same situation the vector-Elm case below handles:
        // reinterpret at the ElemTy granularity, then extract each lane.
        Type *SplitTy = VecUtils::getWideType(ElemTy, OpBits / ElemBits);
        Elm = reinterpretSameWidth(Elm, OpElemTy, SplitTy, DL, WhereIt, Ctx);
      }
    }
    // An element can be either scalar or vector at this point. We need to
    // generate different IR for each case.
    if (Elm->getType()->isVectorTy()) {
      unsigned NumElms =
          cast<FixedVectorType>(Elm->getType())->getNumElements();
      for (auto ExtrLane : seq<int>(0, NumElms)) {
        // We generate extract-insert pairs, for each lane in `Elm`.
        Constant *ExtrLaneC =
            ConstantInt::getSigned(Type::getInt32Ty(Ctx), ExtrLane);
        // This may return a Constant if Elm is a Constant.
        auto *ExtrI =
            ExtractElementInst::create(Elm, ExtrLaneC, WhereIt, Ctx, "VPack");
        if (!isa<Constant>(ExtrI))
          WhereIt = std::next(cast<Instruction>(ExtrI)->getIterator());
        Constant *InsertLaneC =
            ConstantInt::getSigned(Type::getInt32Ty(Ctx), InsertIdx++);
        // This may also return a Constant if ExtrI is a Constant.
        auto *InsertI = InsertElementInst::create(
            LastInsert, ExtrI, InsertLaneC, WhereIt, Ctx, "VPack");
        LastInsert = InsertI;
        if (!isa<Constant>(InsertI))
          WhereIt = std::next(cast<Instruction>(LastInsert)->getIterator());
      }
    } else {
      Constant *InsertLaneC =
          ConstantInt::getSigned(Type::getInt32Ty(Ctx), InsertIdx++);
      // This may be folded into a Constant if LastInsert is a Constant. In
      // that case we only collect the last constant.
      LastInsert = InsertElementInst::create(LastInsert, Elm, InsertLaneC,
                                             WhereIt, Ctx, "Pack");
      if (auto *NewI = dyn_cast<Instruction>(LastInsert))
        WhereIt = std::next(NewI->getIterator());
    }
  }
  return LastInsert;
}

/// Accepts the transaction if vectorizing was profitable, reverts it otherwise.
/// \Returns true if the transaction was accepted.
static bool acceptIfProfitable(Context &Ctx, const ScoreBoard &SB,
                               InstructionCost CostBefore) {
  InstructionCost CostAfter = SB.getAfterCost() - SB.getBeforeCost();
  InstructionCost CostGain = CostAfter - CostBefore;
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX_LOCAL << "CostGain=" << CostGain
                    << " (After=" << CostAfter << " Before=" << CostBefore
                    << ")\n");
  if (CostGain > CostThreshold) {
    LLVM_DEBUG(dbgs() << DEBUG_PREFIX_LOCAL << "Not profitable, reverting.\n");
    Ctx.revert();
    return false;
  }
  LLVM_DEBUG(dbgs() << DEBUG_PREFIX_LOCAL << "Profitable accepting.\n");
  Ctx.accept();
  return true;
}

bool LoadStoreVec::vectorizeStores(ArrayRef<Instruction *> Bndl, Region &Rgn,
                                   Scheduler &Sched, const Analyses &A) {
  // The seeds must form a consecutive, vectorizable store chain.
  if (!VecUtils::areConsecutive<StoreInst, Instruction>(
          Bndl, A.getScalarEvolution(), *DL))
    return false;
  if (!canVectorize(Bndl, Sched))
    return false;

  Function &F = *Bndl[0]->getParent()->getParent();
  auto &Ctx = F.getContext();

  SmallVector<Value *, 4> Operands;
  Operands.reserve(Bndl.size());
  for (auto *I : Bndl)
    Operands.push_back(cast<StoreInst>(I)->getValueOperand());

  const auto &SB = cast<RegionWithScore>(Rgn).getScoreboard();
  InstructionCost CostBefore = SB.getAfterCost() - SB.getBeforeCost();

  // Vectorizing mixed floats and integers with external uses may not be
  // profitable on some targets, so save state here.
  Ctx.save();

  BasicBlock *BB = Bndl[0]->getParent();
  Value *VecOp = packOperands(Operands, *DL, BB);

  // Generate vector store.
  Value *StPtr = cast<StoreInst>(Bndl[0])->getPointerOperand();
  // TODO: Compute alignment.
  Align StAlign(1);
  auto StWhereIt = std::next(VecUtils::getLowest(Bndl)->getIterator());
  StoreInst::create(VecOp, StPtr, StAlign, StWhereIt, Ctx);

  tryEraseDeadInstrs(Bndl, Operands);

  return acceptIfProfitable(Ctx, SB, CostBefore);
}

bool LoadStoreVec::runOnRegion(Region &Rgn, const Analyses &A) {
  SmallVector<Instruction *, 8> Bndl(Rgn.getAux().begin(), Rgn.getAux().end());
  if (Bndl.size() < 2)
    return false;
  Function &F = *Bndl[0]->getParent()->getParent();
  DL = &F.getParent()->getDataLayout();
  Scheduler Sched(A.getAA(), F.getContext(), SchedDirection::BottomUp);
  return vectorizeStores(Bndl, Rgn, Sched, A);
}

} // namespace sandboxir

} // namespace llvm
