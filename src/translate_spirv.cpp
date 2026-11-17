#include "translate_spirv.hpp"
#include "soft_math.hpp"

#include "weft_msl.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

// SPIR-V -> C++ for the same runtime the Metal and PTX front ends target.
namespace {

enum : uint32_t {
  OpNop = 0, OpUndef = 1, OpSource = 3, OpName = 5, OpExtInstImport = 11, OpExtInst = 12,
  OpMemoryModel = 14, OpEntryPoint = 15, OpExecutionMode = 16, OpCapability = 17,
  OpTypeVoid = 19, OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23,
  OpTypeMatrix = 24, OpTypeImage = 25, OpTypeSampler = 26, OpTypeSampledImage = 27,
  OpTypeArray = 28, OpTypeRuntimeArray = 29, OpTypeStruct = 30, OpTypePointer = 32,
  OpTypeFunction = 33, OpTypeForwardPointer = 39,
  OpConstantTrue = 41, OpConstantFalse = 42, OpConstant = 43, OpConstantComposite = 44,
  OpConstantNull = 46, OpFunction = 54, OpFunctionParameter = 55, OpFunctionEnd = 56,
  OpFunctionCall = 57, OpVariable = 59, OpImageTexelPointer = 60, OpLoad = 61, OpStore = 62,
  OpAccessChain = 65, OpInBoundsAccessChain = 66, OpPtrAccessChain = 67,
  OpArrayLength = 68, OpInBoundsPtrAccessChain = 70,
  OpDecorate = 71, OpMemberDecorate = 72, OpVectorExtractDynamic = 77, OpVectorInsertDynamic = 78,
  OpVectorShuffle = 79, OpCompositeConstruct = 80, OpCompositeExtract = 81,
  OpCompositeInsert = 82, OpCopyObject = 83, OpTranspose = 84, OpSampledImage = 86,
  OpImageSampleImplicitLod = 87, OpImageSampleExplicitLod = 88,
  OpImageFetch = 95, OpImageGather = 96, OpImageRead = 98, OpImageWrite = 99, OpImage = 100,
  OpImageQuerySizeLod = 103, OpImageQuerySize = 104,
  OpConvertFToU = 109, OpConvertFToS = 110, OpConvertSToF = 111, OpConvertUToF = 112,
  OpUConvert = 113, OpSConvert = 114, OpFConvert = 115, OpQuantizeToF16 = 116,
  OpSatConvertSToU = 118, OpSatConvertUToS = 119, OpBitcast = 124, OpSNegate = 126,
  OpFNegate = 127, OpIAdd = 128, OpFAdd = 129, OpISub = 130, OpFSub = 131, OpIMul = 132,
  OpFMul = 133, OpUDiv = 134, OpSDiv = 135, OpFDiv = 136, OpUMod = 137, OpSRem = 138,
  OpSMod = 139, OpFRem = 140, OpFMod = 141, OpVectorTimesScalar = 142,
  OpMatrixTimesScalar = 143, OpVectorTimesMatrix = 144, OpMatrixTimesVector = 145,
  OpMatrixTimesMatrix = 146, OpOuterProduct = 147, OpDot = 148, OpIAddCarry = 149,
  OpISubBorrow = 150, OpUMulExtended = 151, OpSMulExtended = 152,
  OpAny = 154, OpAll = 155, OpIsNan = 156, OpIsInf = 157, OpIsFinite = 158,
  OpIsNormal = 159, OpSignBitSet = 160, OpLessOrGreater = 161, OpOrdered = 162,
  OpUnordered = 163, OpLogicalEqual = 164, OpLogicalNotEqual = 165, OpLogicalOr = 166,
  OpLogicalAnd = 167, OpLogicalNot = 168, OpSelect = 169, OpIEqual = 170,
  OpINotEqual = 171, OpUGreaterThan = 172, OpSGreaterThan = 173, OpUGreaterThanEqual = 174,
  OpSGreaterThanEqual = 175, OpULessThan = 176, OpSLessThan = 177, OpULessThanEqual = 178,
  OpSLessThanEqual = 179, OpFOrdEqual = 180, OpFUnordEqual = 181, OpFOrdNotEqual = 182,
  OpFUnordNotEqual = 183, OpFOrdLessThan = 184, OpFUnordLessThan = 185,
  OpFOrdGreaterThan = 186, OpFUnordGreaterThan = 187, OpFOrdLessThanEqual = 188,
  OpFUnordLessThanEqual = 189, OpFOrdGreaterThanEqual = 190, OpFUnordGreaterThanEqual = 191,
  OpShiftRightLogical = 194, OpShiftRightArithmetic = 195, OpShiftLeftLogical = 196,
  OpBitwiseOr = 197, OpBitwiseXor = 198, OpBitwiseAnd = 199, OpNot = 200,
  OpBitFieldInsert = 201, OpBitFieldSExtract = 202, OpBitFieldUExtract = 203,
  OpBitReverse = 204, OpBitCount = 205, OpControlBarrier = 224, OpMemoryBarrier = 225,
  OpAtomicLoad = 227, OpAtomicStore = 228, OpAtomicExchange = 229,
  OpAtomicCompareExchange = 230, OpAtomicCompareExchangeWeak = 231, OpAtomicIIncrement = 232,
  OpAtomicIDecrement = 233, OpAtomicIAdd = 234, OpAtomicISub = 235, OpAtomicSMin = 236,
  OpAtomicUMin = 237, OpAtomicSMax = 238, OpAtomicUMax = 239, OpAtomicAnd = 240,
  OpAtomicOr = 241, OpAtomicXor = 242, OpPhi = 245, OpLoopMerge = 246, OpSelectionMerge = 247,
  OpLabel = 248, OpBranch = 249, OpBranchConditional = 250, OpSwitch = 251, OpKill = 252,
  OpReturn = 253, OpReturnValue = 254, OpUnreachable = 255,
  // subgroup (SPIR-V 1.3)
  OpGroupNonUniformElect = 333, OpGroupNonUniformAll = 334, OpGroupNonUniformAny = 335,
  OpGroupNonUniformAllEqual = 336, OpGroupNonUniformBroadcast = 337,
  OpGroupNonUniformBroadcastFirst = 338, OpGroupNonUniformBallot = 339,
  OpGroupNonUniformInverseBallot = 340, OpGroupNonUniformBallotBitExtract = 341,
  OpGroupNonUniformBallotBitCount = 342, OpGroupNonUniformBallotFindLSB = 343,
  OpGroupNonUniformBallotFindMSB = 344, OpGroupNonUniformShuffle = 345,
  OpGroupNonUniformShuffleXor = 346, OpGroupNonUniformShuffleUp = 347,
  OpGroupNonUniformShuffleDown = 348, OpGroupNonUniformQuadBroadcast = 365,
  OpGroupNonUniformQuadSwap = 366,
  OpGroupNonUniformIAdd = 349, OpGroupNonUniformFAdd = 350,
  OpGroupNonUniformIMul = 351, OpGroupNonUniformFMul = 352,
  OpGroupNonUniformSMin = 353, OpGroupNonUniformUMin = 354,
  OpGroupNonUniformFMin = 355, OpGroupNonUniformSMax = 356,
  OpGroupNonUniformUMax = 357, OpGroupNonUniformFMax = 358,
  OpGroupNonUniformBitwiseAnd = 359, OpGroupNonUniformBitwiseOr = 360,
  OpGroupNonUniformBitwiseXor = 361, OpGroupNonUniformLogicalAnd = 362,
  OpGroupNonUniformLogicalOr = 363, OpGroupNonUniformLogicalXor = 364,
  OpGroupNonUniformRotateKHR = 4431,
  // atomic float add (SPV_EXT_shader_atomic_float_add)
  OpAtomicFAddEXT = 6035,
  // atomic float min/max (SPV_EXT_shader_atomic_float_min_max).
  OpAtomicFMinEXT = 5614, OpAtomicFMaxEXT = 5615,
};

enum : uint32_t { DArrayStride = 6, DBuiltIn = 11, DOffset = 35, DSpecId = 1, DBinding = 33 };

enum : uint32_t {
  BINumWorkgroups = 24, BIWorkgroupSize = 25, BIWorkgroupId = 26, BILocalInvocationId = 27,
  BIGlobalInvocationId = 28, BILocalInvocationIndex = 29, BIWorkDim = 30, BIGlobalSize = 31,
  BINumEnqueuedWorkgroups = 32, BIGlobalOffset = 33, BINumSubgroups = 38,
  BINumEnqueuedSubgroups = 39, BISubgroupSize = 35, BISubgroupId = 36,
  BISubgroupLocalInvocationId = 40,
};

struct Instr {
  uint32_t op = 0;
  std::vector<uint32_t> a;
};

enum Kind {
  KVoid, KBool, KInt, KFloat, KVector, KMatrix, KPointer, KArray, KStruct, KFunction,
  KImage, KSampler, KSampledImage, KOther
};

struct SType {
  Kind kind = KOther;
  uint32_t width = 0;
  bool is_signed = false;
  uint32_t count = 0;   // vector components
  uint32_t elem = 0;    // vector / array / pointer element type
  uint32_t length_id = 0;
  uint32_t length = 0;  // resolved after all constants are read
  uint32_t storage = 0;
  uint32_t arrayed = 0;  // OpTypeImage: the coordinate carries a layer
  std::vector<uint32_t> members;
};

struct Global {
  uint32_t type = 0;
  uint32_t storage = 0;
  bool has_builtin = false;
  bool initialized = false;  // an OpVariable with an initialiser operand
  uint32_t builtin = 0;
};

struct Entry {
  uint32_t exec_model = 0;
  uint32_t function = 0;
  std::string name;
  std::vector<uint32_t> iface;
  uint32_t lx = 0, ly = 1, lz = 1;
};

struct Function {
  uint32_t id = 0;
  bool defined = false;
  std::vector<Instr> body;
};

struct Mod {
  std::map<uint32_t, SType> types;
  std::map<uint32_t, std::string> cexpr;  // scalar constants
  std::map<uint32_t, uint32_t> ctype;     // constant id -> type id
  std::map<uint32_t, std::string> names;
  std::map<uint32_t, Global> globals;
  std::map<uint32_t, uint32_t> array_stride;
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> member_offset;
  std::map<uint32_t, Function> funcs;
  std::map<uint32_t, uint32_t> ext_set;  // result id -> set id
  std::map<uint32_t, std::string> ext_name;
  std::vector<Entry> entries;
};

// Instructions that write no value: everything else starts with (type, result).
bool no_result(uint32_t op) {
  switch (op) {
    case OpStore: case OpBranch: case OpBranchConditional: case OpSwitch: case OpReturn:
    case OpReturnValue: case OpKill: case OpUnreachable: case OpControlBarrier:
    case OpMemoryBarrier: case OpLoopMerge: case OpSelectionMerge: case OpLabel: case OpNop:
    case OpFunctionEnd: case OpFunctionParameter: case OpVariable: case OpDecorate:
    case OpMemberDecorate: case OpAtomicStore: case OpImageWrite: case OpTypeForwardPointer:
    case OpExtInstImport:
      return true;
    default:
      return false;
  }
}

std::string literal_at(const std::vector<uint32_t>& a, size_t off) {
  std::string s;
  for (size_t i = off; i < a.size(); i++)
    for (int b = 0; b < 4; b++) {
      const char c = (char)((a[i] >> (8 * b)) & 0xFF);
      if (c == 0) return s;
      s.push_back(c);
    }
  return s;
}

size_t literal_words(const std::string& s) { return (s.size() + 1 + 3) / 4; }

bool parse_module(Mod& m, const std::string& bytes, std::string& err) {
  if (bytes.size() < 20 || (bytes.size() % 4) != 0) {
    err = "not a SPIR-V module (" + std::to_string(bytes.size()) + " bytes)";
    return false;
  }
  std::vector<uint32_t> w(bytes.size() / 4);
  std::memcpy(w.data(), bytes.data(), bytes.size());
  if (w[0] != 0x07230203u) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "not a SPIR-V module (magic 0x%08x)", w[0]);
    err = buf;
    return false;
  }

  size_t i = 5;
  uint32_t cur = 0;
  while (i < w.size()) {
    const uint32_t head = w[i];
    const uint32_t wc = head >> 16, op = head & 0xFFFFu;
    if (wc == 0 || i + wc > w.size()) { err = "truncated SPIR-V instruction stream"; return false; }
    Instr ins;
    ins.op = op;
    ins.a.assign(w.begin() + i + 1, w.begin() + i + wc);
    i += wc;
    if (ins.a.empty() && op != OpNop && op != OpReturn && op != OpFunctionEnd &&
        op != OpUnreachable && op != OpKill) {
      err = "empty SPIR-V instruction (opcode " + std::to_string(op) + ")";
      return false;
    }

    switch (op) {
      case OpTypeVoid: case OpTypeBool: {
        SType t;
        t.kind = op == OpTypeVoid ? KVoid : KBool;
        m.types[ins.a[0]] = t;
        break;
      }
      case OpTypeInt:
        if (ins.a.size() < 3) { err = "bad OpTypeInt"; return false; }
        m.types[ins.a[0]] = SType{.kind = KInt, .width = ins.a[1], .is_signed = ins.a[2] != 0};
        break;
      case OpTypeFloat:
        if (ins.a.size() < 2) { err = "bad OpTypeFloat"; return false; }
        m.types[ins.a[0]] = SType{.kind = KFloat, .width = ins.a[1], .is_signed = false};
        break;
      case OpTypeVector: {
        if (ins.a.size() < 3) { err = "bad OpTypeVector"; return false; }
        SType t;
        t.kind = KVector;
        t.elem = ins.a[1];
        t.count = ins.a[2];
        m.types[ins.a[0]] = t;
        break;
      }
      case OpTypeArray: {
        if (ins.a.size() < 3) { err = "bad OpTypeArray"; return false; }
        SType t;
        t.kind = KArray;
        t.elem = ins.a[1];
        t.length_id = ins.a[2];
        m.types[ins.a[0]] = t;
        break;
      }
      case OpTypeRuntimeArray: {
        SType t;
        t.kind = KArray;
        t.elem = ins.a[1];
        t.length = 0;  // open: the size comes from the buffer, never from here
        m.types[ins.a[0]] = t;
        break;
      }
      case OpTypeStruct: {
        SType t;
        t.kind = KStruct;
        t.members.assign(ins.a.begin() + 1, ins.a.end());
        m.types[ins.a[0]] = t;
        break;
      }
      case OpTypePointer: {
        if (ins.a.size() < 3) { err = "bad OpTypePointer"; return false; }
        SType t;
        t.kind = KPointer;
        t.storage = ins.a[1];
        t.elem = ins.a[2];
        m.types[ins.a[0]] = t;
        break;
      }
      case OpTypeFunction: {
        SType t;
        t.kind = KFunction;
        m.types[ins.a[0]] = t;
        break;
      }
      // An image is a handle.
      case OpTypeImage: {
        if (ins.a.size() < 8) { err = "bad OpTypeImage"; return false; }
        SType t;
        t.kind = KImage;
        t.elem = ins.a[1];   // sampled type
        t.count = ins.a[2];  // Dim
        t.storage = ins.a[6];  // Sampled: 0 unknown, 1 sampled, 2 storage
        t.arrayed = ins.a[4];  // Arrayed: a 2-D array, so a layer per coordinate
        m.types[ins.a[0]] = t;
        break;
      }
      case OpTypeSampler:
        m.types[ins.a[0]] = SType{.kind = KSampler, .width = 0, .is_signed = false};
        break;
      case OpTypeSampledImage: {
        SType t;
        t.kind = KSampledImage;
        t.elem = ins.a[1];
        m.types[ins.a[0]] = t;
        break;
      }
      case OpTypeMatrix: {
        if (ins.a.size() < 3) { err = "bad OpTypeMatrix"; return false; }
        auto col = m.types.find(ins.a[1]);
        if (col == m.types.end() || col->second.kind != KVector) {
          err = "a matrix whose column is not a vector";
          return false;
        }
        const uint32_t rows = col->second.count, cols = ins.a[2];
        if (cols < 2 || cols > 4) {
          err = "a " + std::to_string(cols) + "-column matrix: 2, 3 or 4 columns";
          return false;
        }
        if (rows < 2 || rows > 4) {
          err = "a " + std::to_string(rows) + "-by-" + std::to_string(rows) + " matrix";
          return false;
        }
        auto e = m.types.find(col->second.elem);
        if (e == m.types.end() || e->second.kind != KFloat || e->second.width != 32) {
          err = "a matrix of something other than 32-bit floats";
          return false;
        }
        SType t;
        t.kind = KMatrix;
        t.elem = ins.a[1];
        t.count = cols;
        m.types[ins.a[0]] = t;
        break;
      }
      case OpConstantTrue: case OpConstantFalse:
        m.cexpr[ins.a[1]] = op == OpConstantTrue ? "true" : "false";
        m.ctype[ins.a[1]] = ins.a[0];
        break;
      case OpConstantNull: {
        m.ctype[ins.a[1]] = ins.a[0];
        m.cexpr[ins.a[1]] = "0";
        break;
      }
      case OpConstant: {
        if (ins.a.size() < 3) { err = "bad OpConstant"; return false; }
        const uint32_t tid = ins.a[0], id = ins.a[1];
        m.ctype[id] = tid;
        auto it = m.types.find(tid);
        if (it == m.types.end()) { err = "constant of an unknown type"; return false; }
        const SType& t = it->second;
        char buf[96];
        if (t.kind == KFloat && t.width == 32) {
          std::snprintf(buf, sizeof buf, "weft_f32(0x%08xu)", ins.a[2]);
        } else if (t.kind == KFloat && t.width == 64) {
          const uint64_t bits = ((uint64_t)ins.a[3] << 32) | ins.a[2];
          std::snprintf(buf, sizeof buf, "weft_f64(0x%016llxull)", (unsigned long long)bits);
        } else if (t.kind == KFloat && t.width == 16) {
          // A 16-bit float constant.
          std::snprintf(buf, sizeof buf, "weft_h16v(0x%04xu)", ins.a[2] & 0xFFFFu);
        } else if (t.kind == KInt && t.width == 64) {
          const uint64_t bits = ((uint64_t)ins.a[3] << 32) | ins.a[2];
          std::snprintf(buf, sizeof buf, "0x%016llxull", (unsigned long long)bits);
        } else if (t.kind == KInt && t.width == 32) {
          std::snprintf(buf, sizeof buf, "0x%08xu", ins.a[2]);
        } else if (t.kind == KInt && t.width == 16) {
          std::snprintf(buf, sizeof buf, "%uu", ins.a[2] & 0xFFFFu);
        } else if (t.kind == KInt && t.width == 8) {
          std::snprintf(buf, sizeof buf, "%uu", ins.a[2] & 0xFFu);
        } else if (t.kind == KInt && t.width == 1) {
          std::snprintf(buf, sizeof buf, "true");
        } else {
          err = "constant of an unsupported type (kind " + std::to_string((int)t.kind) + ")";
          return false;
        }
        m.cexpr[id] = buf;
        break;
      }
      case OpConstantComposite: {
        m.ctype[ins.a[1]] = ins.a[0];
        std::string joined;
        for (size_t k = 2; k < ins.a.size(); k++) {
          auto c = m.cexpr.find(ins.a[k]);
          if (c == m.cexpr.end()) { m.cexpr.erase(ins.a[1]); break; }  // resolved later or refused
          if (k > 2) joined += ", ";
          joined += c->second;
        }
        if (!joined.empty() || ins.a.size() == 2) m.cexpr[ins.a[1]] = "@C" + joined;
        break;
      }
      case OpName:
        if (ins.a.size() >= 2) m.names[ins.a[0]] = literal_at(ins.a, 1);
        break;
      case OpDecorate:
        if (ins.a.size() >= 2) {
          if (ins.a[1] == DBuiltIn && ins.a.size() >= 3) {
            m.globals[ins.a[0]].has_builtin = true;
            m.globals[ins.a[0]].builtin = ins.a[2];
          } else if (ins.a[1] == DArrayStride && ins.a.size() >= 3) {
            m.array_stride[ins.a[0]] = ins.a[2];
          }
        }
        break;
      case OpMemberDecorate:
        if (ins.a.size() >= 4 && ins.a[2] == DOffset)
          m.member_offset[{ins.a[0], ins.a[1]}] = ins.a[3];
        break;
      case OpVariable:
        if (cur == 0) {
          Global g;
          g.type = ins.a[0];
          g.storage = ins.a[2];
          g.initialized = ins.a.size() > 3;
          auto prev = m.globals.find(ins.a[1]);
          if (prev != m.globals.end()) {
            g.has_builtin = prev->second.has_builtin;
            g.builtin = prev->second.builtin;
            g.initialized = g.initialized || prev->second.initialized;
          }
          m.globals[ins.a[1]] = g;
        } else {
          // A function-scope variable is part of the body, not the module.
          m.funcs[cur].body.push_back(ins);
        }
        break;
      case OpExecutionMode: {
        // `LocalSize x y z`, all three kept.
        if (ins.a.size() >= 5 && ins.a[1] == 17u && ins.a[2] >= 1u && ins.a[3] >= 1u &&
            ins.a[4] >= 1u && ins.a[2] * ins.a[3] * ins.a[4] <= 1024u)
          for (auto& e : m.entries)
            if (e.function == ins.a[0]) { e.lx = ins.a[2]; e.ly = ins.a[3]; e.lz = ins.a[4]; }
        break;
      }
      case OpEntryPoint: {
        Entry e;
        e.exec_model = ins.a[0];
        e.function = ins.a[1];
        e.name = literal_at(ins.a, 2);
        size_t k = 2 + literal_words(e.name);
        for (; k < ins.a.size(); k++) e.iface.push_back(ins.a[k]);
        m.entries.push_back(e);
        break;
      }
      case OpExtInstImport:
        m.ext_name[ins.a[0]] = literal_at(ins.a, 1);
        break;
      case OpFunction: {
        Function f;
        f.id = ins.a[1];
        f.defined = true;
        m.funcs[f.id] = f;
        cur = f.id;
        break;
      }
      case OpFunctionEnd: {
        if (cur != 0) {
          bool any_label = false;
          for (const auto& in : m.funcs[cur].body)
            if (in.op == OpLabel) any_label = true;
          m.funcs[cur].defined = any_label;
        }
        cur = 0;
        break;
      }
      default:
        if (cur != 0) m.funcs[cur].body.push_back(ins);
        break;
    }
  }

  if (m.entries.empty()) { err = "no OpEntryPoint: nothing to run"; return false; }

  // Array lengths reference constants that appear later in the module.
  for (auto& it : m.types) {
    if (it.second.kind != KArray || it.second.length_id == 0) continue;
    auto c = m.cexpr.find(it.second.length_id);
    if (c == m.cexpr.end() || c->second.compare(0, 2, "0x") != 0) {
      err = "array length is not an integer constant";
      return false;
    }
    it.second.length = (uint32_t)std::strtoull(c->second.c_str(), nullptr, 0);
  }
  return true;
}

// Emission
struct Block {
  uint32_t label = 0;
  size_t begin = 0, end = 0;  // indices into the function body
};

struct PhiIn {
  uint32_t phi = 0;
  uint32_t pred = 0;
  uint32_t value = 0;
};

class Emitter {
 public:
  Emitter(Mod& m, const Function& f) : m_(m), f_(f) {}

  bool run();
  std::string err;
  bool sync = false;
  int nargs = 0;
  bool warp_ = false;
  bool warp_blocks_ = false;
  bool warp_step_ = false;
  void warp_target(bool blocks, bool step) {
    warp_ = true;
    warp_blocks_ = blocks;
    warp_step_ = step;
  }
  bool warp_blocks() const { return warp_blocks_; }
  bool warp_step() const { return warp_step_; }
  int max_case() const { return max_case_; }
  int slot_words() const { return slot_off_ < 32 ? 32 : slot_off_; }
  void warp_reason(const std::string& why) { warp_why_ = why; }
  bool wide_atom = false;
  struct RedAcc {
    std::string addr;  // the pointer expression, as emitted
    std::string op;    // add, sub, and, or, xor
    std::string name;  // the local accumulator
  };
  std::vector<RedAcc> red_accs;
  struct ShadowSite {
    std::string buf;     // the binding token, e.g. bufs[0]
    std::string call;    // the atomic call, as emitted
    std::string shadow;  // the local form that replaces it
    std::string decl;    // `uint64_t weft_shN = *(...)` before the walk
    std::string flush;   // and `*(...) = weft_shN` after it
  };
  std::vector<ShadowSite> shadows;
  int ntex = 0, nsamp = 0;
  std::vector<std::string> buf_type;  // one element of each binding, in bytes

  std::string bytes_of(uint32_t pointee) const {
    auto t = m_.types.find(pointee);
    if (t == m_.types.end()) return "";
    uint32_t id = pointee;
    for (int hop = 0; hop < 2; hop++) {
      auto p = m_.types.find(id);
      if (p == m_.types.end() || p->second.kind != KPointer) break;
      id = p->second.elem;
    }
    t = m_.types.find(id);
    if (t == m_.types.end()) return "";
    if (t->second.kind == KArray) {
      auto s = m_.array_stride.find(pointee);
      if (s != m_.array_stride.end()) return std::to_string(s->second);
      id = t->second.elem;
    }
    uint32_t sz = 0;
    if (!tsize(id, sz)) return "";
    return std::to_string(sz ? sz : 1);
  }
  const std::string& decls() const { return decls_; }
  const std::string& code() const { return body_; }
  const std::string& cases() const { return cases_; }
  const std::string& init() const { return init_; }
  void gpu_target(bool on) { gpu = on; }
  uint32_t arena_bytes() const { return wg_bytes_; }
  const std::vector<std::string>& textures() const { return tex_access; }
  // And whether each of those parameters is a 2-D array, in the same order.
  const std::vector<bool>& arrayed_textures() const { return tex_arrayed; }
  // And whether each of those is a 3-D volume, in the same order.
  const std::vector<bool>& volume_textures() const { return tex_vol; }
  // And whether each of those is a cube, in the same order.
  const std::vector<bool>& cube_textures() const { return tex_cube; }
  // And what each of those is, by name, for the launcher's binding check.
  const std::vector<std::string>& texture_shapes() const { return tex_shape; }

 private:
  Mod& m_;
  const Function& f_;
  bool pointer_used_once(uint32_t id) const {
    int seen = 0;
    for (const Instr& in : f_.body)
      for (uint32_t o : in.a)
        if (o == id && ++seen > 1) return false;
    return true;
  }
  bool unused_anywhere(uint32_t id) const {
    int seen = 0;
    for (const Instr& in : f_.body)
      for (uint32_t o : in.a)
        if (o == id && ++seen > 1) return false;
    return true;
  }
  std::map<uint32_t, std::string> base_addr_;  // id -> address expression (no state field)
  std::map<uint32_t, std::string> pval_;       // pointer id -> value it points at
  std::map<uint32_t, std::string> local_;      // id -> computed-once scalar expression
  std::map<uint32_t, uint32_t> vt_;            // id -> type id
  std::map<uint32_t, std::string> fields_;     // id -> state field name
  std::map<uint32_t, std::vector<PhiIn>> phis_;
  std::set<uint32_t> own_storage_;
  std::set<uint32_t> globals_addressed_;
  // Pointer id -> the MSL address space it points into.
  std::map<uint32_t, std::string> ptr_space_;
  std::vector<uint32_t> deferred_ptr_;
  std::vector<std::string> tex_access;
  std::vector<bool> tex_arrayed;
  std::vector<bool> tex_vol;      // the same, for `Dim = 3D`
  std::vector<bool> tex_cube;     // and for `Dim = Cube`, which is filter-only
  std::vector<std::string> tex_shape;
  bool writes_image_ = false;
  bool samples_image_ = false;
  std::set<uint32_t> ptr_globals_;  // module-scope pointer globals: loading one yields its base
  std::vector<Block> blocks_;
  std::map<uint32_t, int> label_of_;
  std::string decls_, body_, cases_, init_;
  int resume_next_ = 1;
  bool subgroup_ = false;
  bool barrier_ = false;
  // Emitted for the GPU: the same decode, Metal instead of C++.
  bool gpu = false;
  std::string warp_why_ = "needs something the warp target cannot provide";
  int max_case_ = 0;
  int slot_off_ = 0;  // words of the slot array handed out so far
  uint32_t cur_block_ = 0;
  uint32_t wg_bytes_ = 0;
  int nbuffers_ = 0, nscalars_ = 0, ntex_ = 0, nsamp_ = 0;
  std::vector<uint32_t> buf_elem_;  // pointee type of each binding, in bind order

  // The space a pointer id points into.
  std::string space_of(uint32_t id) const {
    auto p = ptr_space_.find(id);
    if (p != ptr_space_.end()) return p->second;
    const std::string e = addr(id);
    if (e.find("threadgroup") != std::string::npos) return "threadgroup";
    if (e.find("thread ") != std::string::npos) return "thread";
    return "device";
  }

  std::string addr_space(const char* space, bool cst) const {
    std::string q;
    if (cst) q += "const ";
    if (gpu) { q += space; q += " "; }
    return q;
  }
  std::string math(const std::string& name, const std::string& args) const {
    return gpu ? (name + "(" + args + ")") : ("std::" + name + "(" + args + ")");
  }
  std::string jump(uint32_t label) const {
    if (!gpu) return "goto L" + std::to_string(label) + ";";
    if (warp_step_) return "{ s->pc[l] = " + std::to_string(label) + "u; continue; }";
    return "{ s->pc = " + std::to_string(label) + "u; continue; }";
  }
  std::string case_open(uint32_t id) const {
    const std::string t = std::to_string(id);
    return "      case " + t + "u: { for (uint32_t l = 0; l < 32u; l++) {\n"
           "        if (weft_base + l >= weft_lanes) break;\n"
           "        if (s->pc[l] != " + t + "u) continue;\n"
           "        weft_ran = true;\n"
           "        ctx.lid = l; ctx.gid = (uint64_t)weft_base + (uint64_t)l;\n";
  }
  std::string case_close() const { return "      } } break;\n"; }
  std::string park(uint32_t id) {
    if (id > (uint32_t)max_case_) max_case_ = (int)id;
    return "{ s->pc[l] = " + std::to_string(id) + "u; continue; }\n" + case_close() +
           case_open(id);
  }
  // The warp's slot array is carved into windows of 32, one per rendezvous site.
  std::string slot_mask() const {
    return "((weft_lanes > weft_base) ? (((weft_lanes - weft_base) >= 32u) ? 0xFFFFFFFFu : "
           "((1u << (weft_lanes - weft_base)) - 1u)) : 0u)";
  }
  std::string param_of(int i) const {
    const std::string n = std::to_string(i);
    return gpu ? ("weft::param(ctx, " + n + ")") : ("weft::param(" + n + ")");
  }
  // The identity of a min/max fold.
  std::string inf_expr(const std::string& ty, bool negate) const {
    if (!gpu)
      return std::string(negate ? "-std::numeric_limits<" : "std::numeric_limits<") + ty +
             ">::has_infinity ? " + (negate ? "-std::numeric_limits<" : "std::numeric_limits<") +
             ty + ">::infinity() : " +
             (negate ? "-std::numeric_limits<" : "std::numeric_limits<") + ty +
             ">::" + (negate ? "lowest()" : "max()");
    return negate ? "-INFINITY" : "INFINITY";
  }

  bool ctype(uint32_t tid, std::string& out) const;
  bool tsize(uint32_t tid, uint32_t& out) const;
  std::string val(uint32_t id) const;
  std::string addr(uint32_t id) const;
  uint32_t vtype(uint32_t id) const;
  std::string field(uint32_t id) const;
  std::string fref(uint32_t id) const;
  bool vector_of(uint32_t tid, uint32_t& elem, uint32_t& n) const;
  std::string builtin_expr(uint32_t var, int comp) const;
  bool f64_type(uint32_t tid) const;
  bool shfl_pub(uint32_t tid, const std::string& e, std::string& out);
  bool wide_shfl(uint32_t tid, const std::string& e, const std::string& peer, std::string& out);
  std::map<uint32_t, std::pair<std::string, std::string>> multi_;
  struct FMulDef {
    uint32_t ty = 0, a = 0, b = 0;
    uint32_t block = 0;
  };
  std::map<uint32_t, FMulDef> fmul_;

  bool chain(const Instr& ins);
  bool emit_one(const Instr& ins);
  bool emit_import(const Instr& ins, const std::string& sym);
  bool in_call_ = false;    // a `return` here ends the call, not the kernel
  bool call_done_ = false;  // the inlined body reached its return
  uint32_t call_ret_ = 0;   // the id the call's value lands in, or 0 for a void call
  int call_depth_ = 0;      // nested inlining, and the recursion guard
  std::string edge_copies(uint32_t from, uint32_t to) const;
};

bool Emitter::ctype(uint32_t tid, std::string& out) const {
  auto it = m_.types.find(tid);
  if (it == m_.types.end()) return false;
  const SType& t = it->second;
  switch (t.kind) {
    case KVoid: out = "void"; return true;
    case KBool: out = "bool"; return true;
    case KInt:
      if (t.width == 8) out = t.is_signed ? "int8_t" : "uint8_t";
      else if (t.width == 16) out = t.is_signed ? "int16_t" : "uint16_t";
      else if (t.width == 32) out = t.is_signed ? "int32_t" : "uint32_t";
      else if (t.width == 64) out = t.is_signed ? "int64_t" : "uint64_t";
      else if (t.width == 1) out = "bool";
      else return false;
      return true;
    case KFloat:
      // A 16-bit float.
      if (t.width == 16) { out = gpu ? "half" : "_Float16"; return true; }
      if (t.width == 32) { out = "float"; return true; }
      if (t.width == 64) { out = gpu ? "uint64_t" : "double"; return true; }
      return false;
    case KPointer:
      out = gpu ? "device unsigned char*" : "unsigned char*";
      return true;
    case KImage: {
      // On the GPU an image is the bound texture plus its size, held by value.
      const bool vol = t.count == 2u;
      const bool cb = t.count == 3u;
      out = gpu ? (cb ? "weft::CubeImageRef"
                      : t.arrayed != 0
                            ? (writes_image_ ? "weft::RWArrayImageRef" : "weft::ArrayImageRef")
                            : vol ? (writes_image_ ? "weft::RWImage3DRef" : "weft::Image3DRef")
                                  : (writes_image_ ? "weft::RWImageRef" : "weft::ImageRef"))
                : "const weft::Texture*";
      return true;
    }
    case KSampler:
      out = gpu ? "sampler" : "weft::Sampler";
      return true;
    case KSampledImage: {
      auto e = m_.types.find(t.elem);
      const bool img = gpu && e != m_.types.end() && e->second.kind == KImage;
      const bool arr = img && e->second.arrayed != 0;
      const bool vol = img && !arr && e->second.count == 2u;
      const bool cb = img && e->second.count == 3u;
      out = cb ? "weft::SampledCubeImage"
                : vol ? "weft::SampledImage3D"
                      : arr ? "weft::SampledArrayImage"
                            : "weft::SampledImage";
      return true;
    }
    case KMatrix: {
      auto e = m_.types.find(t.elem);
      if (e == m_.types.end() || e->second.kind != KVector) return false;
      if (e->second.count < 2 || e->second.count > 4) return false;
      if (t.count < 2 || t.count > 4) return false;
      out = "weft_f" + std::to_string(e->second.count) + "x" + std::to_string(t.count);
      return true;
    }
    case KVector: {
      auto e = m_.types.find(t.elem);
      if (e == m_.types.end()) return false;
      if (t.count < 2 || t.count > 4) return false;
      const SType& et = e->second;
      const char* pre = nullptr;
      if (et.kind == KFloat && et.width == 32) pre = "weft_f";
      else if (et.kind == KFloat && et.width == 64) {
        pre = "weft_d";
      }
      else if (et.kind == KFloat && et.width == 16) {
        // A vector of halves is carried as a vector of floats.
        pre = "weft_f";
      }
      else if (et.kind == KInt && et.width == 32) pre = et.is_signed ? "weft_i" : "weft_u";
      else if (et.kind == KInt && et.width == 64) pre = et.is_signed ? "weft_l" : "weft_ul";
      if (et.kind == KBool) {
        out = std::string(gpu ? "bool" : "weft_b") + std::to_string(t.count);
        return true;
      }
      if (!pre) return false;
      out = std::string(pre) + std::to_string(t.count);
      return true;
    }
    default:
      return false;
  }
}

bool Emitter::tsize(uint32_t tid, uint32_t& out) const {
  auto it = m_.types.find(tid);
  if (it == m_.types.end()) return false;
  const SType& t = it->second;
  switch (t.kind) {
    case KBool: out = 1; return true;
    case KInt: out = t.width == 1 ? 1 : t.width / 8; return true;
    case KFloat: out = t.width / 8; return true;
    case KPointer: out = 8; return true;
    case KVector: {
      uint32_t es = 0;
      if (!tsize(t.elem, es)) return false;
      out = es * t.count;  // the generated vector structs are packed, no padding
      return true;
    }
    case KMatrix: {
      uint32_t es = 0;
      if (!tsize(t.elem, es)) return false;
      out = es * t.count;  // a column-major array of float, no padding either
      return true;
    }
    case KArray: {
      uint32_t es = 0;
      auto s = m_.array_stride.find(tid);
      if (s != m_.array_stride.end()) es = s->second;
      else if (!tsize(t.elem, es)) return false;
      out = es * t.length;
      return true;
    }
    case KStruct: {
      uint32_t total = 0, running = 0;
      for (size_t k = 0; k < t.members.size(); k++) {
        uint32_t ms = 0;
        if (!tsize(t.members[k], ms)) return false;
        auto o = m_.member_offset.find({tid, (uint32_t)k});
        const uint32_t off = o != m_.member_offset.end() ? o->second : running;
        running = off + ms;
        if (running > total) total = running;
      }
      out = total;
      return true;
    }
    default:
      return false;
  }
}

bool Emitter::vector_of(uint32_t tid, uint32_t& elem, uint32_t& n) const {
  auto it = m_.types.find(tid);
  if (it == m_.types.end() || it->second.kind != KVector) return false;
  elem = it->second.elem;
  n = it->second.count;
  return true;
}

std::string Emitter::field(uint32_t id) const {
  auto f = fields_.find(id);
  if (f != fields_.end()) return f->second;
  return "v" + std::to_string(id);
}

// A value's reference.
std::string Emitter::fref(uint32_t id) const {
  return "s->" + field(id) + (warp_ ? "[l]" : "");
}

std::string Emitter::val(uint32_t id) const {
  auto c = m_.cexpr.find(id);
  if (c != m_.cexpr.end()) return c->second;
  auto l = local_.find(id);
  if (l != local_.end()) return l->second;
  auto p = pval_.find(id);
  if (p != pval_.end()) return p->second;
  return fref(id);
}

std::string Emitter::addr(uint32_t id) const {
  auto b = base_addr_.find(id);
  if (b != base_addr_.end()) return b->second;
  return warp_ ? fref(id) : "s->" + field(id);
}

uint32_t Emitter::vtype(uint32_t id) const {
  auto c = m_.ctype.find(id);
  if (c != m_.ctype.end()) return c->second;
  auto t = vt_.find(id);
  return t == vt_.end() ? 0u : t->second;
}

// A 64-bit float.
bool Emitter::f64_type(uint32_t tid) const {
  auto t = m_.types.find(tid);
  return t != m_.types.end() && t->second.kind == KFloat && t->second.width == 64;
}

std::string Emitter::builtin_expr(uint32_t var, int comp) const {
  auto g = m_.globals.find(var);
  if (g == m_.globals.end() || !g->second.has_builtin) return "";
  const uint32_t bi = g->second.builtin;
  uint32_t gx = 0, gy = 1, gz = 1;
  for (const auto& en : m_.entries)
    if (en.function == f_.id && en.lx) { gx = en.lx; gy = en.ly; gz = en.lz; break; }
  auto at = [&](int k) -> std::string {
    switch (bi) {
      case BIGlobalInvocationId: return k == 0 ? "ctx.gid" : "0u";
      case BILocalInvocationId:
        if (k == 0) return gx ? ("(ctx.lid % " + std::to_string(gx) + "u)") : "ctx.lid";
        if (k == 1)
          return gy > 1 ? ("((ctx.lid / " + std::to_string(gx) + "u) % " +
                           std::to_string(gy) + "u)")
                        : "0u";
        return gz > 1 ? ("(ctx.lid / " + std::to_string(gx * gy) + "u)") : "0u";
      case BIWorkgroupId: return k == 0 ? "ctx.tgid" : "0u";
      case BIWorkgroupSize:
        if (k == 0) return gx ? (std::to_string(gx) + "u") : "ctx.tg_size";
        return k == 1 ? (std::to_string(gy) + "u") : (std::to_string(gz) + "u");
      case BINumWorkgroups: case BINumEnqueuedWorkgroups:
        return k == 0 ? "((ctx.grid + ctx.tg_size - 1u) / ctx.tg_size)" : "1u";
      case BIGlobalSize: return k == 0 ? "ctx.grid" : "0u";
      case BILocalInvocationIndex: return k == 0 ? "ctx.lid" : "0u";
      case BIWorkDim: return "1u";
      case BIGlobalOffset: return "0u";
      case BINumSubgroups: case BINumEnqueuedSubgroups: return k == 0 ? "1u" : "1u";
      case BISubgroupSize: return k == 0 ? "32u" : "1u";
      case BISubgroupId: return k == 0 ? "(ctx.lid >> 5)" : "0u";
      case BISubgroupLocalInvocationId: return k == 0 ? "(ctx.lid & 31u)" : "0u";
      default: return "";
    }
  };
  // The width is the module's, not this translator's.
  uint64_t bits = 64ull;
  {
    auto pt = m_.types.find(g->second.type);
    if (pt != m_.types.end() && pt->second.kind == KPointer) {
      auto po = m_.types.find(pt->second.elem);
      if (po != m_.types.end() && po->second.kind == KVector) {
        auto el = m_.types.find(po->second.elem);
        if (el != m_.types.end() && el->second.kind == KInt) bits = el->second.width;
      } else if (po != m_.types.end() && po->second.kind == KInt) {
        bits = po->second.width;
      }
    }
  }
  const std::string cast = bits == 64 ? "uint64_t" : "uint32_t";
  auto cv = [&](int k) { return "(" + cast + ")" + at(k); };

  if (comp >= 0) return cv(comp);
  auto pt = m_.types.find(g->second.type);
  if (pt == m_.types.end()) return "";
  auto pointee = m_.types.find(pt->second.elem);
  if (pointee != m_.types.end() && pointee->second.kind == KVector) {
    const uint32_t n = pointee->second.count;
    std::string vt;
    if (!ctype(pt->second.elem, vt)) return "";
    std::string e = vt + "{";
    for (uint32_t k = 0; k < n; k++) e += (k ? ", " : "") + cv((int)k);
    return e + "}";
  }
  return at(0).empty() ? "" : cv(0);
}

bool Emitter::shfl_pub(uint32_t tid, const std::string& e, std::string& out) {
  std::string ty;
  if (!ctype(tid, ty)) { err = "a payload of an unsupported type"; return false; }
  if (ty == "float") out = "weft_bitcast<uint32_t>(" + e + ")";
  else if (ty == "double" || ty == "uint64_t" || ty == "int64_t") {
    err = "a 64-bit payload does not fit the one-word rendezvous";
    return false;
  } else if (ty == "bool") out = "((" + e + ") ? 1u : 0u)";
  else out = "(uint32_t)(" + e + ")";
  return true;
}

bool Emitter::wide_shfl(uint32_t tid, const std::string& e, const std::string& peer,
                        std::string& out) {
  auto it = m_.types.find(tid);
  if (it == m_.types.end()) { err = "a shuffle of an unknown type"; return false; }
  const SType t = it->second;
  subgroup_ = true;

  // Publish this word, wait for the pass, and keep the peer's word.
  auto pass = [&](const std::string& payload) -> std::string {
    if (gpu) {
      if (warp_step_) {
        const int off = slot_off_;
        slot_off_ += 32;
        const int sid = resume_next_++;
        const std::string tmp = "weft_shfl_" + std::to_string(sid);
        decls_ += "  uint32_t " + tmp + "[32];\n";
        body_ += "  weft_slots[" + std::to_string(off) + " + l] = (" + payload + "); " +
                 park((uint32_t)sid) + "\n";
        body_ += "  s->" + tmp + "[l] = weft_slots[" + std::to_string(off) + " + ((" + peer +
                 ") & 31u)];\n";
        return "s->" + tmp + "[l]";
      }
      return "simd_shuffle((" + payload + "), (ushort)((" + peer + ") & 31u))";
    }
    const int id = resume_next_++;
    const std::string tmp = "weft_shfl_" + std::to_string(id);
    decls_ += "  uint32_t " + tmp + ";\n";
    cases_ += "    case " + std::to_string(id) + ": goto R" + std::to_string(id) + ";\n";
    body_ += "  ((uint32_t*)weft::shfl_pub_buf(ctx))[ctx.lid] = (" + payload + "); s->pc = " +
             std::to_string(id) + "u; return;\n  R" + std::to_string(id) + ":; s->" + tmp +
             " = ((const uint32_t*)weft::shfl_read_buf(ctx))[" + peer + "];\n";
    return "s->" + tmp;
  };
  auto pack = [](const std::string& ty, const std::string& ex) -> std::string {
    if (ty == "float") return "weft_bitcast<uint32_t>(" + ex + ")";
    if (ty == "bool") return "((" + ex + ") ? 1u : 0u)";
    return "(uint32_t)(" + ex + ")";
  };
  auto unpack = [](const std::string& ty, const std::string& slot) -> std::string {
    if (ty == "float") return "weft_bitcast<float>(" + slot + ")";
    if (ty == "bool") return "((" + slot + ") != 0u)";
    return "(" + ty + ")(" + slot + ")";
  };

  if (t.kind == KVector) {
    std::string vt, ety;
    if (!ctype(tid, vt)) { err = "a shuffle of an unsupported vector"; return false; }
    if (!ctype(t.elem, ety)) { err = "a shuffle of an unsupported element"; return false; }
    if (t.count == 0 || t.count > 4) { err = "a shuffle of a vector that wide"; return false; }
    static const char* kComp[] = {"x", "y", "z", "w"};
    std::string joined;
    for (uint32_t k = 0; k < t.count; k++) {
      std::string one;
      if (!wide_shfl(t.elem, "(" + e + ")." + kComp[k], peer, one)) return false;
      if (k) joined += ", ";
      joined += one;
    }
    out = vt + "{" + joined + "}";
    return true;
  }

  // A 64-bit scalar goes out as two 32-bit halves, low first.
  if (t.kind == KFloat ? t.width == 64 : (t.kind == KInt && t.width == 64)) {
    const std::string word =
        (t.kind == KFloat && !gpu) ? ("weft_bitcast<uint64_t>(" + e + ")") :
                                     ("(uint64_t)(" + e + ")");
    const std::string lo = pass("(uint32_t)(" + word + " & 0xFFFFFFFFu)");
    const std::string hi = pass("(uint32_t)(" + word + " >> 32)");
    const std::string bits =
        "((uint64_t)(" + lo + ") | ((uint64_t)(" + hi + ") << 32))";
    if (t.kind == KFloat)
      out = gpu ? bits : ("weft_bitcast<double>(" + bits + ")");
    else
      out = std::string("(") + (t.is_signed ? "int64_t" : "uint64_t") + ")(" + bits + ")";
    return true;
  }

  std::string ty;
  if (!ctype(tid, ty)) { err = "a shuffle of an unsupported type"; return false; }
  out = unpack(ty, pass(pack(ty, e)));
  return true;
}

// OpAccessChain and friends.
bool Emitter::chain(const Instr& ins) {
  const uint32_t rty = ins.a[0], rid = ins.a[1], base_id = ins.a[2];

  auto g = m_.globals.find(base_id);
  if (g != m_.globals.end() && g->second.has_builtin) {
    if (ins.a.size() < 4) { pval_[rid] = builtin_expr(base_id, -1); vt_[rid] = rty; return true; }
    if (ins.a.size() != 4) { err = "more than one index into a builtin variable"; return false; }
    auto c = m_.cexpr.find(ins.a[3]);
    if (c == m_.cexpr.end()) { err = "builtin component must be a constant"; return false; }
    const long k = std::strtol(c->second.c_str(), nullptr, 0);
    const std::string bc = builtin_expr(base_id, (int)k);
    if (bc.empty()) { err = "builtin variable has no component " + std::to_string(k); return false; }
    pval_[rid] = bc;
    vt_[rid] = rty;
    return true;
  }

  uint32_t base_type = vtype(base_id);
  auto bt = m_.types.find(base_type);
  if (bt == m_.types.end() || bt->second.kind != KPointer) {
    err = "address chain on something that is not a pointer";
    return false;
  }
  uint32_t cur = bt->second.elem;
  std::string off = "0";
  size_t k = 3;
  const bool scaled = (ins.op == OpPtrAccessChain || ins.op == OpInBoundsPtrAccessChain);
  if (scaled) {
    if (ins.a.size() <= k) { err = "bad pointer access chain"; return false; }
    uint32_t es = 0;
    if (!tsize(cur, es)) { err = "cannot size the element of an access chain"; return false; }
    off = "(" + val(ins.a[k]) + " * " + std::to_string(es) + "ull)";
    k++;
  }
  for (; k < ins.a.size(); k++) {
    auto it = m_.types.find(cur);
    if (it == m_.types.end()) { err = "bad type in an access chain"; return false; }
    const SType& t = it->second;
    if (t.kind == KArray) {
      uint32_t stride = 0;
      auto s = m_.array_stride.find(cur);
      if (s != m_.array_stride.end()) stride = s->second;
      else if (!tsize(t.elem, stride)) { err = "cannot size an array element"; return false; }
      off = "(" + off + " + " + val(ins.a[k]) + " * " + std::to_string(stride) + "ull)";
      cur = t.elem;
    } else if (t.kind == KStruct) {
      auto cst = m_.cexpr.find(ins.a[k]);
      if (cst == m_.cexpr.end()) { err = "struct index must be a constant"; return false; }
      const uint32_t idx = (uint32_t)std::strtoull(cst->second.c_str(), nullptr, 0);
      if (idx >= t.members.size()) { err = "struct index out of range"; return false; }
      uint32_t mo = 0;
      auto o = m_.member_offset.find({cur, idx});
      if (o != m_.member_offset.end()) {
        mo = o->second;
      } else {
        for (uint32_t j = 0; j < idx; j++) {
          uint32_t ms = 0;
          if (!tsize(t.members[j], ms)) { err = "cannot size a struct member"; return false; }
          mo += ms;
        }
      }
      if (mo) off = "(" + off + " + " + std::to_string(mo) + "ull)";
      cur = t.members[idx];
    } else if (t.kind == KVector) {
      uint32_t es = 0;
      if (!tsize(t.elem, es)) { err = "cannot size a vector component"; return false; }
      off = "(" + off + " + " + val(ins.a[k]) + " * " + std::to_string(es) + "ull)";
      cur = t.elem;
    } else {
      err = "access chain through a scalar type";
      return false;
    }
  }

  const std::string base = addr(base_id);
  const std::string sp = space_of(base_id);
  ptr_space_[rid] = sp;
  body_ += "  " + fref(rid) + " = (" + addr_space(sp.c_str(), false) + "unsigned char*)(" +
            base + " + " + off + ");\n";
  vt_[rid] = rty;
  return true;
}

// The parallel copy a phi needs on the edge P -> B: read every incoming value
std::string Emitter::edge_copies(uint32_t from, uint32_t to) const {
  auto it = phis_.find(to);
  if (it == phis_.end()) return "";
  std::string out, writes;
  int n = 0;
  for (const PhiIn& p : it->second) {
    if (p.pred != from) continue;
    std::string ty;
    if (!ctype(vtype(p.phi), ty)) ty = "auto";
    const std::string t = "weft_p" + std::to_string(n++);
    out += "const " + ty + " " + t + " = " + val(p.value) + "; ";
    writes += fref(p.phi) + " = " + t + "; ";
  }
  if (out.empty()) return "";
  return "{ " + out + writes + "} ";
}

bool Emitter::emit_one(const Instr& ins) {
  const auto& a = ins.a;
  const uint32_t op = ins.op;
  const bool res = !no_result(op) && a.size() >= 2;
  const uint32_t rt = res ? a[0] : 0;
  const uint32_t rd = res ? a[1] : 0;

  const bool pair_op = op == OpIAddCarry || op == OpISubBorrow || op == OpUMulExtended ||
                       op == OpSMulExtended;
  std::string rty;
  if (res && !pair_op && !ctype(rt, rty)) {
    err = "result of opcode " + std::to_string(op) + " has an unsupported type";
    return false;
  }
  auto V = [&](size_t k) -> std::string {
    if (k >= a.size()) { err = "opcode " + std::to_string(op) + " is missing operands"; return "0"; }
    return val(a[k]);
  };
  auto set = [&](const std::string& e) { body_ += "  " + fref(rd) + " = " + e + ";\n"; };

  if (in_call_ && (op == OpReturn || op == OpReturnValue)) {
    if (op == OpReturnValue) body_ += "  " + fref(call_ret_) + " = " + val(a[0]) + ";\n";
    call_done_ = true;
    return true;
  }

  if (op == OpExtInst && a.size() > 4 && res && f64_type(rt)) {
    const uint32_t num = a[3];
    const char* fn = nullptr;
    int arity = 1;
    bool int_arg = false;
    switch (num) {
      case 0: fn = "acos_bits"; break;
      case 1: fn = "acosh_bits"; break;
      case 2: fn = "acospi_bits"; break;
      case 3: fn = "asin_bits"; break;
      case 4: fn = "asinh_bits"; break;
      case 5: fn = "asinpi_bits"; break;
      case 6: fn = "atan_bits"; break;
      case 7: fn = "atan2_bits"; arity = 2; break;
      case 8: fn = "atanh_bits"; break;
      case 9: fn = "atanpi_bits"; break;
      case 10: fn = "atan2pi_bits"; arity = 2; break;
      case 11: fn = "cbrt_bits"; break;
      case 13: fn = "copysign_bits"; arity = 2; break;
      case 14: fn = "cos_bits"; break;
      case 15: fn = "cosh_bits"; break;
      case 16: fn = "cospi_bits"; break;
      case 19: fn = "exp_bits"; break;
      case 20: fn = "exp2_bits"; break;
      case 21: fn = "exp10_bits"; break;
      case 22: fn = "expm1_bits"; break;
      case 24: fn = "fdim_bits"; arity = 2; break;
      case 29: fn = "fmod_bits"; arity = 2; break;
      case 32: fn = "hypot_bits"; arity = 2; break;
      case 34: fn = "ldexp_bits"; arity = 2; int_arg = true; break;
      case 37: fn = "log_bits"; break;
      case 38: fn = "log2_bits"; break;
      case 39: fn = "log10_bits"; break;
      case 40: fn = "log1p_bits"; break;
      case 43: fn = "maxmag_bits"; arity = 2; break;
      case 44: fn = "minmag_bits"; arity = 2; break;
      case 47: fn = "nextafter_bits"; arity = 2; break;
      case 48: fn = "pow_bits"; arity = 2; break;
      case 49: fn = "pown_bits"; arity = 2; int_arg = true; break;
      case 50: fn = "powr_bits"; arity = 2; break;
      case 17: fn = "erfc_bits"; break;
      case 18: fn = "erf_bits"; break;
      case 51: fn = "remainder_bits"; arity = 2; break;
      case 65: fn = "tgamma_bits"; break;
      case 54: fn = "rootn_bits"; arity = 2; int_arg = true; break;
      case 57: fn = "sin_bits"; break;
      case 59: fn = "sinh_bits"; break;
      case 60: fn = "sinpi_bits"; break;
      case 62: fn = "tan_bits"; break;
      case 63: fn = "tanh_bits"; break;
      case 64: fn = "tanpi_bits"; break;
      case 67: case 78: case 80: fn = nullptr; break;
      case 69: case 83: fn = "exp_bits"; break;
      case 70: case 84: fn = "exp2_bits"; break;
      case 71: case 85: fn = "exp10_bits"; break;
      case 72: case 86: fn = "log_bits"; break;
      case 73: case 87: fn = "log2_bits"; break;
      case 74: case 88: fn = "log10_bits"; break;
      case 75: case 89: fn = "pow_bits"; arity = 2; break;
      case 92: fn = "sin_bits"; break;
      case 93: fn = "sqrt_bits"; break;
      case 94: fn = "tan_bits"; break;
      default: break;
    }
    if (fn) {
      auto operand = [&](int k) {
        const std::string v = V(4 + k);
        return gpu ? v : ("weft_bitcast<uint64_t>(" + v + ")");
      };
      std::string call = std::string("weft_fp64::") + fn + "(" + operand(0);
      if (arity == 2 && a.size() > 5)
        call += ", " + (int_arg ? ("(int32_t)(" + V(5) + ")") : operand(1));
      call += ")";
      set(gpu ? call : ("weft_bitcast<double>(" + call + ")"));
      return true;
    }
    if (num == 30 || num == 31 || num == 45 || num == 52 || num == 58) {
      const bool two_args = (num == 52);
      const char* first = num == 30   ? "fract_bits"
                        : num == 31   ? "frexp_f_bits"
                        : num == 45   ? "modf_f_bits"
                        : num == 52   ? "remquo_r_bits"
                                      : "sincos_s_bits";
      const char* second = num == 30  ? "floor_bits"
                         : num == 31  ? "frexp_e_bits"
                         : num == 45  ? "modf_i_bits"
                         : num == 52  ? "remquo_q_bits"
                                      : "sincos_c_bits";
      const uint32_t pid = a.back();
      auto pt = m_.types.find(vtype(pid));
      if (pt == m_.types.end() || pt->second.kind != KPointer) {
        err = "an extended instruction with two results needs a pointer for the second";
        return false;
      }
      std::string sty;
      if (!ctype(pt->second.elem, sty)) {
        err = "the second result has an unsupported type";
        return false;
      }
      auto operand = [&](size_t k) {
        const std::string v = V(k);
        return gpu ? v : ("weft_bitcast<uint64_t>(" + v + ")");
      };
      std::string args = operand(4);
      if (two_args) args += ", " + operand(5);
      const bool f64_out = (sty == "double" || sty == "uint64_t");
      const std::string sv = std::string("weft_fp64::") + second + "(" + args + ")";
      const std::string sp = space_of(pid);
      body_ += "  weft_st<" + sty + ">((" + addr_space(sp.c_str(), false) + "void*)" + addr(pid) +
               ", " + (f64_out && !gpu ? ("weft_bitcast<double>(" + sv + ")") : sv) + ");\n";
      const std::string rv = std::string("weft_fp64::") + first + "(" + args + ")";
      set(gpu ? rv : ("weft_bitcast<double>(" + rv + ")"));
      return true;
    }
  }

  if (!gpu && op == OpExtInst && a.size() > 4 && a[3] == 61u && rty == "double") {
    set("std::sqrt(" + V(4) + ")");
    return true;
  }

  if (gpu) {
    const bool r64 = res && f64_type(rt);
    const bool o64 = a.size() > 2 && f64_type(vtype(a[2]));
    if (r64 || o64) {
      auto bin = [&](const char* fn) {
        set(std::string("weft_fp64::") + fn + "(" + V(2) + ", " + V(3) + ")");
      };
      switch (op) {
        case OpFAdd: bin("add"); return true;
        case OpFSub: bin("sub"); return true;
        case OpFMul: bin("mul"); return true;
        case OpFDiv: bin("div"); return true;
        case OpFNegate: set("weft_fp64::neg(" + V(2) + ")"); return true;
        case OpFOrdEqual: bin("eq"); return true;
        case OpFOrdNotEqual: bin("ne"); return true;
        case OpFUnordEqual: set("!weft_fp64::eq(" + V(2) + ", " + V(3) + ")"); return true;
        case OpFUnordNotEqual:
          set("!weft_fp64::ne(" + V(2) + ", " + V(3) + ")");
          return true;
        case OpFOrdLessThan: bin("lt"); return true;
        case OpFUnordLessThan: set("!weft_fp64::ge(" + V(2) + ", " + V(3) + ")"); return true;
        case OpFOrdGreaterThan: bin("gt"); return true;
        case OpFUnordGreaterThan: set("!weft_fp64::le(" + V(2) + ", " + V(3) + ")"); return true;
        case OpFOrdLessThanEqual: bin("le"); return true;
        case OpFUnordLessThanEqual: set("!weft_fp64::gt(" + V(2) + ", " + V(3) + ")"); return true;
        case OpFOrdGreaterThanEqual: bin("ge"); return true;
        case OpFUnordGreaterThanEqual:
          set("!weft_fp64::lt(" + V(2) + ", " + V(3) + ")");
          return true;
        case OpIsNan: set("weft_fp64::is_nan(" + V(2) + ")"); return true;
        case OpIsInf: set("weft_fp64::is_inf(" + V(2) + ")"); return true;
        case OpIsFinite:
          set("(!weft_fp64::is_nan(" + V(2) + ") && !weft_fp64::is_inf(" + V(2) + "))");
          return true;
        case OpSignBitSet: set("((" + V(2) + ") >> 63) != 0ull"); return true;
        case OpConvertFToS:
          set(std::string("weft_fp64::to_i") + (r64 ? "64(" : "32(") + V(2) + ")");
          return true;
        case OpConvertFToU:
          set(std::string("weft_fp64::to_u") + (r64 ? "64(" : "32(") + V(2) + ")");
          return true;
        case OpConvertSToF: set("weft_fp64::from_i64(" + V(2) + ")"); return true;
        case OpConvertUToF: set("weft_fp64::from_u64(" + V(2) + ")"); return true;
        case OpFConvert:
          if (r64)
            set("weft_fp64::from_f32(weft_bitcast<uint32_t>(" + V(2) + "))");
          else
            set("weft_bitcast<float>(weft_fp64::to_f32(" + V(2) + "))");
          return true;
        case OpExtInst: {
          // The numbers are OpenCL.std's, which is the set this front end knows.
          const uint32_t num = a.size() > 3 ? a[3] : 0u;
          const std::string x = a.size() > 4 ? V(4) : "0ull";
          switch (num) {
            case 12: set("weft_fp64::ceil_bits(" + x + ")"); return true;
            case 23: set("weft_fp64::abs_bits(" + x + ")"); return true;
            case 25: set("weft_fp64::floor_bits(" + x + ")"); return true;
            case 55: set("weft_fp64::round_bits(" + x + ")"); return true;
            case 66: set("weft_fp64::trunc_bits(" + x + ")"); return true;
            case 27: set("weft_fp64::max_bits(" + x + ", " + V(5) + ")"); return true;
            case 28: set("weft_fp64::min_bits(" + x + ", " + V(5) + ")"); return true;
            case 26: set("weft_fp64::fma(" + x + ", " + V(5) + ", " + V(6) + ")"); return true;
            case 61: set("weft_fp64::sqrt_bits(" + x + ")"); return true;
            default:
              err = "OpenCL.std instruction " + std::to_string(num) +
                    " over a double has no GPU form: the arithmetic, the comparisons,"
                    " the conversions, floor/ceil/trunc/round, min/max, sqrt and fma do";
              return false;
          }
        }
        case OpLoad: case OpStore: case OpSelect: case OpPhi: case OpCopyObject:
        case OpVariable: case OpFunctionParameter: case OpCompositeExtract:
        case OpCompositeConstruct: case OpCompositeInsert: case OpVectorShuffle:
        case OpGroupNonUniformBroadcast: case OpGroupNonUniformBroadcastFirst:
        case OpGroupNonUniformShuffle: case OpGroupNonUniformShuffleXor:
        case OpGroupNonUniformShuffleUp: case OpGroupNonUniformShuffleDown:
        case OpGroupNonUniformQuadBroadcast: case OpGroupNonUniformQuadSwap:
        case OpGroupNonUniformIAdd: case OpGroupNonUniformFAdd:
        case OpGroupNonUniformIMul: case OpGroupNonUniformFMul:
        case OpGroupNonUniformSMin: case OpGroupNonUniformUMin:
        case OpGroupNonUniformFMin: case OpGroupNonUniformSMax:
        case OpGroupNonUniformUMax: case OpGroupNonUniformFMax:
        case OpGroupNonUniformBitwiseAnd: case OpGroupNonUniformBitwiseOr:
        case OpGroupNonUniformBitwiseXor:
        case OpGroupNonUniformLogicalAnd: case OpGroupNonUniformLogicalOr:
        case OpGroupNonUniformLogicalXor:
        case OpBitcast: case OpFunctionCall:
          break;
        default:
          err = "opcode " + std::to_string(op) + " on a double has no GPU form";
          return false;
      }
    }
  }

  {
    uint32_t ve = 0, vn = 0;
    if (gpu && res && vector_of(rt, ve, vn) && f64_type(ve)) {
      static const char* const kComp = "xyzw";
      auto comp = [&](const std::string& A, uint32_t i) {
        return "(" + A + ")." + std::string(1, kComp[i]);
      };
      auto pair = [&](const char* fn, const std::string& A, const std::string& B) -> std::string {
        std::string e = rty + "{";
        for (uint32_t i = 0; i < vn; i++) {
          if (i) e += ", ";
          e += std::string("weft_fp64::") + fn + "(" + comp(A, i) + ", " + comp(B, i) + ")";
        }
        return e + "}";
      };
      switch (op) {
        case OpFAdd: set(pair("add", V(2), V(3))); return true;
        case OpFSub: set(pair("sub", V(2), V(3))); return true;
        case OpFMul: set(pair("mul", V(2), V(3))); return true;
        case OpFDiv: set(pair("div", V(2), V(3))); return true;
        case OpFNegate: {
          std::string e = rty + "{";
          for (uint32_t i = 0; i < vn; i++) {
            if (i) e += ", ";
            e += "weft_fp64::neg(" + comp(V(2), i) + ")";
          }
          set(e + "}");
          return true;
        }
        default: break;
      }
    }
  }

  // Component-wise binary op on packed vector structs. `op` is a C++ operator.
  auto compwise = [&](const std::string& A, const std::string& B, const char* o) -> std::string {
    uint32_t en = 0, n = 0;
    vector_of(rt, en, n);
    const char* k = "xyzw";
    std::string e = rty + "{";
    for (uint32_t i = 0; i < n; i++) {
      if (i) e += ", ";
      e += "(" + A + ")." + std::string(1, k[i]) + " " + o + " (" + B + ")." + std::string(1, k[i]);
    }
    return e + "}";
  };
  auto is_vec = [&](uint32_t tid) {
    uint32_t e2 = 0, n2 = 0;
    return vector_of(tid, e2, n2);
  };

  switch (op) {
    case OpNop: case OpLoopMerge: case OpSelectionMerge: case OpMemoryBarrier:
    case OpSource: case OpDecorate: case OpMemberDecorate: case OpName:
      return true;

    case OpUndef: set(rty + "()"); return true;
    case OpCopyObject: set(V(2)); return true;

    case OpAccessChain: case OpInBoundsAccessChain: case OpPtrAccessChain:
    case OpInBoundsPtrAccessChain:
      return chain(ins);

    case OpVariable: {
      const uint32_t vid = a.size() > 1 ? a[1] : 0u;
      auto t = m_.types.find(a[0]);
      if (t == m_.types.end() || t->second.kind != KPointer) { err = "bad OpVariable"; return false; }
      uint32_t sz = 0;
      if (!tsize(t->second.elem, sz)) { err = "local variable of an unsupported type"; return false; }
      if (sz == 0) sz = 1;
      const std::string th = addr_space("thread", false);
      decls_ += "  unsigned char " + field(vid) + (warp_ ? "[32][" : "[") +
                std::to_string(sz) + "];\n";
      base_addr_[vid] = "((" + th + "unsigned char*)s->" + field(vid) + (warp_ ? "[l]" : "") + ")";
      vt_[vid] = a[0];
      if (a.size() > 3) {
        const std::string at = "s->" + field(vid) + (warp_ ? "[l]" : "");
        init_ += gpu ? ("  weft::zero_thread(" + at + ", " + std::to_string(sz) + ");\n")
                     : ("  std::memset(" + at + ", 0, " + std::to_string(sz) + ");\n");
      }
      return true;
    }

    case OpLoad: {
      const uint32_t ptr = a[2];
      if (pval_.count(ptr)) { set(pval_[ptr]); return true; }
      if (local_.count(ptr)) { set(local_[ptr]); return true; }
      if (ptr_globals_.count(ptr)) { set(addr(ptr)); return true; }
      auto g = m_.globals.find(ptr);
      if (g != m_.globals.end() && g->second.has_builtin) {
        const std::string b = builtin_expr(ptr, -1);
        if (b.empty()) { err = "unsupported builtin variable"; return false; }
        set(b);
        return true;
      }
      {
        const std::string sp = space_of(ptr);
        set("weft_ld<" + rty + ">((" + addr_space(sp.c_str(), true) + "void*)" + addr(ptr) + ")");
      }
      return true;
    }

    case OpStore: {
      const uint32_t ptr = a[0];
      if (pval_.count(ptr)) { err = "store through a builtin variable"; return false; }
      std::string ty;
      if (!ctype(vtype(a[1]), ty)) { err = "store of an unsupported type"; return false; }
      {
        const std::string sp = space_of(ptr);
        body_ += "  weft_st<" + ty + ">((" + addr_space(sp.c_str(), false) + "void*)" + addr(ptr) +
                 ", " + V(1) + ");\n";
      }
      return true;
    }

    case OpCompositeExtract: {
      auto mul = multi_.find(a[2]);
      if (mul != multi_.end()) {
        if (a.size() != 4 || a[3] > 1) {
          err = "an extended integer result read as something other than its two words";
          return false;
        }
        set(a[3] == 0 ? mul->second.first : mul->second.second);
        return true;
      }
      std::string src = V(2);
      uint32_t cur = vtype(a[2]);
      const char* k = "xyzw";
      for (size_t i = 3; i < a.size(); i++) {
        auto t = m_.types.find(cur);
        if (t == m_.types.end()) { err = "composite extract from an unknown type"; return false; }
        if (t->second.kind == KVector) {
          if (a[i] >= t->second.count) { err = "vector index out of range"; return false; }
          src = "(" + src + ")." + std::string(1, k[a[i]]);
          cur = t->second.elem;
        } else if (t->second.kind == KMatrix) {
          if (a[i] >= t->second.count) { err = "matrix column out of range"; return false; }
          src = "weft_col(" + src + ", " + std::to_string(a[i]) + ")";
          cur = t->second.elem;
        } else {
          err = "composite extract from a non-vector";
          return false;
        }
      }
      set(src);
      return true;
    }

    case OpCompositeConstruct: {
      // A matrix is built from its columns, in order.
      auto mt = m_.types.find(rt);
      if (mt != m_.types.end() && mt->second.kind == KMatrix) {
        if (a.size() - 2 != mt->second.count) {
          err = "matrix built from a different number of columns";
          return false;
        }
        std::string e = "weft_mk(";
        for (size_t i = 2; i < a.size(); i++) {
          const uint32_t vt = vtype(a[i]);
          if (vt != 0 && vt != mt->second.elem) {
            err = "matrix built from something other than its column type";
            return false;
          }
          if (i > 2) e += ", ";
          e += val(a[i]);
        }
        set(e + ")");
        return true;
      }
      uint32_t n = 0, en = 0;
      if (!vector_of(rt, en, n)) { err = "composite construct of a non-vector"; return false; }
      if (a.size() - 2 != n) { err = "vector built from a different number of parts"; return false; }
      std::string e = rty + "{";
      for (size_t i = 2; i < a.size(); i++) {
        if (is_vec(vtype(a[i]))) { err = "vector built out of vectors"; return false; }
        if (i > 2) e += ", ";
        e += val(a[i]);
      }
      set(e + "}");
      return true;
    }

    case OpVectorShuffle: {
      uint32_t n = 0, en = 0;
      if (!vector_of(rt, en, n)) { err = "shuffle of a non-vector"; return false; }
      const std::string A = V(2), B = V(3);
      const char* k = "xyzw";
      std::string e = rty + "{";
      for (size_t i = 4; i < a.size(); i++) {
        const int64_t m = (int64_t)a[i];
        if (i > 4) e += ", ";
        if (m < 0) e += "0";
        else if ((size_t)m < 4) e += "(" + A + ")." + std::string(1, k[m]);
        else e += "(" + B + ")." + std::string(1, k[m - 4]);
      }
      set(e + "}");
      return true;
    }

    case OpVectorTimesScalar: {
      const std::string A = V(2), S = V(3);
      uint32_t en = 0, n = 0;
      vector_of(rt, en, n);
      const char* k = "xyzw";
      std::string e = rty + "{";
      for (uint32_t i = 0; i < n; i++) {
        if (i) e += ", ";
        e += "(" + A + ")." + std::string(1, k[i]) + " * (" + S + ")";
      }
      set(e + "}");
      return true;
    }

    case OpTranspose: set("weft_transpose(" + V(2) + ")"); return true;
    case OpMatrixTimesScalar:
    case OpVectorTimesMatrix:
    case OpMatrixTimesVector:
    case OpMatrixTimesMatrix:
      set("(" + V(2) + " * " + V(3) + ")");
      return true;

    case OpDot: {
      const std::string A = V(2), B = V(3);
      uint32_t en = 0, n = 0;
      vector_of(vtype(a[2]), en, n);
      const char* k = "xyzw";
      std::string e = "(";
      for (uint32_t i = 0; i < n; i++) {
        if (i) e += " + ";
        e += "(" + A + ")." + std::string(1, k[i]) + " * (" + B + ")." + std::string(1, k[i]);
      }
      set(e + ")");
      return true;
    }

    case OpIAddCarry: case OpISubBorrow: case OpUMulExtended: case OpSMulExtended: {
      auto struct_ty = m_.types.find(rt);
      if (struct_ty == m_.types.end() || struct_ty->second.kind != KStruct ||
          struct_ty->second.members.size() != 2) {
        err = "an extended integer operation whose result is not a struct of two words";
        return false;
      }
      auto op_ty = m_.types.find(vtype(a[2]));
      if (op_ty == m_.types.end() || op_ty->second.kind != KInt) {
        err = "an extended integer operation over something that is not an integer";
        return false;
      }
      const uint32_t w = op_ty->second.width;
      std::string lt, ht;
      if (!ctype(struct_ty->second.members[0], lt) || !ctype(struct_ty->second.members[1], ht)) {
        err = "an extended integer operation with an unsupported member type";
        return false;
      }
      const std::string ut = "uint" + std::to_string(w) + "_t";
      const std::string A = "((" + ut + ")(" + V(2) + "))";
      const std::string B = "((" + ut + ")(" + V(3) + "))";
      std::string low, high;
      if (op == OpIAddCarry) {
        if (w != 32 && w != 64) {
          err = "OpIAddCarry at a width other than 32 or 64";
          return false;
        }
        low = "(" + lt + ")(" + A + " + " + B + ")";
        high = (w == 32)
                   ? ("(" + ht + ")((uint64_t)(" + A + ") + (uint64_t)(" + B + ") >> 32)")
                   : ("(" + ht + ")(uint64_t)((" + A + " + " + B + ") < " + A + ")");
      } else if (op == OpISubBorrow) {
        if (w != 32 && w != 64) {
          err = "OpISubBorrow at a width other than 32 or 64";
          return false;
        }
        low = "(" + lt + ")(" + A + " - " + B + ")";
        high = "(" + ht + ")(uint64_t)(" + A + " < " + B + ")";
      } else if (w == 32) {
        const std::string prod =
            (op == OpSMulExtended)
                ? ("((uint64_t)((int64_t)(int32_t)(" + A + ") * (int64_t)(int32_t)(" + B + ")))")
                : ("((uint64_t)(" + A + ") * (uint64_t)(" + B + "))");
        low = "(" + lt + ")(uint32_t)(" + prod + ")";
        high = "(" + ht + ")(uint32_t)((" + prod + ") >> 32)";
      } else {
        err = std::string(op == OpUMulExtended ? "OpUMulExtended" : "OpSMulExtended") +
              " over 64-bit operands: the high word needs a 128-bit product, which neither"
              " target can name; the 32-bit form and OpIAddCarry/OpISubBorrow are here";
        return false;
      }
      multi_[rd] = {low, high};
      return true;
    }

    case OpFAdd: {
      const bool f32 = rty == "float" || rty == "float2" || rty == "float3" ||
                       rty == "float4";
      auto f = fmul_.find(a[2]);
      if (f32 && f != fmul_.end() && f->second.ty == rt && f->second.block == cur_block_) {
        set("weft_fma(" + val(f->second.a) + ", " + val(f->second.b) + ", " + V(3) + ")");
        return true;
      }
      set(is_vec(rt) ? compwise(V(2), V(3), "+") : V(2) + " + " + V(3));
      return true;
    }
    case OpFSub: {
      // `m * n - c` puts the negation on the addend, `c - m * n` on the product.
      if (rty == "float") {
        auto f = fmul_.find(a[2]);
        if (f != fmul_.end() && f->second.ty == rt && f->second.block == cur_block_) {
          set("weft_fma(" + val(f->second.a) + ", " + val(f->second.b) + ", -(" + V(3) + "))");
          return true;
        }
        if (a.size() >= 4) {
          auto g = fmul_.find(a[3]);
          if (g != fmul_.end() && g->second.ty == rt && g->second.block == cur_block_) {
            set("weft_fma(-(" + val(g->second.a) + "), " + val(g->second.b) + ", " + V(2) + ")");
            return true;
          }
        }
      }
      set(is_vec(rt) ? compwise(V(2), V(3), "-") : V(2) + " - " + V(3));
      return true;
    }
    case OpIAdd: set(is_vec(rt) ? compwise(V(2), V(3), "+") : V(2) + " + " + V(3)); return true;
    case OpISub: set(is_vec(rt) ? compwise(V(2), V(3), "-") : V(2) + " - " + V(3)); return true;
    case OpIMul: set(is_vec(rt) ? compwise(V(2), V(3), "*") : V(2) + " * " + V(3)); return true;
    case OpFMul: {
      set(is_vec(rt) ? compwise(V(2), V(3), "*") : V(2) + " * " + V(3));
      if (a.size() >= 4) fmul_[rd] = FMulDef{rt, a[2], a[3], cur_block_};
      return true;
    }
    case OpFDiv: set(is_vec(rt) ? compwise(V(2), V(3), "/") : V(2) + " / " + V(3)); return true;
    case OpUDiv: set("((uint64_t)(" + V(2) + ") / (uint64_t)(" + V(3) + "))"); return true;
    case OpSDiv: set("((int64_t)(" + V(2) + ") / (int64_t)(" + V(3) + "))"); return true;
    case OpUMod: set("((uint64_t)(" + V(2) + ") % (uint64_t)(" + V(3) + "))"); return true;
    case OpSRem: set("((int64_t)(" + V(2) + ") % (int64_t)(" + V(3) + "))"); return true;
    case OpSMod: {
      // Floor modulus: the sign follows the divisor, unlike C's %.
      const std::string x = "((int64_t)(" + V(2) + "))", y = "((int64_t)(" + V(3) + "))";
      set("(((" + x + " % " + y + ") + " + y + ") % " + y + ")");
      return true;
    }
    case OpFRem: set(math("fmod", V(2) + ", " + V(3))); return true;
    case OpFMod: {
      const std::string x = V(2), y = V(3);
      set("(" + x + " - " + y + " * " + math("floor", x + " / " + y) + ")");
      return true;
    }
    case OpSNegate: set("-" + V(2)); return true;
    case OpFNegate: set("-" + V(2)); return true;

    case OpShiftLeftLogical: set("(" + V(2) + " << " + V(3) + ")"); return true;
    case OpShiftRightLogical: set("(" + V(2) + " >> " + V(3) + ")"); return true;
    case OpShiftRightArithmetic: {
      std::string ty2;
      ctype(vtype(a[2]), ty2);
      set("((int64_t)(" + V(2) + ") >> " + V(3) + ")");
      return true;
    }
    case OpBitwiseOr: set("(" + V(2) + " | " + V(3) + ")"); return true;
    case OpBitwiseXor: set("(" + V(2) + " ^ " + V(3) + ")"); return true;
    case OpBitwiseAnd: set("(" + V(2) + " & " + V(3) + ")"); return true;
    case OpNot: set("~" + V(2)); return true;
    case OpLogicalAnd: set("(" + V(2) + " && " + V(3) + ")"); return true;
    case OpLogicalOr: set("(" + V(2) + " || " + V(3) + ")"); return true;
    case OpLogicalNot: set("(!" + V(2) + ")"); return true;
    case OpLogicalEqual: set("(" + V(2) + " == " + V(3) + ")"); return true;
    case OpLogicalNotEqual: set("(" + V(2) + " != " + V(3) + ")"); return true;

    case OpBitCount: set("(uint32_t)__builtin_popcountll((uint64_t)(" + V(2) + "))"); return true;
    case OpBitReverse: {
      const uint32_t w = m_.types[vtype(a[2])].width;
      set("weft_bitrev" + std::to_string(w) + "((uint" + std::to_string(w) + "_t)(" + V(2) + "))");
      return true;
    }
    case OpBitFieldInsert: {
      const std::string base = V(2), ins2 = V(3), off = V(4), cnt = V(5);
      set("((" + base + " & ~(((1ull << " + cnt + ") - 1ull) << " + off + ")) | ((" + ins2 +
          " & ((1ull << " + cnt + ") - 1ull)) << " + off + "))");
      return true;
    }
    case OpBitFieldUExtract:
      set("((" + V(2) + " >> " + V(3) + ") & ((1ull << " + V(4) + ") - 1ull))");
      return true;
    case OpBitFieldSExtract: {
      const std::string x = V(2), off = V(3), cnt = V(4);
      set("(int64_t)(((" + x + " >> " + off + ") << (64u - (" + cnt + "))) >> (64u - (" + cnt + ")))");
      return true;
    }

    case OpConvertFToS: {
      const uint32_t w = m_.types[rt].width;
      if (w != 32u && w != 64u) { set("(int" + std::to_string(w) + "_t)" + V(2)); return true; }
      set("weft_f2i" + std::to_string(w) + "(" + V(2) + ")");
      return true;
    }
    case OpConvertFToU: {
      const uint32_t w = m_.types[rt].width;
      if (w != 32u && w != 64u) { set("(uint" + std::to_string(w) + "_t)" + V(2)); return true; }
      set("weft_f2u" + std::to_string(w) + "(" + V(2) + ")");
      return true;
    }
    case OpConvertSToF: case OpConvertUToF: set("(" + rty + ")" + V(2)); return true;
    case OpUConvert: case OpSConvert: case OpFConvert: {
      uint32_t en = 0, n = 0;
      if (vector_of(rt, en, n) && !is_vec(vtype(a[2]))) {
        std::string et;
        if (!ctype(en, et)) { err = "unsupported element type in a vector convert"; return false; }
        std::string e = rty + "{";
        for (uint32_t i = 0; i < n; i++) {
          if (i) e += ", ";
          e += "(" + et + ")(" + V(2) + ")";
        }
        set(e + "}");
        return true;
      }
      set("(" + rty + ")" + V(2));
      return true;
    }
    case OpSatConvertSToU:
      set("(uint64_t)std::clamp<int64_t>((int64_t)(" + V(2) + "), 0, (int64_t)~(uint64_t)0 >> 1)");
      return true;
    case OpSatConvertUToS:
      set("(int64_t)std::min<uint64_t>((uint64_t)(" + V(2) + "), (uint64_t)~(uint64_t)0 >> 1)");
      return true;
    case OpBitcast: set("weft_bitcast<" + rty + ">(" + V(2) + ")"); return true;

    case OpIEqual: set("(" + V(2) + " == " + V(3) + ")"); return true;
    case OpINotEqual: set("(" + V(2) + " != " + V(3) + ")"); return true;
    case OpUGreaterThan: set("((uint64_t)(" + V(2) + ") > (uint64_t)(" + V(3) + "))"); return true;
    case OpSGreaterThan: set("((int64_t)(" + V(2) + ") > (int64_t)(" + V(3) + "))"); return true;
    case OpUGreaterThanEqual: set("((uint64_t)(" + V(2) + ") >= (uint64_t)(" + V(3) + "))"); return true;
    case OpSGreaterThanEqual: set("((int64_t)(" + V(2) + ") >= (int64_t)(" + V(3) + "))"); return true;
    case OpULessThan: set("((uint64_t)(" + V(2) + ") < (uint64_t)(" + V(3) + "))"); return true;
    case OpSLessThan: set("((int64_t)(" + V(2) + ") < (int64_t)(" + V(3) + "))"); return true;
    case OpULessThanEqual: set("((uint64_t)(" + V(2) + ") <= (uint64_t)(" + V(3) + "))"); return true;
    case OpSLessThanEqual: set("((int64_t)(" + V(2) + ") <= (int64_t)(" + V(3) + "))"); return true;
    // Ordered comparisons are false for NaN, unordered ones are their negation.
    case OpFOrdEqual: set("(" + V(2) + " == " + V(3) + ")"); return true;
    case OpFUnordEqual: set("(!" + V(2) + " != " + V(3) + ")"); return true;
    case OpFOrdNotEqual: set("(" + V(2) + " != " + V(3) + ")"); return true;
    case OpFUnordNotEqual: set("(!" + V(2) + " == " + V(3) + ")"); return true;
    case OpFOrdLessThan: set("(" + V(2) + " < " + V(3) + ")"); return true;
    case OpFUnordLessThan: set("(!" + V(2) + " >= " + V(3) + ")"); return true;
    case OpFOrdGreaterThan: set("(" + V(2) + " > " + V(3) + ")"); return true;
    case OpFUnordGreaterThan: set("(!" + V(2) + " <= " + V(3) + ")"); return true;
    case OpFOrdLessThanEqual: set("(" + V(2) + " <= " + V(3) + ")"); return true;
    case OpFUnordLessThanEqual: set("(!" + V(2) + " > " + V(3) + ")"); return true;
    case OpFOrdGreaterThanEqual: set("(" + V(2) + " >= " + V(3) + ")"); return true;
    case OpFUnordGreaterThanEqual: set("(!" + V(2) + " < " + V(3) + ")"); return true;
    case OpIsNan: set(math("isnan", V(2))); return true;
    case OpIsInf: set(math("isinf", V(2))); return true;
    case OpIsFinite: set(math("isfinite", V(2))); return true;
    case OpSignBitSet: set(math("signbit", V(2))); return true;

    case OpSelect: {
      uint32_t cet = 0, cn = 0;
      if (vector_of(vtype(a[2]), cet, cn) && cn >= 2) {
        auto ot = m_.types.find(rt);
        if (ot == m_.types.end() || ot->second.kind != KVector) {
          err = "a vector select of something that is not a vector";
          return false;
        }
        static const char* kSelComp[] = {"x", "y", "z", "w"};
        const std::string c = V(2), x = V(3), y = V(4);
        std::string e = rty + "{";
        for (uint32_t k = 0; k < ot->second.count && k < 4; k++) {
          if (k) e += ", ";
          e += "(" + c + ")." + kSelComp[k] + " ? (" + x + ")." + kSelComp[k] + " : (" + y +
               ")." + kSelComp[k];
        }
        set(e + "}");
        return true;
      }
      set("(" + V(2) + " ? " + V(3) + " : " + V(4) + ")");
      return true;
    }

    case OpLabel: return true;  // emitted by run()

    case OpBranch: {
      body_ += "  " + edge_copies(cur_block_, a[0]) + jump(label_of_[a[0]]) + "\n";
      return true;
    }
    case OpBranchConditional: {
      const std::string c = V(0);
      body_ += "  if (" + c + ") { " + edge_copies(cur_block_, a[1]) + jump(label_of_[a[1]]) +
               " } else { " + edge_copies(cur_block_, a[2]) + jump(label_of_[a[2]]) + " }\n";
      return true;
    }
    case OpSwitch: {
      if (a.size() < 2) { err = "a switch with no default label"; return false; }
      auto st = m_.types.find(vtype(a[0]));
      if (st == m_.types.end() || st->second.kind != KInt) {
        err = "a switch on something that is not an integer";
        return false;
      }
      const uint32_t w = st->second.width;
      const bool wide = w == 64;
      const size_t step = wide ? 3u : 2u;
      const uint64_t wmask = (w >= 64) ? ~0ull : ((1ull << w) - 1ull);
      if ((a.size() - 2u) % step != 0u) {
        err = "a switch whose literal list does not fill its instruction";
        return false;
      }
      auto known = [&](uint32_t l) {
        if (label_of_.count(l)) return true;
        err = "a switch to label " + std::to_string(l) + ", which this function does not define";
        return false;
      };
      if (!known(a[1])) return false;
      body_ += "  {\n    const uint64_t weft_sel = ((uint64_t)(" + V(0) + ")) & " +
               std::to_string(wmask) + "ull;\n";
      bool first = true;
      for (size_t k = 2; k + step - 1 < a.size(); k += step) {
        uint64_t lit = (wide ? ((uint64_t)a[k] | ((uint64_t)a[k + 1] << 32)) : (uint64_t)a[k]) & wmask;
        const uint32_t target = a[k + step - 1];
        if (!known(target)) return false;
        body_ += "    " + std::string(first ? "if" : "else if") + " (weft_sel == " +
                 std::to_string(lit) + "ull) { " + edge_copies(cur_block_, target) +
                 jump(label_of_[target]) + " }\n";
        first = false;
      }
      body_ += "    else { " + edge_copies(cur_block_, a[1]) + jump(label_of_[a[1]]) + " }\n  }\n";
      return true;
    }
    case OpReturn: case OpReturnValue: case OpKill: case OpUnreachable:
      if (warp_step_) body_ += "  { s->pc[l] = 0xFFFFFFFFu; continue; }\n";
      else if (warp_)
        body_ += warp_blocks_ ? "  { s->pc = 0xFFFFFFFFu; continue; }\n" : "  continue;\n";
      else
        body_ += "  s->pc = 0xFFFFFFFFu; return;\n";
      return true;

    case OpControlBarrier: {
      barrier_ = true;
      if (warp_step_) {
        body_ += "  " + park(resume_next_++) + "\n";
        return true;
      }
      if (gpu) {
        body_ += "  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);\n";
        return true;
      }
      const int id = resume_next_++;
      cases_ += "    case " + std::to_string(id) + ": goto R" + std::to_string(id) + ";\n";
      body_ += "  s->pc = " + std::to_string(id) + "u; return;\n  R" + std::to_string(id) + ":;\n";
      return true;
    }

    case OpGroupNonUniformElect: {
      subgroup_ = true;
      body_ += "  s->" + field(rd) + " = ((ctx.lid & 31u) == 0u);\n";
      return true;
    }

    case OpGroupNonUniformBallot: case OpGroupNonUniformAll: case OpGroupNonUniformAny:
    case OpGroupNonUniformAllEqual: case OpGroupNonUniformBroadcast:
    case OpGroupNonUniformBroadcastFirst: case OpGroupNonUniformShuffle:
    case OpGroupNonUniformShuffleXor: case OpGroupNonUniformShuffleUp:
    case OpGroupNonUniformShuffleDown:
    case OpGroupNonUniformQuadBroadcast: case OpGroupNonUniformQuadSwap: {
      subgroup_ = true;
      std::string payload;
      const bool is_ballot = op == OpGroupNonUniformBallot;
      if (is_ballot || op == OpGroupNonUniformAll || op == OpGroupNonUniformAny)
        payload = "(" + V(3) + ") ? 1u : 0u";

      if (op != OpGroupNonUniformBallot && op != OpGroupNonUniformAll &&
          op != OpGroupNonUniformAny && op != OpGroupNonUniformAllEqual) {
        std::string src;
        if (op == OpGroupNonUniformBroadcast || op == OpGroupNonUniformBroadcastFirst)
          src = op == OpGroupNonUniformBroadcastFirst ? "0u" : "(" + V(4) + ") & 31u";
        else if (op == OpGroupNonUniformShuffle) src = "(" + V(4) + ") & 31u";
        else if (op == OpGroupNonUniformQuadBroadcast) src = "(" + V(4) + ") & 3u";
        else if (op == OpGroupNonUniformQuadSwap)
          src = "((ctx.lid & 3u) ^ (((" + V(4) + ") == 2u) ? 3u : (1u << ((" + V(4) +
                ") & 3u))))";
        else if (op == OpGroupNonUniformShuffleXor) src = "((ctx.lid & 31u) ^ (" + V(4) + ")) & 31u";
        else if (op == OpGroupNonUniformShuffleUp)
          src = "((ctx.lid & 31u) >= (" + V(4) + ") ? (ctx.lid & 31u) - (" + V(4) + ") : (ctx.lid & 31u))";
        else
          src = "((ctx.lid & 31u) + (" + V(4) + ") < 32u ? (ctx.lid & 31u) + (" + V(4) + ") : (ctx.lid & 31u))";
        const bool quad =
            op == OpGroupNonUniformQuadBroadcast || op == OpGroupNonUniformQuadSwap;
        const std::string base = quad ? "(ctx.lid & ~3u)" : "(ctx.lid & ~31u)";
        std::string read;
        if (!wide_shfl(rt, V(3), base + " + (" + src + ")", read)) return false;
        set(read);
        return true;
      }

      const std::string live =
          "const uint32_t weft_live = weft::live_threads(ctx.tgid, ctx.tg_size, ctx.grid); "
          "const uint32_t weft_wb = ctx.lid & ~31u; "
          "const uint32_t weft_wl = (weft_live > weft_wb) ? (weft_live - weft_wb > 32u ? 32u : weft_live - weft_wb) : 0u; "
          "const uint32_t weft_mask = (weft_wl >= 32u) ? 0xFFFFFFFFu : ((1u << weft_wl) - 1u); ";

      if (warp_step_) {
        const std::string wmask = slot_mask();
        if (op == OpGroupNonUniformAllEqual) {
          uint32_t sz = 0;
          if (!tsize(vtype(a[3]), sz)) { err = "a vote on a value of no size"; return false; }
          uint32_t en = 0, nn = 0;
          const bool vec = vector_of(vtype(a[3]), en, nn);
          const auto& et = m_.types[vec ? en : vtype(a[3])];
          const int csize = (int)(et.width / 8);
          if (csize != 4 && csize != 8) {
            err = "a vote on a component that is neither 32 nor 64 bits";
            return false;
          }
          const int comps = vec ? (int)nn : 1;
          const int words = comps * csize / 4;
          if (sz != (uint32_t)(comps * csize)) {
            err = "a vote on a value whose size is not a whole number of components";
            return false;
          }
          const bool flt = et.kind == KFloat;
          const std::string acc = "weft_eq" + std::to_string(rd);
          decls_ += "  uint32_t " + acc + "[32];\n";
          const std::string accref = "s->" + acc + "[l]";
          body_ += "  " + accref + " = 1u;\n";
          for (int w = 0; w < words; w++) {
            const int ci = w / (csize / 4);
            const int half = w % (csize / 4);
            std::string comp = V(3);
            if (vec) comp = "(" + V(3) + ")." + std::string(1, "xyzw"[ci]);
            std::string bits;
            if (csize == 4)
              bits = flt ? ("weft_bitcast<uint32_t>(" + comp + ")") : ("(uint32_t)(" + comp + ")");
            else if (flt)
              bits = half == 0
                         ? ("(uint32_t)(weft_bitcast<uint64_t>(" + comp + ") & 0xFFFFFFFFu)")
                         : ("(uint32_t)(weft_bitcast<uint64_t>(" + comp + ") >> 32)");
            else
              bits = half == 0 ? ("(uint32_t)((uint64_t)(" + comp + ") & 0xFFFFFFFFu)")
                               : ("(uint32_t)((uint64_t)(" + comp + ") >> 32)");
            const int off = slot_off_;
            slot_off_ += 32;
            body_ += "  weft_slots[" + std::to_string(off) + " + l] = " + bits + "; " +
                     park((uint32_t)resume_next_++) + "\n";
            body_ += "  " + accref + " &= weft::slot_all_equal(weft_slots + " +
                     std::to_string(off) + ", " + wmask + ");\n";
          }
          body_ += "  " + fref(rd) + " = (" + accref + " != 0u);\n";
          return true;
        }
        const int off = slot_off_;
        slot_off_ += 32;
        body_ += "  weft_slots[" + std::to_string(off) + " + l] = " + payload + "; " +
                 park((uint32_t)resume_next_++) + "\n";
        uint32_t en = 0, nn = 0;
        const bool wide = is_ballot && vector_of(rt, en, nn) && nn == 4;
        const std::string b = "(weft::slot_ballot(weft_slots + " + std::to_string(off) +
                              ", " + wmask + ") & " + wmask + ")";
        std::string e;
        if (is_ballot) {
          e = b;
          if (wide) e = rty + "{" + e + ", 0u, 0u, 0u}";
        } else if (op == OpGroupNonUniformAll) {
          e = "(" + b + " == " + wmask + " && " + wmask + " != 0u)";
        } else {
          e = "(" + b + " != 0u)";
        }
        body_ += "  " + fref(rd) + " = " + e + ";\n";
        return true;
      }

      if (gpu && op == OpGroupNonUniformAllEqual) {
        std::string peer0;
        if (!wide_shfl(vtype(a[3]), V(3), "0u", peer0)) return false;
        uint32_t en = 0, nn = 0;
        const bool vec = vector_of(vtype(a[3]), en, nn);
        const int comps = vec ? (int)nn : 1;
        static const char* kComp[] = {"x", "y", "z", "w"};
        std::string neq;
        for (int k = 0; k < comps; k++) {
          const std::string sfx = vec ? ("." + std::string(1, kComp[k][0])) : std::string();
          neq += std::string(k ? " || " : "") + "((" + V(3) + ")" + sfx + " != (" + peer0 + ")" +
                 sfx + ")";
        }
        body_ += "  { " + live + "const uint32_t weft_b = ((uint32_t)(simd_vote::vote_t)simd_ballot(" +
                 neq + ")) & weft_mask;\n    s->" + field(rd) + " = (weft_b == 0u); }\n";
        return true;
      }

      if (op == OpGroupNonUniformAllEqual) {
        uint32_t eqsz = 0;
        if (!tsize(vtype(a[3]), eqsz)) { err = "a vote on a value of no size"; return false; }
        std::string eqty;
        if (!ctype(vtype(a[3]), eqty)) { err = "a vote on a value of no type"; return false; }
        const bool eq_wide = eqsz > 4;
        if (eq_wide && eqsz > 16) {
          err = "a vote on a value wider than the sixteen-byte relay slot";
          return false;
        }
        const int ae = resume_next_++;
        cases_ += "    case " + std::to_string(ae) + ": goto R" + std::to_string(ae) + ";\n";
        if (eq_wide) {
          const std::string n = std::to_string(eqsz);
          body_ += "  { const " + eqty + " weft_v = " + V(3) +
                   "; std::memcpy(weft::wide_pub_slot(ctx), &weft_v, " + n + "); }\n"
                   "  s->pc = " +
                   std::to_string(ae) + "u; return;\n  R" + std::to_string(ae) + ":;\n";
          body_ += "  { " + live +
                   " const unsigned char* weft_me = weft::wide_read_slot(ctx, ctx.lid);\n"
                   "    bool weft_same = true;\n"
                   "    for (uint32_t weft_i = 0; weft_i < 32u; weft_i++) {\n"
                   "      if ((weft_mask & (1u << weft_i)) == 0u) continue;\n"
                   "      if (std::memcmp(weft_me, weft::wide_read_slot(ctx, weft_wb + weft_i), " +
                   n +
                   ") != 0) weft_same = false;\n"
                   "    }\n"
                   "    s->" +
                   field(rd) + " = weft_same; }\n";
          return true;
        }
        if (!shfl_pub(vtype(a[3]), V(3), payload)) return false;
        body_ += "  ((uint32_t*)weft::shfl_pub_buf(ctx))[ctx.lid] = " + payload +
                 "; s->pc = " + std::to_string(ae) + "u; return;\n  R" +
                 std::to_string(ae) + ":;\n";
        body_ += "  { " + live +
                 " const uint32_t* weft_rb = (const uint32_t*)weft::shfl_read_buf(ctx);\n"
                 "    const uint32_t weft_me = weft_rb[ctx.lid];\n"
                 "    bool weft_same = true;\n"
                 "    for (uint32_t weft_i = 0; weft_i < 32u; weft_i++)\n"
                 "      if ((weft_mask & (1u << weft_i)) && weft_rb[weft_wb + weft_i] != weft_me) weft_same = false;\n"
                 "    s->" + field(rd) + " = weft_same; }\n";
        return true;
      }

      {
        if (gpu) {
          // One ballot instruction, masked to the lanes that exist.
          const std::string b = "((uint32_t)(simd_vote::vote_t)simd_ballot(" + V(3) + "))";
          uint32_t en = 0, nn = 0;
          const bool wide = is_ballot && vector_of(rt, en, nn) && nn == 4;
          std::string e;
          if (is_ballot) {
            e = "(" + b + " & weft_mask)";
            if (wide) e = rty + "{" + e + ", 0u, 0u, 0u}";
          } else if (op == OpGroupNonUniformAll)
            e = "((" + b + " & weft_mask) == weft_mask && weft_mask != 0u)";
          else e = "((" + b + " & weft_mask) != 0u)";
          body_ += "  { " + live + "s->" + field(rd) + " = " + e + "; }\n";
          return true;
        }
        const int id = resume_next_++;
        cases_ += "    case " + std::to_string(id) + ": goto R" + std::to_string(id) + ";\n";
        const std::string label = "  R" + std::to_string(id) + ":;\n";
        body_ += "  ((uint32_t*)weft::shfl_pub_buf(ctx))[ctx.lid] = " + payload +
                 "; s->pc = " + std::to_string(id) + "u; return;\n" + label;
        std::string e;
        uint32_t en = 0, nn = 0;
        const bool wide = is_ballot && vector_of(rt, en, nn) && nn == 4;
        if (is_ballot) {
          e = "(weft::vote_ballot(ctx) & weft_mask)";
          if (wide) e = rty + "{" + e + ", 0u, 0u, 0u}";
        } else if (op == OpGroupNonUniformAll)
          e = "((weft::vote_ballot(ctx) & weft_mask) == weft_mask && weft_mask != 0u)";
        else e = "((weft::vote_ballot(ctx) & weft_mask) != 0u)";
        body_ += "  { " + live + "s->" + field(rd) + " = " + e + "; }\n";
        return true;
      }

    }

    case OpGroupNonUniformBallotBitCount: {
      const std::string m = "weft_bitcast<uint32_t>((" + V(3) + ").x)";
      const uint32_t go = a.size() > 4 ? a[4] : 0u;
      body_ += "  { const uint32_t weft_bm = " + m + ";\n"
               "    const uint32_t weft_lane = ctx.lid & 31u;\n"
               "    if (weft_lane >= 32u) { s->" + field(rd) + " = 0; }\n"
               "    else {\n";
      if (go == 0)
        body_ += "      s->" + field(rd) + " = (" + rty + ")__builtin_popcount(weft_bm);\n";
      else if (go == 1)
        body_ += "      const uint32_t weft_m = weft_bm & ((weft_lane + 1u >= 32u) ? 0xFFFFFFFFu : "
                 "((1u << (weft_lane + 1u)) - 1u));\n"
                 "      s->" + field(rd) + " = (" + rty + ")__builtin_popcount(weft_m);\n";
      else
        body_ += "      const uint32_t weft_m = weft_lane ? (weft_bm & ((1u << weft_lane) - 1u)) : 0u;\n"
                 "      s->" + field(rd) + " = (" + rty + ")__builtin_popcount(weft_m);\n";
      body_ += "    } }\n";
      return true;
    }

    case OpAtomicLoad: {
      if (gpu && m_.types[vtype(rd)].width == 64) {
        if (!warp_) {
          err = "this host's Metal has no 64-bit atomic, so a 64-bit atomic load has no "
                "GPU form";
          return false;
        }
        wide_atom = true;
        const bool sgn = m_.types[vtype(rd)].is_signed;
        set("weft::load64((device " + std::string(sgn ? "int64_t" : "uint64_t") +
            "*)(uintptr_t)(" + addr(a[2]) + "))");
        return true;
      }
      set("weft::atomic_load<" + rty + ">((" + addr_space("device", false) + "void*)" +
          addr(a[2]) + ")");
      return true;
    }
    case OpAtomicStore: {
      if (gpu && m_.types[vtype(a[3])].width == 64) {
        if (!warp_) {
          err = "this host's Metal has no 64-bit atomic, so a 64-bit atomic store has "
                "no GPU form";
          return false;
        }
        wide_atom = true;
        const bool sgn = m_.types[vtype(a[3])].is_signed;
        const std::string t64 = sgn ? "int64_t" : "uint64_t";
        body_ += "  weft::store64((device " + t64 + "*)(uintptr_t)(" + addr(a[0]) + "), (" +
                 t64 + ")(" + V(3) + "));\n";
        return true;
      }
      body_ += "  weft::atomic_store((" + addr_space("device", false) + "void*)" + addr(a[0]) +
               ", " + V(3) + ");\n";
      return true;
    }
    case OpAtomicExchange: case OpAtomicCompareExchange: case OpAtomicCompareExchangeWeak:
    case OpAtomicIIncrement: case OpAtomicIDecrement: case OpAtomicIAdd: case OpAtomicISub:
    case OpAtomicSMin: case OpAtomicUMin: case OpAtomicSMax: case OpAtomicUMax: case OpAtomicAnd:
    case OpAtomicOr: case OpAtomicXor: case OpAtomicFAddEXT:
    case OpAtomicFMinEXT: case OpAtomicFMaxEXT: {
      const std::string ptr = addr(a[2]);
      // a[3] scope, a[4] semantics, then the value(s)
      std::string fn, val_, cmp;
      const std::string is64 = std::to_string((int)m_.types[vtype(rd)].width);
      (void)is64;
      const bool sgn = m_.types[vtype(rd)].is_signed;
      const bool wide = m_.types[vtype(rd)].width == 64;
      const bool flt = m_.types[vtype(rd)].kind == KFloat;
      // A 64-bit atomic on the GPU.
      const bool dropped = unused_anywhere(rd);
      const bool cas = op == OpAtomicCompareExchange || op == OpAtomicCompareExchangeWeak;
      if (op == OpAtomicIIncrement || op == OpAtomicIDecrement) val_ = "1";
      else if (cas) val_ = V(a.size() - 2);
      else val_ = V(5);
      if (cas) cmp = V(a.size() - 1);

      std::string suf = flt ? "" : (wide ? (sgn ? "_s64" : "_u64") : (sgn ? "_s32" : "_u32"));
      if (flt) suf = "_f32";
      switch (op) {
        case OpAtomicIAdd: fn = "atom_add"; break;
        case OpAtomicISub: fn = "atom_sub"; break;
        case OpAtomicSMin: case OpAtomicUMin: fn = "atom_min"; break;
        case OpAtomicSMax: case OpAtomicUMax: fn = "atom_max"; break;
        case OpAtomicAnd: fn = "atom_and"; break;
        case OpAtomicOr: fn = "atom_or"; break;
        case OpAtomicXor: fn = "atom_xor"; break;
        case OpAtomicExchange: fn = "atom_exch"; break;
        case OpAtomicCompareExchange: case OpAtomicCompareExchangeWeak: fn = "atom_cas"; break;
        case OpAtomicIIncrement: case OpAtomicIDecrement: fn = "atom_add"; break;
        case OpAtomicFAddEXT: fn = "atom_add"; suf = "_f32"; break;
        case OpAtomicFMinEXT: fn = "atom_min"; suf = "_f32"; break;
        case OpAtomicFMaxEXT: fn = "atom_max"; suf = "_f32"; break;
        default: break;
      }
      std::string tyop;
      if (!ctype(vtype(rd), tyop)) { err = "atomic of an unsupported type"; return false; }
      if (gpu && wide) {
        const bool order_free = op == OpAtomicIAdd || op == OpAtomicISub || op == OpAtomicAnd ||
                                op == OpAtomicOr || op == OpAtomicXor;
        if (dropped && order_free) {
          const std::string rn = op == OpAtomicIAdd   ? "add"
                                 : op == OpAtomicISub ? "sub"
                                 : op == OpAtomicAnd  ? "and"
                                 : op == OpAtomicOr   ? "or"
                                                      : "xor";
          const std::string addr = "((device uint64_t*)(uintptr_t)(" + ptr + "))";
          const bool invariant = ptr.find("ctx.lid") == std::string::npos &&
                                 ptr.find("ctx.gid") == std::string::npos &&
                                 ptr.find("[l]") == std::string::npos;
          if (invariant && pointer_used_once(a[2])) {
            std::string name;
            for (const auto& a : red_accs)
              if (a.addr == addr && a.op == rn) name = a.name;
            if (name.empty()) {
              name = "weft_red_acc" + std::to_string(red_accs.size());
              red_accs.push_back(RedAcc{addr, rn, name});
            }
            const char* foldop = rn == "add"   ? "+"
                                 : rn == "sub" ? "-"
                                 : rn == "and" ? "&"
                                 : rn == "or"  ? "|"
                                               : "^";
            body_ += "  " + name + " = " + name + " " + foldop + " (uint64_t)(" + val_ +
                     ");\n";
            return true;
          }
          body_ += "  (void)weft::red_" + rn + "_u64(" + addr + ", (uint64_t)(" + val_ +
                   "));\n";
          return true;
        }
        // Refused, and the two reasons are separate measurements.
        if (warp_) {
          wide_atom = true;
          const std::string t64 = sgn ? "int64_t" : "uint64_t";
          const std::string ptr64 =
              "(" + addr_space("device", false) + t64 + "*)(uintptr_t)(" + ptr + ")";
          std::string name64 = "add";
          std::string v64 = "(" + t64 + ")(" + val_ + ")";
          switch (op) {
            case OpAtomicIAdd: case OpAtomicIIncrement: name64 = "add"; break;
            case OpAtomicISub: case OpAtomicIDecrement:
              if (sgn) { name64 = "add"; v64 = "(-" + v64 + ")"; }
              else name64 = "sub";
              break;
            case OpAtomicSMin: case OpAtomicUMin: name64 = "min"; break;
            case OpAtomicSMax: case OpAtomicUMax: name64 = "max"; break;
            case OpAtomicAnd: name64 = "and"; break;
            case OpAtomicOr: name64 = "or"; break;
            case OpAtomicXor: name64 = "xor"; break;
            case OpAtomicExchange: name64 = "exch"; break;
            case OpAtomicCompareExchange: case OpAtomicCompareExchangeWeak: name64 = "cas"; break;
            default: break;
          }
          std::string call64;
          if (name64 == "cas")
            call64 = "weft::rmw_cas_solo(" + ptr64 + ", (" + t64 + ")(" + cmp + "), " + v64 +
                      ")";
          else call64 = "weft::rmw_" + name64 + "_solo(" + ptr64 + ", " + v64 + ")";
          if (name64 != "cas") {
            const size_t bat = ptr64.find("bufs[");
            const size_t bcl = bat == std::string::npos ? std::string::npos : ptr64.find(']', bat);
            if (bcl != std::string::npos) {
              const std::string sh = "weft_sh" + std::to_string(shadows.size());
              ShadowSite site;
              site.buf = ptr64.substr(bat, bcl - bat + 1);
              site.call = call64;
              site.shadow = "weft::sh64_" + name64 + "(" + sh + ", " + v64 + ")";
              site.decl = std::string(sgn ? "int64_t " : "uint64_t ") + sh + " = *((device " +
                          t64 + "*)(uintptr_t)(" + ptr + "));";
              site.flush = "*((device " + t64 + "*)(uintptr_t)(" + ptr + ")) = " + sh + ";";
              shadows.push_back(site);
            }
          }
          if (dropped) body_ += "  (void)" + call64 + ";\n";
          else set(call64);
          return true;
        }
        err = "a 64-bit atomic that reads the old pair back moves both words as one "
              "value, and only one writer can do that here: lanes in a warp cannot "
              "wait for each other (a warp has one program counter), so the per-lane "
              "back end cannot host the section. The warp target can, by walking "
              "every warp in one thread, but this kernel " + warp_why_;
        return false;      }
      if (op == OpAtomicISub && (sgn || flt)) {
        set("weft::atom_add" + suf + "((" + addr_space("device", false) + tyop +
            "*)(" + addr_space("device", false) + "void*)(" + ptr + "), -(" + val_ + "))");
        return true;
      }
      if (flt && fn == "atom_exch") {
        const std::string w = addr_space("device", false) + "uint32_t*";
        set("weft_bitcast<float>(weft::atom_exch_u32((" + w + ")(" +
            addr_space("device", false) + "void*)(" + ptr + "), weft_bitcast<uint32_t>(" +
            val_ + ")))");
        return true;
      }
      if (flt && fn != "atom_add" && fn != "atom_min" && fn != "atom_max") {
        err = "float atomic compare-exchange does not exist in SPIR-V: its result is "
              "constrained to an integer scalar";
        return false;
      }
      const unsigned tw = m_.types[vtype(rd)].width;
      if (tw == 16) {
        const std::string as = addr_space("device", false);
        const std::string word =
            "(" + as + "uint32_t*)(uintptr_t)((uintptr_t)(" + ptr + ") & ~3ull)";
        const std::string shift = "(uint32_t)(((uintptr_t)(" + ptr + ") & 2ull) * 8u)";
        std::string inner;
        if (flt) {
          if (op != OpAtomicFAddEXT) {
            err = "a 16-bit float atomic is an add: only OpAtomicFAddEXT is named at "
                  "that width, and the 16-bit min and max extensions are the 32-bit ones";
            return false;
          }
          inner = "weft_f16_of(weft_atom_add_narrow(" + word + ", " + shift +
                  ", weft_f16_bits(" + val_ + "), 0))";
        } else {
          int code = -1;
          switch (op) {
            case OpAtomicIAdd: code = 0; break;
            case OpAtomicISub: code = 1; break;
            case OpAtomicUMin: code = 2; break;
            case OpAtomicUMax: code = 3; break;
            case OpAtomicSMin: code = 4; break;
            case OpAtomicSMax: code = 5; break;
            case OpAtomicAnd: code = 6; break;
            case OpAtomicOr: code = 7; break;
            case OpAtomicXor: code = 8; break;
            case OpAtomicExchange: code = 9; break;
            case OpAtomicCompareExchange: case OpAtomicCompareExchangeWeak: code = 10; break;
            default: break;
          }
          if (code < 0) { err = "unsupported atomic at 16 bits: " + fn; return false; }
          inner = "weft_atom_int16(" + word + ", " + shift + ", (uint16_t)(" + val_ + "), " +
                  (code == 10 ? ("(uint16_t)(" + (cmp.empty() ? std::string("0") : cmp) + ")")
                              : std::string("0")) +
                  ", " + std::to_string(code) + ")";
        }
        set((sgn && !flt) ? ("(int16_t)(" + inner + ")") : inner);
        return true;
      }
      std::string call = "weft::" + fn + suf + "((" + addr_space("device", false) + tyop +
                         "*)(" + addr_space("device", false) + "void*)(" + ptr + "), " + val_;
      if (!cmp.empty()) call += ", " + cmp;
      call += ")";
      set(call);
      return true;
    }

    case OpSampledImage: {
      auto st = m_.types.find(vtype(a[2]));
      const bool img = gpu && st != m_.types.end() && st->second.kind == KImage;
      const bool arr = img && st->second.arrayed != 0;
      const bool vol = img && !arr && st->second.count == 2u;
      const bool cb = img && st->second.count == 3u;
      set(std::string(cb ? "weft::SampledCubeImage{"
                          : vol ? "weft::SampledImage3D{"
                                : arr ? "weft::SampledArrayImage{"
                                      : "weft::SampledImage{") +
          V(2) + ", " + V(3) + "}");
      return true;
    }

    case OpImage:  // the image out of a sampled image
      set("(" + V(2) + ").tex");
      return true;

    case OpImageQuerySize: case OpImageQuerySizeLod: {
      const std::string img = V(2);
      uint32_t et = 0, n = 0;
      std::string ety = "uint32_t";
      if (vector_of(a[0], et, n) && !ctype(et, ety)) {
        err = "image size of an unsupported element type";
        return false;
      }
      const std::string dot = gpu ? "." : "->";
      const std::string w = "(" + ety + ")(" + img + ")" + dot + "width";
      const std::string h = "(" + ety + ")(" + img + ")" + dot + "height";
      // A 2-D array reports its slice count as the third component.
      auto it = m_.types.find(vtype(a[2]));
      const bool arr = it != m_.types.end() && it->second.arrayed != 0;
      const bool vol = it != m_.types.end() && it->second.kind == KImage && it->second.count == 2u;
      const std::string field = (gpu && vol && !arr) ? "depth" : "layers";
      const bool cb = it != m_.types.end() && it->second.kind == KImage && it->second.count == 3u;
      const std::string l = cb ? ("(" + ety + ")6")
                               : (arr || vol)
                                     ? ("(" + ety + ")(" + img + ")" + dot + field)
                                     : ("(" + ety + ")1");
      if (n == 3) set(rty + "{" + w + ", " + h + ", " + l + "}");
      else if (n == 2) set(rty + "{" + w + ", " + h + "}");
      else set(w);
      return true;
    }

    case OpImageGather: case OpImageRead: case OpImageFetch: case OpImageSampleImplicitLod:
    case OpImageSampleExplicitLod: {
      const bool gather = op == OpImageGather;
      const bool sample = op == OpImageSampleImplicitLod || op == OpImageSampleExplicitLod ||
                          gather;
      const std::string src = V(2);
      const std::string img = sample ? ("(" + src + ").tex") : src;
      // What a read hands back is the image's sampled element type.
      uint32_t img_tid = vtype(a[2]);
      if (sample) {
        auto st = m_.types.find(img_tid);
        if (st == m_.types.end() || st->second.kind != KSampledImage) {
          err = "sample of something that is not a sampled image";
          return false;
        }
        img_tid = st->second.elem;
      }
      auto it = m_.types.find(img_tid);
      if (it == m_.types.end() || it->second.kind != KImage) {
        err = "read of something that is not an image";
        return false;
      }
      std::string ety;
      if (!ctype(it->second.elem, ety)) { err = "image of an unsupported element type"; return false; }

      std::string x, y, lay, arrw;
      {
        uint32_t cet = 0, cn = 0;
        const std::string c = V(3);
        if (vector_of(vtype(a[3]), cet, cn) && cn >= 2) {
          x = "(" + c + ").x";
          y = "(" + c + ").y";
          // The third component of an arrayed coordinate is the slice.
          if (cn >= 3) lay = "(" + c + ").z";
          // A cube array names its slice after all three direction components.
          if (cn >= 4) arrw = "(" + c + ").w";
        } else {
          x = "(" + c + ")";
          y = (it->second.count == 0u) ? "0.5f" : "0";
        }
      }
      const bool vol = it->second.arrayed == 0 && it->second.count == 2u;
      const std::string lz = (it->second.arrayed != 0 || vol)
                                 ? (", (uint32_t)(" + (lay.empty() ? std::string("0") : lay) + ")")
                                 : std::string();

      // The optional image operands, in the order the spec fixes.
      std::string ox = "0", oy = "0";
      const size_t mask_at = gather ? 5u : 4u;
      if (a.size() > mask_at) {
        const uint32_t mask = a[mask_at];
        size_t k = mask_at + 1;
        auto skip = [&](size_t count) {
          for (size_t j = 0; j < count && k < a.size(); j++, k++) {}
        };
        auto offset_of = [&]() {
          if (k >= a.size()) return;
          const std::string c = val(a[k]);
          uint32_t cet = 0, cn = 0;
          if (vector_of(vtype(a[k]), cet, cn) && cn >= 2) {
            ox = "(" + c + ").x";
            oy = "(" + c + ").y";
          } else {
            ox = "(" + c + ")";
          }
          k++;
        };
        if (mask & 0x1u) skip(1);   // Bias
        if (mask & 0x2u) skip(1);   // Lod
        if (mask & 0x4u) skip(2);   // Grad
        if (mask & 0x8u) offset_of();
        if (mask & 0x10u) offset_of();
        if (mask & 0x20u) skip(1);  // ConstOffsets
        if (mask & 0x40u) skip(1);  // Sample
        if (mask & 0x80u) skip(1);  // MinLod
        if (k != a.size()) { err = "an image operand this front end does not know"; return false; }
      }

      if (gather) {
        const std::string g = "weft_g" + std::to_string(a[1]);
        body_ += "  float " + g + "[4];\n";
        const std::string dx = "(" + x + ") + (float)(" + ox + ") / (float)(" +
                               (gpu ? img : "*" + img) + ").width";
        const std::string dy = "(" + y + ") + (float)(" + oy + ") / (float)(" +
                               (gpu ? img : "*" + img) + ").height";
        const std::string lhs = gpu ? src : ("*" + img);
        const uint32_t dim = it->second.count;
        const bool arr = it->second.arrayed != 0;
        std::string call;
        if (dim == 1u && !arr)
          call = "weft::tex_gather4<float>(" + lhs + ", " + dx + ", " + dy + ", " + g + ")";
        else if (dim == 1u && arr)
          call = "weft::tex_gather4<float>(" + lhs + ", " + dx + ", " + dy + ", (uint32_t)(" +
                 lay + "), " + g + ")";
        else if (dim == 2u) {
          // Not a gap.
          err = "a gather is a 2D, Cube or Rect image in SPIR-V, and a volume is none of them";
          return false;
        } else if (dim == 3u) {
          // A cube gathers by direction, and a cube array also names its layer.
          const std::string tb = "(" + (gpu ? img : "*" + img) + ")";
          const std::string dir = lay.empty() ? std::string("0.0f") : lay;
          const std::string du = "(float)(" + ox + ") / (float)(" + tb + ".width)";
          const std::string dv = "(float)(" + oy + ") / (float)(" + tb + ".height)";
          call = arr ? ("weft::tex_gather4_cube_array_dir<" + ety + ">(" + lhs + ", " + x +
                        ", " + y + ", " + dir + ", (uint32_t)(" +
                        (arrw.empty() ? std::string("0") : arrw) + "), " + du + ", " + dv +
                        ", " + g + ")")
                     : ("weft::tex_gather4_cube_dir<" + ety + ">(" + lhs + ", " + x + ", " +
                        y + ", " + dir + ", " + du + ", " + dv + ", " + g + ")");
        } else {
          err = "a gather of an image this front end cannot name";
          return false;
        }
        body_ += "  " + call + ";\n";
        set(rty + "{" + g + "[0], " + g + "[1], " + g + "[2], " + g + "[3]}");
        return true;
      }
      std::string read;
      if (sample && it->second.count == 3u) {
        if (lay.empty()) {
          err = "a cube sample needs a three-component direction";
          return false;
        }
        read = gpu ? ("weft::tex_sample<" + ety + ">(" + src + ", " + x + ", " + y + ", " +
                      lay + ")")
                   : ("weft::tex_sample_cube<" + ety + ">(*" + img + ", " + x + ", " + y + ", " +
                      lay + ", (" + src + ").s.filter, (" + src + ").s.addr)");
      } else if (sample) {
        std::string u = x, v = y;
        if (ox != "0" || oy != "0") {
          const std::string dot = gpu ? "." : "->";
          u = "(" + x + " + (float)(" + ox + ") / (float)(" + img + ")" + dot + "width)";
          v = "(" + y + " + (float)(" + oy + ") / (float)(" + img + ")" + dot + "height)";
        }
        const std::string third = vol ? (", (float)(" + (lay.empty() ? std::string("0") : lay) + ")")
                                      : lz;
        read = gpu ? ("weft::tex_sample<" + ety + ">(" + src + ", " + u + ", " + v + third + ")")
                   : ("weft::tex_sample<" + ety + ">(*" + img + ", " + u + ", " + v + third +
                      ", (" + src + ").s.filter, (" + src + ").s.addr)");
      } else {
        read = "weft::tex_read<" + ety + ">(" + (gpu ? img : "*" + img) + ", (uint32_t)(" + x +
               " + " + ox + "), (uint32_t)(" + y + " + " + oy + ")" + lz + ")";
      }
      uint32_t ret = 0, rn = 0;
      if (vector_of(a[0], ret, rn) && rn >= 2) {
        const std::string z = "(" + ety + ")0", o = "(" + ety + ")1";
        std::string e = rty + "{" + read + ", " + z + ", " + z + ", " + o + "}";
        if (rn == 2) e = rty + "{" + read + ", " + z + "}";
        else if (rn == 3) e = rty + "{" + read + ", " + z + ", " + z + "}";
        set(e);
      } else {
        set(read);
      }
      return true;
    }

    // A storage image write.
    case OpImageWrite: {
      const std::string img = V(0);
      const std::string coord = V(1);
      const std::string texel = V(2);
      std::string x = "(uint32_t)(" + coord + ").x", y = "(uint32_t)(" + coord + ").y";
      std::string lay;
      uint32_t cet = 0, cn = 0;
      if (!vector_of(vtype(a[1]), cet, cn) || cn < 2) {
        x = "(uint32_t)(" + coord + ")";
        y = "0u";
      } else if (cn >= 3) {
        lay = "(uint32_t)(" + coord + ").z";
      }
      {
        auto wt = m_.types.find(vtype(a[0]));
        const bool vol = wt != m_.types.end() && wt->second.kind == KImage &&
                         wt->second.count == 2u;
        if (wt == m_.types.end() || (wt->second.arrayed == 0 && !vol)) lay.clear();
        else lay = ", " + lay;
      }
      const uint32_t ttype = vtype(a[2]);
      uint32_t elem = ttype, vte = 0, tn = 0;
      if (vector_of(ttype, vte, tn)) elem = vte;
      else tn = 1;
      std::string ety;
      if (!ctype(elem, ety)) { err = "image write of an unsupported element type"; return false; }
      const std::string vt = (ety == "float"     ? "float4"
                              : ety == "int32_t"  ? "int4"
                              : ety == "uint32_t" ? "uint4"
                                                  : "");
      if (vt.empty()) { err = "image write of an unsupported element type"; return false; }
      std::string v = texel;
      if (tn < 4) {
        const std::string z = "(" + ety + ")0";
        v = vt + "{" + texel + ", " + z + ", " + z + ", " + z + "}";
      }
      body_ += "  weft::tex_write(" + (gpu ? img : "*(" + img + ")") + ", " + x + ", " + y +
               lay + ", " + v + ");\n";
      return true;
    }

    case OpGroupNonUniformIAdd: case OpGroupNonUniformFAdd:
    case OpGroupNonUniformIMul: case OpGroupNonUniformFMul:
    case OpGroupNonUniformSMin: case OpGroupNonUniformUMin: case OpGroupNonUniformFMin:
    case OpGroupNonUniformSMax: case OpGroupNonUniformUMax: case OpGroupNonUniformFMax:
    case OpGroupNonUniformBitwiseAnd: case OpGroupNonUniformBitwiseOr:
    case OpGroupNonUniformBitwiseXor:
    case OpGroupNonUniformLogicalAnd: case OpGroupNonUniformLogicalOr:
    case OpGroupNonUniformLogicalXor: {
      if (a.size() < 5) { err = "a subgroup fold is missing its value"; return false; }
      const char* kind =
          (op == OpGroupNonUniformIAdd || op == OpGroupNonUniformFAdd) ? "add"
          : (op == OpGroupNonUniformIMul || op == OpGroupNonUniformFMul) ? "mul"
          : (op == OpGroupNonUniformSMin || op == OpGroupNonUniformUMin ||
             op == OpGroupNonUniformFMin) ? "min"
          : (op == OpGroupNonUniformSMax || op == OpGroupNonUniformUMax ||
             op == OpGroupNonUniformFMax) ? "max"
          : (op == OpGroupNonUniformBitwiseAnd || op == OpGroupNonUniformLogicalAnd)
                ? "and"
          : (op == OpGroupNonUniformBitwiseOr || op == OpGroupNonUniformLogicalOr) ? "or"
                                                                              : "xor";
      const uint32_t action = a[3];
      if (action > 2) {
        err = "a subgroup fold with group operation " + std::to_string(action) +
              ": this reads the form where it is 0, 1 or 2";
        return false;
      }
      if (action > 2) {
        err = "a subgroup fold with group action " + std::to_string(action);
        return false;
      }
      Instr sub = ins;
      sub.a.erase(sub.a.begin() + 3);
      return emit_import(sub, action == 0
                                 ? std::string("sub_group_reduce_") + kind
                                 : std::string("sub_group_scan_") +
                                       (action == 2 ? "exclusive_" : "inclusive_") + kind);
    }

    case OpGroupNonUniformInverseBallot: case OpGroupNonUniformBallotBitExtract:
    case OpGroupNonUniformBallotFindLSB: case OpGroupNonUniformBallotFindMSB: {
      if (a.size() < 4) { err = "a ballot operation is missing its ballot"; return false; }
      std::string bw = V(3);
      uint32_t ben = 0, bnn = 0;
      if (vector_of(vtype(a[3]), ben, bnn) && bnn == 4) bw = "(" + V(3) + ").x";
      if (op == OpGroupNonUniformBallotFindLSB) { set("weft_lsb(" + bw + ")"); return true; }
      if (op == OpGroupNonUniformBallotFindMSB) { set("weft_msb(" + bw + ")"); return true; }
      if (op == OpGroupNonUniformBallotBitExtract) {
        if (a.size() < 5) { err = "a ballot bit extract is missing its index"; return false; }
        set("((((" + bw + ") >> ((" + V(4) + ") & 31u)) & 1u) != 0u)");
        return true;
      }
      set("((((" + bw + ") >> (ctx.lid & 31u)) & 1u) != 0u)");
      return true;
    }

    case OpGroupNonUniformRotateKHR: {
      if (a.size() < 5) { err = "a rotation is missing its value or its delta"; return false; }
      const std::string cl = a.size() > 5 ? ("((uint32_t)(" + V(5) + "))") : std::string("32u");
      const std::string mask = "(" + cl + " - 1u)";
      std::string read;
      if (!wide_shfl(rt, V(3),
                     "((ctx.lid & ~" + mask + ") + ((ctx.lid + (uint32_t)(" + V(4) + ")) & " +
                         mask + "))",
                     read))
        return false;
      set(read);
      return true;
    }

    case OpExtInst: {
      const uint32_t set_id = a[2];
      auto sn = m_.ext_name.find(set_id);
      if (sn == m_.ext_name.end() || sn->second != "OpenCL.std") {
        err = "extended instruction set '" + (sn == m_.ext_name.end() ? std::string("?") : sn->second) +
              "' is not supported";
        return false;
      }
      const uint32_t num = a[3];
      std::string f;
      int arity = 1;
      switch (num) {
        case 0: f = "acos"; break;
        case 1: f = "acosh"; break;
        case 3: f = "asin"; break;
        case 4: f = "asinh"; break;
        case 6: f = "atan"; break;
        case 7: f = "atan2"; arity = 2; break;
        case 8: f = "atanh"; break;
        case 11: f = "cbrt"; break;
        case 12: f = "ceil"; break;
        case 16: f = "cospi"; break;
        case 22: f = "expm1"; break;
        case 41: f = "logb"; break;
        case 43: f = "maxmag"; arity = 2; break;
        case 44: f = "minmag"; arity = 2; break;
        case 47: f = "nextafter"; arity = 2; break;
        case 50: f = "powr"; arity = 2; break;
        case 60: f = "sinpi"; break;
        case 64: f = "tanpi"; break;
        case 13: f = "copysign"; arity = 2; break;
        case 14: f = "cos"; break;
        case 15: f = "cosh"; break;
        case 17: f = "erfc"; break;
        case 18: f = "erf"; break;
        case 19: f = "exp"; break;
        case 20: f = "exp2"; break;
        case 21: f = "exp10"; break;
        case 23: f = "fabs"; break;
        case 24: f = "fdim"; arity = 2; break;
        case 25: f = "floor"; break;
        case 26: f = "fma"; arity = 3; break;
        case 27: f = "fmax"; arity = 2; break;
        case 28: f = "fmin"; arity = 2; break;
        case 29: f = "fmod"; arity = 2; break;
        case 32: f = "hypot"; arity = 2; break;
        case 37: f = "log"; break;
        case 38: f = "log2"; break;
        case 39: f = "log10"; break;
        case 40: f = "log1p"; break;
        case 48: f = "pow"; arity = 2; break;
        case 51: f = "remainder"; arity = 2; break;
        case 53: f = "rint"; break;
        case 55: f = "round"; break;
        case 56: f = "rsqrt"; break;
        case 57: f = "sin"; break;
        case 59: f = "sinh"; break;
        case 61: f = "sqrt"; break;
        case 62: f = "tan"; break;
        case 63: f = "tanh"; break;
        case 65: f = "tgamma"; break;
        case 66: f = "trunc"; break;
        case 67: case 81: f = "cos"; break;
        case 69: case 83: f = "exp"; break;
        case 70: case 84: f = "exp2"; break;
        case 71: case 85: f = "exp10"; break;
        case 72: case 86: f = "log"; break;
        case 73: case 87: f = "log2"; break;
        case 74: case 88: f = "log10"; break;
        case 75: case 89: f = "pow"; arity = 2; break;
        case 76: case 90: f = "recip"; break;
        case 77: case 91: f = "rsqrt"; break;
        case 78: case 92: f = "sin"; break;
        case 79: case 93: f = "sqrt"; break;
        case 80: case 94: f = "tan"; break;
        case 95: f = "abs"; break;  // s_abs
        default: f.clear(); break;
      }
      auto arg = [&](int k) -> std::string { return V(4 + k); };
      if (num == 2 || num == 5 || num == 9) {
        const char* base = num == 2 ? "weft_acos" : (num == 5 ? "weft_asin" : "weft_atan");
        set(std::string(base) + "(" + arg(0) + ") * 0.3183098861837907f");
        return true;
      }
      if (num == 10) {
        set("weft_atan2(" + arg(0) + ", " + arg(1) + ") * 0.3183098861837907f");
        return true;
      }
      if (num == 33) { set("weft_ilogb(" + arg(0) + ")"); return true; }
      if (num == 34) { set("weft_ldexp(" + arg(0) + ", (int)(" + arg(1) + "))"); return true; }
      if (num == 49) { set("weft_pown(" + arg(0) + ", (int)(" + arg(1) + "))"); return true; }
      if (num == 54) { set("weft_rootn(" + arg(0) + ", (int)(" + arg(1) + "))"); return true; }
      auto store_out = [&](const std::string& v) -> bool {
        const uint32_t pid = a.back();
        auto pt = m_.types.find(vtype(pid));
        if (pt == m_.types.end() || pt->second.kind != KPointer) {
          err = "an extended instruction with two results needs a pointer for the second";
          return false;
        }
        std::string ty;
        if (!ctype(pt->second.elem, ty)) {
          err = "the second result has an unsupported type";
          return false;
        }
        const std::string sp = space_of(pid);
        body_ += "  weft_st<" + ty + ">((" + addr_space(sp.c_str(), false) + "void*)" +
                 addr(pid) + ", " + v + ");\n";
        return true;
      };
      if (num == 31) {
        if (!store_out("weft_frexp_e(" + arg(0) + ")")) return false;
        set("weft_frexp_f(" + arg(0) + ")");
        return true;
      }
      if (num == 45) {
        if (!store_out("weft_modf_i(" + arg(0) + ")")) return false;
        set("weft_modf_f(" + arg(0) + ")");
        return true;
      }
      if (num == 58) {
        if (!store_out("weft_sincos_c(" + arg(0) + ")")) return false;
        set("weft_sincos_s(" + arg(0) + ")");
        return true;
      }
      if (num == 52) {
        if (!store_out("weft_remquo_q(" + arg(0) + ", " + arg(1) + ")")) return false;
        set("weft_remquo_r(" + arg(0) + ", " + arg(1) + ")");
        return true;
      }
      if (num == 30) {
        if (a.size() > 5 && !store_out("WEFT_FLOOR(" + arg(0) + ")")) return false;
        set("weft_fract(" + arg(0) + ")");
        return true;
      }
      if (num == 42) {
        set("weft_fma(" + arg(0) + ", " + arg(1) + ", " + arg(2) + ")");
        return true;
      }
      if (num == 46) {
        set("weft_float(0x7FC00000u | (" + arg(0) + " & 0x3FFFFFu))");
        return true;
      }
      if (num == 35) { set("weft_lgamma(" + arg(0) + ")"); return true; }
      if (num == 36) {
        if (!store_out("weft_lgamma_sign(" + arg(0) + ")")) return false;
        set("weft_lgamma(" + arg(0) + ")");
        return true;
      }
      if (num == 68 || num == 82) { set("(" + arg(0) + " / " + arg(1) + ")"); return true; }
      if (num == 55) { set(math("round", arg(0))); return true; }
      if (num == 105) {  // clamp
        set(math("min", math("max", arg(0) + ", " + arg(1)) + ", " + arg(2)));
        return true;
      }
      if (f.empty()) {
        err = "OpenCL.std instruction " + std::to_string(num) + " is not supported";
        return false;
      }
      std::string call;
      if (f == "recip") call = "(1.0f / (" + arg(0) + "))";
      else if (f == "abs") call = math("abs", arg(0));
      else {
        std::string args = arg(0);
        for (int k = 1; k < arity; k++) args += ", " + arg(k);
        static const char* const kPortable[] = {
            "sqrt",  "rsqrt", "sin",   "cos",   "tan",   "sinpi", "cospi", "tanpi",
            "asin",  "acos",  "atan",  "atan2", "sinh",  "cosh",  "tanh",
            "asinh", "acosh", "atanh", "exp",   "exp2",  "exp10", "log",   "log2",
            "log10", "pow",   "powr",  "expm1", "logb",  "maxmag", "minmag",
            "nextafter", "tgamma", "remainder", "erf", "erfc", "cbrt", "hypot",
            "log1p"};
        bool portable = false;
        for (const char* p : kPortable)
          if (f == p) portable = true;
        call = portable ? ("weft_" + f + "(" + args + ")") : math(f, args);
      }
      set(call);
      return true;
    }

    case OpFunctionCall: {
      const uint32_t callee = a[2];
      auto fn = m_.funcs.find(callee);
      if (fn == m_.funcs.end()) { err = "call to an unknown function"; return false; }
      const std::string cname = m_.names.count(callee) ? m_.names[callee] : std::to_string(callee);
      if (!fn->second.defined) return emit_import(ins, cname);

      const Function& cf = fn->second;
      if (call_depth_ >= 8) {
        err = "a call chain deeper than eight, through '" + cname +
              "': a helper that calls itself has no inlined form";
        return false;
      }
      std::vector<const Instr*> params, body, results;
      int labels = 0;
      for (const Instr& ci : cf.body) {
        if (ci.op == OpFunctionParameter) { params.push_back(&ci); continue; }
        if (ci.op == OpLabel) { labels++; continue; }
        switch (ci.op) {
          case OpControlBarrier: case OpMemoryBarrier: case OpVariable:
          case OpGroupNonUniformBallot: case OpGroupNonUniformAll: case OpGroupNonUniformAny:
          case OpGroupNonUniformAllEqual: case OpGroupNonUniformBroadcast:
          case OpGroupNonUniformBroadcastFirst: case OpGroupNonUniformShuffle:
          case OpGroupNonUniformShuffleXor: case OpGroupNonUniformShuffleUp:
          case OpGroupNonUniformShuffleDown: case OpAtomicLoad: case OpAtomicStore:
          case OpAtomicExchange: case OpAtomicIAdd: case OpAtomicISub:
          case OpAtomicCompareExchange:
            err = "a call to '" + cname +
                  "' whose body has a barrier, a shuffle, an atomic or a local variable: "
                  "that is a suspension point, and a call has nowhere to resume to";
            return false;
          default: break;
        }
        body.push_back(&ci);
      }
      if (labels > 1) {
        err = "a call to '" + cname + "' whose body has " + std::to_string(labels) +
              " blocks: only a straight-line helper is inlined";
        return false;
      }
      if (params.size() != a.size() - 3) {
        err = "a call to '" + cname + "' with " + std::to_string(a.size() - 3) +
              " arguments and " + std::to_string(params.size()) + " parameters";
        return false;
      }
      for (size_t k = 0; k < params.size(); k++) {
        std::string pty;
        if (!ctype(params[k]->a[0], pty)) { err = "a helper parameter of an unsupported type"; return false; }
        body_ += "  " + fref(params[k]->a[1]) + " = (" + pty + ")(" + val(a[3 + k]) + ");\n";
      }
      const bool save_call = in_call_;
      const uint32_t save_ret = call_ret_;
      in_call_ = true;
      call_done_ = false;
      call_ret_ = res ? a[1] : 0;
      call_depth_++;
      bool ok = true;
      for (const Instr* ci : body) {
        if (call_done_) break;
        if (!emit_one(*ci)) { ok = false; break; }
      }
      call_depth_--;
      in_call_ = save_call;
      call_ret_ = save_ret;
      if (!ok) return false;
      if (!call_done_) {
        err = "a helper that never returns: '" + cname + "' has no OpReturn in its body";
        return false;
      }
      return true;
    }

    default:
      err = "unsupported SPIR-V opcode " + std::to_string(op);
      return false;
  }
}

const char* const kBuiltinNames[] = {
    "sub_group_scan_exclusive_add", "sub_group_scan_exclusive_min",
    "sub_group_scan_exclusive_max", "sub_group_scan_exclusive_and",
    "sub_group_scan_exclusive_or", "sub_group_scan_exclusive_xor",
    "sub_group_scan_inclusive_add", "sub_group_scan_inclusive_min",
    "sub_group_scan_inclusive_max", "sub_group_scan_inclusive_and",
    "sub_group_scan_inclusive_or", "sub_group_scan_inclusive_xor",
    "sub_group_reduce_add", "sub_group_reduce_min", "sub_group_reduce_max",
    "sub_group_reduce_mul",
    "sub_group_scan_inclusive_mul",
    "sub_group_scan_exclusive_mul",
    "sub_group_reduce_and", "sub_group_reduce_or", "sub_group_reduce_xor",
    "sub_group_shuffle_xor", "sub_group_shuffle_up", "sub_group_shuffle_down",
    "sub_group_shuffle", "sub_group_broadcast", "sub_group_ballot", "sub_group_all",
    "sub_group_any", "sub_group_local_id", "sub_group_size", "work_group_barrier",
    "sub_group_barrier", "mem_fence", "read_mem_fence", "write_mem_fence", "barrier",
};

bool Emitter::emit_import(const Instr& ins, const std::string& sym) {
  const auto& a = ins.a;
  std::string rty;
  if (!ctype(a[0], rty)) { err = "result type of a call is not supported"; return false; }
  const bool has_result = (rty != "void");
  auto V = [&](size_t k) -> std::string {
    if (k >= a.size()) { err = "call to '" + sym + "' is missing an argument"; return "0"; }
    return val(a[k]);
  };
  std::string name;
  for (const char* n : kBuiltinNames)
    if (sym.find(n) != std::string::npos) { name = n; break; }
  if (name.empty()) {
    err = "call to '" + sym + "' is not a builtin this runtime provides";
    return false;
  }
  // a[0]=type a[1]=result a[2]=callee, so the arguments start at 3.
  auto arg = [&](size_t k) { return V(3 + k); };
  auto result = [&](const std::string& e) {
    if (!has_result) return;
    // `fref` is `s->v<id>` on the C++ target and `s->v<id>[l]` where one thread owns
    body_ += "  " + fref(a[1]) + " = " + e + ";\n";
  };

  int last_off = -1;  // the window this call published into, for the fold that reads it
  auto rendezvous = [&](const std::string& payload) -> std::string {
    if (warp_step_) {
      subgroup_ = true;
      last_off = slot_off_;
      slot_off_ += 32;
      body_ += "  weft_slots[" + std::to_string(last_off) + " + l] = (uint32_t)(" + payload +
               "); " + park((uint32_t)resume_next_++) + "\n";
      return "";
    }
    if (gpu) {
      err = "the OpenCL.std subgroup call '" + name +
            "' has no GPU form on the per-lane target: the relay it needs is the "
            "warp target's; the native subgroup instructions say which one it is";
      return "";
    }
    subgroup_ = true;
    const int id = resume_next_++;
    cases_ += "    case " + std::to_string(id) + ": goto R" + std::to_string(id) + ";\n";
    body_ += "  ((uint32_t*)weft::shfl_pub_buf(ctx))[ctx.lid] = (uint32_t)(" + payload +
             "); s->pc = " + std::to_string(id) + "u; return;\n";
    return "  R" + std::to_string(id) + ":;\n";
  };
  // Which lanes of this warp exist.
  static const char* kLive =
      "const uint32_t weft_live = weft::live_threads(ctx.tgid, ctx.tg_size, ctx.grid); "
      "const uint32_t weft_wb = ctx.lid & ~31u; "
      "const uint32_t weft_wl = (weft_live > weft_wb) ? ((weft_live - weft_wb) > 32u ? 32u : "
      "(weft_live - weft_wb)) : 0u; "
      "const uint32_t weft_mask = (weft_wl >= 32u) ? 0xFFFFFFFFu : ((1u << weft_wl) - 1u); ";

  if (name == "barrier" || name == "work_group_barrier" || name == "sub_group_barrier") {
    barrier_ = true;
    if (warp_step_) {
      body_ += "  " + park(resume_next_++) + "\n";
      result("0");
      return true;
    }
    if (gpu) {
      body_ += "  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);\n";
      result("0");
      return true;
    }
    const int id = resume_next_++;
    cases_ += "    case " + std::to_string(id) + ": goto R" + std::to_string(id) + ";\n";
    body_ += "  s->pc = " + std::to_string(id) + "u; return;\n  R" + std::to_string(id) + ":;\n";
    result("0");
    return true;
  }
  if (name == "mem_fence" || name == "read_mem_fence" || name == "write_mem_fence") {
    result("0");
    return true;
  }
  if (name == "sub_group_local_id") { result("(ctx.lid & 31u)"); return true; }
  if (name == "sub_group_size") { result("32u"); return true; }

  if (name == "sub_group_ballot" || name == "sub_group_all" || name == "sub_group_any") {
    if (gpu && !warp_step_) {
      body_ += "  { " + std::string(kLive) +
               "const uint32_t weft_b = ((uint32_t)(simd_vote::vote_t)simd_ballot((" + arg(0) +
               ") ? true : false)) & weft_mask;\n";
      std::string e;
      if (name == "sub_group_ballot") e = "weft_b";
      else if (name == "sub_group_all") e = "(weft_b == weft_mask && weft_mask != 0u)";
      else e = "(weft_b != 0u)";
      body_ += "    " + fref(a[1]) + " = (" + rty + ")(" + e + "); }\n";
      return true;
    }
    const std::string lab = rendezvous("(" + arg(0) + ") ? 1u : 0u");
    body_ += lab;
    if (warp_step_) {
      // Folded out of the window this call published into, over the lanes that exist.
      const std::string mask = slot_mask();
      const std::string b = "(weft::slot_ballot(weft_slots + " + std::to_string(last_off) +
                            ", " + mask + ") & " + mask + ")";
      std::string e;
      if (name == "sub_group_ballot") e = b;
      else if (name == "sub_group_all") e = "(" + b + " == " + mask + " && " + mask + " != 0u)";
      else e = "(" + b + " != 0u)";
      body_ += "  " + fref(a[1]) + " = (" + rty + ")(" + e + ");\n";
      return true;
    }
    std::string e;
    if (name == "sub_group_ballot") e = "(weft::vote_ballot(ctx) & weft_mask)";
    else if (name == "sub_group_all")
      e = "(((weft::vote_ballot(ctx) & weft_mask) == weft_mask) && weft_mask != 0u)";
    else e = "((weft::vote_ballot(ctx) & weft_mask) != 0u)";
    body_ += "  { " + std::string(kLive) + fref(a[1]) + " = (" + rty + ")(" + e + "); }\n";
    return true;
  }

  if (name == "sub_group_shuffle" || name == "sub_group_broadcast" ||
      name == "sub_group_shuffle_xor" || name == "sub_group_shuffle_up" ||
      name == "sub_group_shuffle_down") {
    std::string src;
    const std::string lane = "(ctx.lid & 31u)";
    if (name == "sub_group_shuffle" || name == "sub_group_broadcast")
      src = "(" + arg(1) + ") & 31u";
    else if (name == "sub_group_shuffle_xor") src = "(" + lane + " ^ (" + arg(1) + ")) & 31u";
    else if (name == "sub_group_shuffle_up")
      src = "(" + lane + " >= (" + arg(1) + ") ? " + lane + " - (" + arg(1) + ") : " + lane + ")";
    else
      src = "(" + lane + " + (" + arg(1) + ") < 32u ? " + lane + " + (" + arg(1) + ") : " + lane +
            ")";
    // A value wider than one word is published a word per pass.
    std::string read;
    if (!wide_shfl(a[0], arg(0), "(ctx.lid & ~31u) + (" + src + ")", read)) return false;
    result(read);
    return true;
  }

  if (name.rfind("sub_group_reduce_", 0) == 0 || name.rfind("sub_group_scan_", 0) == 0) {
    const bool scan = name.rfind("sub_group_scan_", 0) == 0;
    const bool incl = name.find("inclusive") != std::string::npos;
    const std::string kind = name.substr(name.find_last_of('_') + 1);
    std::string ety;
    if (!ctype(vtype(a[3]), ety)) { err = "reduction over an unsupported type"; return false; }
    const auto& vtel = m_.types[vtype(a[3])];
    const bool f64_el = gpu && vtel.kind == KFloat && vtel.width == 64;
    std::string seed, comb;
    if (f64_el) {
      if (kind == "add") {
        seed = "0ull";
        comb = "weft_fp64::add(weft_a, weft_v)";
      } else if (kind == "min") {
        seed = "0x7FF0000000000000ull";
        comb = "weft_fp64::min_bits(weft_a, weft_v)";
      } else if (kind == "mul") {
        // 1.0's pattern, and the library's multiply.
        seed = "0x3FF0000000000000ull";
        comb = "weft_fp64::mul(weft_a, weft_v)";
      } else if (kind == "max") {
        seed = "0xFFF0000000000000ull";
        comb = "weft_fp64::max_bits(weft_a, weft_v)";
      } else {
        err = "a subgroup " + kind + " over a double has no GPU form";
        return false;
      }
    } else if (kind == "add") { seed = "0"; comb = "weft_a + weft_v"; }
    else if (kind == "mul") { seed = "1"; comb = "weft_a * weft_v"; }
    else if (kind == "and") { seed = "~(" + ety + ")0"; comb = "weft_a & weft_v"; }
    else if (kind == "or") { seed = "0"; comb = "weft_a | weft_v"; }
    else if (kind == "xor") { seed = "0"; comb = "weft_a ^ weft_v"; }    else if (kind == "min" || kind == "max") {
      const auto& vt = m_.types[vtype(a[3])];
      const bool ints = vt.kind == KInt && vt.width <= 32;
      const bool mn = kind == "min";
      if (!gpu) seed = inf_expr(ety, !mn);
      else if (!ints) seed = mn ? "INFINITY" : "-INFINITY";
      else if (mn) seed = vt.is_signed ? "((" + ety + ")0x7FFFFFFF)" : "(~(" + ety + ")0)";
      else seed = vt.is_signed ? "((" + ety + ")0x80000000)" : "((" + ety + ")0)";
      comb = mn ? "weft_a < weft_v ? weft_a : weft_v" : "weft_a > weft_v ? weft_a : weft_v";
    } else {
      err = "subgroup operation '" + name + "' is not supported";
      return false;
    }
    const bool wide64 = m_.types[vtype(a[3])].width == 64;
    const bool excl = scan && !incl;
    const uint32_t ebits = m_.types[vtype(a[3])].width;
    // The lanes of this warp that exist, and the top one of them.
    const std::string live =
        "const uint32_t weft_live = weft::live_threads(ctx.tgid, ctx.tg_size, ctx.grid); "
        "const uint32_t weft_wb = ctx.lid & ~31u; "
        "const uint32_t weft_wl = (weft_live > weft_wb) ? ((weft_live - weft_wb) > 32u ? 32u : "
        "(weft_live - weft_wb)) : 0u; "
        "const uint32_t weft_mask = (weft_wl >= 32u) ? 0xFFFFFFFFu : ((1u << weft_wl) - 1u); "
        "const uint32_t weft_me = ctx.lid & 31u; "
        "const uint32_t weft_top = weft_wl ? (weft_wl - 1u) : 0u; ";
    const std::string seedv = "(" + ety + ")(" + seed + ")";
    // This lane's own value, with a dead lane's replaced by the identity.
    const std::string mine = "(weft_me < weft_wl) ? (" + ety + ")(" + arg(0) + ") : " + seedv;
    const std::string at = excl ? "(i > 0u ? (i - 1u) : 0u)" : "i";
    const std::string use =
        excl ? "(i > 0u && (weft_mask & (1u << i)))" : "(weft_mask & (1u << i))";
    // A reduce's answer is the top live lane's prefix; a scan keeps its own.
    const std::string keep = scan ? "weft_me" : "weft_top";

    auto tree = [&](const std::string& pre, const std::string& load) {
      body_ += "  { " + live + "\n    " + ety + " weft_pub[32], weft_nxt[32];\n"
               "    for (uint32_t i = 0; i < 32u; i++) {\n" +
               pre + "      weft_pub[i] = " + use + " ? (" + load + ") : " + seedv +
               ";\n    }\n"
               "    for (uint32_t d = 1; d < 32u; d <<= 1) {\n"
               "      for (uint32_t i = 0; i < 32u; i++) {\n"
               "        const " +
               ety + " weft_a = weft_pub[i];\n"
                     "        const " +
               ety + " weft_v = weft_pub[(i >= d) ? (i - d) : i];\n"
               "        weft_nxt[i] = (i >= d) ? (" +
               ety + ")(" + comb + ") : weft_pub[i];\n" +
               "      }\n"
               "      for (uint32_t i = 0; i < 32u; i++) weft_pub[i] = weft_nxt[i];\n"
               "    }\n    " +
               fref(a[1]) + " = weft_pub[" + keep + "]; }\n";
    };

    if (gpu && !warp_step_) {
      auto step = [&](const std::string& expr, const std::string& peer, std::string& out) -> bool {
        if (ebits < 32) {
          out = "((" + ety + ")simd_shuffle((uint32_t)(" + expr + "), (ushort)((" + peer +
                ") & 31u)))";
          return true;
        }
        return wide_shfl(vtype(a[3]), expr, peer, out);
      };
      std::string up;
      if (!step("weft_a",
                scan ? "((ctx.lid & 31u) >= d ? (ctx.lid & 31u) - d : (ctx.lid & 31u))"
                     : "((ctx.lid & 31u) >= d ? (ctx.lid & 31u) - d : (ctx.lid & 31u))",
                up))
        return false;
      std::string start = mine;
      if (excl) {
        std::string prev;
        if (!step("(" + ety + ")(" + arg(0) + ")",
                  "((ctx.lid & 31u) ? (ctx.lid & 31u) - 1u : 0u)", prev))
          return false;
        start = "(weft_me > 0u && weft_me < weft_wl) ? (" + ety + ")(" + prev + ") : " + seedv;
      }
      body_ += "  { " + live + "\n    " + ety + " weft_a = " + start +
               ";\n"
               "    for (uint32_t d = 1; d < 32u; d <<= 1) {\n"
               "      const " +
               ety + " weft_v = (" + ety + ")(" + up +
               ");\n"
               "      if (weft_me >= d) weft_a = (" +
               ety + ")(" + comb + ");\n"
               "    }\n";
      if (scan) {
        body_ += "    " + fref(a[1]) + " = weft_a; }\n";
      } else {
        std::string fin;
        if (!step("weft_a", "weft_top", fin)) return false;
        body_ += "    " + fref(a[1]) + " = (" + ety + ")(" + fin + "); }\n";
      }
      return true;
    }

    if (wide64) {
      const std::string bits = ety == "double" ? ("weft_bitcast<uint64_t>(" + arg(0) + ")")
                                               : ("(uint64_t)(" + arg(0) + ")");
      if (warp_step_) {
        subgroup_ = true;
        const int lo = slot_off_;
        slot_off_ += 64;
        body_ += "  { const uint64_t weft_b = " + bits +
                 ";\n"
                 "    weft_slots[" +
                 std::to_string(lo) + " + l] = (uint32_t)(weft_b & 0xFFFFFFFFu);\n"
                 "    weft_slots[" +
                 std::to_string(lo + 32) + " + l] = (uint32_t)(weft_b >> 32); }\n  " +
                 park((uint32_t)resume_next_++) + "\n";
        const std::string l = std::to_string(lo), h = std::to_string(lo + 32);
        tree("      const uint64_t weft_w = ((uint64_t)weft_slots[" + h + " + " + at +
                 "] << 32) | (uint64_t)weft_slots[" + l + " + " + at + "];\n",
             "weft_bitcast<" + ety + ">(weft_w)");
        return true;
      }
      subgroup_ = true;
      const int id = resume_next_++;
      cases_ += "    case " + std::to_string(id) + ": goto R" + std::to_string(id) + ";\n";
      body_ += "  { const uint64_t weft_b = " + bits +
               "; std::memcpy(weft::wide_pub_slot(ctx), &weft_b, 8); }\n"
               "  s->pc = " +
               std::to_string(id) + "u; return;\n  R" + std::to_string(id) + ":;\n";
      tree("      uint64_t weft_w = 0u;\n      if (" + use +
               ") std::memcpy(&weft_w, weft::wide_read_slot(ctx, weft_wb + " + at + "), 8);\n",
           "weft_bitcast<" + ety + ">(weft_w)");
      return true;
    }

    std::string payload;
    if (!shfl_pub(vtype(a[3]), arg(0), payload)) return false;
    const std::string lab = rendezvous(payload);
    body_ += lab;
    if (warp_step_) {
      const std::string off = std::to_string(last_off);
      tree("", "weft_bitcast<" + ety + ">(weft_slots[" + off + " + " + at + "])");
      return true;
    }
    tree("", "weft_bitcast<" + ety + ">(weft::shfl_read_buf(ctx)[weft_wb + " + at + "])");
    return true;
  }
  err = "call to '" + sym + "': '" + name + "' is not supported";
  return false;
}

bool Emitter::run() {
  const auto& B = f_.body;

  // What the kernel does with its images decides their access mode.
  for (const Instr& ins : B) {
    if (ins.op == OpImageWrite) writes_image_ = true;
    if (ins.op == OpImageSampleImplicitLod || ins.op == OpImageSampleExplicitLod)
      samples_image_ = true;
  }
  if (gpu && writes_image_ && samples_image_) {
    err =
        "a kernel that both samples and writes an image needs two access modes on "
        "one binding";
    return false;
  }

  for (const Instr& ins : B) {
    if (ins.op != OpFunctionParameter) continue;
    const uint32_t tid = ins.a[0], id = ins.a[1];
    vt_[id] = tid;
    auto t = m_.types.find(tid);
    if (t == m_.types.end()) { err = "parameter of an unknown type"; return false; }
    if (t->second.kind == KPointer && t->second.storage == 0) {
      // UniformConstant: an image or a sampler.
      auto e = m_.types.find(t->second.elem);
      if (e == m_.types.end()) { err = "parameter of an unknown opaque type"; return false; }
      if (e->second.kind == KImage) {        const int idx = ntex_++;
        const bool arr = e->second.arrayed != 0;
        const bool vol = e->second.count == 2u && !arr;
        const bool cb = e->second.count == 3u;
        tex_shape.push_back(cb ? "cube" : vol ? "3d" : arr ? "array" : "2d");
        if (gpu) {
          const char* mk =
              cb ? "weft::make_cube_image(weft_t"
                 : vol ? (writes_image_ ? "weft::make_rw_image3d(weft_t" : "weft::make_image3d(weft_t")
                 : arr ? (writes_image_ ? "weft::make_rw_array_image(weft_t"
                                        : "weft::make_array_image(weft_t")
                       : (writes_image_ ? "weft::make_rw_image(weft_t"
                                        : "weft::make_image(weft_t");
          local_[id] = std::string(mk) + std::to_string(idx) + ")";
          tex_access.push_back(writes_image_ ? "read_write" : "sample");
          tex_arrayed.push_back(arr);
          tex_vol.push_back(vol);
          tex_cube.push_back(cb);
        } else {
          local_[id] = "weft::texture_ptr(" + std::to_string(idx) + ")";
        }
      } else if (e->second.kind == KSampler) {
        const int idx = nsamp_++;
        local_[id] = gpu ? ("weft_s" + std::to_string(idx))
                         : ("weft::sampler(" + std::to_string(idx) + ")");
      }
      else { err = "opaque kernel parameter that is neither an image nor a sampler"; return false; }
    } else if (t->second.kind == KPointer) {
      const uint32_t st = t->second.storage;
      if (st == 1 || st == 4 || st == 5) {
        base_addr_[id] = "((" + addr_space("device", false) + "unsigned char*)bufs[" +
                         std::to_string(nbuffers_) + "])";
        buf_elem_.push_back(t->second.elem);
        nbuffers_++;
      } else if (st == 3) {
        // A `__local` kernel argument.
        const uint32_t off = wg_bytes_;
        base_addr_[id] = "((" + addr_space("threadgroup", false) + "unsigned char*)sh + " +
                         std::to_string(off) + "u)";
        wg_bytes_ += 24u * 1024u;
      } else {
        err = "kernel parameter in storage class " + std::to_string(st) + " is not supported";
        return false;
      }
    } else {
      std::string ty;
      if (!ctype(tid, ty)) { err = "parameter of an unsupported type"; return false; }
      local_[id] = "(" + ty + ")" + param_of(nscalars_++);
    }
  }

  for (const Instr& ins : B) {
    if (no_result(ins.op) || ins.a.size() < 2 || ins.op == OpFunctionParameter) continue;
    if (ins.op == OpIAddCarry || ins.op == OpISubBorrow || ins.op == OpUMulExtended ||
        ins.op == OpSMulExtended)
      continue;
    vt_[ins.a[1]] = ins.a[0];
  }
  for (const auto& fn_it : m_.funcs) {
    const Function& f = fn_it.second;
    if (!f.defined || &f.body == &B) continue;
    for (const Instr& ins : f.body) {
      if (ins.op == OpFunctionParameter) { vt_[ins.a[1]] = ins.a[0]; continue; }
      if (no_result(ins.op) || ins.a.size() < 2) continue;
      vt_[ins.a[1]] = ins.a[0];
    }
  }

  for (size_t i = 0; i < B.size(); i++) {
    if (B[i].op != OpLabel) continue;
    Block b;
    b.label = B[i].a[0];
    b.begin = i + 1;
    size_t e = i + 1;
    while (e < B.size() && B[e].op != OpLabel) e++;
    b.end = e;
    if (label_of_.count(b.label)) { err = "duplicate label"; return false; }
    label_of_[b.label] = (int)blocks_.size();
    blocks_.push_back(b);
  }
  if (blocks_.empty()) { err = "kernel has no basic blocks"; return false; }

  for (const Block& b : blocks_)
    for (size_t k = b.begin; k < b.end; k++) {
      const Instr& ins = B[k];
      if (ins.op != OpPhi) continue;
      vt_[ins.a[1]] = ins.a[0];
      for (size_t j = 2; j + 1 < ins.a.size(); j += 2)
        phis_[b.label].push_back(PhiIn{ins.a[1], ins.a[j + 1], ins.a[j]});
    }

  for (const Instr& ins : B)
    if (ins.op == OpVariable) own_storage_.insert(ins.a[1]);

  for (const auto& it : vt_) {
    if (base_addr_.count(it.first) || local_.count(it.first) || pval_.count(it.first) ||
        own_storage_.count(it.first))
      continue;
    std::string ty;
    if (!ctype(it.second, ty)) {
      err = "value " + std::to_string(it.first) + " has an unsupported type";
      return false;
    }
    if (gpu && ty.find('*') != std::string::npos) {
      deferred_ptr_.push_back(it.first);
      continue;
    }
    decls_ += "  " + ty + " " + field(it.first) + (warp_ ? "[32]" : "") + ";\n";
  }

  for (const auto& it : m_.globals) {
    const Global& g = it.second;
    if (g.has_builtin) continue;
    auto pt = m_.types.find(g.type);
    if (pt == m_.types.end() || pt->second.kind != KPointer) continue;
    vt_[it.first] = g.type;  // so an access chain can see what it points at
    // A `UniformConstant` variable is an image or a sampler.
    if (g.storage == 0) {
      auto e = m_.types.find(pt->second.elem);
      if (e == m_.types.end()) { err = "module variable of an unknown opaque type"; return false; }
      if (e->second.kind == KImage) {
        const int idx = ntex_++;
        const bool arr = e->second.arrayed != 0;
        const bool vol = e->second.count == 2u && !arr;
        const bool cb = e->second.count == 3u;
        tex_shape.push_back(cb ? "cube" : vol ? "3d" : arr ? "array" : "2d");
        if (gpu) {
          const char* mk =
              cb ? "weft::make_cube_image(weft_t"
                 : vol ? (writes_image_ ? "weft::make_rw_image3d(weft_t" : "weft::make_image3d(weft_t")
                 : arr ? (writes_image_ ? "weft::make_rw_array_image(weft_t"
                                        : "weft::make_array_image(weft_t")
                       : (writes_image_ ? "weft::make_rw_image(weft_t"
                                        : "weft::make_image(weft_t");
          local_[it.first] = std::string(mk) + std::to_string(idx) + ")";
          tex_access.push_back(writes_image_ ? "read_write" : "sample");
          tex_arrayed.push_back(arr);
          tex_vol.push_back(vol);
          tex_cube.push_back(cb);
        } else {
          local_[it.first] = "weft::texture_ptr(" + std::to_string(idx) + ")";
        }
      } else if (e->second.kind == KSampler) {
        const int idx = nsamp_++;
        local_[it.first] = gpu ? ("weft_s" + std::to_string(idx))
                               : ("weft::sampler(" + std::to_string(idx) + ")");
      } else {
        err = "module variable that is neither an image nor a sampler";
        return false;
      }
      continue;
    }
    uint32_t sz = 0;
    if (!tsize(pt->second.elem, sz)) {
      err = "global variable of an unsupported type";
      return false;
    }
    if (sz == 0) sz = 1;
    if (g.storage == 4 || g.storage == 3) {
      base_addr_[it.first] = "((" + addr_space("threadgroup", false) + "unsigned char*)sh + " +
                             std::to_string(wg_bytes_) + "u)";
      wg_bytes_ += sz;
      if (g.initialized)
        init_ += gpu ? ("  if (ctx.lid == 0u) weft::zero_shared(sh + " +
                        std::to_string(wg_bytes_ - sz) + "u, " + std::to_string(sz) + ");\n")
                     : ("  if (ctx.lid == 0u) std::memset((void*)(sh + " +
                        std::to_string(wg_bytes_ - sz) + "u), 0, " + std::to_string(sz) + ");\n");
    } else if (g.storage == 6) {
      decls_ += "  unsigned char w" + std::to_string(it.first) + "[" + std::to_string(sz) + "];\n";
      base_addr_[it.first] = "((" + addr_space("thread", false) + "unsigned char*)s->w" +
                             std::to_string(it.first) + ")";
      if (g.initialized)
        init_ += gpu ? ("  weft::zero_thread(s->w" + std::to_string(it.first) + ", " +
                        std::to_string(sz) + ");\n")
                     : ("  std::memset(s->w" + std::to_string(it.first) + ", 0, " +
                        std::to_string(sz) + ");\n");
    } else if (g.storage == 12 || g.storage == 2) {
      base_addr_[it.first] = "((" + addr_space("device", false) + "unsigned char*)bufs[" +
                             std::to_string(nbuffers_) + "])";
      uint32_t belt = pt->second.elem;
      auto st = m_.types.find(belt);
      if (st != m_.types.end() && st->second.kind == KStruct && !st->second.members.empty()) {
        auto m0 = m_.types.find(st->second.members[0]);
        if (m0 != m_.types.end() && m0->second.kind == KArray) belt = m0->second.elem;
      }
      buf_elem_.push_back(belt);
      nbuffers_++;
    } else if (g.storage == 5) {
      base_addr_[it.first] = "((" + addr_space("device", false) + "unsigned char*)bufs[" +
                             std::to_string(nbuffers_) + "])";
      buf_elem_.push_back(pt->second.elem);
      auto et = m_.types.find(pt->second.elem);
      if (et != m_.types.end() && et->second.kind == KPointer) ptr_globals_.insert(it.first);
      nbuffers_++;
    }
  }

  for (size_t bi = 0; bi < blocks_.size(); bi++) {
    if (warp_step_) {
      body_ += case_open((uint32_t)bi);
      if ((int)bi > max_case_) max_case_ = (int)bi;
    } else
      body_ += gpu ? ("  case " + std::to_string(bi) + "u:;\n")
                   : ("L" + std::to_string(bi) + ":;\n");
    cur_block_ = blocks_[bi].label;
    for (size_t k = blocks_[bi].begin; k < blocks_[bi].end; k++) {
      if (B[k].op == OpPhi) continue;  // written by the incoming edges
      if (!emit_one(B[k])) return false;
    }
    if (warp_step_) body_ += case_close();
  }
  for (uint32_t id : deferred_ptr_) {
    std::string ty;
    if (!ctype(vtype(id), ty)) continue;
    auto ps = ptr_space_.find(id);
    if (ps != ptr_space_.end() && ty.compare(0, 7, "device ") == 0)
      ty = ps->second + " " + ty.substr(7);
    decls_ += "  " + ty + " " + field(id) + (warp_ ? "[32]" : "") + ";\n";
  }

  for (uint32_t e : buf_elem_) buf_type.push_back(bytes_of(e));
  nargs = nbuffers_;
  ntex = ntex_;
  nsamp = nsamp_;
  sync = barrier_ || subgroup_;
  return true;
}

}  // namespace

static TranslateResult translate_spirv_impl(const std::string& raw,
                                           const std::vector<std::pair<int, uint32_t>>& widths,
                                           bool gpu) {
  TranslateResult r;
  Mod m;
  std::string err;
  if (!parse_module(m, raw, err)) {
    r.error = err;
    return r;
  }

  std::string kernels;
  std::vector<TranslatedKernel> out;
  for (const Entry& e : m.entries) {
    auto f = m.funcs.find(e.function);
    if (f == m.funcs.end() || !f->second.defined) {
      r.error = "entry point " + e.name + " has no function body";
      return r;
    }
    bool warp_pc = false;
    bool warp_step = false;
    std::string warp_why = "needs something the warp target cannot provide";
    auto warp_shape = [&]() -> bool {
      const auto& B = f->second.body;
      int blocks = 0;
      bool wide = false;
      bool sub = false;
      bool vote = false;
      for (const Instr& ins : B) {
        switch (ins.op) {
          case OpLabel: blocks++; break;
          case OpControlBarrier:
            // A barrier is a hand-off between lanes, which step mode provides.
            warp_step = true;
            break;
          case OpImageWrite: case OpImageRead: case OpImageFetch:
          case OpImageSampleImplicitLod: case OpImageSampleExplicitLod:
            warp_why = "reads or writes an image, which is a per-lane binding";
            return false;
          default:
            if (ins.op >= OpGroupNonUniformElect && ins.op <= OpGroupNonUniformShuffleDown) {
              sub = true;
            } else if (ins.op == OpFunctionCall && ins.a.size() > 2) {
              auto nm = m.names.find(ins.a[2]);
              if (nm != m.names.end() && nm->second.find("sub_group_") != std::string::npos)
                sub = true;
            }
            break;
        }
        if (ins.op > OpAtomicLoad && ins.op <= OpAtomicXor && ins.a.size() >= 2) {
          auto t = m.types.find(ins.a[0]);
          if (t != m.types.end() && t->second.kind != KPointer && t->second.width == 64)
            wide = true;
        }
        if (ins.op == OpAtomicLoad && !ins.a.empty()) {
          auto t = m.types.find(ins.a[0]);
          if (t != m.types.end() && t->second.width == 64) wide = true;
        }
        if (ins.op == OpAtomicStore && !ins.a.empty()) {
          auto t = m.types.find(ins.a[0]);
          if (t != m.types.end() && t->second.kind == KPointer) {
            auto e = m.types.find(t->second.elem);
            if (e != m.types.end() && e->second.width == 64) wide = true;
          }
        }
      }
      for (const auto& g : m.globals)
        if (g.second.storage == 3 || g.second.storage == 4) {
          warp_why = "declares threadgroup memory, which belongs to a group of lanes "
                     "rather than to one lane";
          return false;
        }
      if (sub && blocks > 1) { warp_step = true; vote = true; }
      warp_pc = blocks > 1 || warp_step;
      return wide || vote;
    };
    Emitter em(m, f->second);
    em.gpu_target(gpu);
    const bool warp_kernel = gpu && warp_shape();
    em.warp_reason(warp_why);
    if (warp_kernel) em.warp_target(warp_pc, warp_step);
    if (!em.run()) {
      r.error = "kernel " + e.name + ": " + em.err;
      return r;
    }
    const std::string st = "weft_s_" + e.name;
    const std::string code = em.code();
    uint32_t slot_words = 0, slot_per = 0;
    if (gpu && warp_kernel) {
      // One Metal thread runs a whole 32-lane warp.
      const int nbufs = em.nargs;
      // The warp body's program counter, when the kernel has more than one block.
      kernels += "struct " + st + " {\n";
      if (em.warp_step()) kernels += "  uint32_t pc[32];\n";
      else if (em.warp_blocks()) kernels += "  uint32_t pc;\n";
      kernels += em.decls() + "};\n";
      kernels += "kernel void " + e.name + "(";
      for (int i = 0; i < nbufs; i++)
        kernels += "device uint8_t* weft_b" + std::to_string(i) + " [[buffer(" +
                    std::to_string(i) + ")]],\n                 ";
      kernels += "device uint64_t* weft_params [[buffer(" + std::to_string(nbufs) +
                 ")]],\n                 ";
      kernels += "device uint64_t* weft_geom [[buffer(" + std::to_string(nbufs + 1) +
                 ")]],\n                 ";
      kernels += "uint weft_tid [[thread_index_in_threadgroup]],\n                 ";
      kernels += "uint3 weft_tgid [[threadgroup_position_in_grid]],\n                 ";
      kernels += "uint3 weft_tgsz [[threads_per_threadgroup]]) {\n";
      kernels += "  thread " + st + " st;\n  thread " + st + "* s = &st;\n";
      kernels += "  for (uint32_t weft_i = 0; weft_i < (uint32_t)sizeof(st); weft_i++)\n";
      kernels += "    ((thread uint8_t*)&st)[weft_i] = 0u;\n";
      kernels += "  weft::Ctx ctx;\n";
      kernels += "  ctx.tgid = weft_tgid.x; ctx.tg_size = 32u;\n";
      kernels += "  ctx.grid = (uint32_t)weft_geom[0];\n";
      for (int i = 0; i < nbufs; i++)
        kernels += "  ctx.b[" + std::to_string(i) + "] = weft_b" + std::to_string(i) + ";\n";
      kernels += "  ctx.params = weft_params;\n";
      if (nbufs > 0) {
        kernels += "  device uint8_t* bufs[" + std::to_string(nbufs) + "] = {";
        for (int i = 0; i < nbufs; i++)
          kernels += (i ? ", weft_b" : "weft_b") + std::to_string(i);
        kernels += "};\n  (void)bufs;\n";
      }
      kernels += "  ctx.tgid = weft_tgid.x * weft_tgsz.x + weft_tid;\n";
      kernels += "  const uint32_t weft_lanes = (uint32_t)weft_geom[0];\n";
      if (em.warp_step()) {
        slot_words = (uint32_t)em.slot_words();
        slot_per = 6144u / slot_words;  // 24 KiB of threadgroup memory, at most
        if (slot_per > 8u) slot_per = 8u;
        if (slot_per == 0u) slot_per = 1u;
        kernels += "  threadgroup uint32_t weft_slots_all[" +
                   std::to_string((uint64_t)slot_words * slot_per) + "];\n";
        kernels += "  threadgroup uint32_t* weft_slots = weft_slots_all + weft_tid * " +
                   std::to_string(slot_words) + "u;\n";
      }
      std::string body = code;
      std::vector<const Emitter::ShadowSite*> live_shadows;
      if (em.wide_atom) {
        for (const auto& s : em.shadows) {
          size_t uses = 0;
          for (size_t at = body.find(s.buf); at != std::string::npos;
               at = body.find(s.buf, at + s.buf.size()))
            uses++;
          if (uses != 1) continue;
          for (size_t at = body.find(s.call); at != std::string::npos;
               at = body.find(s.call, at + s.shadow.size()))
            body.replace(at, s.call.size(), s.shadow);
          live_shadows.push_back(&s);
        }
      }
      for (const auto* s : live_shadows) kernels += "  " + s->decl + "\n";
      if (em.wide_atom) {
        for (const auto& a : em.red_accs)
          kernels += "  uint64_t " + a.name + " = " +
                     (a.op == "and" ? "~0ull" : "0ull") + ";\n";
        kernels += "  const uint32_t weft_wn = (uint32_t)weft_geom[2];\n";
        kernels += "  for (uint32_t weft_w = 0; weft_w < weft_wn; weft_w++) {\n";
        kernels += "    ctx.tgid = weft_w;\n";
        kernels += "    const uint32_t weft_base = weft_w * 32u;\n";
      } else {
        kernels += "  const uint32_t weft_base = ctx.tgid * 32u;\n";
      }
      const bool wpc = em.warp_blocks();
      const bool wstep = em.warp_step();
      if (wstep) {
        kernels += "  for (uint32_t i = 0; i < " + std::to_string(em.slot_words()) +
                   "u; i++) weft_slots[i] = 0u;\n";
        kernels += "  for (uint32_t l = 0; l < 32u; l++)\n";
        kernels += "    s->pc[l] = (weft_base + l < weft_lanes) ? 0u : 0xFFFFFFFFu;\n";
        kernels += "  for (uint32_t l = 0; l < 32u; l++) {\n";
        kernels += "    if (weft_base + l >= weft_lanes) break;\n";
        kernels += "    ctx.lid = l; ctx.gid = (uint64_t)weft_base + (uint64_t)l;\n";
        kernels += em.init();
        kernels += "  }\n";
        kernels += "  for (;;) {\n";
        kernels += "    bool weft_ran = false;\n";
        kernels += "    for (uint32_t weft_id = 0; weft_id <= " +
                   std::to_string(em.max_case()) + "u; weft_id++) {\n";
        kernels += "      switch (weft_id) {\n";
      } else {
      kernels += "  for (uint32_t l = 0; l < 32u; l++) {\n";
      kernels += "    if (weft_base + l >= weft_lanes) break;\n";
      kernels += "    ctx.lid = l; ctx.gid = (uint64_t)weft_base + (uint64_t)l;\n";
      if (wpc) kernels += "    s->pc = 0u;\n";
      kernels += em.init();
      if (wpc) {
        kernels += "    for (;;) {\n";
        kernels += "      if (s->pc == 0xFFFFFFFFu) break;\n";
        kernels += "      switch (s->pc) {\n";
      } else {
        kernels += "    switch (0u) {\n";
      }
      }  // the per-lane half of `if (wstep)`; the block text is shared below
      kernels += body;
      if (wstep) {
        kernels += "        default: break;\n";
        kernels += "      }\n    }\n    if (!weft_ran) break;\n  }\n";
      } else {
        if (wpc) {
          kernels += "        default: { s->pc = 0xFFFFFFFFu; break; }\n      }\n    }\n";
        } else {
          kernels += "    }\n";
        }
        kernels += "  }\n";
      }
      if (em.wide_atom) kernels += "  }\n";  // the warp walk
      for (const auto* s : live_shadows) kernels += "  " + s->flush + "\n";
      for (const auto& a : em.red_accs)
        kernels += "  (void)weft::red_" + a.op + "_u64(" + a.addr + ", " + a.name + ");\n";
      em.red_accs.clear();
      em.shadows.clear();
      kernels += "}\n\n";
    } else if (gpu) {
      const int nbufs = em.nargs;
      kernels += "struct " + st + " {\n  uint32_t pc;\n" + em.decls() + "};\n";
      kernels += "kernel void " + e.name + "(";
      for (int i = 0; i < nbufs; i++)
        kernels += "device uint8_t* weft_b" + std::to_string(i) + " [[buffer(" +
                    std::to_string(i) + ")]],\n                 ";
      kernels += "device uint64_t* weft_params [[buffer(" + std::to_string(nbufs) +
                 ")]],\n                 ";
      for (int i = 0; i < em.ntex; i++) {
        const bool arr = (size_t)i < em.arrayed_textures().size() && em.arrayed_textures()[i];
        const bool vol = (size_t)i < em.volume_textures().size() && em.volume_textures()[i];
        const bool cb = (size_t)i < em.cube_textures().size() && em.cube_textures()[i];
        kernels += std::string(cb ? "texturecube<float, access::"
                                   : vol ? "texture3d<float, access::"
                                        : arr ? "texture2d_array<float, access::"
                                              : "texture2d<float, access::") +
                   em.textures()[i] + "> weft_t" + std::to_string(i) + " [[texture(" +
                   std::to_string(i) + ")]],\n                 ";
      }
      for (int i = 0; i < em.nsamp; i++)
        kernels += "sampler weft_s" + std::to_string(i) + " [[sampler(" +
                   std::to_string(i) + ")]],\n                 ";
      kernels += "uint weft_lid [[thread_index_in_threadgroup]],\n                 ";
      kernels += "uint3 weft_tgid [[threadgroup_position_in_grid]],\n                 ";
      kernels += "uint3 weft_tgsz [[threads_per_threadgroup]],\n                 ";
      kernels += "uint3 weft_grid [[threads_per_grid]]) {\n";
      kernels += "  thread " + st + " st;\n  thread " + st + "* s = &st;\n";
      kernels += "  s->pc = 0u;\n";
      kernels += "  weft::Ctx ctx;\n";
      kernels += "  ctx.lid = weft_lid; ctx.tgid = weft_tgid.x; ctx.tg_size = weft_tgsz.x;\n";
      kernels += "  ctx.gid = (uint64_t)weft_tgid.x * (uint64_t)weft_tgsz.x + (uint64_t)weft_lid;\n";
      kernels += "  ctx.grid = weft_grid.x;\n";
      for (int i = 0; i < nbufs; i++)
        kernels += "  ctx.b[" + std::to_string(i) + "] = weft_b" + std::to_string(i) + ";\n";
      kernels += "  ctx.params = weft_params;\n";
      kernels += "  (void)ctx;\n";
      if (nbufs > 0) {
        kernels += "  device uint8_t* bufs[" + std::to_string(nbufs) + "] = {";
        for (int i = 0; i < nbufs; i++)
          kernels += (i ? ", weft_b" : "weft_b") + std::to_string(i);
        kernels += "};\n  (void)bufs;\n";
      }
      if (em.arena_bytes() > 0) {
        kernels += "  threadgroup uint8_t weft_sh[" + std::to_string(em.arena_bytes()) + "];\n";
        kernels += "  threadgroup uint8_t* sh = weft_sh;\n  (void)sh;\n";
      }
      kernels += em.init();
      kernels += "  for (;;) {\n    switch (s->pc) {\n";
      kernels += code;
      kernels += "      default: return;\n    }\n  }\n}\n\n";
    } else {
    kernels += "struct " + st + " {\n  uint32_t pc;\n" + em.decls() + "};\n";
    std::string inner;
    inner += "    " + st + "* s = (" + st + "*)st;\n";
    inner += "    (void)ctx; (void)bufs; (void)sh;\n";
    inner += "    if (s->pc != 0u) {\n      switch (s->pc) {\n" + em.cases() +
             "        default: goto weft_done;\n      }\n    }\n";
    inner += em.init();
    inner += em.code();
    inner += "    weft_done:;\n    s->pc = 0xFFFFFFFFu;\n";
    kernels += "static uint32_t weft_g_" + e.name +
               "(weft::Ctx* ctxs, unsigned char* states, size_t stride, uint32_t live,\n"
               " uint32_t pass, void* const* bufs, unsigned char* sh) {\n"
               "  uint32_t left = 0;\n"
               "  for (uint32_t i = 0; i < live; i++) {\n"
               "    unsigned char* st = states + (size_t)i * stride;\n"
               "    if (*(const uint32_t*)st == weft::kThreadDone) continue;\n"
               "    weft::Ctx& ctx = ctxs[i];\n"
               "    ctx.pass = pass;\n"
               "    [&] {\n" + inner +
               "    }();\n"
               "    if (*(const uint32_t*)st != weft::kThreadDone) left++;\n"
               "  }\n"
               "  return left;\n"
               "}\n\n";
    kernels += "static void weft_k_" + e.name +
               "(weft::Ctx& ctx, void* stp, void* const* bufs, unsigned char* sh) {\n"
               "  weft_g_" + e.name + "(&ctx, (unsigned char*)stp, 0, 1, ctx.pass, bufs, sh);\n"
               "}\n\n";
    }

    TranslatedKernel tk;
    tk.name = e.name;
    tk.nargs = em.nargs;
    tk.buf_type = em.buf_type;
    tk.nbufs = tk.nargs;  // no render stages in SPIR-V: its own buffers
    tk.def_threads = e.lx ? e.lx * e.ly * e.lz : 0u;
    // A group wider than a row fixes the lane identity, so it fixes the launch.
    if (e.ly > 1u || e.lz > 1u) tk.fixed_threads = tk.def_threads;
    tk.tex_shape = em.texture_shapes();
    tk.ntex = em.ntex;
    tk.wide = gpu && em.wide_atom;
    tk.gpu_warp = warp_kernel;
    tk.slot_words = slot_words;
    tk.slot_per = slot_per;
    tk.nsamp = em.nsamp;
    tk.sync = em.sync;
    tk.flat = false;
    tk.warp = false;
    out.push_back(tk);
  }
  if (out.empty()) {
    r.error = "no kernel entry point in the module";
    return r;
  }

  if (gpu) {
    // Only the kernels and the runtime.
    std::string g = "// generated by weft (spir-v front end, metal target), do not edit\n";
    g += kWWeftMslPreamble;
    g += kWSoftMath;
    g += kernels;
    r.gpu_src = g;
    r.ok = true;
    r.kernels = out;
    return r;
  }

  std::string code =
      "// generated by weft (spir-v front end), do not edit\n"
      "#include \"weft_runtime.hpp\"\n"
      "#include <algorithm>\n"
      "#include <cmath>\n"
      "#include <cstdint>\n"
      "#include <cstring>\n"
      "#include <limits>\n"
      "struct weft_f2 { float x, y; };\n"
      "struct weft_f3 { float x, y, z; };\n"
      "struct weft_f4 { float x, y, z, w; };\n"
      "struct weft_d2 { double x, y; };\n"
      "struct weft_d3 { double x, y, z; };\n"
      "struct weft_d4 { double x, y, z, w; };\n"
      "struct weft_i2 { int32_t x, y; };\n"
      "struct weft_i3 { int32_t x, y, z; };\n"
      "struct weft_i4 { int32_t x, y, z, w; };\n"
      "struct weft_b2 { bool x, y; };\n"
      "struct weft_b3 { bool x, y, z; };\n"
      "struct weft_b4 { bool x, y, z, w; };\n"
      "struct weft_u2 { uint32_t x, y; };\n"
      "struct weft_u3 { uint32_t x, y, z; };\n"
      "struct weft_u4 { uint32_t x, y, z, w; };\n"
      "struct weft_l2 { uint64_t x, y; };\n"
      "struct weft_l3 { uint64_t x, y, z; };\n"
      "struct weft_l4 { uint64_t x, y, z, w; };\n"
      "struct weft_ul2 { uint64_t x, y; };\n"
      "struct weft_ul3 { uint64_t x, y, z; };\n"
      "struct weft_ul4 { uint64_t x, y, z, w; };\n"
      "// Matrices, column-major: `m[column][row]`, the way SPIR-V, Metal and every GPU lay\n"
      "// them out, so a matrix times a vector is a linear combination of its columns. One\n"
      "// template covers every shape SPIR-V allows, two to four columns of two to four\n"
      "// rows, because one shader multiplies a 4x4 by a 4-vector and a 3x4 skinning matrix\n"
      "// by a 3-vector in the same file, and the square-only set refused the second. Every\n"
      "// accumulation is an explicit fused multiply-add in a fixed order, so the two engines\n"
      "// run the same instruction sequence and agree to the bit. Everything here takes its\n"
      "// operands by value: Metal wants an address space on every pointer and every\n"
      "// reference, and this one text is compiled by both engines.\n"
      "template <int R, int C>\n"
      "struct weft_mat { float m[C][R]; };\n"
      "using weft_f2x2 = weft_mat<2, 2>;\n"
      "using weft_f2x3 = weft_mat<2, 3>;\n"
      "using weft_f2x4 = weft_mat<2, 4>;\n"
      "using weft_f3x2 = weft_mat<3, 2>;\n"
      "using weft_f3x3 = weft_mat<3, 3>;\n"
      "using weft_f3x4 = weft_mat<3, 4>;\n"
      "using weft_f4x2 = weft_mat<4, 2>;\n"
      "using weft_f4x3 = weft_mat<4, 3>;\n"
      "using weft_f4x4 = weft_mat<4, 4>;\n"
      "\n"
      "// The components of a row on their way into a vector, as a value rather than a\n"
      "// pointer so no address space has to be named.\n"
      "struct weft_n { float v[4]; };\n"
      "// The bit a ballot says is there: the lowest and the highest one that is set.\n"
      "// A loop rather than a builtin, `ctz` is spelled differently on the two targets and\n"
      "// this runs once per ballot, not once per lane.\n"
      "static inline uint32_t weft_lsb(uint32_t v) {\n"
      "  for (uint32_t i = 0; i < 32u; i++)\n"
      "    if (v & (1u << i)) return i;\n"
      "  return 0xFFFFFFFFu;\n"
      "}\n"
      "static inline uint32_t weft_msb(uint32_t v) {\n"
      "  for (int i = 31; i >= 0; i--)\n"
      "    if (v & (1u << (uint32_t)i)) return (uint32_t)i;\n"
      "  return 0xFFFFFFFFu;\n"
      "}\n"
      "\n"
      "// The width of a vector type, so a matrix can name the vector it multiplies and the\n"
      "// loops below are written once for all nine shapes instead of once per shape.\n"
      "template <int N> struct weft_vt;\n"
      "template <> struct weft_vt<2> {\n"
      "  using type = weft_f2;\n"
      "  static type mk(weft_n t) { return weft_f2{t.v[0], t.v[1]}; }\n"
      "};\n"
      "template <> struct weft_vt<3> {\n"
      "  using type = weft_f3;\n"
      "  static type mk(weft_n t) { return weft_f3{t.v[0], t.v[1], t.v[2]}; }\n"
      "};\n"
      "template <> struct weft_vt<4> {\n"
      "  using type = weft_f4;\n"
      "  static type mk(weft_n t) { return weft_f4{t.v[0], t.v[1], t.v[2], t.v[3]}; }\n"
      "};\n"
      "template <class V> struct weft_vlen;\n"
      "template <> struct weft_vlen<weft_f2> { enum { N = 2 }; };\n"
      "template <> struct weft_vlen<weft_f3> { enum { N = 3 }; };\n"
      "template <> struct weft_vlen<weft_f4> { enum { N = 4 }; };\n"
      "\n"
      "static inline float weft_comp(weft_f2 v, int k) { return k == 0 ? v.x : v.y; }\n"
      "static inline float weft_comp(weft_f3 v, int k) {\n"
      "  return k == 0 ? v.x : (k == 1 ? v.y : v.z);\n"
      "}\n"
      "static inline float weft_comp(weft_f4 v, int k) {\n"
      "  return k == 0 ? v.x : (k == 1 ? v.y : (k == 2 ? v.z : v.w));\n"
      "}\n"
      "\n"
      "// A matrix from its columns, which is the order a composite build arrives in.\n"
      "template <class V0, class... Rest>\n"
      "static inline weft_mat<weft_vlen<V0>::N, 1 + (int)sizeof...(Rest)> weft_mk(V0 p0,\n"
      "                                                                     Rest... rest) {\n"
      "  const V0 cols[1 + sizeof...(Rest)] = {p0, rest...};\n"
      "  weft_mat<weft_vlen<V0>::N, 1 + (int)sizeof...(Rest)> r{};\n"
      "  for (int c = 0; c < 1 + (int)sizeof...(Rest); c++)\n"
      "    for (int i = 0; i < weft_vlen<V0>::N; i++) r.m[c][i] = weft_comp(cols[c], i);\n"
      "  return r;\n"
      "}\n"
      "\n"
      "// One column of a matrix, as the vector of its rows.\n"
      "template <int R, int C>\n"
      "static inline typename weft_vt<R>::type weft_col(weft_mat<R, C> a, int c) {\n"
      "  weft_n t{};\n"
      "  for (int i = 0; i < R; i++) t.v[i] = a.m[c][i];\n"
      "  return weft_vt<R>::mk(t);\n"
      "}\n"
      "\n"
      "// Matrix times vector: row `i` of the result is the dot of row `i` with the vector,\n"
      "// which is `m[0][i] * v[0]` and then one fused multiply-add a column.\n"
      "template <int R, int C>\n"
      "static inline typename weft_vt<R>::type operator*(weft_mat<R, C> a,\n"
      "                                                 typename weft_vt<C>::type v) {\n"
      "  weft_n t{};\n"
      "  for (int i = 0; i < R; i++) {\n"
      "    float acc = a.m[0][i] * weft_comp(v, 0);\n"
      "    for (int k = 1; k < C; k++) acc = std::fma(a.m[k][i], weft_comp(v, k), acc);\n"
      "    t.v[i] = acc;\n"
      "  }\n"
      "  return weft_vt<R>::mk(t);\n"
      "}\n"
      "\n"
      "// Vector times matrix: the vector multiplies the rows, so column `c` of the result is\n"
      "// the dot of the vector with column `c`.\n"
      "template <int R, int C>\n"
      "static inline typename weft_vt<C>::type operator*(typename weft_vt<R>::type v,\n"
      "                                                 weft_mat<R, C> a) {\n"
      "  weft_n t{};\n"
      "  for (int c = 0; c < C; c++) {\n"
      "    float acc = weft_comp(v, 0) * a.m[c][0];\n"
      "    for (int k = 1; k < R; k++) acc = std::fma(weft_comp(v, k), a.m[c][k], acc);\n"
      "    t.v[c] = acc;\n"
      "  }\n"
      "  return weft_vt<C>::mk(t);\n"
      "}\n"
      "\n"
      "// Matrix times matrix: each column of the result is the first matrix times the\n"
      "// matching column of the second.\n"
      "template <int R, int C, int P>\n"
      "static inline weft_mat<R, P> operator*(weft_mat<R, C> a, weft_mat<C, P> b) {\n"
      "  weft_mat<R, P> r{};\n"
      "  for (int c = 0; c < P; c++)\n"
      "    for (int i = 0; i < R; i++) {\n"
      "      float acc = a.m[0][i] * b.m[c][0];\n"
      "      for (int k = 1; k < C; k++) acc = std::fma(a.m[k][i], b.m[c][k], acc);\n"
      "      r.m[c][i] = acc;\n"
      "    }\n"
      "  return r;\n"
      "}\n"
      "\n"
      "template <int R, int C>\n"
      "static inline weft_mat<R, C> operator*(weft_mat<R, C> a, float s) {\n"
      "  weft_mat<R, C> r{};\n"
      "  for (int c = 0; c < C; c++)\n"
      "    for (int i = 0; i < R; i++) r.m[c][i] = a.m[c][i] * s;\n"
      "  return r;\n"
      "}\n"
      "template <int R, int C>\n"
      "static inline weft_mat<R, C> operator*(float s, weft_mat<R, C> a) { return a * s; }\n"
      "\n"
      "template <int R, int C>\n"
      "static inline weft_mat<C, R> weft_transpose(weft_mat<R, C> a) {\n"
      "  weft_mat<C, R> r{};\n"
      "  for (int c = 0; c < C; c++)\n"
      "    for (int i = 0; i < R; i++) r.m[i][c] = a.m[c][i];\n"
      "  return r;\n"
      "}\n"
      "static inline float weft_f32(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }\n"
      "static inline double weft_f64(uint64_t b) { double d; std::memcpy(&d, &b, 8); return d; }\n"
      "static inline int32_t weft_f2i32(double x) {\n"
      "  if (x != x) return 0;\n"
      "  if (x >= 2147483648.0) return 2147483647;\n"
      "  if (x <= -2147483649.0) return (-2147483647 - 1);\n"
      "  return (int32_t)x;\n"
      "}\n"
      "static inline uint32_t weft_f2u32(double x) {\n"
      "  if (x != x) return 0;\n"
      "  if (x >= 4294967296.0) return 4294967295u;\n"
      "  if (x <= -1.0) return 0u;\n"
      "  return (uint32_t)x;\n"
      "}\n"
      "static inline int64_t weft_f2i64(double x) {\n"
      "  if (x != x) return 0;\n"
      "  if (x >= 9223372036854775808.0) return (int64_t)9223372036854775807ll;\n"
      "  if (x <= -9223372036854775808.0) return (int64_t)(-9223372036854775807ll - 1);\n"
      "  return (int64_t)x;\n"
      "}\n"
      "static inline uint64_t weft_f2u64(double x) {\n"
      "  if (x != x) return 0;\n"
      "  if (x >= 18446744073709551616.0) return 18446744073709551615ull;\n"
      "  if (x <= -1.0) return 0u;\n"
      "  return (uint64_t)x;\n"
      "}\n"
      "template <class T, class U>\n"
      "// A narrow source, an integer literal under a 64-bit destination, fills the low\n"
      "// bytes of a zeroed destination instead of reading past itself.\n"
      "static inline T weft_bitcast(U u) {\n"
      "  T t{};\n"
      "  std::memcpy(&t, &u, sizeof(U) < sizeof(T) ? sizeof(U) : sizeof(T));\n"
      "  return t;\n"
      "}\n"
      // A bit reversal written as masks, not as a builtin.
      "static inline uint32_t weft_bitrev32(uint32_t v) {\n"
      "  v = ((v >> 1) & 0x55555555u) | ((v & 0x55555555u) << 1);\n"
      "  v = ((v >> 2) & 0x33333333u) | ((v & 0x33333333u) << 2);\n"
      "  v = ((v >> 4) & 0x0F0F0F0Fu) | ((v & 0x0F0F0F0Fu) << 4);\n"
      "  v = ((v >> 8) & 0x00FF00FFu) | ((v & 0x00FF00FFu) << 8);\n"
      "  return (v >> 16) | (v << 16);\n"
      "}\n"
      "static inline uint64_t weft_bitrev64(uint64_t v) {\n"
      "  return ((uint64_t)weft_bitrev32((uint32_t)v) << 32) |\n"
      "         (uint64_t)weft_bitrev32((uint32_t)(v >> 32));\n"
      "}\n"
      "template <class T>\n"
      "static inline T weft_ld(const void* p) { T t; std::memcpy(&t, p, sizeof(T)); return t; }\n"
      "template <class T>\n"
      "static inline void weft_st(void* p, T v) { std::memcpy(p, &v, sizeof(T)); }\n";
  code += weft_math_preamble();
  code += "\n" + kernels + "static const weft::KernelDef weft_table[] = {\n";
  for (const auto& k : out) {
    code += "  {\"" + k.name + "\", nullptr, " + std::to_string(k.nargs) + ", " +
            (k.sync ? "true" : "false") + ", nullptr, weft_k_" + k.name + ", sizeof(weft_s_" +
            k.name + "), nullptr, nullptr, 0, weft_g_" + k.name + "},\n";
  }
  code +=
      "};\n"
      "extern \"C\" void weft_set_param(int i, uint64_t v) { weft::set_param(i, v); }\n"
      "extern \"C\" void weft_set_texture(int i, void* d, uint32_t w, uint32_t h, uint32_t e) {\n"
      "  weft::set_texture(i, d, w, h, e);\n"
      "}\n"
      "extern \"C\" void weft_set_texture_layers(int i, void* d, uint32_t w, uint32_t h,\n"
      "                                       uint32_t e, uint32_t l) {\n"
      "  weft::set_texture_layers(i, d, w, h, e, l);\n"
      "}\n"
      "extern \"C\" void weft_run(int i, void** bufs, uint32_t grid, uint32_t tg) {\n"
      "  weft::dispatch(weft_table[i], bufs, grid, tg);\n"
      "}\n";
  // A width the caller states wins over anything this front end could infer.
  for (const auto& w : widths) {
    if (w.first < 0) continue;
    for (auto& k : out) {
      if (k.buf_type.size() < (size_t)k.nbufs) k.buf_type.resize((size_t)k.nbufs);
      for (auto& t : k.buf_type)
        if (t.empty()) t = "0";  // unknown: the ABI reports 0 rather than a guess
      if ((size_t)w.first < k.buf_type.size())
        k.buf_type[(size_t)w.first] = std::to_string(w.second);  // digits: a byte count
    }
  }
  code += emit_module_abi(out);

  r.code = code;
  r.ok = true;
  r.kernels = out;
  return r;
}

TranslateResult translate_spirv(const std::string& raw,
                              const std::vector<std::pair<int, uint32_t>>& widths) {
  return translate_spirv_impl(raw, widths, false);
}

TranslateResult translate_spirv_gpu(const std::string& raw,
                                    const std::vector<std::pair<int, uint32_t>>& widths) {
  return translate_spirv_impl(raw, widths, true);
}
