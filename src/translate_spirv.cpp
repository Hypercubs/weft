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
//
// A SPIR-V kernel is SSA with structured control flow: every value is computed
// once, blocks are joined with OpPhi, and a barrier is an instruction rather than
// a keyword. The translation keeps all of that visible: each SSA result becomes a
// field of a per-thread state struct, each block becomes a label, and a barrier
// saves a resume point, the same state machine the PTX front end emits, so the
// runtime does not need to know which front end produced a kernel.
//
// A phi is resolved the classic way: the edge from block P to block B copies B's
// value for P into a temporary before the jump, and B reads it on entry. The
// copies go through locals so a two-way swap cannot clobber itself.
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
  // atomic float min/max (SPV_EXT_shader_atomic_float_min_max). The numbers are
  // the ones the assembler wrote, read back out of the binary with a word dump
  // rather than taken from memory.
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
  // `OpExecutionMode LocalSize x y z`: the group the module was written to run
  // in. A one-dimensional group is a default for the launch; a 2-D or 3-D one is
  // a constraint, because the lane identity is decomposed by its shape.
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
    // An import has no type operand: its first word is the result id and the rest
    // is the set's name. Without this it was read as "type, result", so the string
    // literal's words became a value id and a type id, which showed up much later
    // as a state field that referred to a number nobody had declared.
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
      // An image is a handle. What this runtime needs from the declaration is the
      // sampled element type (what a read hands back) and whether the image is
      // sampled or storage; the dimensionality it gets from the coordinate.
      case OpTypeImage: {
        if (ins.a.size() < 8) { err = "bad OpTypeImage"; return false; }
        SType t;
        t.kind = KImage;
        t.elem = ins.a[1];   // sampled type
        t.count = ins.a[2];  // Dim
        // `count` keeps the Dim, and the two shapes that are not a plane are read
        // back from it: 0 is 1-D, which is a plane one row high, and 3 is a cube,
        // six square faces sampled by direction. Both are legal in the Shader
        // model, `Sampled1D`, `Image1D` and `Shader` are the capabilities the
        // validator asks for, and a GLCompute module may declare all of them, so
        // they are decoded rather than named.
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
        // `OpTypeMatrix %column %count`: a matrix is a list of columns, and the
        // column type is the vector type it is built from. SPIR-V wants those
        // columns to be floating-point; they are 32-bit here, and the shape is
        // square because those are the ones a kernel multiplies.
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
        // Null of a scalar is 0; null of a composite is a zeroed aggregate, which
        // the emitter handles by zeroing the storage it initialises.
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
        // `LocalSize x y z`, all three kept. A one-dimensional group only sets a
        // default launch width; a wider one also fixes how a lane id decomposes,
        // so its product becomes the launch width rather than a suggestion.
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
          // A declaration is `OpFunction ... OpFunctionParameter* OpFunctionEnd`
          // with no label in between: no body means it is an import.
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
  // The warp target: one Metal thread runs all 32 lanes, so a peer's value is this
  // thread's own data and a 64-bit pair has exactly one writer.
  bool warp_ = false;
  // A warp kernel with more than one block carries a program counter: the lane
  // loop finishes a lane before it starts the next, so one counter is enough, but
  // a branch has to re-enter the block dispatch instead of falling through.
  bool warp_blocks_ = false;
  // Step mode: one program counter per lane and a dispatch that runs the lowest
  // one first. It is what a barrier needs, because a lane that parks on the
  // barrier's block cannot be passed by any lane that has not arrived yet.
  bool warp_step_ = false;
  void warp_target(bool blocks, bool step) {
    warp_ = true;
    warp_blocks_ = blocks;
    warp_step_ = step;
  }
  bool warp_blocks() const { return warp_blocks_; }
  bool warp_step() const { return warp_step_; }
  // The largest case id a step-mode kernel uses: the dispatch sweeps up to it.
  int max_case() const { return max_case_; }
  // How many words the warp's slot array needs: one window of 32 per rendezvous
  // site, and at least one window.
  int slot_words() const { return slot_off_ < 32 ? 32 : slot_off_; }
  // Why the warp target was not available, when it was not. The refusal for a pair
  // the per-lane target cannot carry names it, so the message says which two
  // shapes disagreed rather than just that they did.
  void warp_reason(const std::string& why) { warp_why_ = why; }
  // The kernel takes the 64-bit lock: the launcher has to bind the table of lock
  // words for it, and the kernel declares that binding.
  bool wide_atom = false;
  // A 64-bit counter whose address is the same for every lane is folded into a
  // local and flushed once per kernel. A `red` is order-free, so folding in the
  // walk and adding once is the same answer as one atomic per lane, and it is
  // 1,048,576 pairs of device-scope atomics turned into one. Each of those is an
  // L2 round trip, which is what made the 1M-lane 64-bit counter 825 ms.
  struct RedAcc {
    std::string addr;  // the pointer expression, as emitted
    std::string op;    // add, sub, and, or, xor
    std::string name;  // the local accumulator
  };
  std::vector<RedAcc> red_accs;
  // A 64-bit word that nothing else in the kernel touches can keep its value in a
  // local for the whole walk: the old values still come back in program order, and
  // the buffer is read once and written once instead of twice per lane. See
  // WEFT_SH64 in src/weft_msl.hpp.
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

  // One element of a binding, in bytes. A storage buffer's pointer names an
  // array (usually a runtime array) and a caller sizes buffers in elements, so
  // the answer is the element's stride: the module's own ArrayStride when it
  // decorates one, otherwise the element type's size. SPIR-V states its layout
  // explicitly (ArrayStride, Offset) where C++ leaves it to the compiler, so
  // this is a number rather than a `sizeof`.
  std::string bytes_of(uint32_t pointee) const {
    auto t = m_.types.find(pointee);
    if (t == m_.types.end()) return "";
    uint32_t id = pointee;
    // `__global float* p` at module scope is a *pointer* global: what the caller
    // supplies is the float array, so the width is a float, not the 8-byte
    // pointer the variable happens to hold.
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
  // The second back end, chosen by the caller: the same decode, Metal instead of
  // C++. A member function rather than a public field so the flag stays with the
  // emission state it belongs to.
  void gpu_target(bool on) { gpu = on; }
  // The threadgroup arena this kernel needs; the GPU target declares it.
  uint32_t arena_bytes() const { return wg_bytes_; }
  // The access mode of each `[[texture(N)]]` parameter, in binding order: part
  // of the parameter's type, so the caller emits the signature from these.
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
  // Does anything read this id? An atomic whose result nobody reads is a
  // reduction: the discarded old value is the only thing `atom` has that `red`
  // does not, and the reduction form is the one that exists at 64 bits on the GPU
  // (src/weft_msl.hpp folds the halves with two 32-bit atomics). The definition
  // contributes one mention, so one mention means unused, and a use anywhere in
  // the function, reachable from this block or not, means live.
  // Is this pointer used by anything but the instruction asking? Folding a
  // counter into a local is the same program only while the word is touched by
  // nothing else in the kernel: a later load through the same pointer, or a read
  // after a barrier, has to see the increments, and the local is paid back only
  // once the whole walk is over.
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
  // Pointer id -> the MSL address space it points into. A GPU pointer carries
  // its space in its type, so a chain into threadgroup memory has to be cast to
  // a threadgroup pointer, not a device one, all the way to the field that holds
  // it. Empty on the C++ back end, where a pointer is just a pointer.
  std::map<uint32_t, std::string> ptr_space_;
  // Pointer fields, declared after the blocks rather than with the rest: a field
  // that holds an address is typed by the space the emitting code put the
  // address in, so it cannot be declared before the code that fills it exists.
  std::vector<uint32_t> deferred_ptr_;
  // A texture's access mode is part of its MSL type, so it is decided once for
  // the whole kernel. `tex_access` is one entry per `[[texture(N)]]` parameter.
  std::vector<std::string> tex_access;
  std::vector<bool> tex_arrayed;
  std::vector<bool> tex_vol;      // the same, for `Dim = 3D`
  std::vector<bool> tex_cube;     // and for `Dim = Cube`, which is filter-only
  // What each binding is, by name, so the launcher can check the shape it was
  // handed against the shape the module declared.
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
  // Emitted for the GPU: the same decode, Metal instead of C++. One flag, and the
  // handful of places below where the two targets word the same instruction
  // differently are the only things it switches.
  bool gpu = false;
  std::string warp_why_ = "needs something the warp target cannot provide";
  // The largest case id a step-mode kernel uses: the dispatch sweeps up to it.
  int max_case_ = 0;
  int slot_off_ = 0;  // words of the slot array handed out so far
  uint32_t cur_block_ = 0;
  uint32_t wg_bytes_ = 0;
  int nbuffers_ = 0, nscalars_ = 0, ntex_ = 0, nsamp_ = 0;
  std::vector<uint32_t> buf_elem_;  // pointee type of each binding, in bind order

  // The space a pointer id points into. A chain result carries it explicitly; a
  // buffer or a local array says it in its address expression.
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
  // A math call by the name the target gives it: `std::` on the CPU, the bare
  // name on the GPU, where `using namespace metal` puts the same functions in
  // scope with the same rounding.
  std::string math(const std::string& name, const std::string& args) const {
    return gpu ? (name + "(" + args + ")") : ("std::" + name + "(" + args + ")");
  }
  // A branch: a goto on the CPU, and a program counter with a restart on the
  // GPU, where there is no goto (tools/msl_probe.mm).
  std::string jump(uint32_t label) const {
    if (!gpu) return "goto L" + std::to_string(label) + ";";
    if (warp_step_) return "{ s->pc[l] = " + std::to_string(label) + "u; continue; }";
    return "{ s->pc = " + std::to_string(label) + "u; continue; }";
  }
  // What this host's Metal compiler will not take, measured rather than assumed
  // (tools/msl_bisect.py over the emitted source): neither a loop-carried minimum
  // over the lane counters nor a `switch` whose selector is read from them, both
  // take the back end down with XPC_ERROR_CONNECTION_INTERRUPTED, which names the
  // transport and not the construct. A selector that is the dispatch's own scalar
  // loop counter is fine, so the sweep below is what picks a block, and a lane's
  // counter is only ever compared against it.
  std::string case_open(uint32_t id) const {
    const std::string t = std::to_string(id);
    return "      case " + t + "u: { for (uint32_t l = 0; l < 32u; l++) {\n"
           "        if (weft_base + l >= weft_lanes) break;\n"
           "        if (s->pc[l] != " + t + "u) continue;\n"
           "        weft_ran = true;\n"
           "        ctx.lid = l; ctx.gid = (uint64_t)weft_base + (uint64_t)l;\n";
  }
  std::string case_close() const { return "      } } break;\n"; }
  // Park a lane on `id`: it writes its own program counter and the block it is
  // heading for becomes a case of its own, so nothing past this point runs for
  // that lane until the next sweep comes back to it.
  std::string park(uint32_t id) {
    if (id > max_case_) max_case_ = (int)id;
    return "{ s->pc[l] = " + std::to_string(id) + "u; continue; }\n" + case_close() +
           case_open(id);
  }
  // The warp's slot array is carved into windows of 32, one per rendezvous site.
  // A window per site rather than one shared array, because lanes interleave inside
  // a case: a lane that publishes the next word would otherwise overwrite a slot
  // another lane has not read yet, which is exactly how the first attempt at the
  // 64-bit broadcast came out wrong (measured: 17 distinct ballots and two
  // different slot values where there must be one of each).
  // Which lanes of this warp exist: the last warp of a group is short when the lane
  // count is not a multiple of 32, and a vote over it must not count lanes that
  // never ran.
  std::string slot_mask() const {
    return "((weft_lanes > weft_base) ? (((weft_lanes - weft_base) >= 32u) ? 0xFFFFFFFFu : "
           "((1u << (weft_lanes - weft_base)) - 1u)) : 0u)";
  }
  std::string param_of(int i) const {
    const std::string n = std::to_string(i);
    return gpu ? ("weft::param(ctx, " + n + ")") : ("weft::param(" + n + ")");
  }
  // The identity of a min/max fold: Metal has no numeric_limits, so the infinite
  // values are the constants.
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
  bool chain(const Instr& ins);
  bool emit_one(const Instr& ins);
  bool emit_import(const Instr& ins, const std::string& sym);
  // A defined function is inlined where it is called: these are the call's own state,
  // saved and restored around the body it splices in.
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
      // A 16-bit float. `_Float16` is two bytes wide and converts to a float, which
      // is exactly the pair a texel read needs: `tex_read<T>` memcpys `sizeof(T)`
      // against the image's element width, and `tex_at_f<T>` applies `(float)T`. That
      // is why an `R16Float` texel is this type and not `uint16_t`, the same width,
      // but the bit pattern read as an integer (`0x3C00` answering 15360). MSL spells
      // the type `half`, and `weft_h16` is the decode underneath both.
      if (t.width == 16) { out = gpu ? "half" : "_Float16"; return true; }
      if (t.width == 32) { out = "float"; return true; }
      // Metal has no 64-bit float, so on the GPU a double is carried as the uint64_t
      // its bits live in and every operation on it is a `weft_fp64` helper. Off the
      // GPU it is the host's own double.
      if (t.width == 64) { out = gpu ? "uint64_t" : "double"; return true; }
      return false;
    case KPointer:
      // An address points into a buffer, which on the GPU is the device address
      // space and has to say so.
      out = gpu ? "device unsigned char*" : "unsigned char*";
      return true;
    case KImage: {
      // On the GPU an image is the bound texture plus its size, held by value.
      // A kernel that writes it needs the read_write access mode, and an arrayed
      // image is a different handle again, `texture2d_array`, with the slice
      // count alongside, so those two choices are the type.
      // `count` is the declaration's Dim: 0 is 1-D, 1 is 2-D, 2 is 3-D and 3 is a
      // cube. A volume is its own handle on the GPU because `texture3d` is its own
      // type there, and a cube is `texturecube`, which cannot be read by texel at
      // all, a cube is filterable only.
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
      // A sampler on the GPU is the bound state object the parameter names, so
      // the type is the language's own `sampler`.
      out = gpu ? "sampler" : "weft::Sampler";
      return true;
    case KSampledImage: {
      // The sampled-image type names its image, so an array image joins the
      // sampler in the array handle and a volume in the 3-D one.
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
      // The name carries both extents, from the column's own width and the
      // column count: `weft_f4x4` is four `weft_f4` columns. The preamble
      // defines the square ones for both targets, so the name is the type.
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
        // A vector of doubles: the same software double 2, 3 or 4 times over, so
        // every operation on one is the fp64 helper applied per component rather
        // than an operator on two 64-bit halves.
        pre = "weft_d";
      }
      else if (et.kind == KFloat && et.width == 16) {
        // A vector of halves is carried as a vector of floats. An `OpImageRead` of
        // an `R16Float` image hands back `%v4h`, and its components are widened as
        // they are built, each is a texel read through the half element, which is
        // a float already. A second family of vector types whose only difference is
        // a narrowing no kernel here asks for would be the other way to spell it.
        pre = "weft_f";
      }
      else if (et.kind == KInt && et.width == 32) pre = et.is_signed ? "weft_i" : "weft_u";
      else if (et.kind == KInt && et.width == 64) pre = et.is_signed ? "weft_l" : "weft_ul";
      // A vector of booleans: the condition of a componentwise select, which a
      // GLSL emitter produces for `mix` and for a comparison folded into a
      // ternary. The GPU has `bool2/3/4` already; the CPU needs the same shape.
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

// A value's reference: `s->v20` on the per-lane target, `s->v20[l]` when one
// thread owns all 32 lanes and the field is an array of them.
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
  // A pointer that an access chain produced is an SSA value like any other, so
  // on the warp target it is one per lane too. The address it holds is a device
  // address, which is the same on every lane when the lanes differ, but a
  // kernel that walks a pointer per lane has one per lane, so index it.
  return warp_ ? fref(id) : "s->" + field(id);
}

uint32_t Emitter::vtype(uint32_t id) const {
  auto c = m_.ctype.find(id);
  if (c != m_.ctype.end()) return c->second;
  auto t = vt_.find(id);
  return t == vt_.end() ? 0u : t->second;
}

// A 64-bit float. On the GPU its values are carried as the uint64_t their bits live
// in, because Metal rejects the type; the CPU target uses the host's double.
bool Emitter::f64_type(uint32_t tid) const {
  auto t = m_.types.find(tid);
  return t != m_.types.end() && t->second.kind == KFloat && t->second.width == 64;
}

// A builtin variable's value, or one component of it (comp < 0 means the whole
// thing). The launch geometry is all in Ctx, so this is a plain expression.
std::string Emitter::builtin_expr(uint32_t var, int comp) const {
  auto g = m_.globals.find(var);
  if (g == m_.globals.end() || !g->second.has_builtin) return "";
  const uint32_t bi = g->second.builtin;
  // The group shape the module declared, when it declared one. A 2-D group's lane
  // `l` is (l % x, l / x), so the lane identity comes from the shape and not from
  // the launch width alone, which is why a shape wider than a row also fixes the
  // launch (`fixed_threads`).
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
  // The width is the module's, not this translator's. OpenCL declares
  // `GlobalInvocationId` as a `size_t` vector, which is 64-bit here; GLSL declares
  // the same builtin as a `uvec3`, which is 32-bit. The launch geometry is 64-bit
  // either way, so the cast has to narrow to whatever the declaration holds , 
  // picking it from the builtin's name was right for one front end's convention
  // and an error for the other's.
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

// The rendezvous slot is 32 bits wide and carries the value's bits, not its
// value: a float published with `(uint32_t)x` would arrive as its integer part.
// This is the one-word form, used by the votes and by the reductions; a shuffle
// of something wider goes through `wide_shfl`, which publishes a word per pass.
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

// A shuffle of anything wider than one 32-bit word, a 64-bit value, or a
// vector, is published one word per pass: the rendezvous slot is one word per
// lane, so a wider value takes several suspensions, and they compose because
// each pass carries exactly one word. `peer` is the expression for the lane to
// read from (already clamped by the caller).
bool Emitter::wide_shfl(uint32_t tid, const std::string& e, const std::string& peer,
                        std::string& out) {
  auto it = m_.types.find(tid);
  if (it == m_.types.end()) { err = "a shuffle of an unknown type"; return false; }
  const SType t = it->second;
  subgroup_ = true;

  // Publish this word, wait for the pass, and keep the peer's word.
  //
  // The read has to happen at the resume label, not later where the expression
  // is used: `shfl_read_buf` reads the half of the *pass the read happens in*,
  // and a two-word value suspends twice, so an expression held to the end would
  // read the later half twice and the low word would come back as the high one.
  auto pass = [&](const std::string& payload) -> std::string {
    if (gpu) {
      if (warp_step_) {
        // One thread owns these lanes, so the word moves through a window of the
        // warp's slot array: every lane publishes it, the sweep brings the lanes
        // back, and the peer's slot is read on the way in.
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
      // One instruction: the hardware moves the word between lanes, so there is
      // no second pass and nothing to save. `peer` is the absolute lane; modulo
      // the warp's width that is the index simd_shuffle takes.
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
  //
  // A double has to go out as its *bits*: `(uint64_t)1.0` is 1, whose pattern is
  // five thousand times smaller than the smallest subnormal, so a cast here would
  // deliver zero for every value that is not already an integer pattern. The GPU
  // target already holds the bits in a `uint64_t`, which is why this is the CPU
  // side of the same expression.
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


// OpAccessChain and friends. The address is materialised into the state even
// when it looks recomputable: the index it was built from is usually a loop
// counter whose state field is overwritten on the next iteration, so an
// expression that reads it later would compute a different address.
bool Emitter::chain(const Instr& ins) {
  const uint32_t rty = ins.a[0], rid = ins.a[1], base_id = ins.a[2];

  // A chain into a builtin variable is not memory: it is one component of the
  // launch geometry.
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
// into a local first, then write them all, so a swap (a<->b) works.
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

  std::string rty;
  if (res && !ctype(rt, rty)) {
    err = "result of opcode " + std::to_string(op) + " has an unsupported type";
    return false;
  }
  auto V = [&](size_t k) -> std::string {
    if (k >= a.size()) { err = "opcode " + std::to_string(op) + " is missing operands"; return "0"; }
    return val(a[k]);
  };
  auto set = [&](const std::string& e) { body_ += "  " + fref(rd) + " = " + e + ";\n"; };

  // Inside an inlined call a `return` ends the call: the value goes to the id the call
  // was made for and the rest of the helper's body is skipped. The kernel's own return
  // is the same instruction with no call open, and falls through to its handler below.
  if (in_call_ && (op == OpReturn || op == OpReturnValue)) {
    if (op == OpReturnValue) body_ += "  " + fref(call_ret_) + " = " + val(a[0]) + ";\n";
    call_done_ = true;
    return true;
  }

  //
  // Metal rejects `double` outright, the type does not exist on this hardware, so a
  // kernel carrying one could not be compiled at all. Every GPU without fp64 carries
  // one in software, and this target does the same: `ctype` hands a double the
  // `uint64_t` its bits live in, and each operation below is an exact IEEE-754 helper
  // from `weft_fp64`. Memory, `select`, `phi`, a copy and a shuffle need nothing: they
  // move the bits, and the bits are the value.
  //
  // A `double` goes through `weft_fp64` on both columns (src/weft_f64.hpp): the GPU
  // has no fp64 at all, and the host's libm is not a reference either, its `exp`
  // differs from the correctly rounded one in the last bit on some inputs, so a
  // comparison against it would report a disagreement that is nobody's kernel. One
  // text compiled twice answers with the same bits on either side, and that text is
  // swept against an exact rational reference rather than against a library
  // (`tools/f64_math.py`).
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
      // The "half_" and "native_" spellings are the same functions: there is no
      // reduced-precision path here, and the precise answer beats refusing a name.
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
        // On the GPU the value already is the word its bits live in; on the CPU it
        // is the host's double, so it is reinterpreted on the way in and out.
        return gpu ? v : ("weft_bitcast<uint64_t>(" + v + ")");
      };
      std::string call = std::string("weft_fp64::") + fn + "(" + operand(0);
      if (arity == 2 && a.size() > 5)
        call += ", " + (int_arg ? ("(int32_t)(" + V(5) + ")") : operand(1));
      call += ")";
      set(gpu ? call : ("weft_bitcast<double>(" + call + ")"));
      return true;
    }
    // The four that answer with two numbers: the second one is written through the
    // pointer the instruction carries as its last operand, and on this target a
    // `double` there is a `uint64_t` of bits on the GPU and the host's own double on
    // the CPU, so the store takes the same conversion the result does.
    // `fract` is one of these over a double as well: the fraction is the result and
    // the floor goes through the pointer, which is what instruction 30 says.
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

  // The square root over a double on the CPU target: the host's own `std::sqrt`, which
  // is the correctly rounded answer the software one is compared against. It is written
  // here rather than routed through the portable library because that library's `sqrt`
  // is a float overload set and a double in the middle of it is ambiguous.
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
        case OpConvertFToS: set("weft_fp64::to_i64(" + V(2) + ")"); return true;
        case OpConvertFToU: set("weft_fp64::to_u64(" + V(2) + ")"); return true;
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
            // Fused multiply-add: one rounding, and the product is 106 bits wide
            // before it is, so it cannot be built out of a multiply and an add.
            case 26: set("weft_fp64::fma(" + x + ", " + V(5) + ", " + V(6) + ")"); return true;
            // The square root is the one member of that library IEEE pins exactly, so
            // it is carried here rather than left to the host.
            case 61: set("weft_fp64::sqrt_bits(" + x + ")"); return true;
            default:
              err = "OpenCL.std instruction " + std::to_string(num) +
                    " over a double has no GPU form: the arithmetic, the comparisons,"
                    " the conversions, floor/ceil/trunc/round, min/max, sqrt and fma do";
              return false;
          }
        }
        // The bits move and the bits are the value, so these fall through to the
        // ordinary translation: `break` leaves this switch, not the function.
        case OpLoad: case OpStore: case OpSelect: case OpPhi: case OpCopyObject:
        case OpVariable: case OpFunctionParameter: case OpCompositeExtract:
        case OpCompositeConstruct: case OpCompositeInsert: case OpVectorShuffle:
        // A subgroup shuffle of a double is the same instruction as a shuffle of a
        // `long`: it takes the value apart into words and moves them, which is what
        // the shuffle path already does for every 64-bit type. On the GPU the value
        // is the `uint64_t` its bits live in, so there is nothing to convert.
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

  //
  // A double is the `uint64_t` its bits live in, so a vector of them is a struct of
  // those words and each operation is the helper applied per component. On the CPU
  // the components are the host's own doubles, where the operator is already right.
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
      // A function-scope variable: storage for an array or an address-taken
      // scalar. It is a field of the state, like every other SSA value.
      //
      // `OpVariable` is in `no_result`, the field pass treats it as storage rather
      // than as a value, so the shared `rd` is zero for it and the result id has to
      // come from the instruction's own second word. Reading `rd` declared a field
      // named for value 0, which no load could ever name: an address-taken local
      // translated, then read someone else's slot.
      const uint32_t vid = a.size() > 1 ? a[1] : 0u;
      auto t = m_.types.find(a[0]);
      if (t == m_.types.end() || t->second.kind != KPointer) { err = "bad OpVariable"; return false; }
      uint32_t sz = 0;
      if (!tsize(t->second.elem, sz)) { err = "local variable of an unsupported type"; return false; }
      if (sz == 0) sz = 1;
      // One thread walking 32 lanes needs one copy of a local per lane: the lanes
      // are data there, not threads, so the storage is a row each.
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
      // An image or a sampler parameter names a binding rather than storage, so
      // loading it hands back the host's handle.
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
          // Indexing a matrix yields a column, which is the vector it is built
          // from: `weft_col(m, i)` is the same accessor the preamble's product
          // uses, so a caller and the arithmetic cannot disagree about which is
          // a column and which is a row.
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
      // A matrix is built from its columns, in order: `weft_mk(c0, c1, ...)`
      // takes the column type, so a caller cannot pass rows without the compiler
      // saying so.
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

    //
    // The preamble's operators are the arithmetic, and the preamble is one text
    // compiled by both targets, so a product here is the product a GPU runs
    // rather than a second implementation kept in step by hand. Column-major on
    // both sides: `m[c]` is a column, and a matrix times a vector is a linear
    // combination of them.
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

    case OpIAdd: case OpFAdd: set(is_vec(rt) ? compwise(V(2), V(3), "+") : V(2) + " + " + V(3)); return true;
    case OpISub: case OpFSub: set(is_vec(rt) ? compwise(V(2), V(3), "-") : V(2) + " - " + V(3)); return true;
    case OpIMul: case OpFMul: set(is_vec(rt) ? compwise(V(2), V(3), "*") : V(2) + " * " + V(3)); return true;
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

    case OpConvertFToS: set("(int64_t)" + V(2)); return true;
    case OpConvertFToU: set("(uint64_t)" + V(2)); return true;
    case OpConvertSToF: case OpConvertUToF: set("(" + rty + ")" + V(2)); return true;
    case OpUConvert: case OpSConvert: case OpFConvert: {
      uint32_t en = 0, n = 0;
      if (vector_of(rt, en, n) && !is_vec(vtype(a[2]))) {
        // A scalar converted into a vector: the backend emits this for a
        // component of a vector builtin id that was loaded on its own.
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
      // A vector condition selects component by component, which is the one thing
      // `?:` cannot say: it takes a scalar. A scalar condition is the ordinary
      // expression, and it is what every OpenCL module writes.
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
      const std::string sel = "((uint64_t)" + V(0) + ")";
      body_ += "  switch (" + sel + ") {\n";
      for (size_t k = 1; k + 1 < a.size(); k += 2) {
        body_ += "    case " + std::to_string((unsigned long long)a[k]) + "ull: { " +
                 edge_copies(cur_block_, a[k + 1]) + jump(label_of_[a[k + 1]]) + " }\n";
      }
      body_ += "    default: { " + edge_copies(cur_block_, a.back()) + jump(label_of_[a.back()]) +
               " }\n  }\n";
      return true;
    }
    case OpReturn: case OpReturnValue: case OpKill: case OpUnreachable:
      // On the warp target this ends the lane, not the thread: the lane loop
      // moves to the next lane, which is what a hardware lane does when it exits.
      // With more than one block the lane's body is a dispatch loop of its own, so
      // the end of the lane is the sentinel that loop reads at its head, a bare
      // `continue` there would re-enter the same case for ever.
      if (warp_step_) body_ += "  { s->pc[l] = 0xFFFFFFFFu; continue; }\n";
      else if (warp_)
        body_ += warp_blocks_ ? "  { s->pc = 0xFFFFFFFFu; continue; }\n" : "  continue;\n";
      else
        body_ += "  s->pc = 0xFFFFFFFFu; return;\n";
      return true;

    case OpControlBarrier: {
      barrier_ = true;
      if (warp_step_) {
        // A barrier between lanes that one thread runs is a hand-off between
        // them, not a hardware fence: park this lane and let the dispatch, which
        // always runs the lowest program counter first, bring it back only once
        // every lane has arrived. Nothing needs flushing, because one thread's
        // accesses are already in program order.
        body_ += "  " + park(resume_next_++) + "\n";
        return true;
      }
      if (gpu) {
        // The hardware's barrier: every thread of the group arrives, and nothing
        // is suspended, so there is no program counter to save.
        body_ += "  threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);\n";
        return true;
      }
      const int id = resume_next_++;
      cases_ += "    case " + std::to_string(id) + ": goto R" + std::to_string(id) + ";\n";
      body_ += "  s->pc = " + std::to_string(id) + "u; return;\n  R" + std::to_string(id) + ":;\n";
      return true;
    }

    //
    // Every one of these is a rendezvous among 32 lanes, and the only rendezvous
    // the runtime has is "hand the worker back and resume later". So: publish my
    // value, yield, then read what the warp published on the next pass.
    case OpGroupNonUniformElect: {
      // One lane of the group is chosen and which one is the implementation's
      // choice; the lowest lane is the one every engine here picks.
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
      // a[0]=type a[1]=result a[2]=execution scope, then the operands
      std::string payload;
      const bool is_ballot = op == OpGroupNonUniformBallot;
      if (is_ballot || op == OpGroupNonUniformAll || op == OpGroupNonUniformAny)
        payload = "(" + V(3) + ") ? 1u : 0u";

      if (op != OpGroupNonUniformBallot && op != OpGroupNonUniformAll &&
          op != OpGroupNonUniformAny && op != OpGroupNonUniformAllEqual) {
        // Shuffles: one suspension per 32-bit word of the value, so 64-bit and
        // vector types go a word at a time.
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
        // A quad operation addresses four lanes, so its base is the quad's; the rest
        // address the warp's 32. A swap direction names a lane to XOR, 1, 2, then 3
        // for the diagonal, rather than being used as one.
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

      // One thread owns these lanes, so a vote is a relay through the warp's slot
      // array: every lane publishes, the sweep brings them back, and the answer is
      // folded over the slots. The park is the rendezvous, see case_open.
      if (warp_step_) {
        const std::string wmask = slot_mask();
        if (op == OpGroupNonUniformAllEqual) {
          // The value's own words, one pass each: a word that differs anywhere
          // makes the answer false, and every word has to agree. The accumulator is
          // per lane because the passes interleave across lanes.
          //
          // A word of a value is its *bits*, not its value: a float published as
          // `(uint32_t)x` arrives as its integer part, so two different floats
          // between 1.0 and 2.0 would compare equal. The same holds for the halves of
          // a double, so each component is bitcast before it is split.
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
          // The accumulator is a member of the state, and one thread owns the
          // lanes, so it is written through `s->` with the lane index.
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
        // One broadcast of lane 0's value, one predicate, one ballot: the lanes
        // whose value differs from lane 0's are the lanes the mask marks. The
        // comparison is by value, which is what the instruction means, a bit
        // compare would call -0.0 and 0.0 different.
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
        // Every lane publishes its value's bits; the answer is whether they agree.
        // One rendezvous is enough because the published array holds all of them.
        //
        // A value wider than a word goes into the eight-byte wide slot instead of
        // the word-wide one, the same relay with a wider payload, not a second
        // mechanism, and the shuffle path already double-buffers those slots by
        // pass. Comparing the bytes covers every width it can hold at once: a long,
        // a double, a `uint4`. Nothing here needs to know which of those it is,
        // which is the point, the alternative is a refusal per type.
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
        // A ballot's result is a vec4 of 32-bit words, 128 bits of mask, of
        // which the first word is all a 32-wide subgroup can fill.
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

    // Counting the bits of a ballot is local work: every lane already holds the
    // same mask, so a prefix count needs no rendezvous.
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
        // The warp target is one thread for the whole grid, so nothing can
        // interleave with the pair: two 32-bit atomic loads are the 64-bit load,
        // read through the coherent point exactly as the pair updates are.
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
      // A 64-bit atomic on the GPU: `metal_atomic` rejects `atomic_ulong` for
      // every fetch-op, so one is built from the 32-bit atomics that exist. When
      // nobody reads the old value the operation is a reduction, and a reduction
      // of an order-free operation is two 32-bit atomics; otherwise the pair has
      // to move as one value under a lock. That decision is made below, once the
      // operand and the type are in hand.
      const bool dropped = unused_anywhere(rd);
      if (op == OpAtomicIIncrement || op == OpAtomicIDecrement) val_ = "1";
      else val_ = V(5);
      if (op == OpAtomicCompareExchange || op == OpAtomicCompareExchangeWeak) cmp = V(6);

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
          // A lane-dependent address cannot be hoisted: each lane's word is its
          // own, and folding them together would be a different program.
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
        //
        // The hang this used to show was not the lock. The runtime emulated every
        // 32-bit atomic with a compare-exchange retry loop, and a retry costs each
        // contender one line transfer per success, so one hot counter at 65,536
        // lanes cost about 4 billion of them and never finished. The proof that the
        // lock was innocent is the reduction-only module: no lock, no 64-bit
        // anything, same hang at 65,536 and 262,144. The fetch forms are native
        // now and that module is 0.029 ms at 262,144 lanes, exact. That fix stands
        // and applies to every GPU path, this one included.
        //
        // What was left was the pair itself, and it was never the lock's ordering
        // that was wrong: it was *which threads can be inside the critical section
        // at once*. On the per-lane target a lane can enter between the holder's
        // payload store and its unlock, because this Metal declares only
        // `memory_order_relaxed` and no instruction here orders two words without a
        // fence, so the values came out as garbage magnitudes rather than one lost
        // increment, at 16,384 and 65,536 lanes.
        //
        // The warp target has no lane to enter: one Metal thread owns all 32, so it
        // takes the lock, moves the pair and drops the lock with no other lane
        // inside. Every access to the pair is an atomic from that one thread. That
        // is why the PTX warp back end's 64-bit atomics are exact, and this branch
        // is the same helper it uses, the lock, the pair, the same protocol.
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
          // `_raw`: the warp body holds the launch's lock already, so the pair has
          // one writer without a second take.
          std::string call64;
          if (name64 == "cas")
            call64 = "weft::rmw_cas_solo(" + ptr64 + ", (" + t64 + ")(" + cmp + "), " + v64 +
                      ")";
          else call64 = "weft::rmw_" + name64 + "_solo(" + ptr64 + ", " + v64 + ")";
          // The local form is recorded here and only used if the emitted body
          // touches that binding once, which is checked once the body exists.
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
        // Why the per-lane back end cannot have this, measured rather than
        // assumed (tools/lock_probe.mm is the probe, and it is A/B: `control` and
        // `handoff` are exact at every shape while `locked`, `fenced` and
        // `exchanged` do not finish in any of them, including two threadgroups of
        // 32 threads). Ordering is not the obstacle, a device-scope
        // `atomic_thread_fence` exists and a relaxed store reaches a waiter in
        // another threadgroup. Progress is: a lane that wins a section cannot run
        // it until the other lanes in its warp leave the spin loop, and they
        // cannot leave until it releases, because a warp is one program counter. A
        // threadgroup of a single thread does not escape that either, such
        // threadgroups are co-scheduled into one warp and deadlock across each
        // other instead of within each. So this shape is refused by name rather
        // than emitted as a launch that hangs; the same kernel runs on the warp
        // target above, where one thread owns all 32 lanes and there is no section
        // to wait on at all.
        err = "a 64-bit atomic that reads the old pair back moves both words as one "
              "value, and only one writer can do that here: lanes in a warp cannot "
              "wait for each other (a warp has one program counter), so the per-lane "
              "back end cannot host the section. The warp target can, by walking "
              "every warp in one thread, but this kernel " + warp_why_;
        return false;      }
      // sub is only defined for unsigned in the runtime; signed subtraction is
      // an add of the negation.
      if (op == OpAtomicISub && (sgn || flt)) {
        set("weft::atom_add" + suf + "((" + addr_space("device", false) + tyop +
            "*)(" + addr_space("device", false) + "void*)(" + ptr + "), -(" + val_ + "))");
        return true;
      }
      // A float exchange needs no float form: the bits are the value, so it is the
      // 32-bit exchange with a bitcast on each side, which is exact for every float
      // including the NaNs a numeric min would have to choose a winner for.
      if (flt && fn == "atom_exch") {
        const std::string w = addr_space("device", false) + "uint32_t*";
        set("weft_bitcast<float>(weft::atom_exch_u32((" + w + ")(" +
            addr_space("device", false) + "void*)(" + ptr + "), weft_bitcast<uint32_t>(" +
            val_ + ")))");
        return true;
      }
      // A float compare-exchange is not a spelling this model has, and that is the
      // spec rather than a gap here: the assembler refuses the instruction outright
      //, `AtomicCompareExchange: expected Result Type to be integer scalar type`.
      // Add, min, max and exchange are the float atomics that exist.
      if (flt && fn != "atom_add" && fn != "atom_min" && fn != "atom_max") {
        err = "float atomic compare-exchange does not exist in SPIR-V: its result is "
              "constrained to an integer scalar";
        return false;
      }
      std::string call = "weft::" + fn + suf + "((" + addr_space("device", false) + tyop +
                         "*)(" + addr_space("device", false) + "void*)(" + ptr + "), " + val_;
      if (!cmp.empty()) call += ", " + cmp;
      call += ")";
      set(call);
      return true;
    }

    //
    // An image in SPIR-V is an opaque handle and a sampler is a value the host
    // creates; `OpSampledImage` joins the two. The runtime keeps the image as a
    // pointer to the bound texture and the sampler as the two choices that can
    // change an answer, which is everything a compute kernel can observe.
    case OpSampledImage: {
      // An array image joins the sampler in its own handle type: the layer is
      // part of every sample's coordinate, and the GPU form is a different
      // texture type. The host form is one struct either way.
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
      // The size a kernel asks for is a field of the handle on the GPU and a
      // field of the host texture through a pointer on the CPU.
      const std::string dot = gpu ? "." : "->";
      const std::string w = "(" + ety + ")(" + img + ")" + dot + "width";
      const std::string h = "(" + ety + ")(" + img + ")" + dot + "height";
      // A 2-D array reports its slice count as the third component. A plain
      // image has no third component and answers one, which is the same number
      // the Metal front end reports for `get_array_size()`.
      auto it = m_.types.find(vtype(a[2]));
      const bool arr = it != m_.types.end() && it->second.arrayed != 0;
      // A volume's third component is its depth, which is the same outermost
      // extent an array reports as its slice count.
      const bool vol = it != m_.types.end() && it->second.kind == KImage && it->second.count == 2u;
      // The GPU's 3-D handle calls that extent `depth`; the host texture stores a
      // layered image and a volume the same way and calls it `layers`.
      const std::string field = (gpu && vol && !arr) ? "depth" : "layers";
      // A cube's third extent is the six faces it is made of, which the handle
      // does not carry, it is `texturecube`, with a width and a height only.
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
      // A gather is the four texels of the 2x2 footprint, one per channel, instead
      // of the blend a filter would produce, the SPIR-V twin of PTX's `tld4`. It
      // carries a component operand the others do not, which is why its image
      // operands start one word later.
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

      // A 1-D image is a plane one row high, so a sample sits at 0.5, the centre
      // of that row, which is the texel itself under any filter and any address
      // mode. A cube's coordinate is a direction, and its three components are
      // exactly what the sampler takes, so they are read out as they arrive.
      std::string x, y, lay;
      {
        uint32_t cet = 0, cn = 0;
        const std::string c = V(3);
        if (vector_of(vtype(a[3]), cet, cn) && cn >= 2) {
          x = "(" + c + ").x";
          y = "(" + c + ").y";
          // The third component of an arrayed coordinate is the slice.
          if (cn >= 3) lay = "(" + c + ").z";
        } else {
          x = "(" + c + ")";
          y = (it->second.count == 0u) ? "0.5f" : "0";
        }
      }
      // A layered image and 3-D image both put their outermost index in the third
      // component; what differs is only what the sampler does with it, so the read
      // path is shared and the sample below picks its own form.
      const bool vol = it->second.arrayed == 0 && it->second.count == 2u;
      const std::string lz = (it->second.arrayed != 0 || vol)
                                 ? (", (uint32_t)(" + (lay.empty() ? std::string("0") : lay) + ")")
                                 : std::string();

      // The optional image operands, in the order the spec fixes: Bias, Lod,
      // Grad (two ids), ConstOffset, Offset, ConstOffsets, Sample, MinLod. The
      // levels are ignored, a bound texture has one, and the two offsets are
      // how a kernel asks for a neighbouring texel.
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
        // The footprint is a 2x2 block of one plane, and every shape this front
        // end decodes has one: a plain image, a slice of an array, a plane of a
        // volume, or the face of a cube the direction lands on. The answer is
        // four channels of the same sample type, which is the result vector.
        const std::string g = "weft_g" + std::to_string(a[1]);
        body_ += "  float " + g + "[4];\n";
        // The offset is a texel count, so it is added as a fraction of the extent
        //, the same thing the sample path does with it.
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
          // Not a gap: `OpImageGather` is defined for a 2D, Cube or Rect image
          // only, and the validator says so ("Expected Image 'Dim' to be 2D, Cube,
          // or Rect"), so a volume can never carry one.
          err = "a gather is a 2D, Cube or Rect image in SPIR-V, and a volume is none of them";
          return false;
        } else if (dim == 3u) {
          // A cube gathers by direction. Metal has no `gather` on a `texturecube` bound
          // with `access::sample`, so the four texels are read off the face the
          // direction picks, which is what a `tld4` naming the face does on the PTX
          // column, through the same rule and the same four reads.
          const std::string tb = "(" + (gpu ? img : "*" + img) + ")";
          call = "weft::tex_gather4_cube_dir<" + ety + ">(" + lhs + ", " + x + ", " + y +
                 ", " + (lay.empty() ? std::string("0.0f") : lay) + ", (float)(" + ox +
                 ") / (float)(" + tb + ".width), (float)(" + oy + ") / (float)(" + tb +
                 ".height), " + g + ")";
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
        // The direction goes to the cube sampler, which picks the face and the
        // place on it: the same rule the CPU runtime models. All three components
        // are the direction, so a coordinate that does not carry three is a
        // module this cannot answer rather than one it can answer wrongly.
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
        // On the GPU the filter and the address mode live in the bound sampler
        // state, so the sample names the sampler and nothing else.
        // A volume's third coordinate is normalised and stays a float; a layered
        // one is an integer slice, which is what `lz` carries.
        const std::string third = vol ? (", (float)(" + (lay.empty() ? std::string("0") : lay) + ")")
                                      : lz;
        read = gpu ? ("weft::tex_sample<" + ety + ">(" + src + ", " + u + ", " + v + third + ")")
                   : ("weft::tex_sample<" + ety + ">(*" + img + ", " + u + ", " + v + third +
                      ", (" + src + ").s.filter, (" + src + ").s.addr)");
      } else {
        read = "weft::tex_read<" + ety + ">(" + (gpu ? img : "*" + img) + ", (uint32_t)(" + x +
               " + " + ox + "), (uint32_t)(" + y + " + " + oy + ")" + lz + ")";
      }
      // A read of a sampled image answers with four channels, exactly as the
      // Metal front end's does: (value, 0, 0, 1).
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

    // NOTE: `OpImageGather` is deliberately absent. It looks like the SPIR-V twin
    // of the PTX front end's `tld4`, and an emitter for it was written; the
    // validator refused the module with `Opcode ImageGather requires one of these
    // capabilities: Shader`. It is Shader-only, and this front end decodes the
    // OpenCL/Kernel model, which has no gather, so there is nothing to translate
    // here and no fixture that could be valid. The PTX side still has `tld4`.

    // A storage image write. The texel arrives as a vector of channels and the
    // launcher creates single-channel images, so the first channel is the texel:
    // `tex_write` copies `elem` bytes from the value's first member, which is
    // where channel 0 of a vector lives.
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
      // Only an arrayed image has a slice to write into; a plain one ignores the
      // third coordinate, which is what a 2-D image does with a spare component.
      {
        auto wt = m_.types.find(vtype(a[0]));
        const bool vol = wt != m_.types.end() && wt->second.kind == KImage &&
                         wt->second.count == 2u;
        if (wt == m_.types.end() || (wt->second.arrayed == 0 && !vol)) lay.clear();
        else lay = ", " + lay;
      }
      // What the image holds: the texel's own type, widened to four channels if
      // the module wrote a scalar or a short vector.
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

    //
    // `subgroupAdd` in a shader is OpGroupNonUniformFAdd, not the OpenCL.std
    // extended instruction, and these were refused as unsupported opcodes: the fold
    // that answers them was already here, reachable only by the other spelling. The
    // operands are (type, result, scope, group action, value[, cluster size]) and the
    // group action is a constant, so dropping it leaves exactly the shape the named
    // form takes, the fold that runs is the same one, on every engine.
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
      // 0 reduces, 1 is an inclusive scan, 2 an exclusive one. The group operation is a
      // literal in this instruction, that is why an assembler writes it as `Reduce`
      // rather than as a constant id, so it is already the operand's own number. A
      // module that carried it as an id instead is resolved through the constants.
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

    //
    // A ballot is one word here, so every question about it is local arithmetic on
    // that word and needs no rendezvous. The four were refused as unsupported
    // opcodes while BallotBitCount was handled; only the first of the 128-bit form's
    // four words can be set for a 32-wide subgroup, so the extra ones are dropped.
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

    //
    // `subgroupRotate` names a lane to take the value from, `Delta` lanes along and
    // wrapped inside the cluster of `ClusterSize` lanes (32 when it is not given, and
    // always a power of two, which is what makes the wrap a mask). A negative Delta
    // wraps the same way through unsigned arithmetic.
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
        // The numbers are OpenCL.std's own, what an assembler encodes and what a
        // compiler emits. This window used to be off by one: `cbrt` is 11 and `ceil`
        // is 12, because `cospi` and `expm1` are in the list between `cosh` and
        // `erfc`. As it stood, a kernel calling `ceil` was translated into a cube
        // root, silently. `cospi` and `expm1` have no name on both targets, so they
        // are refused by number instead of being mapped to something else.
        // The inverse trigonometry and the pi-scaled family. These were missing
        // outright: a kernel calling `atan2`, which graphics code calls for every
        // angle, was refused while the library that answers it was already here.
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
        // The "half_" and "native_" families are downgraded to the precise ones:
        // this runtime has no reduced-precision path, and answering with a
        // slower but correct value beats refusing a kernel for its spelling.
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
      // The "pi" forms divide by pi and have no `std` spelling, so they are built
      // from the library's own function: one expression, the same on both targets.
      if (num == 2 || num == 5 || num == 9) {
        const char* base = num == 2 ? "weft_acos" : (num == 5 ? "weft_asin" : "weft_atan");
        set(std::string(base) + "(" + arg(0) + ") * 0.3183098861837907f");
        return true;
      }
      if (num == 10) {
        set("weft_atan2(" + arg(0) + ", " + arg(1) + ") * 0.3183098861837907f");
        return true;
      }
      // The three that take an integer: the module's second operand is an `int`,
      // and the library's own signature says so.
      if (num == 33) { set("weft_ilogb(" + arg(0) + ")"); return true; }
      if (num == 34) { set("weft_ldexp(" + arg(0) + ", (int)(" + arg(1) + "))"); return true; }
      if (num == 49) { set("weft_pown(" + arg(0) + ", (int)(" + arg(1) + "))"); return true; }
      if (num == 54) { set("weft_rootn(" + arg(0) + ", (int)(" + arg(1) + "))"); return true; }
      // The four that compute two numbers: the first is the result and the second
      // is written through the pointer the instruction takes as its last operand.
      // One call computes both halves, so a caller cannot receive a fraction whose
      // exponent came from a different input.
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
      // `fract` answers twice, which is easy to miss: instruction 30 returns
      // `fmin(x - floor(x), 0x1.fffffep-1f)` and writes `floor(x)` through the pointer
      // it carries. Both halves come from the same library text, so the sweep covers
      // the value and the two columns are compared on the floor as well. A module
      // that carries no pointer still gets its value.
      if (num == 30) {
        if (a.size() > 5 && !store_out("WEFT_FLOOR(" + arg(0) + ")")) return false;
        set("weft_fract(" + arg(0) + ")");
        return true;
      }
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
        // The names the portable library defines go through it, on both targets:
        // one text compiled as MSL for the GPU and C++ for the CPU, which is the
        // only way the two agree to the bit. A builtin that is exact either way
        //, `floor`, `fabs`, `fmax`, keeps the language's own spelling.
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

      // A helper function is inlined where it is called, and its values keep the
      // module's own ids, unique across the module, so a second call site simply
      // recomputes the same fields and nothing has to be renamed.
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
          // Each of these suspends the caller mid-call or hands off to the other
          // lanes, and an inlined body has nowhere to resume to.
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
      // The parameters are the helper's own fields: assign them, run the body, read the
      // value out of the id the call was made for.
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

// Imports: the OpenCL C builtins a module links against
//
// LLVM's SPIR-V backend lowers barriers and atomics to instructions, but leaves
// the subgroup (warp) builtins as calls to imported functions: the module is
// only complete once those imports are bound, and this runtime is the binding.
// Longest name first, because "sub_group_shuffle_xor" contains
// "sub_group_shuffle" and "_Z7barrierj" contains "barrier".
const char* const kBuiltinNames[] = {
    "sub_group_scan_exclusive_add", "sub_group_scan_exclusive_min",
    "sub_group_scan_exclusive_max", "sub_group_scan_exclusive_and",
    "sub_group_scan_exclusive_or", "sub_group_scan_exclusive_xor",
    "sub_group_scan_inclusive_add", "sub_group_scan_inclusive_min",
    "sub_group_scan_inclusive_max", "sub_group_scan_inclusive_and",
    "sub_group_scan_inclusive_or", "sub_group_scan_inclusive_xor",
    "sub_group_reduce_add", "sub_group_reduce_min", "sub_group_reduce_max",
    // OpenCL.std has no product fold, so the name only ever arrives from the
    // core opcodes: the fold is the same one, and it needs a name to look up.
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
    // the lanes, which is what the warp target's imports write into.
    body_ += "  " + fref(a[1]) + " = " + e + ";\n";
  };

  // Every rendezvous below is the same shape: publish this lane's value, give
  // the worker back, and read the warp's published array on the next pass.
  int last_off = -1;  // the window this call published into, for the fold that reads it
  auto rendezvous = [&](const std::string& payload) -> std::string {
    if (warp_step_) {
      // One thread owns these lanes, so the rendezvous is a window of the warp's
      // slot array and a park: the sweep brings the lanes back in order, and the
      // fold reads the window. Same relay as the native subgroup instructions.
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
  // Which lanes of this warp exist. The last warp of a group is short when the
  // threadgroup size is not a multiple of 32, and a vote over it must not count
  // lanes that never ran.
  static const char* kLive =
      "const uint32_t weft_live = weft::live_threads(ctx.tgid, ctx.tg_size, ctx.grid); "
      "const uint32_t weft_wb = ctx.lid & ~31u; "
      "const uint32_t weft_wl = (weft_live > weft_wb) ? ((weft_live - weft_wb) > 32u ? 32u : "
      "(weft_live - weft_wb)) : 0u; "
      "const uint32_t weft_mask = (weft_wl >= 32u) ? 0xFFFFFFFFu : ((1u << weft_wl) - 1u); ";

  if (name == "barrier" || name == "work_group_barrier" || name == "sub_group_barrier") {
    // A sub-group barrier is weaker than a work-group one, but a threadgroup
    // rendezvous is the only kind this runtime has, and it is sound to over-wait.
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
    // Threads of a group only ever hand off at a rendezvous, so the compiler's
    // ordering of their accesses is already the GPU's.
    result("0");
    return true;
  }
  if (name == "sub_group_local_id") { result("(ctx.lid & 31u)"); return true; }
  if (name == "sub_group_size") { result("32u"); return true; }

  if (name == "sub_group_ballot" || name == "sub_group_all" || name == "sub_group_any") {
    if (gpu && !warp_step_) {
      // The hardware's ballot, with the lanes that exist masked on: a vote is the
      // one subgroup operation that needs nothing but the predicate.
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
    // A double on the GPU is the uint64_t its bits live in, so its identity and its
    // combine are the helpers rather than the operators: `weft_a + weft_v` over two
    // bit patterns would be integer addition.
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
        // 1.0's pattern, and the library's multiply: `weft_a * weft_v` over two
        // bit patterns would be an integer multiply of their mantissas.
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
      // The identity a fold starts from: `INFINITY` on the GPU, which has no
      // numeric_limits, and the type's own extreme for an integer, casting a float
      // infinity to an int is not the integer's largest value.
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
    //
    // A reduce and a scan are the same doubling tree wherever they run, which is
    // what makes a float answer the same in both columns: step d adds the value d
    // lanes below, so a lane ends holding the fold of the lanes under it. A scan
    // keeps that, and a reduce reads it from the top lane that exists. On the
    // hardware that is one `simd_shuffle` a step and the value never leaves a
    // register; on the CPU it is one array read a step of what the warp already
    // published, so the whole tree still costs one rendezvous rather than one per
    // step.
    //
    // It is a tree downwards and not an XOR butterfly for a reason a short warp
    // measured: the butterfly folds lane i with lane i^d, and in the last warp of
    // a launch the lanes past the grid never ran, so the hardware hands back
    // whatever their registers hold. At 24 lanes that read a sum that was 82 off.
    // A downward tree only ever *uses* lanes below, and the ones that never ran
    // still stand in with the fold's identity.
    //
    // `simd_sum` and `simd_prefix_inclusive_sum` are the hardware's own folds in
    // the hardware's own order, and for floats a different order of additions is a
    // different answer, which is why the tree is written out instead of called.
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
    // Which lane the tree reads for lane `i`: itself for a reduce and an inclusive
    // scan, the lane below it for an exclusive one, the shift an exclusive prefix
    // is defined to be. Clamped, so a dead index is still inside the array.
    const std::string at = excl ? "(i > 0u ? (i - 1u) : 0u)" : "i";
    const std::string use =
        excl ? "(i > 0u && (weft_mask & (1u << i)))" : "(weft_mask & (1u << i))";
    // A reduce's answer is the top live lane's prefix; a scan keeps its own.
    const std::string keep = scan ? "weft_me" : "weft_top";

    // The tree, over whatever `pre` + `load` fills the warp's values with.
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
      // Full width, one shuffle a step, nothing published and nothing to wait for.
      // A narrower element than 32 bits goes through its 32-bit form: that is what
      // `simd_shuffle` moves.
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
        // The total is the top live lane's prefix, read with one more shuffle so
        // every lane holds it. The lane that is read ran, so nothing here depends
        // on what a lane that never ran left behind.
        std::string fin;
        if (!step("weft_a", "weft_top", fin)) return false;
        body_ += "    " + fref(a[1]) + " = (" + ety + ")(" + fin + "); }\n";
      }
      return true;
    }

    if (wide64) {
      // An integer is already its own bits; a double's have to be carried as the
      // bits they are, which `weft_bitcast` does on both targets.
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
      // The C++ target publishes the eight bytes into its wide slot, which the
      // shuffle path already double-buffers by pass.
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

  // What the kernel does with its images decides their access mode. Sampling
  // and writing one binding at once has no MSL spelling, so it is named.
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
      // UniformConstant: an image or a sampler. The host binds both before the
      // launch, exactly as a GPU's descriptor set does, so the parameter is a
      // handle onto that binding and not a copied value.
      auto e = m_.types.find(t->second.elem);
      if (e == m_.types.end()) { err = "parameter of an unknown opaque type"; return false; }
      if (e->second.kind == KImage) {        const int idx = ntex_++;
        const bool arr = e->second.arrayed != 0;
        // A volume is its own handle on the GPU, so it needs its own maker, the
        // field it fills is declared from the same Dim by `ctype` above.
        const bool vol = e->second.count == 2u && !arr;
        const bool cb = e->second.count == 3u;
        // The shape is a property of the binding, not of the target, so it is
        // recorded on both passes: the launcher reads its kernel table from the
        // CPU module, where `gpu` is false.
        // A 1-D image is a plane one row high on this engine, bound as `Wx1`,
        // which is a 2-D binding, so it names itself as one. The Metal front end
        // is the column that binds a real `texture1d`, and its bindings say so.
        tex_shape.push_back(cb ? "cube" : vol ? "3d" : arr ? "array" : "2d");
        if (gpu) {
          // The binding is a kernel parameter, and a helper cannot reach a
          // parameter, so the handle is made where the parameter is in scope.
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
        // A `__local` kernel argument: OpenCL passes its size at enqueue time and
        // this runtime has one arena per group, so the argument gets a slice of
        // it. Slices are even splits, which is the only division that cannot
        // overrun: a kernel whose allocations fit in 48 KiB always fits.
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
    vt_[ins.a[1]] = ins.a[0];
  }
  // A helper function is inlined where it is called, so its values are fields of this
  // kernel too. The ids are the module's and unique across it, so they are collected
  // here beside the entry's, including each parameter, which is a plain field the call
  // assigns before the body runs rather than a local of the entry point.
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

  // Function-scope variables own their storage, so they get a byte array rather
  // than a value field.
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
    // A pointer field is typed by the space it was built in, and the blocks
    // below are what build it, so pointers are declared when they end.
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
    // A `UniformConstant` variable is an image or a sampler. The Shader model
    // binds those at module scope, a GLCompute entry point takes no parameters , 
    // so they take the same indices the parameter path hands out, in the order
    // the module declares them.
    if (g.storage == 0) {
      auto e = m_.types.find(pt->second.elem);
      if (e == m_.types.end()) { err = "module variable of an unknown opaque type"; return false; }
      if (e->second.kind == KImage) {
        const int idx = ntex_++;
        const bool arr = e->second.arrayed != 0;
        const bool vol = e->second.count == 2u && !arr;
        const bool cb = e->second.count == 3u;
        // The shape is a property of the binding, not of the target, so it is
        // recorded on both passes: the launcher reads its kernel table from the
        // CPU module, where `gpu` is false.
        // A 1-D image is a plane one row high on this engine, bound as `Wx1`,
        // which is a 2-D binding, so it names itself as one. The Metal front end
        // is the column that binds a real `texture1d`, and its bindings say so.
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
      // threadgroup memory: one arena per group, laid out in declaration order
      base_addr_[it.first] = "((" + addr_space("threadgroup", false) + "unsigned char*)sh + " +
                             std::to_string(wg_bytes_) + "u)";
      wg_bytes_ += sz;
      // Threadgroup memory without an initialiser starts undefined, exactly like
      // a `__shared__` array on the GPU. One with an initialiser is cleared by
      // thread 0 only: every thread clearing it would erase whatever the
      // threads before it had already written.
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
      // A Shader-model buffer: an SSBO or a UBO, which is a `Block` struct around
      // the storage. The host supplies it like any other buffer, but what the
      // launcher needs to know is the element inside the wrapper, it sizes a
      // buffer in elements, and the wrapper is one struct whose first member is
      // the array. A runtime array has no length of its own (the launch says how
      // much there is), so its element is the stride.
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
      // A module-scope buffer. In the OpenCL model that is a `__global` array
      // (storage 5); in the Shader model a compute shader's SSBO or UBO arrives
      // the same way, as a `StorageBuffer` or `Uniform` variable outside every
      // function, because a GLCompute entry point takes no parameters at all.
      // Either way the host supplies the storage, so it takes the next buffer
      // index, the same order the launcher fills them in.
      base_addr_[it.first] = "((" + addr_space("device", false) + "unsigned char*)bufs[" +
                             std::to_string(nbuffers_) + "])";
      buf_elem_.push_back(pt->second.elem);
      // `__global float arr[N]` is the storage itself, but `__global float* p`
      // is a variable holding a pointer, and the launcher can only hand over
      // the array, so the global's value is that array's base address.
      auto et = m_.types.find(pt->second.elem);
      if (et != m_.types.end() && et->second.kind == KPointer) ptr_globals_.insert(it.first);
      nbuffers_++;
    }
  }

  for (size_t bi = 0; bi < blocks_.size(); bi++) {
    // In step mode a block is a case whose body is run for every lane sitting at
    // that program counter, one lane at a time; the lane loop is opened and closed
    // here, and a barrier splits the case in two where it stands.
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
    // The warp target, chosen from the module rather than from the attempt: a
    // 64-bit atomic, which is the reason it exists, and nothing the lanes have to do
    // together, no barrier and no subgroup vote or shuffle, because a single
    // thread runs the lanes one after another and they are never in flight at the
    // same time. Loops, phis and function-scope storage are all fine: one program
    // counter per lane is enough when the lane loop finishes a lane before it
    // starts the next, and a phi is then a per-lane value like any other.
    bool warp_pc = false;
    bool warp_step = false;
    std::string warp_why = "needs something the warp target cannot provide";
    auto warp_shape = [&]() -> bool {
      const auto& B = f->second.body;
      int blocks = 0;
      bool wide = false;
      // A subgroup operation is a hand-off between lanes, and the per-lane target
      // has the hardware for it: `simd_shuffle`, `simd_ballot`. So it is a reason
      // to take the warp target only when the body has more than one block, where
      // a lane runs its own program counter and the lanes diverge, a hardware
      // rendezvous among lanes that have diverged has no answer. One block has
      // nothing to diverge on, and full width is the faster of the two by the
      // measure in the README's warp scan.
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
            // A vote or a shuffle is a relay through the warp's slot array, which is
            // step mode's own machinery: publish, let the sweep bring the lanes
            // back, fold the slots. The OpenCL.std subgroup builtins arrive as calls
            // to imported functions and need exactly the same relay.
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
        // A 64-bit atomic load or store is a pair read or written as one value, so
        // it needs the same single writer the pair updates need.
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
    // The body is generated before the signature, because the signature has to
    // declare the lock table when a 64-bit atomic needs one.
    const std::string code = em.code();
    // The relay area, when the kernel has one: one slice of threadgroup memory per
    // warp, sized here so the launcher and the shader agree on both the width of a
    // slice and how many warps a threadgroup may hold.
    uint32_t slot_words = 0, slot_per = 0;
    if (gpu && warp_kernel) {
      // One Metal thread runs a whole 32-lane warp. Every SSA value is an array of
      // 32, the lane loop sets the context per lane so the emitted code, which
      // reads `ctx.lid` and `ctx.gid`, is unchanged, and the block body is the
      // same text the per-lane target emits.
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
      // Which warp this thread owns: a group can hold more than one warp, so it is
      // the group's position times the threads in it, plus the thread's own index.
      kernels += "  ctx.tgid = weft_tgid.x * weft_tgsz.x + weft_tid;\n";
      kernels += "  const uint32_t weft_lanes = (uint32_t)weft_geom[0];\n";
      if (em.warp_step()) {
        // The warp's slot array, declared at kernel scope: a vote or a shuffle among
        // lanes that one thread owns is a relay through it, in one window per site.
        //
        // A threadgroup holds several metal threads, and every one of them needs
        // its own relay: the windows are indexed by site and lane, not by thread,
        // so two warps sharing an array publish into the same slot and a shuffle
        // reads a lane of a warp that has nothing to do with it. Measured on the
        // warp scan below at eight warps per group: a max |diff| of 9.7, and 0
        // once each warp had its own slice. So the array is one slice per warp and
        // the slice budget fixes how many warps may share a threadgroup.
        slot_words = (uint32_t)em.slot_words();
        slot_per = 6144u / slot_words;  // 24 KiB of threadgroup memory, at most
        if (slot_per > 8u) slot_per = 8u;
        if (slot_per == 0u) slot_per = 1u;
        kernels += "  threadgroup uint32_t weft_slots_all[" +
                   std::to_string((uint64_t)slot_words * slot_per) + "];\n";
        kernels += "  threadgroup uint32_t* weft_slots = weft_slots_all + weft_tid * " +
                   std::to_string(slot_words) + "u;\n";
      }
      // A 64-bit pair over two 32-bit words has one writer at a time, so a kernel
      // that needs it is dispatched as one thread walking every warp: the lanes
      // arrive at the atomics in program order, and nothing spins or contends.
      // A lock per warp was measured and is slower than the work it guards.
      // A word nothing else in the kernel touches becomes a local: one read before
      // the walk and one write after it, instead of two device round trips for
      // every lane. The gate is the binding's use count in the emitted body, if
      // anything else reads it, the local would be a stale copy.
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
        // Step mode: every lane starts at block 0, the locals are zeroed once per
        // lane before any block runs, and the dispatch below runs whichever block
        // the lowest program counter names. A barrier parks a lane on a case of
        // its own, so no lane can run past one until every lane has arrived.
        kernels += "  for (uint32_t i = 0; i < " + std::to_string(em.slot_words()) +
                   "u; i++) weft_slots[i] = 0u;\n";
        kernels += "  for (uint32_t l = 0; l < 32u; l++)\n";
        kernels += "    s->pc[l] = (weft_base + l < weft_lanes) ? 0u : 0xFFFFFFFFu;\n";
        kernels += "  for (uint32_t l = 0; l < 32u; l++) {\n";
        kernels += "    if (weft_base + l >= weft_lanes) break;\n";
        kernels += "    ctx.lid = l; ctx.gid = (uint64_t)weft_base + (uint64_t)l;\n";
        kernels += em.init();
        kernels += "  }\n";
        // Sweep after sweep, every lane runs at most one block, and the block
        // after a barrier is just another block. So a lane that parks on a barrier
        // waits out the rest of the sweep and every lane has arrived by the next
        // one: the rendezvous is the sweep itself. The selector is this loop's own
        // counter, never a lane's, for the compiler reason in case_open.
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
      // The block text starts with its own `case 0u:;`, so the switch starts here.
      // A kernel with more than one block needs a dispatch loop around it: a branch
      // writes `s->pc` and continues, the lane walks its own control flow to the
      // end, and the sentinel ends the lane. One counter is enough because the lane
      // loop finishes a lane before it starts the next.
      if (wpc) {
        kernels += "    for (;;) {\n";
        kernels += "      if (s->pc == 0xFFFFFFFFu) break;\n";
        kernels += "      switch (s->pc) {\n";
      } else {
        kernels += "    switch (0u) {\n";
      }
      }  // the per-lane half of `if (wstep)`; the block text is shared below
      // The folds above are paid back into the buffers here, once each, and with
      // the pair forms. That is measured rather than assumed. A one-walker launch
      // is one Metal thread for the whole grid, so its 64-bit counters have a
      // single writer and a *plain* load and store looks like it must be
      // equivalent, it was 825 ms down to 70 ms at 1,048,576 lanes, and it read
      // counters a previous dispatch of the same buffer had left in the device
      // cache: 4,096 and 1,048,576 lanes disagreed while 8 lanes agreed. So the
      // access that reaches the buffer is an atomic, and the fold that saves the
      // 1M round trips is the local above.
      kernels += body;
      if (wstep) {
        // An id with no case is one no lane is waiting on; nothing to do.
        kernels += "        default: break;\n";
        // A sweep in which no lane ran is a sweep with every lane retired, the
        // module never defines an id that no case answers, so a live lane always
        // runs exactly once per sweep.
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
      // One coherent access per counter for the whole launch, which is where the
      // folded local is paid back into the buffer.
      for (const auto& a : em.red_accs)
        kernels += "  (void)weft::red_" + a.op + "_u64(" + a.addr + ", " + a.name + ");\n";
      em.red_accs.clear();
      em.shadows.clear();
      kernels += "}\n\n";
    } else if (gpu) {
      // A kernel: a thread-local register file, the runtime's context, one bound
      // resource per buffer, and a dispatch switch over the program counter.
      // An image or a sampler is a binding, and on the GPU a binding is a
      // kernel parameter: `[[texture(N)]]` and `[[sampler(N)]]`, in the order the
      // CPU runtime hands them to `texture_ptr` and `sampler`.
      const int nbufs = em.nargs;
      kernels += "struct " + st + " {\n  uint32_t pc;\n" + em.decls() + "};\n";
      kernels += "kernel void " + e.name + "(";
      for (int i = 0; i < nbufs; i++)
        kernels += "device uint8_t* weft_b" + std::to_string(i) + " [[buffer(" +
                    std::to_string(i) + ")]],\n                 ";
      kernels += "device uint64_t* weft_params [[buffer(" + std::to_string(nbufs) +
                 ")]],\n                 ";
      // One binding per resource, so the lock table takes the index after the
      // scalars, the same one a warp kernel uses for its geometry.
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
      // The block loop opens with `case 0u:;` for the entry block, so the switch
      // starts here rather than naming the same case twice.
      kernels += "  for (;;) {\n    switch (s->pc) {\n";
      kernels += code;
      kernels += "      default: return;\n    }\n  }\n}\n\n";
    } else {
    kernels += "struct " + st + " {\n  uint32_t pc;\n" + em.decls() + "};\n";
    // The body, once, inside the lane loop: a call per lane per segment was six
    // of the fifteen instructions a visit, measured in the generated assembly.
    // `return` in a lambda leaves the lambda, which is what a segment boundary is.
    std::string inner;
    inner += "    " + st + "* s = (" + st + "*)st;\n";
    inner += "    (void)ctx; (void)bufs; (void)sh;\n";
    inner += "    if (s->pc != 0u) {\n      switch (s->pc) {\n" + em.cases() +
             "        default: goto weft_done;\n      }\n    }\n";
    inner += em.init();
    inner += em.code();
    inner += "    weft_done:;\n    s->pc = 0xFFFFFFFFu;\n";
    // The group form: the lane loop, the pass number and the count of lanes still
    // running all live here, so a segment costs one call per threadgroup instead
    // of an indirect one per lane.
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
    // One lane, for the paths that resume a single thread: the same body, not a
    // second copy of it.
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
    // Only the kernels and the runtime: Metal finds a kernel by name and the
    // launcher does the dispatch, so there is no table here.
    std::string g = "// generated by weft (spir-v front end, metal target), do not edit\n";
    g += kWWeftMslPreamble;
    // The portable library, so a module's transcendental runs one text on both
    // targets. The Metal front end's own source gets this from the launcher; this
    // one is generated here, so it is added here.
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
      // The vector types are packed: a <3 x float> in memory is 12 bytes and the
      // struct has to be the same 12 bytes or a load would read the wrong data.
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
      "template <class T, class U>\n"
      "static inline T weft_bitcast(U u) { T t; std::memcpy(&t, &u, sizeof(T)); return t; }\n"
      // A bit reversal written as masks, not as a builtin: `__builtin_bitreverse32`
      // is clang's alone, and this is the same eight swaps everywhere, gcc folds
      // it back to `rbit` on aarch64 by itself.
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
      // A load and a store through a pointer with no type: memcpy of a constant
      // size is one instruction, and it cannot be misaligned.
      "template <class T>\n"
      "static inline T weft_ld(const void* p) { T t; std::memcpy(&t, p, sizeof(T)); return t; }\n"
      "template <class T>\n"
      "static inline void weft_st(void* p, T v) { std::memcpy(p, &v, sizeof(T)); }\n";
  // The portable library, over the vector types it is written in. A module's
  // transcendental then runs one text on both targets: without it the GPU answers
  // with its own approximation while the CPU answers with libm, and the two differ
  // in the last bits, which is the exactness this engine is for.
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
  // A width the caller states wins over anything this front end could infer. It is
  // applied before the ABI below is written, so the module reports the stated width
  // itself, a run does not have to be told again on its own command line.
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
