#include "translate_ptx.hpp"

#include "weft_msl.hpp"
#include "weft_warp_msl.hpp"
// The portable library, emitted into both of this front end's texts. `.approx` is
// a hardware polynomial and no CPU reproduces its bits, so the answer is the one
// this project already gives for Metal's `sqrt`: both engines run the same text.
#include "soft_math.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace {

bool is_word(char c) { return std::isalnum((unsigned char)c) || c == '_' || c == '$'; }

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace((unsigned char)s[a])) a++;
  while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
  return s.substr(a, b - a);
}

std::string strip_comments(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    if (i + 1 < s.size() && s[i] == '/' && s[i + 1] == '/') {
      while (i < s.size() && s[i] != '\n') i++;
      continue;
    }
    if (i + 1 < s.size() && s[i] == '/' && s[i + 1] == '*') {
      i += 2;
      while (i + 1 < s.size() && !(s[i] == '*' && s[i + 1] == '/')) i++;
      i = (i + 1 < s.size()) ? i + 2 : s.size();
      out.push_back(' ');
      continue;
    }
    out.push_back(s[i++]);
  }
  return out;
}

size_t find_matching(const std::string& s, size_t open, char o, char c) {
  int depth = 0;
  for (size_t i = open; i < s.size(); i++) {
    if (s[i] == o) depth++;
    else if (s[i] == c && --depth == 0) return i;
  }
  return std::string::npos;
}

std::vector<std::string> split_top(const std::string& s, char sep) {
  std::vector<std::string> out;
  int depth = 0;
  size_t start = 0;
  for (size_t i = 0; i < s.size(); i++) {
    char c = s[i];
    if (c == '[' || c == '(' || c == '{') depth++;
    else if (c == ']' || c == ')' || c == '}') depth--;
    else if (c == sep && depth == 0) {
      out.push_back(trim(s.substr(start, i - start)));
      start = i + 1;
    }
  }
  out.push_back(trim(s.substr(start)));
  return out;
}

// Does `body` name this identifier? A substring test is not enough: a kernel with
// thirty parameters has `..._param_3` inside `..._param_30`.
bool mentions(const std::string& body, const std::string& name) {
  if (name.empty()) return false;
  for (size_t p = body.find(name); p != std::string::npos; p = body.find(name, p + 1)) {
    const size_t e = p + name.size();
    const bool left = p == 0 || !(std::isalnum((unsigned char)body[p - 1]) || body[p - 1] == '_');
    const bool right =
        e >= body.size() || !(std::isalnum((unsigned char)body[e]) || body[e] == '_');
    if (left && right) return true;
  }
  return false;
}

bool has_part(const std::vector<std::string>& parts, const char* want) {
  for (const auto& p : parts)
    if (p == want) return true;
  return false;
}

std::string cpp_type(const std::string& ptx) {
  if (ptx == ".f32") return "float";
  if (ptx == ".f64") return "double";
  if (ptx == ".b32" || ptx == ".u32") return "uint32_t";
  if (ptx == ".s32") return "int32_t";
  if (ptx == ".b64" || ptx == ".u64") return "uint64_t";
  if (ptx == ".s64") return "int64_t";
  if (ptx == ".b16" || ptx == ".u16") return "uint16_t";
  if (ptx == ".s16") return "int16_t";
  if (ptx == ".b8" || ptx == ".u8") return "uint8_t";
  if (ptx == ".s8") return "int8_t";
  if (ptx == ".pred") return "bool";
  return "";
}

// `%f12` -> base "f", index 12. `%tid.x` -> special "tid.x". `%p` -> "p", 0.
struct Reg {
  std::string base;
  std::string special;
  int index = 0;
  bool ok = false;
};

Reg parse_reg(const std::string& tok) {
  Reg r;
  if (tok.empty() || tok[0] != '%') return r;
  size_t i = 1, s = 1;
  while (i < tok.size() && is_word(tok[i])) i++;
  std::string head = tok.substr(s, i - s);
  if (i < tok.size() && tok[i] == '.') {  // %tid.x and friends
    r.special = head + tok.substr(i);
    r.ok = true;
    return r;
  }
  size_t d = head.size();
  while (d > 0 && std::isdigit((unsigned char)head[d - 1])) d--;
  r.base = head.substr(0, d);
  r.index = d < head.size() ? std::atoi(head.c_str() + d) : 0;
  r.ok = !r.base.empty();
  return r;
}

std::string sanitize(const std::string& s) {
  std::string out;
  for (char c : s) out.push_back(is_word(c) ? c : '_');
  return out;
}

int cpp_size(const std::string& t) {
  if (t == "float" || t == "uint32_t" || t == "int32_t") return 4;
  if (t == "double" || t == "uint64_t" || t == "int64_t") return 8;
  if (t == "uint16_t" || t == "int16_t") return 2;
  if (t == "uint8_t" || t == "int8_t" || t == "bool") return 1;
  return 0;
}

struct Emitter {
  std::string err;
  bool uses_barrier = false;
  bool uses_shfl = false;  // counted separately: a shuffle does not tie warps together

  // ---- target ----
  // One decode, two back ends. `msl` switches the few places where the same
  // instruction is worded differently for Metal — a branch, a barrier, a
  // shuffle, the address space of a pointer — and nothing else. The parsing, the
  // register model and the expression builders are shared, which is what makes
  // the GPU output comparable to the CPU output rather than a second
  // implementation that happens to agree.
  bool msl = false;
  // The kernel uses a 64-bit atomic, which this host has no instruction for. The
  // emission names a helper that builds one out of 32-bit atomics under a lock
  // (see weft_warp_msl.hpp), and only the warp back end carries that runtime — so
  // a kernel that holds one is emitted for the warp back end.
  bool wide_atom = false;
  // How many times each register's name appears in the body, its own declaration
  // aside. One appearance means nothing reads the value, which is what turns an
  // `atom` into the `red` of the same operation — the rewrite ptxas makes, and the
  // one that gives a 64-bit counter a GPU column at all.
  std::map<std::string, int> reg_reads;
  // ---- texture bindings ----
  // A `.tex` or `.surf` declaration names one; the declarations, in order, are
  // the indices `--textures` fills. A sampled texture reads through the sampler
  // state at the same index; a surface is a texture the kernel also writes,
  // which is the only thing that changes the GPU's access mode.
  std::map<std::string, int> tex_index;
  int ntex = 0;
  std::vector<bool> tex_write;
  // The bindings an `.a2d` instruction names. PTX's `.tex` says nothing about
  // dimensionality, so the array form is a property of the instruction: the body
  // is scanned for it before emission, and both the GPU parameter type and
  // `txq.depth` read the answer.
  std::vector<bool> tex_array;
  // The same question for `.3d`: a volume on the GPU is `texture3d`, and its
  // outermost index is a plane rather than a slice.
  std::vector<bool> tex_3d;
  // And for `.cube`: six square faces, which on the GPU are `texturecube` — a
  // parameter type of its own, and one that cannot be read or written by texel.
  std::vector<bool> tex_cube;
  int shared_bytes = 0;      // the threadgroup arena this kernel needs, at compile time
  bool shared_extern = false;  // it was `extern .shared`: the launch chooses the size
  bool uses_red = false;     // a reduction barrier: needs its publish slots

  // ---- warp-vector mode ----
  // Registers become r[reg][lane] and every instruction is emitted as a 32-lane
  // loop, so a shuffle is a gather inside the register file. The expression
  // builders are unchanged: they just address `[l]` instead of a scalar.
  bool vec = false;
  bool vec_ok = true;      // false once something here cannot be vectorised
  std::string vec_pred;    // per-lane predicate for the current instruction
  bool vec_control = false;  // emitted text is control flow: do not wrap it
  bool vec_side = false;     // emitted text already contains its lane loop
  bool vec_select = true;    // store() masks by lane; off inside a lane loop
  int label_count = 1;                 // 0 is the entry
  std::map<std::string, int> label_ids;

  int label_id(const std::string& name) {
    auto it = label_ids.find(name);
    if (it != label_ids.end()) return it->second;
    const int id = label_count++;
    label_ids[name] = id;
    return id;
  }
  // ---- lane-affine values ----
  // What a register holds, as `base + stride * lane`, with `base` the same for
  // every lane of the warp. This is tracked only to answer one question at a
  // memory instruction: does lane l touch the address next to lane l-1? If it
  // does, the 32 lanes are a contiguous run and the access becomes a vector
  // load or store instead of a 32-step scalar gather. Anything the model does
  // not understand is simply unknown, which costs speed and not correctness.
  struct VecVal {
    bool ok = false;
    std::string base;  // C++ text with no lane reference in it
    long stride = 0;   // coefficient of the lane index
  };
  std::map<std::string, VecVal> vec_vals;

  static std::string reg_key(const Reg& r) { return r.base + "#" + std::to_string(r.index); }

  static bool vec_num(const std::string& t, long& out) {
    if (t.empty()) return false;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(t.c_str(), &end, 0);
    if (!end || *end != '\0' || end == t.c_str()) return false;
    out = (long)v;
    return true;
  }

  VecVal vec_sym(const std::string& tok) {
    const std::string t = trim(tok);
    VecVal v;
    if (t.empty()) return v;
    if (t[0] != '%') {
      long n = 0;
      if (vec_num(t, n)) { v.ok = true; v.base = std::to_string(n); }
      return v;
    }
    Reg r = parse_reg(t);
    if (!r.ok) return v;
    if (!r.special.empty()) {
      if (r.special == "tid.x") { v.ok = true; v.base = "ctx.lid0"; v.stride = 1; }
      else if (r.special == "laneid") { v.ok = true; v.base = "0u"; v.stride = 1; }
      else if (r.special == "ntid.x") { v.ok = true; v.base = "ctx.tg_size"; }
      else if (r.special == "ctaid.x") { v.ok = true; v.base = "ctx.tgid"; }
      return v;
    }
    auto it = vec_vals.find(reg_key(r));
    return it == vec_vals.end() ? VecVal{} : it->second;
  }

  VecVal vec_add(const VecVal& a, const VecVal& b, bool negate) {
    if (!a.ok || !b.ok) return {};
    VecVal v;
    v.ok = true;
    v.base = "(" + a.base + (negate ? " - " : " + ") + b.base + ")";
    v.stride = a.stride + (negate ? -b.stride : b.stride);
    return v;
  }

  VecVal vec_scale(const VecVal& a, long k) {
    if (!a.ok) return {};
    VecVal v;
    v.ok = true;
    v.base = "(" + a.base + " * " + std::to_string(k) + ")";
    v.stride = a.stride * k;
    return v;
  }

  void vec_forget(const std::string& tok) {
    Reg r = parse_reg(tok);
    if (!r.ok || !r.special.empty()) return;
    vec_vals[reg_key(r)] = VecVal{};
  }

  // Feed one instruction's dataflow through the model. Called for every
  // statement; unrecognised opcodes poison the destination, which is what keeps
  // the model from claiming a value it cannot justify.
  void vec_update(const std::string& opcode, const std::vector<std::string>& args) {
    if (!vec || args.empty()) return;
    // A brace list writes several registers at once; forget all of them.
    if (args[0].find('{') != std::string::npos) {
      std::string list = args[0].substr(args[0].find('{') + 1);
      list = list.substr(0, list.find('}'));
      for (const auto& t : split_top(list, ',')) vec_forget(t);
      return;
    }
    Reg d = parse_reg(args[0]);
    if (!d.ok || !d.special.empty()) return;

    const std::string op = opcode.substr(0, opcode.find('.'));
    VecVal a = args.size() > 1 ? vec_sym(args[1]) : VecVal{};
    VecVal b = args.size() > 2 ? vec_sym(args[2]) : VecVal{};
    VecVal c = args.size() > 3 ? vec_sym(args[3]) : VecVal{};
    VecVal r;

    if (op == "mov" || op == "cvta") {
      r = a;
    } else if (op == "add" || op == "addc") {
      r = vec_add(a, b, false);
    } else if (op == "sub") {
      r = vec_add(a, b, true);
    } else if (op == "mul") {
      long n = 0;
      if (b.ok && vec_num(trim(b.base), n)) r = vec_scale(a, n);
      else if (a.ok && vec_num(trim(a.base), n)) r = vec_scale(b, n);
      else if (a.ok && b.ok && a.stride == 0 && b.stride == 0) {
        r.ok = true;
        r.base = "(" + a.base + " * " + b.base + ")";
      }
    } else if (op == "mad") {
      long n = 0;
      VecVal prod;
      if (b.ok && vec_num(trim(b.base), n)) prod = vec_scale(a, n);
      else if (a.ok && vec_num(trim(a.base), n)) prod = vec_scale(b, n);
      else if (a.ok && b.ok && a.stride == 0 && b.stride == 0) {
        prod.ok = true;
        prod.base = "(" + a.base + " * " + b.base + ")";
      }
      if (prod.ok) r = vec_add(prod, c, false);
    } else if (op == "shl") {
      long n = 0;
      if (b.ok && vec_num(trim(b.base), n)) r = vec_scale(a, 1L << n);
    } else if (op == "shr") {
      // A lane-independent value shifted by a lane-independent count is still
      // lane-independent, which is what a shuffle delta is in a reduction loop.
      if (a.ok && a.stride == 0 && b.ok && b.stride == 0) {
        r.ok = true;
        r.base = "(" + a.base + " >> " + b.base + ")";
      }
    } else if (op == "cvt") {
      if (a.ok) {
        r.ok = true;
        r.stride = a.stride;
        auto it = reg_types.find(d.base);
        r.base = (it != reg_types.end() && cpp_size(it->second) == 8)
                     ? "(int64_t)(" + a.base + ")"
                     : a.base;
      }
    } else if (op == "ld" && args.size() > 1 && args[1].find('[') != std::string::npos) {
      // A pointer parameter is a uniform base with no lane term: that is the
      // root of every address in a real kernel, so it has to be tracked.
      std::string inner = trim(args[1]);
      inner = trim(inner.substr(1, inner.size() - 2));
      auto it = params.find(inner);
      if (it != params.end()) {
        r.ok = true;
        r.base = buf_base(it->second);
      }
    }
    vec_vals[reg_key(d)] = r;
  }

  // `[%rd5]` / `[%rd5+16]` / `[tile]` -> the lane-affine value it points at.
  VecVal mem_sym(const std::string& op) {
    std::string s = trim(op);
    if (s.size() < 2 || s.front() != '[' || s.back() != ']') return {};
    s = trim(s.substr(1, s.size() - 2));
    if (s.empty() || s[0] != '%') return {};
    size_t cut = std::string::npos;
    bool negative = false;
    for (size_t i = 1; i < s.size(); i++) {
      if (s[i] == '+' || s[i] == '-') { cut = i; negative = (s[i] == '-'); break; }
    }
    VecVal v = vec_sym(cut == std::string::npos ? s : trim(s.substr(0, cut)));
    if (!v.ok) return v;
    if (cut != std::string::npos) {
      const std::string off = trim(s.substr(cut + 1));
      v.base = "(" + v.base + (negative ? " - " : " + ") + off + ")";
    }
    return v;
  }

  bool warp_only = false;     // the body needs the whole warp's registers
  int next_resume = 1;
  std::string resume_cases;   // "case 1: goto R1;"
  std::string resume_labels;  // "R1:;" spliced in after each barrier
  const std::string* shared_size = nullptr;
  // Each `.shared` array's byte offset inside the threadgroup's shared window.
  // A kernel may declare several; they are laid out one after another, aligned,
  // so `tile2[0]` is not `tile1[0]`.
  std::map<std::string, uint64_t> shared_offsets;
  std::map<std::string, int> params;   // buffer parameters
  std::map<std::string, int> scalars;  // plain value parameters
  std::vector<std::string> unused;     // parameters the body never names
  // What each buffer parameter is *addressed as*, by the instructions that reach
  // it. PTX binds an address, not a type: a parameter is `.u64` and its register
  // is `.b64`, so nothing in the declaration says a buffer holds floats. The load
  // or store does — `ld.global.f32` is a four-byte element, `ld.global.u8` one —
  // and that is the only place the width exists. A parameter reached at two
  // different widths has no single answer and reports 0, which the ABI reads as
  // unknown rather than as four bytes.
  std::map<int, uint32_t> param_bytes;
  // The register holding a parameter's base address, through `ld.param`, `cvta`,
  // `mov` and `add` — the instructions every front end puts between a binding and
  // its first access.
  std::map<std::string, int> ptr_param;
  std::map<std::string, std::string> reg_types;  // "r" -> "uint32_t"
  std::string decls;

  // PTX registers carry a declared type, but the instruction says how the bits
  // are meant right now: nvcc-era PTX loads floats with ld.global.b32 and then
  // does add.rn.f32 on the same register. So every operand is coerced to the
  // instruction's type, and the result is coerced back to the register's type.
  std::string operand_as(const std::string& tok, const std::string& want) {
    std::string t = trim(tok);
    if (t.empty()) return t;
    if (t[0] != '%') {
      if (t.size() == 10 && t.compare(0, 2, "0f") == 0) return "weft_f32(0x" + t.substr(2) + "u)";
      if (t.size() == 18 && t.compare(0, 2, "0d") == 0) {
        // A double on the GPU target is its bits in a `uint64_t` and `weft_f64` is
        // the identity there, so the constant has one spelling for both targets.
        return "weft_f64(0x" + t.substr(2) + "ull)";
      }
      if (!t.empty() && (std::isalpha((unsigned char)t[0]) || t[0] == '_')) {
        // A bare symbol. A shared array's address is its offset inside the
        // threadgroup's shared window, and every [%r] is an offset added to it.
        auto so = shared_offsets.find(t);
        if (so != shared_offsets.end()) return std::to_string(so->second) + "ull";
        auto it = params.find(t);
        if (it != params.end())
          return buf_base(it->second);
        err = "unknown symbol '" + t + "'";
        return t;
      }
      return t;
    }
    std::string expr = reg_expr(t);
    Reg r = parse_reg(t);
    if (!r.ok || want.empty() || !r.special.empty()) return expr;
    auto it = reg_types.find(r.base);
    if (it == reg_types.end() || it->second == want) return expr;
    if (cpp_size(it->second) == cpp_size(want)) return "weft_bitcast<" + want + ">(" + expr + ")";
    return "(" + want + ")(" + expr + ")";
  }

  std::string store(const std::string& dst_tok, const std::string& expr, const std::string& want) {
    std::string dst = reg_expr(dst_tok);
    Reg r = parse_reg(dst_tok);
    std::string core = expr;
    if (r.ok && !want.empty() && r.special.empty()) {
      auto it = reg_types.find(r.base);
      if (it != reg_types.end() && it->second != want) {
        if (cpp_size(it->second) == cpp_size(want))
          core = "weft_bitcast<" + it->second + ">(" + expr + ")";
        else
          core = "(" + it->second + ")(" + expr + ")";
      }
    }
    if (!vec || !vec_select) return dst + " = " + core + ";";
    // Inactive lanes keep what they had: they are on a different path. This is a
    // bitwise select rather than a ternary because a ternary over a per-lane
    // mask is the one shape the vectoriser refuses: as a ternary these loops
    // compile to 32 scalar steps, as a mask they become 8 NEON ops.
    std::string dt = "uint32_t";
    if (r.ok) {
      auto it = reg_types.find(r.base);
      if (it != reg_types.end()) dt = it->second;
    }
    const bool wide = cpp_size(dt) == 8;
    const std::string mk = wide ? "am64[l]" : "am[l]";
    std::string body;
    if (dt == "float") {
      body = "weft_bitcast<float>((weft_bitcast<uint32_t>(" + dst + ") & ~" + mk +
             ") | (weft_bitcast<uint32_t>(" + core + ") & " + mk + "))";
    } else if (wide) {
      body = "(" + dst + " & ~" + mk + ") | ((uint64_t)(" + core + ") & " + mk + ")";
    } else {
      body = "(" + dst + " & ~" + mk + ") | ((uint32_t)(" + core + ") & " + mk + ")";
    }
    return dst + " = " + body + ";";
  }

  // The lanes this instruction applies to, as a C++ boolean over the lane index.
  std::string lane_cond() const {
    return vec_pred.empty() ? "am[l]" : "(am[l] && (" + vec_pred + "))";
  }

  // A statement that writes memory has to run inside a lane loop: a load for an
  // inactive lane would use an address that lane never computed.
  std::string lane_block(const std::string& body) {
    vec_side = true;
    return "for (uint32_t l = 0; l < 32u; l++) { if (" + lane_cond() + ") { " + body + " } }";
  }

  std::string reg_type_of(const std::string& tok) {
    Reg r = parse_reg(tok);
    if (!r.ok) return "";
    auto it = reg_types.find(r.base);
    return it == reg_types.end() ? "" : it->second;
  }

  std::string reg_expr(const std::string& tok) {
    Reg r = parse_reg(tok);
    if (!r.ok) {
      err = "expected a register, got '" + tok + "'";
      return "0";
    }
    if (!r.special.empty()) {
      if (r.special == "tid.x") return vec ? "(ctx.lid0 + l)" : "ctx.lid";
      if (r.special == "tid.y" || r.special == "tid.z") return "0u";
      if (r.special == "ntid.x") return "ctx.tg_size";
      if (r.special == "ntid.y" || r.special == "ntid.z") return "1u";
      if (r.special == "ctaid.x") return "ctx.tgid";
      if (r.special == "ctaid.y" || r.special == "ctaid.z") return "0u";
      if (r.special == "nctaid.x") return "((ctx.grid + ctx.tg_size - 1) / ctx.tg_size)";
      if (r.special == "laneid") return vec ? "l" : "(ctx.lid & 31u)";
      err = "unsupported special register %" + r.special;
      return "0";
    }
    return "s->" + r.base + "[" + std::to_string(r.index) + "]" + (vec ? "[l]" : "");
  }

  // `[%rd1+16]` -> ("rd[1] + 16", base register). `[tile]` -> ("0", "").
  bool mem_operand(const std::string& op, std::string& addr, std::string& param_name,
                   std::string* base_reg = nullptr) {
    std::string s = trim(op);
    if (s.size() < 2 || s.front() != '[' || s.back() != ']') {
      err = "expected a memory operand, got '" + op + "'";
      return false;
    }
    s = trim(s.substr(1, s.size() - 2));
    if (base_reg) base_reg->clear();
    if (!s.empty() && s[0] != '%') {
      // A symbol, and possibly an offset into it: `[tile]` or `[tile + 4]`. The name
      // has to stay the array's, because that is the key the arena is laid out by,
      // and the offset belongs in the address every access adds to that base.
      // Keeping both inside the name put every `[tile + n]` at the array's first
      // byte, so the second word of anything read as its first.
      const size_t cut = s.find_first_of("+-");
      if (cut == std::string::npos) {
        param_name = s;  // a symbol: param or shared array name
        addr = "0";
        return true;
      }
      param_name = trim(s.substr(0, cut));
      std::string off = trim(s.substr(cut + 1));
      if (!off.empty() && off[0] == '%') off = reg_expr(off);
      addr = s[cut] == '-' ? ("(0 - " + off + ")") : ("(0 + " + off + ")");
      return true;
    }
    std::string base;
    bool negative = false;
    size_t cut = std::string::npos;
    for (size_t i = 1; i < s.size(); i++) {
      if (s[i] == '+' || s[i] == '-') { cut = i; negative = (s[i] == '-'); break; }
    }
    if (cut == std::string::npos) {
      addr = "(" + reg_expr(s) + ")";
      if (base_reg) {
        Reg r = parse_reg(s);
        *base_reg = (r.ok && r.special.empty()) ? reg_key(r) : std::string();
      }
      return err.empty();
    }
    const std::string first = trim(s.substr(0, cut));
    base = reg_expr(first);
    if (base_reg) {
      Reg r = parse_reg(first);
      *base_reg = (r.ok && r.special.empty()) ? reg_key(r) : std::string();
    }
    std::string off = trim(s.substr(cut + 1));
    addr = "(" + base + (negative ? " - " : " + ") + off + ")";
    return err.empty();
  }

  // The parameter a token names: the symbol itself, or the register an address is
  // walking through. -1 when it names neither.
  int ptr_index(const std::string& tok) const {
    const std::string t = trim(tok);
    if (t.empty()) return -1;
    if (t[0] != '%') {
      auto it = params.find(t);
      return it == params.end() ? -1 : it->second;
    }
    Reg r = parse_reg(t);
    if (!r.ok || !r.special.empty()) return -1;
    auto it = ptr_param.find(reg_key(r));
    return it == ptr_param.end() ? -1 : it->second;
  }

  // Follow an address from the binding to the register that holds it. Only a
  // 64-bit destination is a pointer, which is what keeps `add.s32 %r5, %r4, %r1`
  // from being mistaken for one.
  void track_ptr(const std::string& op, const std::vector<std::string>& parts,
                 const std::vector<std::string>& args) {
    if (args.size() < 2) return;
    Reg d = parse_reg(args[0]);
    if (!d.ok || !d.special.empty()) return;
    if (cpp_size(reg_type_of(args[0])) != 8) return;
    // `ld.param.u64 %rd1, [buf];` is where the kernel's own parameter becomes a
    // register, so it is the head of every address chain.
    if (op == "ld" && has_part(parts, "param")) {
      std::string s = trim(args[1]);
      if (s.size() >= 2 && s.front() == '[' && s.back() == ']') s = trim(s.substr(1, s.size() - 2));
      const int i = ptr_index(s);
      if (i >= 0) ptr_param[reg_key(d)] = i;
      return;
    }
    if (op == "cvta" || op == "mov" || op == "add" || op == "lea") {
      const int i = ptr_index(args[1]);
      if (i >= 0) ptr_param[reg_key(d)] = i;
    }
  }

  // Record what width one access addresses a buffer parameter at. `sym` is the
  // operand when it names a symbol outright (`[buf]`), `reg` the register it
  // walks through otherwise.
  void note_width(const std::string& sym, const std::string& reg, const std::string& ty) {
    int i = -1;
    auto p = params.find(sym);
    if (p != params.end()) {
      i = p->second;
    } else if (!reg.empty()) {
      auto r = ptr_param.find(reg);
      if (r != ptr_param.end()) i = r->second;
    }
    if (i < 0) return;
    const int n = cpp_size(ty);
    if (!n) return;
    auto it = param_bytes.find(i);
    if (it == param_bytes.end()) param_bytes[i] = (uint32_t)n;
    else if (it->second != (uint32_t)n) it->second = 0;  // two widths: no answer
  }

  // One memory access for the 32 lanes of the warp. When what the lanes touch
  // is one contiguous run — which is every global access in a kernel that
  // indexes by global thread id — a full warp becomes a plain loop over that
  // run: no mask, no lane index, and the compiler turns it into vector loads.
  // A partial warp, or an address the value model cannot describe, falls back to
  // the masked per-lane loop, which is always correct.
  std::string vec_mem_access(const std::string& memop, int esz, const std::string& full_body,
                             const std::string& masked_body, const std::string& ptr_ty) {
    vec_side = true;
    std::string fallback = lane_block(masked_body);
    const VecVal as = mem_sym(memop);
    if (!vec_pred.empty() || !as.ok || as.stride != esz) return fallback;
    return "{ if (weft_full) { " + ptr_ty + "* weft_p = (" + ptr_ty + "*)(uintptr_t)(" + as.base +
           "); for (uint32_t l = 0; l < 32u; l++) { " + full_body + " } } else { " + fallback +
           " } }";
  }

  // ---- how each target spells the same thing ----
  // The address space of a pointer is part of its type in Metal and absent in
  // C++, so a cast carries it or does not; the rest is a name.
  std::string addr_space(const char* space, bool cst) const {
    std::string q;
    if (cst) q += "const ";
    if (msl) { q += space; q += " "; }
    return q;
  }
  std::string buf_base(int i) const {
    const std::string n = std::to_string(i);
    return msl ? ("weft::buf_addr(ctx, " + n + ")")
               : ("(uint64_t)(uintptr_t)weft::buffer(ctx, " + n + ")");
  }
  std::string buf_base_as(const std::string& ty, int i) const {
    const std::string n = std::to_string(i);
    return "(" + ty + ")" + (msl ? ("weft::buf_addr(ctx, " + n + ")")
                                  : ("(uintptr_t)weft::buffer(ctx, " + n + ")"));
  }
  std::string param_of(int i) const {
    const std::string n = std::to_string(i);
    return msl ? ("weft::param(ctx, " + n + ")") : ("weft::param(" + n + ")");
  }
  // A math call, named the way the target names it: `std::` on the CPU, the
  // bare name on the GPU, where `using namespace metal` puts the same functions
  // in scope with the same rounding.
  std::string math(const std::string& name, const std::string& args) const {
    return msl ? (name + "(" + args + ")") : ("std::" + name + "(" + args + ")");
  }
  // The identity of a min/max fold: the value it leaves the other operand alone
  // for. Metal has no numeric_limits, so the limits are the constants.
  std::string lim_expr(const std::string& ty, const std::string& rop) const {
    if (!msl)
      return "std::numeric_limits<" + ty + ">::" + (rop == "min" ? "max()" : "lowest()");
    const bool low = (rop != "min");
    if (ty == "float") return low ? "-INFINITY" : "INFINITY";
    if (ty == "int32_t") return low ? "INT_MIN" : "INT_MAX";
    if (ty == "uint32_t") return low ? "0u" : "UINT_MAX";
    if (ty == "int8_t") return low ? "-128" : "127";
    return low ? "0" : "255";
  }
  // A rounding function applied to a value on its way to an integer. The CPU
  // widens to double because that keeps .rn exact for every f32 input; the GPU
  // has no double, and rint on the f32 itself is exact for the same reason — a
  // float has no bits below its own ulp to lose.
  std::string rounded_math(const std::string& fn, const std::string& src) const {
    if (msl) return (fn == "nearbyint" ? std::string("rint") : fn) + "(" + src + ")";
    return "std::" + fn + "((double)" + src + ")";
  }
  std::string isnan_of(const std::string& x) const {
    // Metal has no double, so the test is on the float itself: same answer, no
    // widening the hardware cannot do.
    return msl ? ("isnan(" + x + ")") : ("std::isnan((double)" + x + ")");
  }

  std::string emit(const std::string& raw);
  std::string translate_body(const std::string& body);
};

// The warp back end for the GPU.
//
// One Metal thread runs one whole 32-lane warp, with the register file in thread
// memory, so a peer lane's register is data rather than a hardware lane. Every
// tensor-core kernel the front end translates — the `mma` forms and the
// `ldmatrix` staging around them — gets a GPU column from here, and each one
// agrees with its CPU twin value for value.

// The warp body as a Metal kernel body.
//
// The CPU warp back end is a computed goto: a label per block, a jump table
// indexed by the block the lanes are on, `goto dispatch` to re-decide. A shader
// has neither labels nor goto (`labeled statements are not supported in Metal`),
// so the same control flow is written as a switch inside a loop:
//
//   - each `Lk:;` opens `case ku`, and the block that follows ends with `break`;
//   - `goto dispatch` becomes that `break`, which leaves the switch for the loop
//     head, where the masks are recomputed from the pc values — so the lanes
//     that come back out are the same lanes that went in;
//   - `goto Lk` becomes a write of k into every active lane's pc and then the
//     same break, which is the same thing the jump did: those lanes run block k
//     next, and the dispatcher picks k because that is where they all are.
static std::string warp_switch_body(const std::string& code) {
  std::vector<std::string> segs(1);
  std::vector<int> idx(1, 0);
  size_t pos = 0;
  while (pos < code.size()) {
    const size_t eol = code.find('\n', pos);
    const size_t end = (eol == std::string::npos) ? code.size() : eol;
    const std::string line = code.substr(pos, end - pos);
    const std::string t = trim(line);
    bool label = false;
    if (t.size() > 3 && t[0] == 'L') {
      size_t d = 1;
      while (d < t.size() && t[d] >= '0' && t[d] <= '9') d++;
      if (d > 1 && t.compare(d, std::string::npos, ":;") == 0) {
        // The label opens the block that follows it, so the segment the number
        // belongs to is the one just begun. The segment before it keeps the
        // number it had — which for the prologue is 0, the pc every lane starts
        // on, so the first dispatch enters the first block and not the block
        // after the first barrier.
        label = true;
        segs.emplace_back();
        idx.push_back(std::atoi(t.c_str() + 1));
      }
    }
    if (!label) {
      segs.back() += line;
      segs.back() += "\n";
    }
    if (eol == std::string::npos) break;
    pos = eol + 1;
  }
  // The first segment is whatever came before the first label, and the last index
  // entry is the placeholder opened by the final label.
  if (!segs.empty() && trim(segs.front()).empty()) {
    segs.erase(segs.begin());
    idx.erase(idx.begin());
  }
  std::string out;
  for (size_t i = 0; i < segs.size(); i++) {
    std::string seg;
    size_t at = 0;
    const std::string& s = segs[i];
    while (at < s.size()) {
      const size_t g = s.find("goto ", at);
      if (g == std::string::npos) {
        seg += s.substr(at);
        break;
      }
      seg += s.substr(at, g - at);
      const size_t semi = s.find(';', g);
      if (semi == std::string::npos) {
        seg += s.substr(g);
        break;
      }
      const std::string target = trim(s.substr(g + 5, semi - g - 5));
      if (target == "dispatch") {
        seg += "break;";
      } else if (target.size() > 1 && target[0] == 'L') {
        // Braced, because a jump is a *statement* and can be the whole body of an
        // `if`: `if (!sy) goto L1;` is how a partially-predicated branch sends the
        // lanes that all agree. Written without braces, the `break` that follows
        // would be the next statement after the `if` and would run whatever the
        // condition said — which threw away the pc write and sent every lane back
        // to the case it had just run, forever.
        seg += "{ for (uint32_t l = 0; l < weft::kWarpLanes; l++) if (am[l]) s->pc[l] = " +
               target.substr(1) + "u; break; }";
      } else {
        seg += s.substr(g, semi - g + 1);
      }
      at = semi + 1;
    }
    // A block that runs into the next one has to say so. In the goto form the
    // next block is simply the next statement; here it is another `case`, so "the
    // instruction after this one" is a program-counter write. Without it a block
    // that does not itself transfer control leaves every lane where it was and the
    // dispatcher picks the same case again — an infinite loop that looks like a
    // kernel that never finishes.
    if (i + 1 < segs.size() && seg.find("s->pc[l]") == std::string::npos &&
        seg.find("return") == std::string::npos) {
      seg += "  { for (uint32_t l = 0; l < weft::kWarpLanes; l++) if (am[l]) s->pc[l] = " +
             std::to_string(idx[i + 1]) + "u; }\n";
    }
    out += "      case " + std::to_string(idx[i]) + "u: {\n" + seg + "      break; }\n";
  }
  return out;
}

// One PTX statement -> one C++ line.
std::string Emitter::emit(const std::string& raw) {
  std::string s = trim(raw);
  vec_pred.clear();  // each instruction carries its own predicate
  if (s.empty()) return "";

  // Predication: @%p1 / @!%p1
  std::string guard;
  if (s[0] == '@') {
    size_t i = 1;
    bool negated = false;
    if (i < s.size() && s[i] == '!') { negated = true; i++; }
    size_t st = i;
    while (i < s.size() && !std::isspace((unsigned char)s[i])) i++;
    std::string pred = s.substr(st, i - st);
    std::string pe = reg_expr(pred);
    const std::string cond = negated ? "!(" + pe + ")" : pe;
    if (vec)
      vec_pred = vec_pred.empty() ? cond : "(" + vec_pred + ") && (" + cond + ")";
    else
      guard = "if (" + cond + ") ";
    s = trim(s.substr(i));
  }
  vec_control = false;
  vec_side = false;

  size_t sp = s.find_first_of(" \t");
  std::string opcode = sp == std::string::npos ? s : s.substr(0, sp);
  std::string rest = sp == std::string::npos ? "" : trim(s.substr(sp + 1));

  std::vector<std::string> parts = split_top(opcode, '.');
  if (parts.empty()) return "";
  const std::string& op = parts[0];

  std::vector<std::string> args = split_top(rest, ',');
  std::string ty;
  bool shared = false, param = false, global = false, is_const = false;
  for (const auto& p : parts) {
    std::string t = cpp_type("." + p);  // split on '.' drops the dot
    if (!t.empty()) ty = t;
    if (p == "shared") shared = true;
    if (p == "param") param = true;
    if (p == "global") global = true;
    if (p == "const") is_const = true;
  }

  auto need = [&](size_t n) {
    if (args.size() < n) err = "instruction '" + opcode + "' needs " + std::to_string(n) + " operands";
    return args.size() >= n;
  };

  // Every statement flows through the value model before it is emitted, so a
  // load or a store later on can see whether the lanes are walking consecutive
  // addresses.
  vec_update(opcode, args);
  track_ptr(op, parts, args);

  if (op == "ld" || op == "st") {
    if (!need(2)) return "";
    if (ty.empty()) ty = "uint32_t";
    if (shared && !shared_size) {
      err = "shared memory used without a .shared declaration";
      return "";
    }
    (void)global;
    // .v2 / .v4 load or store a small vector: the same instruction four times
    // at consecutive addresses. The GPU coalesces them; the CPU just does four.
    int lanes = 1;
    for (const auto& p : parts) {
      if (p == "v2") lanes = 2;
      else if (p == "v4") lanes = 4;
    }
    const int esz = cpp_size(ty) ? cpp_size(ty) : 4;
    // A 64-bit float is read and written as the word its bits are on the GPU
    // target, because the type itself does not exist there. The width is the same
    // eight bytes either way, so the addresses do not change.
    const std::string ty_mem = (msl && ty == "double") ? "uint64_t" : ty;

    auto at = [&](const std::string& a, int k) {
      return k == 0 ? a : "(" + a + " + " + std::to_string(k * esz) + ")";
    };

    if (op == "ld") {
      std::string addr, name, base;
      if (!mem_operand(args[1], addr, name, &base)) return "";
      // `ld.param.u64 %rd1, [buf]` reads the *binding*, which is an eight-byte
      // address; it says nothing about what the buffer holds.
      if (!param) note_width(name, base, ty);
      auto load_at = [&](const std::string& a) -> std::string {
        if (param) {
          auto b = params.find(name);
          auto sc = scalars.find(name);
          // `[buf + 8]` is a field inside what the binding points at, so the offset
          // is part of the address rather than part of the name.
          if (b != params.end()) {
            const std::string at_base = buf_base_as(ty, b->second);
            return addr == "0" ? at_base : ("(" + at_base + " + " + addr + ")");
          }
          if (sc != scalars.end()) return "(" + ty + ")" + param_of(sc->second);
          err = "unknown parameter '" + name + "'";
          return "0";
        }
        if (shared)
          return "*(" + addr_space("threadgroup", is_const) + ty_mem + "*)(sh + " + a + ")";
        return "*(" + addr_space("device", is_const) + ty_mem + "*)(uintptr_t)(" + a + ")";
      };
      std::string dsts = trim(args[0]);
      if (lanes == 1 || dsts.find('{') == std::string::npos) {
        if (vec) {
          const bool sel = vec_select;
          vec_select = false;
          std::string plain = store(args[0], "weft_p[l]", ty);
          vec_select = sel;
          return vec_mem_access(args[1], esz, plain, store(args[0], load_at(addr), ty),
                                "const " + ty_mem);
        }
        return guard + store(args[0], load_at(addr), ty);
      }
      dsts = dsts.substr(dsts.find('{') + 1);
      dsts = dsts.substr(0, dsts.find('}'));
      auto regs = split_top(dsts, ',');
      if ((int)regs.size() != lanes) { err = "vector load register count mismatch"; return ""; }
      std::string out;
      for (int k = 0; k < lanes; k++) out += store(regs[k], load_at(at(addr, k)), ty) + " ";
      if (vec) {
        const bool sel = vec_select;
        vec_select = false;
        out.clear();
        for (int k = 0; k < lanes; k++) out += store(regs[k], load_at(at(addr, k)), ty) + " ";
        vec_select = sel;
        return lane_block(out);
      }
      return guard + out;
    }

    std::string addr, name, base;
    if (!mem_operand(args[0], addr, name, &base)) return "";
    if (!param) note_width(name, base, ty);
    auto store_at = [&](const std::string& a, const std::string& v) -> std::string {
      if (shared)
        return "*((" + addr_space("threadgroup", false) + ty_mem + "*)(sh + " + a + ")) = " + v +
               ";";
      return "*((" + addr_space("device", false) + ty_mem + "*)(uintptr_t)(" + a + ")) = " + v +
             ";";
    };
    std::string srcs = trim(args[1]);
    if (lanes == 1 || srcs.find('{') == std::string::npos) {
      if (vec) {
        std::string val = operand_as(args[1], ty_mem);
        return vec_mem_access(args[0], esz, "weft_p[l] = " + val + ";", store_at(addr, val), ty);
      }
      return guard + store_at(addr, operand_as(args[1], ty_mem));
    }
    srcs = srcs.substr(srcs.find('{') + 1);
    srcs = srcs.substr(0, srcs.find('}'));
    auto vals = split_top(srcs, ',');
    if ((int)vals.size() != lanes) { err = "vector store register count mismatch"; return ""; }
    std::string out;
    for (int k = 0; k < lanes; k++)
      out += store_at(at(addr, k), operand_as(vals[k], ty_mem)) + " ";
    if (vec) return lane_block(out);
    return guard + out;
  }

  if (op == "mov") {
    if (!need(2)) return "";
    if (ty.empty()) ty = reg_type_of(args[0]);
    // mov.b64 %rd, {%lo, %hi};  and  mov.b64 {%lo, %hi}, %rd;
    auto braced = [](const std::string& s) { return s.find('{') != std::string::npos; };
    if (braced(args[0])) {
      std::string list = trim(args[0]);
      list = list.substr(list.find('{') + 1);
      list = list.substr(0, list.find('}'));
      auto parts2 = split_top(list, ',');
      if (parts2.size() != 2) { err = "mov with a brace list needs two registers"; return ""; }
      return guard + store(parts2[0], "(uint32_t)(" + reg_expr(args[1]) + " & 0xFFFFFFFFu)", "uint32_t") +
             " " + store(parts2[1], "(uint32_t)((" + reg_expr(args[1]) + ") >> 32)", "uint32_t");
    }
    if (braced(args[1])) {
      std::string list = trim(args[1]);
      list = list.substr(list.find('{') + 1);
      list = list.substr(0, list.find('}'));
      auto parts2 = split_top(list, ',');
      if (parts2.size() != 2) { err = "mov with a brace list needs two registers"; return ""; }
      return guard + store(args[0],
                           "((uint64_t)(uint32_t)" + reg_expr(parts2[0]) + " | ((uint64_t)(uint32_t)" +
                               reg_expr(parts2[1]) + " << 32))",
                           ty);
    }
    // A software double is moved as the word it lives in, not converted.
    if (msl && ty == "double")
      return guard + store(args[0], operand_as(args[1], "uint64_t"), "uint64_t");
    return guard + store(args[0], operand_as(args[1], ty), ty);
  }

  if (op == "cvta") {
    if (!need(2)) return "";
    return guard + reg_expr(args[0]) + " = " + reg_expr(args[1]) + ";";
  }

  // cvt.rn.{f16,bf16}x2.f32 d, a, b: two floats to two 16-bit values, the first
  // source in the high half. This is the form LLVM's NVPTX backend emits to pack
  // a pair, and it is not a scalar conversion, so it is handled before the table
  // below.
  if (op == "cvt" && (has_part(parts, "f16x2") || has_part(parts, "bf16x2") ||
                      has_part(parts, "e4m3x2") || has_part(parts, "e5m2x2"))) {
    if (!need(3)) return "";
    const char* pack = has_part(parts, "e4m3x2") ? "weft_e4m3x2("
                       : has_part(parts, "e5m2x2") ? "weft_e5m2x2("
                       : has_part(parts, "bf16x2") ? "weft_bf16x2("
                                                    : "weft_f16x2(";
    return store(args[0], std::string(pack) + operand_as(args[1], "float") + ", " +
                               operand_as(args[2], "float") + ")",
                 "uint32_t");
  }

  // cvt[.rn|.rz|.rm|.rp][.ftz].dst.src d, a
  //
  // The destination type is the first type in the modifier list and the source
  // is the last, because a rounding mode sits between them. Reading only
  // parts[1] made every `cvt.rn.f32.s32` an error — and rounding-mode converts
  // are what `__f2i_rn` and every float-to-int idiom in CUDA compile to.
  if (op == "cvt") {
    if (!need(2)) return "";
    std::string dstty, mode;
    // `.rni` and `.rn` are two different instructions, not two spellings: the
    // three-letter form rounds the *value* to an integer and keeps the float type
    // (`rint`), and the two-letter form rounds the significand to the destination's
    // precision. Treating the first as the second turns 2.5 into 2.
    bool integer_round = false;
    for (size_t i = 1; i < parts.size(); i++) {
      const std::string t = cpp_type("." + parts[i]);
      if (!t.empty()) {
        if (dstty.empty()) dstty = t;
      } else if ((parts[i] == "rn" || parts[i] == "rz" || parts[i] == "rm" ||
                  parts[i] == "rp" || parts[i] == "rni" || parts[i] == "rzi" ||
                  parts[i] == "rmi" || parts[i] == "rpi")) {
        // PTX spells these .rni/.rzi/.rmi/.rpi; the trailing `i` is what makes
        // them round to an integer rather than to the destination's precision.
        mode = parts[i].substr(0, 2);
        integer_round = parts[i].size() == 3;
      } else if (parts[i] == "ftz" || parts[i] == "sat") {
        // Renormalising a denormal and clamping to the destination range are
        // both what the CPU does anyway for the conversions reached here.
      } else {
        err = "unsupported conversion modifier in '" + opcode + "'";
        return "";
      }
    }
    if (dstty.empty()) { err = "unsupported conversion '" + opcode + "'"; return ""; }
    // A conversion to or from a 64-bit float on the GPU target goes through the
    // software double: the value is bits, so a cast would convert the pattern.
    if (msl && (dstty == "double" || ty == "double")) {
      const std::string b = operand_as(args[1], "uint64_t");
      if (dstty == "double") {
        if (ty == "double") return guard + store(args[0], b, "uint64_t");
        if (ty == "float")
          return guard + store(args[0],
                               "weft_fp64::from_f32(weft_bitcast<uint32_t>(" +
                                   operand_as(args[1], "float") + "))",
                               "uint64_t");
        const bool sgn = ty == "int32_t" || ty == "int64_t";
        const std::string wide = sgn ? ("(int64_t)" + operand_as(args[1], "int64_t"))
                                     : ("(uint64_t)" + operand_as(args[1], "uint64_t"));
        return guard + store(args[0],
                             std::string("weft_fp64::from_") + (sgn ? "i64(" : "u64(") + wide + ")",
                             "uint64_t");
      }
      // A double read as a float: the narrowing *is* the rounding, and `to_f32`
      // rounds the significand to nearest even. Rounding to an integer first would
      // be a different number — 2.5 would come back as 2 — so the mode is not
      // applied here at all.
      if (dstty == "float") {
        // `.rni` to a float is a round to an integer that stays a float.
        std::string fv = b;
        if (integer_round && mode == "rn") fv = "weft_fp64::rne_bits(" + fv + ")";
        return guard + store(args[0], "weft_bitcast<float>(weft_fp64::to_f32(" + fv + "))",
                             "float");
      }
      // A double read as an integer: the mode rounds the double to an integer
      // first, because the hardware rounds before it truncates too.
      std::string v = b;
      if (!mode.empty() && mode != "rz")
        v = std::string("weft_fp64::") +
            (mode == "rn" ? "rne_bits(" : mode == "rm" ? "floor_bits(" : "ceil_bits(") + v + ")";
      const bool sgn = dstty == "int32_t" || dstty == "int64_t";
      const std::string c = std::string("weft_fp64::to_") + (sgn ? "i64(" : "u64(") + v + ")";
      return guard + store(args[0], "(" + dstty + ")" + c, dstty);
    }
    // The source is coerced to the *instruction's* type, not the register's:
    // `cvt.rzi.s32.f32` on a `.b32` register holds a float in it, and reading
    // those bits as an integer converts the bit pattern rather than the value.
    const std::string src = operand_as(args[1], ty);
    const bool dst_f = dstty == "float" || dstty == "double";
    const bool src_f = ty == "float" || ty == "double";
    // Rounding only exists between floats, and between a float and an integer.
    // Everything else is a truncation and a cast.
    std::string e;
    // Rounding is a step between two floats. An integer source has no fraction
    // to round off, and on the GPU `rint` of an unsigned int is ambiguous
    // between the half and float overloads, so the conversion is a plain cast.
    if (!mode.empty() && dst_f && src_f) {
      if (integer_round) {
        // `.rni` rounds the value to an integer and keeps the float type, which is
        // `rint` — what a compiler emits before narrowing to a half.
        const char* fn = mode == "rn" ? "nearbyint"
                       : mode == "rz" ? "trunc"
                       : mode == "rm" ? "floor"
                                       : "ceil";
        e = "(" + dstty + ")(" + rounded_math(fn, src) + ")";
      } else if (mode == "rn") {
        // Between two floats, `.rn` is what a cast already does: it rounds the
        // significand to the destination's precision, ties to even. Rounding to an
        // integer first would be a different number — `cvt.rn.f32.f64` of
        // 1234.5678 came out as 1235.
        e = "(" + dstty + ")(" + src + ")";
      } else {
        const char* fn = mode == "rz" ? "trunc" : (mode == "rm" ? "floor" : "ceil");
        e = "(" + dstty + ")(" + rounded_math(fn, src) + ")";
      }
    } else if (!mode.empty() && src_f) {
      // Float to integer: the cast truncates, so .rz needs nothing and the
      // other three round first. Rounding a float in double keeps .rn exact for
      // every f32 input, halves included.
      if (mode == "rz") e = "(" + dstty + ")(" + src + ")";
      else {
        const char* fn = mode == "rn" ? "nearbyint" : (mode == "rm" ? "floor" : "ceil");
        e = "(" + dstty + ")(" + rounded_math(fn, src) + ")";
      }
    } else {
      e = "(" + dstty + ")(" + src + ")";
    }
    return guard + store(args[0], e, dstty);
  }

  if (op == "setp") {
    if (!need(3)) return "";
    std::string cmp = parts.size() > 1 ? parts[1] : "";
    const char* sym = nullptr;
    if (cmp == "eq") sym = "==";
    else if (cmp == "ne") sym = "!=";
    else if (cmp == "lt") sym = "<";
    else if (cmp == "le") sym = "<=";
    else if (cmp == "gt") sym = ">";
    else if (cmp == "ge") sym = ">=";
    else if (cmp == "lo") sym = "<";
    else if (cmp == "ls") sym = "<=";
    else if (cmp == "hi") sym = ">";
    else if (cmp == "hs") sym = ">=";
    // NaN tests are not comparisons: .nan is true when *either* operand is a
    // NaN, .num when neither is, and every other predicate is ordered.
    if (msl && ty == "double" && (cmp == "nan" || cmp == "num")) {
      const std::string a = operand_as(args[1], "uint64_t"), b = operand_as(args[2], "uint64_t");
      const std::string any = "(weft_fp64::is_nan(" + a + ") || weft_fp64::is_nan(" + b + "))";
      return guard + store(args[0], cmp == "nan" ? any : ("(!" + any + ")"), "bool");
    }
    if (cmp == "nan" || cmp == "num") {
      const std::string a = operand_as(args[1], ty), b = operand_as(args[2], ty);
      const std::string an = isnan_of(a);
      const std::string bn = isnan_of(b);
      return guard + store(args[0], cmp == "nan" ? "(" + an + " || " + bn + ")"
                                                  : "(!" + an + " && !" + bn + ")",
                           "bool");
    }
    if (!sym) { err = "unsupported comparison '" + opcode + "'"; return ""; }
    // An ordered comparison of two software doubles is the helper, not an
    // operator: the same functions the SPIR-V front end calls for the same
    // predicates.
    if (msl && ty == "double") {
      const std::string a = operand_as(args[1], "uint64_t"), b = operand_as(args[2], "uint64_t");
      const std::string e = cmp == "eq" ? ("weft_fp64::eq(" + a + ", " + b + ")")
                          : cmp == "ne" ? ("(!weft_fp64::eq(" + a + ", " + b + "))")
                          : cmp == "lt" ? ("weft_fp64::lt(" + a + ", " + b + ")")
                          : cmp == "le" ? ("weft_fp64::le(" + a + ", " + b + ")")
                          : cmp == "gt" ? ("weft_fp64::gt(" + a + ", " + b + ")")
                                        : ("weft_fp64::ge(" + a + ", " + b + ")");
      return guard + store(args[0], e, "bool");
    }
    std::string boolop;
    for (const auto& p : parts)
      if (p == "and" || p == "or" || p == "xor") boolop = p;
    std::string cond = "(" + operand_as(args[1], ty) + " " + sym + " " + operand_as(args[2], ty) + ")";
    if (!boolop.empty() && args.size() >= 4) {
      const char* bs = boolop == "and" ? " && " : (boolop == "or" ? " || " : " ^ ");
      cond = cond + bs + operand_as(args[3], "bool");
    }
    return guard + store(args[0], cond, "bool");
  }

  if (op == "bra") {
    if (!need(1)) return "";
    vec_control = true;
    if (!vec) {
      // Metal has no goto (tools/msl_probe.mm), so a branch is a program counter
      // and a restart of the dispatch switch: the block it names is a case, and
      // the ids are handed out by label name so a forward jump lands on the case
      // the label later defines.
      if (msl)
        return guard + "{ s->pc = " + std::to_string(label_id(trim(args[0]))) +
               "u; continue; }";
      return guard + "goto " + sanitize(args[0]) + ";";
    }

    const int tgt = label_id(args[0]);
    if (vec_pred.empty()) return "goto L" + std::to_string(tgt) + ";";

    // A predicated branch splits the warp. Lanes that go and lanes that stay get
    // their own program counters and the dispatcher picks up the pieces; when
    // both groups reach the same instruction their counters match again and they
    // merge into one group, which is what reconvergence is.
    const int fall = label_count++;
    std::string out = "{ uint32_t tk = 0, sy = 0;\n";
    out += "    for (uint32_t l = 0; l < 32u; l++) { if (!am[l]) continue; if (" + vec_pred +
           ") tk |= 1u << l; else sy |= 1u << l; }\n";
    out += "    if (!sy) goto L" + std::to_string(tgt) + ";\n";
    // Unconditional, not `if (tk)`: when the predicate is false for *every* lane
    // the group still has to move to the fall-through block, and skipping the
    // block on an empty `tk` left every program counter where it was, so the
    // dispatcher picked the same case until the launch was killed.
    out += "    {\n";
    out += "      for (uint32_t l = 0; l < 32u; l++) if (am[l])\n";
    out += "        s->pc[l] = ((tk >> l) & 1u) ? " + std::to_string(tgt) + "u : " +
           std::to_string(fall) + "u;\n";
    out += "      goto dispatch;\n    } }\n  L" + std::to_string(fall) + ":;";
    return out;
  }

  if (op == "bar" || op == "membar" || op == "fence") {
    if (op != "bar") return guard + ";";
    // bar.red.popc/or/and is `__syncthreads_count` and friends: a barrier whose
    // result is a fold over every thread in the group. It is two rendezvous —
    // publish and arrive, then the per-warp totals are known and every warp
    // folds them — because the one-rendezvous version makes every thread read
    // tg_size slots, and a barrier is cheaper than 256 loads.
    bool is_red = false;
    for (const auto& p : parts)
      if (p == "red") is_red = true;
    uses_barrier = true;
    if (is_red) {
      int mode = -1;
      for (const auto& p : parts) {
        if (p == "popc") mode = 0;
        else if (p == "and") mode = 1;
        else if (p == "or") mode = 2;
      }
      // bar.red.<op>.<type> d, a, b — a is the barrier id and b the predicate.
      if (mode < 0 || args.size() < 3) {
        err = "unsupported reduction barrier '" + opcode + "'";
        return "";
      }
      const std::string m = std::to_string(mode);
      const std::string pred = operand_as(args[2], "bool");
      if (msl) {
        // On the GPU this is a publish, a real barrier, and a fold over the
        // group: the two rendezvous the CPU side has to book are what the
        // hardware's barrier already is.
        uses_red = true;
        return guard + "{ weft_red_slots[ctx.lid] = (" + pred + ") ? 1u : 0u;"
                       " threadgroup_barrier(mem_flags::mem_threadgroup); " +
               store(args[0], "weft::red_fold(weft_red_slots, ctx, " + m + ")", "uint32_t") +
               " threadgroup_barrier(mem_flags::mem_threadgroup); }";
      }
      const std::string pub = "weft::red_publish(ctx, (" + pred + ") ? 1u : 0u);";

      if (vec) {
        vec_control = true;
        const int id1 = label_count++, id2 = label_count++;
        const std::string at1 = "L" + std::to_string(id1), at2 = "L" + std::to_string(id2);
        std::string out;
        out += std::string("{ for (uint32_t l = 0; l < 32u; l++) if (am[l]) weft::red_publish(ctx, l, ") +
               "(" + pred + ") ? 1u : 0u);\n";
        out += "    for (uint32_t l = 0; l < 32u; l++) if (am[l]) s->pc[l] = " +
               std::to_string(id1) + "u; return weft::kWarpBarrier; }\n  " + at1 + ":;\n";
        out += "  { (void)weft::red_warp(ctx, " + m + "); }\n";
        out += "  { for (uint32_t l = 0; l < 32u; l++) if (am[l]) s->pc[l] = " +
               std::to_string(id2) + "u; return weft::kWarpBarrier; }\n  " + at2 + ":;\n";
        // The answer is the same in every lane, so it is stored under the usual
        // active mask: lanes on another path keep what they had.
        out += "  { const uint32_t weft_rg = weft::red_group(ctx, " + m +
               "); for (uint32_t l = 0; l < 32u; l++) { " + store(args[0], "weft_rg", "uint32_t") +
               " } }";
        return out;
      }

      const int id1 = next_resume++, id2 = next_resume++;
      resume_cases += "    case " + std::to_string(id1) + ": goto R" + std::to_string(id1) +
                      ";\n    case " + std::to_string(id2) + ": goto R" + std::to_string(id2) + ";\n";
      std::string out = pub + "\n  s->pc = " + std::to_string(id1) + "u; return;\n  R" +
                        std::to_string(id1) + ":;\n  (void)weft::red_warp(ctx, " + m +
                        ");\n  s->pc = " + std::to_string(id2) + "u; return;\n  R" +
                        std::to_string(id2) + ":;\n  " + store(args[0], "weft::red_group(ctx, " + m + ")", "uint32_t");
      return out;
    }
    if (vec) {
      vec_control = true;
      const int next = label_count++;
      return "{ for (uint32_t l = 0; l < 32u; l++) if (am[l]) s->pc[l] = " +
             std::to_string(next) + "u; return weft::kWarpBarrier; }\n  L" +
             std::to_string(next) + ":;";
    }
    if (msl) {
      // A barrier is the hardware's: every thread of the group arrives, nothing
      // has to be suspended, so there is no program counter to save and no
      // second entry into the body.
      return guard +
             "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);";
    }
    const int id = next_resume++;
    resume_cases += "    case " + std::to_string(id) + ": goto R" + std::to_string(id) + ";\n";
    // A barrier is just "save where I am and give the worker back".
    std::string label = "R" + std::to_string(id) + ":;";
    if (!guard.empty())
      return guard + "{ s->pc = " + std::to_string(id) + "u; return; }\n  " + label;
    return "s->pc = " + std::to_string(id) + "u; return;\n  " + label;
  }

  if (op == "ret" || op == "exit") {
    if (vec) {
      vec_control = true;
      return "{ for (uint32_t l = 0; l < 32u; l++) if (am[l]) s->pc[l] = 0xFFFFFFFFu; "
             "goto dispatch; }";
    }
    return guard + "s->pc = 0xFFFFFFFFu; return;";
  }

  // shfl.sync.<mode>.b32 d, a, b, clamp, mask
  //
  // A shuffle is a rendezvous among the 32 lanes of one warp, and the only
  // rendezvous primitive this runtime has is "give the worker back and resume
  // later". So: publish my lane's value, yield, read the source lane on the
  // next pass. Same shape as a barrier, which is why it costs the same.
  if (op == "shfl") {
    if (args.size() < 3) { err = "unsupported shuffle '" + opcode + "'"; return ""; }
    std::string mode;
    for (size_t i = 1; i < parts.size(); i++)
      if (parts[i] == "idx" || parts[i] == "up" || parts[i] == "down" || parts[i] == "bfly")
        mode = parts[i];
    if (mode.empty()) { err = "unsupported shuffle mode in '" + opcode + "'"; return ""; }

    uses_shfl = true;

    if (vec) {
      // 32 lanes are one vector here, so a shuffle is a gather inside the
      // register file: no publication, no suspension, no scheduler.
      Reg rs = parse_reg(args[1]);
      if (!rs.ok || !rs.special.empty()) {
        vec_ok = false;
        err = "shuffle of a value that is not a register";
        return "";
      }
      const std::string c = std::to_string([&] {
        uint32_t cl = (mode == "up") ? 0u : 31u;
        if (args.size() >= 4) {
          const std::string t = trim(args[3]);
          char* end = nullptr;
          const long v = std::strtol(t.c_str(), &end, 0);
          if (end && *end == '\0' && v >= 0 && v <= 31) cl = (uint32_t)v;
          else vec_ok = false;
        }
        return cl;
      }());
      const std::string lane = "l";
      const std::string b = operand_as(args[2], "uint32_t");
      std::string peer;
      if (mode == "idx")
        peer = "(" + lane + " & ~" + c + "u) | (" + b + " & " + c + "u)";
      else if (mode == "bfly")
        peer = "(" + lane + " & ~" + c + "u) | ((" + lane + " ^ " + b + ") & " + c + "u)";
      else if (mode == "up")
        peer = "(" + lane + " >= " + b + " + " + c + "u ? " + lane + " - " + b + " : " + lane + ")";
      else
        peer = "(" + lane + " + " + b + " <= " + c + "u ? " + lane + " + " + b + " : " + lane + ")";
      const std::string arr = "s->" + rs.base + "[" + std::to_string(rs.index) + "]";
      const std::string src = arr + "[" + peer + "]";
      const std::string masked = store(args[0], src, "uint32_t");

      // The common case: a delta that is the same in every lane, which is what
      // a reduction loop looks like (16, 8, 4, 2, 1). Then the peer of lane l
      // is lane l+d, so the whole shuffle is a shift of the vector — one run of
      // consecutive elements, which the compiler vectorises. As a per-lane
      // index it could not: `l + d` has a runtime term, so the loop goes scalar
      // and this becomes the most expensive instruction in the kernel.
      const VecVal dv = vec_sym(args[2]);
      const bool uniform_delta = dv.ok && dv.stride == 0;
      if (uniform_delta && vec_pred.empty() && (mode == "down" || mode == "up" || mode == "idx")) {
        std::string dtext;
        Reg dr = parse_reg(trim(args[2]));
        if (dr.ok && dr.special.empty()) {
          // Same value in every lane, so lane 0 holds it — and a full warp is
          // required for that to be current, which is exactly when this runs.
          dtext = "s->" + dr.base + "[" + std::to_string(dr.index) + "][0]";
        } else {
          dtext = operand_as(args[2], "uint32_t");
        }
        const bool sel = vec_select;
        vec_select = false;
        const std::string move0 = store(args[0], "weft_v[l - weft_d]", "uint32_t");
        const std::string move1 = store(args[0], "weft_v[l + weft_d]", "uint32_t");
        const std::string move_self = store(args[0], "weft_v[l]", "uint32_t");
        const std::string move_one = store(args[0], "weft_sv", "uint32_t");
        vec_select = sel;
        std::string shift;
        if (mode == "idx") {
          shift = "const uint32_t weft_s = (" + dtext + ") & " + c +
                  "u; const uint32_t* weft_v = " + arr +
                  "; const uint32_t weft_sv = weft_v[weft_s];" +
                  " for (uint32_t l = 0; l < 32u; l++) { " + move_one + " }";
        } else if (mode == "down") {
          shift = "const uint32_t weft_d = " + dtext + "; const uint32_t* weft_v = " + arr +
                  "; for (uint32_t l = 0; l + weft_d < 32u; l++) { " + move1 +
                  " } for (uint32_t l = (weft_d < 32u ? 32u - weft_d : 0u); l < 32u; l++) { " +
                  move_self + " }";
        } else {
          shift = "const uint32_t weft_d = (" + dtext + " < 32u ? " + dtext +
                  " : 32u); const uint32_t* weft_v = " + arr +
                  "; for (uint32_t l = 0; l < 32u; l++) { " + move_self +
                  " } for (uint32_t l = weft_d; l < 32u; l++) { " + move0 + " }";
        }
        // A full warp writes the shifted values straight back. A partial one
        // falls back to the per-lane gather: routing it through a staging array
        // measured slower than the scalar loop it would replace.
        vec_side = true;
        return "{ if (weft_full) { " + shift + " } else { " + lane_block(masked) + " } }";
      }
      return masked;
    }
    const int id = next_resume++;
    resume_cases += "    case " + std::to_string(id) + ": goto R" + std::to_string(id) + ";\n";
    const std::string label = "R" + std::to_string(id) + ":;";

    // `c` is the clamp: the far edge of the segment a lane may read from, in the
    // direction of travel. CUDA passes 0x1f for .down/.idx/.bfly and 0 for .up,
    // and a lane whose source falls outside keeps its own value — that is what
    // the hardware does, so it is what this does.
    uint32_t clamp = (mode == "up") ? 0u : 31u;
    if (args.size() >= 4) {
      const std::string t = trim(args[3]);
      char* end = nullptr;
      const long v = std::strtol(t.c_str(), &end, 0);
      if (!end || *end != '\0' || v < 0 || v > 31) {
        err = "unsupported shuffle clamp '" + t + "' in '" + opcode + "'";
        return "";
      }
      clamp = (uint32_t)v;
    }
    const std::string c = std::to_string(clamp);
    const std::string lane = "(ctx.lid & 31u)";
    const std::string base = "(ctx.lid & ~31u)";
    const std::string b = operand_as(args[2], "uint32_t");
    std::string src;
    if (mode == "idx")
      src = "(" + lane + " & ~" + c + "u) | (" + b + " & " + c + "u)";
    else if (mode == "bfly")
      src = "(" + lane + " & ~" + c + "u) | ((" + lane + " ^ " + b + ") & " + c + "u)";
    else if (mode == "up")
      src = "(" + lane + " >= " + b + " + " + c + "u ? " + lane + " - " + b + " : " + lane +
            ")";
    else
      src = "(" + lane + " + " + b + " <= " + c + "u ? " + lane + " + " + b + " : " + lane +
            ")";

    if (msl) {
      // The hardware moves the value between lanes, so a shuffle is one
      // instruction: no publication, no yield, no second pass. `src` is already
      // the source lane inside the warp, which is what simd_shuffle takes.
      return guard + store(args[0],
                           "simd_shuffle(" + operand_as(args[1], "uint32_t") +
                               ", (ushort)(" + src + "))",
                           "uint32_t");
    }

    // Pass parity: a lane publishes in one pass and reads in the next, so the
    // halves alternate and a lane that has moved on cannot clobber the value a
    // lagging lane still has to read.
    const std::string publish =
        "((uint32_t*)weft::shfl_pub_buf(ctx))[ctx.lid] = " + operand_as(args[1], "uint32_t") +
        "; s->pc = " + std::to_string(id) + "u; return;\n  ";
    const std::string read = "weft::shfl_read_buf(ctx)[" + base + " + (" + src + ")]";
    const std::string refine = store(args[0], read, "uint32_t");
    if (!guard.empty()) return guard + "{ " + publish + "}\n  " + label + "\n  " + guard + "{ " + refine + " }";
    return "{ " + publish + "}\n  " + label + "\n  " + refine;
  }

  // vote.sync.<ballot|all|any|uni>.b32 d, p, membermask;
  //
  // The warp has to agree before the answer exists, so this is the shuffle
  // rendezvous with a one-bit payload: publish the predicate, yield, then fold
  // what the whole warp published into a mask.
  if (op == "vote") {
    if (args.size() < 2) { err = "unsupported vote '" + opcode + "'"; return ""; }
    std::string mode;
    for (size_t i = 1; i < parts.size(); i++)
      if (parts[i] == "ballot" || parts[i] == "all" || parts[i] == "any" || parts[i] == "uni")
        mode = parts[i];
    if (mode.empty()) { err = "unsupported vote mode in '" + opcode + "'"; return ""; }
    uses_shfl = true;  // needs the same publish-yield-read rendezvous as a shuffle
    std::string member = "0xFFFFFFFFu";
    if (args.size() >= 3) {
      const std::string t = trim(args[2]);
      char* end = nullptr;
      const long v = std::strtol(t.c_str(), &end, 0);
      if (!end || *end != '\0') { err = "unsupported vote mask '" + t + "'"; return ""; }
      member = std::to_string((uint32_t)v) + "u";
    }

    if (vec) {
      // 32 lanes side by side: fold the predicate across the vector.
      const std::string p = operand_as(args[1], "bool");
      std::string core = "uint32_t weft_b = 0; for (uint32_t l = 0; l < 32u; l++) "
                         "if (am[l] && (" + p + ")) weft_b |= 1u << l;\n    weft_b &= " + member + ";";
      if (mode == "ballot") {
        vec_side = true;
        return "{ " + core + " for (uint32_t l = 0; l < 32u; l++) " +
               store(args[0], "weft_b", "uint32_t") + " }";
      }
      const std::string cond =
          mode == "all" ? "(weft_b == " + member + ")"
          : mode == "any" ? "(weft_b != 0u)"
                           : "(weft_b == 0u || weft_b == " + member + ")";
      vec_side = true;
      return "{ " + core + " for (uint32_t l = 0; l < 32u; l++) " + store(args[0], cond, "bool") +
             " }";
    }

    if (msl) {
      // One instruction again: the ballot is the hardware's, and the member mask
      // is the AND the PTX asked for.
      const std::string p = operand_as(args[1], "bool");
      std::string value;
      // simd_vote converts to its vote_t explicitly, so the cast goes through
      // that before the member mask.
      if (mode == "ballot")
        value = "((uint32_t)(simd_vote::vote_t)simd_ballot(" + p + ") & " + member + ")";
      else if (mode == "all") value = "simd_all(" + p + ")";
      else if (mode == "any") value = "simd_any(" + p + ")";
      else value = "(simd_all(" + p + ") || simd_all(!(" + p + ")))";
      return guard + store(args[0], value, mode == "ballot" ? "uint32_t" : "bool");
    }

    const int id = next_resume++;
    resume_cases += "    case " + std::to_string(id) + ": goto R" + std::to_string(id) + ";\n";
    const std::string label = "R" + std::to_string(id) + ":;";
    const std::string publish = "((uint32_t*)weft::shfl_pub_buf(ctx))[ctx.lid] = (" +
                                operand_as(args[1], "bool") + ") ? 1u : 0u; s->pc = " +
                                std::to_string(id) + "u; return;\n  ";
    const std::string ballot = "weft::vote_ballot(ctx)";
    std::string value;
    if (mode == "ballot") value = "(" + ballot + " & " + member + ")";
    else if (mode == "all") value = "((" + ballot + " & " + member + ") == " + member + ")";
    else if (mode == "any") value = "((" + ballot + " & " + member + ") != 0u)";
    else
      value = "(((" + ballot + " & " + member + ") == 0u) || ((" + ballot + " & " + member +
              ") == " + member + "))";
    const std::string refine =
        store(args[0], value, mode == "ballot" ? "uint32_t" : "bool");
    if (!guard.empty())
      return guard + "{ " + publish + "}\n  " + label + "\n  " + guard + "{ " + refine + " }";
    return "{ " + publish + "}\n  " + label + "\n  " + refine;
  }

  if (op == "selp") {
    if (!need(3)) return "";
    std::string p = args.size() > 3 ? reg_expr(args[3]) : "true";
    if (msl && ty == "double")
      return guard + store(args[0],
                           p + " ? " + operand_as(args[1], "uint64_t") + " : " +
                               operand_as(args[2], "uint64_t"),
                           "uint64_t");
    return guard + store(args[0],
                         p + " ? " + operand_as(args[1], ty) + " : " + operand_as(args[2], ty), ty);
  }

  static const std::map<std::string, const char*> binop = {
      {"add", "+"}, {"sub", "-"}, {"mul", "*"}, {"and", "&"}, {"or", "|"},
      {"xor", "^"}, {"shl", "<<"}, {"shr", ">>"}, {"div", "/"}, {"rem", "%"}};

  // mul.{lo,hi,wide}[.u32|.s32|.u64|.s64]
  //
  // The three forms differ only in which half of the product is kept, and only
  // .lo is a plain `*`: .hi is the high word, which is how a division by a
  // constant compiles (a magic-number multiply), so `mul.hi.u64` used to fall
  // through to the generic multiply and quietly return the *low* word. That is
  // a wrong answer rather than a refusal, so it is handled here explicitly.
  if (op == "mul" && parts.size() > 1 && args.size() >= 3) {
    bool wide = false, hi = false;
    for (size_t i = 1; i < parts.size(); i++) {
      if (parts[i] == "wide") wide = true;
      else if (parts[i] == "hi") hi = true;
    }
    const bool is64 = (ty == "uint64_t" || ty == "int64_t");
    const bool sgn = (ty == "int64_t" || ty == "int32_t" || ty == "int16_t" || ty == "int8_t");
    std::string dstty = reg_type_of(args[0]);
    const std::string a = operand_as(args[1], ty), b = operand_as(args[2], ty);
    if (wide) {
      const std::string w = sgn ? "int64_t" : "uint64_t";
      if (dstty.empty()) dstty = w;
      return guard + store(args[0], "(" + w + ")" + a + " * (" + w + ")" + b, dstty);
    }
    if (hi) {
      if (dstty.empty()) dstty = ty;
      // 64-bit: the high word is a helper rather than a 128-bit multiply. Both
      // targets define the helper (four 32x32 products and a carry chain), and
      // both are then exact — a `__int128` scalar compiles here, but this host's
      // Metal back end cannot turn the 128-bit *multiply* into machine code, and
      // the failure is a dead compiler (XPC_ERROR_CONNECTION_INTERRUPTED), not a
      // diagnostic.
      if (is64) {
        return guard + store(args[0],
                             std::string(sgn ? "weft_mulhi_s64(" : "weft_mulhi_u64(") + a + ", " +
                                 b + ")",
                             dstty);
      }
      const std::string prod = sgn ? "((int64_t)" + a + " * (int64_t)" + b + ")"
                                   : "((uint64_t)(uint32_t)" + a + " * (uint64_t)(uint32_t)" + b + ")";
      return guard + store(args[0], "(" + dstty + ")(" + prod + " >> 32)", dstty);
    }
  }

  // shf.{l,r}.{wrap,clamp}.b32 d, a, b, c: a funnel shift of the 64-bit pair
  // (a:b). LLVM emits it for a rotate and for byte-packing, and it is what a
  // 32-bit slice of a 64-bit shift-with-variable-amount becomes.
  if (op == "shf" && args.size() >= 4 && parts.size() >= 3) {
    const bool left = parts[1] == "l";
    bool clamp = false;
    for (size_t i = 2; i < parts.size(); i++) {
      if (parts[i] == "clamp") clamp = true;
      else if (parts[i] != "wrap" && parts[i] != "b32") {
        err = "unsupported funnel shift '" + opcode + "'";
        return "";
      }
    }
    const std::string call = std::string("weft_shf_") + (left ? "l" : "r") + "(" +
                             operand_as(args[1], "uint32_t") + ", " +
                             operand_as(args[2], "uint32_t") + ", " +
                             operand_as(args[3], "uint32_t") + (clamp ? ", 1)" : ", 0)");
    return guard + store(args[0], call, "uint32_t");
  }

  // lop3.b32 d, a, b, c, lut: an arbitrary three-input bitwise function, given
  // as a truth table in the low byte of the immediate. LLVM 23 does not emit it
  // at -O2 (it writes and/xor pairs instead), so this exists for hand-written
  // PTX and for whatever the combiner decides later.
  if (op == "lop3" && args.size() >= 4) {
    const std::string lut = args.size() >= 5 ? operand_as(args[4], "uint32_t") : "0u";
    return guard + store(args[0],
                         "weft_lop3(" + operand_as(args[1], "uint32_t") + ", " +
                             operand_as(args[2], "uint32_t") + ", " +
                             operand_as(args[3], "uint32_t") + ", " + lut + ")",
                         "uint32_t");
  }

  // testp.<class>.f32 p, a: the floating-point class tests.
  if (op == "testp" && args.size() >= 2) {
    std::string cls;
    for (size_t i = 1; i < parts.size(); i++)
      if (parts[i] != "f32" && parts[i] != "f64" && parts[i] != "ftz") cls = parts[i];
    // No cast to double on the way in: a float subnormal becomes a *normal*
    // double, so `testp.normal.f32` and `.subnormal.f32` would both answer wrong.
    const std::string v = operand_as(args[1], ty);
    std::string e;
    if (cls == "finite") e = math("isfinite", v);
    else if (cls == "infinite") e = math("isinf", v);
    else if (cls == "number") e = "!" + math("isnan", v);
    else if (cls == "normal") e = math("isnormal", v);
    else if (cls == "subnormal")
      // Metal has no fpclassify, and a subnormal is exactly "not zero and below
      // the smallest normal", so the test is the two comparisons.
      e = msl ? ("((" + v + " != 0.0f) && (fabs(" + v + ") < 0x1p-126f))")
              : ("(std::fpclassify(" + v + ") == FP_SUBNORMAL)");
    else if (cls == "nan") e = math("isnan", v);
    if (e.empty()) { err = "unsupported class test '" + opcode + "'"; return ""; }
    return guard + store(args[0], e, "bool");
  }

  // prmt.b32 d, a, b, c: four nibbles of `c` pick four bytes out of the eight in
  // (a:b). The default .f4e mode sign-replicates when a nibble's top bit is set.
  if (op == "prmt" && args.size() >= 3) {
    for (size_t i = 1; i < parts.size(); i++) {
      if (parts[i] != "b32" && parts[i] != "f4e") {
        err = "unsupported byte permute '" + opcode + "'";
        return "";
      }
    }
    const std::string c = args.size() >= 4 ? operand_as(args[3], "uint32_t") : "0u";
    return guard + store(args[0],
                         "weft_prmt(" + operand_as(args[1], "uint32_t") + ", " +
                             operand_as(args[2], "uint32_t") + ", " + c + ")",
                         "uint32_t");
  }

  // A 64-bit float on the GPU target is bits in a `uint64_t`, so there is no
  // operator to apply: each of these is the software double's own function, over
  // the same text the SPIR-V front end calls. The CPU column keeps the host's
  // double, which is what makes the two comparable bit for bit.
  if (msl && ty == "double" && args.size() >= 3) {
    const char* f64fn = op == "add"   ? "add"
                        : op == "sub" ? "sub"
                        : op == "mul" ? "mul"
                        : op == "div" ? "div"
                                      : nullptr;
    if (f64fn)
      return guard + store(args[0],
                           std::string("weft_fp64::") + f64fn + "(" +
                               operand_as(args[1], "uint64_t") + ", " +
                               operand_as(args[2], "uint64_t") + ")",
                           "uint64_t");
  }

  auto it = binop.find(op);
  if (it != binop.end() && args.size() >= 3) {
    return guard + store(args[0],
                         operand_as(args[1], ty) + " " + it->second + " " + operand_as(args[2], ty),
                         ty);
  }

  if (op == "fma" || op == "mad") {
    if (!need(4)) return "";
    // `mad` rounds twice and `fma` rounds once, so on the software double the first
    // is two calls and the second is the one helper that keeps the intermediate
    // exact — a 192-bit accumulator with a single rounding at the end.
    if (msl && ty == "double") {
      if (op == "mad")
        return guard + store(args[0],
                             "weft_fp64::add(weft_fp64::mul(" + operand_as(args[1], "uint64_t") +
                                 ", " + operand_as(args[2], "uint64_t") + "), " +
                                 operand_as(args[3], "uint64_t") + ")",
                             "uint64_t");
      return guard + store(args[0],
                           "weft_fp64::fma(" + operand_as(args[1], "uint64_t") + ", " +
                               operand_as(args[2], "uint64_t") + ", " +
                               operand_as(args[3], "uint64_t") + ")",
                           "uint64_t");
    }
    return guard + store(args[0],
                         operand_as(args[1], ty) + " * " + operand_as(args[2], ty) + " + " +
                             operand_as(args[3], ty),
                         ty);
  }

  if (op == "neg") {
    if (!need(2)) return "";
    if (msl && ty == "double")
      return guard + store(args[0], "weft_fp64::neg(" + operand_as(args[1], "uint64_t") + ")",
                           "uint64_t");
    return guard + store(args[0], "-" + operand_as(args[1], ty), ty);
  }

  if (op == "abs") {
    if (!need(2)) return "";
    if (msl && ty == "double")
      return guard + store(args[0],
                           "weft_fp64::abs_bits(" + operand_as(args[1], "uint64_t") + ")",
                           "uint64_t");
    return guard + store(args[0], math("abs", operand_as(args[1], ty)), ty);
  }

  if (op == "min" || op == "max") {
    if (!need(3)) return "";
    // minNum/maxNum on the software double: a NaN operand means the other one.
    if (msl && ty == "double")
      return guard + store(args[0],
                           std::string("weft_fp64::") + op + "_bits(" +
                               operand_as(args[1], "uint64_t") + ", " +
                               operand_as(args[2], "uint64_t") + ")",
                           "uint64_t");
    return guard + store(
        args[0], math(op, operand_as(args[1], ty) + ", " + operand_as(args[2], ty)), ty);
  }

  if (op == "sqrt") {
    if (!need(2)) return "";
    // The square root is the one function in that library with an exact
    // specification, so the software double answers it and the host's own `sqrt`
    // is the same number rather than an approximation of it.
    if (msl && ty == "double")
      return guard + store(args[0],
                           "weft_fp64::sqrt_bits(" + operand_as(args[1], "uint64_t") + ")",
                           "uint64_t");
    return guard + store(args[0], math("sqrt", operand_as(args[1], ty)), ty);
  }

  if (op == "not") {
    if (!need(2)) return "";
    // not.pred is logical, not.b32 is bitwise. Reading a predicate as any other
    // integer complements the wrong value and the result is always true.
    const bool pred = ty == "bool" || reg_type_of(args[1]) == "bool";
    if (pred) return guard + store(args[0], "!" + operand_as(args[1], "bool"), "bool");
    return guard + store(args[0], "~" + operand_as(args[1], ty), ty);
  }

  // ---- bit manipulation ----
  if (op == "bfe" && args.size() >= 4) {
    const bool sgn = ty == "int32_t";
    std::string sh = operand_as(args[2], "uint32_t");
    std::string w = operand_as(args[3], "uint32_t");
    std::string body = "((" + operand_as(args[1], ty) + " >> " + sh + ") & ((1u << " + w + ") - 1u))";
    if (sgn) body = "(int32_t)(" + body + ")";
    return guard + store(args[0], body, ty);
  }
  if (op == "bfi" && args.size() >= 4) {
    std::string mask = "((1u << " + operand_as(args[3], "uint32_t") + ") - 1u)";
    std::string sh = operand_as(args[3], "uint32_t");
    (void)sh;
    std::string ins = "((" + operand_as(args[1], "uint32_t") + " & " + mask + ") << " +
                      operand_as(args[3], "uint32_t") + ")";
    std::string keep = "(" + operand_as(args[2], "uint32_t") + " & ~(" + mask + " << " +
                       operand_as(args[3], "uint32_t") + "))";
    return guard + store(args[0], "(" + keep + " | " + ins + ")", "uint32_t");
  }
  // ---- integer dot products ----
  //
  // `dp4a.atype.btype d, a, b, c` takes four bytes from `a`, four from `b`, the
  // sign of each chosen by its own modifier, and accumulates the four products
  // into `c`. It is the inner instruction of an int8 GEMM and both Triton and
  // cuBLAS emit it, so a kernel carrying one was being refused outright.
  //
  // It is pure integer arithmetic, which is why one expression is exact on both
  // targets — the columns cannot drift apart the way a `mul` on floats can.
  //
  // `satfinite` saturates the result to [0, INT_MAX]. With byte inputs the largest
  // magnitude that can reach it is 4 * 255 * 255 = 260,100, so the clamp is
  // unreachable and is not emitted; the modifier is accepted so that a kernel
  // spelling it stays translatable rather than being refused for a word.
  if (op == "dp4a" && args.size() >= 4) {
    bool sgn_a = true, sgn_b = true, seen_a = false, seen_b = false;
    for (size_t i = 1; i < parts.size(); i++) {
      const std::string& p = parts[i];
      if (p == "u8" || p == "s8") {
        if (!seen_a) { sgn_a = (p == "s8"); seen_a = true; }
        else if (!seen_b) { sgn_b = (p == "s8"); seen_b = true; }
      } else if (p == "satfinite" || p == "b32" || p == "s32" || p == "u32") {
        continue;
      } else {
        err = "unsupported dp4a modifier '" + p + "'";
        return "";
      }
    }
    auto byte_at = [](const std::string& w, int i) {
      return "(((uint32_t)(" + w + ") >> " + std::to_string(8 * i) + ") & 0xFFu)";
    };
    auto extend = [](const std::string& b, bool sgn) {
      return sgn ? ("((int32_t)(int8_t)" + b + ")") : ("((int32_t)" + b + ")");
    };
    const std::string a = operand_as(args[1], "uint32_t");
    const std::string b = operand_as(args[2], "uint32_t");
    std::string sum = operand_as(args[3], "int32_t");
    for (int i = 0; i < 4; i++)
      sum += " + " + extend(byte_at(a, i), sgn_a) + " * " + extend(byte_at(b, i), sgn_b);
    return guard + store(args[0], "(" + sum + ")", ty.empty() ? "int32_t" : ty);
  }

  if (op == "clz" && args.size() >= 2)
    return guard + store(args[0], "(uint32_t)__builtin_clz(" + operand_as(args[1], ty) + ")", ty);
  if (op == "popc" && args.size() >= 2)
    return guard + store(args[0], "(uint32_t)__builtin_popcount(" + operand_as(args[1], ty) + ")", ty);
  if (op == "brev" && args.size() >= 2)
    return guard + store(args[0], "weft_bitrev32(" + operand_as(args[1], ty) + ")", ty);

  // ---- transcendental and approximate math ----
  if ((op == "ex2" || op == "lg2" || op == "rcp" || op == "rsqrt" || op == "div" || op == "sqrt") &&
      args.size() >= 2) {
    std::string a = operand_as(args[1], ty);
    std::string e;
    // The library, not the target's math. `sqrt.approx` on the GPU is bit-for-bit
    // `x * rsqrt(x)`, and its `ex2` and `lg2` are the unit's own polynomials, so
    // lowering these to the hardware made the two columns disagree in the last
    // bits (measured at 3.4e-07 relative over 65,536 lanes). `div` stays a
    // division, which is correctly rounded on both.
    if (op == "ex2") e = "weft_exp2(" + a + ")";
    else if (op == "lg2") e = "weft_log2(" + a + ")";
    else if (op == "rcp") e = "(1.0f / (" + a + "))";
    else if (op == "rsqrt") e = "weft_rsqrt(" + a + ")";
    else if (op == "sqrt") e = "weft_sqrt(" + a + ")";
    else e = a + " / " + (args.size() > 2 ? operand_as(args[2], ty) : "1.0f");
    return guard + store(args[0], e, ty);
  }
  // `mufu.<op>.approx.f32` is the multifunction unit's own approximation — the
  // only mode it has — and it is the same bargain as the `.approx` forms above:
  // the silicon's polynomial, which no CPU reproduces. So it runs the portable
  // library like everything else, and one expression is exact on both columns.
  if (op == "mufu" && args.size() >= 2) {
    std::string f;
    for (size_t i = 1; i < parts.size(); i++) {
      const std::string& p = parts[i];
      if (p == "rcp" || p == "rsqrt" || p == "sqrt" || p == "sin" || p == "cos" ||
          p == "ex2" || p == "lg2" || p == "tanh")
        f = p;
      else if (p == "approx" || p == "ftz" || p == "f32" || p == "rn")
        continue;
      else {
        err = "unsupported mufu operation '" + p + "'";
        return "";
      }
    }
    if (f.empty()) {
      err = "mufu without an operation";
      return "";
    }
    const std::string arg = operand_as(args[1], "float");
    std::string e;
    if (f == "rcp") e = "(1.0f / (" + arg + "))";
    else if (f == "ex2") e = "weft_exp2(" + arg + ")";
    else if (f == "lg2") e = "weft_log2(" + arg + ")";
    else e = "weft_" + f + "(" + arg + ")";
    return guard + store(args[0], e, ty.empty() ? "float" : ty);
  }

  if (op == "sin" && args.size() >= 2)
    return guard + store(args[0], "weft_sin(" + operand_as(args[1], ty) + ")", ty);
  if (op == "cos" && args.size() >= 2)
    return guard + store(args[0], "weft_cos(" + operand_as(args[1], ty) + ")", ty);
  if (op == "tanh" && args.size() >= 2)
    return guard + store(args[0], "weft_tanh(" + operand_as(args[1], ty) + ")", ty);

  // ---- atomics ----
  // ---- tensor cores ----
  //
  // ldmatrix and mma are warp-wide: a fragment is distributed across the 32
  // lanes, so one lane's operand is another lane's register. That is expressible
  // only where the whole warp's registers are addressable at once, which is the
  // warp backend; the per-lane state machine holds one lane's registers and has
  // no way to read a peer's without a rendezvous. A kernel that uses them is
  // therefore emitted warp-only, and the module's table records that.
  // ---- warp reductions ----
  //
  // `redux.sync.op d, a, mask` folds one value across the lanes of the warp: what
  // `__reduce_add_sync` compiles to, and what Triton emits for a `tl.sum`/`tl.max`
  // over an integer block. In the warp backend every lane's registers are
  // addressable at once, so the fold is a butterfly over the register array —
  // five steps, each combining a lane with the peer one bit away. An integer fold
  // has one right answer whatever order it is formed in, so there is nothing here
  // for an order to disagree about.
  if (op == "redux") {
    if (!vec && !msl) {
      warp_only = true;
      err = "redux.sync folds a value across the warp's lanes, so it needs the warp backend";
      return "";
    }
    if (!need(3)) return "";
    const std::string rop = parts.size() > 2 ? parts[2] : "";
    if (ty == "float" || ty == "double") {
      err = "redux.sync is an integer fold in PTX; a float reduction is the shfl form, "
            "which is supported";
      return "";
    }
    if (ty.empty()) ty = "uint32_t";
    std::string comb, ident;
    if (rop == "add") {
      comb = "(weft_a + weft_b)";
      ident = "0";
    } else if (rop == "min" || rop == "max") {
      comb = rop == "min" ? "(weft_a < weft_b ? weft_a : weft_b)"
                          : "(weft_a > weft_b ? weft_a : weft_b)";
      ident = lim_expr(ty, rop);
    } else if (rop == "and" || rop == "or" || rop == "xor") {
      comb = rop == "and" ? "(weft_a & weft_b)"
             : rop == "or" ? "(weft_a | weft_b)" : "(weft_a ^ weft_b)";
      ident = rop == "and" ? "~0u" : "0";
    } else {
      err = "unsupported redux op '" + rop + "'";
      return "";
    }
    // On the GPU the fold is one group instruction rather than a butterfly over
    // 32 sets of registers, so there is nothing for a per-lane state machine to
    // reach: the hardware's own reduction over the simdgroup is the answer, and
    // it is the same answer, because an integer fold has one whatever order it is
    // formed in. A lane outside the member mask stands in with the fold's
    // identity, so a partial mask means what it says here too.
    if (msl) {
      const std::string fn = rop == "add"   ? "simd_sum"
                             : rop == "min" ? "simd_min"
                             : rop == "max" ? "simd_max"
                             : rop == "and" ? "simd_and"
                             : rop == "or"  ? "simd_or"
                                            : "simd_xor";
      const std::string val = operand_as(args[1], ty);
      const std::string mask = trim(args[2]);
      const bool full = mask == "0xffffffff" || mask == "0xffffffffu" ||
                        mask == "0xFFFFFFFF" || mask == "0xFFFFFFFFu";
      const std::string e = full
                                ? (fn + "(" + val + ")")
                                : (fn + "((((" + operand_as(args[2], "uint32_t") +
                                   ") >> (ctx.lid & 31u)) & 1u) ? (" + val + ") : (" + ty + ")(" +
                                   ident + "))");
      return guard + store(args[0], e, ty);
    }
    // Lanes outside the member mask contribute the fold's identity: that is what
    // makes a partial mask mean what it says instead of folding in whatever the
    // inactive lanes happened to hold.
    const std::string mask = trim(args[2]);
    const bool full_mask = mask == "0xffffffff" || mask == "0xffffffffu" ||
                           mask == "0xFFFFFFFF" || mask == "0xFFFFFFFFu";
    std::string active = "am[l]";
    if (!full_mask)
      active += " && ((((uint32_t)" + operand_as(args[2], "uint32_t") + ") >> l) & 1u)";
    const std::string dst = reg_expr(args[0]), src = reg_expr(args[1]);
    vec_side = true;
    // Five butterflies, not one running sum: a stage is 32 independent combines,
    // and an accumulator is one dependency chain. Measured, in the README.
    std::string out = "{ " + ty + " weft_t[32]; ";
    out += "for (uint32_t l = 0; l < 32u; l++) weft_t[l] = (" + active + ") ? " + src + " : (" +
           ty + ")(" + ident + "); ";
    out += "for (int weft_o = 16; weft_o > 0; weft_o >>= 1) { " + ty + " weft_u[32]; ";
    out += "for (uint32_t l = 0; l < 32u; l++) { const " + ty + " weft_a = weft_t[l]; const " + ty +
           " weft_b = weft_t[l ^ (uint32_t)weft_o]; weft_u[l] = " + comb + "; } ";
    out += "for (uint32_t l = 0; l < 32u; l++) weft_t[l] = weft_u[l]; } ";
    out += "for (uint32_t l = 0; l < 32u; l++) if (am[l]) " + dst + " = weft_t[l]; ";
    out += "}";
    return out;
  }

  if (op == "ldmatrix") {
    if (!vec) {
      warp_only = true;
      err = "ldmatrix reads registers belonging to other lanes, so it needs the warp backend";
      return "";
    }
    if (!need(2)) return "";
    if (!has_part(parts, "m8n8")) {
      err = "unsupported ldmatrix shape (only 8x8 tiles of 16-bit elements)";
      return "";
    }
    int count = 1;
    for (const auto& p : parts)
      if (p.size() == 2 && p[0] == 'x' && p[1] >= '1' && p[1] <= '4') count = p[1] - '0';
    const bool trans = has_part(parts, "trans");
    std::string dsts = trim(args[0]);
    if (dsts.find('{') == std::string::npos) {
      err = "ldmatrix needs a register list";
      return "";
    }
    dsts = dsts.substr(dsts.find('{') + 1);
    dsts = dsts.substr(0, dsts.find('}'));
    const std::vector<std::string> regs = split_top(dsts, ',');
    if ((int)regs.size() != count) {
      err = "ldmatrix.x" + std::to_string(count) + " takes " + std::to_string(count) +
            " destination registers";
      return "";
    }
    std::string addr, sym;
    if (!mem_operand(args[1], addr, sym)) return "";
    // A named symbol is `[tile]` or `[tile+128]`: the address a lane would supply
    // is a place inside the shared window that is known at translation time, so
    // nothing has to be read out of a peer's register — the rows are the 16-byte
    // ones the tile was declared with.
    if (!sym.empty()) {
      std::string nm = sym;
      long long k = 0;
      const size_t cut = sym.find_first_of("+/-");
      if (cut != std::string::npos) {
        nm = trim(sym.substr(0, cut));
        k = std::strtoll(sym.c_str() + cut + 1, nullptr, 0);
        if (sym[cut] == '-') k = -k;
      }
      nm = trim(nm);
      auto so = shared_offsets.find(nm);
      if (so == shared_offsets.end()) {
        err = "'" + nm + "' is not a shared array: ldmatrix reads threadgroup memory";
        return "";
      }
      addr = std::to_string((long long)so->second + k);
    }
    // The address a lane supplies points at one row of 8 halves, and another
    // lane is the one that reads it: matrix k's row i is addressed by lane 8k+i,
    // and lane l wants row l>>2 of it. So the address is read out of the peer's
    // slot, not this lane's.
    // A register address names one row of the tile, and the peer is the lane that
    // holds it, so the row is read out of that slot. A named symbol names the whole
    // tile instead, so the row is an offset from its start — 16 bytes per row.
    const bool uniform = !sym.empty();
    auto row_addr = [&](const std::string& who) -> std::string {
      if (uniform) return "(" + addr + "ull + 16u * (" + who + "))";
      std::string s = addr;
      for (size_t p = 0; (p = s.find("[l]", p)) != std::string::npos;)
        s.replace(p, 3, "[" + who + "]");
      return s;
    };
    vec_side = true;
    std::string out = "{ const uint32_t weft_r = (l >> 2) & 7u; const uint32_t weft_c = (l & 3u); ";
    for (int k = 0; k < count; k++) {
      const std::string m = std::to_string(k);
      if (trans) {
        // .trans hands back the transpose: element (r, c) of the result is
        // element (c, r) of the tile, so the two halves sit in two different
        // rows, at the column this lane owns.
        const std::string rowlo = "8u * " + m + "u + 2u * weft_c";
        const std::string rowhi = "8u * " + m + "u + 2u * weft_c + 1u";
        out += "const uint16_t weft_lo" + m + " = *(" + addr_space("threadgroup", true) +
               "uint16_t*)(sh + " + row_addr(rowlo) + " + 2u * weft_r); ";
        out += "const uint16_t weft_hi" + m + " = *(" + addr_space("threadgroup", true) +
               "uint16_t*)(sh + " + row_addr(rowhi) + " + 2u * weft_r); ";
        out += store(regs[k],
                     "((uint32_t)weft_lo" + m + " | ((uint32_t)weft_hi" + m + " << 16))",
                     "uint32_t");
      } else {
        out += store(regs[k],
                     "*(" + addr_space("threadgroup", true) +
                     "uint32_t*)(sh + " + row_addr("8u * " + m + "u + weft_r") +
                         " + 4u * weft_c)",
                     "uint32_t");
      }
      out += " ";
    }
    out += "}";
    // The block names the lane index, so it carries its own lane loop: there is
    // nothing here that can be batched with the surrounding arithmetic, because
    // each lane reads a different lane's registers.
    return "for (uint32_t l = 0; l < 32u; l++) { " + out + " }";
  }

  // ---- wgmma: the warpgroup-level tensor-core instruction ----
  //
  // One instruction is issued by a warpgroup — 128 lanes — and reads *both*
  // operands out of shared memory through a 64-bit descriptor. That last part is
  // what makes it translatable here at all: unlike `mma`, no lane ever needs
  // another lane's register, so one lane's worth of work is enough to compute one
  // lane's accumulators. The per-lane backend runs a thread per threadgroup lane,
  // so `ctx.lid` is the warpgroup lane, 0 to 127. The 32-lane backend runs a
  // thread per *warp*, so its lanes are `ctx.lid0 + l` and the same body runs
  // once per lane there under the warp's active mask.
  //
  // The descriptor (PTX ISA "Matrix Descriptor") packs five fields: the shared
  // memory start address in bits 0-13, the leading byte offset (between core
  // matrices along K) in 16-29, the stride byte offset (along M or N) in 32-45, a
  // base offset for alignment in 49-51, and the swizzle mode in 62-63, all of the
  // offsets in 16-byte units. A core matrix is 8 rows by 16 bytes, so element
  // (i, k) of an unswizzled K-major operand is at
  // start + (i/8)*SBO + (k/8)*LBO + (i%8)*16 + (k%8)*element_size.
  if (op == "wgmma") {
    if (parts.size() >= 2 && (parts[1] == "fence" || parts[1] == "commit_group" ||
                              parts[1] == "wait_group")) {
      // Group bookkeeping for the async proxy: on a GPU these order the wgmma
      // against later reads of the accumulator. Here every element is computed
      // once, in program order, so there is nothing left to wait for.
      return guard + ";";
    }
    if (parts.size() < 2 || parts[1] != "mma_async") {
      err = "unsupported wgmma instruction '" + opcode + "'";
      return "";
    }
    // One layout rule, four element widths. A core-matrix row is 16 bytes, so it
    // holds 16/elements values and the instruction's K is 32/elements: 32 for the
    // 8-bit types (fp8 and the integers), 16 for f16 and bf16, 8 for tf32. The
    // descriptor fields, the row walk and the accumulator pattern do not depend on
    // which, so this is a few derived numbers and one decode rather than a decoder
    // per format.
    const char* fp8dec = has_part(parts, "e5m2") ? "weft_e5m2("
                         : has_part(parts, "e4m3") ? "weft_e4m3(" : nullptr;
    // An 8-bit integer operand is the same one-byte read as an fp8 one: only the
    // reading of the byte changes, which is why signedness is a decode and not a
    // second layout.
    // An accumulator is f32 for every float operand and s32 for the 8-bit
    // integers, which is the pair the instruction set offers; the integer form
    // also decodes its bytes as integers, because a sum that leaves 2^24 is no
    // longer the same number in a float.
    const bool iacc = has_part(parts, "s32");
    const char* i8dec = has_part(parts, "s8") ? (iacc ? "weft_s8_i(" : "weft_s8(")
                        : has_part(parts, "u8") ? (iacc ? "weft_u8_i(" : "weft_u8(")
                                                 : nullptr;
    const char* bytedec = fp8dec ? fp8dec : i8dec;
    const std::string dty = iacc ? "int32_t" : "float";
    const bool bw = has_part(parts, "bf16");
    const bool tf = has_part(parts, "tf32");
    if (parts.size() < 5) {
      err = "unsupported wgmma operands in '" + opcode +
            "': f16, bf16, tf32, fp8 (e4m3/e5m2) and 8-bit integer operands with "
            "an f32 accumulator are translated";
      return "";
    }
    if (iacc && !i8dec) {
      err = "unsupported wgmma accumulator in '" + opcode +
            "': an s32 accumulator is the 8-bit integer form (s8 or u8 operands); "
            "f16, bf16, tf32 and fp8 accumulate in f32";
      return "";
    }
    const int esz = bytedec ? 1 : tf ? 4 : 2;  // bytes of one operand element
    const int ktot = 32 / esz;                 // the instruction's K
    int ncols = 0;
    for (const auto& p : parts) {
      const size_t np = p.find('n'), kp = p.find('k');
      if (p.empty() || p[0] != 'm' || np == std::string::npos || kp == std::string::npos)
        continue;
      const std::string ms = p.substr(1, np - 1), ns = p.substr(np + 1, kp - np - 1),
                        ks = p.substr(kp + 1);
      if (ms != "64" || ks != std::to_string(ktot)) {
        err = "unsupported wgmma shape '" + p + "': with these operands the tile "
              "shape must be m64nNk" + std::to_string(ktot) + " (f16/bf16 m64nNk16, "
              "tf32 m64nNk8, fp8 and integer m64nNk32)";
        return "";
      }
      ncols = std::atoi(ns.c_str());
      break;
    }
    if (ncols <= 0 || ncols > 256 || ncols % 8 != 0) {
      err = "unsupported wgmma shape in '" + opcode + "'";
      return "";
    }
    if (args.size() < 4) { err = "bad " + opcode; return ""; }
    // After the operands the immediates run in a fixed order: scale-d, scale-a,
    // scale-b, transpose-a, transpose-b. PTX says the trailing ones may be left
    // off, so a run of one to five is legal and whatever is missing is zero — a
    // shorter list is not a different instruction, and refusing it refused
    // assembly GCC and Triton both emit.
    const size_t nargs = args.size();
    if (nargs > 8) {
      err = "wgmma '" + opcode + "' takes at most 5 immediates (scale-d, then "
            "scale-a scale-b, then transpose-a transpose-b), got " +
            std::to_string(nargs - 3);
      return "";
    }
    bool flag[5] = {false, false, false, false, false};
    static const char* const kImmName[5] = {"scale-d", "scale-a", "scale-b",
                                            "transpose-a", "transpose-b"};
    for (size_t i = 0; i < 5 && 3 + i < nargs; i++) {
      const std::string v = trim(args[3 + i]);
      if (v != "0" && v != "1") {
        err = "wgmma '" + opcode + "': the " + kImmName[i] +
              " immediate must be 0 or 1, got '" + v + "'";
        return "";
      }
      flag[i] = (v == "1");
    }
    const bool scale_d = flag[0], neg_a = flag[1], neg_b = flag[2];
    const bool trans_a = flag[3], trans_b = flag[4];
    std::string dlist = trim(args[0]);
    if (dlist.find('{') == std::string::npos) {
      err = "wgmma needs an accumulator register list";
      return "";
    }
    dlist = dlist.substr(dlist.find('{') + 1);
    dlist = dlist.substr(0, dlist.find('}'));
    const std::vector<std::string> dregs = split_top(dlist, ',');
    const int groups = ncols / 8;
    if ((int)dregs.size() != 4 * groups) {
      err = "wgmma m64n" + std::to_string(ncols) + "k" + std::to_string(ktot) + " takes " +
            std::to_string(4 * groups) + " accumulator registers, got " +
            std::to_string(dregs.size());
      return "";
    }
    const std::string da = operand_as(args[1], "uint64_t");
    const std::string db = operand_as(args[2], "uint64_t");
    auto elem_at = [&](const std::string& off) {
      if (bytedec) {
        // One byte, and the decode is the format's own: e4m3 saturates at 448 and
        // has no infinities, e5m2 has them, an s8 sign-extends and a u8 does not —
        // the helpers `mma` reads its operands with, so the paths cannot drift
        // apart.
        return std::string(bytedec) + "(uint32_t)*(" + addr_space("threadgroup", true) +
               "uint8_t*)(sh + (" + off + ")), 0)";
      }
      if (tf) {
        // tf32 is a float whose low 13 mantissa bits the instruction ignores, so
        // the read is four bytes and the decode is the same truncation `mma` uses.
        const std::string raw = "(uint32_t)*(" + addr_space("threadgroup", true) +
                                "uint32_t*)(sh + (" + off + "))";
        return "weft_tf32(weft_f32(" + raw + "))";
      }
      const std::string raw = "(uint32_t)*(" + addr_space("threadgroup", true) +
                              "uint16_t*)(sh + (" + off + "))";
      return (bw ? "weft_bf16(" : "weft_h16(") + raw + ", 0)";
    };
    std::string out =
        "const uint64_t weft_da = " + da + ", weft_db = " + db + "; "
        "const uint32_t weft_sma = (uint32_t)(weft_da >> 62), "
        "weft_smb = (uint32_t)(weft_db >> 62); "
        // Swizzle: mode 1, 2 or 3 is an atom 128, 64 or 32 bytes wide, and over
        // byte offsets that is CUTLASS's Swizzle<3,4,3>, <2,4,3> or <1,4,3> —
        // `offset ^ (((offset >> 7) & mask) << 4)`. No swizzle (mode 0) is the
        // 16-byte core matrix, which is the same rule with the shift at 4 and an
        // empty mask, so one walk covers all four and nothing here has to be
        // refused. The width is a shift held in a variable rather than a
        // constant because the mode is a field of the descriptor, which is data.
        "const uint32_t weft_as = weft_sma ? 8u - weft_sma : 4u, "
        "weft_bs = weft_smb ? 8u - weft_smb : 4u; "
        "const uint32_t weft_aw = 1u << weft_as, weft_bw = 1u << weft_bs; "
        // Two different masks, one for where a byte sits inside an atom (all of
        // its width) and one for the XOR (the 16-byte chunks, one per 16 bytes of
        // that width). Using the second for the first drops every byte past the
        // eighth, which reads the wrong half rather than a nearby one.
        "const uint32_t weft_ak = weft_aw - 1u, weft_bk = weft_bw - 1u; "
        "const uint32_t weft_am = (weft_aw >> 4u) - 1u, weft_bm = (weft_bw >> 4u) - 1u; "
        // Where the pattern starts, when the tile does not begin on one: bits 49-51
        // hold the distance from the pattern origin to the descriptor's start in
        // 128-byte units, and it shifts the phase of the XOR without moving any
        // address. A tile that does begin on a pattern boundary carries zero here.
        "const uint32_t weft_ao = (uint32_t)((weft_da >> 49u) & 7u) << 7u, "
        "weft_bo = (uint32_t)((weft_db >> 49u) & 7u) << 7u; "
        "const uint64_t weft_ba = (weft_da & 0x3FFFu) << 4, "
        "weft_bb = (weft_db & 0x3FFFu) << 4; "
        "const uint64_t weft_la = ((weft_da >> 16) & 0x3FFFu) << 4, "
        "weft_sa = ((weft_da >> 32) & 0x3FFFu) << 4; "
        "const uint64_t weft_lb = ((weft_db >> 16) & 0x3FFFu) << 4, "
        "weft_sb = ((weft_db >> 32) & 0x3FFFu) << 4; "
        "";
    std::string body;
    const bool weft_sel = vec_select;
    vec_select = false;
    // The swizzle, applied to an offset measured from the descriptor's start. An
    // atom is a run of eight rows and the pattern restarts every atom, so the
    // strides the descriptor carries (both whole atoms) leave the phase alone and
    // the XOR is the only thing that moves an address.
    auto swz = [&](const std::string& d, const char* sfx) {
      const std::string x = "(((" + d + ") + weft_" + sfx + "o) >> 7u)";
      return "((" + d + ") ^ ((" + x + " & weft_" + sfx + "m) << 4u))";
    };
    // A: row 16w+g (and 16w+g+8), column k. B: row k, column 8q+2t+c.
    //
    // Majorness does not move the descriptor's two offsets — SBO steps between MN
    // groups and LBO between K groups either way. It decides which direction is
    // contiguous inside an atom: K-major puts K in it and the eight rows an atom
    // width apart, MN-major is the mirror image of that.
    auto aval = [&](int rowhalf, int k) {
      const std::string rowg = rowhalf ? "(2u * weft_w + 1u)" : "(2u * weft_w)";
      const std::string rowi = rowhalf ? "(16u * weft_w + weft_g + 8u)"
                                       : "(16u * weft_w + weft_g)";
      // The byte column of this k inside a row. With no swizzle that is a 16-byte
      // core matrix and LBO steps between them; a wider atom holds the K of a
      // whole instruction, so the step lands on zero and the same atom strides it.
      const std::string kb = "(" + std::to_string(esz * k) + "u >> weft_as)";
      const std::string ko = "(" + std::to_string(esz * k) + "u & weft_ak)";
      std::string d;
      if (trans_a) {
        // Along M, so an atom spans the atom width of M and eight rows of K: the
        // K row is one atom wide and LBO steps eight of them.
        const std::string mb = "((" + rowi + ") * " + std::to_string(esz) + "u)";
        d = "(" + mb + " >> weft_as) * weft_sa + " + std::to_string(k / 8) +
            "u * weft_la + (" + mb + " & weft_ak) + " + std::to_string(k % 8) +
            "u * weft_aw";
      } else {
        d = rowg + " * weft_sa + " + kb + " * weft_la + weft_g * weft_aw + " + ko;
      }
      const std::string v = elem_at("weft_ba + " + swz(d, "a"));
      return neg_a ? ((iacc ? "(0 - " : "(0.0f - ") + v + ")") : v;
    };
    // B is K rows by N columns. The two descriptor offsets do not change with
    // majorness — SBO steps between MN groups and LBO between K groups either way —
    // but which direction lives contiguously inside an atom does. In a K-major
    // operand K is the contiguous one: an atom holds values of K, and the eight
    // values of N are an atom width apart. That is exactly the mirror of A's
    // K-major walk and of what this used to do, which put K 16 bytes apart and N
    // contiguous — the MN-major arrangement, under an instruction that says K-major.
    // Every fixture gave B an LBO and an SBO of the same value, so the two readings
    // produced identical addresses and no test could tell them apart.
    auto bval = [&](int q, int c, int k) {
      const std::string n = "(8u * " + std::to_string(q) + "u + 2u * weft_t + " +
                            std::to_string(c) + "u)";
      const std::string kb = "(" + std::to_string(esz * k) + "u >> weft_bs)";
      const std::string ko = "(" + std::to_string(esz * k) + "u & weft_bk)";
      std::string d;
      if (trans_b) {
        const std::string nb = "((" + n + ") * " + std::to_string(esz) + "u)";
        d = "(" + nb + " >> weft_bs) * weft_sb + " + std::to_string(k / 8) +
            "u * weft_lb + (" + nb + " & weft_bk) + " + std::to_string(k % 8) +
            "u * weft_bw";
      } else {
        d = "(" + n + ") / 8u * weft_sb + " + kb + " * weft_lb + (" + n +
            " % 8u) * weft_bw + " + ko;
      }
      const std::string v = elem_at("weft_bb + " + swz(d, "b"));
      return neg_b ? ((iacc ? "(0 - " : "(0.0f - ") + v + ")") : v;
    };
    for (int q = 0; q < groups; q++) {
      for (int j = 0; j < 4; j++) {
        const int rowhalf = j / 2, colhalf = j % 2;
        std::string sum;
        for (int k = 0; k < ktot; k++) {
          if (k) sum += " + ";
          sum += aval(rowhalf, k) + " * " + bval(q, colhalf, k);
        }
        // scale-d = 1 is D = A*B + D, and the register is read before it is
        // written. With scale-d = 0 the accumulator's old value is ignored, which is
        // what the instruction says and what a first issue into a tile wants.
        const std::string acc =
            scale_d ? ("(" + operand_as(dregs[(size_t)(4 * q + j)], dty) + ") + (" +
                       sum + ")")
                    : sum;
        body += store(dregs[(size_t)(4 * q + j)], acc, dty);
        body += " ";
      }
    }
    vec_select = weft_sel;
    // The lane-dependent half. On the per-lane backend the thread *is* one lane.
    // On the 32-lane backend it is one warp of the warpgroup, so its lanes are
    // `ctx.lid0 + l` and each accumulator is that lane's element of the vector —
    // which works only because wgmma reads both operands out of shared memory and
    // never another lane's register.
    const std::string ldecl =
        "const uint32_t weft_l = " +
        std::string(vec ? "(ctx.lid0 + l)" : "ctx.lid") +
        ", weft_w = weft_l >> 5u, weft_g = (weft_l >> 2u) & 7u, weft_t = weft_l & 3u; ";
    // Everything goes inside the lane loop on the vector backend: the descriptor
    // fields are read out of per-lane registers there, so they cannot be hoisted.
    const std::string inner = out + ldecl + body;
    if (!vec) return "{ " + inner + "}";
    return "{ " + lane_block(inner) + "}";
  }

  if (op == "mma") {
    if (!vec) {
      warp_only = true;
      err = "mma reads registers belonging to other lanes, so it needs the warp backend";
      return "";
    }
    // The four shapes Triton emits for `tl.dot`. Operand width decides the
    // fragment layout: one element per register for tf32, two for a 16-bit type,
    // four for an 8-bit one, and the K dimension scales with it.
    const bool bf16 = has_part(parts, "bf16");
    const bool e4m3 = has_part(parts, "e4m3");
    const bool e5m2 = has_part(parts, "e5m2");
    const bool k16 = has_part(parts, "m16n8k16") && (has_part(parts, "f16") || bf16);
    const bool k8 = has_part(parts, "m16n8k8") && has_part(parts, "tf32");
    const bool k32 = has_part(parts, "m16n8k32") && (e4m3 || e5m2);
    if (!k16 && !k8 && !k32) {
      err = "unsupported mma '" + opcode + "': m16n8k16 with f16/bf16, m16n8k8 with "
            "tf32, and m16n8k32 with e4m3/e5m2 are the forms translated";
      return "";
    }
    if (!has_part(parts, "row") || !has_part(parts, "col")) {
      err = "unsupported mma layout in '" + opcode + "' (Triton emits row.col)";
      return "";
    }
    auto regs_of = [&](const std::string& arg, int want) -> std::vector<std::string> {
      std::string s = trim(arg);
      if (s.find('{') == std::string::npos) return {};
      s = s.substr(s.find('{') + 1);
      s = s.substr(0, s.find('}'));
      std::vector<std::string> v = split_top(s, ',');
      if ((int)v.size() != want) v.clear();
      return v;
    };
    std::vector<std::string> d = regs_of(args[0], 4), a = regs_of(args[1], 4),
                             b = regs_of(args[2], 2);
    if (d.empty() || a.empty() || b.empty()) {
      err = "mma takes {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3}";
      return "";
    }
    const std::vector<std::string> c = args.size() > 3 ? regs_of(args[3], 4) : std::vector<std::string>();

    const int K = k32 ? 32 : (k16 ? 16 : 8);
    const int per = k32 ? 4 : (k16 ? 2 : 1);  // elements packed in one register
    const bool halves = k16;
    auto own = [&](const std::string& tok, const std::string& lane) {
      Reg r = parse_reg(tok);
      if (!r.ok) { err = "expected a register, got '" + tok + "'"; return std::string("0.0f"); }
      const std::string v = "s->" + r.base + "[" + std::to_string(r.index) + "][" + lane + "]";
      auto it = reg_types.find(r.base);
      return (it != reg_types.end() && it->second == "float") ? v : ("weft_f32(" + v + ")");
    };
    // One element of an operand register, as a float. A fragment register holds
    // two halves when the operands are 16-bit, and the operand names which; a
    // tf32 operand is a 32-bit value the instruction reduces to 10 mantissa bits,
    // which this host does the same way rather than borrowing the extra
    // precision.
    auto elem = [&](const std::string& tok, const std::string& lane, int which) {
      if (k32 || halves) {
        Reg r = parse_reg(tok);
        if (!r.ok) { err = "expected a register, got '" + tok + "'"; return std::string("0.0f"); }
        const std::string v = "s->" + r.base + "[" + std::to_string(r.index) + "][" + lane + "]";
        auto it = reg_types.find(r.base);
        const std::string bits = (it != reg_types.end() && it->second == "float") ?
                                     ("weft_bits(" + v + ")") : v;
        const char* fn = k32 ? (e5m2 ? "weft_e5m2(" : "weft_e4m3(")
                     : bf16 ? "weft_bf16("
                            : "weft_h16(";
        return std::string(fn) + bits + ", " + std::to_string(which) + ")";
      }
      return "weft_tf32(" + own(tok, lane) + ")";
    };
    // A is 16 rows by K: the row group this lane is in, and the group 8 above it.
    // Within a register the elements run along K, `per` of them at a time, and
    // the half of K from K/2 up lives in the third and fourth register.
    auto aval = [&](const std::string& wg, int rowhalf, int k) {
      const int reg = rowhalf + (k >= K / 2 ? 2 : 0);
      const int within = (k & (K / 2 - 1)) / per;
      return elem(a[reg], "4u * " + wg + " + " + std::to_string(within), k & (per - 1));
    };
    // B is K by 8, in the column layout `row.col` names: lane 4n+q holds B[q][n].
    auto bval = [&](int colhalf, int k) {
      const int reg = k >= K / 2 ? 1 : 0;
      const int within = (k & (K / 2 - 1)) / per;
      return elem(b[reg], "8u * weft_t + " + std::to_string(4 * colhalf + within),
                  k & (per - 1));
    };
    vec_side = true;
    std::string out = "{ const uint32_t weft_g = (l >> 2) & 7u; const uint32_t weft_t = (l & 3u); ";
    // D[r][n] = sum_k A[r][k] * B[k][n] + C[r][n], and this lane holds
    // D[g][2t], D[g][2t+1], D[g+8][2t] and D[g+8][2t+1].
    for (int q = 0; q < 4; q++) {
      const int rowhalf = q / 2, colhalf = q % 2;
      const std::string wg = rowhalf ? std::string("weft_g") : std::string("weft_g");
      std::string sum = c.empty() ? std::string("0.0f") : own(c[(size_t)q], "l");
      for (int k = 0; k < K; k++)
        sum += " + " + aval(wg, rowhalf, k) + " * " + bval(colhalf, k);
      out += store(d[(size_t)q], sum, "float");
      out += " ";
    }
    out += "}";
    return "for (uint32_t l = 0; l < 32u; l++) { " + out + " }";
  }

  if (op == "atom" || op == "red") {
    // `atom` names its result, `red` does not — so the operands sit at different
    // positions in the two spellings, and the *instruction as written* decides
    // where they are. Whether the result is kept is a separate question, below.
    const bool is_atom = (op == "atom");
    bool has_result = is_atom;
    // An atomic whose result nobody reads is a reduction, and ptxas emits the
    // reduction itself; this front end has to make the same rewrite because what
    // it is handed is already codegen output. It matters most at 64 bits, where
    // the `red` forms exist and the returning ones do not.
    if (is_atom && !args.empty()) {
      const std::string d = trim(args[0]);
      if (!d.empty() && d[0] == '%' && d.find('{') == std::string::npos) {
        auto it = reg_reads.find(d.substr(1));
        if (it != reg_reads.end() && it->second <= 1) has_result = false;
      }
    }
    const size_t ai = is_atom ? 1 : 0;
    const size_t vi = is_atom ? 2 : 1;
    if (args.size() < vi + 1) { err = "bad " + opcode; return ""; }
    std::string addr, name, base;
    if (!mem_operand(args[ai], addr, name, &base)) return "";
    note_width(name, base, ty);
    // The operation is one of several modifiers: real codegen writes
    // `atom.acquire.sys.global.max.s64`, so the scope and the semantic
    // qualifiers are skipped rather than assumed away.
    std::string aop;
    for (size_t i = 1; i < parts.size(); i++) {
      const std::string& p = parts[i];
      if (p == "add" || p == "sub" || p == "min" || p == "max" || p == "and" ||
          p == "or" || p == "xor" || p == "exch" || p == "cas") {
        aop = p;
        break;
      }
    }
    if (aop.empty()) { err = "unsupported atomic '" + opcode + "'"; return ""; }
    std::string fn;
    if (ty == "float") {
      if (aop == "add") fn = "atom_add_f32";
      else if (aop == "min") fn = "atom_min_f32";
      else if (aop == "max") fn = "atom_max_f32";
    } else if (ty == "uint64_t" || ty == "int64_t") {
      // 64-bit counters: `atom.global.max.s64` on a size or a histogram bound.
      const bool s = (ty == "int64_t");
      if (aop == "add") fn = s ? "atom_add_s64" : "atom_add_u64";
      else if (aop == "sub") fn = "atom_sub_u64";
      else if (aop == "min") fn = s ? "atom_min_s64" : "atom_min_u64";
      else if (aop == "max") fn = s ? "atom_max_s64" : "atom_max_u64";
      else if (aop == "and") fn = "atom_and_u64";
      else if (aop == "or") fn = "atom_or_u64";
      else if (aop == "xor") fn = "atom_xor_u64";
      else if (aop == "exch") fn = "atom_exch_u64";
      else if (aop == "cas") fn = "atom_cas_u64";
    } else {
      const bool s = (ty == "int32_t");
      if (aop == "add") fn = s ? "atom_add_s32" : "atom_add_u32";
      else if (aop == "sub") fn = "atom_sub_u32";
      else if (aop == "min") fn = s ? "atom_min_s32" : "atom_min_u32";
      else if (aop == "max") fn = s ? "atom_max_s32" : "atom_max_u32";
      else if (aop == "and") fn = "atom_and_u32";
      else if (aop == "or") fn = "atom_or_u32";
      else if (aop == "xor") fn = "atom_xor_u32";
      else if (aop == "exch") fn = "atom_exch_u32";
      else if (aop == "cas") fn = "atom_cas_u32";
    }
    if (fn.empty()) { err = "unsupported atomic '" + opcode + "'"; return ""; }
    if (msl && cpp_size(ty) == 8) {
      // A `red` has no result, so an order-free operation does not need the 64
      // bits read as one value: `src/weft_msl.hpp` folds each half with a 32-bit
      // atomic, and for add/sub carries the low half's carry into the high one,
      // which sums correctly whatever order the threads land in.
      const bool order_free = !has_result && (aop == "add" || aop == "sub" || aop == "and" ||
                                              aop == "or" || aop == "xor");
      if (order_free)
        return guard + "(void)weft::red_" + aop + "_u64((" + addr_space("device", false) + ty +
               "*)(uintptr_t)(" + addr + "), " + operand_as(args[vi], ty) + ");";
      // Everything else has to have the pair read and written as one value, and
      // this host has no instruction for that. `atomic_ulong` is declared in the
      // header, but every entry point that could read, write, exchange or
      // compare one is constrained to 32-bit types, and the two that are not
      // compile only to be refused by the device at pipeline creation
      // (tools/atom64_probe.mm asks for each answer separately). So the helper is
      // a lock over the two halves, built from the 32-bit atomics the hardware
      // does have, and `wide_atom` marks the kernel as needing that runtime — the
      // warp back end, at the same width as any other launch.
      wide_atom = true;
      const std::string p =
          "((" + addr_space("device", false) + ty + "*)(uintptr_t)(" + addr + "))";
      // In the warp back end one thread runs all 32 lanes, so an address that is
      // the same for every lane is one word to lock rather than 32 trips to the
      // coherent point: the lock goes around the loop instead of inside it. A
      // lane-dependent address cannot be hoisted — each lane's word is its own —
      // and keeps the per-lane form below. Hoisting every address onto one global
      // lock instead was tried and measured slower (19 ms against 10 ms at 4,096
      // lanes, 660 ms against 150 ms at 262,144): serialising threads that were
      // working on different words costs more than the lock trips saved.
      if (vec && p.find("[l]") == std::string::npos) {
        std::string raw = "weft::rmw_" + aop + "_raw(" + p + ", " + operand_as(args[vi], ty);
        if (aop == "cas" && args.size() > vi + 1) raw += ", " + operand_as(args[vi + 1], ty);
        raw += ")";
        const bool sel = vec_select;
        vec_select = false;  // the loop below masks by lane; the store must not
        const std::string inner = has_result ? store(args[0], raw, ty) : ("(void)" + raw + ";");
        vec_select = sel;
        vec_side = true;
        return "{ device uint32_t* weft_lk = weft::lock_of(ctx.locks, (device const void*)" +
               p + ");\n"
               "  weft::lock_take(weft_lk);\n"
               "  for (uint32_t l = 0; l < 32u; l++) { if (" +
               lane_cond() + ") { " + inner + " } }\n"
               "  weft::lock_drop(weft_lk); }";
      }
      std::string call = "weft::rmw_" + aop + "(ctx.locks, " + p + ", " + operand_as(args[vi], ty);
      if (aop == "cas" && args.size() > vi + 1) call += ", " + operand_as(args[vi + 1], ty);
      call += ")";
      if (has_result) return guard + store(args[0], call, ty);
      return guard + "(void)" + call + ";";
    }
    std::string call = "weft::" + fn + "((" + addr_space("device", false) + ty +
                       "*)(uintptr_t)(" + addr + "), " +
                       operand_as(args[vi], ty);
    if (aop == "cas" && args.size() > vi + 1) call += ", " + operand_as(args[vi + 1], ty);
    call += ")";
    if (has_result) return guard + store(args[0], call, ty);
    return guard + "(void)" + call + ";";
  }

  // cp.async.ca.shared.global [dst], [src], bytes
  //
  // The GPU's asynchronous copy has no CPU counterpart that is worth having: a
  // group of copies is already a handful of loads and stores, and a thread that
  // has to wait for them gains nothing by waiting later. So the copy is issued
  // as a memcpy and the commit/wait instructions are no-ops — which is exactly
  // the ordering the GPU promises once the wait has been reached.
  if (op == "cp" && parts.size() >= 2 && parts[1] == "async") {
    for (size_t i = 2; i < parts.size(); i++) {
      const std::string& p = parts[i];
      if (p == "commit_group" || p == "wait_group" || p == "wait_all") return guard + ";";
    }
    bool to_shared = false;
    for (const auto& p : parts) if (p == "shared") to_shared = true;
    if (!to_shared || args.size() < 3) {
      err = "unsupported asynchronous copy '" + opcode + "'";
      return "";
    }
    std::string dst, dname, src, sname;
    if (!mem_operand(args[0], dst, dname)) return "";
    if (!mem_operand(args[1], src, sname)) return "";
    std::string bytes = trim(args[2]);
    if (bytes.empty() || bytes.find_first_not_of("0123456789") != std::string::npos) {
      err = "cp.async size must be a constant";
      return "";
    }
    if (msl)
      return guard + "weft::copy_shared(sh + " + dst + ", (const device void*)(uintptr_t)(" +
             src + "), " + bytes + ");";
    return guard + "std::memcpy((void*)(sh + " + dst + "), (const void*)(uintptr_t)(" + src +
           "), " + bytes + ");";
  }

  // ---- textures and surfaces ----
  //
  // `tex.2d.v4.f32.f32 {d0,d1,d2,d3}, [t, {u,v}]` samples the binding a `.tex`
  // declaration names, `txq` asks that binding for its size, and `suld`/`sust`
  // read and write a texel exactly, which is what a surface is for. The warp
  // backend has no form for any of them — a texture read is not register
  // arithmetic — so a kernel that uses one stays on the per-lane machine.
  if (op == "tex" || op == "tld4" || op == "txq" || op == "suld" || op == "sust") {
    vec_ok = false;
    if (args.empty()) { err = "bad " + opcode; return ""; }
    // PTX's cube form is `(s, t, face)` — a face number and a position on it —
    // where the sampler here takes the direction that decomposes into exactly
    // those three. The two are the same information, so `weft_cube_dir` rebuilds
    // the direction that lands back on that face at that place; both engines run
    // that text, so both build the same bits and the sample agrees.
    const bool cube = opcode.find("cube") != std::string::npos;
    const bool d3 = !cube && opcode.find("3d") != std::string::npos;
    // A 1-D texture is a 2-D one with a single row: the binding is `Wx1` and the
    // sample sits at 0.5, the centre of that row, which is the texel itself under
    // any filter and any address mode. So the lowering is exact rather than
    // approximate, and `txq` still answers with what was actually bound.
    const bool d1 = !cube && !d3 && opcode.find("1d") != std::string::npos;
    if (op != "txq" && !cube && opcode.find("2d") == std::string::npos && !d3 && !d1) {
      err = "only a 1-D, 2-D or 3-D texture is bound here, and '" + opcode + "' is not one";
      return "";
    }
    // A gather is a 2x2 footprint, and every form of the binding has one: a volume
    // gathers in the plane its third coordinate names, and a 1-D binding is the
    // `Wx1` plane its sample already reads, so its footprint is the two texels along
    // x and the second row is outside the image — zero, on both engines.
    // `tld4` is a gather rather than a filter: the same footprint, each texel on
    // its own channel. Both targets have the helper, so the two runs compare the
    // same four reads.

    // The binding an operand names, plus whatever `{...}` list follows it.
    auto binding = [&](const std::string& tok, std::vector<std::string>& list) -> int {
      const std::string m = trim(tok);
      const size_t lb = m.find('['), rb = m.rfind(']');
      if (lb == std::string::npos || rb == std::string::npos || rb < lb) {
        err = "bad texture operand in '" + opcode + "'";
        return -1;
      }
      std::vector<std::string> parts = split_top(m.substr(lb + 1, rb - lb - 1), ',');
      if (parts.empty()) { err = "bad texture operand in '" + opcode + "'"; return -1; }
      const std::string nm = trim(parts[0]);
      auto it = tex_index.find(nm);
      if (it == tex_index.end()) {
        // The bindings a module declares, named: a misspelled reference is worth
        // an answer, not a count.
        std::string have;
        for (const auto& kv : tex_index) have += (have.empty() ? "" : ", ") + kv.first;
        err = "'" + nm + "' is not a texture: the module declares " +
              (have.empty() ? std::string("none") : have) + " (with .tex or .surf)";
        return -1;
      }
      for (size_t i = 1; i < parts.size(); i++) {
        std::string c = trim(parts[i]);
        if (!c.empty() && c[0] == '{') c = c.substr(1, c.rfind('}') - 1);
        for (const auto& t2 : split_top(c, ','))
          if (!trim(t2).empty()) list.push_back(trim(t2));
      }
      return it->second;
    };
    auto reg_list = [&](const std::string& tok) {
      std::vector<std::string> out;
      const std::string d = trim(tok);
      if (!d.empty() && d[0] == '{') {
        const std::string in = d.substr(1, d.rfind('}') - 1);
        for (const auto& t2 : split_top(in, ','))
          if (!trim(t2).empty()) out.push_back(trim(t2));
      } else if (!d.empty()) {
        out.push_back(d);
      }
      return out;
    };
    // The two ways to name the same image: a handle the state carries on the
    // CPU, and the kernel parameter the launcher binds on the GPU.
    auto image_of = [&](int idx) {
      const std::string i = std::to_string(idx);
      if (!msl) return "*weft::texture_ptr(" + i + ")";
      const bool rw = idx < (int)tex_write.size() && tex_write[idx];
      const bool arr = idx < (int)tex_array.size() && tex_array[idx];
      if (idx < (int)tex_cube.size() && tex_cube[idx])
        return "weft::make_cube_image(weft_t" + i + ")";
      if (idx < (int)tex_3d.size() && tex_3d[idx])
        return std::string(rw ? "weft::make_rw_image3d(weft_t" : "weft::make_image3d(weft_t") + i +
               ")";
      if (arr)
        return std::string(rw ? "weft::make_rw_array_image(weft_t" : "weft::make_array_image(weft_t") +
               i + ")";
      return std::string(rw ? "weft::make_rw_image(weft_t" : "weft::make_image(weft_t") + i +
             ")";
    };

    if (op == "txq") {
      if (args.size() < 2) { err = "bad " + opcode; return ""; }
      std::vector<std::string> list;
      const int idx = binding(args[1], list);
      if (idx < 0) return "";
      std::string what;
      const bool arr = (size_t)idx < tex_array.size() && tex_array[idx];
      const bool vol = (size_t)idx < tex_3d.size() && tex_3d[idx];
      // A cube's third extent is the six faces it is made of, which is a number
      // neither its width nor its height can produce.
      const bool cb = (size_t)idx < tex_cube.size() && tex_cube[idx];
      if (opcode.find("width") != std::string::npos) what = "width";
      else if (opcode.find("height") != std::string::npos) what = "height";
      // A layered binding has slices rather than a depth, and a volume has planes;
      // how many is the one number neither can derive from anything else it is
      // given.
      else if (opcode.find("depth") != std::string::npos)
        what = (arr || vol) ? "layers" : (cb ? "6" : "1");
      else { err = "txq asks for a texture's width, height or depth"; return ""; }
      const std::string i = std::to_string(idx);
      // The GPU handle names that extent `depth` on a volume and `layers` on a
      // layered image; the host `weft::Texture` calls it `layers` either way,
      // because it stores both the same way.
      const std::string msl_field = (what == "layers" && vol) ? "depth" : what;
      const std::string expr =
          what == "6" ? std::string("6u")
                       : what == "1" ? std::string("1u")
                       : (msl ? ("(uint32_t)" + image_of(idx) + "." + msl_field)
                              : ("(uint32_t)weft::texture_ptr(" + i + ")->" + what));
      return guard + store(args[0], expr, "uint32_t");
    }

    if (op == "suld" || op == "sust") {
      const bool read = op == "suld";
      if (args.size() < (read ? 2u : 2u)) { err = "bad " + opcode; return ""; }
      std::vector<std::string> coords;
      const int idx = binding(args[read ? 1 : 0], coords);
      if (idx < 0) return "";
      const bool arr = (size_t)idx < tex_array.size() && tex_array[idx];
      const bool vol = (size_t)idx < tex_3d.size() && tex_3d[idx];
      if (coords.size() < ((arr || vol) ? 3u : (d1 ? 1u : 2u))) {
        err = vol ? "a 3-D surface needs x, y and a plane"
                  : arr ? "a 2-D array surface needs x, y and a layer"
                  : d1  ? "a 1-D surface needs one coordinate"
                        : "a 2-D surface needs two coordinates";
        return "";
      }
      if (!read && idx < (int)tex_write.size()) tex_write[idx] = true;
      const std::string x = "(uint32_t)(" + operand_as(coords[0], "uint32_t") + ")";
      const std::string y = d1 ? std::string("0u")
                               : "(uint32_t)(" + operand_as(coords[1], "uint32_t") + ")";
      const std::string lay = (arr || vol)
                                  ? (", (uint32_t)(" + operand_as(coords[2], "uint32_t") + ")")
                                  : std::string();
      if (read) {
        return guard + store(args[0],
                             "weft::tex_read<uint32_t>(" + image_of(idx) + ", " + x + ", " + y +
                                 lay + ")",
                             "uint32_t");
      }
      std::vector<std::string> vals = reg_list(args[1]);
      if (vals.empty()) { err = "a surface store needs a value"; return ""; }
      // The image is one channel wide, so the first register is the texel and
      // the rest of a vector has nowhere to go — what an R32 format does on the
      // hardware too.
      const std::string v = operand_as(vals[0], "float");
      const std::string call =
          msl ? ("weft::tex_write(" + image_of(idx) + ", " + x + ", " + y + lay + ", float4(" +
                 v + ", 0.0f, 0.0f, 0.0f))")
              : ("weft::tex_write(" + image_of(idx) + ", " + x + ", " + y + lay + ", (float)(" +
                 v + "))");
      return guard + call + ";";
    }

    // A cube sample: the face and the position on it, turned into the direction
    // the sampler takes. `weft_cube_dir` is the inverse of the rule that picks a
    // face, so the direction comes back to this face at this place.
    if (cube) {
      std::vector<std::string> coords;
      const int idx = binding(args[1], coords);
      if (idx < 0) return "";
      if (coords.size() < 3) {
        err = "a cube sample needs s, t and a face";
        return "";
      }
      const std::string s = operand_as(coords[0], "float");
      const std::string t = operand_as(coords[1], "float");
      const std::string f = "(uint32_t)(" + operand_as(coords[2], "uint32_t") + ")";
      const std::string i = std::to_string(idx);
      // A binding the kernel also writes is declared read_write on this target,
      // and a filter cannot read through that mode — so the sample reaches the
      // second, sampled view of the same texture that the module declares.
      const std::string tn = (msl && idx < (int)tex_write.size() && tex_write[idx])
                                 ? ("weft_t" + i + "_s")
                                 : ("weft_t" + i);
      const std::string img =
          msl ? ("weft::SampledCubeImage{weft::make_cube_image(" + tn + "), weft_s" + i + "}")
              : ("*weft::texture_ptr(" + i + ")");
      // A gather keeps the four texels apart, and a face is a layer here, so it
      // needs no direction: `(s, t, face)` rebases straight to a coordinate.
      const bool cgather = op == "tld4";
      const std::string call =
          cgather ? ("weft::tex_gather4_cube<float>(" + img + ", " + s + ", " + t + ", " + f +
                     ", weft_t4)")
          : msl ? ("weft::tex_sample4<float>(" + img + ", weft_dx, weft_dy, weft_dz, weft_t4)")
                : ("weft::tex_sample4_cube<float>(" + img +
                    ", weft_dx, weft_dy, weft_dz, weft::sampler(" + i + ").filter, " +
                    "weft::sampler(" + i + ").addr, weft_t4)");
      std::vector<std::string> dsts = reg_list(args[0]);
      std::string out = guard + "{ float weft_t4[4]; " +
                        (cgather ? std::string()
                                 : "float weft_dx, weft_dy, weft_dz; weft::weft_cube_dir(" + f +
                                       ", " + s + ", " + t + ", weft_dx, weft_dy, weft_dz); ") +
                        call + ";";
      for (size_t k = 0; k < dsts.size() && k < 4; k++)
        out += " " + store(dsts[k], "weft_t4[" + std::to_string(k) + "]", "float");
      out += " }";
      return out;
    }

    // `tex`: a filtered sample at normalised coordinates. The filter and the
    // address mode are the bound sampler state, which is where the GPU keeps
    // them and what the CPU runtime models with the same two choices.
    if (args.size() < 2) { err = "bad " + opcode; return ""; }
    std::vector<std::string> coords;
    const int idx = binding(args[1], coords);
    if (idx < 0) return "";
    const bool arr = (size_t)idx < tex_array.size() && tex_array[idx];
    const bool vol = (size_t)idx < tex_3d.size() && tex_3d[idx];
    if (coords.size() < ((arr || vol) ? 3u : (d1 ? 1u : 2u))) {
      err = vol ? "a 3-D texture needs u, v and w"
                : arr ? "a 2-D array texture needs u, v and a layer"
                : d1  ? "a 1-D texture needs one coordinate"
                      : "a 2-D texture needs two coordinates";
      return "";
    }
    const std::string u = operand_as(coords[0], "float");
    // One row, so 0.5 is its centre: every filter and address mode lands on the
    // texel the coordinate names instead of mixing in a row that is not there.
    const std::string v = d1 ? std::string("0.5f") : operand_as(coords[1], "float");
    // The layer is a float coordinate in the array form — CUDA's `tex2DLayered`
    // passes the slice as one — so it is truncated to the index the slice has.
    // A volume's third coordinate stays a float: it is a normalised position in
    // the stack of planes, not a slice index to truncate.
    const std::string lay = vol ? (", " + operand_as(coords[2], "float"))
                                : arr ? (", (uint32_t)(" + operand_as(coords[2], "float") + ")")
                                      : std::string();
    const std::string i = std::to_string(idx);
    // The sampled element type comes from the instruction's own `.f32`, so the
    // channels land in registers of the type PTX asked for.
    const bool gather = op == "tld4";
    // A binding the kernel also writes is `access::read_write` on this target, and
    // no filter reads through that mode: the sample takes the second, sampled view
    // of the same texture instead. Same storage, other access mode.
    const std::string tn = (msl && idx < (int)tex_write.size() && tex_write[idx])
                               ? ("weft_t" + i + "_s")
                               : ("weft_t" + i);
    const std::string img =
        msl ? (std::string(vol ? "weft::SampledImage3D{weft::make_image3d("
                               : arr ? "weft::SampledArrayImage{weft::make_array_image("
                                     : "weft::SampledImage{weft::make_image(") +
               tn + "), weft_s" + i + "}")
            : ("*weft::texture_ptr(" + i + ")");
    std::string call;
    if (gather)
      call = "weft::tex_gather4<float>(" + img + ", " + u + ", " + v + lay + ", weft_t4)";
    else if (msl)
      call = "weft::tex_sample4<float>(" + img + ", " + u + ", " + v + lay + ", weft_t4)";
    else
      call = "weft::tex_sample4<float>(" + img + ", " + u + ", " + v + lay +
             ", weft::sampler(" + i + ").filter, weft::sampler(" + i + ").addr, weft_t4)";;
    std::vector<std::string> dsts = reg_list(args[0]);
    std::string out = guard + "{ float weft_t4[4]; " + call + ";";
    for (size_t k = 0; k < dsts.size() && k < 4; k++)
      out += " " + store(dsts[k], "weft_t4[" + std::to_string(k) + "]", "float");
    out += " }";
    return out;
  }

  if (op == "nop") return guard + ";";

  err = "unsupported PTX instruction '" + opcode + "'";
  return "";
}

}  // namespace

static TranslateResult translate_ptx_impl(const std::string& raw,
                                          const std::vector<std::pair<int, uint32_t>>& widths,
                                          bool gpu) {
  TranslateResult r;
  std::string src = strip_comments(raw);

  // ---- texture declarations ----
  //
  // `.tex .u32 name, 1, 1, 1, 1;` declares a sampled texture and `.surf .u64
  // name, ...;` a surface the kernel also writes into. The declarations are
  // numbered in order, and that order is the order `--textures` fills, so a
  // kernel and its binding agree without the caller being handed an index.
  std::map<std::string, int> tex_index;
  int ntex = 0;
  {
    const size_t first = src.find(".entry");
    const std::string head = src.substr(0, first == std::string::npos ? src.size() : first);
    // One declaration per line: the module prologue is not semicolon-terminated
    // throughout (`.version 7.0` has no `;`), so splitting on `;` would swallow
    // the first declaration into the version line.
    size_t at = 0;
    while (at < head.size()) {
      size_t nl = head.find('\n', at);
      if (nl == std::string::npos) nl = head.size();
      std::string s = trim(head.substr(at, nl - at));
      at = nl + 1;
      if (!s.empty() && s.back() == ';') s.pop_back();
      s = trim(s);
      if (s.rfind(".tex", 0) != 0 && s.rfind(".surf", 0) != 0) continue;
      // `.texref t` carries no type token; `.tex .u32 t` does.
      std::string rest = trim(s.substr(s.find_first_of(" \t")));
      std::string nm = rest;
      const size_t sp = rest.find_first_of(" \t");
      if (sp != std::string::npos) nm = rest.substr(sp + 1);
      nm = trim(nm.substr(0, nm.find(',')));
      if (nm.empty() || tex_index.count(nm)) continue;
      tex_index[nm] = ntex++;
    }
  }

  std::string body_out;
  std::vector<TranslatedKernel> kernels;

  size_t pos = 0;
  while (true) {
    size_t e = src.find(".entry", pos);
    if (e == std::string::npos) break;

    size_t i = e + 6;
    while (i < src.size() && std::isspace((unsigned char)src[i])) i++;
    size_t name_start = i;
    while (i < src.size() && is_word(src[i])) i++;
    std::string name = src.substr(name_start, i - name_start);

    size_t paren = src.find('(', i);
    if (paren == std::string::npos) { r.error = "malformed .entry"; return r; }
    size_t close = find_matching(src, paren, '(', ')');
    if (close == std::string::npos) { r.error = "unbalanced parameter list in " + name; return r; }
    size_t brace = src.find('{', close);
    if (brace == std::string::npos) { r.error = "kernel " + name + " has no body"; return r; }
    size_t end = find_matching(src, brace, '{', '}');
    if (end == std::string::npos) { r.error = "unbalanced body in " + name; return r; }

    // `.maxntid` sits immediately before the entry it applies to: the launch size
    // the module was compiled for. Triton writes it from `num_warps`, and it is
    // the only record of the geometry a kernel was indexed against.
    uint32_t tg_hint = 0;
    {
      const size_t m = src.rfind(".maxntid", e);
      const size_t semi = m == std::string::npos ? std::string::npos : src.find(';', m);
      if (m != std::string::npos && m >= pos) {
        const size_t v = src.find_first_of("0123456789", m);
        if (v != std::string::npos && semi != std::string::npos && v < semi)
          tg_hint = (uint32_t)std::strtoul(src.c_str() + v, nullptr, 10);
      }
    }

    Emitter em;
    em.tex_index = tex_index;
    em.ntex = ntex;
    em.tex_write.assign((size_t)ntex, false);
    // The body says which bindings are arrays — an `.a2d` instruction names one
    // with `[name, {...}]` — so it is scanned once here rather than guessed from
    // the declaration, which does not carry dimensionality.
    em.tex_array.assign((size_t)ntex, false);
    // The same scan for `.3d`: the declaration says nothing about dimensionality,
    // so the instruction is what makes a binding a volume.
    em.tex_3d.assign((size_t)ntex, false);
    em.tex_cube.assign((size_t)ntex, false);
    {
      size_t at = brace;
      while (at < end) {
        const size_t hit = src.find(".3d", at);
        if (hit == std::string::npos || hit >= end) break;
        const size_t semi = src.find(';', hit);
        const size_t lb = src.find('[', hit);
        if (lb != std::string::npos && lb < end && (semi == std::string::npos || lb < semi)) {
          size_t e3 = lb + 1;
          while (e3 < src.size() && std::isspace((unsigned char)src[e3])) e3++;
          const size_t b3 = e3;
          while (e3 < src.size() && is_word(src[e3])) e3++;
          auto t3 = tex_index.find(src.substr(b3, e3 - b3));
          if (t3 != tex_index.end() && (size_t)t3->second < em.tex_3d.size())
            em.tex_3d[t3->second] = true;
        }
        at = hit + 3;
      }
    }
    {
      size_t at = brace;
      while (at < end) {
        const size_t hit = src.find(".cube", at);
        if (hit == std::string::npos || hit >= end) break;
        const size_t semi = src.find(';', hit);
        const size_t lb = src.find('[', hit);
        if (lb != std::string::npos && lb < end && (semi == std::string::npos || lb < semi)) {
          size_t ec = lb + 1;
          while (ec < src.size() && std::isspace((unsigned char)src[ec])) ec++;
          const size_t bc = ec;
          while (ec < src.size() && is_word(src[ec])) ec++;
          auto tc = tex_index.find(src.substr(bc, ec - bc));
          if (tc != tex_index.end() && (size_t)tc->second < em.tex_cube.size())
            em.tex_cube[tc->second] = true;
        }
        at = hit + 5;
      }
    }
    {
      size_t at = brace;
      while (at < end) {
        const size_t hit = src.find("a2d", at);
        if (hit == std::string::npos || hit >= end) break;
        const size_t semi = src.find(';', hit);
        const size_t lb = src.find('[', hit);
        if (lb != std::string::npos && lb < end && (semi == std::string::npos || lb < semi)) {
          size_t e = lb + 1;
          while (e < src.size() && std::isspace((unsigned char)src[e])) e++;
          const size_t b = e;
          while (e < src.size() && is_word(src[e])) e++;
          auto t2 = tex_index.find(src.substr(b, e - b));
          if (t2 != tex_index.end() && (size_t)t2->second < em.tex_array.size())
            em.tex_array[t2->second] = true;
        }
        at = hit + 3;
      }
    }
    // ---- parameters, in declaration order ----
    //
    // A pointer the body never names is not bound. Triton hands every kernel a
    // `global_scratch` pointer and most kernels never touch it; a CUDA kernel
    // declares parameters an `#if` compiled out. Binding one would make the
    // caller invent a buffer for a slot nothing reads, so buffer indices are
    // assigned over the pointers that are actually used — which is the order
    // Triton's own launcher binds them in.
    std::vector<std::pair<std::string, bool>> pdecl;  // name, is a pointer
    for (const std::string& p : split_top(src.substr(paren + 1, close - paren - 1), ',')) {
      if (p.empty()) continue;
      std::vector<std::string> toks = split_top(p, ' ');
      std::string pname = toks.back();
      bool is_ptr = false;
      for (const auto& t : toks)
        if (t == ".u64" || t == ".b64" || t == ".s64" || t == ".ptr") is_ptr = true;
      pdecl.emplace_back(pname, is_ptr);
    }

    std::string body = src.substr(brace + 1, end - brace - 1);

    // How often each register is named in the body, its declaration aside. A
    // register named once is one nothing reads: a result that is written and
    // dropped. That is the difference between `atom.add.u64` — which has to
    // return the pair as one value and cannot here — and `red.add.u64`, whose
    // answer is the same and which this host has.
    {
      std::map<std::string, int> uses;
      size_t p = 0;
      while (p < body.size()) {
        size_t nl = body.find('\n', p);
        if (nl == std::string::npos) nl = body.size();
        const std::string line = trim(body.substr(p, nl - p));
        p = nl + 1;
        if (line.rfind(".reg", 0) == 0) continue;
        for (size_t i = 0; i < line.size(); i++) {
          if (line[i] != '%') continue;
          size_t j = i + 1;
          while (j < line.size() && (std::isalnum((unsigned char)line[j]) || line[j] == '_')) j++;
          if (j > i + 1) uses[line.substr(i + 1, j - i - 1)]++;
          i = j ? j - 1 : i;
        }
      }
      em.reg_reads = uses;
    }

    for (const auto& pd : pdecl) {
      if (!pd.second) {
        // Scalars keep their declared positions: `--arg I` names the I-th scalar
        // the kernel declares, whether or not this build reads it.
        em.scalars[pd.first] = (int)em.scalars.size();
      } else if (mentions(body, pd.first)) {
        em.params[pd.first] = (int)em.params.size();
      } else {
        em.unused.push_back(pd.first);
      }
    }
    std::string shared_decl;

    // Shared memory is declared before use, but scan it out first so the
    // instruction pass knows the arena exists.
    //
    // Every `.shared` array gets its own offset in the group's window, laid out
    // in declaration order and aligned to 16 bytes, so a kernel with two tiles
    // sees two tiles. `extern .shared` — a size chosen by the launch — sits after
    // the static ones, which is where CUDA's dynamic shared memory lives; weft's
    // launch gives the whole 48 KiB arena to it.
    {
      uint64_t off = 0;
      bool any = false;
      std::vector<std::pair<std::string, bool>> externs;
      for (size_t p = 0; (p = body.find(".shared", p)) != std::string::npos; p += 7) {
        // Declarations only: `.shared` is also the space qualifier of
        // `st.shared.f32`, which follows an opcode rather than a statement start.
        size_t q = p;
        while (q > 0 && (body[q - 1] == ' ' || body[q - 1] == '\t')) q--;
        bool decl_start = (q == 0 || body[q - 1] == '\n' || body[q - 1] == ';' || body[q - 1] == '{');
        // `.extern .shared` is a declaration too: the token before `.shared` is
        // the `extern` qualifier, not the start of a statement.
        if (!decl_start && q >= 6 && body.compare(q - 6, 6, "extern") == 0) decl_start = true;
        if (!decl_start) continue;
        const size_t semi = body.find(';', p);
        if (semi == std::string::npos) break;
        const std::string d = trim(body.substr(p + 7, semi - p - 7));
        const size_t lb = d.find('[');
        std::string name_field = lb == std::string::npos ? d : trim(d.substr(0, lb));
        const size_t sp = name_field.find_last_of(" \t");
        const std::string nm = sp == std::string::npos ? name_field : trim(name_field.substr(sp + 1));
        std::string extent;
        if (lb != std::string::npos) {
          const size_t rb = d.find(']', lb);
          if (rb != std::string::npos) extent = trim(d.substr(lb + 1, rb - lb - 1));
        }
        if (!extent.empty() && extent.find_first_not_of("0123456789") != std::string::npos) {
          r.error = "kernel " + name + ": shared array size '" + extent + "' is not a constant";
          return r;
        }
        any = true;
        if (extent.empty()) {
          externs.emplace_back(nm, true);
          continue;
        }
        off = (off + 15) & ~(uint64_t)15;
        em.shared_offsets[nm] = off;
        off += std::strtoull(extent.c_str(), nullptr, 10);
        shared_decl += extent + " ";
      }
      // The dynamic array starts where the static ones end.
      for (const auto& e : externs) em.shared_offsets[e.first] = off;
      if (any && shared_decl.empty()) shared_decl = " ";  // extern only: still memory
      // The GPU needs the size at compile time, because the arena is a
      // `threadgroup` array. An extern-only arena gets the 48 KiB the CPU
      // runtime hands out, so both sides can address the same slots.
      em.shared_extern = !externs.empty();
      em.shared_bytes = (int)(externs.empty() ? off : off + 48u * 1024u);
    }
    em.shared_size = &shared_decl;

    // Walk the body once per backend. The statement loop and the expression
    // builders are shared; only the framing differs — one lane at a time, or the
    // 32 lanes of a warp as one vector.
    auto walk = [&](Emitter& e, std::string& out, std::string& struct_decls) -> bool {
      std::string stmt, lane_batch;
      // Can this statement be fused into a lane loop with its neighbours? Only if
      // every element it touches is indexed by the lane number itself. A gather
      // (`r[18][l + d]`, what a shuffle compiles to) has a runtime index, and
      // one of those in the loop stops the vectoriser for the whole loop — which
      // costs far more than the loop entry it saves.
      auto elementwise = [](const std::string& s) {
        for (size_t i = 0; i < s.size(); i++) {
          if (s[i] != '[') continue;
          const size_t e = s.find(']', i);
          if (e == std::string::npos) return false;
          const std::string in = s.substr(i + 1, e - i - 1);
          if (in == "l") { i = e; continue; }
          if (in.empty() || in.find_first_not_of("0123456789") != std::string::npos) return false;
          if (s.compare(e + 1, 3, "[l]") != 0) return false;  // a register index, not lane data
          i = e + 2;
        }
        return true;
      };
      auto flush_lanes = [&]() {
        if (lane_batch.empty()) return;
        out += "  for (uint32_t l = 0; l < 32u; l++) { " + lane_batch + " }\n";
        lane_batch.clear();
      };
      for (size_t k = 0; k <= body.size(); k++) {
        char c = k < body.size() ? body[k] : ';';
        if (c != ';') { stmt.push_back(c); continue; }

        std::string s = trim(stmt);
        stmt.clear();
        // Block braces are separators, but `{ %r1, %r2 }` is an operand list.
        while (!s.empty() && (s.front() == '{' || s.front() == '}')) s.erase(s.begin());
        while (!s.empty() && (s.back() == '{' || s.back() == '}')) s.pop_back();
        s = trim(s);
        if (s.empty()) continue;

        // A label glued to its instruction: "$L_1: add.f32 ..."
        size_t colon = s.find(':');
        if (colon != std::string::npos && s.substr(0, colon).find_first_of(" \t") == std::string::npos) {
          flush_lanes();
          const std::string lab = trim(s.substr(0, colon));
          if (e.vec)
            out += "  L" + std::to_string(e.label_id(lab)) + ":;\n";
          else if (e.msl)
            out += "  case " + std::to_string(e.label_id(lab)) + "u:;\n";
          else
            out += "  " + sanitize(lab) + ":;\n";
          s = trim(s.substr(colon + 1));
          if (s.empty()) continue;
        }

        if (s.rfind(".reg", 0) == 0) {
          std::vector<std::string> toks = split_top(s, ' ');
          std::string ty, reg;
          for (const auto& t : toks) {
            if (!t.empty() && t[0] == '.') { if (!cpp_type(t).empty()) ty = cpp_type(t); }
            else reg = t;
          }
          Reg rr = parse_reg(reg);
          if (ty.empty() || !rr.ok) {
            e.err = "bad register declaration '" + s + "'";
            return false;
          }
          int count = 1;
          size_t lt = reg.find('<');
          if (lt != std::string::npos) count = std::atoi(reg.c_str() + lt + 1);
          // A double on the GPU target is the `uint64_t` its bits live in — the
          // language has no fp64 type and every operation on one is a `weft_fp64`
          // helper. The register is the same storage either way, so the decode
          // written for the CPU column is the decode that runs here.
          // A double is stored as the word its bits are, on both back ends that
          // cannot hold one natively: Metal has no fp64 type, and the warp backend
          // masks every register as integers. `operand_as` and `store` bitcast
          // across, so the decode above them is unchanged either way.
          if ((e.msl || e.vec) && ty == "double") ty = "uint64_t";
          if (e.vec && ty == "bool") ty = "uint32_t";
          e.reg_types[rr.base] = ty;
          struct_decls += "  " + ty + " " + rr.base + "[" + std::to_string(count) + "]" +
                          (e.vec ? "[32]" : "") + ";\n";
          continue;
        }
        if (s.rfind(".shared", 0) == 0 || s.rfind(".extern", 0) == 0) continue;
        if (!s.empty() && s[0] == '.') continue;  // .loc, .pragma, .maxnreg ...

        std::string line = e.emit(s);
        if (!e.err.empty()) return false;
        // `@%p op`: the lanes that fail the predicate must keep what they had.
        // The active mask alone does not say that — a predicate selects lanes
        // *within* the group — so the statement is guarded by it as well.
        if (e.vec && !e.vec_control && !e.vec_side && !e.vec_pred.empty())
          line = "if (" + e.vec_pred + ") { " + line + " }";
        if (e.vec) {
          // Straight-line register work is batched into one lane loop instead of
          // one loop per instruction. That is what lets the intermediates live in
          // vector registers: reopen the loop for every PTX instruction and every
          // temporary is written to the state arrays and read back, which is
          // several times the cost of the arithmetic itself.
          if (e.vec_control || e.vec_side || !elementwise(line)) {
            flush_lanes();
            out += "  " + (e.vec_control || e.vec_side
                               ? line
                               : "for (uint32_t l = 0; l < 32u; l++) { " + line + " }")
                   + "\n";
          } else {
            lane_batch += line + " ";
          }
        } else {
          out += "  " + line + "\n";
        }
      }
      flush_lanes();
      return true;
    };

    em.msl = gpu;
    std::string code, struct_decls;
    const bool lane_ok = walk(em, code, struct_decls);

    // The warp backend is optional: anything the vector path cannot express
    // leaves the kernel on the per-lane state machine instead of failing.
    Emitter emv;
    emv.vec = true;
    // The warp body becomes Metal source on this back end, so its math and its
    // helper names have to be the GPU's spellings, not `std::`.
    emv.msl = gpu;
    emv.tex_index = tex_index;
    emv.ntex = ntex;
    emv.tex_write.assign((size_t)ntex, false);
    emv.tex_cube = em.tex_cube;
    emv.params = em.params;
    emv.scalars = em.scalars;
    emv.reg_reads = em.reg_reads;
    emv.shared_offsets = em.shared_offsets;
    emv.shared_size = em.shared_size;
    std::string code_v, decls_v;
    const bool have_vec = walk(emv, code_v, decls_v) && emv.vec_ok && !decls_v.empty() &&
                          !code_v.empty();

    // A kernel with no per-lane body — a tensor-core fragment — is emitted for the
    // GPU through the warp back end instead: one Metal thread runs the whole
    // 32-lane warp, so a peer's register is `r[l]` and no shuffle is involved.
    // What that back end cannot express yet — a barrier, a shuffle, threadgroup
    // memory — still has no GPU form, and the message says which one stopped it.
    // The per-lane body stops at the first instruction it cannot express — a
    // tensor-core one — so a buffer reached only after that point is typed by the
    // warp body instead. Merge what each pass saw, keeping 0 wherever the two
    // disagree about a width.
    if (have_vec) {
      for (const auto& pb : emv.param_bytes) {
        auto it = em.param_bytes.find(pb.first);
        if (it == em.param_bytes.end()) em.param_bytes[pb.first] = pb.second;
        else if (it->second != pb.second) it->second = 0;
      }
    }
    const bool warp_only = !lane_ok && em.warp_only && have_vec;
    // A 64-bit atomic is built out of 32-bit ones in the warp runtime, which is
    // where the lock table lives — so a kernel that holds one is emitted for the
    // warp back end and launched at the same width as any other warp kernel.
    const bool wide = em.wide_atom || emv.wide_atom;
    const bool wide_gpu = gpu && wide && have_vec;
    bool gpu_warp_kernel = false;
    if (gpu && wide && !have_vec) {
      r.error = "kernel " + name +
                ": a 64-bit atomic is built from 32-bit ones in the warp runtime, and the "
                "warp body did not translate" +
                (emv.err.empty() ? "" : (": " + emv.err));
      return r;
    }
    if (gpu && !lane_ok && !warp_only && !wide_gpu) {
      r.error = "kernel " + name + ": " + (em.err.empty() ? "no per-lane body" : em.err);
      return r;
    }
    if (!lane_ok && !warp_only && !wide_gpu) {
      r.error = "kernel " + name + ": " + em.err;
      return r;
    }

    if (gpu && ((!lane_ok && warp_only) || wide_gpu)) {
      // The warp kernel: the same register file the CPU warp back end runs, as a
      // thread-local struct, with the dispatcher's computed goto written as a
      // switch — a shader has no jump table, but it has cases and labels.
      gpu_warp_kernel = true;
      const std::string wst = "weft_ws_" + name;
      const std::string wfn = "weft_gw_" + name;
      const int wnbufs = (int)em.params.size();
      // Threadgroup memory is a `threadgroup` array here, sized from the module.
      // One thread holds the whole warp, so an arena the lanes share is a slot
      // this thread's stack does not have to carry, and a slot another lane
      // writes is a slot it reads. The reduction slots live in the same window,
      // so the cap leaves room for them.
      int arena = em.shared_bytes;
      const int arena_cap = 32768 - 4096;
      if (arena > arena_cap) {
        if (!em.shared_extern) {
          r.error = "kernel " + name + ": " + std::to_string(arena) +
                    " bytes of threadgroup memory, more than the " +
                    std::to_string(arena_cap) + " a threadgroup can hold with its reduction slots";
          return r;
        }
        arena = arena_cap;
      }
      body_out += "struct " + wst + " {\n  uint32_t pc[32];\n" + decls_v + "};\n";
      // The body is a function, not the kernel, because one instruction leaves
      // it: a barrier ends a warp segment. In this target every lane arrives at
      // once, so there is nothing to suspend — the kernel calls the body again
      // and the pc values resume it at the next segment, which is what the CPU
      // scheduler does with the same barrier.
      body_out += "static uint32_t " + wfn + "(thread weft::WCtx& ctx, thread " + wst + "* s,\n";
      body_out += "    thread uint32_t* am, thread uint64_t* am64, thread uint32_t& weft_full,\n";
      body_out += "    threadgroup uint8_t* sh) {\n";
      body_out += "  (void)ctx; (void)sh; (void)am; (void)am64; (void)weft_full;\n";
      // Metal has no labels and no goto, so the dispatch is a loop around a
      // switch: one case per block, and a jump in the body is a `break` back to
      // the loop head, which re-reads the pc values and picks again.
      body_out += "  for (;;) {\n";
      body_out += "    uint32_t best = 0xFFFFFFFFu, m = 0;\n";
      body_out += "    for (uint32_t l = 0; l < weft::kWarpLanes; l++) { const uint32_t p = s->pc[l];\n";
      body_out += "      if (p == 0xFFFFFFFFu) continue;\n";
      body_out += "      if (p < best) { best = p; m = 1u << l; }\n";
      body_out += "      else if (p == best) m |= 1u << l; }\n";
      body_out += "    if (best == 0xFFFFFFFFu) return weft::kWarpDone;\n";
      body_out += "    weft_full = (m == 0xFFFFFFFFu) ? 1u : 0u;\n";
      body_out += "    for (uint32_t l = 0; l < weft::kWarpLanes; l++) {\n";
      body_out += "      am[l] = ((m >> l) & 1u) ? 0xFFFFFFFFu : 0u;\n";
      body_out += "      am64[l] = ((uint64_t)am[l] << 32) | am[l]; }\n";
      body_out += "    switch (best) {\n";
      body_out += warp_switch_body(code_v);
      body_out += "      default: return weft::kWarpDone;\n    }\n  }\n}\n\n";
      body_out += "kernel void " + name + "(";
      for (int i = 0; i < wnbufs; i++)
        body_out += "device uint8_t* weft_b" + std::to_string(i) + " [[buffer(" +
                    std::to_string(i) + ")]],\n                 ";
      body_out += "device uint64_t* weft_params [[buffer(" + std::to_string(wnbufs) +
                   ")]],\n                 ";
      // The launch the caller asked for, as two numbers. `threads_per_grid` is
      // not it: the group count rounds up, so the attribute reports a warp count
      // with the padding folded in, and a kernel told that number thinks lanes
      // past the end of the program are live.
      body_out += "device uint64_t* weft_geom [[buffer(" + std::to_string(wnbufs + 1) +
                   ")]],\n                 ";
      body_out += "device uint32_t* weft_locks [[buffer(" + std::to_string(wnbufs + 2) +
                   ")]],\n                 ";
      body_out += "uint weft_lid [[thread_index_in_threadgroup]],\n                 ";
      body_out += "uint3 weft_tgid [[threadgroup_position_in_grid]],\n                 ";
      body_out += "uint3 weft_tgsz [[threads_per_threadgroup]],\n                 ";
      body_out += "uint3 weft_grid [[threads_per_grid]]) {\n";
      body_out += "  thread " + wst + " st;\n  thread " + wst + "* s = &st;\n";
      // The register file starts zeroed: the CPU back end hands the warp a zeroed
      // blob, so a kernel that reads a register before writing it has to see the
      // same thing on both engines.
      body_out += "  for (uint32_t weft_i = 0; weft_i < (uint32_t)sizeof(st); weft_i++)\n";
      body_out += "    ((thread uint8_t*)&st)[weft_i] = 0u;\n";
      body_out += "  for (uint32_t l = 0; l < weft::kWarpLanes; l++) s->pc[l] = 0u;\n";
      body_out += "  weft::WCtx ctx;\n";
      // One thread is one warp. `%ctaid.x` is the block and `%tid.x` picks the warp
      // inside it — a lane's index is its position inside the threadgroup, so a
      // kernel that derives its fragment from `tid` sees 0..tg-1 in every group.
      body_out += "  ctx.lid0 = weft_lid * weft::kWarpLanes;\n";
      body_out += "  ctx.tgid = weft_tgid.x;\n";
      body_out += "  ctx.tg_size = (uint32_t)weft_geom[1];\n";
      body_out += "  ctx.grid = (uint32_t)weft_geom[0];\n";
      for (int i = 0; i < wnbufs; i++)
        body_out += "  ctx.b[" + std::to_string(i) + "] = weft_b" + std::to_string(i) + ";\n";
      body_out += "  ctx.params = weft_params;\n";
      // The lock words a 64-bit read-modify-write spins on: a bound buffer, zeroed
      // by the launcher before each dispatch.
      body_out += "  ctx.locks = weft_locks;\n";
      body_out += "  threadgroup uint8_t weft_sh[" +
                   std::to_string(arena > 0 ? arena : 1) + "];\n";
      body_out += "  threadgroup uint32_t weft_red_slots[1024];\n";
      // The same rule the CPU scheduler applies to a group the launch did not
      // fill: its missing lanes leave slots unread-by-anyone, so the arena is
      // zeroed rather than left holding whatever the last launch left there.
      // Zeroing it unconditionally was tried and made the dispatch less reliable
      // rather than more: with everything else equal the same fixture went from
      // every run answering to two runs in five never completing, so the shared
      // arena is not what the stall is made of and this stays as it was.
      body_out += "  if (weft::live_threads(ctx.tgid, ctx.tg_size, ctx.grid) < ctx.tg_size)\n";
      body_out += "    for (uint32_t weft_i = 0; weft_i < (uint32_t)sizeof(weft_sh); weft_i++)\n";
      body_out += "      weft_sh[weft_i] = 0u;\n";
      body_out += "  threadgroup uint8_t* sh = weft_sh;\n  (void)sh;\n";
      body_out += "  ctx.red = weft_red_slots;\n";
      if (emv.uses_red)
        body_out += "  for (uint32_t weft_i = 0; weft_i < 1024u; weft_i++) weft_red_slots[weft_i] = 0u;\n";
      body_out += "  uint32_t am[32];\n  uint64_t am64[32];\n  uint32_t weft_full = 0;\n";
      body_out += "  (void)am; (void)am64; (void)weft_full;\n";
      // One call per barrier segment: the body returns either "every lane is
      // done" or "a segment ended", and a resume after a barrier is the next
      // call.
        body_out += "  for (;;) {\n";
        body_out += "    if (" + wfn +
                     "(ctx, s, am, am64, weft_full, weft_sh) == weft::kWarpDone) return;\n";
        body_out += "  }\n}\n\n";
    } else if (lane_ok && gpu) {
      // The GPU body is the same instruction text as the CPU one, framed as a
      // kernel: a thread-local register file, the runtime's context, and a
      // dispatch switch over the program counter. A branch is `pc = N` and a
      // restart of the switch; a barrier is the hardware's; the state a lane
      // keeps between barriers is just its locals.
      const std::string st = "weft_st_" + name;
      body_out += "struct " + st + " {\n  uint32_t pc;\n" + struct_decls + "};\n";
      // Each buffer is its own `[[buffer(i)]]` resource, the way a handwritten
      // Metal kernel declares one, because the host cannot know a GPU address:
      // the launcher binds the resource and the kernel takes the pointer. The
      // scalar parameters follow, at the index after the last buffer.
      const int nbufs = (int)em.params.size();
      body_out += "kernel void " + name + "(";
      for (int i = 0; i < nbufs; i++)
        body_out += "device uint8_t* weft_b" + std::to_string(i) + " [[buffer(" +
                    std::to_string(i) + ")]],\n                 ";
      body_out += "device uint64_t* weft_params [[buffer(" + std::to_string(nbufs) +
                   ")]],\n                 ";

      // An image is a binding, and on the GPU a binding is a kernel parameter.
      // The access mode is part of the type: a surface needs `read_write` and a
      // filter needs `sample`, which is why a written binding is declared twice.
      for (int i = 0; i < em.ntex; i++) {
        const bool rw = (size_t)i < em.tex_write.size() && em.tex_write[i];
        const bool arr = (size_t)i < em.tex_array.size() && em.tex_array[i];
        const bool vol = (size_t)i < em.tex_3d.size() && em.tex_3d[i];
        const bool cb = (size_t)i < em.tex_cube.size() && em.tex_cube[i];
        const std::string shape =
            cb ? "texturecube<float, access::"
            : vol ? "texture3d<float, access::"
                 : arr ? "texture2d_array<float, access::" : "texture2d<float, access::";
        const std::string ni = std::to_string(i);
        body_out += shape + (rw ? "read_write" : "sample") + "> weft_t" + ni + " [[texture(" +
                    ni + ")]],\n                 ";
        // A binding the kernel writes needs `access::read_write` for the write,
        // and no filter reads through that mode. So the same texture is declared
        // a second time as a sampled view, one past the last binding, and the
        // launcher binds the one handle to both: the storage is the same, only the
        // access mode differs, and a filtered read reaches what the writes put
        // there.
        const std::string si = std::to_string(rw ? i + em.ntex : i);
        if (rw)
          body_out += shape + "sample> weft_t" + ni + "_s [[texture(" + si +
                      ")]],\n                 ";
        body_out += "sampler weft_s" + ni + " [[sampler(" + si + ")]],\n                 ";
      }
      body_out += "uint weft_lid [[thread_index_in_threadgroup]],\n                 ";
      body_out += "uint3 weft_tgid [[threadgroup_position_in_grid]],\n                 ";
      body_out += "uint3 weft_tgsz [[threads_per_threadgroup]],\n                 ";
      body_out += "uint3 weft_grid [[threads_per_grid]]) {\n";
      body_out += "  thread " + st + " st;\n  thread " + st + "* s = &st;\n";
      body_out += "  s->pc = 0u;\n";
      body_out += "  weft::Ctx ctx;\n";
      body_out += "  ctx.lid = weft_lid; ctx.tgid = weft_tgid.x; ctx.tg_size = weft_tgsz.x;\n";
      body_out += "  ctx.grid = weft_grid.x;\n";
      for (int i = 0; i < nbufs; i++)
        body_out += "  ctx.b[" + std::to_string(i) + "] = weft_b" + std::to_string(i) + ";\n";
      body_out += "  ctx.params = weft_params;\n  (void)ctx;\n";
      // A threadgroup has 32 KiB of memory on this hardware; an `extern` arena
      // is sized by the launch, so it takes what is there rather than the 48 KiB
      // the CPU runtime hands out. A static declaration that does not fit has no
      // GPU answer, and the run says so instead of failing in the compiler.
      int arena = em.shared_bytes;
      if (arena > 32768) {
        if (!em.shared_extern) {
          r.error = "kernel " + name + ": " + std::to_string(arena) +
                    " bytes of threadgroup memory, more than the 32 KiB a threadgroup has";
          return r;
        }
        arena = 32768;
      }
      if (arena > 0) {
        body_out += "  threadgroup uint8_t weft_sh[" + std::to_string(arena) + "];\n";
        body_out += "  threadgroup uint8_t* sh = weft_sh;\n  (void)sh;\n";
      }
      if (em.uses_red)
        body_out += "  threadgroup uint32_t weft_red_slots[1024];\n  (void)weft_red_slots;\n";
      body_out += "  for (;;) {\n    switch (s->pc) {\n      case 0u:;\n";
      body_out += code;
      body_out += "      default: return;\n    }\n  }\n}\n\n";
    } else if (lane_ok) {
      const std::string st = "weft_st_" + name;
      std::string head = "struct " + st + " {\n  uint32_t pc;\n" + struct_decls + "};\n";
      // The lane loop is inside the body: the switch and its resume labels are
      // reached by one call per threadgroup per barrier segment, and the loop
      // keeps the state blob in cache between lanes. The same code was a call
      // per lane before, and at ~20 cycles a lane the call was most of it.
      head += "static uint32_t weft_g_" + name +
              "(weft::Ctx* ctxs, unsigned char* states, size_t stride, uint32_t live,\n"
              " uint32_t weft_pass, void* const* bufs, unsigned char* sh) {\n";
      head += "  uint32_t weft_left = 0;\n";
      head += "  for (uint32_t i = 0; i < live; i++) {\n";
      head += "    " + st + "* s = (" + st + "*)(states + (size_t)i * stride);\n";
      head += "    if (s->pc == weft::kThreadDone) continue;\n";
      // A segment ends with `return;`, which must leave the lane and not the
      // group — so the body is a lambda invoked once per lane, and `return` is
      // the lane's own. Labels and the switch stay inside it.
      head += "    [&]() {\n";
      head += "      weft::Ctx& ctx = ctxs[i];\n";
      head += "      ctx.pass = weft_pass;\n";
      head += "      (void)bufs; (void)sh;\n";
      head += "      switch (s->pc) {\n" + em.resume_cases + "      }\n";
      head += code;
      head += "      s->pc = weft::kThreadDone;\n";
      head += "    }();\n";
      head += "    if (s->pc != weft::kThreadDone) weft_left++;\n";
      head += "  }\n  return weft_left;\n}\n";
      // One lane, for the paths that still resume a single thread: a direct call
      // into the loop above rather than a second copy of the body.
      head += "static void weft_k_" + name +
              "(weft::Ctx& ctx, void* stp, void* const* bufs, unsigned char* sh) {\n";
      head += "  weft_g_" + name + "(&ctx, (unsigned char*)stp, 0, 1, ctx.pass, bufs, sh);\n}\n\n";
      body_out += head;
    }

    if (have_vec && !gpu) {
      const std::string ws = "weft_ws_" + name;
      body_out += "struct " + ws + " {\n  uint32_t pc[32];\n" + decls_v + "};\n";
      body_out += "static uint32_t weft_w_" + name + "(weft::WCtx& ctx, void* stp) {\n";
      // `restrict`: the state blob is the only thing this function reaches
      // through a pointer, so a write to a register must not be assumed to
      // clobber the lane masks held in local registers — without it the mask is
      // reloaded around every write and the lane loop stops vectorising.
      body_out += "  " + ws + "* __restrict s = (" + ws + "*)stp;\n";
      body_out += "  unsigned char* sh = ctx.shared;\n";
      body_out += "  static void* const weft_tbl[] = {";
      for (int i = 0; i < emv.label_count; i++)
        body_out += (i ? ", &&L" : "&&L") + std::to_string(i);
      body_out += "};\n";
      body_out += "  uint32_t am[32];\n  uint64_t am64[32];\n  uint32_t weft_full = 0;\n";
      body_out += "  (void)ctx; (void)sh; (void)am; (void)am64; (void)weft_tbl; (void)weft_full;\n";
      body_out += "  goto dispatch;\n";
      body_out += "  L0:;\n";
      body_out += code_v;
      body_out += "  { for (uint32_t l = 0; l < 32u; l++) if (am[l]) s->pc[l] = 0xFFFFFFFFu; }\n";
      body_out +=
          "  dispatch:\n"
          "  { uint32_t best = 0xFFFFFFFFu, m = 0;\n"
          "    for (uint32_t l = 0; l < 32u; l++) { const uint32_t p = s->pc[l];\n"
          "      if (p == 0xFFFFFFFFu) continue;\n"
          "      if (p < best) { best = p; m = 1u << l; }\n"
          "      else if (p == best) m |= 1u << l; }\n"
          "    if (best == 0xFFFFFFFFu) return weft::kWarpDone;\n"
          "    weft_full = (m == 0xFFFFFFFFu) ? 1u : 0u;\n"
          "    for (uint32_t l = 0; l < 32u; l++) { am[l] = ((m >> l) & 1u) ? 0xFFFFFFFFu : 0u;\n"
          "      am64[l] = ((uint64_t)am[l] << 32) | am[l]; }\n"
          "    goto *weft_tbl[best];\n"
          "  }\n"
          "}\n\n";
    }

    TranslatedKernel tk;
    tk.gpu_warp = gpu_warp_kernel;
    tk.wide = wide_gpu;
    tk.name = name;
    tk.nargs = (int)em.params.size();
    tk.nbufs = tk.nargs;  // a PTX kernel has no render stages: its own buffers
    // Each binding's element width, from the instructions that reach it. A
    // parameter with no typed access keeps 0 — unknown, not four bytes.
    tk.buf_type.assign((size_t)tk.nbufs, std::string("0"));
    for (const auto& pb : em.param_bytes)
      if (pb.first >= 0 && pb.first < tk.nbufs && pb.second)
        tk.buf_type[(size_t)pb.first] = std::to_string(pb.second);
    tk.sync = em.uses_barrier || em.uses_shfl || !shared_decl.empty();
    tk.flat = false;
    tk.warp = have_vec;
    tk.tg_hint = tg_hint;
    tk.ntex = em.ntex;
    // What each binding is, by name. PTX's `.tex` says nothing about the shape, so
    // the instruction that reads it is what names it — and a 1-D texture lowers
    // onto a plane one row high, so its binding is a 2-D one.
    for (int i = 0; i < em.ntex; i++) {
      const bool cb = (size_t)i < em.tex_cube.size() && em.tex_cube[i];
      const bool vol = (size_t)i < em.tex_3d.size() && em.tex_3d[i];
      const bool arr = (size_t)i < em.tex_array.size() && em.tex_array[i];
      tk.tex_shape.push_back(cb ? "cube" : vol ? "3d" : arr ? "array" : "2d");
    }
    tk.nsamp = em.ntex;  // the sampler state that goes with each binding
    tk.tex_rw = em.tex_write;
    tk.unused = em.unused;
    tk.warp_only = warp_only;
    kernels.push_back(tk);

    pos = end + 1;
  }

  if (kernels.empty()) {
    r.error = "no .entry found (expected `.visible .entry name(...)`)";
    return r;
  }

  if (gpu) {
    // The GPU module is only the kernels and their runtime: there is no kernel
    // table, because Metal finds a kernel by name and the launcher does the
    // dispatch.
    std::string g = "// generated by weft (ptx front end, metal target) — do not edit\n";
    g += kWWeftMslPreamble;
    g += kWWeftWarpMsl;
    // The portable library. Metal's `sqrt` is `x * rsqrt(x)` bit for bit and its
    // `ex2`, `lg2`, `sin` and `cos` are the unit's own polynomials, so a `.approx`
    // form lowered to the hardware answers differently from the same form lowered
    // to the host's libm — measured at 3.4e-07 relative on 65,536 lanes before
    // this line existed. One text on both targets is the answer the rest of this
    // project already gives.
    g += kWSoftMath;
    g += body_out;
    r.gpu_src = g;
    r.ok = true;
    r.kernels = kernels;
    return r;
  }

  std::string code =
      "// generated by weft (ptx front end) — do not edit\n"
      "#include \"weft_runtime.hpp\"\n"
      "#include <algorithm>\n"
      "#include <limits>\n"
      "#include <cmath>\n"
      "#include <cstdint>\n"
      "#include <cstring>\n"
      // The vector and half types the portable library names, with Metal's sizes
      // and alignments and the operators Metal defines for them: the library is
      // written once and expects the same language on both sides.
      "using half = _Float16;\n"
      // Metal spells the 32-bit unsigned type `uint`; the library is written once
      // and uses it, so C++ gets the same name rather than a second copy of the
      // text with every type renamed.
      "using uint = uint32_t;\n"
      "struct alignas(8) float2 { float x = 0, y = 0;\n"
      "  float2() {}\n"
      "  float2(float v) : x(v), y(v) {}\n"
      "  float2(float a, float b) : x(a), y(b) {}\n"
      "};\n"
      "struct alignas(16) float3 { float x = 0, y = 0, z = 0, _pad = 0;\n"
      "  float3() {}\n"
      "  float3(float v) : x(v), y(v), z(v) {}\n"
      "  float3(float a, float b, float c) : x(a), y(b), z(c) {}\n"
      "};\n"
      "struct alignas(16) float4 { float x = 0, y = 0, z = 0, w = 0;\n"
      "  float4() {}\n"
      "  float4(float v) : x(v), y(v), z(v), w(v) {}\n"
      "  float4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}\n"
      "};\n"
      "struct alignas(4) half2 { half x = 0, y = 0;\n"
      "  half2() {}\n"
      "  half2(half v) : x(v), y(v) {}\n"
      "  half2(half a, half b) : x(a), y(b) {}\n"
      "};\n"
      "struct alignas(8) half3 { half x = 0, y = 0, z = 0, _pad = 0;\n"
      "  half3() {}\n"
      "  half3(half v) : x(v), y(v), z(v) {}\n"
      "  half3(half a, half b, half c) : x(a), y(b), z(c) {}\n"
      "};\n"
      "struct alignas(8) half4 { half x = 0, y = 0, z = 0, w = 0;\n"
      "  half4() {}\n"
      "  half4(half v) : x(v), y(v), z(v), w(v) {}\n"
      "  half4(half a, half b, half c, half d) : x(a), y(b), z(c), w(d) {}\n"
      "};\n"
      // The operators the library's vector overloads use: negate, add, subtract,
      // multiply and divide, by a vector or by a scalar.
      // One macro per arity, because a `float4` has four members and a `float2`
      // two: the first attempt took a fixed third member and could not build the
      // four-element types at all.
      "#define WEFT_VEC2(T, S) \\\n"
      "  static inline T operator-(T a) { return T(-a.x, -a.y); } \\\n"
      "  static inline T operator+(T a, T b) { return T(a.x + b.x, a.y + b.y); } \\\n"
      "  static inline T operator-(T a, T b) { return T(a.x - b.x, a.y - b.y); } \\\n"
      "  static inline T operator*(T a, T b) { return T(a.x * b.x, a.y * b.y); } \\\n"
      "  static inline T operator/(T a, T b) { return T(a.x / b.x, a.y / b.y); } \\\n"
      "  static inline T operator*(T a, S s) { return T(a.x * s, a.y * s); } \\\n"
      "  static inline T operator*(S s, T a) { return T(a.x * s, a.y * s); } \\\n"
      "  static inline T operator/(T a, S s) { return T(a.x / s, a.y / s); }\n"
      "#define WEFT_VEC3(T, S) \\\n"
      "  static inline T operator-(T a) { return T(-a.x, -a.y, -a.z); } \\\n"
      "  static inline T operator+(T a, T b) { return T(a.x + b.x, a.y + b.y, a.z + b.z); } \\\n"
      "  static inline T operator-(T a, T b) { return T(a.x - b.x, a.y - b.y, a.z - b.z); } \\\n"
      "  static inline T operator*(T a, T b) { return T(a.x * b.x, a.y * b.y, a.z * b.z); } \\\n"
      "  static inline T operator/(T a, T b) { return T(a.x / b.x, a.y / b.y, a.z / b.z); } \\\n"
      "  static inline T operator*(T a, S s) { return T(a.x * s, a.y * s, a.z * s); } \\\n"
      "  static inline T operator*(S s, T a) { return T(a.x * s, a.y * s, a.z * s); } \\\n"
      "  static inline T operator/(T a, S s) { return T(a.x / s, a.y / s, a.z / s); }\n"
      "#define WEFT_VEC4(T, S) \\\n"
      "  static inline T operator-(T a) { return T(-a.x, -a.y, -a.z, -a.w); } \\\n"
      "  static inline T operator+(T a, T b) { return T(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); } \\\n"
      "  static inline T operator-(T a, T b) { return T(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); } \\\n"
      "  static inline T operator*(T a, T b) { return T(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); } \\\n"
      "  static inline T operator/(T a, T b) { return T(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w); } \\\n"
      "  static inline T operator*(T a, S s) { return T(a.x * s, a.y * s, a.z * s, a.w * s); } \\\n"
      "  static inline T operator*(S s, T a) { return T(a.x * s, a.y * s, a.z * s, a.w * s); } \\\n"
      "  static inline T operator/(T a, S s) { return T(a.x / s, a.y / s, a.z / s, a.w / s); }\n"
      "WEFT_VEC2(float2, float)\n"
      "WEFT_VEC3(float3, float)\n"
      "WEFT_VEC4(float4, float)\n"
      // The integer vectors with Metal's sizes: a 2-element to twice the element,
      // a 3- or 4-element to four times it, a short3 padded to 8 and a char3 to 4.
      "struct alignas(8) int2 { int32_t x = 0, y = 0;\n"
      "  int2() {}\n"
      "  int2(int32_t v) : x(v), y(v) {}\n"
      "  int2(int32_t a, int32_t b) : x(a), y(b) {}\n"
      "};\n"
      "struct alignas(16) int3 { int32_t x = 0, y = 0, z = 0, _pad = 0;\n"
      "  int3() {}\n"
      "  int3(int32_t v) : x(v), y(v), z(v) {}\n"
      "  int3(int32_t a, int32_t b, int32_t c) : x(a), y(b), z(c) {}\n"
      "};\n"
      "struct alignas(16) int4 { int32_t x = 0, y = 0, z = 0, w = 0;\n"
      "  int4() {}\n"
      "  int4(int32_t v) : x(v), y(v), z(v), w(v) {}\n"
      "  int4(int32_t a, int32_t b, int32_t c, int32_t d) : x(a), y(b), z(c), w(d) {}\n"
      "};\n"
      "struct alignas(8) uint2 { uint32_t x = 0, y = 0;\n"
      "  uint2() {}\n"
      "  uint2(uint32_t v) : x(v), y(v) {}\n"
      "  uint2(uint32_t a, uint32_t b) : x(a), y(b) {}\n"
      "};\n"
      "struct alignas(16) uint3 { uint32_t x = 0, y = 0, z = 0, _pad = 0;\n"
      "  uint3() {}\n"
      "  uint3(uint32_t v) : x(v), y(v), z(v) {}\n"
      "  uint3(uint32_t a, uint32_t b, uint32_t c) : x(a), y(b), z(c) {}\n"
      "};\n"
      "struct alignas(16) uint4 { uint32_t x = 0, y = 0, z = 0, w = 0;\n"
      "  uint4() {}\n"
      "  uint4(uint32_t v) : x(v), y(v), z(v), w(v) {}\n"
      "  uint4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) : x(a), y(b), z(c), w(d) {}\n"
      "};\n"
      "struct alignas(4) short2 { int16_t x = 0, y = 0; };\n"
      "struct alignas(8) short3 { int16_t x = 0, y = 0, z = 0, _pad = 0; };\n"
      "struct alignas(8) short4 { int16_t x = 0, y = 0, z = 0, w = 0; };\n"
      "struct alignas(4) ushort2 { uint16_t x = 0, y = 0; };\n"
      "struct alignas(8) ushort3 { uint16_t x = 0, y = 0, z = 0, _pad = 0; };\n"
      "struct alignas(8) ushort4 { uint16_t x = 0, y = 0, z = 0, w = 0; };\n"
      "struct alignas(4) char2 { int8_t x = 0, y = 0; };\n"
      "struct alignas(4) char3 { int8_t x = 0, y = 0, z = 0, _pad = 0; };\n"
      "struct alignas(4) char4 { int8_t x = 0, y = 0, z = 0, w = 0; };\n"
      "struct alignas(4) uchar2 { uint8_t x = 0, y = 0; };\n"
      "struct alignas(4) uchar3 { uint8_t x = 0, y = 0, z = 0, _pad = 0; };\n"
      "struct alignas(4) uchar4 { uint8_t x = 0, y = 0, z = 0, w = 0; };\n"
      "struct alignas(2) bool2 { bool x = false, y = false;\n"
      "  bool2() {}\n"
      "  bool2(bool a, bool b) : x(a), y(b) {}\n"
      "};\n"
      "struct alignas(4) bool3 { bool x = false, y = false, z = false, _pad = false;\n"
      "  bool3() {}\n"
      "  bool3(bool a, bool b, bool c) : x(a), y(b), z(c) {}\n"
      "};\n"
      "struct alignas(4) bool4 { bool x = false, y = false, z = false, w = false;\n"
      "  bool4() {}\n"
      "  bool4(bool a, bool b, bool c, bool d) : x(a), y(b), z(c), w(d) {}\n"
      "};\n"
      "WEFT_VEC2(half2, half)\n"
      "WEFT_VEC3(half3, half)\n"
      "WEFT_VEC4(half4, half)\n"
      "WEFT_VEC2(int2, int32_t)\n"
      "WEFT_VEC3(int3, int32_t)\n"
      "WEFT_VEC4(int4, int32_t)\n"
      "WEFT_VEC2(uint2, uint32_t)\n"
      "WEFT_VEC3(uint3, uint32_t)\n"
      "WEFT_VEC4(uint4, uint32_t)\n"
      "using std::min;\n"
      "using std::max;\n"
      "using std::abs;\n"
      "using std::sqrt;\n"
      "static inline float weft_f32(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }\n"
      "static inline double weft_f64(uint64_t b) { double d; std::memcpy(&d, &b, 8); return d; }\n"

      "template <class T, class U>\n"
      "static inline T weft_bitcast(U u) { T t; std::memcpy(&t, &u, sizeof(T)); return t; }\n"
      // A tensor-core fragment register holds two halves, low half first, which
      // is the order the hardware packs them in.
      // `weft_bits` comes from the portable library, which defines it for both
      // targets; defining it here as well was a redefinition the moment the
      // library was emitted into this text.
      // `brev.b32` as eight mask swaps rather than `__builtin_bitreverse32`,
      // which is clang's alone; gcc folds this back to `rbit` on aarch64.
      "static inline uint32_t weft_bitrev32(uint32_t v) {\n"
      "  v = ((v >> 1) & 0x55555555u) | ((v & 0x55555555u) << 1);\n"
      "  v = ((v >> 2) & 0x33333333u) | ((v & 0x33333333u) << 2);\n"
      "  v = ((v >> 4) & 0x0F0F0F0Fu) | ((v & 0x0F0F0F0Fu) << 4);\n"
      "  v = ((v >> 8) & 0x00FF00FFu) | ((v & 0x00FF00FFu) << 8);\n"
      "  return (v >> 16) | (v << 16);\n"
      "}\n"
      "static inline uint16_t weft_h16_bits(float x) {\n"
      "  const _Float16 h = (_Float16)x;\n"
      "  uint16_t u; std::memcpy(&u, &h, 2); return u;\n"
      "}\n"
      "static inline float weft_h16(uint32_t b, int i) {\n"
      "  const uint16_t u = (uint16_t)(b >> (16 * i));\n"
      "  _Float16 h; std::memcpy(&h, &u, 2); return (float)h;\n"
      "}\n"
      // cvt.rn.f16x2.f32: two floats to two halves, the first source in the high
      // half — the form LLVM's NVPTX backend emits when it packs a pair.
      "static inline uint32_t weft_f16x2(float hi, float lo) {\n"
      "  return (uint32_t)weft_h16_bits(lo) | ((uint32_t)weft_h16_bits(hi) << 16);\n"
      "}\n"
      // bf16 is the high half of an f32 with the low 16 bits rounded away: eight
      // mantissa bits, the same exponent range as f32.
      "static inline uint16_t weft_bf16_bits(float x) {\n"
      "  uint32_t u; std::memcpy(&u, &x, 4);\n"
      "  if ((u & 0x7F800000u) != 0x7F800000u) u += 0x7FFFu + ((u >> 16) & 1u);\n"
      "  return (uint16_t)(u >> 16);\n"
      "}\n"
      "static inline uint32_t weft_bf16x2(float hi, float lo) {\n"
      "  return (uint32_t)weft_bf16_bits(lo) | ((uint32_t)weft_bf16_bits(hi) << 16);\n"
      "}\n"
      "static inline float weft_bf16(uint32_t b, int i) {\n"
      "  const uint32_t bits = (uint32_t)(uint16_t)(b >> (16 * i)) << 16;\n"
      "  float f; std::memcpy(&f, &bits, 4); return f;\n"
      "}\n"
      // e4m3: 3 mantissa bits and a 4-bit exponent, biased 7, saturating at 448
      // rather than overflowing to an infinity — what `cvt.rn.satfinite.e4m3x2.f32`
      // produces, and what an fp8 `mma` operand holds.
      "static inline uint8_t weft_e4m3_bits(float x) {\n"
      "  uint32_t u; std::memcpy(&u, &x, 4);\n"
      "  const uint8_t sign = (u & 0x80000000u) ? 0x80u : 0u;\n"
      "  if ((u & 0x7FFFFFFFu) == 0u) return sign;\n"
      "  if (!std::isfinite(x)) return (uint8_t)(sign | 0x7Fu);\n"
      "  const int E = (int)((u >> 23) & 0xFFu) - 127 + 7;\n"
      "  if (E >= 16) return (uint8_t)(sign | 0x7Eu);\n"
      "  if (E <= 0) {\n"
      "    const double q = std::nearbyint(std::fabs((double)x) * 512.0);\n"
      "    if (q >= 8.0) return (uint8_t)(sign | 0x08u);\n"
      "    return (uint8_t)(sign | (uint8_t)q);\n"
      "  }\n"
      "  const uint32_t mant = u & 0x7FFFFFu;\n"
      "  uint32_t r = mant >> 20;\n"
      "  const uint32_t rem = mant & 0xFFFFFu;\n"
      "  if (rem > 0x80000u || (rem == 0x80000u && (r & 1u))) r++;\n"
      // A carry out of the mantissa is a carry into the exponent: OR would drop
      // it, because bit 3 of the exponent field is already set for E with its
      // low bit on. Rounding 500.0 up to 512 is exactly that case.
      "  const uint32_t out = ((uint32_t)E << 3) + r;\n"
      "  return (uint8_t)(sign | (uint8_t)(out >= 0x7Fu ? 0x7Eu : out));\n"
      "}\n"
      "static inline uint32_t weft_e4m3x2(float hi, float lo) {\n"
      "  return (uint32_t)weft_e4m3_bits(lo) | ((uint32_t)weft_e4m3_bits(hi) << 8);\n"
      "}\n"
      "static inline float weft_e4m3(uint32_t b, int i) {\n"
      "  const uint32_t v = (b >> (8 * i)) & 0xFFu;\n"
      "  const uint32_t mag = v & 7u, E = (v >> 3) & 0xFu;\n"
      // 2^(E-10) as an exponent field, so the scale is a bit pattern and not a
      // library call — the same arithmetic the Metal side writes, which is what
      // makes the two decodes bit-for-bit the same function.
      "  const uint32_t sb = (E + 117u) << 23;\n"
      "  float sc; std::memcpy(&sc, &sb, 4);\n"
      "  const float d = (E == 0) ? (float)mag / 512.0f : (float)(8u + mag) * sc;\n"
      "  return (float)((v & 0x80u) ? -d : d);\n"
      "}\n"
      // e5m2: 2 mantissa bits and a 5-bit exponent, with infinities and NaN — the
      // other fp8 format, and the one gradients are usually kept in.
      "static inline uint8_t weft_e5m2_bits(float x) {\n"
      "  uint32_t u; std::memcpy(&u, &x, 4);\n"
      "  const uint8_t sign = (u & 0x80000000u) ? 0x80u : 0u;\n"
      "  if ((u & 0x7FFFFFFFu) == 0u) return sign;\n"
      "  if (!std::isfinite(x)) return (uint8_t)(sign | 0x7Fu);\n"
      "  const int E = (int)((u >> 23) & 0xFFu) - 127 + 15;\n"
      "  if (E >= 31) return (uint8_t)(sign | 0x7Bu);\n"
      "  if (E <= 0) {\n"
      "    const double q = std::nearbyint(std::fabs((double)x) * 65536.0);\n"
      "    if (q >= 4.0) return (uint8_t)(sign | 0x04u);\n"
      "    return (uint8_t)(sign | (uint8_t)q);\n"
      "  }\n"
      "  const uint32_t mant = u & 0x7FFFFFu;\n"
      "  uint32_t r = mant >> 21;\n"
      "  const uint32_t rem = mant & 0x1FFFFFu;\n"
      "  if (rem > 0x100000u || (rem == 0x100000u && (r & 1u))) r++;\n"
      "  const uint32_t out = ((uint32_t)E << 2) + r;\n"
      "  return (uint8_t)(sign | (uint8_t)(out >= 0x7Cu ? 0x7Bu : out));\n"
      "}\n"
      "static inline uint32_t weft_e5m2x2(float hi, float lo) {\n"
      "  return (uint32_t)weft_e5m2_bits(lo) | ((uint32_t)weft_e5m2_bits(hi) << 8);\n"
      "}\n"
      // The same bytes with an integer accumulator. The sum of 32 products of
      // bytes is exact in s32, and converting to float would stop being the same
      // number past 2^24 — so the integer form reads the byte as an integer.
      "static inline int32_t weft_s8_i(uint32_t b, int i) {\n"
      "  return (int32_t)(int8_t)(uint8_t)((b >> (8 * i)) & 0xFFu);\n"
      "}\n"
      "static inline int32_t weft_u8_i(uint32_t b, int i) {\n"
      "  return (int32_t)((b >> (8 * i)) & 0xFFu);\n"
      "}\n"
      // The 8-bit integers are the same one-byte operand read a different way: a
      // signed byte sign-extends, an unsigned one does not. Both are exact in f32,
      // so an integer tile's products and the sum of 32 of them are exact.
      "static inline float weft_s8(uint32_t b, int i) {\n"
      "  return (float)(int)(int8_t)(uint8_t)((b >> (8 * i)) & 0xFFu);\n"
      "}\n"
      "static inline float weft_u8(uint32_t b, int i) {\n"
      "  return (float)((b >> (8 * i)) & 0xFFu);\n"
      "}\n"
      "static inline float weft_e5m2(uint32_t b, int i) {\n"
      "  const uint32_t v = (b >> (8 * i)) & 0xFFu;\n"
      "  const uint32_t mag = v & 3u, E = (v >> 2) & 0x1Fu;\n"
      "  if (E == 31u) return (mag == 0u) ? ((v & 0x80u) ? -INFINITY : INFINITY) : NAN;\n"
      "  const uint32_t sb = (E + 110u) << 23;\n"
      "  float sc; std::memcpy(&sc, &sb, 4);\n"
      "  const float d = (E == 0) ? (float)mag / 65536.0f : (float)(4u + mag) * sc;\n"
      "  return (float)((v & 0x80u) ? -d : d);\n"
      "}\n"
      // tf32 keeps 10 mantissa bits, rounded to nearest-even: an operand is
      // reduced to that width before it is multiplied, which is what the
      // instruction does rather than borrowing the extra precision.
      "static inline float weft_tf32(float x) {\n"
      "  uint32_t u; std::memcpy(&u, &x, 4);\n"
      "  if ((u & 0x7F800000u) == 0x7F800000u) return x;\n"
      "  u += 0x1000u + ((u >> 13) & 1u);\n"
      "  u &= 0xFFFFE000u;\n"
      "  float r; std::memcpy(&r, &u, 4); return r;\n"
      "}\n"
      // shf.l / shf.r: a funnel shift over the 64-bit pair {b:a} — `a` is the
      // low word and `b` the high one, and the result is the 32 bits of the
      // shifted pair on the side the direction names. .wrap takes the count
      // modulo 32, .clamp saturates it, and both keep the shift-by-32 case out
      // of C++'s undefined behaviour.
      "static inline uint32_t weft_shf_l(uint32_t a, uint32_t b, uint32_t c, int clamp) {\n"
      "  if (clamp) { if (c >= 32u) return a; } else c &= 31u;\n"
      "  return c ? ((b << c) | (a >> (32u - c))) : b;\n"
      "}\n"
      "static inline uint32_t weft_shf_r(uint32_t a, uint32_t b, uint32_t c, int clamp) {\n"
      "  if (clamp) { if (c >= 32u) return b; } else c &= 31u;\n"
      "  return c ? ((a >> c) | (b << (32u - c))) : a;\n"
      "}\n"
      // lop3's truth table: bit ((a<<2)|(b<<1)|c) of the immediate is the result
      // for that combination of input bits, applied to all 32 of them at once.
      "static inline uint32_t weft_lop3(uint32_t a, uint32_t b, uint32_t c, uint32_t lut) {\n"
      "  uint32_t out = 0;\n"
      "  for (uint32_t i = 0; i < 8u; i++) {\n"
      "    const uint32_t m = ((i & 4u) ? a : ~a) & ((i & 2u) ? b : ~b) & ((i & 1u) ? c : ~c);\n"
      "    if (lut & (1u << i)) out |= m;\n"
      "  }\n"
      "  return out;\n"
      "}\n"
      // prmt.b32's default .f4e mode: nibble i of the selector picks byte i of
      // the result out of the eight bytes of (a:b), and a nibble with its top
      // bit set returns the sign of the byte it names (0x00 or 0xFF).
      "static inline uint32_t weft_prmt(uint32_t a, uint32_t b, uint32_t c) {\n"
      "  const uint64_t src = (uint64_t)a | ((uint64_t)b << 32);\n"
      "  uint32_t out = 0;\n"
      "  for (int i = 0; i < 4; i++) {\n"
      "    const uint32_t sel = (c >> (4 * i)) & 0xFu;\n"
      "    const uint32_t byte = (uint32_t)((src >> ((sel & 7u) * 8u)) & 0xFFu);\n"
      "    out |= ((sel & 8u) ? ((byte & 0x80u) ? 0xFFu : 0u) : byte) << (8 * i);\n"
      "  }\n"
      "  return out;\n"
      "}\n"
      // mul.hi's high word, from four 32x32 multiplies. The same helper the Metal
      // preamble defines: the GPU cannot build a 128-bit multiply in a kernel
      // this size, so both targets spell the product the same way and the two
      // columns compute it with the same instructions.
      "static inline uint64_t weft_mulhi_u64(uint64_t a, uint64_t b) {\n"
      "  const uint64_t a0 = a & 0xFFFFFFFFu, a1 = a >> 32, b0 = b & 0xFFFFFFFFu, b1 = b >> 32;\n"
      "  const uint64_t p0 = a0 * b0, p1 = a0 * b1, p2 = a1 * b0, p3 = a1 * b1;\n"
      "  const uint64_t cy = ((p0 >> 32) + (p1 & 0xFFFFFFFFu) + (p2 & 0xFFFFFFFFu)) >> 32;\n"
      "  return p3 + (p1 >> 32) + (p2 >> 32) + cy;\n"
      "}\n"
      "static inline int64_t weft_mulhi_s64(int64_t a, int64_t b) {\n"
      "  uint64_t h = weft_mulhi_u64((uint64_t)a, (uint64_t)b);\n"
      "  if (a < 0) h -= (uint64_t)b;\n"
      "  if (b < 0) h -= (uint64_t)a;\n"
      "  return (int64_t)h;\n"
      "}\n"
      "\n" + std::string(kWSoftMath) +
      body_out + "static const weft::KernelDef weft_table[] = {\n";
  for (const auto& k : kernels) {
    code += "  {\"" + k.name + "\", nullptr, " + std::to_string(k.nargs) + ", " +
            (k.sync ? "true" : "false") + ", nullptr, " +
            (k.warp_only ? std::string("nullptr") : ("weft_k_" + k.name)) + ", " +
            (k.warp_only ? std::string("0") : ("sizeof(weft_st_" + k.name + ")")) +
            ", nullptr, " + (k.warp ? ("weft_w_" + k.name) : std::string("nullptr")) + ", " +
            (k.warp ? ("sizeof(weft_ws_" + k.name + ")") : std::string("0")) + ", " +
            (k.warp_only ? std::string("nullptr") : ("weft_g_" + k.name)) + "},\n";
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
  // itself — a run does not have to be told again on its own command line.
  for (const auto& w : widths) {
    if (w.first < 0) continue;
    for (auto& k : kernels) {
      if (k.buf_type.size() < (size_t)k.nbufs) k.buf_type.resize((size_t)k.nbufs);
      for (auto& t : k.buf_type)
        if (t.empty()) t = "0";  // unknown: the ABI reports 0 rather than a guess
      if ((size_t)w.first < k.buf_type.size())
        k.buf_type[(size_t)w.first] = std::to_string(w.second);  // digits: a byte count
    }
  }
  code += emit_module_abi(kernels);

  r.code = code;
  r.ok = true;
  r.kernels = kernels;
  return r;
}

TranslateResult translate_ptx(const std::string& raw,
                              const std::vector<std::pair<int, uint32_t>>& widths) {
  return translate_ptx_impl(raw, widths, false);
}

TranslateResult translate_ptx_gpu(const std::string& raw,
                                  const std::vector<std::pair<int, uint32_t>>& widths) {
  return translate_ptx_impl(raw, widths, true);
}
