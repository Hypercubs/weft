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

// A qualifier can carry an argument that the dot splitter keeps attached:
// `.scaled::n1::ue8m0` arrives as one part, so this one matches a prefix.
bool has_qual(const std::vector<std::string>& parts, const char* want) {
  const std::string w(want);
  for (const auto& p : parts)
    if (p.compare(0, w.size(), w) == 0) return true;
  return false;
}

std::string cpp_type(const std::string& ptx) {
  if (ptx == ".f32") return "float";
  if (ptx == ".f64") return "double";
  if (ptx == ".b32" || ptx == ".u32") return "uint32_t";
  if (ptx == ".s32") return "int32_t";
  if (ptx == ".b64" || ptx == ".u64") return "uint64_t";
  if (ptx == ".b128") return "weft_u128";
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
  // The special registers that carry no dot. PTX reserves these names, so a register a
  // module declares cannot shadow one, and the whole token is compared rather than the
  // word before a dot: `lanemask_eq` has an underscore, which the name scanner does not
  // read, and the index stripping below would otherwise turn it into a plain register.
  static const char* const kNoDot[] = {"laneid",  "warpid",      "nwarpid",
                                       "lanemask_eq", "lanemask_lt", "lanemask_le",
                                       "lanemask_gt", "lanemask_ge"};
  for (const char* name : kNoDot)
    if (tok.compare(1, std::string::npos, name) == 0) {
      r.special = name;
      r.ok = true;
      return r;
    }
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
  if (t == "weft_u128") return 16;
  if (t == "uint16_t" || t == "int16_t") return 2;
  if (t == "uint8_t" || t == "int8_t" || t == "bool") return 1;
  return 0;
}

struct Emitter {
  std::string err;
  bool uses_barrier = false;
  bool uses_shfl = false;  // counted separately: a shuffle does not tie warps together

  // One decode, two back ends. `msl` switches the few places where the same
  // instruction is worded differently for Metal, a branch, a barrier, a
  // shuffle, the address space of a pointer, and nothing else. The parsing, the
  // register model and the expression builders are shared, which is what makes
  // the GPU output comparable to the CPU output rather than a second
  // implementation that happens to agree.
  bool msl = false;
  // The kernel uses a 64-bit atomic, which this host has no instruction for. The
  // emission names a helper that builds one out of 32-bit atomics under a lock
  // (see weft_warp_msl.hpp), and only the warp back end carries that runtime, so
  // a kernel that holds one is emitted for the warp back end.
  bool wide_atom = false;
  // How many times each register's name appears in the body, its own declaration
  // aside. One appearance means nothing reads the value, which is what turns an
  // `atom` into the `red` of the same operation, the rewrite ptxas makes, and the
  // one that gives a 64-bit counter a GPU column at all.
  std::map<std::string, int> reg_reads;
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
  // And for `.cube`: six square faces, which on the GPU are `texturecube`, a
  // parameter type of its own, and one that cannot be read or written by texel.
  std::vector<bool> tex_cube;
  int shared_bytes = 0;      // the threadgroup arena this kernel needs, at compile time
  bool shared_extern = false;  // it was `extern .shared`: the launch chooses the size
  bool uses_red = false;     // a reduction barrier: needs its publish slots

  // Registers become r[reg][lane] and every instruction is emitted as a 32-lane
  // loop, so a shuffle is a gather inside the register file. The expression
  // builders are unchanged: they just address `[l]` instead of a scalar.
  bool vec = false;
  bool vec_ok = true;      // false once something here cannot be vectorised
  std::string vec_pred;    // per-lane predicate for the current instruction
  bool vec_control = false;  // emitted text is control flow: do not wrap it
  bool vec_side = false;     // emitted text already contains its lane loop
  bool vec_select = true;    // store() masks by lane; off inside a lane loop
  // The condition code: PTX's one flag that outlives the instruction that set it.
  // `add.cc` and `sub.cc` write it, `addc` and `subc` read it and may write it
  // again, and that chain is how a 64-bit or a 128-bit integer add is spelled. It is
  // a member of the per-thread state for the same reason a register is, because the
  // instruction that reads it is a different one. The vector back end keeps a
  // register per lane and has no place for a flag that is not one of them, so a
  // kernel holding a chain stays on the per-lane back end, where a thread has one.
  std::vector<std::string> cc_members;
  std::string cc_name;
  int cc_next = 0;
  // The member is declared here and reached the way a register is, through `s`, so the
  // statement that writes it and the one that reads it name the same slot.
  std::string cc_member() {
    if (cc_name.empty()) {
      cc_name = "weft_cc" + std::to_string(cc_next++);
      cc_members.push_back("uint32_t " + cc_name + ";");
    }
    return "s->" + cc_name;
  }
  int label_count = 1;                 // 0 is the entry
  std::map<std::string, int> label_ids;
  // `ts: .branchtargets L1, L2, L3;` names a jump table a `brx.idx` indexes, and
  // the label at the start of that line is what the instruction names.
  std::map<std::string, std::vector<std::string>> brxtab;

  int label_id(const std::string& name) {
    auto it = label_ids.find(name);
    if (it != label_ids.end()) return it->second;
    const int id = label_count++;
    label_ids[name] = id;
    return id;
  }
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
  // How many times the body writes each register. The model folds a register's
  // value into the text a memory instruction carries, so a register written twice
  // would be folded in with the number it held at the first write: a loop counter
  // then reads as a constant and every iteration stores to the first iteration's
  // address. Only a register with a single definition anywhere in the body is
  // followed, which is what a real kernel's address chain looks like.
  std::map<std::string, int> reg_defs;

  void scan_defs(const std::string& body) {
    size_t q = 0;
    while (q < body.size()) {
      size_t semi = body.find(';', q);
      if (semi == std::string::npos) semi = body.size();
      std::string s = trim(body.substr(q, semi - q));
      q = semi + 1;
      const size_t colon = s.find(':');
      if (colon != std::string::npos && colon > 0 &&
          s.substr(0, colon).find_first_of(" \t.") == std::string::npos)
        s = trim(s.substr(colon + 1));
      if (s.empty() || s[0] == '.') continue;
      if (s[0] == '@') {
        size_t i = 1;
        while (i < s.size() && !std::isspace((unsigned char)s[i])) i++;
        s = trim(s.substr(i));
      }
      const size_t sp = s.find_first_of(" \t");
      if (sp == std::string::npos) continue;
      std::string dst = trim(s.substr(sp + 1));
      const size_t comma = dst.find(',');
      if (comma != std::string::npos) dst = trim(dst.substr(0, comma));
      std::string list;
      if (dst.find('{') != std::string::npos) {
        list = dst.substr(dst.find('{') + 1);
        if (list.find('}') != std::string::npos) list = list.substr(0, list.find('}'));
      }
      for (const auto& t : split_top(list.empty() ? dst : list, ',')) {
        Reg r = parse_reg(trim(t));
        if (!r.ok || !r.special.empty()) continue;
        reg_defs[reg_key(r)]++;
      }
    }
  }

  // A `call.uni` names a `.func` in this module, and PTX gives a function its own
  // register file: the callee's `%rd1` is not the caller's. So the callee is
  // translated into a function of its own, with a state struct of its own, and the
  // call site passes the arguments by value. `callees` is the module's table of
  // functions that translated, `callee_errs` the reason for the ones that did not,
  // and `callee_args` maps a callee's parameter names to its own arguments, which
  // is what `ld.param` inside it reads.
  std::map<std::string, std::vector<std::string>> callees;
  std::map<std::string, std::string> callee_errs;
  std::map<std::string, std::string> callee_args;
  std::vector<std::pair<std::string, std::string>> callee_members;
  // A callee's answer is a field of its state struct rather than the generated
  // function's return type: the body is a switch over a program counter, so the `ret`
  // that ends it is a `return` from the middle of that switch, and a value has to be
  // somewhere a jump cannot drop. `callee_rets` is what the module knows about each
  // function it translated (empty for one that returns nothing), `callee_ret` is this
  // function's own parameter name for the slot it writes.
  std::map<std::string, std::string> callee_rets;
  std::string callee_ret;
  // A function that can give the worker back in the middle -- a rendezvous inside it,
  // or a call to one -- makes its call site a resume point, so the caller has to know
  // that before it writes the call.
  std::map<std::string, bool> callee_suspends;

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
    if (it == vec_vals.end()) return VecVal{};
    auto d = reg_defs.find(reg_key(r));
    if (d != reg_defs.end() && d->second > 1) return VecVal{};
    return it->second;
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
    // A predicated write leaves the register alone in the lanes that fail the
    // predicate, so what it holds afterwards is conditional. Folding that into an
    // address would be the same mistake as folding a loop counter: the value is
    // right for the lanes that ran and stale for the ones that did not.
    if (!vec_pred.empty()) {
      vec_vals[reg_key(d)] = VecVal{};
      return;
    }

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
  // or store does, `ld.global.f32` is a four-byte element, `ld.global.u8` one , 
  // and that is the only place the width exists. A parameter reached at two
  // different widths has no single answer and reports 0, which the ABI reads as
  // unknown rather than as four bytes.
  std::map<int, uint32_t> param_bytes;
  // The register holding a parameter's base address, through `ld.param`, `cvta`,
  // `mov` and `add`, the instructions every front end puts between a binding and
  // its first access.
  std::map<std::string, int> ptr_param;
  std::map<std::string, std::string> reg_types;  // "r" -> "uint32_t"
  // How many registers `%r<N>` holds. A body that names `%rd13` under `%rd<10>`
  // used to be generated anyway, and the field it addressed does not exist in the
  // state struct, the write landed in whatever followed it, which reads as a
  // wrong number rather than as a bad fixture. It is an error now.
  std::map<std::string, int> reg_counts;
  std::string decls;

  // PTX registers carry a declared type, but the instruction says how the bits
  // are meant right now: nvcc-era PTX loads floats with ld.global.b32 and then
  // does add.rn.f32 on the same register. So every operand is coerced to the
  // instruction's type, and the result is coerced back to the register's type.
  std::string operand_as(const std::string& tok, const std::string& want) {
    std::string t = trim(tok);
    if (t.empty()) return t;
    if (t[0] != '%') {
      if (t.size() == 10 && t.compare(0, 2, "0f") == 0) {
        // `0f` is a float constant, and its bits are the value when the operand is
        // read as an integer: `mov.u32 %r, 0f3F800000` moves those 32 bits rather
        // than converting 1.0 to 1. Converting by value turned every float immediate
        // in a `mov` into its integer part.
        if (want != "float") return "(uint32_t)0x" + t.substr(2) + "u";
        return "weft_f32(0x" + t.substr(2) + "u)";
      }
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
    // A 128-bit value is a struct, so the per-lane bitwise select that stands in for a
    // predicate has no operands to work on: assign on the lanes that ran and leave the
    // others alone, which is what the predicate asks for.
    if (r.ok) {
      auto wide = reg_types.find(r.base);
      if (wide != reg_types.end() && cpp_size(wide->second) > 8)
        return "if (am[l]) " + dst + " = " + core + ";";
    }
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
      // The lane masks are the warp's own bit patterns and nothing more: each is a shift
      // of one or two ones, which is what a warp intrinsic asks for when it wants the
      // lanes below or above it. They are written as complements of a shift rather than
      // shifts of the complement, so the lane at 31 never shifts by 32, which is not a
      // shift this language defines. `full` and `activemask` are not these: they depend on
      // which lanes are actually running, so they are still refused rather than guessed.
      {
        const std::string lane = vec ? "l" : std::string("(ctx.lid & 31u)");
        if (r.special == "lanemask_eq") return "(1u << " + lane + ")";
        if (r.special == "lanemask_lt") return "((1u << " + lane + ") - 1u)";
        if (r.special == "lanemask_le") return "((2u << " + lane + ") - 1u)";
        if (r.special == "lanemask_gt") return "~((2u << " + lane + ") - 1u)";
        if (r.special == "lanemask_ge") return "~((1u << " + lane + ") - 1u)";
      }
      // Which warp of the block this lane is in, and how many the block has: both are the
      // block's geometry, not the hardware's, so a 64-lane block has two and lanes 0..31
      // are in the first.
      if (r.special == "warpid") return vec ? "0u" : "(ctx.lid / 32u)";
      if (r.special == "nwarpid") return "((ctx.tg_size + 31u) / 32u)";
      err = "unsupported special register %" + r.special;
      return "0";
    }
    auto rc = reg_counts.find(r.base);
    // A register that is *never* declared used to be emitted as a field of the state
    // struct that does not exist -- `%fd1` became `s->fd[1]` -- so the module failed in
    // the C++ compiler rather than here, with an error that names generated code. The
    // name is what is wrong, so the name is what the report carries.
    if (rc == reg_counts.end()) {
      err = "register %" + r.base + std::to_string(r.index) + " is used but never declared";
      return "0";
    }
    if ((long)r.index >= rc->second) {
      err = "register %" + r.base + std::to_string(r.index) + " is outside the declared %" +
            r.base + "<" + std::to_string(rc->second) + ">";
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
      param_name = cut == std::string::npos ? s : trim(s.substr(0, cut));
      std::string off;
      if (cut != std::string::npos) {
        off = trim(s.substr(cut + 1));
        if (!off.empty() && off[0] == '%') off = reg_expr(off);
        if (s[cut] == '-') off = "(0 - " + off + ")";
      }
      // A shared array's address is its offset inside the threadgroup window, and
      // that is known here; a parameter's is the binding, which is resolved by the
      // access itself. Without the base every `.shared` array started at the window's
      // first byte, so the second array of a kernel wrote over the first.
      auto so = shared_offsets.find(param_name);
      if (so != shared_offsets.end()) {
        addr = std::to_string(so->second);
        if (!off.empty()) addr = "(" + addr + " + " + off + ")";
        return true;
      }
      addr = off.empty() ? "0" : ("(0 + " + off + ")");
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
  // is one contiguous run, which is every global access in a kernel that
  // indexes by global thread id, a full warp becomes a plain loop over that
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
  // has no double, and rint on the f32 itself is exact for the same reason, a
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
// tensor-core kernel the front end translates, the `mma` forms and the
// `ldmatrix` staging around them, gets a GPU column from here, and each one
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
//     head, where the masks are recomputed from the pc values, so the lanes
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
        // number it had, which for the prologue is 0, the pc every lane starts
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
        // condition said, which threw away the pc write and sent every lane back
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
    // dispatcher picks the same case again, an infinite loop that looks like a
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
    // `.shared`, `.shared::cta` and `.shared::cluster` are the same space to a launch
    // that is one CTA wide: the sub-qualifier picks which CTA's window the address is
    // taken in, and there is only one window here.
    if (p.rfind("shared", 0) == 0) shared = true;
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
    // A 128-bit move is the one value wider than the arithmetic types: `weft_u128` is
    // a pair of 64-bit halves, and the load and the store each read or write the whole
    // sixteen bytes in one access, which is what the isa promises. Nothing computes
    // with it, so it never meets the scalar paths. The register it names has to be
    // declared to hold sixteen bytes: a `.b64` name in that position is not a narrower
    // load, it is a module ptxas would reject, so it is refused by name here too.
    if (has_part(parts, "b128")) {
      const std::string& tok = (op == "ld") ? args[0] : args[1];
      Reg br = parse_reg(trim(tok));
      auto bt = br.ok ? reg_types.find(br.base) : reg_types.end();
      if (bt == reg_types.end() || cpp_size(bt->second) != 16) {
        err = "'" + opcode + "' moves sixteen bytes, which needs a .b128 register, not '" +
              trim(tok) + "'";
        return "";
      }
    }
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
          // A callee's own parameter is an argument the call site passed, not a
          // binding: `ld.param` inside a `.func` reads the value, not the buffer.
          auto ca = callee_args.find(name);
          if (ca != callee_args.end()) return "(" + ty + ")" + ca->second;
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
          const std::string qt =
              "const " + addr_space(shared ? "threadgroup" : "device", false) + ty_mem;
          return vec_mem_access(args[1], esz, plain, store(args[0], load_at(addr), ty), qt);
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
    if (param) {
      // `st.param.b32 [rv], %r;` is how a called function hands a value back: the
      // parameter is the return slot, and here that is a field of the state the caller
      // reads after the call. A store to any other parameter is not a move this side
      // has, and is refused by name rather than aimed at address zero.
      if (!callee_ret.empty() && name == callee_ret) {
        const std::string vty = (msl && ty == "double") ? "uint64_t" : ty;
        return guard + "s->weft_ret = " + operand_as(args[1], vty) + ";";
      }
      err = "a store to the parameter '" + name + "', which is not the function's return slot";
      return "";
    }
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
        // A batched store builds a pointer of its own, and the address space is part
        // of the type on this target: without it the emitted kernel does not compile.
        const std::string qt = addr_space(shared ? "threadgroup" : "device", false) + ty;
        return vec_mem_access(args[0], esz, "weft_p[l] = " + val + ";", store_at(addr, val), qt);
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
      // The halves are split at the source's own width: a `.b64` gives 32 bits a
      // half, a `.b32` (the f16x2 and packed-narrow words) 16.
      const bool wide = reg_type_of(args[1]) == "uint64_t";
      const std::string mask = wide ? "0xFFFFFFFFu" : "0xFFFFu";
      const std::string sh = wide ? "32" : "16";
      return guard + store(parts2[0], "(uint32_t)(" + reg_expr(args[1]) + " & " + mask + ")", "uint32_t") +
             " " + store(parts2[1], "(uint32_t)((" + reg_expr(args[1]) + ") >> " + sh + ")", "uint32_t");
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
  // The pair converts understand a packed narrow type as either end: the last type
  // in the modifier list is the destination, so `cvt.rn.f16x2.e2m1x2` unpacks two
  // narrow values into two halves and `cvt.rn.satfinite.e2m1x2.f16x2` packs two
  // halves into them. Either way it is one decode or one encode of the same lanes.
  {
    static const char* kPair[5] = {"e4m3x2", "e5m2x2", "e2m1x2", "e2m3x2", "e3m2x2"};
    auto pos_of = [&parts](const char* t) {
      for (size_t i = 0; i < parts.size(); i++)
        if (parts[i] == t) return (int)i;
      return -1;
    };
    int psel = -1, ppos = -1;
    for (int i = 0; i < 5; i++) {
      const int q = pos_of(kPair[i]);
      if (q > 0) { psel = i; ppos = q; }
    }
    int hpos = pos_of("f16x2");
    const int bpos = pos_of("bf16x2");
    const bool bf_half = hpos < 0 && bpos > 0;
    if (bf_half) hpos = bpos;
    if (op == "cvt" && hpos > 0 && psel >= 0 && !has_part(parts, "rs") && !has_part(parts, "x4")) {
      if (!need(2)) return "";
      static const char* kDec[5] = {"weft_e4m3(", "weft_e5m2(", "weft_e2m1(",
                                   "weft_e2m3(", "weft_e3m2("};
      static const char* kEnc[5] = {"weft_e4m3x2(", "weft_e5m2x2(", "weft_e2m1x2(",
                                   "weft_e2m3x2(", "weft_e3m2x2("};
      const bool relu = has_part(parts, "relu");
      auto clamp = [relu](const std::string& v) {
        return relu ? "(((" + v + ") < 0.0f) ? 0.0f : (" + v + "))" : v;
      };
      // `.pzo` asks for positive zeros only, and `-0.0 == 0.0` makes that fold a
      // comparison rather than a sign test.
      const bool pzo = has_part(parts, "pzo");
      auto pz = [pzo](const std::string& v) {
        return pzo ? "(((" + v + ") == 0.0f) ? 0.0f : (" + v + "))" : v;
      };
      // `.scaled` carries one or two ue8m0 factors: a single one divides both values
      // on the way down to a narrow destination, and a pair multiplies each on the way
      // up to a 16-bit one. The operand is optional and defaults to 0x7f, or 0x7f7f
      // for the pair, 1.0 either way.
      const bool scaled = has_qual(parts, "scaled");
      const std::string src = operand_as(args[1], "uint32_t");
      if (hpos < ppos) {
        // A packed source read out into two halves. Every one of these formats is
        // exact in a 16-bit float, so the conversion itself rounds nothing, only the
        // scale can, and a power of two only overflows.
        const std::string sc = scaled
            ? (args.size() > 2 ? operand_as(args[2], "uint32_t") : std::string("0x7F7Fu"))
            : std::string();
        auto up = [scaled, &sc](const std::string& v, int i) {
          return scaled ? "weft_scale_up(" + v + ", (" + sc + " >> " +
                              std::to_string(8 * i) + ") & 0xFFu)"
                        : v;
        };
        const std::string hi = pz(up(clamp(std::string(kDec[psel]) + src + ", 1)"), 1));
        const std::string lo = pz(up(clamp(std::string(kDec[psel]) + src + ", 0)"), 0));
        return store(args[0],
                     (bf_half ? "weft_bf16x2(" : "weft_f16x2(") + hi + ", " + lo + ")",
                     "uint32_t");
      }
      // Two halves packed into the narrow format.
      const std::string sc = scaled
          ? (args.size() > 2 ? operand_as(args[2], "uint32_t") : std::string("0x7Fu"))
          : std::string();
      auto down = [scaled, &sc](const std::string& v) {
        return scaled ? "weft_scale_down(" + v + ", " + sc + ")" : v;
      };
      const char* half = bf_half ? "weft_bf16(" : "weft_h16(";
      const std::string hi = pz(down(clamp(std::string(half) + src + ", 1)")));
      const std::string lo = pz(down(clamp(std::string(half) + src + ", 0)")));
      const char* wid = psel == 2 ? "uint8_t"
                        : (psel == 3 || psel == 4) ? "uint16_t"
                                                   : "uint32_t";
      if (!has_part(parts, "rz"))
        return store(args[0], std::string(kEnc[psel]) + hi + ", " + lo + ")", wid);
      // `.rz` truncates each value instead of rounding it, and the pack is the same.
      static const char* kEncRz[5] = {"weft_e4m3_bits_rz(", "weft_e5m2_bits_rz(",
                                      "weft_e2m1_rz(", "weft_e2m3_rz(",
                                      "weft_e3m2_rz("};
      const std::string bits = psel == 2 ? "4" : "8";
      return store(args[0], "weft_pack2(" + std::string(kEncRz[psel]) + hi + "), " +
                                std::string(kEncRz[psel]) + lo + "), " + bits + ")",
                   wid);
    }
  }

  // cvt.rs{.relu}{.satfinite}.{f16x2,bf16x2}.f32 d, a, b, rbits
  //
  // Stochastic rounding, which a quantization pass uses instead of nearest so the
  // error averages out instead of biasing. The ISA's figure splits the random word:
  // bits[12:0] below the low result and bits[28:16] below the high one for a half,
  // bits[15:0] and [31:16] for a bf16, as many bits as the destination drops, so the
  // carry is the plain integer addition the ISA describes.
  if (op == "cvt" && has_part(parts, "rs") &&
      (has_part(parts, "f16x2") || has_part(parts, "bf16x2"))) {
    if (!need(4)) return "";
    const bool bf = has_part(parts, "bf16x2");
    const std::string fn = bf ? "weft_bf16_bits_rs(" : "weft_h16_bits_rs(";
    const std::string mask = bf ? "0xFFFFu" : "0x1FFFu";
    const std::string satf = has_part(parts, "satfinite") ? "1" : "0";
    std::string a = operand_as(args[1], "float");
    std::string b = operand_as(args[2], "float");
    if (has_part(parts, "relu")) {
      a = "((" + a + " < 0.0f) ? 0.0f : (" + a + "))";
      b = "((" + b + " < 0.0f) ? 0.0f : (" + b + "))";
    }
    const std::string rb = operand_as(args[3], "uint32_t");
    const std::string hi = "((uint32_t)" + fn + a + ", (" + rb + " >> 16) & " + mask + ", " +
                           satf + ") << 16)";
    const std::string lo = "(uint32_t)" + fn + b + ", " + rb + " & " + mask + ", " + satf + ")";
    return store(args[0], "(" + lo + " | " + hi + ")", "uint32_t");
  }

  // cvt.rn.satfinite{.relu}.{e2m1x4,e2m3x4,e3m2x4}.f32 d, {a, b, e, f}
  //
  // The four-value forms: fp4 packs into a 16-bit destination and the f6 formats
  // into a 32-bit one, first source in the high field, the order an mma takes its
  // operand in, so a quantization pass writes the register it is about to feed.
  if (op == "cvt" && (has_part(parts, "e2m1x4") || has_part(parts, "e2m3x4") ||
                      has_part(parts, "e3m2x4") || has_part(parts, "e4m3x4") ||
                      has_part(parts, "e5m2x4"))) {
    if (args.size() < 2) { err = "'" + opcode + "' needs four sources"; return ""; }
    // The four sources come in braces, and the operand splitter keeps the brace that
    // opens them attached to the first one, so that operand is unwrapped and split on
    // its own, and anything after it (the random word of `.rs`) is appended as it
    // stands. An assembly that leaves the braces off reads the same way.
    std::string list = trim(args[1]);
    if (!list.empty() && list[0] == '{') list.erase(0, 1);
    if (!list.empty() && list[list.size() - 1] == '}') list.erase(list.size() - 1);
    std::vector<std::string> src = split_top(list, ',');
    for (size_t i = 2; i < args.size(); i++) src.push_back(trim(args[i]));
    const bool rs = has_part(parts, "rs");
    if (src.size() != (rs ? 5u : 4u)) {
      err = "'" + opcode + "' takes four sources" + (rs ? " and a random word" : "") +
            ", and this one names " + std::to_string(src.size());
      return "";
    }
    std::string v[4];
    for (int i = 0; i < 4; i++) {
      v[i] = operand_as(src[i], "float");
      if (has_part(parts, "relu")) v[i] = "((" + v[i] + " < 0.0f) ? 0.0f : (" + v[i] + "))";
    }
    const bool f4 = has_part(parts, "e2m1x4");
    // `.rs` reads the random word as its fifth source; a convert without it asks for
    // no carry at all, which is the same call with a word of zeros.
    const std::string rb = rs ? operand_as(src[4], "uint32_t") : std::string("0u");
    const std::string mode = rs ? "1" : "0";
    std::string call;
    if (has_part(parts, "e4m3x4") || has_part(parts, "e5m2x4")) {
      call = "weft_f8x4(" + v[0] + ", " + v[1] + ", " + v[2] + ", " + v[3] + ", " + rb +
             ", " + (has_part(parts, "e4m3x4") ? "3" : "4") + ", " + mode + ")";
    } else {
      const char* ebmb = has_part(parts, "e2m3x4") ? "2, 3"
                         : has_part(parts, "e3m2x4") ? "3, 2" : "2, 1";
      call = "weft_mx4(" + v[0] + ", " + v[1] + ", " + v[2] + ", " + v[3] + ", " + rb +
             ", " + ebmb + ", " + mode + ")";
    }
    return store(args[0], call, f4 ? "uint16_t" : "uint32_t");
  }

  // cvt.rn.{f16,bf16,e4m3,e5m2,e2m1,e2m3,e3m2}x2.f32 d, a, b: two floats to two
  // packed values, the first source in the high half (or for the f6 formats in the
  // high six bits of a byte). This is the form LLVM's NVPTX backend emits to pack a
  // pair, and it is not a scalar conversion, so it is handled before the table below.
  if (op == "cvt" && (has_part(parts, "f16x2") || has_part(parts, "bf16x2") ||
                      has_part(parts, "e4m3x2") || has_part(parts, "e5m2x2") ||
                      has_part(parts, "e2m1x2") || has_part(parts, "e2m3x2") ||
                      has_part(parts, "e3m2x2"))) {
    if (!need(3)) return "";
    const bool rz = has_part(parts, "rz");
    const bool pzo = has_part(parts, "pzo");
    // `.scaled::n1::ue8m0` divides both sources by one ue8m0 factor before the
    // conversion, and flushes a subnormal input to a zero of its own sign first. The
    // operand is optional and defaults to 0x7f, which is 1.0.
    const bool scaled = has_qual(parts, "scaled");
    const std::string sc = scaled
        ? (args.size() > 3 ? operand_as(args[3], "uint32_t") : std::string("0x7Fu"))
        : std::string();
    auto prep = [&](std::string v) {
      if (has_part(parts, "relu")) v = "((" + v + " < 0.0f) ? 0.0f : (" + v + "))";
      if (scaled) v = "weft_scale_down(" + v + ", " + sc + ")";
      if (pzo) v = "((" + v + " == 0.0f) ? 0.0f : (" + v + "))";
      return v;
    };
    const std::string a = prep(operand_as(args[1], "float"));
    const std::string b = prep(operand_as(args[2], "float"));
    // One encoder per value and a shift to pack them: the two 16-bit destinations take
    // a whole word, an e2m1 pair a nibble each and the rest a byte, and `.rz` picks
    // the truncating encoder for every one of them.
    const char* one;
    int bits;
    const char* wid;
    if (has_part(parts, "f16x2")) {
      one = rz ? "weft_h16_bits_rz(" : "weft_h16_bits(";
      bits = 16; wid = "uint32_t";
    } else if (has_part(parts, "bf16x2")) {
      one = rz ? "weft_bf16_bits_rz(" : "weft_bf16_bits(";
      bits = 16; wid = "uint32_t";
    } else if (has_part(parts, "e4m3x2")) {
      one = rz ? "weft_e4m3_bits_rz(" : "weft_e4m3_bits(";
      bits = 8; wid = "uint32_t";
    } else if (has_part(parts, "e5m2x2")) {
      one = rz ? "weft_e5m2_bits_rz(" : "weft_e5m2_bits(";
      bits = 8; wid = "uint32_t";
    } else if (has_part(parts, "e2m1x2")) {
      one = rz ? "weft_e2m1_rz(" : "weft_e2m1_bits(";
      bits = 4; wid = "uint8_t";
    } else if (has_part(parts, "e2m3x2")) {
      one = rz ? "weft_e2m3_rz(" : "weft_e2m3_bits(";
      bits = 8; wid = "uint16_t";
    } else {
      one = rz ? "weft_e3m2_rz(" : "weft_e3m2_bits(";
      bits = 8; wid = "uint16_t";
    }
    // `.satfinite` needs no step of its own: every one of these formats clamps to its
    // own range instead of overflowing, which is what the modifier asks for.
    const std::string hi = std::string(one) + a + ")";
    const std::string lo = std::string(one) + b + ")";
    return store(args[0],
                 "weft_pack2(" + hi + ", " + lo + ", " + std::to_string(bits) + ")", wid);
  }

  // add, sub, mul and fma over four packed narrow values, which a Blackwell tensor
  // path uses to keep an accumulator in fp8 or fp6: `add.rn.satfinite.e4m3x4.e2m3x4
  // d, a, c`. Four lanes at once, so the operand types come from the modifier list
  // in order, the destination first, then the sources.
  if (op == "add" || op == "sub" || op == "mul" || op == "fma") {
    struct Narrow { const char* name; int code; };
    static const Narrow kNarrow[7] = {{"e2m1x4", 0}, {"e2m3x4", 1}, {"e3m2x4", 2},
                                      {"e4m3x4", 3}, {"e5m2x4", 4}, {"ue8m0x4", 5},
                                      {"e2m1p4x4", 6}};
    std::vector<int> codes;
    for (const auto& p : parts)
      for (const auto& n : kNarrow)
        if (p == n.name) codes.push_back(n.code);
    if (!codes.empty()) {
      const bool fma = op == "fma";
      const size_t want = (op == "add" || op == "sub") ? 2u : 3u;
      if (codes.size() != want) {
        err = "'" + opcode + "' names its destination and " +
              std::to_string(want - 1) + " source types, and this one names " +
              std::to_string(codes.size());
        return "";
      }
      if (!need(fma ? 4u : 3u)) return "";
      const int dc = codes[0];
      if (dc != 3 && dc != 4) {
        err = "'" + opcode + "' writes .e4m3x4 or .e5m2x4 and this one writes " +
              parts.back() + ": the isa's destination table holds only those two";
        return "";
      }
      const std::string sat = has_part(parts, "satfinite") ? "1" : "0";
      const std::string A = operand_as(args[1], "uint32_t");
      if (op == "add" || op == "sub") {
        const std::string C = operand_as(args[2], "uint32_t");
        return store(args[0],
                     "weft_mx4_op(" + A + ", 0u, " + C + ", " + std::to_string(codes[1]) +
                         ", " + std::to_string(codes[1]) + ", " + std::to_string(codes[1]) +
                         ", " + std::to_string(dc) + ", " + (op == "add" ? "0" : "1") + ", " +
                         sat + ")",
                     "uint32_t");
      }
      if (op == "mul") {
        const std::string C = operand_as(args[2], "uint32_t");
        return store(args[0],
                     "weft_mx4_op(" + A + ", " + C + ", " + C + ", " +
                         std::to_string(codes[1]) + ", " + std::to_string(codes[2]) + ", " +
                         std::to_string(codes[2]) + ", " + std::to_string(dc) + ", 2, " + sat +
                         ")",
                     "uint32_t");
      }
      const std::string B = operand_as(args[2], "uint32_t");
      const std::string C = operand_as(args[3], "uint32_t");
      return store(args[0],
                   "weft_mx4_op(" + A + ", " + B + ", " + C + ", " +
                       std::to_string(codes[1]) + ", " + std::to_string(codes[2]) + ", " +
                       std::to_string(dc) + ", " + std::to_string(dc) + ", 3, " + sat + ")",
                   "uint32_t");
    }
  }

  // cvt[.rn|.rz|.rm|.rp][.ftz].dst.src d, a
  //
  // The destination type is the first type in the modifier list and the source
  // is the last, because a rounding mode sits between them. Reading only
  // parts[1] made every `cvt.rn.f32.s32` an error, and rounding-mode converts
  // are what `__f2i_rn` and every float-to-int idiom in CUDA compile to.
  if (op == "cvt") {
    if (!need(2)) return "";
    std::string dstty, mode;
    // `.rni` and `.rn` are two different instructions, not two spellings: the
    // three-letter form rounds the *value* to an integer and keeps the float type
    // (`rint`), and the two-letter form rounds the significand to the destination's
    // precision. Treating the first as the second turns 2.5 into 2.
    bool integer_round = false;
    // A 16-bit float destination: `.f16` and `.bf16` are not register types in the
    // map, a `b16` register holds their bits, so the width is read here.
    std::string narrow;
    // A 16-bit float read as a *source*: its bits sit in a b16 register, so the
    // conversion starts by widening them back to a float, the exact inverse of what
    // the encode does on the way down, and the form `__half2float` compiles to.
    // cvt.pack.sat.<narrow>.<wide>.b32 d, a, b, c;
    //
    // A different instruction that shares the opcode, so it is answered before the
    // conversion modifiers are read. Both sources are saturated to the narrow type
    // and packed into d; the eight- and sixteen-bit forms differ in how much of c
    // survives, because two narrow values fill the low half and leave the high half
    // of d alone.
    if (opcode.find("pack") != std::string::npos) {
      if (!need(4)) return "";
      std::string narrow, wide;
      for (const auto& t : parts) {
        if (t == "u8" || t == "s8" || t == "u16" || t == "s16") narrow = t;
        else if (t == "s32" || t == "u32") wide = t;
      }
      if (narrow.empty() || wide.empty()) {
        err = "'" + opcode + "' does not name the narrow type it packs into";
        return "";
      }
      if (opcode.find(".sat") == std::string::npos) {
        err = "'" + opcode + "' without .sat is a different instruction, and the only "
              "form here is the saturating pack";
        return "";
      }
      auto pclamp = [&](const std::string& v, const std::string& lo, const std::string& hi) {
        return math("min", math("max", "(int32_t)(" + v + "), " + lo) + ", " + hi);
      };
      const std::string a = operand_as(args[1], "uint32_t");
      const std::string b = operand_as(args[2], "uint32_t");
      const std::string c = operand_as(args[3], "uint32_t");
      const bool low = narrow[0] == 'u';
      const bool small = narrow == "u8" || narrow == "s8";
      const std::string lo_v = low ? "0" : (small ? "-128" : "-32768");
      const std::string hi_v = low ? (small ? "255" : "65535") : (small ? "127" : "32767");
      std::string e;
      if (small) {
        e = "((" + c + ") & 0xFFFF0000u) | ((uint32_t)(" + pclamp(a, lo_v, hi_v) +
            ") & 0xFFu) | (((uint32_t)(" + pclamp(b, lo_v, hi_v) + ") & 0xFFu) << 8)";
      } else {
        e = "((uint32_t)(" + pclamp(a, lo_v, hi_v) + ") & 0xFFFFu) | (((uint32_t)(" +
            pclamp(b, lo_v, hi_v) + ") & 0xFFFFu) << 16)";
      }
      return guard + store(args[0], e, "uint32_t");
    }

    std::string src_half;
    for (size_t i = 1; i < parts.size(); i++) {
      const std::string t = cpp_type("." + parts[i]);
      if (!t.empty()) {
        if (dstty.empty()) dstty = t;
      } else if (parts[i] == "f16" || parts[i] == "bf16") {
        // The first 16-bit float in the list is the destination; one after a
        // destination is that same type read as a source.
        if (!dstty.empty()) {
          ty = "uint16_t";
          src_half = parts[i] == "f16" ? "weft_h16(" : "weft_bf16(";
          continue;
        }
        narrow = parts[i] == "f16" ? "half" : "bf16";
        dstty = "uint16_t";
      } else if (parts[i] == "rna") {
        // Three letters, but not one of the `.rni` family: `.rna` rounds the
        // significand away from zero, and it only exists for tf32.
        mode = "rna";
      } else if ((parts[i] == "rn" || parts[i] == "rz" || parts[i] == "rm" ||
                  parts[i] == "rp" || parts[i] == "rni" || parts[i] == "rzi" ||
                  parts[i] == "rmi" || parts[i] == "rpi")) {
        // PTX spells these .rni/.rzi/.rmi/.rpi; the trailing `i` is what makes
        // them round to an integer rather than to the destination's precision.
        mode = parts[i].substr(0, 2);
        integer_round = parts[i].size() == 3;
      } else if (parts[i] == "ftz" || parts[i] == "sat" || parts[i] == "tf32" ||
                 ((parts[i] == "satfinite" || parts[i] == "relu") &&
                  (has_part(parts, "tf32") || has_part(parts, "f16") ||
                   has_part(parts, "bf16")))) {
        // `.satfinite` and `.relu` are only accepted where the rounding is emitted:
        // the tf32 path and the 16-bit one both apply them, and anywhere else they
        // would be dropped silently, which is a wrong answer rather than a refusal.
        // Renormalising a denormal and clamping to the destination range are
        // both what the CPU does anyway for the conversions reached here. `tf32`
        // is a destination width rather than a register type, and its two
        // modifiers are read where the rounding is emitted.
      } else {
        err = "unsupported conversion modifier in '" + opcode + "'";
        return "";
      }
    }
    if (dstty.empty()) { err = "unsupported conversion '" + opcode + "'"; return ""; }
    // `cvt.{rna|rn|rz}{.satfinite}{.relu}.tf32.f32 d, a`: ten mantissa bits, and the
    // value a multiply sees is the rounded one. The grammar puts `.relu` on the
    // rn and rz forms only, so it is refused where the ISA does not name it.
    if (has_part(parts, "tf32")) {
      const bool satf = has_part(parts, "satfinite");
      const bool relu = has_part(parts, "relu");
      if (relu && mode == "rna") {
        err = "'cvt.rna.tf32.f32' has no .relu form: the isa gives it .satfinite only";
        return "";
      }
      const int m = mode == "rna" ? 1 : (mode == "rz" ? 2 : 0);
      return guard + store(args[0],
                           "weft_tf32_bits(" + operand_as(args[1], "uint32_t") + ", " +
                               std::to_string(m) + ", " + (satf ? "1" : "0") + ", " +
                               (relu ? "1" : "0") + ")",
                           "uint32_t");
    }
    // `cvt.rn{.relu}{.satfinite}.f16.f32` and its bf16 twin, what `__float2half`
    // and its bf16 sibling compile to. `.rn` is the mode, because the 16-bit
    // conversion both engines already have rounds to nearest even; `.rz` would need
    // a truncation neither carries, so it says so rather than rounding instead.
    if (!narrow.empty()) {
      const bool to_zero = mode == "rz";
      const bool satf = has_part(parts, "satfinite");
      const bool relu = has_part(parts, "relu");
      // A double source is bits on the GPU target and a number on the CPU one, so it is
      // read rather than cast: `(float)` of a `uint64_t` would convert the pattern.
      std::string v;
      if (!src_half.empty()) v = src_half + operand_as(args[1], "uint16_t") + ", 0)";
      else if (has_part(parts, "f64"))
        v = msl ? "weft_bitcast<float>(weft_fp64::to_f32(" + operand_as(args[1], "uint64_t") + "))"
                : "(float)" + operand_as(args[1], "double");
      else v = operand_as(args[1], "float");
      if (relu) v = "((" + v + " < 0.0f) ? 0.0f : (" + v + "))";
      // `.sat` is the [0, 1] clamp with a NaN flushed to +0; `.satfinite` is the
      // destination's largest finite value. Two modifiers, two answers.
      if (has_part(parts, "sat")) {
        v = "((" + v + " != " + v + ") ? 0.0f : ((" + v + " < 0.0f) ? 0.0f : ((" + v +
            " > 1.0f) ? 1.0f : (" + v + "))))";
      }
      if (satf) {
        // `.satfinite` clamps to the largest finite value of the *destination* and
        // answers zero for a NaN: a bf16 has a float's exponent range but eight
        // significand bits, so the clamp is not the float's.
        const std::string mx =
            narrow == "half" ? "weft_h16(0x7BFFu, 0)" : "weft_bf16(0x7F7Fu, 0)";
        v = "((" + v + " != " + v + ") ? 0.0f : ((" + v + " > " + mx + ") ? " + mx +
            " : ((" + v + " < -" + mx + ") ? -" + mx + " : (" + v + "))))";
      }
      return guard + store(args[0],
                           narrow == "half"
                               ? (to_zero ? ("weft_h16_bits_rz(" + v + ")")
                                          : ("(uint16_t)weft_h16_bits(" + v + ")"))
                               : (to_zero ? ("weft_bf16_bits_rz(" + v + ")")
                                          : ("weft_bf16_bits(" + v + ")")),
                           "uint16_t");
    }
    // `cvt.f32.f16` and `cvt.f32.bf16`: the 16-bit float read back as a float, which
    // is what `__half2float` compiles to and what a half-precision kernel does before
    // it multiplies anything. The bits sit in a `b16` register, so this is the exact
    // inverse of the encode above: a widening, not a cast of the pattern to a number.
    // Every finite value of both formats is exact in binary32 except the range, which
    // is why the answer is the host's own number rather than an approximation.
    if (!src_half.empty() && dstty == "float")
      return guard + store(args[0],
                           src_half + operand_as(args[1], "uint16_t") + ", 0)", "float");

    // A conversion to or from a 64-bit float on the GPU target goes through the
    // software double: the value is bits, so a cast would convert the pattern. `.sat`
    // on a double destination is the [0, 1] clamp the float form is, its own
    // comparisons, so a NaN becomes positive zero and -0.0 keeps its sign.
    if (msl && (dstty == "double" || ty == "double")) {
      const std::string b = operand_as(args[1], "uint64_t");
      auto dstore = [&](std::string v) {
        if (dstty == "double" && has_part(parts, "sat"))
          v = "weft_fp64::sat_bits(" + v + ")";
        return guard + store(args[0], v, "uint64_t");
      };
      if (dstty == "double") {
        if (ty == "double") return dstore(b);
        if (ty == "float" || !src_half.empty())
          return dstore("weft_fp64::from_f32(weft_bitcast<uint32_t>(" +
                        (src_half.empty()
                             ? operand_as(args[1], "float")
                             : src_half + operand_as(args[1], "uint16_t") + ", 0)") +
                        "))");
        const bool sgn = ty == "int32_t" || ty == "int64_t";
        const std::string wide = sgn ? ("(int64_t)" + operand_as(args[1], "int64_t"))
                                     : ("(uint64_t)" + operand_as(args[1], "uint64_t"));
        return dstore(std::string("weft_fp64::from_") + (sgn ? "i64(" : "u64(") + wide + ")");
      }
      // A double read as a float: the narrowing *is* the rounding, and `to_f32`
      // rounds the significand to nearest even. Rounding to an integer first would
      // be a different number, 2.5 would come back as 2, so the mode is not
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
      // A 32-bit destination clamps to its own range: the 64-bit answer narrowed to 32
      // bits wraps, and 1e20 came back as 0x63100000 instead of INT32_MAX.
      const bool narrow = dstty == "int32_t" || dstty == "uint32_t";
      const std::string c = std::string("weft_fp64::to_") + (sgn ? "i" : "u") +
                            (narrow ? "32(" : "64(") + v + ")";
      return guard + store(args[0], "(" + dstty + ")" + c, dstty);
    }
    // The source is coerced to the *instruction's* type, not the register's:
    // `cvt.rzi.s32.f32` on a `.b32` register holds a float in it, and reading
    // those bits as an integer converts the bit pattern rather than the value.
    std::string src = operand_as(args[1], ty);
    if (!src_half.empty()) src = src_half + src + ", 0)";
    const bool dst_f = dstty == "float" || dstty == "double";
    const bool src_f = ty == "float" || ty == "double" || !src_half.empty();
    // A conversion to an integer clamps: the destination's ends for a value past
    // them, zero for a NaN. Anything else keeps the cast it had.
    auto f2i = [&](const std::string& v) -> std::string {
      if (dstty == "int32_t") return "weft_f2i32(" + v + ")";
      if (dstty == "uint32_t") return "weft_f2u32(" + v + ")";
      if (dstty == "int64_t") return "weft_f2i64(" + v + ")";
      if (dstty == "uint64_t") return "weft_f2u64(" + v + ")";
      return "(" + dstty + ")(" + v + ")";
    };
    // Rounding only exists between floats, and between a float and an integer.
    // Everything else is a truncation and a cast.
    std::string e;
    // Rounding is a step between two floats. An integer source has no fraction
    // to round off, and on the GPU `rint` of an unsigned int is ambiguous
    // between the half and float overloads, so the conversion is a plain cast.
    if (!mode.empty() && dst_f && src_f) {
      if (integer_round) {
        // `.rni` rounds the value to an integer and keeps the float type, which is
        // `rint`, what a compiler emits before narrowing to a half.
        const char* fn = mode == "rn" ? "nearbyint"
                       : mode == "rz" ? "trunc"
                       : mode == "rm" ? "floor"
                                       : "ceil";
        e = "(" + dstty + ")(" + rounded_math(fn, src) + ")";
      } else if (mode == "rn") {
        // Between two floats, `.rn` is what a cast already does: it rounds the
        // significand to the destination's precision, ties to even. Rounding to an
        // integer first would be a different number, `cvt.rn.f32.f64` of
        // 1234.5678 came out as 1235.
        e = "(" + dstty + ")(" + src + ")";
      } else {
        const char* fn = mode == "rz" ? "trunc" : (mode == "rm" ? "floor" : "ceil");
        e = "(" + dstty + ")(" + rounded_math(fn, src) + ")";
      }
    } else if (!mode.empty() && src_f) {
      // Float to integer: the device clamps an out-of-range value to the
      // destination's ends and answers zero for a NaN (the `.NTZ` bit every `F2I`
      // carries), so the conversion is a helper rather than a cast. `(int)` of an
      // infinity is undefined in C++ and answered 0, where the device answers
      // INT_MAX. `.rz` truncates and the other three round before the clamp, so
      // rounding first and clamping once is the same order the hardware uses.
      if (mode == "rz") e = f2i(src);
      else {
        const char* fn = mode == "rn" ? "nearbyint" : (mode == "rm" ? "floor" : "ceil");
        e = f2i(rounded_math(fn, src));
      }
    } else if (src_f) {
      e = f2i(src);
    } else {
      e = "(" + dstty + ")(" + src + ")";
    }
    // `.sat` on a floating-point destination is not the destination's range: the
    // ISA states it as [0, 1] with a NaN flushed to positive zero. Skipping the
    // modifier answered 2.0 for `cvt.sat.f32.f32` of 2.0.
    if (has_part(parts, "sat") && (dstty == "float" || dstty == "double")) {
      e = "((" + e + ") != (" + e + ") ? 0.0f : ((" + e + ") < 0.0f ? 0.0f : ((" + e +
          ") > 1.0f ? 1.0f : (" + e + "))))";
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
    // A narrow float's word order is not its value order: the bits of -1 sit above
    // the bits of 1, so the word compare this block would otherwise make calls -0.5
    // greater than 0.25. Widening to float is exact, so compare the widened pair.
    if (has_part(parts, "f16x2") || has_part(parts, "bf16x2")) {
      // `setp.lt.f16x2 p|q, a, b` answers one predicate per half, and the pair is
      // written as two destinations joined by a bar.
      if (!sym || args.size() < 3) {
        err = "'" + opcode + "' is not a packed comparison of two halves";
        return "";
      }
      const std::string dsts = trim(args[0]);
      const size_t bar = dsts.find('|');
      if (bar == std::string::npos) {
        err = "'" + opcode + "' compares two packed halves and answers two predicates, "
              "which have to be named as p|q";
        return "";
      }
      const std::string dl = dsts.substr(0, bar);
      const std::string dh = dsts.substr(bar + 1);
      const std::string pre = has_part(parts, "bf16x2") ? "weft_bf16_" : "weft_f16_";
      const std::string A = operand_as(args[1], "uint32_t");
      const std::string B = operand_as(args[2], "uint32_t");
      const std::string la = "(float)" + pre + "val((uint16_t)" + A + ")";
      const std::string lb = "(float)" + pre + "val((uint16_t)" + B + ")";
      const std::string ha = "(float)" + pre + "val((uint16_t)(" + A + " >> 16))";
      const std::string hb = "(float)" + pre + "val((uint16_t)(" + B + " >> 16))";
      return guard + store(dl, "(" + la + " " + sym + " " + lb + ")", "bool") + " " +
             store(dh, "(" + ha + " " + sym + " " + hb + ")", "bool");
    }
    if (has_part(parts, "f16") || has_part(parts, "bf16")) {
      if (!need(3)) return "";
      const std::string pre = has_part(parts, "bf16") ? "weft_bf16_" : "weft_f16_";
      const std::string va = "((float)" + pre + "val(" + operand_as(args[1], "uint16_t") + "))";
      const std::string vb = "((float)" + pre + "val(" + operand_as(args[2], "uint16_t") + "))";
      if (cmp == "nan" || cmp == "num") {
        const std::string any = "(" + va + " != " + va + " || " + vb + " != " + vb + ")";
        return guard + store(args[0], cmp == "nan" ? any : ("(!" + any + ")"), "bool");
      }
      if (!sym) {
        err = "'" + opcode + "' is not a comparison of two halves: ." + cmp +
              " orders unsigned integers";
        return "";
      }
      std::string both = "(" + va + " " + sym + " " + vb + ")";
      // `setp.lt.and p|q, a, b, c` folds in a second predicate; the fold is the same
      // operator whichever type the comparison was on.
      for (const auto& p : parts) {
        if (p != "and" && p != "or" && p != "xor") continue;
        if (args.size() < 4) {
          err = "'" + opcode + "' names ." + p + " without the predicate it combines with";
          return "";
        }
        const char* bs = p == "and" ? " && " : (p == "or" ? " || " : " ^ ");
        both = both + bs + operand_as(args[3], "bool");
      }
      return guard + store(args[0], both, "bool");
    }
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

  // `set` is the mask answer to the same comparison `setp` turns into a predicate:
  // every bit of the destination set for true, and 1.0 for a float one.
  if (op == "set") {
    if (!need(3)) return "";
    const std::string cmp = parts.size() > 1 ? parts[1] : "";
    const char* sym = cmp == "eq"   ? "=="
                      : cmp == "ne" ? "!="
                      : cmp == "lt" ? "<"
                      : cmp == "le" ? "<="
                      : cmp == "gt" ? ">"
                      : cmp == "ge" ? ">="
                      : cmp == "lo" ? "<"
                      : cmp == "ls" ? "<="
                      : cmp == "hi" ? ">"
                      : cmp == "hs" ? ">="
                                    : nullptr;
    std::string dty = reg_type_of(args[0]);
    if (dty.empty()) dty = "uint32_t";
    const std::string on = dty == "float"    ? "1.0f"
                           : dty == "double" ? "1.0"
                           : cpp_size(dty) == 8 ? "0xFFFFFFFFFFFFFFFFull"
                           : cpp_size(dty) == 2 ? "(uint16_t)0xFFFFu"
                                                : "0xFFFFFFFFu";
    std::string cond;
    if (has_part(parts, "f16x2") || has_part(parts, "bf16x2")) {
      // A packed comparison answers a mask per half in one word, which is the same
      // widened compare run twice: the low answer in the low half of the result.
      if (!sym) {
        err = "'" + opcode + "' is not a packed comparison of two halves: ." + cmp +
              " orders unsigned integers";
        return "";
      }
      const std::string pre2 = has_part(parts, "bf16x2") ? "weft_bf16_" : "weft_f16_";
      const std::string A = operand_as(args[1], "uint32_t");
      const std::string B = operand_as(args[2], "uint32_t");
      const std::string la = "(float)" + pre2 + "val((uint16_t)" + A + ")";
      const std::string lb = "(float)" + pre2 + "val((uint16_t)" + B + ")";
      const std::string ha = "(float)" + pre2 + "val((uint16_t)(" + A + " >> 16))";
      const std::string hb = "(float)" + pre2 + "val((uint16_t)(" + B + " >> 16))";
      const std::string m = "((" + la + " " + sym + " " + lb + ") ? 0xFFFFu : 0u) | ((" + ha +
                            " " + sym + " " + hb + ") ? 0xFFFF0000u : 0u)";
      return guard + store(args[0], m, dty);
    }
    if (has_part(parts, "f16") || has_part(parts, "bf16")) {
      // The same widening `setp` does: a narrow float's words are not ordered like
      // its values.
      const std::string pre = has_part(parts, "bf16") ? "weft_bf16_" : "weft_f16_";
      const std::string va = "((float)" + pre + "val(" + operand_as(args[1], "uint16_t") + "))";
      const std::string vb = "((float)" + pre + "val(" + operand_as(args[2], "uint16_t") + "))";
      if (cmp == "nan" || cmp == "num") {
        const std::string any = "(" + va + " != " + va + " || " + vb + " != " + vb + ")";
        cond = cmp == "nan" ? any : ("(!" + any + ")");
      } else if (sym) {
        cond = "(" + va + " " + sym + " " + vb + ")";
      }
    } else if (cmp == "nan" || cmp == "num") {
      if (ty != "float" && ty != "double") {
        err = "'" + opcode + "' tests for a NaN, which an integer is not";
        return "";
      }
      if (msl && ty == "double") {
        const std::string a = operand_as(args[1], "uint64_t"), b = operand_as(args[2], "uint64_t");
        const std::string any =
            "(weft_fp64::is_nan(" + a + ") || weft_fp64::is_nan(" + b + "))";
        cond = cmp == "nan" ? any : ("(!" + any + ")");
      } else {
        const std::string a = operand_as(args[1], ty), b = operand_as(args[2], ty);
        const std::string any = "(" + a + " != " + a + " || " + b + " != " + b + ")";
        cond = cmp == "nan" ? any : ("(!" + any + ")");
      }
    } else if (sym) {
      if (msl && ty == "double") {
        const std::string a = operand_as(args[1], "uint64_t"), b = operand_as(args[2], "uint64_t");
        cond = cmp == "eq"   ? "weft_fp64::eq(" + a + ", " + b + ")"
               : cmp == "ne" ? "(!weft_fp64::eq(" + a + ", " + b + "))"
               : cmp == "lt" ? "weft_fp64::lt(" + a + ", " + b + ")"
               : cmp == "le" ? "weft_fp64::le(" + a + ", " + b + ")"
               : cmp == "gt" ? "weft_fp64::gt(" + a + ", " + b + ")"
                             : "weft_fp64::ge(" + a + ", " + b + ")";
      } else {
        cond = "(" + operand_as(args[1], ty) + " " + sym + " " + operand_as(args[2], ty) + ")";
      }
    }
    if (cond.empty()) {
      err = "unsupported comparison '" + opcode + "'";
      return "";
    }
    return guard + store(args[0], "(" + cond + " ? " + on + " : 0)", dty);
  }

  // `slct.dtype.s32 d, a, b, c` is a select on the sign of c: a when c is not
  // negative, b when it is. The comparison is in c's own width, which is why the
  // `.f32` spelling compares against zero as a float. A NaN c is not negative, so
  // it takes b, which is what `>= 0` answers.
  if (op == "slct") {
    if (!need(4)) return "";
    // The sign of c is the whole test, and the register holding it is a `.b32`,
    // which carries no sign: read as unsigned, -5 is larger than zero and the
    // select picks the wrong argument. So the type suffix names how c is compared.
    const bool isf = has_part(parts, "f32") || has_part(parts, "f64");
    const std::string cty = isf ? (has_part(parts, "f64") ? "double" : "float")
                                : (has_part(parts, "u32") ? "uint32_t" : "int32_t");
    return guard + store(args[0],
                         "((" + operand_as(args[3], cty) + ") >= " + (isf ? "0.0" : "0") +
                             " ? " + operand_as(args[1], ty) + " : " +
                             operand_as(args[2], ty) + ")",
                         ty);
  }

  // `brx.idx a, tlist`: branch to the entry the index picks out of the table the
  // named `.branchtargets` directive lists. The index comes first and the table
  // second (llc writes `brx.idx %r1, $L_brx_0`), and the table is a label rather
  // than a value, so it is looked up among the lists parsed beside the labels.
  // An index past the end is undefined in the isa and every producer guards it
  // itself, llc with a `setp`/`bra` pair before the instruction; the next
  // instruction is what a jump table leaves there, so that is where it goes.
  if (op == "brx") {
    if (parts.size() < 2 || parts[1] != "idx") {
      err = "unsupported branch '" + opcode + "'";
      return "";
    }
    if (args.size() < 2) { err = "brx.idx with no table operand"; return ""; }
    auto tb = brxtab.find(trim(args[1]));
    if (tb == brxtab.end()) {
      err = "brx.idx names '" + trim(args[1]) +
            "', which no .branchtargets directive declares";
      return "";
    }
    vec_control = true;
    const std::string ix = "(uint32_t)(" + operand_as(args[0], "uint32_t") + ")";
    if (!vec) {
      std::string out = guard + "{ const uint32_t weft_ix = " + ix + "; ";
      for (size_t k = 0; k < tb->second.size(); k++)
        out += std::string(k ? "else if" : "if") + " (weft_ix == " + std::to_string(k) +
               "u) " +
               (msl ? "{ s->pc = " + std::to_string(label_id(tb->second[k])) + "u; continue; }"
                    : "goto " + sanitize(tb->second[k]) + ";") +
               " ";
      return out + "}";
    }
    // The warp target keeps a program counter per lane, so the table is a choice
    // per lane: the index is read inside the lane loop, and a lane whose index
    // picks nothing moves to the instruction after the table.
    const int fall = label_count++;
    const std::string fallu = std::to_string(fall) + "u";
    std::string pick = fallu;
    for (size_t k = tb->second.size(); k-- > 0;)
      pick = "weft_ix == " + std::to_string(k) + "u ? " +
             std::to_string(label_id(tb->second[k])) + "u : " + pick;
    std::string out = "{ for (uint32_t l = 0; l < 32u; l++) { if (!(am[l])) continue;\n";
    out += "    const uint32_t weft_ix = " + ix + ";\n";
    out += "    s->pc[l] = " +
           (vec_pred.empty() ? pick : "((" + vec_pred + ") ? " + pick + " : " + fallu + ")") +
           ";\n  }\n  goto dispatch; }\n  L" + std::to_string(fall) + ":;";
    return out;
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
    // result is a fold over every thread in the group. It is two rendezvous , 
    // publish and arrive, then the per-warp totals are known and every warp
    // folds them, because the one-rendezvous version makes every thread read
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
      // bar.red.<op>.<type> d, a, b, a is the barrier id and b the predicate.
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

  // mbarrier: an arrival-counted barrier with a phase, in threadgroup memory.
  //
  // The object is eight bytes and this runtime defines what is in them, because
  // nothing else writes it: the first word is how many arrivals are still owed and
  // the second is the initial count shifted up one place, with the phase in its low
  // bit. When the owed count reaches zero the count is restored and the phase flips,
  // which is the whole rule.
  //
  // A wait cannot spin on the CPU, one worker runs one lane at a time, so a lane that
  // spun would never let the lane that owes the arrival run, so it publishes a
  // resume point and gives the worker back, which is the shape `bar.sync` has and what
  // the warp scheduler releases once nothing else can run. On the GPU the lanes of a
  // group are resident together, so the phase is readable state and the caller's loop
  // around the branch is the waiting.
  if (op == "mbarrier") {
    std::string mode;
    for (const auto& p : parts)
      if (p == "init" || p == "arrive" || p == "try_wait" || p == "test_wait" ||
          p == "expect_tx" || p == "inval" || p == "invalidate")
        mode = p;
    if (mode.empty() || args.empty()) { err = "unsupported mbarrier '" + opcode + "'"; return ""; }
    // The object is whichever operand is bracketed.
    int ai = -1;
    for (size_t i = 0; i < args.size(); i++)
      if (!args[i].empty() && args[i][0] == '[') { ai = (int)i; break; }
    if (ai < 0) { err = "an mbarrier instruction without an address"; return ""; }
    std::string inner = trim(args[ai]);
    const size_t rb = inner.find(']');
    inner = trim(rb == std::string::npos ? inner.substr(1) : inner.substr(1, rb - 1));
    std::string nm = inner, extra;
    const size_t plus = inner.find('+');
    if (plus != std::string::npos) {
      nm = trim(inner.substr(0, plus));
      extra = trim(inner.substr(plus + 1));
    }
    auto so = shared_offsets.find(nm);
    if (so == shared_offsets.end()) {
      err = "'" + nm + "' is not a shared array: an mbarrier lives in threadgroup memory";
      return "";
    }
    std::string addr = std::to_string(so->second);
    if (!extra.empty()) addr += " + " + operand_as(extra, "uint64_t");
    // On the GPU the object is the only channel between the thread that arrives and
    // the threads that wait on it, so every access is a volatile one: a plain load
    // can be kept in a register across the caller's retry loop and the notification
    // would never be seen.
    const std::string ct = msl ? "threadgroup " : "";
    const std::string cv = msl ? "volatile " : "";
    const std::string w0 = "*(( " + cv + ct + "uint32_t*)(sh + " + addr + "))";
    const std::string w1 = "*(( " + cv + ct + "uint32_t*)(sh + " + addr + " + 4u))";
    // A waiting lane reads the phase on every pass, so on the GPU the read is one the
    // compiler may not keep in a register: that read is the whole notification.
    const std::string phase = msl
        ? ("*((volatile " + ct + "const uint32_t*)(sh + " + addr + " + 4u))")
        : w1;
    // One warp body runs a statement for all 32 lanes at once, so every mode that
    // touches the object says which lanes it applies to.
    // The predicate of `@%p mbarrier...` narrows which lanes act, so it belongs in the
    // loop condition: a warp body runs the statement for all 32 lanes otherwise, and
    // thirty-two arrivals on a barrier that was told to expect one never completes.
    const std::string lane = "for (uint32_t l = 0; l < 32u; l++) if (" + lane_cond() + ")";

    if (mode == "init") {
      if ((int)args.size() <= ai + 1) { err = "an mbarrier init without a count"; return ""; }
      const std::string n = "(uint32_t)(" + operand_as(args[ai + 1], "uint32_t") + ")";
      if (vec) {
        // `vec_side` rather than `vec_control`: the lane loop is written here, and the
        // statement still has to sit under the predicate of `@%p` if it carries one.
        vec_side = true;
        return "{ " + lane + " { const uint32_t weft_n = " + n + "; " + w0 + " = weft_n; " +
               w1 + " = (weft_n << 1); } }";
      }
      return guard + "{ const uint32_t weft_n = " + n + "; " + w0 + " = weft_n; " + w1 +
             " = (weft_n << 1); }";
    }
    // `mbarrier.expect_tx` says how many bytes the phase is also waiting for: the copy
    // that moves them completes that part of the same count, and the phase flips when
    // the sum reaches zero. `arrive.expect_tx` does both in one instruction, which is
    // the form a TMA load emits.
    bool also_arrive = false;
    for (const auto& p : parts)
      if (p == "arrive") also_arrive = true;
    auto expect_tx_code = [&]() -> std::string {
      if ((int)args.size() <= ai + 1) {
        err = "an mbarrier expect_tx without a byte count";
        return "";
      }
      const std::string tx = "((uint32_t)(" + operand_as(args[ai + 1], "uint32_t") + "))";
      // Two lanes can expect at once on the GPU, so the add is one atomic there too.
      if (msl)
        return "{ threadgroup atomic_uint* weft_x = (threadgroup atomic_uint*)(sh + " + addr +
               "); atomic_fetch_add_explicit(weft_x, " + tx + ", memory_order_relaxed); }";
      return "{ const uint32_t weft_tx = " + tx + "; " + w0 + " = " + w0 + " + weft_tx; }";
    };
    if (mode == "expect_tx" && !also_arrive) {
      const std::string code = expect_tx_code();
      if (code.empty()) return "";
      if (vec) { vec_side = true; return "{ " + lane + " " + code + " }"; }
      return guard + code;
    }
    if (mode == "arrive" || mode == "expect_tx") {
      // Only the fused form carries an expectation; a bare arrive has none.
      const std::string pre = (mode == "expect_tx") ? (expect_tx_code() + " ") : std::string();
      // The lanes of a group arrive at the same instant, so on the GPU the decrement
      // is one atomic. A read, a subtract and a write let two arrivals see the same
      // count and one of them never be counted, which is a wait that never ends , 
      // measured at 32 lanes, where the phase stayed put and the launch hung. The CPU
      // column runs one lane at a time, so there the plain form is already exact.
      const std::string body = msl
          ? ("threadgroup atomic_uint* weft_a = (threadgroup atomic_uint*)(sh + " + addr +
             "); const uint32_t weft_left = atomic_fetch_sub_explicit(weft_a, 1u, "
             "memory_order_relaxed); if (weft_left == 1u) { "
             "atomic_store_explicit(weft_a, (uint32_t)(*((volatile threadgroup const "
             "uint32_t*)(sh + " + addr + " + 4u)) >> 1u), memory_order_relaxed); "
             "*((volatile threadgroup uint32_t*)(sh + " + addr + " + 4u)) = "
             "(*((volatile threadgroup const uint32_t*)(sh + " + addr +
             " + 4u)) ^ 1u); }")
          : ("if (" + w0 + " != 0u) { " + w0 + " = " + w0 +
             " - 1u; if (" + w0 + " == 0u) { " + w0 + " = " + w1 +
             " >> 1; " + w1 + " ^= 1u; } }");
      // The state the instruction hands back is not read here: a parity wait names the
      // phase it is waiting for instead.
      bool have_state = false;
      std::string state;
      if (ai > 0 && !args[0].empty()) {
        const std::string rd = trim(args[0]);
        if (!rd.empty() && rd[0] == '%') {
          have_state = true;
          state = store(rd, "0ull", "uint64_t") + ";";
        }
      }
      if (vec) {
        vec_side = true;
        return "{ " + lane + " { " + pre + body + (have_state ? " " + state : std::string()) +
               " } }";
      }
      std::string out = guard + "{ " + pre + body;
      if (have_state) out += " " + state;
      return out + " }";
    }
    if (mode == "try_wait" || mode == "test_wait") {
      if ((int)args.size() <= ai + 1) { err = "an mbarrier wait without a phase"; return ""; }
      const std::string ph = "(uint32_t)(" + operand_as(args[ai + 1], "uint32_t") + ")";
      const std::string done = "((" + phase + " & 1u) != " + ph + ")";
      // A test never waits. Nor does anyone on the GPU: the lanes of a group are
      // resident together and an arrival from another warp is ordered through a
      // `bar.sync`, which is a rendezvous, so the read is the answer and the caller's
      // loop around the branch is the waiting.
      if (mode == "test_wait" || msl || vec) return guard + store(args[0], done, "bool");
      // The CPU column has no other way: the retry is a resume point, so the phase is
      // read again on the way back in rather than spun on.
      const int id = next_resume++;
      resume_cases += "    case " + std::to_string(id) + ": goto R" + std::to_string(id) + ";\n";
      const std::string sid = std::to_string(id);
      // `@%p` narrows the wait: a lane whose predicate is false does not wait at all,
      // so only a lane that was asked to wait may park.
      std::string cond;
      if (guard.size() > 6 && guard.compare(0, 4, "if (") == 0)
        cond = guard.substr(4, guard.size() - 6);
      return "R" + sid + ":;\n  if (!(" + done + ")" +
             (cond.empty() ? std::string() : " && (" + cond + ")") + ") { s->pc = " + sid +
             "u; return; }\n  " + guard + store(args[0], done, "bool");
    }
    // An expectation and an invalidation only matter to a bulk copy, which has no form
    // here yet: a kernel that initialises a barrier and arrives on it needs neither.
    return guard + ";";
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
    // A lane's registers are a member of the state struct, so a pointer into one has to
    // name the address space on the Metal target; the host's C++ has no such qualifier.
    const std::string addr_q = msl ? "thread " : "";
    // `c` packs two fields: a clamp value in bits 4:0 and a sub-segment mask in bits
    // 12:8. The spec's model is that a lane may only read inside its segment, the
    // source index is clamped at the segment's end, and a lane whose source falls
    // outside that clamp keeps its own value. Read as a single number in 0..31 -- the
    // only form this used to accept -- every segment is the whole warp, which is the
    // common case but not the general one.
    uint32_t cval = (mode == "up") ? 0u : 31u, segmask = 0u;
    bool c_lit = true;
    std::string cexpr;
    if (args.size() >= 4) {
      const std::string t = trim(args[3]);
      char* end = nullptr;
      const long v = std::strtol(t.c_str(), &end, 0);
      if (end && *end == '\0' && v >= 0 && v <= 0x1fff) {
        cval = (uint32_t)v & 31u;
        segmask = ((uint32_t)v >> 8) & 31u;
      } else if (!t.empty() &&
                 (t[0] == '%' || t[0] == '(' || t[0] == '_' ||
                  std::isalnum((unsigned char)t[0]))) {
        // A clamp in a register is how a front end writes a segment end that differs
        // per lane -- a `width` below 32 -- so its two fields are taken from the value
        // at run time rather than refused.
        cexpr = operand_as(t, "uint32_t");
        if (cexpr.empty() || !err.empty()) return "";
        c_lit = false;
      } else {
        err = "unsupported shuffle clamp '" + t + "' in '" + opcode + "'";
        return "";
      }
    }
    const std::string cval_e = c_lit ? std::to_string(cval) : "((uint32_t)(" + cexpr + ") & 31u)";
    const std::string seg_e =
        c_lit ? std::to_string(segmask) : "((((uint32_t)(" + cexpr + ") >> 8) & 31u))";
    auto max_lane = [&](const std::string& lane) -> std::string {
      if (c_lit && segmask == 0) return cval_e;
      return "((" + lane + " & " + seg_e + ") | (" + cval_e + " & ~" + seg_e + "))";
    };
    auto peer_of = [&](const std::string& lane, const std::string& b) -> std::string {
      const std::string hi = max_lane(lane);
      if (mode == "idx") {
        const std::string j =
            (c_lit && segmask == 0)
                ? b
                : "((" + lane + " & " + seg_e + ") | (" + b + " & ~" + seg_e + "))";
        if (c_lit && segmask == 0 && cval == 31) return j;
        return "(" + j + " <= " + hi + " ? " + j + " : " + lane + ")";
      }
      if (mode == "bfly") {
        const std::string j = "(" + lane + " ^ " + b + ")";
        if (c_lit && segmask == 0 && cval == 31) return j;
        return "(" + j + " <= " + hi + " ? " + j + " : " + lane + ")";
      }
      if (mode == "up")
        return "((int32_t)" + lane + " - (int32_t)(" + b + ") >= (int32_t)" + hi + " ? " +
               lane + " - " + b + " : " + lane + ")";
      return "(" + lane + " + " + b + " <= " + hi + " ? " + lane + " + " + b + " : " + lane + ")";
    };
    // The shifted-vector form below is exact only for the full-warp clamp: a segment
    // edge moves where the shift stops, and `.idx` reads one lane from everywhere.
    const bool full_clamp =
        c_lit && segmask == 0 &&
        ((mode == "up" && cval == 0) || ((mode == "down" || mode == "idx") && cval == 31));

    if (vec) {
      // 32 lanes are one vector here, so a shuffle is a gather inside the
      // register file: no publication, no suspension, no scheduler.
      Reg rs = parse_reg(args[1]);
      if (!rs.ok || !rs.special.empty()) {
        vec_ok = false;
        err = "shuffle of a value that is not a register";
        return "";
      }
      const std::string lane = "l";
      const std::string b = operand_as(args[2], "uint32_t");
      const std::string peer = peer_of(lane, b);
      const std::string arr = "s->" + rs.base + "[" + std::to_string(rs.index) + "]";
      const std::string src = arr + "[" + peer + "]";
      const std::string masked = store(args[0], src, "uint32_t");

      // The common case: a delta that is the same in every lane, which is what
      // a reduction loop looks like (16, 8, 4, 2, 1). Then the peer of lane l
      // is lane l+d, so the whole shuffle is a shift of the vector, one run of
      // consecutive elements, which the compiler vectorises. As a per-lane
      // index it could not: `l + d` has a runtime term, so the loop goes scalar
      // and this becomes the most expensive instruction in the kernel.
      const VecVal dv = vec_sym(args[2]);
      const bool uniform_delta = dv.ok && dv.stride == 0;
      if (uniform_delta && full_clamp && vec_pred.empty() &&
          (mode == "down" || mode == "up" || mode == "idx")) {
        std::string dtext;
        Reg dr = parse_reg(trim(args[2]));
        if (dr.ok && dr.special.empty()) {
          // Same value in every lane, so lane 0 holds it, and a full warp is
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
          shift = "const uint32_t weft_s = (" + dtext + ") & " + cval_e +
                  "; const " + addr_q + "uint32_t* weft_v = " + arr +
                  "; const uint32_t weft_sv = weft_v[weft_s];" +
                  " for (uint32_t l = 0; l < 32u; l++) { " + move_one + " }";
        } else if (mode == "down") {
          shift = "const uint32_t weft_d = " + dtext + "; const " + addr_q +
                  "uint32_t* weft_v = " + arr +
                  "; for (uint32_t l = 0; l + weft_d < 32u; l++) { " + move1 +
                  " } for (uint32_t l = (weft_d < 32u ? 32u - weft_d : 0u); l < 32u; l++) { " +
                  move_self + " }";
        } else {
          shift = "const uint32_t weft_d = (" + dtext + " < 32u ? " + dtext +
                  " : 32u); const " + addr_q + "uint32_t* weft_v = " + arr +
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

    const std::string lane = "(ctx.lid & 31u)";
    const std::string base = "(ctx.lid & ~31u)";
    const std::string b = operand_as(args[2], "uint32_t");
    const std::string src = peer_of(lane, b);

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

  // `elect.sync d|p, membermask` elects one lane out of the mask: every lane learns
  // which lane it is and only that lane's predicate is true. The choice every engine
  // here makes is the lowest lane of the mask, which is the one a hardware leader
  // election is allowed to make, and the answer is the same on both columns.
  if (op == "elect") {
    if (args.empty()) { err = "elect.sync with no destination"; return ""; }
    std::vector<std::string> two = split_top(args[0], '|');
    if (two.size() != 2) { err = "elect.sync needs the 'd|p' destination form"; return ""; }
    const std::string d_dst = trim(two[0]), p_dst = trim(two[1]);
    const bool want_d = !d_dst.empty() && d_dst != "_";
    const bool want_p = !p_dst.empty() && p_dst != "_";
    std::string member = "0xFFFFFFFFu";
    if (args.size() >= 2) {
      const std::string t = trim(args[1]);
      char* end = nullptr;
      const long v = std::strtol(t.c_str(), &end, 0);
      if (!end || *end != '\0') { err = "unsupported member mask '" + t + "'"; return ""; }
      member = std::to_string((uint32_t)v) + "u";
    }
    uses_shfl = true;
    const std::string pick =
        "const uint32_t weft_e = weft_m ? (uint32_t)__builtin_ctz(weft_m) : 0xFFFFFFFFu; ";
    if (vec) {
      // All 32 lanes are one thread's registers, so the mask is the lane mask and
      // the election is a count of trailing zeros, no rendezvous at all.
      vec_side = true;
      std::string out = "{ uint32_t weft_m = 0; for (uint32_t l = 0; l < 32u; l++) "
                        "if (am[l]) weft_m |= 1u << l; weft_m &= " + member + "; " + pick;
      if (want_d || want_p) {
        out += "for (uint32_t l = 0; l < 32u; l++) { ";
        if (want_d) out += store(d_dst, "weft_e", "uint32_t") + " ";
        if (want_p) out += store(p_dst, "(uint32_t)l == weft_e", "bool") + " ";
        out += "} ";
      }
      return out + "}";
    }
    if (msl) {
      std::string body = guard + "{ const uint32_t weft_m = (uint32_t)(simd_vote::vote_t)" +
                          std::string("simd_ballot(true) & ") + member + "; " + pick;
      if (want_d) body += store(d_dst, "weft_e", "uint32_t") + " ";
      if (want_p) body += store(p_dst, "((ctx.lid & 31u) == weft_e)", "bool") + " ";
      return body + "}";
    }
    // The host has no warp, so the active lanes have to agree first: publish, give
    // the worker back, and fold the published bits once every lane has arrived.
    const int id = next_resume++;
    resume_cases += "    case " + std::to_string(id) + ": goto R" + std::to_string(id) + ";\n";
    const std::string label = "R" + std::to_string(id) + ":;";
    const std::string publish =
        "((uint32_t*)weft::shfl_pub_buf(ctx))[ctx.lid] = 1u; s->pc = " + std::to_string(id) +
        "u; return;\n  ";
    std::string refine = "{ const uint32_t weft_m = weft::vote_ballot(ctx) & " + member + "; " + pick;
    if (want_d) refine += store(d_dst, "weft_e", "uint32_t") + " ";
    if (want_p) refine += store(p_dst, "((ctx.lid & 31u) == weft_e)", "bool") + " ";
    refine += "}";
    if (!guard.empty()) return guard + "{ " + publish + "}\n  " + label + "\n  " + guard + refine;
    return "{ " + publish + "}\n  " + label + "\n  " + refine;
  }

  // `match.any.sync d, a, membermask` answers the mask of lanes whose value equals
  // this lane's, and `match.all.sync d|p, a, membermask` answers all ones (and a
  // true predicate) only if every live lane of the mask holds the same value. Only
  // live lanes count, so a short warp's absent lanes are not compared: on the host
  // that is the thread count, and on the GPU it is the ballot of the lanes that ran.
  if (op == "match") {
    bool any = false, all = false;
    for (const auto& p : parts) {
      if (p == "any") any = true;
      else if (p == "all") all = true;
    }
    if (!any && !all) { err = "unsupported match mode in '" + opcode + "'"; return ""; }
    if (args.size() < 2) { err = "match with no value"; return ""; }
    std::string d_dst = trim(args[0]), p_dst;
    if (all) {
      std::vector<std::string> two = split_top(args[0], '|');
      if (two.size() != 2) { err = "match.all.sync needs the 'd|p' destination form"; return ""; }
      d_dst = trim(two[0]);
      p_dst = trim(two[1]);
    }
    const bool want_d = !d_dst.empty() && d_dst != "_";
    const bool want_p = !p_dst.empty() && p_dst != "_";
    std::string member = "0xFFFFFFFFu";
    if (args.size() >= 3) {
      const std::string t = trim(args[2]);
      char* end = nullptr;
      const long v = std::strtol(t.c_str(), &end, 0);
      if (!end || *end != '\0') { err = "unsupported member mask '" + t + "'"; return ""; }
      member = std::to_string((uint32_t)v) + "u";
    }
    uses_shfl = true;
    const std::string value = operand_as(args[1], "uint32_t");
    // What each column adds after the same-value mask is known.
    auto emit_from = [&](const std::string& scope) {
      std::string out = scope;
      if (all) {
        out += "const bool weft_all = (weft_same == weft_am); ";
        if (want_d) out += store(d_dst, "(weft_all ? " + member + " : 0u)", "uint32_t") + " ";
        if (want_p) out += store(p_dst, "weft_all", "bool") + " ";
      } else if (want_d) {
        out += store(d_dst, "weft_same", "uint32_t") + " ";
      }
      return out;
    };
    if (vec) {
      Reg rs = parse_reg(trim(args[1]));
      if (!rs.ok || !rs.special.empty()) {
        err = "match of a value that is not a register";
        return "";
      }
      vec_side = true;
      const std::string arr = "s->" + rs.base + "[" + std::to_string(rs.index) + "]";
      std::string out = "{ uint32_t weft_am = 0; for (uint32_t k = 0; k < 32u; k++) "
                        "if (am[k]) weft_am |= 1u << k; weft_am &= " + member +
                        "; for (uint32_t l = 0; l < 32u; l++) { if (!am[l]) continue; "
                        "uint32_t weft_same = 0; for (uint32_t k = 0; k < 32u; k++) "
                        "if (am[k] && " + arr + "[k] == " + arr + "[l]) weft_same |= 1u << k; ";
      out += emit_from(std::string());
      return out + "} }";
    }
    if (msl) {
      // One shuffle per lane position: the warp's values compared against this
      // lane's, which is the same comparison the host folds out of its slot array.
      std::string out = guard + "{ const uint32_t weft_v = " + value +
                        "; const uint32_t weft_am = (uint32_t)(simd_vote::vote_t)" +
                        "simd_ballot(true) & " + member + "; uint32_t weft_same = 0; "
                        "for (uint32_t k = 0; k < 32u; k++) "
                        "if (simd_shuffle(weft_v, (ushort)k) == weft_v) weft_same |= 1u << k; "
                        "weft_same &= weft_am; ";
      out += emit_from(std::string());
      return out + "}";
    }
    const int id = next_resume++;
    resume_cases += "    case " + std::to_string(id) + ": goto R" + std::to_string(id) + ";\n";
    const std::string label = "R" + std::to_string(id) + ":;";
    const std::string publish = "((uint32_t*)weft::shfl_pub_buf(ctx))[ctx.lid] = " + value +
                                "; s->pc = " + std::to_string(id) + "u; return;\n  ";
    const std::string scope = "{ const uint32_t weft_wb = ctx.lid & ~31u; "
                              "const uint32_t weft_lv = weft::live_threads(ctx.tgid, ctx.tg_size, ctx.grid); "
                              "const uint32_t weft_wl = (weft_lv > weft_wb) ? "
                              "((weft_lv - weft_wb) > 32u ? 32u : (weft_lv - weft_wb)) : 0u; "
                              "const uint32_t weft_am = ((weft_wl >= 32u ? 0xFFFFFFFFu : "
                              "((1u << weft_wl) - 1u)) & " + member + "); "
                              "const uint32_t weft_v = " + value + "; "
                              "const uint32_t* weft_pv = weft::shfl_read_buf(ctx) + weft_wb; "
                              "uint32_t weft_same = 0; for (uint32_t k = 0; k < 32u; k++) "
                              "if (((weft_am >> k) & 1u) && weft_pv[k] == weft_v) weft_same |= 1u << k; ";
    std::string refine = emit_from(scope);
    refine += "}";
    if (!guard.empty()) return guard + "{ " + publish + "}\n  " + label + "\n  " + guard + refine;
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
      // both are then exact, a `__int128` scalar compiles here, but this host's
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
    else if (cls == "normal")
      // The isa counts positive and negative zero as normal numbers, and `isnormal` does
      // not: it is true for a value that is neither zero nor subnormal. The isa's reading
      // is "finite and not subnormal, or zero", which is what this says. Measured: zero
      // answered 0 before this line and 1 after it.
      e = msl ? ("(((" + v + ") == 0.0f) || (isfinite(" + v + ") && fabs(" + v + ") >= 0x1p-126f))")
              : ("(std::fpclassify(" + v + ") == FP_NORMAL || (" + v + ") == 0)");
    else if (cls == "subnormal")
      // Metal has no fpclassify, and a subnormal is exactly "not zero and below
      // the smallest normal", so the test is the two comparisons.
      e = msl ? ("((" + v + " != 0.0f) && (fabs(" + v + ") < 0x1p-126f))")
              : ("(std::fpclassify(" + v + ") == FP_SUBNORMAL)");
    // The ISA spells it `.notanumber`; `.nan` is what some producers write.
    else if (cls == "nan" || cls == "notanumber") e = math("isnan", v);
    if (e.empty()) { err = "unsupported class test '" + opcode + "'"; return ""; }
    return guard + store(args[0], e, "bool");
  }

  // prmt.b32{.mode} d, a, b, c: four bytes picked out of the eight of (a:b). No
  // mode reads a nibble a byte and sign-replicates when its top bit is set; the
  // six named modes read one 2-bit selector and walk, repeat or clamp.
  if (op == "prmt" && args.size() >= 3) {
    int mode = -1;
    for (size_t i = 1; i < parts.size(); i++) {
      const std::string& p = parts[i];
      if (p == "b32") continue;
      if (p == "f4e") mode = 0;
      else if (p == "b4e") mode = 1;
      else if (p == "rc8") mode = 2;
      else if (p == "ecl") mode = 3;
      else if (p == "ecr") mode = 4;
      else if (p == "rc16") mode = 5;
      else {
        err = "unsupported byte permute '" + opcode + "'";
        return "";
      }
    }
    const std::string c = args.size() >= 4 ? operand_as(args[3], "uint32_t") : "0u";
    return guard + store(args[0],
                         "weft_prmt(" + operand_as(args[1], "uint32_t") + ", " +
                             operand_as(args[2], "uint32_t") + ", " + c + ", " +
                             std::to_string(mode) + ")",
                         "uint32_t");
  }

  // add/sub/mul/fma/min/max/neg/abs of a narrow float: `.f16` is a half's 5-bit
  // exponent over a 10-bit significand, `.bf16` a float's 8-bit exponent over 7,
  // and the `b16` register holds the bits of either. The generic integer path added
  // the patterns instead (0x3C00 + 0x4000 answered 0x7C00, an infinity, where
  // 1.0 + 2.0 is 3.0), so the type suffix is what says to read them as a float at
  // all. Both widths are answered from one exact routine, so neither form has to be
  // refused for want of a single rounding.
  if ((op == "add" || op == "sub" || op == "mul" || op == "fma" || op == "mad" ||
       op == "min" || op == "max" || op == "neg" || op == "abs" || op == "div" ||
       op == "rcp" || op == "sqrt" || op == "ex2" || op == "lg2" || op == "rsqrt" ||
       op == "sin" || op == "cos" || op == "tanh") &&
      (has_part(parts, "f16") || has_part(parts, "f16x2") || has_part(parts, "bf16") ||
       has_part(parts, "bf16x2"))) {
    // Everything the operation does not name is refused rather than dropped: `.rz`
    // is a second rounding rule, `.ftz` flushes subnormals, and an unknown qualifier
    // would be a different answer that looks like this one. `min`/`max` also carry
    // `.NaN` and the `min.xorsign.abs` pair, each with its own helper.
    for (const auto& p : parts) {
      if (p == op || p == "f16" || p == "f16x2" || p == "bf16" || p == "bf16x2") continue;
      if (p == "rn") continue;
      if (p == "sat" && (op == "add" || op == "sub" || op == "mul")) continue;
      // `.approx` asks only that the answer be close, and the portable library
      // rounded once to the narrow grid is closer than "close": it is the exact
      // answer, which every approximation is allowed to be.
      if (p == "approx" && (op == "ex2" || op == "lg2" || op == "rsqrt" || op == "sin" ||
                            op == "cos" || op == "tanh" || op == "rcp" || op == "sqrt"))
        continue;
      if ((op == "min" || op == "max") && (p == "NaN" || p == "xorsign" || p == "abs"))
        continue;
      err = "'" + opcode + "' carries a qualifier the 16-bit path does not know: ." + p;
      return "";
    }
    if ((op == "min" || op == "max") && has_part(parts, "abs") &&
        !has_part(parts, "xorsign")) {
      err = "'" + opcode + "' names .abs without .xorsign, which is the only form the isa has";
      return "";
    }
    if (op == "mad") {
      err = "'" + opcode + "' is not a 16-bit float instruction: the fused form the isa "
            "names is `fma`";
      return "";
    }
    const bool packed = has_part(parts, "f16x2") || has_part(parts, "bf16x2");
    const bool bf = has_part(parts, "bf16") || has_part(parts, "bf16x2");
    const std::string wt = packed ? "uint32_t" : "uint16_t";
    // The type and the packing only pick which family answers: `weft_bf16x2_fma` is
    // the packed bf16 fused form, and every one of them names the same operation.
    const std::string pre = packed ? (bf ? "weft_bf16x2_" : "weft_f16x2_")
                                   : (bf ? "weft_bf16_" : "weft_f16_");
    if (op == "neg" || op == "abs") {
      if (!need(2)) return "";
      return guard + store(args[0], pre + op + "(" + operand_as(args[1], wt) + ")", wt);
    }
    if (op == "min" || op == "max") {
      if (!need(3)) return "";
      // Three answers: the plain form hands a NaN operand back as the other one,
      // `.NaN` answers a canonical NaN instead, and `.xorsign.abs` compares the
      // magnitudes and keeps the exclusive or of the two signs.
      const std::string fn =
          pre + op +
          (has_part(parts, "xorsign") ? "_xorabs" : (has_part(parts, "NaN") ? "_nan" : ""));
      return guard + store(args[0],
                           fn + "(" + operand_as(args[1], wt) + ", " +
                               operand_as(args[2], wt) + ")",
                           wt);
    }
    // The approximate transcendentals answer the portable library rounded once, the
    // same bargain the f32 forms strike: no CPU reproduces the silicon's polynomial.
    // The packed forms are that answer per half, so `weft_f16x2_tanh` is two independent
    // roundings rather than one rounding of a wider expression.
    if (op == "ex2" || op == "lg2" || op == "rsqrt" || op == "sin" || op == "cos" ||
        op == "tanh") {
      if (!need(2)) return "";
      return guard + store(args[0], pre + op + "(" + operand_as(args[1], wt) + ")", wt);
    }
    // Division, reciprocal and square root end in the same one-rounding pack the
    // fused form uses, so a narrow quotient or root is the nearest value on the
    // grid rather than whatever a widening to float would have rounded twice.
    if (op == "div") {
      if (!need(3)) return "";
      return guard + store(args[0],
                           pre + "div(" + operand_as(args[1], wt) + ", " +
                               operand_as(args[2], wt) + ")",
                           wt);
    }
    if (op == "rcp") {
      if (!need(2)) return "";
      return guard + store(args[0], pre + "rcp(" + operand_as(args[1], wt) + ")", wt);
    }
    if (op == "sqrt") {

      if (!need(2)) return "";
      return guard + store(args[0], pre + "sqrt(" + operand_as(args[1], wt) + ")", wt);
    }
    if (op == "fma") {
      if (!need(4)) return "";
      std::string e = pre + "fma(" + operand_as(args[1], wt) + ", " +
                      operand_as(args[2], wt) + ", " + operand_as(args[3], wt) + ")";
      if (has_part(parts, "sat")) {
        err = "'" + opcode + "' asks for .sat on a fused multiply-add, which the isa does "
              "not name";
        return "";
      }
      return guard + store(args[0], e, wt);
    }
    if (!need(3)) return "";
    std::string e = pre + op + "(" + operand_as(args[1], wt) + ", " +
                    operand_as(args[2], wt) + ")";
    if (has_part(parts, "sat")) e = pre + "sat(" + e + ")";
    return guard + store(args[0], e, wt);
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

  // The condition code, PTX's one flag that survives between instructions.
  // `add.cc`, `sub.cc` and `mad.cc` write it, `addc`, `subc` and `madc` read it and
  // write it again when they carry `.cc`, and that chain is how a 64-bit or a
  // 128-bit integer add is spelled. The carry out is the one the sum leaves: a sum
  // below its first addend wrapped, and a difference whose first source is below the
  // second borrowed. With a carry-in the flag is the two events together, which is
  // what makes the second word of a wide addition come out right.
  {
    const bool writes_cc = has_part(parts, "cc");
    const bool has_hi = has_part(parts, "hi");
    const bool is_carry_op = (op == "addc" || op == "subc" || op == "madc");
    const bool mad_op = (op == "mad" || op == "madc");
    const bool sub_op = (op == "sub" || op == "subc");
    const bool reads_cc = is_carry_op;
    // A product with a word picked out belongs here even without a flag: `.hi` is the
    // high word, and the add that would otherwise take it answers with the low one,
    // which is a wrong number rather than a refusal.
    const bool word_pick = mad_op && has_hi;
    if (is_carry_op || word_pick || (writes_cc && (op == "add" || op == "sub" || mad_op))) {
      if (!need(mad_op ? 4u : 3u)) return "";
      if (ty != "uint32_t" && ty != "int32_t" && ty != "uint64_t" && ty != "int64_t") {
        err = "'" + opcode + "' is an integer instruction; the condition code is an integer flag";
        return "";
      }
      if (has_part(parts, "wide")) {
        err = "'" + opcode + "': a wide product is two words, and this lowers one";
        return "";
      }
      // One flag per lane is the per-lane back end's; `r[reg][lane]` has nowhere to
      // put one that is not a register. A word of a product needs no flag, so it is
      // still a per-lane expression the vector back end can carry.
      if (vec && (reads_cc || writes_cc)) vec_ok = false;
      const std::string ut = cpp_size(ty) == 8 ? "uint64_t" : "uint32_t";
      const bool sgn = (ty == "int32_t" || ty == "int64_t");
      const std::string f = (reads_cc || writes_cc) ? cc_member() : std::string();
      const std::string a = operand_as(args[1], ut);
      const std::string b = operand_as(args[2], ut);
      const std::string q = mad_op ? operand_as(args[3], ut) : b;
      // The word of the product. `.lo` is the plain multiply, which wraps. `.hi` is
      // the high word: exact for unsigned, sign-extended for signed, and the same two
      // answers `mul.hi` already gives, so a magic-number multiply and an
      // extended-precision one agree down both roads.
      std::string prod = mad_op ? "(" + a + " * " + b + ")" : a;
      if (has_hi) {
        const std::string as = operand_as(args[1], ty), bs = operand_as(args[2], ty);
        if (ut == "uint32_t")
          prod = sgn ? "((uint32_t)(((int64_t)" + as + " * (int64_t)" + bs + ") >> 32))"
                     : "((uint32_t)(((uint64_t)(uint32_t)" + a + " * (uint64_t)(uint32_t)" + b +
                           ") >> 32))";
        else
          prod = sgn ? "((uint64_t)weft_mulhi_s64(" + as + ", " + bs + "))"
                     : "weft_mulhi_u64(" + a + ", " + b + ")";
      }
      // `weft_` names throughout: the lane back end reaches its state through `s`,
      // and a temporary called `s` would shadow the whole register file.
      std::string line = "{ " + ut + " weft_p = " + prod + "; " + ut +
                         " weft_q = " + q + "; " + ut + " weft_s = weft_p " +
                         (sub_op ? "-" : "+") + " weft_q; ";
      if (reads_cc)
        line += ut + " weft_r = weft_s " + (sub_op ? "-" : "+") + " " + f + "; ";
      if (writes_cc) {
        std::string flag = sub_op ? "weft_p < weft_q" : "weft_s < weft_p";
        if (reads_cc) flag += sub_op ? " || weft_s < " + f : " || weft_r < weft_s";
        line += f + " = (" + flag + ") ? 1u : 0u; ";
      }
      line += store(args[0], reads_cc ? "weft_r" : "weft_s", ty) + " }";
      return guard + line;
    }
    if (writes_cc) {
      err = "'" + opcode +
            "': only add, sub, mad, addc, subc and madc carry the condition code";
      return "";
    }
  }

  auto it = binop.find(op);
  if (it != binop.end() && args.size() >= 3) {
    // A shift by at least the register width is defined by the ISA: the count is
    // clamped to the width, so `shl.b32` by 32 is zero and `shr.s32` by 32 is the
    // sign. C++ leaves it undefined, and the two compilers here did different
    // things with it, the CPU column happened to get zero, the Metal column
    // folded the count modulo 32 and pulled elements that belonged past the word
    // back into it, which the 6-bit `ldmatrix` fixture caught. So the count is
    // guarded; a constant stays a plain shift, where the guard folds away.
    if (op == "shl" || op == "shr") {
      const std::string a = operand_as(args[1], ty);
      const std::string sh = operand_as(args[2], ty);
      const int bits = cpp_size(ty) * 8;
      const std::string lit = trim(args[2]);
      char* end = nullptr;
      const long n = std::strtol(lit.c_str(), &end, 0);
      const bool constant = !lit.empty() && end && *end == '\0';
      // What the clamp leaves: nothing for a left shift or an unsigned right one, the
      // sign for a signed right one. The unsigned case fills with zero, and reading the
      // sign bit instead answered 1 for `shr.u32 0xFFFFFFFF, 32`.
      const bool keeps_sign = (op == "shr" && ty[0] == 'i');
      const std::string fill =
          keeps_sign ? ("(" + a + " >> " + std::to_string(bits - 1) + ")") : "0";
      if (constant && n >= bits) return guard + store(args[0], fill, ty);
      if (constant && n >= 0)
        return guard + store(args[0],
                             a + (op == "shl" ? " << " : " >> ") + std::to_string(n), ty);
      return guard + store(args[0],
                           "((uint64_t)(" + sh + ") >= " + std::to_string(bits) + "ull ? (" + ty +
                               ")(" + fill + ") : (" + a + (op == "shl" ? " << (" : " >> (") +
                               sh + ")))",
                           ty);
    }
    return guard + store(args[0],
                         operand_as(args[1], ty) + " " + it->second + " " + operand_as(args[2], ty),
                         ty);
  }

  if (op == "fma" || op == "mad") {
    if (!need(4)) return "";
    // `mad` rounds twice and `fma` rounds once, so on the software double the first
    // is two calls and the second is the one helper that keeps the intermediate
    // exact, a 192-bit accumulator with a single rounding at the end.
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

  if (op == "copysign") {
    if (!need(3)) return "";
    // The answer is the first operand's bits with the second's sign on top, so this is
    // the bit operation rather than a library call: it keeps a NaN payload, which is
    // what "copied without modification" means for the value that is not a number.
    const std::string A = operand_as(args[1], ty), B = operand_as(args[2], ty);
    if (msl && ty == "double")
      return guard + store(args[0],
                           "weft_fp64::copysign_bits(" + operand_as(args[1], "uint64_t") + ", " +
                               operand_as(args[2], "uint64_t") + ")",
                           "uint64_t");
    if (ty == "double")
      return guard + store(args[0],
                           "weft_bitcast<double>((weft_bitcast<uint64_t>(" + A +
                               ") & 0x7FFFFFFFFFFFFFFFull) | (weft_bitcast<uint64_t>(" + B +
                               ") & 0x8000000000000000ull))",
                           "double");
    return guard + store(args[0],
                         "weft_bitcast<float>((weft_bitcast<uint32_t>(" + A +
                             ") & 0x7FFFFFFFu) | (weft_bitcast<uint32_t>(" + B +
                             ") & 0x80000000u))",
                         "float");
  }

  if (op == "min" || op == "max") {
    if (!need(3)) return "";
    // The three-input form (`{min,max}{.NaN}{.abs}.f32 d, a, b, c`, sm_100) asks for an
    // answer over three magnitudes, which is not two answers folded: reading two of its
    // operands and dropping the third is a different number that looks like this one.
    if (args.size() > 3) {
      err = "'" + opcode + "': the three-input min/max form is not translated";
      return "";
    }
    const std::string A = operand_as(args[1], ty), B = operand_as(args[2], ty);
    const bool flt = (ty == "float" || ty == "double");
    if (!flt) {
      if (has_part(parts, "NaN") || has_part(parts, "xorsign") || has_part(parts, "abs")) {
        err = "'" + opcode + "' carries a floating-point modifier on an integer type";
        return "";
      }
      return guard + store(args[0], math(op, A + ", " + B), ty);
    }
    // The isa spells the modifiers on the .f32 form -- `{min,max}{.NaN}{.xorsign.abs}.f32`,
    // with `.abs` alone only in the sm_100 three-input form -- and `{min,max}.f64` takes
    // none at all. Reading one on a double would be inventing an instruction.
    if (ty == "double" && (has_part(parts, "NaN") || has_part(parts, "xorsign") ||
                           has_part(parts, "abs"))) {
      err = "'" + opcode + "': min/max of a double takes no modifier";
      return "";
    }
    if (has_part(parts, "abs") && !has_part(parts, "xorsign")) {
      err = "'" + opcode + "' names .abs without .xorsign, which the two-input form does"
            " not have";
      return "";
    }
    if (has_part(parts, "xorsign")) {
      // Not this instruction with an extra: the magnitude is the comparison of the two
      // magnitudes and the sign is the exclusive or of both, so `min.xorsign.abs(-1, 2)`
      // is -1 where `min(-1, 2)` is -1 for another reason, and `max.xorsign.abs(1, -4)`
      // is -4. The comparison runs on the magnitude's bit pattern, where a non-negative
      // float's order is its integer order, and a NaN magnitude answers the canonical
      // NaN, which is the all-ones word for a float.
      const std::string ua = "weft_bitcast<uint32_t>(" + A + ")";
      const std::string ub = "weft_bitcast<uint32_t>(" + B + ")";
      const std::string ma = "(" + ua + " & 0x7FFFFFFFu)";
      const std::string mb = "(" + ub + " & 0x7FFFFFFFu)";
      const std::string mag =
          "((" + ma + (op == "min" ? " < " : " > ") + mb + ") ? " + ma + " : " + mb + ")";
      return guard + store(args[0],
                           "(((" + mag + " & 0x7F800000u) == 0x7F800000u && (" + mag +
                               " & 0x007FFFFFu) != 0u) ? weft_bitcast<float>(0x7FFFFFFFu) :"
                               " weft_bitcast<float>(" + mag + " | ((" + ua + " ^ " + ub +
                               ") & 0x80000000u)))",
                           "float");
    }
    // The plain form is minNum/maxNum: a NaN operand hands back the other one, and a
    // comparison with both NaNs answers a NaN. `std::min` is not that -- it returns its
    // first operand whenever the comparison is false, so a NaN in the first operand came
    // back out -- and it treats -0.0 and +0.0 as equal, where the isa puts -0.0 below.
    // Both are said here rather than in a helper so the expression is the definition.
    const std::string canon = ty == "double" ? "weft_bitcast<double>(0x7FF8000000000000ull)"
                                             : "weft_bitcast<float>(0x7FFFFFFFu)";
    const std::string isnanA = "(" + math("isnan", A) + ")", isnanB = "(" + math("isnan", B) + ")";
    // Equal values are only a question when they are the two zeros: the isa puts -0.0
    // below +0.0 and a C comparison calls them equal, so the sign bit breaks that tie.
    // `min(-0.0, +0.0)` answered +0.0 with the comparison alone.
    const std::string gt = (op == "min") ? " < " : " > ";
    const std::string lt = (op == "min") ? " > " : " < ";
    const std::string tie = "(" + math("signbit", A) + " ? " + (op == "min" ? A : B) + " : " +
                            (op == "min" ? B : A) + ")";
    const std::string pick =
        (std::string("(") + isnanA + " ? (" + isnanB + " ? " + canon + " : " + B + ") : (" +
         isnanB + " ? " + A + " : ((" + A + gt + B + ") ? " + A + " : ((" + B + lt + A + ") ? " + B +
         " : " + tie + "))))");
    // minNum/maxNum on the software double is already this rule, the zeros and the
    // both-NaN answer included.
    if (msl && ty == "double")
      return guard + store(args[0],
                           std::string("weft_fp64::") + op + "_bits(" +
                               operand_as(args[1], "uint64_t") + ", " +
                               operand_as(args[2], "uint64_t") + ")",
                           "uint64_t");
    if (has_part(parts, "NaN"))
      return guard + store(args[0], "(" + isnanA + " || " + isnanB + ") ? " + canon + " : " + pick,
                           ty);
    return guard + store(args[0], pick, ty);
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
  //
  // `dp4a.atype.btype d, a, b, c` takes four bytes from `a`, four from `b`, the
  // sign of each chosen by its own modifier, and accumulates the four products
  // into `c`. It is the inner instruction of an int8 GEMM and both Triton and
  // cuBLAS emit it, so a kernel carrying one was being refused outright.
  //
  // It is pure integer arithmetic, which is why one expression is exact on both
  // targets, the columns cannot drift apart the way a `mul` on floats can.
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

  // `dp2a.mode.atype.btype d, a, b, c` is the 16x8 sibling of `dp4a`: `a` holds two
  // 16-bit lanes, `b` four bytes, and the mode picks which byte pair of `b` takes
  // part -- `.lo` the low two bytes, `.hi` the high two. Both of `a`'s lanes always
  // take part, one per byte of the pair, and each is folded in at its own
  // signedness: atype names a's lanes, btype names b's bytes, and `c` is signed
  // only when one of them is. Pure integer arithmetic again, so one expression is
  // exact on both columns.
  if (op == "dp2a" && args.size() >= 4) {
    bool sgn_a = true, sgn_b = true, hi = false, seen_a = false, seen_b = false;
    for (size_t i = 1; i < parts.size(); i++) {
      const std::string& p = parts[i];
      if (p == "lo") hi = false;
      else if (p == "hi") hi = true;
      else if (p == "u32" || p == "s32") {
        if (!seen_a) { sgn_a = (p == "s32"); seen_a = true; }
        else if (!seen_b) { sgn_b = (p == "s32"); seen_b = true; }
      } else if (p == "satfinite") {
        continue;  // unreachable at these widths, as for dp4a
      } else {
        err = "unsupported dp2a modifier '" + p + "'";
        return "";
      }
    }
    if (!seen_b) { err = "dp2a without both lane types in '" + opcode + "'"; return ""; }
    auto lane16 = [](const std::string& w, int i, bool sgn) {
      const std::string m = "(((uint32_t)(" + w + ") >> " + std::to_string(16 * i) +
                            ") & 0xFFFFu)";
      return sgn ? ("((int32_t)(int16_t)" + m + ")") : ("((int32_t)" + m + ")");
    };
    auto byte8 = [](const std::string& w, int i, bool sgn) {
      const std::string m = "(((uint32_t)(" + w + ") >> " + std::to_string(8 * i) +
                            ") & 0xFFu)";
      return sgn ? ("((int32_t)(int8_t)" + m + ")") : ("((int32_t)" + m + ")");
    };
    const std::string a = operand_as(args[1], "uint32_t");
    const std::string b = operand_as(args[2], "uint32_t");
    const int base = hi ? 2 : 0;
    // The sum is taken in 64 bits and cut to 32 at the end, so the wrap is the
    // 32-bit wrap the destination has and not an overflow nobody defined.
    std::string sum = "((int64_t)(" + operand_as(args[3], "uint32_t") + "))";
    for (int i = 0; i < 2; i++)
      sum += " + " + lane16(a, i, sgn_a) + " * " + byte8(b, base + i, sgn_b);
    return guard + store(args[0], "(uint32_t)(" + sum + ")", ty.empty() ? "int32_t" : ty);
  }

  // The scalar video instructions are one operation on the whole word, with the two
  // sources read at the types their suffixes name: `vmin.u32.u32.u32` on 0xFFFFFFFF
  // takes the unsigned maximum, and `vmin.s32.s32.s32` on the same bits takes -1 and is
  // therefore the minimum. The plain form is what a producer emits and its answer is the
  // operation cut back to the destination's width, so the arithmetic runs in 64 bits
  // where the sum of two words cannot overflow the language. `.sat` saturates instead of
  // wrapping, the `.asel`/`.bsel` selectors name a part of an operand, and `.add` folds
  // in a fourth operand; all three are refused by name rather than dropped.
  if ((op == "vadd" || op == "vsub" || op == "vmin" || op == "vmax" || op == "vabsdiff" ||
       op == "vmad") &&
      args.size() >= 3) {
    const bool mad_op = (op == "vmad");
    bool seen_sat = false, seen_add = false;
    std::string types[3];
    int ntypes = 0;
    for (size_t i = 1; i < parts.size(); i++) {
      const std::string& p = parts[i];
      if (p == "s32" || p == "u32") {
        if (ntypes == 3) { err = "'" + opcode + "' names three types"; return ""; }
        types[ntypes++] = p;
      } else if (p == "sat") {
        seen_sat = true;
      } else if (p == "add") {
        seen_add = true;
      } else {
        err = "'" + opcode + "' carries a qualifier the plain form does not lower: ." + p;
        return "";
      }
    }
    if (ntypes != 3) {
      err = "'" + opcode + "' names a destination type and two source types";
      return "";
    }
    if (seen_sat) { err = "'" + opcode + ".sat' saturates, which this does not"; return ""; }
    if (seen_add || (!mad_op && args.size() > 3)) {
      err = "'" + opcode + "': the fourth operand of the merge form is not translated";
      return "";
    }
    const int nops = mad_op ? 4 : 3;
    for (int i = 0; i < nops && (size_t)i < args.size(); i++) {
      const std::string t = trim(args[(size_t)i]);
      if (t.find('.') != std::string::npos || (!t.empty() && t[0] == '-')) {
        err = "a part selector or a negated operand on '" + opcode + "' is not translated";
        return "";
      }
    }
    const bool dsgn = (types[0] == "s32");
    const std::string dst = dsgn ? "int32_t" : "uint32_t";
    // The low 32 bits of a 32x32 product plus a third word are the same whether the
    // sources are signed or not, so one unsigned 64-bit expression is exact here.
    if (mad_op) {
      if (!need(4)) return "";
      const std::string p = "((uint64_t)(uint32_t)" + operand_as(args[1], "uint32_t") +
                            " * (uint64_t)(uint32_t)" + operand_as(args[2], "uint32_t") + ")";
      const std::string sum =
          "(" + p + " + (uint32_t)" + operand_as(args[3], "uint32_t") + ")";
      return guard + store(args[0], std::string("(") + dst + ") (uint32_t)(" + sum + ")", dst);
    }
    auto wide_of = [&](const std::string& tok, bool sgn) {
      return "((int64_t)" + std::string(sgn ? "(int32_t)" : "(uint32_t)") +
             operand_as(tok, sgn ? "int32_t" : "uint32_t") + ")";
    };
    const std::string ta = wide_of(args[1], types[1] == "s32");
    const std::string tb = wide_of(args[2], types[2] == "s32");
    std::string val;
    if (op == "vadd") val = "(" + ta + " + " + tb + ")";
    else if (op == "vsub") val = "(" + ta + " - " + tb + ")";
    else if (op == "vmin") val = "(" + ta + " < " + tb + " ? " + ta + " : " + tb + ")";
    else if (op == "vmax") val = "(" + ta + " > " + tb + " ? " + ta + " : " + tb + ")";
    else
      val = "(" + ta + " > " + tb + " ? " + ta + " - " + tb + " : " + tb + " - " + ta + ")";
    return guard + store(args[0], "(" + dst + ") (" + val + ")", dst);
  }

  // `vshl`/`vshr` are the scalar video shifts: one shift of the whole 32-bit word, and
  // the mode says what a count wider than the register means. `.clamp` stops the count
  // at 32 and `.wrap` takes its low five bits, which are two different answers for a
  // count of 36. Both run in 64 bits, so the clamped count of 32 is a shift the language
  // defines rather than one it does not, and the fill comes from the source type: signed
  // shifts bring the sign down and unsigned ones bring zero. The `.sat` form, the merge
  // and the `.asel`/`.bsel` selectors are refused by name rather than dropped, and a
  // count with no mode is refused too, because the isa leaves it undefined.
  if ((op == "vshl" || op == "vshr") && args.size() >= 3) {
    bool clamp = false, wrap = false, sat = false, sgn_a = false;
    std::string dtype;
    int word_types = 0;
    for (size_t i = 1; i < parts.size(); i++) {
      const std::string& p = parts[i];
      if (p == "clamp") clamp = true;
      else if (p == "wrap") wrap = true;
      else if (p == "sat") sat = true;
      else if (p == "s32" || p == "u32") {
        if (++word_types == 1) dtype = p;
        else if (word_types == 2) sgn_a = (p == "s32");
        else if (p != "u32") {
          err = "'" + opcode + "': the shift amount is .u32";
          return "";
        }
      } else {
        err = "'" + opcode + "' carries a qualifier the scalar form does not lower: ." + p;
        return "";
      }
    }
    if (word_types < 3) {
      err = "'" + opcode + "' names a destination type, a source type and an amount type";
      return "";
    }
    if (!clamp && !wrap) {
      err = "'" + opcode + "' needs .clamp or .wrap: without one a count past the width is undefined";
      return "";
    }
    if (sat) { err = "'" + opcode + ".sat' saturates, which this does not"; return ""; }
    for (int i = 0; i < 3; i++)
      if (args[(size_t)i].find('.') != std::string::npos) {
        err = "a part selector on '" + opcode + "' is not translated";
        return "";
      }
    const std::string amt = "((uint32_t)" + operand_as(args[2], "uint32_t") + ")";
    const std::string cnt =
        wrap ? ("(" + amt + " & 31u)") : ("(" + amt + " > 32u ? 32u : " + amt + ")");
    const std::string src = sgn_a
                                ? ("((int64_t)(int32_t)" + operand_as(args[1], "int32_t") + ")")
                                : ("((uint64_t)(uint32_t)" + operand_as(args[1], "uint32_t") + ")");
    const std::string sh =
        op == "vshl" ? (src + " << " + cnt) : (src + " >> " + cnt);
    const bool dsgn = (dtype == "s32");
    return guard + store(args[0], (dsgn ? "(int32_t)(" : "(uint32_t)(") + sh + ")",
                         dsgn ? "int32_t" : "uint32_t");
  }

  // `vset.atype.btype.cmp d, a, b` compares two words at the types the suffixes name and
  // answers 1 or 0, so a predicate lands in a register. The two types must agree: a
  // signed word against an unsigned one has no comparison the isa defines, and a
  // sign-extended word read as unsigned is a different answer rather than a rounding.
  // The merge form and the part selectors are refused, as they are next door.
  if (op == "vset" && args.size() >= 3) {
    std::string at, bt, cmp;
    for (size_t i = 1; i < parts.size(); i++) {
      const std::string& p = parts[i];
      if (p == "s32" || p == "u32") {
        if (at.empty()) at = p;
        else if (bt.empty()) bt = p;
        else { err = "'" + opcode + "' names two types and a comparison"; return ""; }
      } else if (p == "eq" || p == "ne" || p == "lt" || p == "le" || p == "gt" || p == "ge") {
        cmp = p;
      } else {
        err = "'" + opcode + "' carries a qualifier this form does not lower: ." + p;
        return "";
      }
    }
    if (at.empty() || bt.empty() || cmp.empty()) {
      err = "'" + opcode + "' names two source types and a comparison";
      return "";
    }
    for (int i = 0; i < 2; i++)
      if (args[(size_t)i].find('.') != std::string::npos) {
        err = "a part selector on '" + opcode + "' is not translated";
        return "";
      }
    // Both sides are widened to 64 bits and compared there. An unsigned word becomes a
    // positive 64-bit value and a signed one keeps its sign, so a single comparison is
    // right for both, and the two types are allowed to differ: that is a form the isa
    // defines, and widening first is what makes `s32` against `u32` well defined.
    auto wide_of = [&](const std::string& tok, bool sgn) {
      return "((int64_t)" + std::string(sgn ? "(int32_t)" : "(uint32_t)") +
             operand_as(tok, sgn ? "int32_t" : "uint32_t") + ")";
    };
    const std::string a = wide_of(args[1], at == "s32");
    const std::string b = wide_of(args[2], bt == "s32");
    const std::string sym = cmp == "eq" ? "==" : cmp == "ne" ? "!=" : cmp == "lt" ? "<"
                            : cmp == "le" ? "<=" : cmp == "gt" ? ">" : ">=";
    return guard + store(args[0], "((" + a + " " + sym + " " + b + ") ? 1u : 0u)", "uint32_t");
  }

  // `vset2`/`vset4` are the packed comparisons: each half (two lanes) or byte (four) of
  // the two sources is compared at its own element type and the answer is 1 or 0 in that
  // same field, which is not the all-ones mask a vector compare gives. A lane of a
  // `.u32` source is an unsigned half or byte, so the same bits compared as `.s32` are a
  // different answer, and one row of the fixture is there to say so. The destination mask,
  // the part selectors and the `.add` form are the merge variants and are refused by
  // name rather than dropped.
  if ((op == "vset2" || op == "vset4") && args.size() >= 3) {
    const int lanes = (op == "vset2") ? 2 : 4;
    const int width = 32 / lanes;
    std::string at, bt, cmp;
    for (size_t i = 1; i < parts.size(); i++) {
      const std::string& p = parts[i];
      if (p == "s32" || p == "u32") {
        if (at.empty()) at = p;
        else if (bt.empty()) bt = p;
        else { err = "'" + opcode + "' names two element types and a comparison"; return ""; }
      } else if (p == "eq" || p == "ne" || p == "lt" || p == "le" || p == "gt" || p == "ge") {
        cmp = p;
      } else {
        err = "'" + opcode + "' carries a qualifier this form does not lower: ." + p;
        return "";
      }
    }
    if (at.empty() || bt.empty() || cmp.empty()) {
      err = "'" + opcode + "' names two element types and a comparison";
      return "";
    }
    // The packed form names its merge operand `c` even when no mask selects it, and the
    // mask that is absent means every lane, so `c` is not read here. A mask on the
    // destination or a selector on any operand makes it matter, and those are refused.
    if (args.size() > 4) {
      err = "'" + opcode + "' has more operands than the packed form does";
      return "";
    }
    for (size_t i = 0; i < args.size(); i++)
      if (args[i].find('.') != std::string::npos) {
        err = "a part selector on '" + opcode + "' is not translated";
        return "";
      }
    const std::string field = lanes == 2 ? "0xFFFFu" : "0xFFu";
    auto lane_val = [&](const std::string& tok, bool sgn, int i) {
      const std::string m = "(((uint32_t)(" + operand_as(tok, "uint32_t") + ") >> " +
                            std::to_string(width * i) + ") & " + field + ")";
      return sgn ? ("((int32_t)(int" + std::to_string(width == 16 ? 16 : 8) + "_t)" + m + ")")
                 : ("((int32_t)" + m + ")");
    };
    const std::string sym = cmp == "eq" ? "==" : cmp == "ne" ? "!=" : cmp == "lt" ? "<"
                            : cmp == "le" ? "<=" : cmp == "gt" ? ">" : ">=";
    std::string packed = "0u";
    for (int i = 0; i < lanes; i++) {
      const std::string x = lane_val(args[1], at == "s32", i);
      const std::string y = lane_val(args[2], bt == "s32", i);
      const std::string t = "((" + x + " " + sym + " " + y + ") ? 1u : 0u)";
      packed = "(" + packed + " | (" + t + " << " + std::to_string(width * i) + "))";
    }
    return guard + store(args[0], packed, "uint32_t");
  }

  //
  // The video instructions are packed arithmetic that does not carry between lanes.
  // The plain form is what a producer emits -- no `.asel`/`.bsel`, no destination
  // mask, no `.add` -- and there each lane of `a` and `b` is widened at the signedness
  // its own suffix names, the operation runs in 32 bits, and the answer is cut back
  // to the lane width. The fourth operand `c` belongs to the merge form and is not
  // read here, which is why a `.add` or a selector is refused by name instead of
  // being accepted and quietly dropped.
  {
    int lanes = 0;
    std::string vop;
    if (op == "vadd2") { lanes = 2; vop = "add"; }
    else if (op == "vsub2") { lanes = 2; vop = "sub"; }
    else if (op == "vmin2") { lanes = 2; vop = "min"; }
    else if (op == "vmax2") { lanes = 2; vop = "max"; }
    else if (op == "vavrg2") { lanes = 2; vop = "avrg"; }
    else if (op == "vabsdiff2") { lanes = 2; vop = "absdiff"; }
    else if (op == "vadd4") { lanes = 4; vop = "add"; }
    else if (op == "vsub4") { lanes = 4; vop = "sub"; }
    else if (op == "vmin4") { lanes = 4; vop = "min"; }
    else if (op == "vmax4") { lanes = 4; vop = "max"; }
    else if (op == "vavrg4") { lanes = 4; vop = "avrg"; }
    else if (op == "vabsdiff4") { lanes = 4; vop = "absdiff"; }
    if (lanes) {
      if (!need(4)) return "";
      if (args[0].find('.') != std::string::npos || args[1].find('.') != std::string::npos ||
          args[2].find('.') != std::string::npos) {
        err = "the merge form of '" + op +
              "' (a destination mask or an .asel/.bsel selector) is not translated";
        return "";
      }
      bool sgn_a = false, sgn_b = false;
      int ti = 0;
      for (size_t i = 1; i < parts.size(); i++) {
        const std::string& p = parts[i];
        if (p == "sat" || p == "add") {
          err = "'" + op + "' with ." + p + " is the merge form and is not translated";
          return "";
        }
        if (p != "u32" && p != "s32") {
          err = "unsupported " + op + " operand type '." + p + "'";
          return "";
        }
        if (ti == 1) sgn_a = (p == "s32");
        if (ti == 2) sgn_b = (p == "s32");
        ti++;
      }
      if (ti < 3) {
        err = op + " needs three operand types: destination, a and b";
        return "";
      }
      const int width = lanes == 2 ? 16 : 8;
      const std::string lm = width == 16 ? "0xFFFFu" : "0xFFu";
      auto ext = [&](const std::string& w, int i, bool sgn) {
        const std::string m = "(((uint32_t)(" + w + ") >> " + std::to_string(width * i) +
                              ") & " + lm + ")";
        if (width == 16)
          return sgn ? ("((int32_t)(int16_t)" + m + ")") : ("((int32_t)" + m + ")");
        return sgn ? ("((int32_t)(int8_t)" + m + ")") : ("((int32_t)" + m + ")");
      };
      const std::string a = operand_as(args[1], "uint32_t");
      const std::string b = operand_as(args[2], "uint32_t");
      std::string packed = "0u";
      for (int i = 0; i < lanes; i++) {
        const std::string x = ext(a, i, sgn_a);
        const std::string y = ext(b, i, sgn_b);
        std::string t;
        if (vop == "add") t = "(" + x + " + " + y + ")";
        else if (vop == "sub") t = "(" + x + " - " + y + ")";
        else if (vop == "min") t = "(" + x + " < " + y + " ? " + x + " : " + y + ")";
        else if (vop == "max") t = "(" + x + " > " + y + " ? " + x + " : " + y + ")";
        else if (vop == "absdiff")
          t = "(" + x + " > " + y + " ? " + x + " - " + y + " : " + y + " - " + x + ")";
        else
          t = "(((" + x + " + " + y + ") >= 0) ? ((" + x + " + " + y +
              " + 1) >> 1) : ((" + x + " + " + y + ") >> 1))";
        const std::string cut = "(((uint32_t)" + t + ") & " + lm + ")";
        packed = "(" + packed + " | (" + cut + " << " + std::to_string(width * i) + "))";
      }
      return guard + store(args[0], packed, ty.empty() ? "uint32_t" : ty);
    }
  }

  if (op == "clz" && args.size() >= 2)
    return guard + store(args[0], "(uint32_t)__builtin_clz(" + operand_as(args[1], ty) + ")", ty);
  if (op == "popc" && args.size() >= 2)
    return guard + store(args[0], "(uint32_t)__builtin_popcount(" + operand_as(args[1], ty) + ")", ty);
  if (op == "brev" && args.size() >= 2)
    return guard + store(args[0], "weft_bitrev32(" + operand_as(args[1], ty) + ")", ty);

  // `bswap.b32 d, a` and `bswap.b64 d, a` reverse the bytes of the register, which is
  // how a kernel turns a word read from memory into host order. It is not `brev`: the
  // bits move eight at a time and stay inside their byte, so a value with one byte set
  // comes back with that byte at the other end. The 64-bit form reverses the eight bytes
  // as a whole, so its two 32-bit halves trade places too.
  if (op == "bswap" && args.size() >= 2) {
    const bool wide = (ty == "uint64_t" || ty == "int64_t");
    return guard + store(args[0],
                         std::string(wide ? "weft_bswap64(" : "weft_bswap32(") +
                             operand_as(args[1], wide ? "uint64_t" : "uint32_t") + ")",
                         ty.empty() ? (wide ? "uint64_t" : "uint32_t") : ty);
  }

  // `cnot.type d, a` is the C logical negation and not the bitwise one: the answer is 1
  // when the register is zero and 0 when it is anything else, so a word of all ones
  // answers 0 rather than `not`'s all zeros. The width only picks the register to read,
  // which is why `.b16`, `.b32` and `.b64` all have the same answer.
  if (op == "cnot" && args.size() >= 2)
    return guard + store(args[0], "(" + operand_as(args[1], ty) + " == 0 ? 1 : 0)",
                         ty.empty() ? "uint32_t" : ty);

  // `bfind.u32 d, a` is the position of the highest set bit, counted from the least
  // significant end, and 0xFFFFFFFF when there is no set bit. `.s32` searches for
  // the highest bit that differs from the sign, so a negative value is searched
  // inverted and both 0 and -1 have no such bit; `.shiftamt` answers the shift that
  // moves that bit to the sign position, 31 - pos, and 0xFFFFFFFF for none. All
  // three are one helper on both columns, so the answer cannot differ between them.
  if (op == "bfind" && args.size() >= 2) {
    const bool sgn = ty == "int32_t";
    bool shiftamt = false;
    for (const auto& p : parts)
      if (p == "shiftamt") shiftamt = true;
    const std::string call = std::string(sgn ? "weft_bfind_s32(" : "weft_bfind_u32(") +
                             operand_as(args[1], sgn ? "int32_t" : "uint32_t") + ")";
    const std::string body =
        shiftamt ? "(" + call + " == 0xFFFFFFFFu ? 0xFFFFFFFFu : (31u - " + call + "))" : call;
    return guard + store(args[0], body, ty.empty() ? "uint32_t" : ty);
  }
  // `fns.b32 d, mask, base, offset` is the position of the offset-th set bit of
  // `mask`, counting upward from `base`, and 0xFFFFFFFF if it never meets one.
  if (op == "fns" && args.size() >= 4)
    return guard + store(args[0],
                         "weft_fns_b32(" + operand_as(args[1], "uint32_t") + ", " +
                             operand_as(args[2], "uint32_t") + ", " +
                             operand_as(args[3], "uint32_t") + ")",
                         ty.empty() ? "uint32_t" : ty);

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
  // `mufu.<op>.approx.f32` is the multifunction unit's own approximation, the
  // only mode it has, and it is the same bargain as the `.approx` forms above:
  // the silicon's polynomial, which no CPU reproduces. So it runs the portable
  // library like everything else, and one expression is exact on both columns.
  if (op == "mufu" && args.size() >= 2) {
    std::string f;
    for (size_t i = 1; i < parts.size(); i++) {
      const std::string& p = parts[i];
      if (p == "rcp" || p == "rsqrt" || p == "sqrt" || p == "sin" || p == "cos" ||
          p == "ex2" || p == "lg2" || p == "tanh" || p == "tan")
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
  if (op == "tan" && args.size() >= 2)
    return guard + store(args[0], "weft_tan(" + operand_as(args[1], ty) + ")", ty);

  //
  // ldmatrix and mma are warp-wide: a fragment is distributed across the 32
  // lanes, so one lane's operand is another lane's register. That is expressible
  // only where the whole warp's registers are addressable at once, which is the
  // warp backend; the per-lane state machine holds one lane's registers and has
  // no way to read a peer's without a rendezvous. A kernel that uses them is
  // therefore emitted warp-only, and the module's table records that.
  //
  // `redux.sync.op d, a, mask` folds one value across the lanes of the warp: what
  // `__reduce_add_sync` compiles to, and what Triton emits for a `tl.sum`/`tl.max`
  // over an integer block. In the warp backend every lane's registers are
  // addressable at once, so the fold is a butterfly over the register array , 
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
    // Three families: an 8x8 tile of 16-bit elements, a 16x16 tile of bytes, and the
    // sub-byte shapes, which widen a 6- or 4-bit element to a byte on the way in.
    const bool b8 = has_part(parts, "b8");
    const bool p6 = has_part(parts, "b6x16_p32");
    const bool p4 = has_part(parts, "b4x16_p64") || has_part(parts, "s4");
    const bool packed = p6 || p4;
    const bool sgn = has_part(parts, "s4");
    const bool m16 = has_part(parts, "m16n16");
    if (!has_part(parts, "m8n8") && !(b8 && m16) &&
        !(packed && (has_part(parts, "m8n16") || m16))) {
      err = "unsupported ldmatrix shape: an 8x8 tile of 16-bit elements, the 16x16 "
            "tile of bytes, and the 8x16 or 16x16 tile of 6- and 4-bit elements";
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
    // A lane holds four bytes of a container row, so a tile is rows/8 registers a lane:
    // one for the 8-row shapes, two for the 16-row ones.
    const int rows = m16 ? 16 : 8;
    const int per = packed ? rows / 8 : (b8 ? 2 : 1);
    const int want_regs = per * count;
    if ((int)regs.size() != want_regs) {
      const std::string fmt = packed
                                  ? (p6 ? ".b6x16_p32" : (sgn ? ".s8.s4" : ".b4x16_p64"))
                                  : (b8 ? ".b8" : "");
      err = "ldmatrix.x" + std::to_string(count) + fmt + " takes " +
            std::to_string(want_regs) + " destination registers";
      return "";
    }
    std::string addr, sym;
    if (!mem_operand(args[1], addr, sym)) return "";
    // A named symbol is `[tile]` or `[tile+128]`: the address a lane would supply
    // is a place inside the shared window that is known at translation time, so
    // nothing has to be read out of a peer's register, the rows are the 16-byte
    // ones the tile was declared with.
    // `addr` already carries the array's offset in the shared window and any constant
    // in the operand, so the symbol is only here to say whether the address is shared
    // by the warp or a per-lane register. Re-deriving the base from the bare name threw
    // the constant away: `[tile+256]` read `tile`.
    if (!sym.empty() && shared_offsets.find(sym) == shared_offsets.end()) {
      err = "'" + sym + "' is not a shared array: ldmatrix reads threadgroup memory";
      return "";
    }
    // The address a lane supplies points at one row of 8 halves, and another
    // lane is the one that reads it: matrix k's row i is addressed by lane 8k+i,
    // and lane l wants row l>>2 of it. So the address is read out of the peer's
    // slot, not this lane's.
    // A register address names one row of the tile, and the peer is the lane that
    // holds it, so the row is read out of that slot. A named symbol names the whole
    // tile instead, so the row is an offset from its start, 16 bytes per row.
    const bool uniform = !sym.empty();
    auto row_addr = [&](const std::string& who) -> std::string {
      if (uniform) return "((uint64_t)(" + addr + ") + 16u * (" + who + "))";
      std::string s = addr;
      for (size_t p = 0; (p = s.find("[l]", p)) != std::string::npos;)
        s.replace(p, 3, "[" + who + "]");
      return s;
    };
    if (packed) {
      // The sub-byte shapes. The source format is the ISA's: one 16-byte container
      // holds sixteen 6-bit elements in its first 96 bits, or sixteen 4-bit elements
      // in its first 64, and the rest of the container is padding. The destination
      // side is CUTLASS's, `SM100_SU6_DU8x16_x1_LDSM_N` and its fp4 twin in
      // `cute/atom/copy_traits_sm100.hpp` say the warp reads eight containers (lane t
      // supplies container t&7) and that a lane's slice is the 32 bits at 32*t of a
      // 1024-bit reference, four consecutive bytes in order. Composed with a container
      // being one row of sixteen elements, lane l holds row l>>2 at columns
      // 4*(l&3)..+3, and a 16-row tile is the same rule with eight bytes a lane.
      const int bits = p6 ? 6 : 4;
      vec_side = true;
      std::string out = "{ ";
      for (int m = 0; m < count; m++) {
        for (int rg = 0; rg < per; rg++) {
          std::vector<std::string> bs;
          for (int q = 0; q < 4; q++) {
            const int j = 4 * rg + q;  // which byte of this lane's slice
            const std::string idx =
                "(" + std::to_string(4 * per) + "u * l + " + std::to_string(j) + "u)";
            const std::string row = "(" + idx + " >> 4)";
            const std::string col = "(" + idx + " & 15u)";
            std::string chunk;
            if (uniform) {
              chunk = "((uint64_t)(" + addr + ") + " + std::to_string(16 * rows * m) +
                      "u + 16u * " + row + ")";
            } else {
              // Container `rows*m + row` is the one lane `rows*m + row` supplied, the
              // same pairing the byte shape uses.
              const std::string peer = "(" + std::to_string(rows * m) + "u + " + row + ")";
              std::string s2 = addr;
              for (size_t p2 = 0; (p2 = s2.find("[l]", p2)) != std::string::npos;)
                s2.replace(p2, 3, "[" + peer + "]");
              chunk = "(" + s2 + ")";
            }
            std::string v;
            if (bits == 4) {
              // Two elements a byte, low nibble first, and the payload stops after
              // eight bytes, the second half of the container is padding.
              v = "((uint32_t)*(" + addr_space("threadgroup", true) +
                  "uint8_t*)(sh + " + chunk + " + (" + col + " >> 1)) >> (4u * (" + col +
                  " & 1u))) & 0xFu";
              if (sgn)
                v = "((uint32_t)(((int32_t)((" + v + ") << 28)) >> 28) & 0xFFu)";
            } else {
              // Six-bit fields straddle byte boundaries, so the field is the two bytes
              // its start falls in, shifted down. The last column starts in byte 11, so
              // the pair always lies inside the container.
              const std::string at = "sh + " + chunk + " + ((3u * " + col + ") >> 2)";
              v = "(((((uint32_t)*(" + addr_space("threadgroup", true) + "uint8_t*)(" + at +
                  ")) | (((uint32_t)*(" + addr_space("threadgroup", true) +
                  "uint8_t*)(" + at + " + 1u)) << 8)) >> ((6u * " + col +
                  ") & 7u)) & 0x3Fu)";
            }
            // Parenthesised here rather than at the assembly below: the field
            // expression ends in a mask, and `& 0xFu << 8` would bind the shift to the
            // mask and zero the byte.
            bs.push_back("(" + v + ")");
          }
          const std::string w = "weft_pw" + std::to_string(m) + std::to_string(rg);
          out += "const uint32_t " + w + " = " + bs[0] + " | (" + bs[1] + " << 8) | (" +
                 bs[2] + " << 16) | (" + bs[3] + " << 24); ";
          out += store(regs[(size_t)(per * m + rg)], w, "uint32_t");
          out += " ";
        }
      }
      out += "}";
      return "for (uint32_t l = 0; l < 32u; l++) { " + out + " }";
    }

    if (b8) {
      // The byte shape. Where each byte comes from is fixed by two halves of one
      // statement in CUTLASS (`cute/arch/copy_sm100.hpp` and `copy_traits_sm100.hpp`):
      // the reference layout of `SM100_U8x8_LDSM_T` says the fragment covers a 16x16
      // tile as
      //     row = 2*(l&3) + (s&1) + 8*(s>>2),  col = (l>>2) + 8*((s>>1)&1)
      // for logical byte s of a lane, and the shuffle its `copy()` applies to the raw
      // result says logical byte s is byte q of register r where
      //     s = 2*r + (q&1) + 4*(q>>1).
      // Composing them is the hardware's own answer, which is why a kernel written
      // against CUTLASS sees the bytes it expects. Two registers a matrix.
      vec_side = true;
      std::string out = "{ ";
      for (int m = 0; m < count; m++) {
        for (int r = 0; r < 2; r++) {
          std::vector<std::string> bs;
          for (int q = 0; q < 4; q++) {
            const int s = 2 * r + (q & 1) + 4 * (q >> 1);
            const int a = s & 1, bq = (s >> 1) & 1, c = s >> 2;
            const std::string row = "2u * (l & 3u) + " + std::to_string(a + 8 * c) + "u";
            const std::string col = "(l >> 2) + " + std::to_string(8 * bq) + "u";
            std::string at;
            if (uniform) {
              at = "((uint64_t)(" + addr + ") + " + std::to_string(256 * m) + "u + 16u * (" +
                   row + ") + (" + col + "))";
            } else {
              // The address lane `16m + f(row)` supplied is the start of that row, so
              // the peer's slot plus the column is the byte. f is the inverse of the
              // row map above, which collapses to 4*(l&3) + a + 2c.
              const std::string peer = "(" + std::to_string(16 * m) + "u + 4u * (l & 3u) + " +
                                       std::to_string(a + 2 * c) + "u)";
              std::string s2 = addr;
              for (size_t p = 0; (p = s2.find("[l]", p)) != std::string::npos;)
                s2.replace(p, 3, "[" + peer + "]");
              at = "(" + s2 + " + (" + col + "))";
            }
            bs.push_back("(uint32_t)*(" + addr_space("threadgroup", true) +
                         "uint8_t*)(sh + " + at + ")");
          }
          const std::string w = "weft_bw" + std::to_string(m) + std::to_string(r);
          out += "const uint32_t " + w + " = " + bs[0] + " | (" + bs[1] + " << 8) | (" +
                 bs[2] + " << 16) | (" + bs[3] + " << 24); ";
          out += store(regs[(size_t)(2 * m + r)], w, "uint32_t");
          out += " ";
        }
      }
      out += "}";
      return "for (uint32_t l = 0; l < 32u; l++) { " + out + " }";
    }
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

  // activemask.b32 d;  -- one bit per lane of this warp that is executing.
  //
  // The warp body already carries that as the predicate it dispatches on, so the
  // mask is a fold over it and the answer is the same number in every lane that ran:
  // a converged kernel reports all of its live lanes, and one whose lanes are on
  // different paths reports the block the scheduler ran, which is exactly what the
  // instruction promises.
  if (op == "activemask") {
    if (!vec) {
      warp_only = true;
      err = "activemask answers which lanes of the warp are executing, which only the "
            "warp body knows";
      return "";
    }
    if (!need(1)) return "";
    // A predicated instruction deactivates its lane just as a branch does, and the
    // spec counts a predicated-off lane as a 0 in this mask. The predicate is a
    // per-lane expression, so the fold runs over the lane index instead of being
    // unrolled with a constant one.
    const std::string p = vec_pred.empty() ? "1" : vec_pred;
    vec_side = true;
    return "{ uint32_t weft_m = 0; for (uint32_t l = 0; l < 32u; l++) if (am[l] && (" + p +
           ")) weft_m |= 1u << l; for (uint32_t l = 0; l < 32u; l++) if (am[l] && (" + p +
           ")) { " + store(args[0], "weft_m", "uint32_t") + " } }";
  }

  // `movmatrix` is the transpose of the fragment `ldmatrix` builds and `stmatrix`
  // takes apart, laid out the same way: lane l holds row l>>2, at columns 2(l&3) and
  // 2(l&3)+1. Transposed, this lane holds columns l>>2 of the matrix at rows 2(l&3)
  // and 2(l&3)+1, that is original elements (2(l&3), l>>2) and (2(l&3)+1, l>>2).
  // Element (c, r) of the original lives in lane 4c+(r>>1), in the half that r&1
  // names, so the two halves this lane needs come from peers four lanes apart -- and
  // because the map is the one the two matrix instructions already agree on, a lane
  // layout that drifted would break the round trip rather than go unnoticed.
  if (op == "movmatrix") {
    if (!vec) {
      warp_only = true;
      err = "movmatrix reads registers belonging to other lanes, so it needs the warp backend";
      return "";
    }
    if (!need(2)) return "";
    if (!has_part(parts, "m8n8") || !has_part(parts, "b16")) {
      err = "unsupported movmatrix form: the 8x8 tile of 16-bit elements, transposed";
      return "";
    }
    Reg src = parse_reg(args[1]);
    Reg dst = parse_reg(args[0]);
    if (!src.ok || !dst.ok) {
      err = "movmatrix takes one destination and one source register";
      return "";
    }
    const std::string sreg = "s->" + src.base + "[" + std::to_string(src.index) + "]";
    const std::string dreg = "s->" + dst.base + "[" + std::to_string(dst.index) + "]";
    auto rt = reg_types.find(src.base);
    const bool fl = rt != reg_types.end() && rt->second == "float";
    auto word = [&](const std::string& lane) {
      return fl ? ("weft_bits(" + sreg + "[" + lane + "])")
                : ("((uint32_t)" + sreg + "[" + lane + "])");
    };
    // Without `.trans` the source and destination are the same matrix in the same
    // fragment layout, so the move is the identity: the lane keeps the pair it holds.
    // This was refused for a while because it measured as a bus error at a full warp.
    // The fault was in the fixture that fed it, which added the output address to an
    // address that already held the input address, so the store landed at in+out+tid*4.
    // With that fixed the identity runs at 1, 8 and 32 lanes and round-trips the warp.
    if (!has_part(parts, "trans")) {
      std::string id = "for (uint32_t l = 0; l < 32u; l++) { ";
      id += "const uint32_t weft_w = " + word("l") + "; ";
      id += "if (am[l]) " + dreg + "[l] = weft_w; }";
      return id;
    }
    std::string out = "for (uint32_t l = 0; l < 32u; l++) { ";
    out += "const uint32_t weft_r = (l >> 2) & 7u; const uint32_t weft_c = 2u * (l & 3u); ";
    out += "const uint32_t weft_p = 4u * weft_c + (weft_r >> 1u); ";
    out += "const uint32_t weft_w0 = " + word("weft_p") + "; ";
    out += "const uint32_t weft_w1 = " + word("weft_p + 4u") + "; ";
    out += "const uint32_t weft_sh = 16u * (weft_r & 1u); ";
    out += "const uint32_t weft_lo = (weft_w0 >> weft_sh) & 0xFFFFu; ";
    out += "const uint32_t weft_hi = (weft_w1 >> weft_sh) & 0xFFFFu; ";
    out += "if (am[l]) " + dreg + "[l] = weft_lo | (weft_hi << 16); }";
    return out;
  }

  // `stmatrix` is `ldmatrix` backwards: the same fragment map, walked from the
  // registers into shared memory. Lane l holds row l>>2 and the two halves at
  // columns 2(l&3) and 2(l&3)+1, and the address it stores through is the one lane
  // 8m+(l>>2) supplied, the same pairing the load reads through, which is what
  // makes the two instructions a round trip rather than two maps that happen to
  // look alike. `.trans` stores the fragment transposed: the two halves go to two
  // rows of the column this lane owns.
  if (op == "stmatrix") {
    if (!vec) {
      warp_only = true;
      err = "stmatrix hands registers belonging to other lanes to shared memory, so it "
            "needs the warp backend";
      return "";
    }
    if (!need(2)) return "";
    const bool b8 = has_part(parts, "b8");
    if (!has_part(parts, "m8n8") && !(b8 && has_part(parts, "m16n8"))) {
      err = "unsupported stmatrix shape: an 8x8 tile of 16-bit elements, and the 16x8 "
            "tile of bytes";
      return "";
    }
    int count = 1;
    for (const auto& p : parts)
      if (p.size() == 2 && p[0] == 'x' && p[1] >= '1' && p[1] <= '4') count = p[1] - '0';
    const bool trans = has_part(parts, "trans");
    // The other way round from `ldmatrix`: the address is first and the registers
    // are last.
    std::string srcs = trim(args.back());
    if (srcs.find('{') == std::string::npos) {
      err = "stmatrix needs a register list";
      return "";
    }
    srcs = srcs.substr(srcs.find('{') + 1);
    srcs = srcs.substr(0, srcs.find('}'));
    const std::vector<std::string> regs = split_top(srcs, ',');
    if ((int)regs.size() != count) {
      err = "stmatrix.x" + std::to_string(count) + " takes " + std::to_string(count) +
            " source registers";
      return "";
    }
    std::string addr, sym;
    if (!mem_operand(args[0], addr, sym)) return "";
    // `addr` already carries the array's offset inside the shared window *and* any
    // constant written in the operand, so the symbol is only here to say whether the
    // address is one the warp shares or a per-lane register. Re-deriving the base
    // from the bare name threw the constant away: `[tile+256]` stored at `tile`.
    const bool uniform = !sym.empty();
    if (uniform && shared_offsets.find(sym) == shared_offsets.end()) {
      err = "'" + sym + "' is not a shared array: stmatrix writes threadgroup memory";
      return "";
    }
    auto row_addr = [&](const std::string& who) -> std::string {
      if (uniform) return "((uint64_t)(" + addr + ") + 16u * (" + who + "))";
      std::string s = addr;
      for (size_t p = 0; (p = s.find("[l]", p)) != std::string::npos;)
        s.replace(p, 3, "[" + who + "]");
      return s;
    };
    // The word a lane's register holds, as the 32 bits that are the two halves.
    auto word = [&](const std::string& tok, const std::string& lane) -> std::string {
      Reg r = parse_reg(tok);
      if (!r.ok) {
        err = "expected a register, got '" + tok + "'";
        return std::string("0u");
      }
      const std::string v = "s->" + r.base + "[" + std::to_string(r.index) + "][" + lane + "]";
      auto it = reg_types.find(r.base);
      const bool fl = it != reg_types.end() && it->second == "float";
      return fl ? ("weft_bits(" + v + ")") : ("(uint32_t)(" + v + ")");
    };
    if (b8) {
      // The byte shape, and the other half of the same statement: CUTLASS's
      // `SM100_U8x4_STSM_T` reference layout puts logical byte s of a lane at
      //     row = 4*(l&3) + 2*(s&1) + ((s>>1)&1),  col = l>>2
      // of a 16x8 tile, one register a matrix, and it applies no shuffle to the source
      // registers, so this is the store's own order rather than an inversion of one.
      vec_side = true;
      std::string out = "{ ";
      for (int m = 0; m < count; m++) {
        const std::string w = "weft_sw" + std::to_string(m);
        out += "const uint32_t " + w + " = " + word(regs[(size_t)m], "l") + "; ";
        for (int q = 0; q < 4; q++) {
          const int a = q & 1, bq = (q >> 1) & 1;
          const std::string row = "4u * (l & 3u) + " + std::to_string(2 * a + bq) + "u";
          std::string at;
          if (uniform) {
            at = "((uint64_t)(" + addr + ") + " + std::to_string(128 * m) + "u + 8u * (" +
                 row + ") + (l >> 2))";
          } else {
            // The address operand names a 16-byte block of the tile, two rows at a
            // time, supplied by lane `8m + block`: eight addresses a matrix, the
            // groups of eight lanes the PTX ISA gives each matrix. The byte inside
            // that block is the row parity times eight plus the column.
            const std::string peer = "(" + std::to_string(8 * m) + "u + 2u * (l & 3u) + " +
                                     std::to_string(a) + "u)";
            std::string s2 = addr;
            for (size_t p = 0; (p = s2.find("[l]", p)) != std::string::npos;)
              s2.replace(p, 3, "[" + peer + "]");
            at = "(" + s2 + " + " + std::to_string(8 * bq) + "u + (l >> 2))";
          }
          out += "*(" + addr_space("threadgroup", false) + "uint8_t*)(sh + " + at +
                 ") = (uint8_t)((" + w + " >> " + std::to_string(8 * q) + ") & 0xFFu); ";
        }
      }
      out += "}";
      return "for (uint32_t l = 0; l < 32u; l++) { " + out + " }";
    }
    vec_side = true;
    std::string out = "{ const uint32_t weft_r = (l >> 2) & 7u; const uint32_t weft_c = (l & 3u); ";
    for (int k = 0; k < count; k++) {
      const std::string m = std::to_string(k);
      const std::string w = "weft_w" + m;
      out += "const uint32_t " + w + " = " + word(regs[(size_t)k], "l") + "; ";
      if (trans) {
        out += "*(" + addr_space("threadgroup", false) +
               "uint16_t*)(sh + " + row_addr("8u * " + m + "u + 2u * weft_c") +
               " + 2u * weft_r) = (uint16_t)(" + w + " & 0xFFFFu); ";
        out += "*(" + addr_space("threadgroup", false) +
               "uint16_t*)(sh + " + row_addr("8u * " + m + "u + 2u * weft_c + 1u") +
               " + 2u * weft_r) = (uint16_t)(" + w + " >> 16); ";
      } else {
        out += "*(" + addr_space("threadgroup", false) +
               "uint32_t*)(sh + " + row_addr("8u * " + m + "u + weft_r") + " + 4u * weft_c) = " +
               w + "; ";
      }
    }
    out += "}";
    return "for (uint32_t l = 0; l < 32u; l++) { " + out + " }";
  }

  //
  // One instruction is issued by a warpgroup, 128 lanes, and reads *both*
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
    // scale-b, transpose-a, transpose-b. ptxas wants all five, but the dumps write
    // shorter lists, so a missing one reads as the identity: zero for the flags,
    // and for the two scales, where zero and an explicit 1 are the same answer.
    const size_t nargs = args.size();
    if (nargs > 8) {
      err = "wgmma '" + opcode + "' takes at most 5 immediates (scale-d, then "
            "scale-a scale-b, then transpose-a transpose-b), got " +
            std::to_string(nargs - 3);
      return "";
    }
    int imm[5] = {0, 0, 0, 0, 0};
    static const char* const kImmName[5] = {"scale-d", "scale-a", "scale-b",
                                            "transpose-a", "transpose-b"};
    for (size_t i = 0; i < 5 && 3 + i < nargs; i++) {
      const std::string v = trim(args[3 + i]);
      if (v != "0" && v != "1" && v != "-1") {
        err = "wgmma '" + opcode + "': the " + kImmName[i] +
              " immediate must be 0, 1 or -1, got '" + v + "'";
        return "";
      }
      imm[i] = v == "-1" ? -1 : (v == "1" ? 1 : 0);
    }
    // The operand scales are multipliers, not flags: -1 negates the operand and 1
    // leaves it, which is why ptxas takes only those two there. The 0 an older dump
    // writes means the same as 1. The rest are ordinary flags.
    const bool scale_d = imm[0] != 0;
    const bool neg_a = imm[1] == -1, neg_b = imm[2] == -1;
    const bool trans_a = imm[3] != 0, trans_b = imm[4] != 0;
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
        // has no infinities, e5m2 has them, an s8 sign-extends and a u8 does not , 
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
        // byte offsets that is CUTLASS's Swizzle<3,4,3>, <2,4,3> or <1,4,3> , 
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
    // Majorness does not move the descriptor's two offsets, SBO steps between MN
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
    // majorness, SBO steps between MN groups and LBO between K groups either way , 
    // but which direction lives contiguously inside an atom does. In a K-major
    // operand K is the contiguous one: an atom holds values of K, and the eight
    // values of N are an atom width apart. That is exactly the mirror of A's
    // K-major walk and of what this used to do, which put K 16 bytes apart and N
    // contiguous, the MN-major arrangement, under an instruction that says K-major.
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
    // `ctx.lid0 + l` and each accumulator is that lane's element of the vector , 
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
    // `.kind::block_scale` (and `.kind::mxf4`) names a *different* instruction: the
    // scale operands that follow the accumulators multiply each block of the
    // product, so the product alone is a different number rather than a coarser
    // one. Those spellings would otherwise match a shape arm below, the tile and
    // the type are the same, and the scales would be dropped in silence, which is
    // the one thing a translator must not do.
    //
    // `.kind::f8f6f4` is the exception, and ptxas settles it: it is the plain
    // fp4/fp6 multiply on `sm_120a`, four operands and no scales, so it is read as
    // the form it is. The block-scaled spellings keep their rejection.
    for (const auto& p : parts) {
      if (p == "kind::f8f6f4" && args.size() <= 4) continue;
      if (p.rfind("kind", 0) == 0 || p == "block_scale" || p.rfind("scale_vec", 0) == 0) {
        err = "mma '" + opcode + "': a block-scaled form carries the per-block scale "
              "operands after the accumulators, and the product without them is a "
              "different number";
        return "";
      }
    }
    // Four shapes, and an operand list no longer than `d, a, b[, c]`. A longer one
    // is an instruction whose extra operands mean something, the block scales
    // above, or a form this does not know, and reading only the first four would
    // answer for a product nobody asked for.
    if (args.size() > 4) {
      err = "mma '" + opcode + "' has " + std::to_string(args.size()) +
            " operands: the forms translated are accumulators, A, B and optionally C";
      return "";
    }
    // The four shapes Triton emits for `tl.dot`. Operand width decides the
    // fragment layout: one element per register for tf32, two for a 16-bit type,
    // four for an 8-bit one, and the K dimension scales with it.
    const bool bf16 = has_part(parts, "bf16");
    const bool e4m3 = has_part(parts, "e4m3");
    const bool e5m2 = has_part(parts, "e5m2");
    // The integer dot: `tl.dot` over int8 lowers to the same 8-bit fragment map as
    // the fp8 form and reads the bytes as signed or unsigned.
    const bool s8 = has_part(parts, "s8");
    const bool u8 = has_part(parts, "u8");
    // The four-bit operands. A register holds eight of them where a byte operand
    // holds four, so the fragment map is the same shape with `per` of eight, the
    // standard's table for `m16n8k64.s4` is `col = 8*(lane%4) + (i&7)`, `row =
    // lane>>2`, which is that formula at K of 64. Only the extraction differs, and
    // the accumulate is still `s32`.
    const bool s4 = has_part(parts, "s4");
    const bool u4 = has_part(parts, "u4");
    // The narrow float operands: four-bit e2m1 and six-bit e2m3/e3m2. The standard's
    // table puts four of them in a register for `m16n8k32`, the same fragment map
    // the byte forms use, and eight for `m16n8k64`, which is the map the four-bit
    // integer form uses. Their accumulate is `f32`, not `s32`.
    const bool e3m2 = has_part(parts, "e3m2");
    const bool e2m3 = has_part(parts, "e2m3");
    const bool e2m1 = has_part(parts, "e2m1");
    const bool f16 = has_part(parts, "f16");
    const bool tf32 = has_part(parts, "tf32");
    // `K` is the reduction width and `per` how many elements one register holds,
    // which is what decides the fragment map: a tf32 operand is one value a
    // register, a 16-bit one is two, an 8-bit one four. The eight-row shape keeps
    // the same map with one row group instead of two and two accumulators a lane.
    const bool wide8 = has_part(parts, "m8n8k16") && (s8 || u8);
    int K = 0;
    int per = 0;
    if (has_part(parts, "m16n8k16") && (f16 || bf16)) { K = 16; per = 2; }
    else if (has_part(parts, "m16n8k8") && (f16 || bf16)) { K = 8; per = 2; }
    else if (has_part(parts, "m16n8k8") && tf32) { K = 8; per = 1; }
    else if (has_part(parts, "m16n8k4") && tf32) { K = 4; per = 1; }
    else if (has_part(parts, "m16n8k32") &&
             (e4m3 || e5m2 || e3m2 || e2m3 || e2m1 || s8 || u8)) { K = 32; per = 4; }
    else if (has_part(parts, "m16n8k64") && (s4 || u4 || e2m1)) { K = 64; per = 8; }
    else if (has_part(parts, "m16n8k32") && (s4 || u4)) { K = 32; per = 8; }
    else if (wide8) { K = 16; per = 4; }
    if (K == 0) {
      err = "unsupported mma '" + opcode + "': m16n8k4 and m16n8k8 with tf32, "
            "m16n8k8 and m16n8k16 with f16/bf16, m16n8k32 with "
            "e4m3/e5m2/e3m2/e2m3/e2m1/s8/u8, m16n8k32 and m16n8k64 with s4/u4, "
            "m16n8k64 with e2m1, and m8n8k16 with s8/u8 are the forms translated";
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
    // How many of K one register pair covers, so the register, the lane and the
    // element within the register all follow from it. A 16-row shape has two row
    // groups a register each; the eight-row one has a single row group.
    const int split = 4 * per;
    const int groups = (K + split - 1) / split;
    const int nd = wide8 ? 2 : 4;
    std::vector<std::string> d = regs_of(args[0], nd), a = regs_of(args[1], groups * (wide8 ? 1 : 2)),
                             b = regs_of(args[2], groups);
    if (d.empty() || a.empty() || b.empty()) {
      err = "mma '" + opcode + "' takes " + std::to_string(nd) + " accumulators, " +
            std::to_string(groups * (wide8 ? 1 : 2)) + " A registers and " +
            std::to_string(groups) + " B registers";
      return "";
    }
    const std::vector<std::string> c =
        args.size() > 3 ? regs_of(args[3], nd) : std::vector<std::string>();
    const bool int8op = s8 || u8;
    const bool int4op = s4 || u4;
    // Whatever the operand width, a four- or eight-bit dot lands in an `s32`
    // register, so the conversion in and out is the same for both.
    const bool intop = int8op || int4op;
    auto own = [&](const std::string& tok, const std::string& lane) {
      Reg r = parse_reg(tok);
      if (!r.ok) { err = "expected a register, got '" + tok + "'"; return std::string("0.0f"); }
      const std::string v = "s->" + r.base + "[" + std::to_string(r.index) + "][" + lane + "]";
      auto it = reg_types.find(r.base);
      return (it != reg_types.end() && it->second == "float") ? v : ("weft_f32(" + v + ")");
    };
    // The raw word a register holds, with no reading of its bits: what an integer
    // accumulate needs when it converts a value rather than reinterpreting one.
    auto raw = [&](const std::string& tok, const std::string& lane) {
      Reg r = parse_reg(tok);
      if (!r.ok) { err = "expected a register, got '" + tok + "'"; return std::string("0u"); }
      return "s->" + r.base + "[" + std::to_string(r.index) + "][" + lane + "]";
    };
    // One element of an operand register, as a float. A fragment register holds
    // two halves when the operands are 16-bit, and the operand names which; a
    // tf32 operand is a 32-bit value the instruction reduces to 10 mantissa bits,
    // which this host does the same way rather than borrowing the extra
    // precision.
    auto elem = [&](const std::string& tok, const std::string& lane, int which) {
      if (per > 1) {
        Reg r = parse_reg(tok);
        if (!r.ok) { err = "expected a register, got '" + tok + "'"; return std::string("0.0f"); }
        const std::string v = "s->" + r.base + "[" + std::to_string(r.index) + "][" + lane + "]";
        auto it = reg_types.find(r.base);
        const std::string bits = (it != reg_types.end() && it->second == "float") ?
                                     ("weft_bits(" + v + ")") : v;
        // An 8-bit integer operand: the byte is the element, widened to float. A
        // product of two 8-bit values is exact in f32 and so is the sum of thirty-two
        // of them, which is why the integer dot goes through the same accumulator
        // path the fp8 form does rather than a separate one.
        if (int8op)
          return std::string(s8 ? "((float)(int8_t)(" : "((float)(uint8_t)(") + bits + " >> " +
                 std::to_string(8 * which) + "u))";
        // A nibble, widened with its sign for `s4`: the shift left then the
        // arithmetic shift right is what sign-extends four bits to thirty-two.
        if (int4op) {
          const std::string sh = std::to_string(4 * which) + "u";
          if (s4)
            return "((float)(int32_t)((int32_t)(((uint32_t)(" + bits + ") >> " + sh +
                   ") << 28u) >> 28))";
          return "((float)(((uint32_t)(" + bits + ") >> " + sh + ") & 0xFu))";
        }
        // A four-bit float: the nibble is the value's code, widened by the same
        // table the `cvt` forms decode with, the sign is the top bit, so a decode
        // that read it as magnitude would answer 6 where the code says -6.
        if (e2m1)
          return "weft_e2m1(" + bits + ", " + std::to_string(which) + ")";
        // A six-bit one sits in the low bits of its byte, which is the packing the
        // `cvt` forms of this format already use.
        const char* fn = e4m3 ? "weft_e4m3("
                         : e5m2 ? "weft_e5m2("
                         : e3m2 ? "weft_e3m2("
                         : e2m3 ? "weft_e2m3("
                                : (bf16 ? "weft_bf16(" : "weft_h16(");
        return std::string(fn) + bits + ", " + std::to_string(which) + ")";
      }
      return "weft_tf32(" + own(tok, lane) + ")";
    };
    // A is 16 rows by K: the row group this lane is in, and the group 8 above it.
    // Within a register the elements run along K, `per` of them at a time, and
    // the half of K from K/2 up lives in the third and fourth register.
    auto aval = [&](const std::string& wg, int rowhalf, int k) {
      const int reg = rowhalf + (k >= split ? 2 : 0);
      const int within = (k % split) / per;
      return elem(a[reg], "4u * " + wg + " + " + std::to_string(within), k % per);
    };
    // B is K by 8, in the column layout `row.col` names: lane 4n+q holds B[q][n].
    auto bval = [&](int colhalf, int k) {
      const int reg = k >= split ? 1 : 0;
      const int within = (k % split) / per;
      return elem(b[reg], "8u * weft_t + " + std::to_string(4 * colhalf + within),
                  k % per);
    };
    vec_side = true;
    std::string out = "{ const uint32_t weft_g = (l >> 2) & 7u; const uint32_t weft_t = (l & 3u); ";
    // D[r][n] = sum_k A[r][k] * B[k][n] + C[r][n]. A 16-row shape has this lane
    // holding D[g][2t], D[g][2t+1], D[g+8][2t] and D[g+8][2t+1]; the eight-row one
    // holds the first two and stops, because it has one row group.
    for (int q = 0; q < nd; q++) {
      const int rowhalf = q / 2, colhalf = q % 2;
      // The integer dot accumulates in float, a product of two 8-bit values is
      // exact there, and so is the sum of thirty-two of them, while its result
      // register is an `s32`. So C is converted on the way in and the total
      // converted on the way out: reinterpreting the bits instead would store the
      // float's encoding, which is a different number from the sum.
      std::string sum;
      if (c.empty()) sum = "0.0f";
      else if (intop) sum = "(float)(int32_t)(" + raw(c[(size_t)q], "l") + ")";
      else sum = own(c[(size_t)q], "l");
      for (int k = 0; k < K; k++)
        sum += " + " + aval("weft_g", rowhalf, k) + " * " + bval(colhalf, k);
      out += store(d[(size_t)q], intop ? ("(int32_t)(" + sum + ")") : sum,
                   intop ? "int32_t" : "float");
      out += " ";
    }
    out += "}";
    return "for (uint32_t l = 0; l < 32u; l++) { " + out + " }";
  }

  if (op == "atom" || op == "red") {
    // `atom` names its result, `red` does not, so the operands sit at different
    // positions in the two spellings, and the *instruction as written* decides
    // where they are. Whether the result is kept is a separate question, below.
    const bool is_atom = (op == "atom");
    // `atom.cas.b128` compares and writes sixteen bytes as one value. The widest
    // atomic instruction here is 64 bits, so the sixteen bytes are read, compared and
    // written under a lock: on the GPU the lock the 64-bit pair already takes, on the
    // host the stripe table in src/soft_math.hpp. The shape is handled below, once the
    // operation and the address are known, because a compare on a prefix is a different
    // instruction with the same spelling and only a whole-value compare is this one.
    const bool b128atom = has_part(parts, "b128");
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
          p == "or" || p == "xor" || p == "exch" || p == "cas" || p == "inc" ||
          p == "dec") {
        aop = p;
        break;
      }
    }
    if (aop.empty()) { err = "unsupported atomic '" + opcode + "'"; return ""; }
    std::string fn;
    // An atomic in shared memory is refused by name. `.shared`, `.shared::cta` and
    // `.shared::cluster` name a window the launch owns, and its address is an offset
    // inside that window, not a pointer: the library's atomics in src/weft_msl.hpp are
    // declared over device pointers, and on the GPU the address space is part of the
    // pointer type, so one helper cannot serve both spaces. Handing the offset to the
    // device form was a crash, an atomic on address zero, not an atomic on the window.
    // Loads and stores in the same space are translated, the `.v2`/`.v4` cluster forms
    // included; this is the atomic case alone.
    // The address an atomic takes, in the space the instruction names. A shared address
    // is an offset inside the threadgroup window, so it is added to `sh`; a device one
    // is the value the register holds. `.shared`, `.shared::cta` and `.shared::cluster`
    // are one window to a launch this wide, and on the GPU the window and the heap are
    // different pointer types -- which is why src/weft_msl.hpp carries the atomic family
    // twice and the narrow helpers in src/soft_math.hpp are templates over their pointer.
    auto atom_ptr = [&](const std::string& t) -> std::string {
      if (shared)
        return "(" + addr_space("threadgroup", false) + t + "*)(sh + (" + addr + "))";
      return "(" + addr_space("device", false) + t + "*)(uintptr_t)(" + addr + ")";
    };
    // The word that holds a 16-bit value or a packed pair: the address masked down to
    // it. That arithmetic is the same in both spaces, so only the base and the cast
    // differ. A shared atomic is not the device one with a cast: the window is not the
    // heap, and casting an offset was an atomic on address zero.
    auto atom_word = [&](const std::string& t, const std::string& a) -> std::string {
      if (shared)
        return "(" + addr_space("threadgroup", false) + t + "*)(sh + (" + a + " & ~3ull))";
      return "(" + addr_space("device", false) + t + "*)(uintptr_t)(" + a + " & ~3ull)";
    };
    if (b128atom) {
      if (aop != "cas") {
        err = "'" + opcode + "' is a ." + aop + " atomic on .b128: the isa names only "
              ".cas at that width";
        return "";
      }
      if (shared) {
        err = "'" + opcode + "' is a 128-bit atomic in shared memory: that width has no "
              "atomic instruction in the window on this host's GPU, and the lock the "
              "device form takes orders the heap rather than the window";
        return "";
      }
      if (args.size() <= vi + 1) { err = "bad " + opcode; return ""; }
      const std::string p = atom_ptr("weft_u128");
      const std::string c = operand_as(args[vi], "weft_u128");
      const std::string val = operand_as(args[vi + 1], "weft_u128");
      if (msl) {
        wide_atom = true;
        // One thread runs all 32 lanes, so an address every lane shares takes the
        // lock around the loop: a lane that waited for a peer in its own warp could
        // never be resumed to release it.
        if (vec && p.find("[l]") == std::string::npos) {
          const bool sel = vec_select;
          vec_select = false;
          const std::string raw = "weft::rmw_cas_b128_raw(" + p + ", " + c + ", " + val + ")";
          const std::string inner = has_result ? store(args[0], raw, "weft_u128")
                                               : ("(void)" + raw + ";");
          vec_select = sel;
          vec_side = true;
          return "{ device uint32_t* weft_lk = weft::lock_of(ctx.locks, (device const void*)" +
                 p + ");\n"
                 "  weft::lock_take(weft_lk);\n"
                 "  for (uint32_t l = 0; l < 32u; l++) { if (" +
                 lane_cond() + ") { " + inner + " } }\n"
                 "  weft::lock_drop(weft_lk); }";
        }
        const std::string call =
            "weft::rmw_cas_b128(ctx.locks, " + p + ", " + c + ", " + val + ")";
        if (!has_result) return guard + "(void)" + call + ";";
        return guard + store(args[0], call, "weft_u128");
      }
      const std::string call = "weft_atom_cas_b128(" + p + ", " + c + ", " + val + ")";
      if (!has_result) return guard + "(void)" + call + ";";
      return guard + store(args[0], call, "weft_u128");
    }
    // One width stays refused: a 64-bit atomic in the window. This host's GPU has no
    // atomic at that width in threadgroup memory, and the lock the device form takes
    // orders the heap, not the window, so answering with two 32-bit words would tear the
    // value a reader is promised. Everything at 32 bits and below is translated.
    if (shared && cpp_size(ty) == 8) {
      err = "'" + opcode + "' is a 64-bit atomic in shared memory: that width has no "
            "atomic instruction in the window on this host's GPU, and the lock the "
            "device form takes orders the heap rather than the window";
      return "";
    }
    // Vector atomics: `atom.global.v4.f16.max.noftz {%hd0,%hd1,%hd2,%hd3}, [gbl],
    // {%h0,%h1,%h2,%h3}`. Table 41 keeps these in a table of their own, and it is not a
    // wider copy of the scalar one: a half takes `.add`, `.min` and `.max` at `.v2`,
    // `.v4` and `.v8`, a packed pair takes them at `.v2` and `.v4`, and no scalar half is
    // spelled `.min` at all -- which is why this is its own case. The isa says the
    // atomicity is per element and not for the group, so each element is one
    // compare-exchange on the word that holds it and the element beside it keeps its
    // value. `.noftz` is required by the isa on these and is what the library does
    // anyway, so a form written without it lands on the same answer.
    int vn = 0;
    for (size_t i = 1; i < parts.size(); i++) {
      const std::string& p = parts[i];
      if (p.size() == 2 && p[0] == 'v' && p[1] >= '2' && p[1] <= '8') vn = p[1] - '0';
    }
    if (vn) {
      const bool packv = has_part(parts, "f16x2") || has_part(parts, "bf16x2");
      const bool halfv = has_part(parts, "f16") || has_part(parts, "bf16");
      if (!halfv && !packv) {
        err = "'" + opcode + "' is a vector atomic, and this front end carries the half "
              "ones: a vector of floats or of integers is refused rather than written "
              "as one step, since the isa makes each element its own atomic";
        return "";
      }
      if (vn != 2 && vn != 4) {
        if (vn != 8) {
          err = "'" + opcode + "' is a vector atomic, and the isa names .v2, .v4 and "
                ".v8 only";
          return "";
        }
        if (packv) {
          err = "'" + opcode + "' is a .v8 atomic of half pairs, which the isa does not "
                "name: the packed vector forms are .v2 and .v4";
          return "";
        }
      }
      if (aop != "add" && aop != "min" && aop != "max") {
        err = "'" + opcode + "' is a ." + aop + " vector atomic, and the isa names only "
              ".add, .min and .max on a vector";
        return "";
      }
      // The statement pre-pass strips a brace from each end of a line, because a block
      // brace and an operand brace are the same character there, so a list that ends the
      // instruction arrives without its closer. The elements are all still there.
      auto elems = [&](const std::string& tok, std::vector<std::string>& out) {
        std::string t = trim(tok);
        if (t.size() < 2 || t.front() != '{') return false;
        t = t.substr(1);
        if (!t.empty() && t.back() == '}') t.pop_back();
        out = split_top(t, ',');
        return true;
      };
      // The bit bucket is how a reduction gives up its result, and the vector form
      // writes it the same way the scalar one does.
      const bool bucket = !args.empty() && trim(args[0]) == "_";
      std::vector<std::string> dsts, srcs;
      if (!elems(args[vi], srcs) || srcs.size() != (size_t)vn) {
        err = "'" + opcode + "' names " + std::to_string(vn) +
              " source halfs inside one pair of braces";
        return "";
      }
      if (has_result && !bucket && (!elems(args[0], dsts) || dsts.size() != (size_t)vn)) {
        err = "'" + opcode + "' names " + std::to_string(vn) +
              " destination halfs inside one pair of braces";
        return "";
      }
      const bool bf = has_part(parts, "bf16") || has_part(parts, "bf16x2");
      const bool ismax = (aop == "max");
      std::string code;
      for (int k = 0; k < vn; k++) {
        const std::string a =
            "(" + addr + " + " + std::to_string(packv ? 4 * k : 2 * k) + "ull)";
        std::string call;
        if (packv) {
          const std::string w = atom_word("uint32_t", a) + ", ";
          call = aop == "add"
                     ? "weft_atom_add_pack(" + w +
                           operand_as(srcs[(size_t)k], "uint32_t") + ", " +
                           (bf ? "1" : "0") + ")"
                     : "weft_atom_minmax_pack(" + w +
                           operand_as(srcs[(size_t)k], "uint32_t") + ", " +
                           (bf ? "1" : "0") + ", " + (ismax ? "1" : "0") + ")";
        } else {
          const std::string w =
              atom_word("uint32_t", a) + ", (uint32_t)(((" + a + ") & 2ull) * 8u), ";
          call = aop == "add"
                     ? "weft_atom_add_narrow(" + w +
                           operand_as(srcs[(size_t)k], "uint16_t") + ", " +
                           (bf ? "1" : "0") + ")"
                     : "weft_atom_minmax_narrow(" + w +
                           operand_as(srcs[(size_t)k], "uint16_t") + ", " +
                           (bf ? "1" : "0") + ", " + (ismax ? "1" : "0") + ")";
        }
        if (!has_result || bucket) code += "(void)" + call + "; ";
        else code += store(dsts[(size_t)k], call, packv ? "uint32_t" : "uint16_t") + " ";
      }
      // One instruction, one predicate, so the elements sit in one block.
      return guard + "{ " + code + "}";
    }
    // 16-bit integer atomics. Table 41 of the isa gives the 16-bit width exactly one
    // operation, .cas, and this host's atomics are 32 bits wide, so the half
    // addressed is compared and written on its own: narrowing to a 32-bit atomic
    // would overwrite the half beside it. Anything else at this width is refused by
    // name rather than translated as something it is not (.exch, for one, the isa
    // lists only at 32 bits and above).
    if (ty == "uint16_t" || ty == "int16_t") {
      if (aop != "cas" || args.size() <= vi + 1) {
        err = "'" + opcode + "' is a ." + aop + " atomic on a 16-bit value: the isa "
              "names only .cas at that width";
        return "";
      }
      const std::string call =
          "weft_atom_cas_b16(" + atom_word("uint32_t", addr) + ", (uint32_t)(((" + addr +
          ") & 2ull) * 8u), " + operand_as(args[vi], "uint16_t") + ", " +
          operand_as(args[vi + 1], "uint16_t") + ")";
      if (!has_result) return guard + "(void)" + call + ";";
      return guard + store(args[0], call, "uint16_t");
    }
    // A 16-bit float atomic. The isa names only `.add` at that width, and the
    // hardware's atomics are 32 bits, so the add is a compare-exchange on the word
    // that holds the half: the value beside it in the same word is left alone, and
    // a packed pair is the whole word and needs no split.
    if (has_part(parts, "f16") || has_part(parts, "bf16") || has_part(parts, "f16x2") ||
        has_part(parts, "bf16x2")) {
      const bool pack = has_part(parts, "f16x2") || has_part(parts, "bf16x2");
      const bool bf = has_part(parts, "bf16") || has_part(parts, "bf16x2");
      if (aop != "add") {
        err = "'" + opcode + "' is a ." + aop + " atomic on a 16-bit float, which the isa "
              "does not name: .add is the only 16-bit float atomic";
        return "";
      }
      const std::string bfs = bf ? "1" : "0";
      std::string call;
      if (pack) {
        call = "weft_atom_add_pack(" + atom_word("uint32_t", addr) + ", " +
               operand_as(args[vi], "uint32_t") + ", " + bfs + ")";
      } else {
        call = "weft_atom_add_narrow(" + atom_word("uint32_t", addr) +
               ", (uint32_t)(((" + addr + ") & 2ull) * 8u), " +
               operand_as(args[vi], "uint16_t") + ", " + bfs + ")";
      }
      if (!has_result) return guard + "(void)" + call + ";";
      return guard + store(args[0], call, pack ? "uint32_t" : "uint16_t");
    }
    // .inc and .dec: table 41 gives them the .u32 width and nothing else, and this
    // host has no instruction for either, so the step round the cycle bounded by b
    // is a compare-exchange loop. A counter that wraps at b is what a histogram
    // bound or a segmented ballot does, and the answer after N lanes is the start
    // advanced N steps whatever order they land in.
    if (aop == "inc" || aop == "dec") {
      if (ty != "uint32_t") {
        err = "'" + opcode + "' is a ." + aop + " atomic on " + ty +
              ": the isa names .inc and .dec only for .u32";
        return "";
      }
      const std::string call = "weft_atom_" + aop + "_u32(" + atom_ptr("uint32_t") + ", " +
                               operand_as(args[vi], "uint32_t") + ")";
      if (!has_result) return guard + "(void)" + call + ";";
      return guard + store(args[0], call, "uint32_t");
    }
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
      // does have, and `wide_atom` marks the kernel as needing that runtime, the
      // warp back end, at the same width as any other launch.
      wide_atom = true;
      const std::string p =
          "((" + addr_space("device", false) + ty + "*)(uintptr_t)(" + addr + "))";
      // In the warp back end one thread runs all 32 lanes, so 32 trips to the
      // coherent point for one word are 32 serialised trips for nothing: a counter, a
      // reduction target or a head pointer is one word for the whole warp. When the
      // emitted address says so -- no lane in it -- the lock goes around the loop. A
      // lane-array register, `s->rd[1][l]`, never says so even when all 32 values
      // agree, which is exactly the counter case, so the test is made at run time
      // there: up to 31 pointer compares, and then one lock instead of 32.
      // Hoisting every address onto one global lock instead was tried and measured
      // slower (19 ms against 10 ms at 4,096 lanes, 660 ms against 150 ms at
      // 262,144): serialising threads that were working on different words costs more
      // than the lock trips saved.
      const std::string val = operand_as(args[vi], ty);
      // A no-result max/min whose value the word already holds needs no lock, and
      // `satisfied_*` proves that without one (src/weft_msl.hpp). Only a value that is
      // the same for every lane can be tested outside the lane loop.
      const bool reducible = !has_result && (aop == "max" || aop == "min") &&
                             val.find("[l]") == std::string::npos;
      auto already = [&](const std::string& at) {
        return "weft::satisfied_" + aop + (ty == "int64_t" ? "_s64(" : "_u64(") + "(device " + ty +
               "*)(" + at + "), " + val + ")";
      };
      if (vec) {
        std::string extra;
        if (aop == "cas" && args.size() > vi + 1) extra = ", " + operand_as(args[vi + 1], ty);
        const bool sel = vec_select;  // the loops below mask by lane; a store must not
        vec_select = false;
        const std::string raw_call = "weft::rmw_" + aop + "_raw(" + p + ", " + val + extra + ")";
        const std::string lock_call =
            "weft::rmw_" + aop + "(ctx.locks, " + p + ", " + val + extra + ")";
        const std::string loop_raw =
            "for (uint32_t l = 0; l < 32u; l++) { if (" + lane_cond() + ") { " +
            (has_result ? store(args[0], raw_call, ty) : ("(void)" + raw_call + ";")) + " } }";
        const std::string loop_lock =
            "for (uint32_t l = 0; l < 32u; l++) { if (" + lane_cond() + ") { " +
            (has_result ? store(args[0], lock_call, ty) : ("(void)" + lock_call + ";")) + " } }";
        vec_select = sel;
        vec_side = true;
        std::string first = p;
        for (size_t q = 0; (q = first.find("[l]", q)) != std::string::npos; q += 3)
          first.replace(q, 3, "[0]");
        auto locked = [&](const std::string& at) {
          return "{ device uint32_t* weft_lk = weft::lock_of(ctx.locks, (device const void*)" + at +
                 ");\n  weft::lock_take(weft_lk);\n  " + loop_raw +
                 "\n  weft::lock_drop(weft_lk); }";
        };
        if (first == p)
          return reducible ? "{ if (!" + already(first) + ") " + locked(first) + " }"
                           : locked(first);
        // The first *active* lane names the word, not lane 0: a divergent warp can
        // have lane 0 parked on another path, and locking its address would leave the
        // word the live lanes are updating unlocked.
        return "{ device const void* weft_p0 = nullptr; bool weft_uni = true;\n"
               "  for (uint32_t l = 0; l < 32u; l++) if (am[l]) { if (weft_p0 == nullptr) "
               "weft_p0 = (device const void*)" + p + "; else if ((device const void*)" + p +
               " != weft_p0) weft_uni = false; }\n"
               "  if (weft_uni) " +
               (reducible ? "{ if (!" + already("weft_p0") + ") " + locked("weft_p0") + " }"
                          : locked("weft_p0")) +
               " else { " + loop_lock + " } }";
      }
      std::string call = "weft::rmw_" + aop + "(ctx.locks, " + p + ", " + val;
      if (aop == "cas" && args.size() > vi + 1) call += ", " + operand_as(args[vi + 1], ty);
      call += ")";
      if (has_result) return guard + store(args[0], call, ty);
      if (reducible) return guard + "{ if (!" + already(p) + ") (void)" + call + "; }";
      return guard + "(void)" + call + ";";
    }
    std::string call = "weft::" + fn + "(" + atom_ptr(ty) + ", " + operand_as(args[vi], ty);
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
  // as a memcpy and the commit/wait instructions are no-ops, which is exactly
  // the ordering the GPU promises once the wait has been reached.
  // The bulk form below is a different instruction with a different shape: it reports
  // its own completion rather than joining a commit/wait group, so it is not this one.
  if (op == "cp" && parts.size() >= 2 && parts[1] == "async" &&
      !(parts.size() >= 3 && parts[2] == "bulk")) {
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

  // cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [dst], [src], size
  //
  // The bulk store's reduce form: the threadgroup values are combined into the buffer
  // instead of overwriting it. It is the same descriptor-free shape as the plain store
  //, the size is written in the instruction and the group commit is a no-op, so the
  // only new thing is the combining step.
  if (op == "cp" && parts.size() >= 3 && parts[1] == "reduce" && parts[2] == "async") {
    std::string opname, ty;
    for (const auto& p : parts) {
      if (p == "add" || p == "min" || p == "max" || p == "and" || p == "or" || p == "xor" ||
          p == "inc" || p == "dec")
        opname = p;
      else if (p == "f32" || p == "u32" || p == "s32" || p == "u64" || p == "s64" ||
               p == "f16" || p == "bf16" || p == "f64")
        ty = p;
    }
    if (args.size() < 3) { err = "a reduce copy names a destination, a source and a size"; return ""; }
    // The element types the ISA's table names for a global destination, each at
    // its own width, so the loop's `size / sizeof(element)` is the element count
    // the instruction means. `bf16` and a software double are carried as their
    // bits, the GPU has neither type, so their combine is the helper rather
    // than an operator; a half is a real `half` on both engines.
    const bool f64 = (ty == "f64");
    const bool bf16 = (ty == "bf16");
    const std::string cty = ty == "f32" ? "float"
                          : ty == "f16" ? "half"
                          : bf16 ? "uint16_t"
                          : f64 ? (msl ? "uint64_t" : "double")
                          : ty == "u32" ? "uint32_t"
                          : ty == "s32" ? "int32_t"
                          : ty == "u64" ? "uint64_t"
                          : ty == "s64" ? "int64_t"
                                        : std::string();
    if (cty.empty()) {
      err = "unsupported reduction type in '" + opcode +
            "': f16, bf16, f32, f64 and the 32- and 64-bit integers have a "
            "bulk reduction here";
      return "";
    }
    const bool bitwise = (opname == "and" || opname == "or" || opname == "xor");
    if (bitwise && (ty == "f32" || ty == "f16" || bf16 || f64)) {
      err = "a floating-point bulk store has no '" + opname + "': it is a bitwise reduction";
      return "";
    }
    // The ISA's table: `.f64` has only `.add`, and `.inc`/`.dec` are the integer
    // forms. A min or max of doubles would be a defined-looking guess here.
    if (f64 && opname != "add") {
      err = "a bulk reduction of .f64 has only .add: " + opname + " is not in the isa's table";
      return "";
    }
    if ((opname == "inc" || opname == "dec") && ty != "u32") {
      err = "a '" + opname + "' bulk reduction is .u32 only, and this one is ." + ty;
      return "";
    }
    std::string combine;
    if (opname == "add") {
      // A bf16 sum widens both elements to float, adds, and rounds back to bf16:
      // both engines run that text, so they cannot disagree, and a bf16 pair sums
      // exactly in a float unless the exponents are more than 24 apart, the one
      // corner where rounding here could differ from rounding in bf16 alone.
      combine = f64 ? (msl ? "weft_d[k] = weft_fp64::add(weft_d[k], weft_s[k]);"
                           : "weft_d[k] = weft_d[k] + weft_s[k];")
              : bf16 ? "weft_d[k] = (uint16_t)weft_bf16_bits(weft_bf16((uint32_t)weft_d[k], 0) + "
                       "weft_bf16((uint32_t)weft_s[k], 0));"
                     : "weft_d[k] = weft_d[k] + weft_s[k];";
    } else if (opname == "min" || opname == "max") {
      const std::string cmp = opname == "min" ? " < " : " > ";
      combine = bf16 ? std::string("weft_d[k] = (weft_bf16((uint32_t)weft_s[k], 0)") + cmp +
                           "weft_bf16((uint32_t)weft_d[k], 0)) ? weft_s[k] : weft_d[k];"
                     : std::string("weft_d[k] = (weft_d[k]") + cmp + "weft_s[k]) ? weft_d[k] : weft_s[k];";
    } else if (opname == "inc") {
      // "The .inc and .dec operations return a result in the range [0..x] where x
      // is the value at the source state space": the source element is the bound.
      combine = "weft_d[k] = (weft_d[k] >= weft_s[k]) ? 0u : weft_d[k] + 1u;";
    } else if (opname == "dec") {
      combine = "weft_d[k] = (weft_d[k] == 0u || weft_d[k] > weft_s[k]) ? weft_s[k] : weft_d[k] - 1u;";
    } else if (opname == "and") combine = "weft_d[k] = weft_d[k] & weft_s[k];";
    else if (opname == "or") combine = "weft_d[k] = weft_d[k] | weft_s[k];";
    else if (opname == "xor") combine = "weft_d[k] = weft_d[k] ^ weft_s[k];";
    else { err = "unsupported reduction in '" + opcode + "'"; return ""; }
    std::string dst, dname, src, sname;
    if (!mem_operand(args[0], dst, dname)) return "";
    if (!mem_operand(args[1], src, sname)) return "";
    const std::string bytes = "((uint32_t)(" + operand_as(args[2], "uint32_t") + "))";
    // Metal has no unqualified pointers: the destination is a `device` one and the
    // source is a `threadgroup` one, and a cast that leaves the space off does not
    // compile at all.
    const std::string dq = msl ? "device " : "";
    const std::string sq = msl ? "threadgroup const " : "const ";
    return guard + "{ " + dq + cty + "* weft_d = (" + dq + cty + "*)(uintptr_t)(" + dst +
           "); " + sq + cty + "* weft_s = (" + sq + cty + "*)(sh + (" + src + ")); "
           "for (uint32_t k = 0; k < (" + bytes + ") / sizeof(" + cty + "); k++) { " + combine +
           " } }";
  }

  // cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [dst], [src], bytes, [mbr]
  //
  // The bulk copy is the one that reports itself: the bytes it moves are subtracted
  // from what the barrier was told to expect, and the phase flips when that sum runs
  // out. So it is a copy followed by the completion, the same two words the arrive
  // touches, rather than a commit/wait group. `expect_tx` is the other half of the
  // pair, and a kernel that uses one without the other either never flips its phase
  // or flips it on bytes that never moved, exactly as on the hardware.
  if (op == "cp" && parts.size() >= 3 && parts[2] == "bulk") {
    for (size_t i = 3; i < parts.size(); i++)
      if (parts[i] == "commit_group" || parts[i] == "wait_group" || parts[i] == "wait_all")
        return guard + ";";
    bool from_global = false, to_shared = false, completes = false;
    int gq = -1, sq = -1;  // where each space qualifier sits: the order is the direction
    for (size_t i = 3; i < parts.size(); i++) {
      const std::string& p = parts[i];
      if (p == "global") { from_global = true; gq = (int)i; }
      if (p.rfind("shared", 0) == 0) { to_shared = true; if (sq < 0) sq = (int)i; }
      if (p.rfind("mbarrier", 0) == 0 && p.find("complete_tx") != std::string::npos) completes = true;
    }
    // The mnemonic writes destination first, so a threadgroup qualifier ahead of a global
    // one is a load and the other way round is a store.
    const bool tensor_load = sq >= 0 && gq >= 0 && sq < gq;

    // The barrier a copy reports to, as the object's offset in the window.
    auto mbar_slot = [&](const std::string& operand, std::string& bad) -> bool {
      std::string mo = trim(operand);
      const size_t lb = mo.find('['), rb2 = mo.rfind(']');
      if (lb == std::string::npos || rb2 == std::string::npos || rb2 < lb) {
        err = "a bulk copy needs the mbarrier it completes, in brackets";
        return false;
      }
      std::string inner = trim(mo.substr(lb + 1, rb2 - lb - 1));
      std::string mname = inner, mextra;
      const size_t plus = inner.find('+');
      if (plus != std::string::npos) {
        mname = trim(inner.substr(0, plus));
        mextra = trim(inner.substr(plus + 1));
      }
      auto so = shared_offsets.find(mname);
      if (so == shared_offsets.end()) {
        err = "'" + mname + "' is not a shared array: a bulk copy completes an mbarrier";
        return false;
      }
      bad = std::to_string(so->second);
      if (!mextra.empty()) bad += " + " + operand_as(mextra, "uint64_t");
      return true;
    };

    // What a copy that reports itself does to the barrier: the bytes it moved come off
    // what the phase was told to expect, and when that sum runs out the count is restored
    // and the phase flips. One text serves the linear copy, whose size is written in the
    // instruction, and the tensor copy, whose size is the box the descriptor names, a box
    // that hangs off the tensor still transfers all of it, with the outside zero-filled.
    auto complete = [&](const std::string& bad, const std::string& tx) -> std::string {
      if (msl)
        return std::string("{ threadgroup atomic_uint* weft_a = (threadgroup atomic_uint*)(sh + ") +
               bad + "); const uint32_t weft_tx = " + tx +
               "; const uint32_t weft_left = atomic_fetch_sub_explicit(weft_a, weft_tx, "
               "memory_order_relaxed); if (weft_left == weft_tx) { atomic_store_explicit(weft_a, "
               "(uint32_t)(*((volatile threadgroup const uint32_t*)(sh + " + bad +
               " + 4u)) >> 1u), memory_order_relaxed); *((volatile threadgroup uint32_t*)(sh + " +
               bad + " + 4u)) = (*((volatile threadgroup const uint32_t*)(sh + " + bad +
               " + 4u)) ^ 1u); } }";
      return std::string("{ const uint32_t weft_tx = ") + tx + "; *((uint32_t*)(sh + " + bad +
             ")) = (*((uint32_t*)(sh + " + bad + ")) >= weft_tx) ? (*((uint32_t*)(sh + " + bad +
             ")) - weft_tx) : 0u; if (*((uint32_t*)(sh + " + bad + ")) == 0u) { "
             "*((uint32_t*)(sh + " + bad + ")) = *((uint32_t*)(sh + " + bad + " + 4u)) >> 1; "
             "*((uint32_t*)(sh + " + bad + " + 4u)) ^= 1u; } }";
    };
    if (parts.size() >= 4 && parts[3] == "tensor") {
      // A tensor copy takes its geometry from a 128-byte object instead of from its
      // operands: `[map, {cx, cy}]` names the map and the corner of the box, and the map
      // says where the tensor is and how far apart its elements sit. The layout of that
      // object is weft's own, see the note over the decoder in `src/weft_msl.hpp`, so
      // both engines read the same bytes and a descriptor that is not one of ours is
      // refused by its tag instead of being decoded from a guess.
      // `.im2col` is the convolution form of the same instruction: the object describes
      // a column of the implicit GEMM, channels by pixels, and the copy names the
      // corner, so the filter window's position rides alongside it as a second list.
      // PTX spells the qualifier `.tile::im2col`, so the token is not bare `im2col`;
      // accept both and treat them as the one form.
      const bool column = has_part(parts, "im2col") || has_part(parts, "tile::im2col");
      // One copy serves the whole threadgroup, so on the warp back end it is issued once,
      // by the thread that owns the whole warp, which is lane 0 of that warp. The loop
      // below runs a single lane for that reason: it binds `l` to 0, so a value the copy
      // is aimed with and the predicate it carries are the leader's, and the copy is one
      // statement per warp rather than one per lane.
      auto one_copy = [&](const std::string& body) -> std::string {
        if (!vec) return guard + body;
        return "{ for (uint32_t l = 0; l < 1u; l++) { if (" + lane_cond() + ") " + body + " } }";
      };
      // The operand carrying the map is the one with the braces in it.
      int mi = -1;
      for (size_t i = 0; i < args.size(); i++)
        if (args[i].find('{') != std::string::npos) { mi = (int)i; break; }
      if (mi < 0) {
        err = "a tensor copy names the map and a box coordinate: '" + opcode + "'";
        return "";
      }
      std::string spec = trim(args[(size_t)mi]);
      if (!spec.empty() && spec[0] == '[') spec.erase(0, 1);
      if (!spec.empty() && spec[spec.size() - 1] == ']') spec.pop_back();
      size_t cut = std::string::npos;
      int depth = 0;
      for (size_t i = 0; i < spec.size(); i++) {
        if (spec[i] == '{') depth++;
        else if (spec[i] == '}') depth--;
        else if (spec[i] == ',' && depth == 0) { cut = i; break; }
      }
      if (cut == std::string::npos) {
        err = "a tensor copy names the map and a box coordinate: '" + opcode + "'";
        return "";
      }
      const std::string mapname = trim(spec.substr(0, cut));
      std::string cspec = trim(spec.substr(cut + 1));
      if (!cspec.empty() && cspec[0] == '{') cspec.erase(0, 1);
      if (!cspec.empty() && cspec[cspec.size() - 1] == '}') cspec.pop_back();
      // A descriptor is an address in the constant, parameter or global space, that is
      // where the instruction takes it, so a threadgroup object here is reported rather
      // than handed a window offset that is not a pointer.
      {
        std::string nm = mapname;
        const size_t plus = mapname.find('+');
        if (plus != std::string::npos) nm = trim(mapname.substr(0, plus));
        if (!nm.empty() && nm[0] != '%' && shared_offsets.find(nm) != shared_offsets.end()) {
          err = "'" + nm + "' is threadgroup memory: a tensor map is an address in the "
                "constant, parameter or global space, which is where a descriptor is kept";
          return "";
        }
      }
      const std::string map = "(const " + std::string(msl ? "device " : "") +
                              "void*)(uintptr_t)(" + operand_as(mapname, "uint64_t") + ")";
      std::vector<std::string> co;
      {
        std::string cur;
        for (char ch : cspec) {
          if (ch == ',') { co.push_back(trim(cur)); cur.clear(); }
          else cur.push_back(ch);
        }
        if (!trim(cur).empty()) co.push_back(trim(cur));
      }
      if (co.empty()) {
        err = "a tensor copy names the box corner: '" + opcode + "' has no coordinate";
        return "";
      }
      if (co.size() > 5) {
        err = "a tensor copy has at most five coordinates, one per dimension: '" + opcode +
              "' has " + std::to_string(co.size());
        return "";
      }
      // A rank carries fewer coordinates than five, and the decode reads the rank; the
      // ones the instruction does not name are zero so the call has one shape.
      std::string cargs;
      for (int i = 0; i < 5; i++)
        cargs += ", " + ((size_t)i < co.size()
                             ? ("(int32_t)(" + operand_as(co[(size_t)i], "int32_t") + ")")
                             : std::string("0"));
      // `.im2col` carries the filter window's position beside the corner: one entry per
      // spatial dimension, and any the instruction does not name are zero.
      auto filter_offsets = [&](int at) -> std::string {
        if (at >= (int)args.size()) {
          err = "an im2col copy names its filter offsets: '" + opcode + "'";
          return "";
        }
        std::string ospec = trim(args[(size_t)at]);
        if (!ospec.empty() && ospec[0] == '{') ospec.erase(0, 1);
        if (!ospec.empty() && ospec[ospec.size() - 1] == '}') ospec.pop_back();
        const std::vector<std::string> of = split_top(ospec, ',');
        std::string oargs;
        for (int i = 0; i < 3; i++)
          oargs += ", " + (((size_t)i < of.size() && !of[(size_t)i].empty())
                               ? ("(int32_t)(" + operand_as(of[(size_t)i], "int32_t") + ")")
                               : std::string("0"));
        return oargs;
      };
      if (tensor_load) {
        if (args.size() < 3) {
          err = "a tensor copy that completes an mbarrier names the barrier";
          return "";
        }
        const int di = (mi == 0) ? 1 : 0;
        std::string dst, dname;
        if (!mem_operand(args[(size_t)di], dst, dname)) return "";
        std::string bad;
        if (!mbar_slot(args[args.size() - 1], bad)) return "";
        // The tile lands in threadgroup memory and the copy reports the box's bytes to the
        // barrier, which is the half `expect_tx` was sized on.
        // `vec_side` says this statement owns its own placement rather than joining the
        // lane loop the arithmetic around it is batched into.
        vec_side = true;
        if (column) {
          const std::string oargs = filter_offsets(mi + 1);
          if (oargs.empty()) return "";
          const std::string call =
              "weft::tma_im2col(sh + (" + dst + "), " + map + cargs + oargs + ")";
          return one_copy("{ const uint32_t weft_moved = " + call + "; " +
                          complete(bad, "weft_moved") + " }");
        }
        const std::string call = "weft::tma_load(sh + (" + dst + "), " + map + cargs + ")";
        return one_copy("{ const uint32_t weft_moved = " + call + "; " +
                        complete(bad, "weft_moved") + " }");
      }
      // The other direction: the column comes out of threadgroup memory and is scattered
      // into the window, reported through a bulk group rather than an mbarrier, so the
      // commit and the wait stay no-ops. The shared source is the last operand either way , 
      // the filter offsets sit between the map and it.
      std::string src, sname;
      if (!mem_operand(args.back(), src, sname)) return "";
      vec_side = true;
      if (column) {
        const std::string oargs = filter_offsets(mi + 1);
        if (oargs.empty()) return "";
        return one_copy("weft::tma_im2col_store(" + map + cargs + oargs + ", sh + (" + src +
                        "));");
      }
      return one_copy("weft::tma_store(" + map + cargs + ", sh + (" + src + "));");
    }
    // A prefetch moves nothing and reports nothing: `cp.async.bulk.prefetch` and the
    // tensor form only ask for a cache line. Dropping it is the whole translation, and
    // it is exact, a hint has no effect on what the program can observe.
    if (has_part(parts, "prefetch")) return guard + ";";
    if (!from_global || !to_shared || args.size() < 3) {
      err = "unsupported bulk copy '" + opcode +
            "': global to threadgroup (completing an mbarrier) and threadgroup to global "
            "are translated";
      return "";
    }
    if (!completes) {
      // The other direction: threadgroup memory back out to the buffer, reported through
      // a bulk group rather than an mbarrier, so the commit and wait are no-ops.
      std::string sdst, sdname, ssrc, ssname;
      if (!mem_operand(args[0], sdst, sdname)) return "";
      if (!mem_operand(args[1], ssrc, ssname)) return "";
      const std::string sbytes = "((uint32_t)(" + operand_as(args[2], "uint32_t") + "))";
      if (msl)
        return guard + "weft::copy_global((device void*)(uintptr_t)(" + sdst + "), sh + (" +
               ssrc + "), " + sbytes + ");";
      return guard + "std::memcpy((void*)(uintptr_t)(" + sdst + "), (const void*)(sh + (" +
             ssrc + ")), " + sbytes + ");";
    }
    if (args.size() < 4) {
      err = "a bulk copy that completes an mbarrier names the barrier";
      return "";
    }
    std::string dst, dname, src, sname;
    if (!mem_operand(args[0], dst, dname)) return "";
    if (!mem_operand(args[1], src, sname)) return "";
    const std::string bytes = "((uint32_t)(" + operand_as(args[2], "uint32_t") + "))";
    std::string bad;
    if (!mbar_slot(args[3], bad)) return "";
    const std::string copy = msl
        ? ("weft::copy_shared(sh + (" + dst + "), (const device void*)(uintptr_t)(" + src +
           "), " + bytes + ");")
        : ("std::memcpy((void*)(sh + (" + dst + ")), (const void*)(uintptr_t)(" + src + "), " +
           bytes + ");");
    // The bytes are in the window before this runs, which is the order the hardware
    // promises once the transaction completes.
    return guard + "{ " + copy + " " + complete(bad, bytes) + " }";
  }

  //
  // `tex.2d.v4.f32.f32 {d0,d1,d2,d3}, [t, {u,v}]` samples the binding a `.tex`
  // declaration names, `txq` asks that binding for its size, and `suld`/`sust`
  // read and write a texel exactly, which is what a surface is for. The warp
  // backend has no form for any of them, a texture read is not register
  // arithmetic, so a kernel that uses one stays on the per-lane machine.
  if (op == "tex" || op == "tld4" || op == "txq" || op == "suq" || op == "suld" ||
      op == "sust") {
    vec_ok = false;
    if (args.empty()) { err = "bad " + opcode; return ""; }
    // PTX's cube form is `(s, t, face)`, a face number and a position on it , 
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
    if (op != "txq" && op != "suq" && !cube && opcode.find("2d") == std::string::npos &&
        !d3 && !d1) {
      err = "only a 1-D, 2-D or 3-D texture is bound here, and '" + opcode + "' is not one";
      return "";
    }
    // A gather is a 2x2 footprint, and every form of the binding has one: a volume
    // gathers in the plane its third coordinate names, and a 1-D binding is the
    // `Wx1` plane its sample already reads, so its footprint is the two texels along
    // x and the second row is outside the image, zero, on both engines.
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

    // suq.width/height/depth d, [surf];
    //
    // A surface's extent is the binding's own number, which is what makes the answer
    // the launcher's rather than something the kernel assumes from its geometry.
    // The format and channel-type queries are refused by name: this binding is texel
    // storage laid out as the arena is, and it carries neither field, so inventing
    // one would be an answer the hardware would not give.
    if (op == "suq") {
      if (args.size() < 2) { err = "bad " + opcode; return ""; }
      std::vector<std::string> list;
      const int idx = binding(args[1], list);
      if (idx < 0) return "";
      const bool arr = (size_t)idx < tex_array.size() && tex_array[idx];
      const bool vol = (size_t)idx < tex_3d.size() && tex_3d[idx];
      std::string field;
      if (opcode.find("width") != std::string::npos) field = "width";
      else if (opcode.find("height") != std::string::npos) field = "height";
      else if (opcode.find("depth") != std::string::npos)
        field = (arr || vol) ? "layers" : "one";
      else {
        err = "'" + opcode + "' asks a surface for its format or its channel type, and "
              "this binding carries neither: a surface here is texel storage in the "
              "one format the arena is laid out as";
        return "";
      }
      const std::string expr =
          field == "one"
              ? std::string("1u")
              : (msl ? ("(uint32_t)" + image_of(idx) + "." +
                        (field == "layers" && vol ? "depth" : field))
                     : ("(uint32_t)weft::texture_ptr(" + std::to_string(idx) + ")->" + field));
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
      // the rest of a vector has nowhere to go, what an R32 format does on the
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
      // and a filter cannot read through that mode, so the sample reaches the
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
    // The layer is a float coordinate in the array form, CUDA's `tex2DLayered`
    // passes the slice as one, so it is truncated to the index the slice has.
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

  // Prefetches and a discard are hints. `prefetch.global.L2` and `prefetchu.L1` ask
  // for a line to be brought up; `discard.global.L2` says the contents are dead. The
  // first two have no observable effect at all, and the third makes the data
  // undefined, which keeping it satisfies, there is no cache here to invalidate.
  if (op == "prefetch" || op == "prefetchu" || op == "discard") return guard + ";";

  // The remaining hints and the cluster operations a one-CTA launch cannot have:
  // a cache priority applied to a line, a launch that may start, a performance
  // counter event, a register-count request, a policy descriptor for a hint that
  // is dropped. None of them produces a value a kernel can observe, so accepting
  // them is exact rather than approximate: there is no cache here to hint and no
  // scheduler to steer. `mapa` maps an address of another CTA's shared memory into
  // this CTA's view, and a cluster is one CTA wide here, so it is the identity;
  // `barrier.cluster` waits for CTAs that do not exist and is nothing.
  if (op == "applypriority" || op == "griddepcontrol" || op == "pmevent" ||
      op == "setmaxnreg")
    return guard + ";";
  if (op == "createpolicy") return guard + store(args[0], "0ull", ty.empty() ? "uint64_t" : ty);
  if (op == "mapa" && args.size() >= 3)
    return guard + store(args[0], operand_as(args[1], ty.empty() ? "uint32_t" : ty),
                         ty.empty() ? "uint32_t" : ty);
  if (op == "barrier" && !parts.empty() && parts.size() > 1 && parts[1] == "cluster")
    return guard + ";";

  // `multimem` names a multicast address: one address whose value every replica
  // holds. This runtime has one replica, so a reduce over the replicas of a value is
  // that value and a multicast store writes the one copy that exists -- the mapping
  // cannot disagree with the hardware while there is a single device. `.red` is
  // refused by name because the operation it carries would become a plain store.
  if (op == "multimem") {
    if (has_part(parts, "red")) {
      err = "multimem.red is an atomic at a multicast address: refused rather than "
            "turned into a plain store";
      return "";
    }
    if (ty.empty()) {
      err = "multimem needs the type of the value it moves";
      return "";
    }
    const bool is_st = has_part(parts, "st");
    if (!need(2)) return "";
    const std::string ty_mem = (msl && ty == "double") ? "uint64_t" : ty;
    std::string addr, name, base;
    std::string code;
    if (is_st) {
      if (!mem_operand(args[0], addr, name, &base)) return "";
      code = "*((" + addr_space("device", false) + ty_mem + "*)(uintptr_t)(" + addr +
             ")) = " + operand_as(args[1], ty_mem) + ";";
    } else {
      if (!mem_operand(args[1], addr, name, &base)) return "";
      const std::string rd =
          "*(" + addr_space("device", true) + ty_mem + "*)(uintptr_t)(" + addr + ")";
      code = store(args[0], rd, ty) + " ";
    }
    return vec ? lane_block(code) : guard + code;
  }

  // bmsk.b32 d, a, b;  -- the run of bits from a up to b and nothing else. The isa
  // clamps instead of wrapping: a width of zero or less sets no bit at all, and a width
  // the type cannot hold sets every bit, so `bmsk d, 0, 32` on a .b32 is all ones and
  // not a shift by 32. The width is b - a in 64 bits, because a difference that does not
  // fit the register is still the answer's width, and the low bit is a masked to the
  // type's width, which is the only place the shift can be named without asking a
  // compiler for a shift it would fold away or trap on.
  if (op == "bmsk") {
    if (!need(3)) return "";
    const bool w64 = opcode.find(".b64") != std::string::npos;
    const std::string ty = w64 ? "uint64_t" : "uint32_t";
    const std::string one = w64 ? "1ull" : "1u";
    const std::string bits = w64 ? "64" : "32";
    const std::string rng = w64 ? "63u" : "31u";
    const std::string a = operand_as(args[1], ty);
    const std::string b = operand_as(args[2], ty);
    const std::string wh = "(int64_t)((uint64_t)(" + b + ") - (uint64_t)(" + a + "))";
    const std::string e = "(" + wh + " <= 0 ? (" + ty + ")0 : " + wh + " >= " + bits +
                          " ? ~(" + ty + ")0 : ((" + one + " << (unsigned)" + wh + ") - 1) << ((" +
                          a + ") & " + rng + "))";
    return guard + store(args[0], e, ty);
  }

  // sad.u32 d, a, b, c;  -- the sum of the absolute differences of the four byte
  // lanes, added to c. `.u32` reads the lanes unsigned and `.s32` signed, which is
  // the only thing the suffix changes.
  if (op == "sad") {
    if (!need(4)) return "";
    const bool sgn = opcode.find(".s32") != std::string::npos;
    const std::string a = operand_as(args[1], "uint32_t");
    const std::string b = operand_as(args[2], "uint32_t");
    std::string sum = "(int32_t)(" + operand_as(args[3], "uint32_t") + ")";
    for (int k = 0; k < 4; k++) {
      const std::string sh = std::to_string(8 * k);
      const std::string ea = sgn ? ("(int32_t)(int8_t)(((" + a + ") >> " + sh + ") & 0xFFu)")
                                 : ("(int32_t)(((" + a + ") >> " + sh + ") & 0xFFu)");
      const std::string eb = sgn ? ("(int32_t)(int8_t)(((" + b + ") >> " + sh + ") & 0xFFu)")
                                 : ("(int32_t)(((" + b + ") >> " + sh + ") & 0xFFu)");
      sum += " + " + math("abs", ea + " - " + eb);
    }
    return guard + store(args[0], "(uint32_t)(" + sum + ")", "uint32_t");
  }

  // mul24/mad24: both operands are the low 24 bits, widened at the suffix's
  // signedness, so the product is 48 bits and `.lo`/`.hi` pick a half of it.
  if (op == "mul24" || op == "mad24") {
    const bool mad = op == "mad24";
    if (!need(mad ? 4 : 3)) return "";
    const bool sgn = opcode.find(".s32") != std::string::npos;
    const bool hi = opcode.find(".hi") != std::string::npos;
    auto w24 = [&](const std::string& tok) -> std::string {
      const std::string v = operand_as(tok, "uint32_t");
      return sgn ? ("(int64_t)(((int32_t)((" + v + ") << 8)) >> 8)")
                 : ("(int64_t)(((" + v + ") & 0xFFFFFFu))");
    };
    std::string p = "(" + w24(args[1]) + " * " + w24(args[2]) + ")";
    if (mad) {
      const std::string c = operand_as(args[3], "uint32_t");
      p = "(" + p + " + (int64_t)" + (sgn ? "(int32_t)" : "(uint32_t)") + "(" + c + "))";
    }
    const std::string e = hi ? ("(uint32_t)((uint64_t)(" + p + ") >> 32)")
                             : ("(uint32_t)(uint64_t)(" + p + ")");
    return guard + store(args[0], e, "uint32_t");
  }

  // A sleep is a scheduling hint, not an operation: nothing it does can reach a
  // register or a buffer, so the answer is the same with it and without it.
  if (op == "nanosleep") return guard + "(void)0;";

  // call{.uni} target, (a, b);  and  call{.uni} (retvals), target, (a, b);
  //
  // The callee is a function in this module, translated once with a register file
  // of its own (see the module pass), and the call is a C++ call to it. A value it
  // hands back rides in its state struct: the generated function is a switch over a
  // program counter, so the `ret` that ends it is a `return` from the middle of that
  // switch. What is refused is refused by name: a call to a function whose body is
  // not in this module has nothing to inline, and a return list with more than one
  // slot has no field here to land in.
  if (op == "call") {
    if (args.size() < 2) { err = "malformed '" + opcode + "'"; return ""; }
    // `call.uni (%r1), f, (a);` names where the answer goes, then the function, then
    // the arguments.
    size_t next = 0;
    std::string tgt = trim(args[next++]);
    std::string ret_dst;
    if (!tgt.empty() && tgt[0] == '(') {
      const std::string inner = trim(tgt.substr(1, tgt.size() - 2));
      const std::vector<std::string> slots =
          inner.empty() ? std::vector<std::string>() : split_top(inner, ',');
      if (slots.size() != 1) {
        err = "'" + opcode + "' returns " + std::to_string(slots.size()) +
              " values through a parenthesised list, and a call here carries one";
        return "";
      }
      ret_dst = trim(slots[0]);
      if (next >= args.size()) { err = "malformed '" + opcode + "'"; return ""; }
      tgt = trim(args[next++]);
    }
    auto f = callees.find(tgt);
    if (f == callees.end()) {
      auto bad = callee_errs.find(tgt);
      if (bad != callee_errs.end())
        err = "'" + opcode + " " + tgt + "': the callee did not translate: " + bad->second;
      else
        err = "'" + opcode + "' names '" + tgt + "', which has no body in this module: "
              "a call to a function defined elsewhere has nothing to inline here";
      return "";
    }
    std::string list = (next < args.size()) ? trim(args[next]) : std::string();
    if (!list.empty() && list.front() == '(' && list.back() == ')')
      list = list.substr(1, list.size() - 2);
    const std::vector<std::string> avals =
        trim(list).empty() ? std::vector<std::string>() : split_top(list, ',');
    if (avals.size() != f->second.size()) {
      err = "'" + opcode + " " + tgt + "' passes " + std::to_string(avals.size()) +
            " arguments and the function declares " + std::to_string(f->second.size());
      return "";
    }
    // The answer the call has to place, or the function's answer the call has nowhere
    // to put: both are said rather than dropped.
    auto rt = callee_rets.find(tgt);
    const std::string rty = (rt == callee_rets.end()) ? std::string() : rt->second;
    if (ret_dst.empty() && !rty.empty()) {
      err = "'" + opcode + " " + tgt + "' returns a value and the call has no list to put it in";
      return "";
    }
    if (!ret_dst.empty() && rty.empty()) {
      err = "'" + opcode + " " + tgt + "' is in a return list, and that function returns nothing";
      return "";
    }
    const std::string mem = "wcf_" + tgt;
    bool have = false;
    for (const auto& m : callee_members)
      if (m.second == mem) have = true;
    if (!have) callee_members.emplace_back("weft_cf_" + tgt, mem);
    // A callee is per-thread scalar work: it has no lane dimension, so a kernel
    // that calls one is emitted for the per-lane body rather than the warp one.
    vec_ok = false;
    std::string call = "weft_f_" + tgt + "(ctx, &s->" + mem;
    for (size_t i = 0; i < avals.size(); i++)
      call += ", (" + f->second[i] + ")(" + operand_as(avals[i], f->second[i]) + ")";
    call += ");";
    // A callee that can give the worker back in the middle makes the call site a
    // resume point too: the fresh call starts the callee at its top, and a re-entry
    // jumps straight back to the call rather than walking to it again. The callee's
    // counter says where it stopped, and until that is the finished sentinel the caller
    // stops here as well -- one level up per call, until the kernel's own counter is
    // what parks.
    auto sp = callee_suspends.find(tgt);
    const bool suspends = (sp != callee_suspends.end() && sp->second);
    if (!suspends) {
      if (!ret_dst.empty()) call += " " + store(ret_dst, "s->" + mem + ".weft_ret", rty);
      return guard + call;
    }
    const int rid = next_resume++;
    const std::string rids = std::to_string(rid);
    resume_cases += "    case " + rids + ": goto R" + rids + ";\n";
    if (!guard.empty()) {
      // A predicated call to a callee that stops: `if (cond) ` in front of the call
      // would be gone by the time the lane came back, because a resume jumps straight
      // to the label after it. So the decision is recorded in a bit of this call site's
      // own state, the label goes after the fresh-call reset, and the re-entry asks the
      // record rather than a register that may have been rewritten since.
      const std::string cond = guard.substr(4, guard.size() - 6);
      const std::string cg = "cg" + rids;
      callee_members.emplace_back("uint32_t", cg);
      std::string out = "s->" + cg + " = (" + cond + ") ? 1u : 0u;\n  if (s->" + cg +
                        ") s->" + mem + ".pc = 0u;\n  R" + rids + ":;\n  if (s->" + cg +
                        ") {\n  " + call;
      out += "\n  if (s->" + mem + ".pc != 0xFFFFFFFFu) { s->pc = " + rids +
             "u; return; }";
      if (!ret_dst.empty())
        out += "\n  " + store(ret_dst, "s->" + mem + ".weft_ret", rty);
      out += "\n  }";
      return out;
    }
    std::string out = "s->" + mem + ".pc = 0u;\n  R" + rids + ":;\n  " + call;
    out += "\n  if (s->" + mem + ".pc != 0xFFFFFFFFu) { s->pc = " + std::to_string(rid) +
           "u; return; }";
    if (!ret_dst.empty()) out += " " + store(ret_dst, "s->" + mem + ".weft_ret", rty);
    return out;
  }

  err = "unsupported PTX instruction '" + opcode + "'";
  return "";
}

}  // namespace

static TranslateResult translate_ptx_impl(const std::string& raw,
                                          const std::vector<std::pair<int, uint32_t>>& widths,
                                          bool gpu) {
  TranslateResult r;
  std::string src = strip_comments(raw);
  // The module prologue: everything before the first `.entry`. A `.shared` array
  // declared here is file scope in PTX, one instance per threadgroup, and every
  // kernel that names it addresses that same arena; a declaration inside an entry
  // is that kernel's own. Both have to be laid out, or a module-scope tile is
  // addressed at offset 0 of an arena the GPU never declares.
  const size_t first_entry = src.find(".entry");
  const std::string module_head =
      src.substr(0, first_entry == std::string::npos ? src.size() : first_entry);

  //
  // `.tex .u32 name, 1, 1, 1, 1;` declares a sampled texture and `.surf .u64
  // name, ...;` a surface the kernel also writes into. The declarations are
  // numbered in order, and that order is the order `--textures` fills, so a
  // kernel and its binding agree without the caller being handed an index.
  std::map<std::string, int> tex_index;
  std::vector<std::string> tex_names;  // index -> the declared name
  int ntex = 0;
  {
    const std::string& head = module_head;
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
      // Three spellings name a binding: `.global .texref t` and `.global .surfref t`,
      // which is the form PTX accepts today, plus the older `.tex .u32 t, ...` and
      // `.texref t` the dumps use. The name is the token after the directive.
      std::string nm;
      if (s.rfind(".global .texref", 0) == 0 || s.rfind(".global .surfref", 0) == 0) {
        nm = trim(s.substr(s.find_last_of(" \t") + 1));
      } else if (s.rfind(".texref", 0) == 0 || s.rfind(".surfref", 0) == 0) {
        nm = trim(s.substr(s.find_first_of(" \t") + 1));
      } else if (s.rfind(".tex", 0) == 0 || s.rfind(".surf", 0) == 0) {
        std::string rest = trim(s.substr(s.find_first_of(" \t")));
        nm = rest;
        const size_t sp = rest.find_first_of(" \t");
        if (sp != std::string::npos) nm = rest.substr(sp + 1);
        nm = trim(nm.substr(0, nm.find(',')));
      } else {
        continue;
      }
      if (nm.empty() || tex_index.count(nm)) continue;
      tex_index[nm] = ntex++;
      tex_names.push_back(nm);
    }
  }

  std::string body_out;
  std::vector<TranslatedKernel> kernels;

  //
  // The module's `.func` bodies, translated once each and emitted ahead of the
  // kernels that call them. A callee is a function of its own rather than its
  // statements pasted into the caller, because PTX gives it its own register file:
  // its `%rd1` is not the caller's, and pasting would make the two one slot.
  // Only a function a `call` names is translated: a module can carry device
  // functions no kernel reaches, and failing the module over one of those would be
  // a refusal for code nothing runs.
  std::map<std::string, std::vector<std::string>> callee_types;
  std::map<std::string, std::string> callee_rets;
  std::map<std::string, bool> callee_suspends;
  std::map<std::string, std::string> callee_errs;
  {
    //
    // The module's `.func` bodies are collected before any of them is translated,
    // because a callee can call one written further down the file: what a `call`
    // inside a body names is only known once every body has been read.
    struct FuncDef {
      std::string name;
      std::vector<std::string> pnames, ptypes;
      std::string ret_name, ret_type;  // the return slot, if the list names one
      std::vector<std::string> deps;
      std::string body;  // the text between the braces
      bool called = false;
    };
    std::vector<FuncDef> defs;
    std::map<std::string, size_t> def_at;

    // The functions a body calls: the token after the `.uni`, or the one after the
    // return list when the call carries one.
    auto call_targets = [](const std::string& body, std::vector<std::string>& out) {
      size_t q = 0;
      while ((q = body.find("call", q)) != std::string::npos) {
        if (q > 0 && (is_word(body[q - 1]) || body[q - 1] == '.' || body[q - 1] == '%')) {
          q += 4;
          continue;
        }
        size_t p = q + 4;
        while (p < body.size() &&
               (is_word(body[p]) || body[p] == '.' || body[p] == ':')) p++;
        for (;;) {
          while (p < body.size() && std::isspace((unsigned char)body[p])) p++;
          if (p < body.size() && body[p] == '(') {
            const size_t rc = find_matching(body, p, '(', ')');
            if (rc == std::string::npos) { p = body.size(); break; }
            p = rc + 1;
            continue;
          }
          if (p < body.size() && body[p] == ',') { p++; continue; }
          break;
        }
        const size_t ns = p;
        while (p < body.size() && (is_word(body[p]) || body[p] == '$' || body[p] == '.')) p++;
        if (p > ns) out.push_back(body.substr(ns, p - ns));
        q = p > q + 4 ? p : q + 4;
      }
    };

    size_t at = 0;
    while ((at = src.find(".func", at)) != std::string::npos) {
      at += 5;
      size_t p = at;
      while (p < src.size() && std::isspace((unsigned char)src[p])) p++;
      // `.func (.param .b32 rv) name(...)` puts the return list before the name.
      // The name is taken whichever side of it the list is on.
      std::string ret_decl;
      if (p < src.size() && src[p] == '(') {
        const size_t rc = find_matching(src, p, '(', ')');
        if (rc == std::string::npos) break;
        ret_decl = src.substr(p + 1, rc - p - 1);
        p = rc + 1;
        while (p < src.size() && std::isspace((unsigned char)src[p])) p++;
      }
      const size_t ns = p;
      while (p < src.size() && is_word(src[p])) p++;
      const std::string fname = src.substr(ns, p - ns);
      if (fname.empty()) continue;
      const size_t lp = src.find('(', p);
      if (lp == std::string::npos) continue;
      const size_t rp = find_matching(src, lp, '(', ')');
      if (rp == std::string::npos) continue;
      // A declaration ends at its `;`. It names a function for a call written above
      // the body, and the body that follows belongs to the next `.func`: reading on
      // for a `{` here would give this name someone else's calls.
      size_t after = rp + 1;
      while (after < src.size() && std::isspace((unsigned char)src[after])) after++;
      if (after < src.size() && src[after] == ';') {
        at = after + 1;
        continue;
      }
      const size_t lb = src.find('{', rp);
      if (lb == std::string::npos) continue;
      const size_t rb = find_matching(src, lb, '{', '}');
      if (rb == std::string::npos) continue;

      FuncDef d;
      d.name = fname;
      d.body = src.substr(lb + 1, rb - lb - 1);
      // A call that names the function is what makes it worth translating: a
      // module can carry device functions no kernel reaches, and failing the
      // module over one of those would be a refusal for code nothing runs.
      for (size_t q = 0; (q = src.find(fname, q)) != std::string::npos; q += fname.size()) {
        const size_t back = q > 24 ? q - 24 : 0;
        const size_t c = src.find("call", back);
        if (c != std::string::npos && c < q) { d.called = true; break; }
      }
      bool bad = false;
      for (const std::string& pt : split_top(src.substr(lp + 1, rp - lp - 1), ',')) {
        if (trim(pt).empty()) continue;
        const std::vector<std::string> toks = split_top(trim(pt), ' ');
        std::string nm = toks.empty() ? std::string() : toks.back();
        if (!nm.empty() && nm[0] == '%') nm = nm.substr(1);
        std::string ty;
        for (const auto& t : toks) {
          if (t.empty() || t[0] != '.') continue;
          const std::string ct = cpp_type(t);
          if (!ct.empty() && cpp_size(ct)) { ty = ct; break; }
        }
        if (nm.empty()) { bad = true; break; }
        d.pnames.push_back(nm);
        d.ptypes.push_back(ty.empty() ? "uint64_t" : ty);
      }
      if (bad) {
        callee_errs[fname] = "a parameter of the function is not a name";
        continue;
      }
      // The return list is one slot: PTX allows several, and the state struct has one
      // field to put an answer in, so a longer list is refused by name rather than
      // folded into something the caller would have to unpack.
      if (!trim(ret_decl).empty()) {
        const std::vector<std::string> slots = split_top(trim(ret_decl), ',');
        if (slots.size() != 1) {
          callee_errs[fname] = "it returns " + std::to_string(slots.size()) +
                               " values through its parenthesised list, and a call here "
                               "carries one";
          continue;
        }
        const std::vector<std::string> toks = split_top(trim(slots[0]), ' ');
        std::string nm = toks.empty() ? std::string() : toks.back();
        if (!nm.empty() && nm[0] == '%') nm = nm.substr(1);
        std::string ty;
        for (const auto& t : toks) {
          if (t.empty() || t[0] != '.') continue;
          const std::string ct = cpp_type(t);
          if (!ct.empty() && cpp_size(ct)) { ty = ct; break; }
        }
        if (nm.empty()) {
          callee_errs[fname] = "the return slot of the function is not a name";
          continue;
        }
        d.ret_name = nm;
        d.ret_type = ty.empty() ? "uint32_t" : ty;
      }
      call_targets(d.body, d.deps);
      if (def_at.count(fname)) continue;  // one definition per name
      def_at[fname] = defs.size();
      defs.push_back(d);
    }

    // What has to be translated: the functions a `call` names, plus everything they
    // call, because a call carries the callee's state as a member of its own and
    // that member has to have a body.
    std::map<std::string, bool> needed;
    {
      std::vector<std::string> work;
      for (const FuncDef& d : defs)
        if (d.called) { needed[d.name] = true; work.push_back(d.name); }
      while (!work.empty()) {
        const std::string nm = work.back();
        work.pop_back();
        auto it = def_at.find(nm);
        if (it == def_at.end()) continue;
        for (const std::string& dep : defs[it->second].deps)
          if (def_at.count(dep) && !needed.count(dep)) {
            needed[dep] = true;
            work.push_back(dep);
          }
      }
    }

    std::map<std::string, int> state;  // 1 translated, 2 refused: both are answers
    std::string callee_out;

    auto translate = [&](size_t i) -> bool {
      const std::string& fname = defs[i].name;
      Emitter ef;
      ef.msl = gpu;
      ef.callees = callee_types;
      ef.callee_errs = callee_errs;
      ef.callee_rets = callee_rets;
      ef.callee_suspends = callee_suspends;
      ef.callee_ret = defs[i].ret_name;
      ef.reg_defs.clear();
      ef.scan_defs(defs[i].body);
      for (size_t k = 0; k < defs[i].pnames.size(); k++)
        ef.callee_args[defs[i].pnames[k]] = "weft_a" + std::to_string(k);
      std::string decls, code;
      bool ok = true;
      size_t q = 0;
      while (q < defs[i].body.size()) {
        // One statement, up to its semicolon: `emit` takes a statement, and a `;`
        // left on the end is not part of the operand list (`[src];` is not a memory
        // operand).
        size_t semi = defs[i].body.find(';', q);
        if (semi == std::string::npos) semi = defs[i].body.size();
        const std::string line = trim(defs[i].body.substr(q, semi - q));
        q = semi + 1;
        if (line.empty()) continue;
        if (line.rfind(".reg", 0) == 0) {
          // PTX separates tokens by any run of whitespace, and generators indent
          // with tabs, so the declaration is split on tabs as well as spaces.
          std::string decl = line;
          for (char& c : decl) if (c == '\t') c = ' ';
          std::vector<std::string> toks = split_top(decl, ' ');
          std::string ty, reg;
          for (const auto& t : toks) {
            if (!t.empty() && t[0] == '.') {
              const std::string ct = cpp_type(t);
              if (!ct.empty()) ty = ct;
            } else {
              reg = t;
            }
          }
          if ((gpu || ef.msl) && ty == "double") ty = "uint64_t";
          Reg rr = parse_reg(reg);
          if (ty.empty() || !rr.ok) {
            ef.err = "bad register declaration '" + line + "'";
            ok = false;
            break;
          }
          int count = 1;
          const size_t lt = reg.find('<');
          if (lt != std::string::npos) count = std::atoi(reg.c_str() + lt + 1);
          ef.reg_types[rr.base] = ty;
          ef.reg_counts[rr.base] = count;
          decls += "  " + ty + " " + rr.base + "[" + std::to_string(count) + "];\n";
          continue;
        }
        // A label inside a function. The body becomes the same switch a kernel body
        // is -- a branch is a write to `pc` and a restart of the case table -- because
        // Metal has no goto (tools/msl_probe.mm); on the host column a label and a
        // `goto` are the language's own, and the function is a plain function.
        std::string stmt = line;
        const size_t colon = stmt.find(':');
        if (colon != std::string::npos && colon > 0 &&
            stmt.substr(0, colon).find_first_of(" \t.:") == std::string::npos) {
          const std::string lab = trim(stmt.substr(0, colon));
          if (ef.msl)
            code += "    case " + std::to_string(ef.label_id(lab)) + "u:;\n";
          else
            code += "  " + sanitize(lab) + ":;\n";
          stmt = trim(stmt.substr(colon + 1));
          if (stmt.empty()) continue;
        }
        if (!stmt.empty() && stmt[0] == '.') {
          ef.err = "statement '" + stmt +
                   "' inside the function is a directive the call does not carry";
          ok = false;
          break;
        }
        const std::string st = ef.emit(stmt);
        if (!ef.err.empty()) { ok = false; break; }
        code += "  " + st + "\n";
      }
      if (!ok) {
        callee_errs[fname] = ef.err.empty() ? "the body translated to nothing" : ef.err;
        state[fname] = 2;
        return false;
      }
      // A callee's state holds the state of every function it calls, by value: those
      // structs were emitted first, so this one is complete here. Two functions that
      // call each other have no such shape, and never reach this.
      // The value the function hands back is a field of its own state, next to the
      // register file and the states of the functions it calls. Its zeroing is the
      // entry to the body, so a path that writes no answer does not report the last
      // call's.
      const std::string rty =
          (gpu && defs[i].ret_type == "double") ? std::string("uint64_t") : defs[i].ret_type;
      // A body that can give the worker back in the middle -- a rendezvous, or a call to
      // something that has one -- stops at a counter the caller resumes, and every one of
      // those stops leaves a resume case behind. A branch writes the counter too, but as
      // `continue` within its own switch, so what is left in `resume_cases` is exactly
      // the set of ways out of the middle of this body.
      const bool suspends = !ef.resume_cases.empty();
      std::string members = decls;
      for (const auto& m : ef.callee_members)
        members += "  " + m.first + " " + m.second + ";\n";
      if (!defs[i].ret_name.empty()) members += "  " + rty + " weft_ret;\n";
      std::string sig;
      for (size_t k = 0; k < defs[i].pnames.size(); k++)
        sig += ", " + defs[i].ptypes[k] + " weft_a" + std::to_string(k);
      callee_out += "struct weft_cf_" + fname + " {\n  uint32_t pc;\n" + members + "};\n";
      // A body with labels is a switch over the program counter, so the `case` a branch
      // writes is the case it lands on. A body that never stops is entered at its top;
      // one that can stop is entered at the counter the caller left, which is why the
      // reset only happens on a fresh call and the resume cases are part of the switch.
      std::string fbody = "  (void)ctx; (void)s;\n";
      if (!defs[i].ret_name.empty())
        fbody += std::string("  ") + (suspends ? "if (s->pc == 0u) " : "") + "s->weft_ret = " +
                 rty + "();\n";
      if (gpu) {
        if (!suspends) fbody += "  s->pc = 0u;\n";
        fbody += "  for (;;) {\n    switch (s->pc) {\n";
        fbody += ef.resume_cases;
        fbody += "      case 0u:;\n";
        fbody += code;
        fbody += "      default: return;\n    }\n  }\n";
      } else {
        // The host body is a plain function, so its resume cases are a switch at the
        // top rather than the switch the GPU body already is.
        if (suspends)
          fbody += "  if (s->pc != 0u) {\n    switch (s->pc) {\n" + ef.resume_cases +
                   "      default: break;\n    }\n  }\n";
        fbody += code;
      }
      callee_out += "static void weft_f_" + fname + "(" +
                    std::string(gpu ? "thread weft::Ctx& ctx" : "weft::Ctx& ctx") + ", " +
                    (gpu ? "thread " : "") + "weft_cf_" + fname + "* s" + sig +
                    ") {\n" + fbody + "}\n";
      callee_types[fname] = defs[i].ptypes;
      callee_rets[fname] = rty;
      callee_suspends[fname] = suspends;
      state[fname] = 1;
      return true;
    };

    // A callee is translated only after the functions it calls. A pass that
    // translates nothing is the point at which one is waiting on one that waits on
    // it, which is the shape a struct cannot hold.
    for (bool progress = true; progress;) {
      progress = false;
      for (size_t i = 0; i < defs.size(); i++) {
        const std::string& nm = defs[i].name;
        if (!needed.count(nm) || state.count(nm)) continue;
        bool waiting = false;
        std::string that;
        for (const std::string& dep : defs[i].deps) {
          if (dep == nm) {
            that = "'" + nm + "' calls itself, and the state a call carries is a struct "
                   "that would have to contain itself: refused by name rather than given "
                   "a depth limit";
            break;
          }
          if (!needed.count(dep)) continue;
          auto st = state.find(dep);
          if (st == state.end()) { waiting = true; continue; }
          if (st->second == 2) {
            that = "the function it calls, '" + dep + "', did not translate: " + callee_errs[dep];
            break;
          }
        }
        if (!that.empty()) {
          callee_errs[nm] = that;
          state[nm] = 2;
          progress = true;
        } else if (!waiting) {
          translate(i);
          progress = true;
        }
      }
    }
    // Whatever is left is a chain that comes back to itself, one name at a time.
    for (size_t i = 0; i < defs.size(); i++) {
      const std::string& nm = defs[i].name;
      if (!needed.count(nm) || state.count(nm)) continue;
      callee_errs[nm] = "it calls a function that calls it back, and the state a call "
                        "carries is a struct that would have to contain itself: refused "
                        "by name rather than given a depth limit";
      state[nm] = 2;
    }
    body_out += callee_out;
  }

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

    // `.maxntid` describes the entry it belongs to: the launch size the module was
    // compiled for. Triton writes it from `num_warps`, and it is the only record of
    // the geometry a kernel was indexed against. The legal place is between the
    // parameter list and the body; the dumps that put it above the `.entry` line are
    // not legal PTX, and both spellings mean the same thing here.
    uint32_t tg_hint = 0;
    {
      size_t m = src.rfind(".maxntid", e);
      if (m == std::string::npos || m < pos) m = src.find(".maxntid", close);
      if (m != std::string::npos && m < brace) {
        const size_t semi = src.find(';', m);
        const size_t v = src.find_first_of("0123456789", m);
        if (v != std::string::npos && semi != std::string::npos && v < semi)
          tg_hint = (uint32_t)std::strtoul(src.c_str() + v, nullptr, 10);
      }
    }

    Emitter em;
    em.callees = callee_types;
    em.callee_errs = callee_errs;
    em.callee_rets = callee_rets;
    em.callee_suspends = callee_suspends;
    em.tex_index = tex_index;
    em.ntex = ntex;
    em.tex_write.assign((size_t)ntex, false);
    // The body says which bindings are arrays, an `.a2d` instruction names one
    // with `[name, {...}]`, so it is scanned once here rather than guessed from
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
    //
    // A pointer the body never names is not bound. Triton hands every kernel a
    // `global_scratch` pointer and most kernels never touch it; a CUDA kernel
    // declares parameters an `#if` compiled out. Binding one would make the
    // caller invent a buffer for a slot nothing reads, so buffer indices are
    // assigned over the pointers that are actually used, which is the order
    // Triton's own launcher binds them in.
    struct PtxParam {
      std::string name;
      bool ptr = false;
      std::string type;  // the value's own token, without its dot
    };
    auto value_token = [](const std::string& t) {
      static const char* kTypes[] = {".u8",  ".s8",  ".b8",  ".u16", ".s16", ".b16",
                                     ".f16", ".u32", ".s32", ".b32", ".f32", ".u64",
                                     ".s64", ".b64", ".f64", ".pred"};
      for (const char* k : kTypes)
        if (t == k) return true;
      return false;
    };
    std::vector<PtxParam> pdecl;
    for (const std::string& p : split_top(src.substr(paren + 1, close - paren - 1), ',')) {
      if (p.empty()) continue;
      std::vector<std::string> toks = split_top(p, ' ');
      PtxParam pd;
      pd.name = toks.back();
      for (const auto& t : toks) {
        if (t == ".u64" || t == ".b64" || t == ".s64" || t == ".ptr") pd.ptr = true;
        if (pd.type.empty() && value_token(t)) pd.type = t.substr(1);
      }
      pdecl.push_back(pd);
    }

    std::string body = src.substr(brace + 1, end - brace - 1);

    // How often each register is named in the body, its declaration aside. A
    // register named once is one nothing reads: a result that is written and
    // dropped. That is the difference between `atom.add.u64`, which has to
    // return the pair as one value and cannot here, and `red.add.u64`, whose
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
      if (!pd.ptr) {
        // Scalars keep their declared positions: `--arg I` names the I-th scalar
        // the kernel declares, whether or not this build reads it.
        em.scalars[pd.name] = (int)em.scalars.size();
      } else if (mentions(body, pd.name)) {
        em.params[pd.name] = (int)em.params.size();
      } else {
        em.unused.push_back(pd.name);
      }
    }
    std::string shared_decl;

    // Shared memory is declared before use, but scan it out first so the
    // instruction pass knows the arena exists.
    //
    // Every `.shared` array gets its own offset in the group's window, laid out
    // in declaration order and aligned to 16 bytes, so a kernel with two tiles
    // sees two tiles. `extern .shared`, a size chosen by the launch, sits after
    // the static ones, which is where CUDA's dynamic shared memory lives; weft's
    // launch gives the whole 48 KiB arena to it.
    {
      uint64_t off = 0;
      bool any = false;
      std::vector<std::pair<std::string, bool>> externs;
      // The prologue and the body in one scan, the prologue first: a module-scope
      // array has to sit in the arena the launch hands the kernel, and before the
      // kernel's own arrays so an offset in the body can never land inside it. The
      // newline between them keeps a declaration that opens the body from being
      // glued onto the prologue's last line.
      const std::string scan = module_head + "\n" + body;
      for (size_t p = 0; (p = scan.find(".shared", p)) != std::string::npos; p += 7) {
        // Declarations only: `.shared` is also the space qualifier of
        // `st.shared.f32`, which follows an opcode rather than a statement start.
        size_t q = p;
        while (q > 0 && (scan[q - 1] == ' ' || scan[q - 1] == '\t')) q--;
        bool decl_start = (q == 0 || scan[q - 1] == '\n' || scan[q - 1] == ';' || scan[q - 1] == '{');
        // `.extern .shared` is a declaration too: the token before `.shared` is
        // the `extern` qualifier, not the start of a statement.
        if (!decl_start && q >= 6 && scan.compare(q - 6, 6, "extern") == 0) decl_start = true;
        if (!decl_start) continue;
        const size_t semi = scan.find(';', p);
        if (semi == std::string::npos) break;
        const std::string d = trim(scan.substr(p + 7, semi - p - 7));
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
        // The bracket is a count of elements, not of bytes: `.b32 tile[1024]` is four
        // kilobytes, which is what CUDA means by it and what the kernel writes. Sizing
        // it as 1024 bytes put the array after it inside this one.
        uint64_t esz = 4;
        for (const auto& t : split_top(d, ' ')) {
          if (t.empty() || t[0] != '.') continue;
          const std::string et = cpp_type(t);
          if (!et.empty() && cpp_size(et)) { esz = (uint64_t)cpp_size(et); break; }
        }
        off = (off + 15) & ~(uint64_t)15;
        em.shared_offsets[nm] = off;
        off += std::strtoull(extent.c_str(), nullptr, 10) * esz;
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
    // builders are shared; only the framing differs, one lane at a time, or the
    // 32 lanes of a warp as one vector.
    auto walk = [&](Emitter& e, std::string& out, std::string& struct_decls) -> bool {
      // Counted before the body is emitted, because a memory instruction's text
      // depends on how many times the register it reads is written.
      e.reg_defs.clear();
      e.scan_defs(body);
      std::string stmt, lane_batch;
      // Can this statement be fused into a lane loop with its neighbours? Only if
      // every element it touches is indexed by the lane number itself. A gather
      // (`r[18][l + d]`, what a shuffle compiles to) has a runtime index, and
      // one of those in the loop stops the vectoriser for the whole loop, which
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

        // A label glued to its instruction: "$L_1: add.f32 ...". A label is a name, so
        // it has no dot and no second colon, otherwise `cp.async.bulk.shared::cluster`
        // reads as a label called `cp.async.bulk.shared` followed by an instruction
        // beginning `:cluster`.
        size_t colon = s.find(':');
        std::string lab;
        if (colon != std::string::npos && colon > 0 &&
            s.substr(0, colon).find_first_of(" \t.:") == std::string::npos) {
          flush_lanes();
          lab = trim(s.substr(0, colon));
          if (e.vec)
            out += "  L" + std::to_string(e.label_id(lab)) + ":;\n";
          else if (e.msl)
            out += "  case " + std::to_string(e.label_id(lab)) + "u:;\n";
          else
            out += "  " + sanitize(lab) + ":;\n";
          s = trim(s.substr(colon + 1));
          if (s.empty()) continue;
        }

        // `ts: .branchtargets L1, L2, L3;` is a directive at a label, and the list
        // is the table a `brx.idx` indexes. It is recorded under that label rather
        // than emitted: a jump table is a choice of destinations, not storage. The
        // compact form `N<3>` names N0, N1, N2, which the isa allows here.
        if (s.rfind(".branchtargets", 0) == 0) {
          if (lab.empty()) {
            e.err = "a .branchtargets list with no label to name it";
            return false;
          }
          std::vector<std::string> tgts;
          for (const auto& tok : split_top(trim(s.substr(14)), ',')) {
            std::string nm = trim(tok);
            if (nm.empty()) continue;
            const size_t lt = nm.find('<');
            if (lt != std::string::npos && nm.back() == '>') {
              const int n = std::atoi(nm.c_str() + lt + 1);
              const std::string pre = nm.substr(0, lt);
              for (int k = 0; k < n; k++) tgts.push_back(pre + std::to_string(k));
            } else {
              tgts.push_back(nm);
            }
          }
          if (tgts.empty()) { e.err = "an empty .branchtargets list"; return false; }
          e.brxtab[lab] = tgts;
          continue;
        }

        if (s.rfind(".reg", 0) == 0) {
          std::string decl = s;
          for (char& c : decl) if (c == '\t') c = ' ';
          std::vector<std::string> toks = split_top(decl, ' ');
          std::string ty, reg;
          for (const auto& t : toks) {
            if (!t.empty() && t[0] == '.') {
              std::string ct = cpp_type(t);
              // A `.f16` register holds the 16 bits of a half, which is the same
              // storage `.b16` names; every half instruction here works on those
              // bits, so the register is declared the same way. Triton and ptxas
              // both write `.reg .f16`, and the declaration was refused by name.
              if (ct.empty() && t == ".f16") ct = "uint16_t";
              if (!ct.empty()) ty = ct;
            } else {
              reg = t;
            }
          }
          Reg rr = parse_reg(reg);
          if (ty.empty() || !rr.ok) {
            e.err = "bad register declaration '" + s + "'";
            return false;
          }
          int count = 1;
          size_t lt = reg.find('<');
          if (lt != std::string::npos) count = std::atoi(reg.c_str() + lt + 1);
          // A double on the GPU target is the `uint64_t` its bits live in, the
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
          e.reg_counts[rr.base] = count;
          struct_decls += "  " + ty + " " + rr.base + "[" + std::to_string(count) + "]" +
                          (e.vec ? "[32]" : "") + ";\n";
          continue;
        }
        if (s.rfind(".shared", 0) == 0 || s.rfind(".extern", 0) == 0) continue;
        if (!s.empty() && s[0] == '.') continue;  // .loc, .pragma, .maxnreg ...

        std::string line = e.emit(s);
        if (!e.err.empty()) return false;
        // `@%p op`: the lanes that fail the predicate must keep what they had.
        // The active mask alone does not say that, a predicate selects lanes
        // *within* the group, so the statement is guarded by it as well.
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
      // A callee's register file is a member of the caller's state, because the two
      // are one thread: the struct carries it, the function addresses it.
      for (const auto& m : e.callee_members)
        struct_decls += "  " + m.first + " " + m.second + ";\n";
      // The condition code a carry chain keeps between instructions, declared with
      // the registers because it lives exactly as long as they do.
      for (const auto& m : e.cc_members) struct_decls += "  " + m + "\n";
      return true;
    };

    em.msl = gpu;
    std::string code, struct_decls;
    const bool lane_ok = walk(em, code, struct_decls);

    // The warp backend is optional: anything the vector path cannot express
    // leaves the kernel on the per-lane state machine instead of failing.
    Emitter emv;
    emv.vec = true;
    emv.callees = callee_types;
    emv.callee_errs = callee_errs;
    emv.callee_rets = callee_rets;
    emv.callee_suspends = callee_suspends;
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

    // A kernel with no per-lane body, a tensor-core fragment, is emitted for the
    // GPU through the warp back end instead: one Metal thread runs the whole
    // 32-lane warp, so a peer's register is `r[l]` and no shuffle is involved.
    // What that back end cannot express yet, a barrier, a shuffle, threadgroup
    // memory, still has no GPU form, and the message says which one stopped it.
    // The per-lane body stops at the first instruction it cannot express, a
    // tensor-core one, so a buffer reached only after that point is typed by the
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
    // where the lock table lives, so a kernel that holds one is emitted for the
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
      // Both bodies failed, and the per-lane one stops at the same instruction the
      // warp one is needed for, so the message has to carry the warp body's own
      // reason, or it names the one error that is not the problem.
      r.error = "kernel " + name + ": " + em.err;
      if (have_vec) r.error += "; the warp body could not be used either";
      else if (!emv.err.empty()) r.error += " (warp body: " + emv.err + ")";
      else r.error += " (the warp body translated to nothing)";
      return r;
    }

    if (gpu && ((!lane_ok && warp_only) || wide_gpu)) {
      // The warp kernel: the same register file the CPU warp back end runs, as a
      // thread-local struct, with the dispatcher's computed goto written as a
      // switch, a shader has no jump table, but it has cases and labels.
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
      // once, so there is nothing to suspend, the kernel calls the body again
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
      // inside it, a lane's index is its position inside the threadgroup, so a
      // kernel that derives its fragment from `tid` sees 0..tg-1 in every group.
      body_out += "  ctx.lid0 = weft_lid * weft::kWarpLanes;\n";
      body_out += "  ctx.tgid = weft_tgid.x;\n";
      body_out += "  ctx.tg_size = (uint32_t)weft_geom[1];\n";
      body_out += "  ctx.grid = (uint32_t)weft_geom[0];\n";
      // A lane past the end of the launch does not exist, and a zeroed program
      // counter says "about to start", so the last warp of a short launch would run
      // lanes the program does not have and store past its buffer. This is the same
      // rule from the same geometry as the CPU driver's, so both engines run the same
      // lanes and a warp-only kernel answers the same on each.
      body_out += "  for (uint32_t l = 0; l < weft::kWarpLanes; l++)\n";
      body_out += "    s->pc[l] = ((ctx.lid0 + l) < "
                  "weft::live_threads(ctx.tgid, ctx.tg_size, ctx.grid)) ? 0u : "
                  "0xFFFFFFFFu;\n";

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
      // group, so the body is a lambda invoked once per lane, and `return` is
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
      // clobber the lane masks held in local registers, without it the mask is
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
    // The signature's own order, so a driver that takes the PTX as written is
    // handed the slots in the order the declaration lists them.
    for (const auto& pd : pdecl) {
      TranslatedKernel::Param gp;
      gp.type = pd.type;
      auto b = em.params.find(pd.name);
      if (b != em.params.end()) gp.buf = b->second;
      auto sl = em.scalars.find(pd.name);
      if (sl != em.scalars.end()) gp.scalar = sl->second;
      tk.params.push_back(gp);
    }
    // Each binding's element width, from the instructions that reach it. A
    // parameter with no typed access keeps 0, unknown, not four bytes.
    tk.buf_type.assign((size_t)tk.nbufs, std::string("0"));
    for (const auto& pb : em.param_bytes)
      if (pb.first >= 0 && pb.first < tk.nbufs && pb.second)
        tk.buf_type[(size_t)pb.first] = std::to_string(pb.second);
    tk.sync = em.uses_barrier || em.uses_shfl || !shared_decl.empty();
    tk.flat = false;
    tk.warp = have_vec;
    tk.tg_hint = tg_hint;
    tk.ntex = em.ntex;
    tk.tex_name = tex_names;
    // What each binding is, by name. PTX's `.tex` says nothing about the shape, so
    // the instruction that reads it is what names it, and a 1-D texture lowers
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
    std::string g = "// generated by weft (ptx front end, metal target), do not edit\n";
    g += kWWeftMslPreamble;
    g += kWWeftWarpMsl;
    // The portable library. Metal's `sqrt` is `x * rsqrt(x)` bit for bit and its
    // `ex2`, `lg2`, `sin` and `cos` are the unit's own polynomials, so a `.approx`
    // form lowered to the hardware answers differently from the same form lowered
    // to the host's libm, measured at 3.4e-07 relative on 65,536 lanes before
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
      "// generated by weft (ptx front end), do not edit\n"
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
      // `F2I` saturates: a float past the destination's range clamps to its end and
      // a NaN answers zero, which is the `.NTZ` bit. A C++ cast is undefined for
      // all of those and answered 0 for both infinities, where the device answers
      // the ends. The comparisons are on the double form, so one helper serves an
      // f32 and an f64 source and the bounds stay exact.
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
      "// A narrow source -- an integer literal under a 64-bit destination -- fills the low\n"
      "// bytes of a zeroed destination instead of reading past itself.\n"
      "static inline T weft_bitcast(U u) {\n"
      "  T t{};\n"
      "  std::memcpy(&t, &u, sizeof(U) < sizeof(T) ? sizeof(U) : sizeof(T));\n"
      "  return t;\n"
      "}\n"
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
      // `bswap` is the byte order, not the bit order: each byte keeps its bits and the
      // four (or eight) bytes trade places. Written as masks rather than
      // `__builtin_bswap32` so the same text is the GPU's as well.
      "static inline uint32_t weft_bswap32(uint32_t v) {\n"
      "  return ((v & 0x000000FFu) << 24) | ((v & 0x0000FF00u) << 8) |\n"
      "         ((v & 0x00FF0000u) >> 8) | ((v & 0xFF000000u) >> 24);\n"
      "}\n"
      "static inline uint64_t weft_bswap64(uint64_t v) {\n"
      "  return ((uint64_t)weft_bswap32((uint32_t)v) << 32) |\n"
      "         (uint64_t)weft_bswap32((uint32_t)(v >> 32));\n"
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
      // half, the form LLVM's NVPTX backend emits when it packs a pair.
      "static inline uint32_t weft_f16x2(float hi, float lo) {\n"
      "  return (uint32_t)weft_h16_bits(lo) | ((uint32_t)weft_h16_bits(hi) << 16);\n"
      "}\n"
      // `.rz` truncates toward zero, and the 16-bit conversion above rounds: so the
      // truncation is the rounded value walked back one step whenever it landed away
      // from zero. A finite value past the half range truncates to the largest
      // finite half rather than to an infinity; an infinity or a NaN has no
      // significand to drop and passes through. The same text runs on the GPU.
      "static inline uint16_t weft_h16_bits_rz(float x) {\n"
      "  uint16_t h = weft_h16_bits(x);\n"
      "  if ((h & 0x7C00u) == 0x7C00u) {\n"
      "    const uint32_t ux = weft_bitcast<uint32_t>(x);\n"
      "    if ((ux & 0x7F800000u) == 0x7F800000u) return h;\n"
      "    return (uint16_t)((h & 0x8000u) | 0x7BFFu);\n"
      "  }\n"
      "  const float back = weft_h16((uint32_t)h, 0);\n"
      "  const float ax = x < 0.0f ? -x : x, ab = back < 0.0f ? -back : back;\n"
      "  if (ab > ax && (h & 0x7FFFu) != 0u)\n"
      "    h = (uint16_t)((h & 0x8000u) | ((h & 0x7FFFu) - 1u));\n"
      "  return h;\n"
      "}\n"
      "static inline uint16_t weft_bf16_bits_rz(float x) {\n"
      "  return (uint16_t)(weft_bitcast<uint32_t>(x) >> 16);\n"
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
      // rather than overflowing to an infinity, what `cvt.rn.satfinite.e4m3x2.f32`
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
      // library call, the same arithmetic the Metal side writes, which is what
      // makes the two decodes bit-for-bit the same function.
      "  const uint32_t sb = (E + 117u) << 23;\n"
      "  float sc; std::memcpy(&sc, &sb, 4);\n"
      "  const float d = (E == 0) ? (float)mag / 512.0f : (float)(8u + mag) * sc;\n"
      "  return (float)((v & 0x80u) ? -d : d);\n"
      "}\n"
      // e5m2: 2 mantissa bits and a 5-bit exponent, with infinities and NaN, the
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
      // e2m1, e2m3 and e3m2: the MX formats, what a Blackwell fp4/fp6 convert
      // writes. Each is a sign, `eb` exponent bits and `mb` mantissa bits biased
      // 2^(eb-1)-1, and none of the three has an infinity or a NaN, so the
      // `.satfinite` the ISA makes mandatory on these converts is the whole range
      // rule: a NaN answers the positive MAX_NORM, and an infinity, whose magnitude
      // is past MAX_NORM, answers MAX_NORM keeping its sign.
      "static inline uint32_t weft_mx_bits(float x, int eb, int mb) {\n"
      "  const uint32_t maxbits = (uint32_t)((((1u << eb) - 1u) << mb) | ((1u << mb) - 1u));\n"
      "  uint32_t u; std::memcpy(&u, &x, 4);\n"
      "  const uint32_t sign = u >> 31;\n"
      "  if (!std::isfinite(x)) return (x != x) ? maxbits : ((sign << (eb + mb)) | maxbits);\n"
      "  const uint32_t mag = u & 0x7FFFFFFFu;\n"
      "  if (mag == 0u) return sign << (eb + mb);\n"
      "  const int bias = (1 << (eb - 1)) - 1;\n"
      "  const int E = (int)((mag >> 23) & 0xFFu) - 127 + bias;\n"
      "  if (E >= (1 << eb) - 1) return (sign << (eb + mb)) | maxbits;\n"
      "  const uint32_t mant = mag & 0x7FFFFFu;\n"
      "  uint32_t out;\n"
      "  if (E <= 0) {\n"
      "    // A subnormal sits on a grid of 2^(1-bias-mb). Scaling the magnitude onto\n"
      "    // it is exact, because the scale is a power of two, so the rounding is the\n"
      "    // same `rint` the e4m3 path uses and the two engines agree bit for bit.\n"
      "    out = (uint32_t)std::nearbyint(std::fabs(x) * (float)(1u << (mb + bias - 1)));\n"
      "  } else {\n"
      "    const int sh = 23 - mb;\n"
      "    uint32_t r = mant >> sh;\n"
      "    const uint32_t rem = mant & ((1u << sh) - 1u);\n"
      "    if (rem > (1u << (sh - 1)) || (rem == (1u << (sh - 1)) && (r & 1u))) r++;\n"
      "    out = ((uint32_t)E << mb) + r;\n"
      "  }\n"
      // A carry out of the mantissa is a carry into the exponent, and a rounding
      // that reaches 2^mb off the subnormal grid is the smallest normal: both are
      // already the right integer, so only the top of the range needs clamping.
      "  if (out >= maxbits) out = maxbits;\n"
      "  return (sign << (eb + mb)) | out;\n"
      "}\n"
      // The packed layouts are the ISA's: e2m1 puts the first source in d[7:4] and
      // the second in d[3:0]; the f6 formats put each six-bit value in the low bits
      // of a byte, d[13:8] then d[5:0], with the top two bits of each byte zero , 
      // which the mask leaves zero rather than shifting the value up.
      "static inline uint8_t weft_e2m1x2(float hi, float lo) {\n"
      "  return (uint8_t)((weft_mx_bits(lo, 2, 1) & 0xFu) | ((weft_mx_bits(hi, 2, 1) & 0xFu) << 4));\n"
      "}\n"
      "static inline uint16_t weft_e2m3x2(float hi, float lo) {\n"
      "  return (uint16_t)((weft_mx_bits(lo, 2, 3) & 0x3Fu) | ((weft_mx_bits(hi, 2, 3) & 0x3Fu) << 8));\n"
      "}\n"
      "static inline uint16_t weft_e3m2x2(float hi, float lo) {\n"
      "  return (uint16_t)((weft_mx_bits(lo, 3, 2) & 0x3Fu) | ((weft_mx_bits(hi, 3, 2) & 0x3Fu) << 8));\n"
      "}\n"
      "static inline uint16_t weft_e2m1x4(float a, float b, float e, float f) {\n"
      "  return (uint16_t)(((weft_mx_bits(a, 2, 1) & 0xFu) << 12) |\n"
      "                    ((weft_mx_bits(b, 2, 1) & 0xFu) << 8) |\n"
      "                    ((weft_mx_bits(e, 2, 1) & 0xFu) << 4) | (weft_mx_bits(f, 2, 1) & 0xFu));\n"
      "}\n"
      "static inline uint32_t weft_e2m3x4(float a, float b, float e, float f) {\n"
      "  return ((weft_mx_bits(a, 2, 3) & 0x3Fu) << 24) | ((weft_mx_bits(b, 2, 3) & 0x3Fu) << 16) |\n"
      "         ((weft_mx_bits(e, 2, 3) & 0x3Fu) << 8) | (weft_mx_bits(f, 2, 3) & 0x3Fu);\n"
      "}\n"
      "static inline uint32_t weft_e3m2x4(float a, float b, float e, float f) {\n"
      "  return ((weft_mx_bits(a, 3, 2) & 0x3Fu) << 24) | ((weft_mx_bits(b, 3, 2) & 0x3Fu) << 16) |\n"
      "         ((weft_mx_bits(e, 3, 2) & 0x3Fu) << 8) | (weft_mx_bits(f, 3, 2) & 0x3Fu);\n"
      "}\n"
      // `.rs` is stochastic rounding: the value is truncated onto the destination's
      // grid and then stepped one ulp away from zero when adding the random bits to
      // the discarded mantissa carries out. A float leaves thirteen bits below a
      // half and sixteen below a bf16, and the ISA's figure gives exactly that many
      // random bits per value, so at those widths the test is the plain integer
      // carry the ISA describes. `dropped << w` against `(2^w - r) << shift` is that
      // carry written as a comparison: it answers true for the same `dropped` values
      // (a tie carries, an exact grid value does not), and it stays exact when a
      // subnormal widens the dropped field past the width of the random bits.
      "static inline int weft_rs_carry(uint32_t mag, int e, int mb, int emin, uint32_t r, int w) {\n"
      "  int shift = 23 - mb + (e < emin ? emin - e : 0);\n"
      "  if (shift <= 0) return 0;\n"
      "  if (shift > 23) shift = 23;\n"
      "  const uint64_t dropped = (uint64_t)(mag & ((1u << shift) - 1u));\n"
      "  const uint64_t needbits = (uint64_t)((1u << w) - (r & ((1u << w) - 1u)));\n"
      "  return (dropped << w) >= (needbits << shift) ? 1 : 0;\n"
      "}\n"
      "static inline uint16_t weft_h16_bits_rs(float x, uint32_t rbits, int satfinite) {\n"
      "  uint32_t u; std::memcpy(&u, &x, 4);\n"
      "  const uint16_t sign = (uint16_t)((u >> 31) ? 0x8000u : 0u);\n"
      "  const uint32_t mag = u & 0x7FFFFFFFu;\n"
      "  const int e = (int)((mag >> 23) & 0xFFu) - 127;\n"
      // A NaN keeps its own encoding and an infinity has no significand to round;
      // `.satfinite` is what sends a value past the half range to the largest finite
      // half instead of to an infinity.
      "  if ((mag & 0x7F800000u) == 0x7F800000u) {\n"
      "    if (mag > 0x7F800000u) return (uint16_t)(sign | 0x7E00u);\n"
      "    return (uint16_t)(sign | (satfinite ? 0x7BFFu : 0x7C00u));\n"
      "  }\n"
      "  if (e >= 16) return (uint16_t)(sign | (satfinite ? 0x7BFFu : 0x7C00u));\n"
      "  uint16_t t = weft_h16_bits_rz(x);\n"
      "  if (weft_rs_carry(mag, e, 10, -14, rbits, 13)) {\n"
      "    const uint32_t body = (uint32_t)(t & 0x7FFFu) + 1u;\n"
      "    if (body >= 0x7C00u) return (uint16_t)(sign | (satfinite ? 0x7BFFu : 0x7C00u));\n"
      "    t = (uint16_t)(sign | body);\n"
      "  }\n"
      "  return t;\n"
      "}\n"
      "static inline uint16_t weft_bf16_bits_rs(float x, uint32_t rbits, int satfinite) {\n"
      "  uint32_t u; std::memcpy(&u, &x, 4);\n"
      "  const uint16_t sign = (uint16_t)((u >> 31) ? 0x8000u : 0u);\n"
      "  const uint32_t mag = u & 0x7FFFFFFFu;\n"
      "  const int e = (int)((mag >> 23) & 0xFFu) - 127;\n"
      // A bf16 has a float's exponent range, so only a NaN or an infinity reaches
      // here: the NaN keeps its encoding, and `.satfinite` clamps the infinity.
      "  if ((mag & 0x7F800000u) == 0x7F800000u) {\n"
      "    if (mag > 0x7F800000u) return (uint16_t)(sign | 0x7FC0u);\n"
      "    return (uint16_t)(sign | (satfinite ? 0x7F7Fu : 0x7F80u));\n"
      "  }\n"
      "  uint16_t t = (uint16_t)(mag >> 16);\n"
      "  if (weft_rs_carry(mag, e, 7, -126, rbits, 16)) {\n"
      "    const uint32_t body = (uint32_t)t + 1u;\n"
      "    if (body >= 0x7F80u) return (uint16_t)(sign | (satfinite ? 0x7F7Fu : 0x7F80u));\n"
      "    t = (uint16_t)body;\n"
      "  }\n"
      "  return (uint16_t)(sign | t);\n"
      "}\n"
      // These two decodes are defined with the rest of the packed formats, after the
      // encoders that walk a code back one step need them.
      "static inline float weft_mx_decode(uint32_t v, int eb, int mb);\n"
      "static inline float weft_e5m2(uint32_t b, int i);\n"
      // `.rz` on a narrow destination: the nearest code, walked one grid step closer
      // to zero when that answer landed away from it. The codes are monotone in
      // magnitude and the nearest answer is within half a step, so one step is the
      // most this can need, and a finite value past the range has already saturated
      // to MAX_NORM, which is what truncating it gives too, so the walk only ever
      // happens below saturation.
      "static inline uint8_t weft_e4m3_bits_rz(float x) {\n"
      "  const uint8_t c = weft_e4m3_bits(x);\n"
      "  uint32_t m = c & 0x7Fu;\n"
      "  if (m != 0u && std::fabs(weft_e4m3((uint32_t)c, 0)) > std::fabs(x)) m--;\n"
      "  return (uint8_t)((c & 0x80u) | m);\n"
      "}\n"
      "static inline uint8_t weft_e5m2_bits_rz(float x) {\n"
      "  const uint8_t c = weft_e5m2_bits(x);\n"
      "  uint32_t m = c & 0x7Fu;\n"
      "  if (m != 0u && std::fabs(weft_e5m2((uint32_t)c, 0)) > std::fabs(x)) m--;\n"
      "  return (uint8_t)((c & 0x80u) | m);\n"
      "}\n"
      "static inline uint32_t weft_mx_bits_rz(float x, int eb, int mb) {\n"
      "  const uint32_t mask = (1u << (eb + mb)) - 1u;\n"
      "  const uint32_t c = weft_mx_bits(x, eb, mb);\n"
      "  uint32_t m = c & mask;\n"
      "  if (m != 0u && std::fabs(weft_mx_decode(m, eb, mb)) > std::fabs(x)) m--;\n"
      "  return (c & ~mask) | m;\n"
      "}\n"
      "static inline uint32_t weft_e2m1_bits(float x) { return weft_mx_bits(x, 2, 1); }\n"
      "static inline uint32_t weft_e2m3_bits(float x) { return weft_mx_bits(x, 2, 3); }\n"
      "static inline uint32_t weft_e3m2_bits(float x) { return weft_mx_bits(x, 3, 2); }\n"
      "static inline uint32_t weft_e2m1_rz(float x) { return weft_mx_bits_rz(x, 2, 1); }\n"
      "static inline uint32_t weft_e2m3_rz(float x) { return weft_mx_bits_rz(x, 2, 3); }\n"
      "static inline uint32_t weft_e3m2_rz(float x) { return weft_mx_bits_rz(x, 3, 2); }\n"
      // Two narrow values into one word: the first in the high field, four bits up for
      // the nibble-wide fp4 pair and a byte up for the rest.
      "static inline uint32_t weft_pack2(uint32_t hi, uint32_t lo, int bits) {\n"
      "  return lo | (hi << bits);\n"
      "}\n"
      // `.scaled::n{1,2}::ue8m0`: a ue8m0 factor is the power of two 2^(s-127), which a
      // down conversion divides by and an up conversion multiplies by. Writing the
      // down direction as a multiply by 2^(127-s) is the same number, and exact
      // because it is a power of two. s = 0x7f is 1.0, the value the operand defaults
      // to. The ISA flushes a subnormal input to a zero of its own sign before a
      // scaled down conversion.
      "static inline float weft_ue8m0(uint32_t s) {\n"
      "  const uint32_t sb = s == 0u ? 0x00400000u : (s << 23);\n"
      "  float f; std::memcpy(&f, &sb, 4); return f;\n"
      "}\n"
      "static inline float weft_scale_up(float x, uint32_t s) { return x * weft_ue8m0(s); }\n"
      "static inline float weft_scale_down(float x, uint32_t s) {\n"
      "  uint32_t u; std::memcpy(&u, &x, 4);\n"
      "  if ((u & 0x7F800000u) == 0u && (u & 0x7FFFFFu) != 0u) u &= 0x80000000u;\n"
      "  std::memcpy(&x, &u, 4);\n"
      "  const uint32_t sb = s >= 254u ? (s == 254u ? 0x00400000u : 0x00200000u)\n"
      "                                : (254u - s) << 23;\n"
      "  float sc; std::memcpy(&sc, &sb, 4);\n"
      "  return x * sc;\n"
      "}\n"
      // `.rs` on a packed four: the truncated code, one step further from zero when
      // the integer addition of the random bits to the discarded mantissa carries
      // out. Eight random bits per operand, which is what the ISA's figure gives each
      // of the four.
      "static inline uint32_t weft_rs_step(uint32_t code, uint32_t magbits, uint32_t maxmag,\n"
      "                                    float x, int mb, int emin, uint32_t r) {\n"
      "  uint32_t u; std::memcpy(&u, &x, 4);\n"
      "  const uint32_t mag = u & 0x7FFFFFFFu;\n"
      "  if (mag == 0u || (mag & 0x7F800000u) == 0x7F800000u) return code;\n"
      "  if (!weft_rs_carry(mag, (int)((mag >> 23) & 0xFFu) - 127, mb, emin, r, 8)) return code;\n"
      "  const uint32_t mask = (1u << magbits) - 1u;\n"
      "  const uint32_t m = (code & mask) + 1u;\n"
      "  return (code & ~mask) | (m > maxmag ? maxmag : m);\n"
      "}\n"
      // The rbits word splits by operand, not by value: the byte-wide destinations
      // take a and b from the upper half of the word and e and f from the lower one,
      // the first of each pair in the upper byte; the nibble-wide fp4 packs four to a
      // half-word, so the same rule puts b's random bits in the low byte of the upper
      // half and e's in the high byte of the lower one.
      "static inline uint32_t weft_mx4(float a, float b, float e, float f, uint32_t r,\n"
      "                                int eb, int mb, int rs) {\n"
      "  const int nib = (eb + mb) == 3;\n"
      "  const float v[4] = {a, b, e, f};\n"
      "  const int bit[4] = {24, nib ? 8 : 16, nib ? 16 : 8, 0};\n"
      "  const int put[4] = {nib ? 12 : 24, nib ? 8 : 16, nib ? 4 : 8, 0};\n"
      "  const uint32_t magbits = (uint32_t)(eb + mb);\n"
      "  const uint32_t maxmag = (1u << magbits) - 1u;\n"
      "  const int emin = 2 - (1 << (eb - 1));\n"
      "  uint32_t w = 0;\n"
      "  for (int i = 0; i < 4; i++) {\n"
      "    const uint32_t c = rs ? weft_rs_step(weft_mx_bits_rz(v[i], eb, mb), magbits, maxmag,\n"
      "                                         v[i], mb, emin, (r >> bit[i]) & 0xFFu)\n"
      "                          : weft_mx_bits(v[i], eb, mb);\n"
      "    w |= c << put[i];\n"
      "  }\n"
      "  return w;\n"
      "}\n"
      "static inline uint32_t weft_f8x4(float a, float b, float e, float f, uint32_t r,\n"
      "                                 int code, int rs) {\n"
      "  const float v[4] = {a, b, e, f};\n"
      "  uint32_t w = 0;\n"
      "  for (int i = 0; i < 4; i++) {\n"
      "    const uint32_t rb = (r >> (24 - 8 * i)) & 0xFFu;\n"
      "    const uint32_t c = code == 3\n"
      "        ? (rs ? weft_rs_step(weft_e4m3_bits_rz(v[i]), 7u, 0x7Eu, v[i], 3, -6, rb)\n"
      "              : weft_e4m3_bits(v[i]))\n"
      "        : (rs ? weft_rs_step(weft_e5m2_bits_rz(v[i]), 7u, 0x7Bu, v[i], 2, -14, rb)\n"
      "              : weft_e5m2_bits(v[i]));\n"
      "    w |= c << (24 - 8 * i);\n"
      "  }\n"
      "  return w;\n"
      "}\n"
      // The same bytes with an integer accumulator. The sum of 32 products of
      // bytes is exact in s32, and converting to float would stop being the same
      // number past 2^24, so the integer form reads the byte as an integer.
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
      // The packed narrow formats as one code, so an instruction that names two of
      // them is a pair of small tables rather than a case for every combination:
      // 0 e2m1, 1 e2m3, 2 e3m2, 3 e4m3, 4 e5m2, 5 ue8m0, 6 e2m1p (one nibble a byte).
      "static inline float weft_mx_decode(uint32_t v, int eb, int mb) {\n"
      "  const int bias = (1 << (eb - 1)) - 1;\n"
      "  const uint32_t mant = v & ((1u << mb) - 1u);\n"
      "  const uint32_t E = (v >> mb) & ((1u << eb) - 1u);\n"
      // 2^(E-bias-mb) as an exponent field, and E = 0 reads the format's subnormal
      // grid instead of an implicit 1, the same arithmetic the e4m3 decode above is
      // written with, so both engines land on the same float.
      "  const int ex = (E == 0u ? 1 : (int)E) - bias - mb;\n"
      "  const uint32_t sb = (uint32_t)(ex + 127) << 23;\n"
      "  float sc; std::memcpy(&sc, &sb, 4);\n"
      "  const float d = (float)((E == 0u ? 0u : (1u << mb)) + mant) * sc;\n"
      "  return (v >> (eb + mb)) ? -d : d;\n"
      "}\n"
      "static inline float weft_e2m1(uint32_t b, int i) {\n"
      "  return weft_mx_decode((b >> (4 * i)) & 0xFu, 2, 1);\n"
      "}\n"
      "static inline float weft_e2m3(uint32_t b, int i) {\n"
      "  return weft_mx_decode((b >> (8 * i)) & 0x3Fu, 2, 3);\n"
      "}\n"
      "static inline float weft_e3m2(uint32_t b, int i) {\n"
      "  return weft_mx_decode((b >> (8 * i)) & 0x3Fu, 3, 2);\n"
      "}\n"
      // One lane of a packed four, by code: an eight-bit field, a six-bit one at the
      // low end of each byte, a four-bit one at the low end of each half-byte, or one
      // nibble a byte when the format is the padded fp4.
      "static inline float weft_mx_get(uint32_t w, int i, int code) {\n"
      "  if (code == 0) return weft_mx_decode((w >> (4 * i)) & 0xFu, 2, 1);\n"
      "  if (code == 1) return weft_mx_decode((w >> (8 * i)) & 0x3Fu, 2, 3);\n"
      "  if (code == 2) return weft_mx_decode((w >> (8 * i)) & 0x3Fu, 3, 2);\n"
      "  if (code == 3) return weft_e4m3(w, i);\n"
      "  if (code == 4) return weft_e5m2(w, i);\n"
      "  if (code == 6) return weft_mx_decode((w >> (8 * i)) & 0xFu, 2, 1);\n"
      "  const uint32_t v = (w >> (8 * i)) & 0xFFu;\n"
      "  if (v == 0xFFu) return NAN;\n"
      // ue8m0 is a bare exponent, 2^(v-127), and v = 0 is the format's smallest value
      // rather than zero: a subnormal in f32, and exact.
      "  const uint32_t sb = v == 0u ? 0x00400000u : (v << 23);\n"
      "  float f; std::memcpy(&f, &sb, 4); return f;\n"
      "}\n"
      // The two destinations the narrow arithmetic allows. `.satfinite` sends a value
      // past MAX_NORM to MAX_NORM; without it e4m3 answers a NaN, which is what the
      // ISA says and what the saturating encoder alone would get wrong.
      "static inline uint32_t weft_f8_bits(float x, int code, int satfinite) {\n"
      "  if (code == 3) {\n"
      "    if (!satfinite && std::isfinite(x) && std::fabs(x) > 448.0f) return 0x7Fu;\n"
      "    return weft_e4m3_bits(x);\n"
      "  }\n"
      "  return weft_e5m2_bits(x);\n"
      "}\n"
      // add, sub, mul and fma over four packed narrow values. The ISA computes each
      // lane in infinite precision and rounds once to the destination; every one of
      // these formats decodes to an exactly representable float and a product or a
      // sum of two of them is exact in f32, so the one rounding here is the one the
      // instruction asks for.
      "static inline uint32_t weft_mx4_op(uint32_t a, uint32_t b, uint32_t c, int ac, int bc,\n"
      "                                   int cc, int dc, int op, int sat) {\n"
      "  uint32_t out = 0;\n"
      "  for (int i = 0; i < 4; i++) {\n"
      "    const float fa = weft_mx_get(a, i, ac);\n"
      "    float r;\n"
      "    if (op == 0) r = fa + weft_mx_get(c, i, cc);\n"
      "    else if (op == 1) r = fa - weft_mx_get(c, i, cc);\n"
      "    else if (op == 2) r = fa * weft_mx_get(b, i, bc);\n"
      "    else r = fa * weft_mx_get(b, i, bc) + weft_mx_get(c, i, cc);\n"
      "    out |= (uint32_t)(weft_f8_bits(r, dc, sat) & 0xFFu) << (8 * i);\n"
      "  }\n"
      "  return out;\n"
      "}\n"
      // tf32 keeps ten mantissa bits: a float32 with the low thirteen dropped, and
      // a multiply sees the rounded value rather than the input. `rna` carries a
      // tie away from zero, `rn` to even, `rz` throws the thirteen away; `.relu`
      // zeroes a negative first and `.satfinite` clamps an infinite result to the
      // largest finite tf32, NaN included. Bit-level, so a `.b32` register stays
      // one; the same text runs on the GPU target.
      "static inline uint32_t weft_tf32_bits(uint32_t u, int mode, int satfinite, int relu) {\n"
      "  if (relu && (u & 0x80000000u) != 0u && (u & 0x7FFFFFFFu) != 0u) u = 0u;\n"
      "  const uint32_t sign = u & 0x80000000u;\n"
      "  if ((u & 0x7F800000u) == 0x7F800000u) {\n"
      "    if (!satfinite) return u;\n"
      "    return (u & 0x7FFFFFFFu) > 0x7F800000u ? 0u : (sign | 0x7F7FE000u);\n"
      "  }\n"
      "  if (mode == 1) u += 0x1000u;\n"
      "  else if (mode == 0) u += 0x0FFFu + ((u >> 13) & 1u);\n"
      "  u &= 0xFFFFE000u;\n"
      "  if (satfinite && (u & 0x7F800000u) == 0x7F800000u) u = sign | 0x7F7FE000u;\n"
      "  return u;\n"
      "}\n"
      "static inline float weft_tf32_r(float x, int mode) {\n"
      "  uint32_t u; std::memcpy(&u, &x, 4);\n"
      "  u = weft_tf32_bits(u, mode, 0, 0);\n"
      "  float r; std::memcpy(&r, &u, 4); return r;\n"
      "}\n"
      "static inline float weft_tf32(float x) { return weft_tf32_r(x, 0); }\n"
      // shf.l / shf.r: a funnel shift over the 64-bit pair {b:a}, `a` is the
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
      // prmt.b32: four bytes picked out of the eight of (a:b). The generic form
      // (mode -1) reads a nibble for each byte, and a nibble with its top bit set
      // answers with the sign of the byte it names. The six named modes read one
      // 2-bit selector `k` and are the ISA's table: `f4e` walks forward from it,
      // `b4e` backward, `rc8` repeats one byte and `rc16` one half, and the edge
      // clamps hold the low or the high end.
      "static inline uint32_t weft_prmt(uint32_t a, uint32_t b, uint32_t c, int mode) {\n"
      "  const uint64_t src = (uint64_t)a | ((uint64_t)b << 32);\n"
      "  uint32_t out = 0;\n"
      "  if (mode < 0) {\n"
      "    for (int i = 0; i < 4; i++) {\n"
      "      const uint32_t sel = (c >> (4 * i)) & 0xFu;\n"
      "      const uint32_t byte = (uint32_t)((src >> ((sel & 7u) * 8u)) & 0xFFu);\n"
      "      out |= ((sel & 8u) ? ((byte & 0x80u) ? 0xFFu : 0u) : byte) << (8 * i);\n"
      "    }\n"
      "    return out;\n"
      "  }\n"
      "  const uint32_t k = c & 3u;\n"
      "  for (uint32_t j = 0; j < 4u; j++) {\n"
      "    uint32_t idx;\n"
      "    if (mode == 0) idx = (k + j) & 7u;\n"
      "    else if (mode == 1) idx = (k - j) & 7u;\n"
      "    else if (mode == 2) idx = k;\n"
      "    else if (mode == 3) idx = k > j ? k : j;\n"
      "    else if (mode == 4) idx = k < j ? k : j;\n"
      "    else idx = (k & 1u) * 2u + (j & 1u);\n"
      "    out |= (uint32_t)((src >> (idx * 8u)) & 0xFFu) << (8u * j);\n"
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
      // The software double travels with the portable math: the float bridge
      // functions call it, and a kernel that reaches one needs its definitions in
      // this text or the CPU column fails to link while the GPU column runs.
      "\n" + weft_f64_cpp_text() + std::string(kWSoftMath) +
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
  // itself, a run does not have to be told again on its own command line.
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
