#include "translate_ptx.hpp"

#include "weft_msl.hpp"
#include "weft_warp_msl.hpp"
// The portable library, emitted into both of this front end's texts.
#include "soft_math.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace {

bool is_word(char c) { return std::isalnum((unsigned char)c) || c == '_' || c == '$'; }

// The component a `tld4.r|g|b|a` gather asks for. A texture weft binds has one
// channel, so a gather of any other component reads nothing, as a fetch does.
int gather_component(const std::string& opcode) {
  const size_t at = opcode.find("tld4.");
  if (at == std::string::npos || at + 5 >= opcode.size()) return 0;
  switch (opcode[at + 5]) {
    case 'g': return 1;
    case 'b': return 2;
    case 'a': return 3;
    default: return 0;
  }
}

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

// Does `body` name this identifier?
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

// A qualifier can carry an argument that the dot splitter keeps attached.
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
  // The special registers that carry no dot.
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

static std::string fuse_ptx_products(const std::string& body) {
  struct Ins {
    std::string op;
    std::vector<std::string> mods, ops;
  };
  auto parse = [](const std::string& line, Ins& out) {
    std::string s = trim(line);
    if (s.empty()) return false;
    if (s[0] == '@' || s[0] == '/' || s[0] == '.' || s[0] == '{' || s[0] == '}' || s[0] == ';')
      return false;
    if (s.back() == ';') s.pop_back();
    const size_t sp = s.find_first_of(" \t");
    if (sp == std::string::npos) return false;
    const std::vector<std::string> head = split_top(s.substr(0, sp), '.');
    if (head.size() < 2) return false;
    out.op = head[0];
    out.mods.assign(head.begin() + 1, head.end());
    for (const auto& t : split_top(trim(s.substr(sp)), ',')) out.ops.push_back(trim(t));
    return !out.ops.empty();
  };
  auto plain_f32 = [](const Ins& in, const char* op) {
    if (in.op != op) return false;
    for (const auto& m : in.mods)
      if (m != "rn" && m != "f32") return false;
    return true;
  };
  std::vector<std::string> lines;
  for (size_t p = 0; p <= body.size();) {
    const size_t nl = body.find('\n', p);
    if (nl == std::string::npos) {
      lines.push_back(body.substr(p));
      break;
    }
    lines.push_back(body.substr(p, nl - p));
    p = nl + 1;
  }
  std::map<std::string, int> uses;
  for (const auto& line : lines) {
    const std::string t = trim(line);
    if (t.rfind(".reg", 0) == 0) continue;
    for (size_t i = 0; i < t.size(); i++) {
      if (t[i] != '%') continue;
      size_t j = i + 1;
      while (j < t.size() && (std::isalnum((unsigned char)t[j]) || t[j] == '_')) j++;
      if (j > i + 1) uses[t.substr(i + 1, j - i - 1)]++;
      i = j ? j - 1 : i;
    }
  }
  // Every line, parsed once.
  std::vector<Ins> ins(lines.size());
  std::vector<bool> ok(lines.size(), false), pred(lines.size(), false);
  for (size_t k = 0; k < lines.size(); k++) {
    const std::string t = trim(lines[k]);
    pred[k] = !t.empty() && t[0] == '@';
    ok[k] = !pred[k] && parse(lines[k], ins[k]);
    if (!ok[k]) ins[k] = Ins{};
  }
  auto wall = [](const Ins& in) {
    static const char* kOps[] = {"bra", "brx",  "ret",   "call",     "exit", "trap",
                                 "bar", "membar", "fence", "cp",     "wmma", "red",
                                 "atom", "st",   "ldu",   "prefetch", "suld", "sust"};
    if (in.op.empty()) return true;
    for (const char* o : kOps)
      if (in.op == o) return true;
    return false;
  };
  auto writes = [](const Ins& in, const std::string& reg) {
    return !in.ops.empty() && in.ops[0] == reg;
  };
  for (size_t k = 0; k < lines.size(); k++) {
    if (!ok[k] || pred[k] || ins[k].op != "add" || !plain_f32(ins[k], "add")) continue;
    if (ins[k].ops.size() != 3) continue;
    bool fused = false;
    for (int w = 1; w <= 2 && !fused; w++) {
      const std::string d = ins[k].ops[w];
      if (d.empty() || d[0] != '%') continue;
      if (uses[d.substr(1)] != 2) continue;  // read here and nowhere else
      for (size_t j = k; j-- > 0;) {
        if (!ok[j] || pred[j] || wall(ins[j])) break;
        if (ins[j].ops.empty() || ins[j].ops[0] != d) continue;
        if (!plain_f32(ins[j], "mul") || ins[j].ops.size() != 3) break;
        bool safe = true;
        for (size_t q = j + 1; q < k && safe; q++)
          safe = ok[q] && !pred[q] && !wall(ins[q]) &&
                 !writes(ins[q], ins[j].ops[1]) && !writes(ins[q], ins[j].ops[2]);
        if (!safe) break;
        const std::string other = ins[k].ops[w == 1 ? 2 : 1];
        size_t lead = lines[k].find_first_not_of(" \t");
        if (lead == std::string::npos) lead = 0;
        lines[k] = lines[k].substr(0, lead) + "fma.rn.f32 " + ins[k].ops[0] + ", " +
                   ins[j].ops[1] + ", " + ins[j].ops[2] + ", " + other + ";";
        lines[j] = "";  // nothing else reads the product, so one line replaces both
        fused = true;
        break;
      }
    }
  }
  std::string out;
  for (const auto& line : lines) out += line + "\n";
  return out;
}

struct Emitter {
  std::string err;
  bool uses_barrier = false;
  bool uses_shfl = false;  // counted separately: a shuffle does not tie warps together

  // One decode, two back ends.
  bool msl = false;
  bool wide_atom = false;
  std::map<std::string, int> reg_reads;
  std::map<std::string, int> tex_index;
  int ntex = 0;
  std::vector<bool> tex_write;
  // The bindings an `.a2d` instruction names.
  std::vector<bool> tex_array;
  std::vector<bool> tex_3d;
  std::vector<bool> tex_cube;
  int shared_bytes = 0;      // the threadgroup arena this kernel needs, at compile time
  bool shared_extern = false;  // it was `extern .shared`: the launch chooses the size
  bool uses_red = false;     // a reduction barrier: needs its publish slots

  bool vec = false;
  bool vec_ok = true;      // false once something here cannot be vectorised
  std::string vec_pred;    // per-lane predicate for the current instruction
  bool vec_control = false;  // emitted text is control flow: do not wrap it
  bool vec_side = false;     // emitted text already contains its lane loop
  bool vec_select = true;    // store() masks by lane; off inside a lane loop
  // The condition code.
  std::vector<std::string> cc_members;
  std::string cc_name;
  int cc_next = 0;
  std::string cc_member() {
    if (cc_name.empty()) {
      cc_name = "weft_cc" + std::to_string(cc_next++);
      cc_members.push_back("uint32_t " + cc_name + ";");
    }
    return "s->" + cc_name;
  }
  int label_count = 1;                 // 0 is the entry
  std::map<std::string, int> label_ids;
  std::map<std::string, std::vector<std::string>> brxtab;

  int label_id(const std::string& name) {
    auto it = label_ids.find(name);
    if (it != label_ids.end()) return it->second;
    const int id = label_count++;
    label_ids[name] = id;
    return id;
  }
  struct VecVal {
    bool ok = false;
    std::string base;  // C++ text with no lane reference in it
    long stride = 0;   // coefficient of the lane index
  };
  std::map<std::string, VecVal> vec_vals;
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

  std::map<std::string, std::vector<std::string>> callees;
  std::map<std::string, std::string> callee_errs;
  std::map<std::string, std::string> callee_args;
  std::vector<std::pair<std::string, std::string>> callee_members;
  std::map<std::string, std::vector<std::string>> callee_rets;
  std::vector<std::string> callee_ret;
  std::map<std::string, bool> callee_suspends;
  std::map<std::string, bool> callee_deep;
  std::map<std::string, bool> entry_names;
  std::string self_name;
  bool self_tail = false;
  bool self_deep = false;
  std::vector<std::vector<std::string>> self_sites;
  std::vector<std::vector<std::string>> self_site_dsts;
  std::vector<std::string> self_rets;
  std::vector<std::string> self_params;
  std::map<std::string, int> callee_cycle;
  std::map<std::string, int> callee_cycle_at;
  std::map<std::string, std::vector<std::string>> callee_cycle_params;
  int self_cycle = -1;
  std::string cycle_sel;
  bool wctx = false;

  std::string word_read(const std::string& ty, size_t k) const {
    const std::string idx = "wa[" + std::to_string(k) + "]";
    if (ty == "float") return "weft_bitcast<float>((uint32_t)" + idx + ")";
    if (ty == "double") return "weft_bitcast<double>(" + idx + ")";
    return "((" + ty + ")" + idx + ")";
  }

  std::string to_word(const std::string& tok, const std::string& ty) {
    const std::string e = operand_as(tok, ty);
    if (ty == "float") return "(uint64_t)weft_bitcast<uint32_t>(" + e + ")";
    if (ty == "double") return "weft_bitcast<uint64_t>(" + e + ")";
    return "(uint64_t)(" + e + ")";
  }

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

  // Feed one instruction's dataflow through the model.
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
  int next_resume = 65536;
  std::string resume_cases;   // "case 1: goto R1;"
  std::string resume_labels;  // "R1:;" spliced in after each barrier
  const std::string* shared_size = nullptr;
  // Each `.shared` array's byte offset inside the threadgroup's shared window.
  std::map<std::string, uint64_t> shared_offsets;
  std::map<std::string, int> params;   // buffer parameters
  std::map<std::string, int> scalars;  // plain value parameters
  std::vector<std::string> unused;     // parameters the body never names
  std::map<int, uint32_t> param_bytes;
  std::map<std::string, int> ptr_param;
  std::map<std::string, std::string> reg_types;  // "r" -> "uint32_t"
  std::map<std::string, int> reg_counts;
  std::string decls;

  std::string operand_as(const std::string& tok, const std::string& want) {
    std::string t = trim(tok);
    if (t.empty()) return t;
    if (t[0] != '%') {
      if (t.size() == 10 && t.compare(0, 2, "0f") == 0) {
        if (want != "float") return "(uint32_t)0x" + t.substr(2) + "u";
        return "weft_f32(0x" + t.substr(2) + "u)";
      }
      if (t.size() == 18 && t.compare(0, 2, "0d") == 0) {
        return "weft_f64(0x" + t.substr(2) + "ull)";
      }
      if (!t.empty() && (std::isalpha((unsigned char)t[0]) || t[0] == '_')) {
        // A bare symbol.
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
    for (size_t i = 0; i < callee_ret.size(); i++)
      if (trim(dst_tok) == callee_ret[i])
        return "s->weft_ret" + std::to_string(i) + " = " + expr + ";";
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
    if (r.ok) {
      auto wide = reg_types.find(r.base);
      if (wide != reg_types.end() && cpp_size(wide->second) > 8)
        return "if (am[l]) " + dst + " = " + core + ";";
    }
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
      {
        const std::string lane = vec ? "l" : std::string("(ctx.lid & 31u)");
        if (r.special == "lanemask_eq") return "(1u << " + lane + ")";
        if (r.special == "lanemask_lt") return "((1u << " + lane + ") - 1u)";
        if (r.special == "lanemask_le") return "((2u << " + lane + ") - 1u)";
        if (r.special == "lanemask_gt") return "~((2u << " + lane + ") - 1u)";
        if (r.special == "lanemask_ge") return "~((1u << " + lane + ") - 1u)";
      }
      if (r.special == "warpid") return vec ? "0u" : "(ctx.lid / 32u)";
      if (r.special == "nwarpid") return "((ctx.tg_size + 31u) / 32u)";
      err = "unsupported special register %" + r.special;
      return "0";
    }
    auto rc = reg_counts.find(r.base);
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
      const size_t cut = s.find_first_of("+-");
      param_name = cut == std::string::npos ? s : trim(s.substr(0, cut));
      std::string off;
      if (cut != std::string::npos) {
        off = trim(s.substr(cut + 1));
        if (!off.empty() && off[0] == '%') off = reg_expr(off);
        if (s[cut] == '-') off = "(0 - " + off + ")";
      }
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

  void track_ptr(const std::string& op, const std::vector<std::string>& parts,
                 const std::vector<std::string>& args) {
    if (args.size() < 2) return;
    Reg d = parse_reg(args[0]);
    if (!d.ok || !d.special.empty()) return;
    if (cpp_size(reg_type_of(args[0])) != 8) return;
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
  std::string math(const std::string& name, const std::string& args) const {
    return msl ? (name + "(" + args + ")") : ("std::" + name + "(" + args + ")");
  }
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
  // A rounding function applied to a value on its way to an integer.
  std::string rounded_math(const std::string& fn, const std::string& src) const {
    if (msl) return (fn == "nearbyint" ? std::string("rint") : fn) + "(" + src + ")";
    return "std::" + fn + "((double)" + src + ")";
  }
  std::string isnan_of(const std::string& x) const {
    return msl ? ("isnan(" + x + ")") : ("std::isnan((double)" + x + ")");
  }

  std::string emit(const std::string& raw);
  std::string translate_body(const std::string& body);
};

// The warp back end for the GPU.

// The warp body as a Metal kernel body.
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
        seg += "{ for (uint32_t l = 0; l < weft::kWarpLanes; l++) if (am[l]) s->pc[l] = " +
               target.substr(1) + "u; break; }";
      } else {
        seg += s.substr(g, semi - g + 1);
      }
      at = semi + 1;
    }
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
    if (p.rfind("shared", 0) == 0) shared = true;
    if (p == "param") param = true;
    if (p == "global") global = true;
    if (p == "const") is_const = true;
  }

  auto need = [&](size_t n) {
    if (args.size() < n) err = "instruction '" + opcode + "' needs " + std::to_string(n) + " operands";
    return args.size() >= n;
  };

  vec_update(opcode, args);
  track_ptr(op, parts, args);

  if (op == "ld" || op == "st") {
    if (!need(2)) return "";
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
    int lanes = 1;
    for (const auto& p : parts) {
      if (p == "v2") lanes = 2;
      else if (p == "v4") lanes = 4;
    }
    const int esz = cpp_size(ty) ? cpp_size(ty) : 4;
    const std::string ty_mem = (msl && ty == "double") ? "uint64_t" : ty;

    auto at = [&](const std::string& a, int k) {
      return k == 0 ? a : "(" + a + " + " + std::to_string(k * esz) + ")";
    };

    if (op == "ld") {
      std::string addr, name, base;
      if (!mem_operand(args[1], addr, name, &base)) return "";
      if (!param) note_width(name, base, ty);
      auto load_at = [&](const std::string& a) -> std::string {
        if (param) {
          auto ca = callee_args.find(name);
          if (ca != callee_args.end()) return "(" + ty + ")" + ca->second;
          auto b = params.find(name);
          auto sc = scalars.find(name);
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
      for (size_t i = 0; i < callee_ret.size(); i++) {
        if (name != callee_ret[i]) continue;
        const std::string vty = (msl && ty == "double") ? "uint64_t" : ty;
        return guard + "s->weft_ret" + std::to_string(i) + " = " + operand_as(args[1], vty) +
               ";";
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
    auto braced = [](const std::string& s) { return s.find('{') != std::string::npos; };
    if (braced(args[0])) {
      std::string list = trim(args[0]);
      list = list.substr(list.find('{') + 1);
      list = list.substr(0, list.find('}'));
      auto parts2 = split_top(list, ',');
      if (parts2.size() != 2) { err = "mov with a brace list needs two registers"; return ""; }
      const bool wide = reg_type_of(args[1]) == "uint64_t";
      const std::string mask = wide ? "0xFFFFFFFFu" : "0xFFFFu";
      const std::string sh = wide ? "32" : "16";
      const std::string lo = store(parts2[0], "(uint32_t)(" + reg_expr(args[1]) + " & " + mask + ")", "uint32_t");
      const std::string hi = store(parts2[1], "(uint32_t)((" + reg_expr(args[1]) + ") >> " + sh + ")", "uint32_t");
      return guard.empty() ? lo + " " + hi : guard + "{ " + lo + " " + hi + " }";
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
      const bool pzo = has_part(parts, "pzo");
      auto pz = [pzo](const std::string& v) {
        return pzo ? "(((" + v + ") == 0.0f) ? 0.0f : (" + v + "))" : v;
      };
      const bool scaled = has_qual(parts, "scaled");
      const std::string src = operand_as(args[1], "uint32_t");
      if (hpos < ppos) {
        // A packed source read out into two halves.
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
  if (op == "cvt" && (has_part(parts, "e2m1x4") || has_part(parts, "e2m3x4") ||
                      has_part(parts, "e3m2x4") || has_part(parts, "e4m3x4") ||
                      has_part(parts, "e5m2x4"))) {
    if (args.size() < 2) { err = "'" + opcode + "' needs four sources"; return ""; }
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

  if (op == "cvt" && (has_part(parts, "f16x2") || has_part(parts, "bf16x2") ||
                      has_part(parts, "e4m3x2") || has_part(parts, "e5m2x2") ||
                      has_part(parts, "e2m1x2") || has_part(parts, "e2m3x2") ||
                      has_part(parts, "e3m2x2"))) {
    if (!need(3)) return "";
    const bool rz = has_part(parts, "rz");
    const bool pzo = has_part(parts, "pzo");
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
    const std::string hi = std::string(one) + a + ")";
    const std::string lo = std::string(one) + b + ")";
    return store(args[0],
                 "weft_pack2(" + hi + ", " + lo + ", " + std::to_string(bits) + ")", wid);
  }

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

  if (op == "cvt") {
    if (!need(2)) return "";
    std::string dstty, mode;
    bool integer_round = false;
    // A 16-bit float destination.
    std::string narrow;
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
        if (!dstty.empty()) {
          ty = "uint16_t";
          src_half = parts[i] == "f16" ? "weft_h16(" : "weft_bf16(";
          continue;
        }
        narrow = parts[i] == "f16" ? "half" : "bf16";
        dstty = "uint16_t";
      } else if (parts[i] == "rna") {
        // Three letters, but not one of the `.rni` family.
        mode = "rna";
      } else if ((parts[i] == "rn" || parts[i] == "rz" || parts[i] == "rm" ||
                  parts[i] == "rp" || parts[i] == "rni" || parts[i] == "rzi" ||
                  parts[i] == "rmi" || parts[i] == "rpi")) {
        mode = parts[i].substr(0, 2);
        integer_round = parts[i].size() == 3;
      } else if (parts[i] == "ftz" || parts[i] == "sat" || parts[i] == "tf32" ||
                 ((parts[i] == "satfinite" || parts[i] == "relu") &&
                  (has_part(parts, "tf32") || has_part(parts, "f16") ||
                   has_part(parts, "bf16")))) {
      } else {
        err = "unsupported conversion modifier in '" + opcode + "'";
        return "";
      }
    }
    if (dstty.empty()) { err = "unsupported conversion '" + opcode + "'"; return ""; }
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
    if (!narrow.empty()) {
      const bool to_zero = mode == "rz";
      const bool satf = has_part(parts, "satfinite");
      const bool relu = has_part(parts, "relu");
      std::string v;
      if (!src_half.empty()) v = src_half + operand_as(args[1], "uint16_t") + ", 0)";
      else if (has_part(parts, "f64"))
        v = msl ? "weft_bitcast<float>(weft_fp64::to_f32(" + operand_as(args[1], "uint64_t") + "))"
                : "(float)" + operand_as(args[1], "double");
      else v = operand_as(args[1], "float");
      if (relu) v = "((" + v + " < 0.0f) ? 0.0f : (" + v + "))";
      if (has_part(parts, "sat")) {
        v = "((" + v + " != " + v + ") ? 0.0f : ((" + v + " < 0.0f) ? 0.0f : ((" + v +
            " > 1.0f) ? 1.0f : (" + v + "))))";
      }
      if (satf) {
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
    if (!src_half.empty() && dstty == "float")
      return guard + store(args[0],
                           src_half + operand_as(args[1], "uint16_t") + ", 0)", "float");

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
      if (dstty == "float") {
        // `.rni` to a float is a round to an integer that stays a float.
        std::string fv = b;
        if (integer_round && mode == "rn") fv = "weft_fp64::rne_bits(" + fv + ")";
        return guard + store(args[0], "weft_bitcast<float>(weft_fp64::to_f32(" + fv + "))",
                             "float");
      }
      std::string v = b;
      if (!mode.empty() && mode != "rz")
        v = std::string("weft_fp64::") +
            (mode == "rn" ? "rne_bits(" : mode == "rm" ? "floor_bits(" : "ceil_bits(") + v + ")";
      const bool sgn = dstty == "int32_t" || dstty == "int64_t";
      const bool narrow = dstty == "int32_t" || dstty == "uint32_t";
      const std::string c = std::string("weft_fp64::to_") + (sgn ? "i" : "u") +
                            (narrow ? "32(" : "64(") + v + ")";
      return guard + store(args[0], "(" + dstty + ")" + c, dstty);
    }
    // The source is coerced to the *instruction's* type, not the register's.
    std::string src = operand_as(args[1], ty);
    if (!src_half.empty()) src = src_half + src + ", 0)";
    const bool dst_f = dstty == "float" || dstty == "double";
    const bool src_f = ty == "float" || ty == "double" || !src_half.empty();
    auto f2i = [&](const std::string& v) -> std::string {
      if (dstty == "int32_t") return "weft_f2i32(" + v + ")";
      if (dstty == "uint32_t") return "weft_f2u32(" + v + ")";
      if (dstty == "int64_t") return "weft_f2i64(" + v + ")";
      if (dstty == "uint64_t") return "weft_f2u64(" + v + ")";
      return "(" + dstty + ")(" + v + ")";
    };
    // Rounding only exists between floats, and between a float and an integer.
    std::string e;
    // Rounding is a step between two floats.
    if (!mode.empty() && dst_f && src_f) {
      if (integer_round) {
        const char* fn = mode == "rn" ? "nearbyint"
                       : mode == "rz" ? "trunc"
                       : mode == "rm" ? "floor"
                                       : "ceil";
        e = "(" + dstty + ")(" + rounded_math(fn, src) + ")";
      } else if (mode == "rn") {
        e = "(" + dstty + ")(" + src + ")";
      } else {
        const char* fn = mode == "rz" ? "trunc" : (mode == "rm" ? "floor" : "ceil");
        e = "(" + dstty + ")(" + rounded_math(fn, src) + ")";
      }
    } else if (!mode.empty() && src_f) {
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
    if (has_part(parts, "f16x2") || has_part(parts, "bf16x2")) {
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
      const std::string cl = store(dl, "(" + la + " " + sym + " " + lb + ")", "bool");
      const std::string ch = store(dh, "(" + ha + " " + sym + " " + hb + ")", "bool");
      return guard.empty() ? cl + " " + ch : guard + "{ " + cl + " " + ch + " }";
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

  if (op == "slct") {
    if (!need(4)) return "";
    const bool isf = has_part(parts, "f32") || has_part(parts, "f64");
    const std::string cty = isf ? (has_part(parts, "f64") ? "double" : "float")
                                : (has_part(parts, "u32") ? "uint32_t" : "int32_t");
    return guard + store(args[0],
                         "((" + operand_as(args[3], cty) + ") >= " + (isf ? "0.0" : "0") +
                             " ? " + operand_as(args[1], ty) + " : " +
                             operand_as(args[2], ty) + ")",
                         ty);
  }

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
      if (msl)
        return guard + "{ s->pc = " + std::to_string(label_id(trim(args[0]))) +
               "u; continue; }";
      return guard + "goto " + sanitize(args[0]) + ";";
    }

    const int tgt = label_id(args[0]);
    if (vec_pred.empty()) return "goto L" + std::to_string(tgt) + ";";

    // A predicated branch splits the warp.
    const int fall = label_count++;
    std::string out = "{ uint32_t tk = 0, sy = 0;\n";
    out += "    for (uint32_t l = 0; l < 32u; l++) { if (!am[l]) continue; if (" + vec_pred +
           ") tk |= 1u << l; else sy |= 1u << l; }\n";
    out += "    if (!sy) goto L" + std::to_string(tgt) + ";\n";
    out += "    {\n";
    out += "      for (uint32_t l = 0; l < 32u; l++) if (am[l])\n";
    out += "        s->pc[l] = ((tk >> l) & 1u) ? " + std::to_string(tgt) + "u : " +
           std::to_string(fall) + "u;\n";
    out += "      goto dispatch;\n    } }\n  L" + std::to_string(fall) + ":;";
    return out;
  }

  if (op == "bar" || op == "membar" || op == "fence") {
    if (op != "bar") return guard + ";";
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
      return guard +
             "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);";
    }
    const int id = next_resume++;
    resume_cases += "    case " + std::to_string(id) + ": goto R" + std::to_string(id) + ";\n";
    // A barrier is "save where I am and give the worker back".
    std::string label = "R" + std::to_string(id) + ":;";
    if (!guard.empty())
      return guard + "{ s->pc = " + std::to_string(id) + "u; return; }\n  " + label;
    return "s->pc = " + std::to_string(id) + "u; return;\n  " + label;
  }

  // mbarrier: an arrival-counted barrier with a phase, in threadgroup memory.
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
    const std::string ct = msl ? "threadgroup " : "";
    const std::string cv = msl ? "volatile " : "";
    const std::string w0 = "*(( " + cv + ct + "uint32_t*)(sh + " + addr + "))";
    const std::string w1 = "*(( " + cv + ct + "uint32_t*)(sh + " + addr + " + 4u))";
    const std::string phase = msl
        ? ("*((volatile " + ct + "const uint32_t*)(sh + " + addr + " + 4u))")
        : w1;
    const std::string lane = "for (uint32_t l = 0; l < 32u; l++) if (" + lane_cond() + ")";

    if (mode == "init") {
      if ((int)args.size() <= ai + 1) { err = "an mbarrier init without a count"; return ""; }
      const std::string n = "(uint32_t)(" + operand_as(args[ai + 1], "uint32_t") + ")";
      if (vec) {
        vec_side = true;
        return "{ " + lane + " { const uint32_t weft_n = " + n + "; " + w0 + " = weft_n; " +
               w1 + " = (weft_n << 1); } }";
      }
      return guard + "{ const uint32_t weft_n = " + n + "; " + w0 + " = weft_n; " + w1 +
             " = (weft_n << 1); }";
    }
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
      // A test never waits.
      if (mode == "test_wait" || msl || vec) return guard + store(args[0], done, "bool");
      const int id = next_resume++;
      resume_cases += "    case " + std::to_string(id) + ": goto R" + std::to_string(id) + ";\n";
      const std::string sid = std::to_string(id);
      std::string cond;
      if (guard.size() > 6 && guard.compare(0, 4, "if (") == 0)
        cond = guard.substr(4, guard.size() - 6);
      return "R" + sid + ":;\n  if (!(" + done + ")" +
             (cond.empty() ? std::string() : " && (" + cond + ")") + ") { s->pc = " + sid +
             "u; return; }\n  " + guard + store(args[0], done, "bool");
    }
    return guard + ";";
  }

  if (op == "ret" || op == "exit") {
    if (!cycle_sel.empty())
      return guard + (guard.empty()
                          ? std::string("*" + cycle_sel + " = 0xFFFFFFFFu; return;")
                          : ("{ *" + cycle_sel + " = 0xFFFFFFFFu; return; }"));
    if (vec) {
      vec_control = true;
      return "{ for (uint32_t l = 0; l < 32u; l++) if (am[l]) s->pc[l] = 0xFFFFFFFFu; "
             "goto dispatch; }";
    }
    return guard + (guard.empty() ? std::string("s->pc = 0xFFFFFFFFu; return;")
                                  : std::string("{ s->pc = 0xFFFFFFFFu; return; }"));
  }

  if (op == "shfl") {
    if (args.size() < 3) { err = "unsupported shuffle '" + opcode + "'"; return ""; }
    std::string mode;
    for (size_t i = 1; i < parts.size(); i++)
      if (parts[i] == "idx" || parts[i] == "up" || parts[i] == "down" || parts[i] == "bfly")
        mode = parts[i];
    if (mode.empty()) { err = "unsupported shuffle mode in '" + opcode + "'"; return ""; }

    uses_shfl = true;
    const std::string addr_q = msl ? "thread " : "";
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
    const bool full_clamp =
        c_lit && segmask == 0 &&
        ((mode == "up" && cval == 0) || ((mode == "down" || mode == "idx") && cval == 31));

    if (vec) {
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

      const VecVal dv = vec_sym(args[2]);
      const bool uniform_delta = dv.ok && dv.stride == 0;
      if (uniform_delta && full_clamp && vec_pred.empty() &&
          (mode == "down" || mode == "up" || mode == "idx")) {
        std::string dtext;
        Reg dr = parse_reg(trim(args[2]));
        if (dr.ok && dr.special.empty()) {
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
        // A full warp writes the shifted values straight back.
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
      return guard + store(args[0],
                           "simd_shuffle(" + operand_as(args[1], "uint32_t") +
                               ", (ushort)(" + src + "))",
                           "uint32_t");
    }

    const std::string publish =
        "((uint32_t*)weft::shfl_pub_buf(ctx))[ctx.lid] = " + operand_as(args[1], "uint32_t") +
        "; s->pc = " + std::to_string(id) + "u; return;\n  ";
    const std::string read = "weft::shfl_read_buf(ctx)[" + base + " + (" + src + ")]";
    const std::string refine = store(args[0], read, "uint32_t");
    if (!guard.empty()) return guard + "{ " + publish + "}\n  " + label + "\n  " + guard + "{ " + refine + " }";
    return "{ " + publish + "}\n  " + label + "\n  " + refine;
  }

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
      const std::string p = operand_as(args[1], "bool");
      std::string value;
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
    const std::string v = operand_as(args[1], ty);
    std::string e;
    if (cls == "finite") e = math("isfinite", v);
    else if (cls == "infinite") e = math("isinf", v);
    else if (cls == "number") e = "!" + math("isnan", v);
    else if (cls == "normal")
      e = msl ? ("(((" + v + ") == 0.0f) || (isfinite(" + v + ") && fabs(" + v + ") >= 0x1p-126f))")
              : ("(std::fpclassify(" + v + ") == FP_NORMAL || (" + v + ") == 0)");
    else if (cls == "subnormal")
      e = msl ? ("((" + v + " != 0.0f) && (fabs(" + v + ") < 0x1p-126f))")
              : ("(std::fpclassify(" + v + ") == FP_SUBNORMAL)");
    // The ISA spells it `.notanumber`; `.nan` is what some producers write.
    else if (cls == "nan" || cls == "notanumber") e = math("isnan", v);
    if (e.empty()) { err = "unsupported class test '" + opcode + "'"; return ""; }
    return guard + store(args[0], e, "bool");
  }

  // prmt.b32{.mode} d, a, b, c: four bytes picked out of the eight of (a:b).
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

  if ((op == "add" || op == "sub" || op == "mul" || op == "fma" || op == "mad" ||
       op == "min" || op == "max" || op == "neg" || op == "abs" || op == "div" ||
       op == "rcp" || op == "sqrt" || op == "ex2" || op == "lg2" || op == "rsqrt" ||
       op == "sin" || op == "cos" || op == "tanh") &&
      (has_part(parts, "f16") || has_part(parts, "f16x2") || has_part(parts, "bf16") ||
       has_part(parts, "bf16x2"))) {
    for (const auto& p : parts) {
      if (p == op || p == "f16" || p == "f16x2" || p == "bf16" || p == "bf16x2") continue;
      if (p == "rn") continue;
      if (p == "sat" && (op == "add" || op == "sub" || op == "mul")) continue;
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
    // The type and the packing only pick which family answers.
    const std::string pre = packed ? (bf ? "weft_bf16x2_" : "weft_f16x2_")
                                   : (bf ? "weft_bf16_" : "weft_f16_");
    if (op == "neg" || op == "abs") {
      if (!need(2)) return "";
      return guard + store(args[0], pre + op + "(" + operand_as(args[1], wt) + ")", wt);
    }
    if (op == "min" || op == "max") {
      if (!need(3)) return "";
      const std::string fn =
          pre + op +
          (has_part(parts, "xorsign") ? "_xorabs" : (has_part(parts, "NaN") ? "_nan" : ""));
      return guard + store(args[0],
                           fn + "(" + operand_as(args[1], wt) + ", " +
                               operand_as(args[2], wt) + ")",
                           wt);
    }
    if (op == "ex2" || op == "lg2" || op == "rsqrt" || op == "sin" || op == "cos" ||
        op == "tanh") {
      if (!need(2)) return "";
      return guard + store(args[0], pre + op + "(" + operand_as(args[1], wt) + ")", wt);
    }
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
  {
    const bool writes_cc = has_part(parts, "cc");
    const bool has_hi = has_part(parts, "hi");
    const bool is_carry_op = (op == "addc" || op == "subc" || op == "madc");
    const bool mad_op = (op == "mad" || op == "madc");
    const bool sub_op = (op == "sub" || op == "subc");
    const bool reads_cc = is_carry_op;
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
      if (vec && (reads_cc || writes_cc)) vec_ok = false;
      const std::string ut = cpp_size(ty) == 8 ? "uint64_t" : "uint32_t";
      const bool sgn = (ty == "int32_t" || ty == "int64_t");
      const std::string f = (reads_cc || writes_cc) ? cc_member() : std::string();
      const std::string a = operand_as(args[1], ut);
      const std::string b = operand_as(args[2], ut);
      const std::string q = mad_op ? operand_as(args[3], ut) : b;
      // The word of the product.
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
    if (op == "shl" || op == "shr") {
      const std::string a = operand_as(args[1], ty);
      const std::string sh = operand_as(args[2], ty);
      const int bits = cpp_size(ty) * 8;
      const std::string lit = trim(args[2]);
      char* end = nullptr;
      const long n = std::strtol(lit.c_str(), &end, 0);
      const bool constant = !lit.empty() && end && *end == '\0';
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
    if (op == "fma" && ty == "float")
      return guard + store(args[0],
                           "weft_fma(" + operand_as(args[1], ty) + ", " +
                               operand_as(args[2], ty) + ", " + operand_as(args[3], ty) + ")",
                           ty);
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
    const std::string canon = ty == "double" ? "weft_bitcast<double>(0x7FF8000000000000ull)"
                                             : "weft_bitcast<float>(0x7FFFFFFFu)";
    const std::string isnanA = "(" + math("isnan", A) + ")", isnanB = "(" + math("isnan", B) + ")";
    const std::string gt = (op == "min") ? " < " : " > ";
    const std::string lt = (op == "min") ? " > " : " < ";
    const std::string tie = "(" + math("signbit", A) + " ? " + (op == "min" ? A : B) + " : " +
                            (op == "min" ? B : A) + ")";
    const std::string pick =
        (std::string("(") + isnanA + " ? (" + isnanB + " ? " + canon + " : " + B + ") : (" +
         isnanB + " ? " + A + " : ((" + A + gt + B + ") ? " + A + " : ((" + B + lt + A + ") ? " + B +
         " : " + tie + "))))");
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
    if (msl && ty == "double")
      return guard + store(args[0],
                           "weft_fp64::sqrt_bits(" + operand_as(args[1], "uint64_t") + ")",
                           "uint64_t");
    return guard + store(args[0], math("sqrt", operand_as(args[1], ty)), ty);
  }

  if (op == "not") {
    if (!need(2)) return "";
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
    std::string sum = "((int64_t)(" + operand_as(args[3], "uint32_t") + "))";
    for (int i = 0; i < 2; i++)
      sum += " + " + lane16(a, i, sgn_a) + " * " + byte8(b, base + i, sgn_b);
    return guard + store(args[0], "(uint32_t)(" + sum + ")", ty.empty() ? "int32_t" : ty);
  }

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

  // The video instructions are packed arithmetic that does not carry between lanes.
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

  if (op == "bswap" && args.size() >= 2) {
    const bool wide = (ty == "uint64_t" || ty == "int64_t");
    return guard + store(args[0],
                         std::string(wide ? "weft_bswap64(" : "weft_bswap32(") +
                             operand_as(args[1], wide ? "uint64_t" : "uint32_t") + ")",
                         ty.empty() ? (wide ? "uint64_t" : "uint32_t") : ty);
  }

  if (op == "cnot" && args.size() >= 2)
    return guard + store(args[0], "(" + operand_as(args[1], ty) + " == 0 ? 1 : 0)",
                         ty.empty() ? "uint32_t" : ty);

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
    if (op == "ex2") e = "weft_exp2(" + a + ")";
    else if (op == "lg2") e = "weft_log2(" + a + ")";
    else if (op == "rcp") e = "(1.0f / (" + a + "))";
    else if (op == "rsqrt") e = "weft_rsqrt(" + a + ")";
    else if (op == "sqrt") e = "weft_sqrt(" + a + ")";
    else e = a + " / " + (args.size() > 2 ? operand_as(args[2], ty) : "1.0f");
    return guard + store(args[0], e, ty);
  }
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
    const std::string mask = trim(args[2]);
    const bool full_mask = mask == "0xffffffff" || mask == "0xffffffffu" ||
                           mask == "0xFFFFFFFF" || mask == "0xFFFFFFFFu";
    std::string active = "am[l]";
    if (!full_mask)
      active += " && ((((uint32_t)" + operand_as(args[2], "uint32_t") + ") >> l) & 1u)";
    const std::string dst = reg_expr(args[0]), src = reg_expr(args[1]);
    vec_side = true;
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
    if (!sym.empty() && shared_offsets.find(sym) == shared_offsets.end()) {
      err = "'" + sym + "' is not a shared array: ldmatrix reads threadgroup memory";
      return "";
    }
    const bool uniform = !sym.empty();
    auto row_addr = [&](const std::string& who) -> std::string {
      if (uniform) return "((uint64_t)(" + addr + ") + 16u * (" + who + "))";
      std::string s = addr;
      for (size_t p = 0; (p = s.find("[l]", p)) != std::string::npos;)
        s.replace(p, 3, "[" + who + "]");
      return s;
    };
    if (packed) {
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
              const std::string peer = "(" + std::to_string(rows * m) + "u + " + row + ")";
              std::string s2 = addr;
              for (size_t p2 = 0; (p2 = s2.find("[l]", p2)) != std::string::npos;)
                s2.replace(p2, 3, "[" + peer + "]");
              chunk = "(" + s2 + ")";
            }
            std::string v;
            if (bits == 4) {
              v = "((uint32_t)*(" + addr_space("threadgroup", true) +
                  "uint8_t*)(sh + " + chunk + " + (" + col + " >> 1)) >> (4u * (" + col +
                  " & 1u))) & 0xFu";
              if (sgn)
                v = "((uint32_t)(((int32_t)((" + v + ") << 28)) >> 28) & 0xFFu)";
            } else {
              const std::string at = "sh + " + chunk + " + ((3u * " + col + ") >> 2)";
              v = "(((((uint32_t)*(" + addr_space("threadgroup", true) + "uint8_t*)(" + at +
                  ")) | (((uint32_t)*(" + addr_space("threadgroup", true) +
                  "uint8_t*)(" + at + " + 1u)) << 8)) >> ((6u * " + col +
                  ") & 7u)) & 0x3Fu)";
            }
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
    return "for (uint32_t l = 0; l < 32u; l++) { " + out + " }";
  }

  // activemask.b32 d;, one bit per lane of this warp that is executing.
  if (op == "activemask") {
    if (!vec) {
      warp_only = true;
      err = "activemask answers which lanes of the warp are executing, which only the "
            "warp body knows";
      return "";
    }
    if (!need(1)) return "";
    const std::string p = vec_pred.empty() ? "1" : vec_pred;
    vec_side = true;
    return "{ uint32_t weft_m = 0; for (uint32_t l = 0; l < 32u; l++) if (am[l] && (" + p +
           ")) weft_m |= 1u << l; for (uint32_t l = 0; l < 32u; l++) if (am[l] && (" + p +
           ")) { " + store(args[0], "weft_m", "uint32_t") + " } }";
  }

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

  if (op == "wgmma") {
    if (parts.size() >= 2 && (parts[1] == "fence" || parts[1] == "commit_group" ||
                              parts[1] == "wait_group")) {
      return guard + ";";
    }
    if (parts.size() < 2 || parts[1] != "mma_async") {
      err = "unsupported wgmma instruction '" + opcode + "'";
      return "";
    }
    const char* fp8dec = has_part(parts, "e5m2") ? "weft_e5m2("
                         : has_part(parts, "e4m3") ? "weft_e4m3(" : nullptr;
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
        return std::string(bytedec) + "(uint32_t)*(" + addr_space("threadgroup", true) +
               "uint8_t*)(sh + (" + off + ")), 0)";
      }
      if (tf) {
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
        "const uint32_t weft_as = weft_sma ? 8u - weft_sma : 4u, "
        "weft_bs = weft_smb ? 8u - weft_smb : 4u; "
        "const uint32_t weft_aw = 1u << weft_as, weft_bw = 1u << weft_bs; "
        "const uint32_t weft_ak = weft_aw - 1u, weft_bk = weft_bw - 1u; "
        "const uint32_t weft_am = (weft_aw >> 4u) - 1u, weft_bm = (weft_bw >> 4u) - 1u; "
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
    // The swizzle, applied to an offset measured from the descriptor's start.
    auto swz = [&](const std::string& d, const char* sfx) {
      const std::string x = "(((" + d + ") + weft_" + sfx + "o) >> 7u)";
      return "((" + d + ") ^ ((" + x + " & weft_" + sfx + "m) << 4u))";
    };
    // A: row 16w+g (and 16w+g+8), column k.
    auto aval = [&](int rowhalf, int k) {
      const std::string rowg = rowhalf ? "(2u * weft_w + 1u)" : "(2u * weft_w)";
      const std::string rowi = rowhalf ? "(16u * weft_w + weft_g + 8u)"
                                       : "(16u * weft_w + weft_g)";
      // The byte column of this k inside a row.
      const std::string kb = "(" + std::to_string(esz * k) + "u >> weft_as)";
      const std::string ko = "(" + std::to_string(esz * k) + "u & weft_ak)";
      std::string d;
      if (trans_a) {
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
    // B is K rows by N columns.
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
    const std::string ldecl =
        "const uint32_t weft_l = " +
        std::string(vec ? "(ctx.lid0 + l)" : "ctx.lid") +
        ", weft_w = weft_l >> 5u, weft_g = (weft_l >> 2u) & 7u, weft_t = weft_l & 3u; ";
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
    for (const auto& p : parts) {
      if (p == "kind::f8f6f4" && args.size() <= 4) continue;
      if (p.rfind("kind", 0) == 0 || p == "block_scale" || p.rfind("scale_vec", 0) == 0) {
        err = "mma '" + opcode + "': a block-scaled form carries the per-block scale "
              "operands after the accumulators, and the product without them is a "
              "different number";
        return "";
      }
    }
    // Four shapes, and an operand list no longer than `d, a, b[, c]`.
    if (args.size() > 4) {
      err = "mma '" + opcode + "' has " + std::to_string(args.size()) +
            " operands: the forms translated are accumulators, A, B and optionally C";
      return "";
    }
    // The four shapes Triton emits for `tl.dot`.
    const bool bf16 = has_part(parts, "bf16");
    const bool e4m3 = has_part(parts, "e4m3");
    const bool e5m2 = has_part(parts, "e5m2");
    // The integer dot.
    const bool s8 = has_part(parts, "s8");
    const bool u8 = has_part(parts, "u8");
    const bool s4 = has_part(parts, "s4");
    const bool u4 = has_part(parts, "u4");
    // The narrow float operands: four-bit e2m1 and six-bit e2m3/e3m2.
    const bool e3m2 = has_part(parts, "e3m2");
    const bool e2m3 = has_part(parts, "e2m3");
    const bool e2m1 = has_part(parts, "e2m1");
    const bool f16 = has_part(parts, "f16");
    const bool tf32 = has_part(parts, "tf32");
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
    const bool intop = int8op || int4op;
    auto own = [&](const std::string& tok, const std::string& lane) {
      Reg r = parse_reg(tok);
      if (!r.ok) { err = "expected a register, got '" + tok + "'"; return std::string("0.0f"); }
      const std::string v = "s->" + r.base + "[" + std::to_string(r.index) + "][" + lane + "]";
      auto it = reg_types.find(r.base);
      return (it != reg_types.end() && it->second == "float") ? v : ("weft_f32(" + v + ")");
    };
    auto raw = [&](const std::string& tok, const std::string& lane) {
      Reg r = parse_reg(tok);
      if (!r.ok) { err = "expected a register, got '" + tok + "'"; return std::string("0u"); }
      return "s->" + r.base + "[" + std::to_string(r.index) + "][" + lane + "]";
    };
    auto elem = [&](const std::string& tok, const std::string& lane, int which) {
      if (per > 1) {
        Reg r = parse_reg(tok);
        if (!r.ok) { err = "expected a register, got '" + tok + "'"; return std::string("0.0f"); }
        const std::string v = "s->" + r.base + "[" + std::to_string(r.index) + "][" + lane + "]";
        auto it = reg_types.find(r.base);
        const std::string bits = (it != reg_types.end() && it->second == "float") ?
                                     ("weft_bits(" + v + ")") : v;
        // An 8-bit integer operand: the byte is the element, widened to float.
        if (int8op)
          return std::string(s8 ? "((float)(int8_t)(" : "((float)(uint8_t)(") + bits + " >> " +
                 std::to_string(8 * which) + "u))";
        if (int4op) {
          const std::string sh = std::to_string(4 * which) + "u";
          if (s4)
            return "((float)(int32_t)((int32_t)(((uint32_t)(" + bits + ") >> " + sh +
                   ") << 28u) >> 28))";
          return "((float)(((uint32_t)(" + bits + ") >> " + sh + ") & 0xFu))";
        }
        if (e2m1)
          return "weft_e2m1(" + bits + ", " + std::to_string(which) + ")";
        const char* fn = e4m3 ? "weft_e4m3("
                         : e5m2 ? "weft_e5m2("
                         : e3m2 ? "weft_e3m2("
                         : e2m3 ? "weft_e2m3("
                                : (bf16 ? "weft_bf16(" : "weft_h16(");
        return std::string(fn) + bits + ", " + std::to_string(which) + ")";
      }
      return "weft_tf32(" + own(tok, lane) + ")";
    };
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
    // D[r][n] = sum_k A[r][k] * B[k][n] + C[r][n].
    for (int q = 0; q < nd; q++) {
      const int rowhalf = q / 2, colhalf = q % 2;
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
    const bool is_atom = (op == "atom");
    const bool b128atom = has_part(parts, "b128");
    bool has_result = is_atom;
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
    auto atom_ptr = [&](const std::string& t) -> std::string {
      if (shared)
        return "(" + addr_space("threadgroup", false) + t + "*)(sh + (" + addr + "))";
      return "(" + addr_space("device", false) + t + "*)(uintptr_t)(" + addr + ")";
    };
    auto atom_word = [&](const std::string& t, const std::string& a) -> std::string {
      if (shared)
        return "(" + addr_space("threadgroup", false) + t + "*)(sh + (" + a + " & ~3ull))";
      return "(" + addr_space("device", false) + t + "*)(uintptr_t)(" + a + " & ~3ull)";
    };
    if (b128atom) {
      if (aop != "cas" && aop != "exch") {
        err = "'" + opcode + "' is a ." + aop + " atomic on .b128: the isa names only "
              ".cas and .exch at that width";
        return "";
      }
      if (args.size() <= (aop == "cas" ? vi + 1 : vi)) { err = "bad " + opcode; return ""; }
      if (shared) {
        // Sixteen bytes have no atomic instruction, so the lock covers the whole value.
        std::string inner = "{ const weft_u128 weft_v = " +
                            operand_as(args[vi + (aop == "cas" ? 1 : 0)], "weft_u128") +
                            "; ";
        if (aop == "cas")
          inner += "const weft_u128 weft_c = " + operand_as(args[vi], "weft_u128") + "; ";
        inner += addr_space("threadgroup", false) + "weft_u128* weft_p = " +
                 atom_ptr("weft_u128") + "; ";
        inner += "const weft_u128 weft_o = *weft_p; ";
        inner += aop == "cas"
                     ? "if (weft_o.lo == weft_c.lo && weft_o.hi == weft_c.hi) *weft_p = weft_v; "
                     : "*weft_p = weft_v; ";
        if (has_result) inner += store(args[0], "weft_o", "weft_u128") + " ";
        inner += "(void)weft_o; }";
        if (vec) {
          const bool sel = vec_select;
          vec_select = false;
          vec_side = true;
          std::string out = "{ ";
          if (msl) out += "device uint32_t* weft_lk = ctx.locks + (weft::kLockWords - 1u); "
                          "weft::lock_take(weft_lk); ";
          out += "for (uint32_t l = 0; l < 32u; l++) { if (" + lane_cond() + ") " + inner +
                 " } ";
          if (msl) out += "weft::lock_drop(weft_lk); ";
          out += "}";
          vec_select = sel;
          return out;
        }
        if (!msl) return guard + inner;
        wide_atom = true;
        return guard + "{ device uint32_t* weft_lk = ctx.locks + (weft::kLockWords - 1u); "
                       "weft::lock_take(weft_lk); " + inner + " weft::lock_drop(weft_lk); }";
      }
      const std::string p = atom_ptr("weft_u128");
      const std::string c = (aop == "cas") ? operand_as(args[vi], "weft_u128")
                                           : std::string();
      const std::string val =
          operand_as(args[vi + (aop == "cas" ? 1 : 0)], "weft_u128");
      if (msl) {
        wide_atom = true;
        if (vec && p.find("[l]") == std::string::npos) {
          const bool sel = vec_select;
          vec_select = false;
          const std::string raw = aop == "cas"
              ? ("weft::rmw_cas_b128_raw(" + p + ", " + c + ", " + val + ")")
              : ("weft::rmw_exch_b128_raw(" + p + ", " + val + ")");
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
        const std::string call = aop == "cas"
            ? ("weft::rmw_cas_b128(ctx.locks, " + p + ", " + c + ", " + val + ")")
            : ("weft::rmw_exch_b128(ctx.locks, " + p + ", " + val + ")");
        if (!has_result) return guard + "(void)" + call + ";";
        return guard + store(args[0], call, "weft_u128");
      }
      const std::string call = aop == "cas"
          ? ("weft_atom_cas_b128(" + p + ", " + c + ", " + val + ")")
          : ("weft_atom_exch_b128(" + p + ", " + val + ")");
      if (!has_result) return guard + "(void)" + call + ";";
      return guard + store(args[0], call, "weft_u128");
    }
    if (shared && cpp_size(ty) == 8) {
      if (ty != "uint64_t" && ty != "int64_t") {
        err = "'" + opcode + "' is a 64-bit atomic in shared memory: that width has no "
              "atomic instruction in the window on this host's GPU";
        return "";
      }
      std::string expr;
      if (aop == "add") expr = "o + v";
      else if (aop == "sub") expr = "o - v";
      else if (aop == "min") expr = "(o < v ? o : v)";
      else if (aop == "max") expr = "(o > v ? o : v)";
      else if (aop == "and") expr = "o & v";
      else if (aop == "or") expr = "o | v";
      else if (aop == "xor") expr = "o ^ v";
      else if (aop == "exch") expr = "v";
      else if (aop != "cas") {
        err = "'" + opcode + "' is a ." + aop + " atomic in the window at 64 bits, which "
              "the isa does not name";
        return "";
      }
      if (aop == "cas" || has_result) {
        // One thread runs the warp, so the lanes go out in lane order under one lock.
        if (!vec) {
          warp_only = true;
          err = "'" + opcode + "' is a 64-bit atomic in the window whose result is used, "
                "so it needs the warp backend: one thread has to run the lanes in order, "
                "and a lane per thread reaches the lock in the GPU's order";
          return "";
        }
        const bool sel = vec_select;
        vec_select = false;
        vec_side = true;
        const std::string p = atom_ptr(ty);
        std::string inner = "{ ";
        if (aop == "cas")
          inner += "const " + ty + " c = " + operand_as(args[vi], ty) + "; const " + ty +
                   " v = " + operand_as(args[vi + 1], ty) + "; ";
        else
          inner += "const " + ty + " v = " + operand_as(args[vi], ty) + "; ";
        inner += addr_space("threadgroup", false) + ty + "* weft_p = " + p + "; ";
        inner += "const " + ty + " o = *weft_p; ";
        if (aop == "cas") inner += "if (o == c) *weft_p = v; ";
        else inner += "*weft_p = (" + ty + ")(" + expr + "); ";
        if (has_result) inner += store(args[0], "o", ty) + " ";
        inner += "}";
        std::string out = "{ ";
        if (msl) {
          wide_atom = true;
          out += "device uint32_t* weft_lk = ctx.locks + (weft::kLockWords - 1u); "
                 "weft::lock_take(weft_lk); ";
        }
        out += "for (uint32_t l = 0; l < 32u; l++) { if (" + lane_cond() + ") " + inner + " } ";
        if (msl) out += "weft::lock_drop(weft_lk); ";
        out += "}";
        vec_select = sel;
        return out;
      }
      const std::string p = atom_ptr(ty);
      std::string body = "{ const " + ty + " v = " + operand_as(args[vi], ty) + "; ";
      if (msl) {
        wide_atom = true;
        body += "device uint32_t* weft_lk = ctx.locks + (((uint32_t)(" + addr +
                ") >> 3) & (weft::kLockWords - 1u)); weft::lock_take(weft_lk); ";
      }
      body += "const " + ty + " o = *" + p + "; *" + p + " = (" + ty + ")(" + expr + "); ";
      if (msl) body += "weft::lock_drop(weft_lk); ";
      body += "(void)o; }";
      return guard + body;
    }
    // Vector atomics.
    int vn = 0;
    for (size_t i = 1; i < parts.size(); i++) {
      const std::string& p = parts[i];
      if (p.size() == 2 && p[0] == 'v' && p[1] >= '2' && p[1] <= '8') vn = p[1] - '0';
    }
    if (vn) {
      // The isa uses a vector atom only with `.global`, so the window has scalar forms.
      if (shared) {
        err = "'" + opcode + "' is a vector atomic in the window: the isa says a vector "
              "atom may be used only with .global space, so at this width the window has "
              "the scalar forms alone";
        return "";
      }
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
      auto elems = [&](const std::string& tok, std::vector<std::string>& out) {
        std::string t = trim(tok);
        if (t.size() < 2 || t.front() != '{') return false;
        t = t.substr(1);
        if (!t.empty() && t.back() == '}') t.pop_back();
        out = split_top(t, ',');
        return true;
      };
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
    // A 16-bit float atomic.
    if (has_part(parts, "f16") || has_part(parts, "bf16") || has_part(parts, "f16x2") ||
        has_part(parts, "bf16x2")) {
      const bool pack = has_part(parts, "f16x2") || has_part(parts, "bf16x2");
      const bool bf = has_part(parts, "bf16") || has_part(parts, "bf16x2");
      if (aop != "add") {
        err = "'" + opcode + "' is a ." + aop + " atomic on a 16-bit float: the isa names "
              ".add alone at that width, and min and max only in the .v2 and .v4 forms, "
              "which are .global only";
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
      const bool order_free = !has_result && (aop == "add" || aop == "sub" || aop == "and" ||
                                              aop == "or" || aop == "xor");
      if (order_free)
        return guard + "(void)weft::red_" + aop + "_u64((" + addr_space("device", false) + ty +
               "*)(uintptr_t)(" + addr + "), " + operand_as(args[vi], ty) + ");";
      wide_atom = true;
      const std::string p =
          "((" + addr_space("device", false) + ty + "*)(uintptr_t)(" + addr + "))";
      const std::string val = operand_as(args[vi], ty);
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
    // The ISA's table.
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
    const std::string dq = msl ? "device " : "";
    const std::string sq = msl ? "threadgroup const " : "const ";
    return guard + "{ " + dq + cty + "* weft_d = (" + dq + cty + "*)(uintptr_t)(" + dst +
           "); " + sq + cty + "* weft_s = (" + sq + cty + "*)(sh + (" + src + ")); "
           "for (uint32_t k = 0; k < (" + bytes + ") / sizeof(" + cty + "); k++) { " + combine +
           " } }";
  }

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
      const bool column = has_part(parts, "im2col") || has_part(parts, "tile::im2col");
      auto one_copy = [&](const std::string& body) -> std::string {
        if (!vec) return guard + body;
        return "{ for (uint32_t l = 0; l < 1u; l++) { if (" + lane_cond() + ") " + body + " } }";
      };
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
      std::string cargs;
      for (int i = 0; i < 5; i++)
        cargs += ", " + ((size_t)i < co.size()
                             ? ("(int32_t)(" + operand_as(co[(size_t)i], "int32_t") + ")")
                             : std::string("0"));
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
    // A prefetch moves nothing and reports nothing.
    if (has_part(parts, "prefetch")) return guard + ";";
    if (!from_global || !to_shared || args.size() < 3) {
      err = "unsupported bulk copy '" + opcode +
            "': global to threadgroup (completing an mbarrier) and threadgroup to global "
            "are translated";
      return "";
    }
    if (!completes) {
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
    return guard + "{ " + copy + " " + complete(bad, bytes) + " }";
  }

  if (op == "tex" || op == "tld4" || op == "txq" || op == "suq" || op == "suld" ||
      op == "sust") {
    vec_ok = false;
    if (args.empty()) { err = "bad " + opcode; return ""; }
    const bool cube = opcode.find("cube") != std::string::npos;
    const bool d3 = !cube && opcode.find("3d") != std::string::npos;
    const bool d1 = !cube && !d3 && opcode.find("1d") != std::string::npos;
    if (op != "txq" && op != "suq" && !cube && opcode.find("2d") == std::string::npos &&
        !d3 && !d1) {
      err = "only a 1-D, 2-D or 3-D texture is bound here, and '" + opcode + "' is not one";
      return "";
    }

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
    auto image_of = [&](int idx) {
      const std::string i = std::to_string(idx);
      if (!msl) return "*weft::texture_ptr(" + i + ")";
      const bool rw = idx < (int)tex_write.size() && tex_write[idx];
      const bool arr = idx < (int)tex_array.size() && tex_array[idx];
      if (idx < (int)tex_cube.size() && tex_cube[idx])
        return arr ? "weft::make_cube_array_image(weft_t" + i + ")"
                   : "weft::make_cube_image(weft_t" + i + ")";
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
      const bool cb = (size_t)idx < tex_cube.size() && tex_cube[idx];
      if (opcode.find("width") != std::string::npos) what = "width";
      else if (opcode.find("height") != std::string::npos) what = "height";
      else if (opcode.find("depth") != std::string::npos)
        // A cube array's depth is its count of cubes, six faces to a layer.
        what = cb && arr ? "cubes" : (arr || vol) ? "layers" : (cb ? "6" : "1");
      else { err = "txq asks for a texture's width, height or depth"; return ""; }
      const std::string i = std::to_string(idx);
      const std::string field =
          (what == "layers" && vol) ? "depth" : (what == "cubes" ? "layers" : what);
      // The cpu runtime keeps a volume's depth and a cube array's faces in `layers`.
      const std::string cpu_field = what == "cubes" ? "layers" : what;
      const std::string expr =
          what == "6" ? std::string("6u")
                       : what == "1" ? std::string("1u")
                       : (msl ? ("(uint32_t)(" + image_of(idx) + "." + field +
                                 (what == "cubes" ? " / 6u" : "") + ")")
                              : ("((uint32_t)weft::texture_ptr(" + i + ")->" + cpu_field +
                                 (what == "cubes" ? " / 6u" : "") + ")"));
      return guard + store(args[0], expr, "uint32_t");
    }

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
      if (coords.size() < (vol ? 3u : arr ? (d1 ? 2u : 3u) : (d1 ? 1u : 2u))) {
        err = vol ? "a 3-D surface needs x, y and a plane"
                  : arr ? (d1 ? "a 1-D array surface needs a layer and a coordinate"
                              : "a 2-D array surface needs a layer, x and y")
                        : d1 ? "a 1-D surface needs one coordinate"
                             : "a 2-D surface needs two coordinates";
        return "";
      }
      if (!read && idx < (int)tex_write.size()) tex_write[idx] = true;
      // A layered surface names its layer first, as a texture does; a volume names its plane
      // as the third coordinate, which is the layer the runtime takes.
      const std::string x =
          "(uint32_t)(" + operand_as(coords[arr && !vol ? 1 : 0], "uint32_t") + ")";
      const std::string y =
          d1 ? std::string("0u")
             : "(uint32_t)(" + operand_as(coords[arr && !vol ? 2 : 1], "uint32_t") + ")";
      const std::string lay =
          vol ? (", (uint32_t)(" + operand_as(coords[2], "uint32_t") + ")")
              : arr ? (", (uint32_t)(" + operand_as(coords[0], "uint32_t") + ")")
                    : std::string();
      // `.b` addresses the surface in bytes and moves the type's own width.
      uint32_t nbytes = 4;
      for (const std::string& p : parts) {
        if (p == "b8" || p == "s8" || p == "u8") nbytes = 1;
        else if (p == "b16" || p == "s16" || p == "u16") nbytes = 2;
        else if (p == "b64" || p == "s64" || p == "u64") nbytes = 8;
      }
      if (nbytes != 4) {
        err = "'" + opcode + "' moves " + std::to_string(nbytes) +
              " bytes a lane, and a surface here is 4-byte texels addressed in bytes: only .b32";
        return "";
      }
      if (read) {
        return guard + store(args[0],
                             "weft::tex_byte_read(" + image_of(idx) + ", " + x + ", " + y + lay +
                                 ")",
                             "uint32_t");
      }
      std::vector<std::string> vals = reg_list(args[1]);
      if (vals.empty()) { err = "a surface store needs a value"; return ""; }
      const std::string v = operand_as(vals[0], "float");
      return guard + "weft::tex_byte_write(" + image_of(idx) + ", " + x + ", " + y + lay + ", " +
             v + ");";
    }

    if (cube) {
      std::vector<std::string> coords;
      const int idx = binding(args[1], coords);
      if (idx < 0) return "";
      const bool arr = (size_t)idx < tex_array.size() && tex_array[idx];
      if (coords.size() < (arr ? 4u : 3u)) {
        err = arr ? "a cube array sample needs a layer and a direction (s, t, r)"
                  : "a cube sample needs a direction (s, t, r)";
        return "";
      }
      // A cube's coordinates are a direction, (s, t, r), and a cube array names its layer first.
      const std::string lay = arr ? (", (uint32_t)(" + operand_as(coords[0], "uint32_t") + ")")
                                  : std::string();
      const std::string dx = operand_as(coords[arr ? 1 : 0], "float");
      const std::string dy = operand_as(coords[arr ? 2 : 1], "float");
      const std::string dz = operand_as(coords[arr ? 3 : 2], "float");
      const std::string i = std::to_string(idx);
      const std::string tn = (msl && idx < (int)tex_write.size() && tex_write[idx])
                                 ? ("weft_t" + i + "_s")
                                 : ("weft_t" + i);
      const std::string img =
          msl ? (arr ? ("weft::SampledCubeArrayImage{weft::make_cube_array_image(" + tn +
                        "), weft_s" + i + "}")
                     : ("weft::SampledCubeImage{weft::make_cube_image(" + tn + "), weft_s" + i +
                        "}"))
              : ("*weft::texture_ptr(" + i + ")");
      const bool cgather = op == "tld4";
      // A gather of a component the one-channel binding does not have is zero,
      // as a fetch of one is: `tld4.b` off an r32 image reads nothing.
      const bool gather_absent = cgather && gather_component(opcode) != 0;
      const std::string call =
          cgather ? ((arr ? "weft::tex_gather4_cube_array_dir<float>(" :
                             "weft::tex_gather4_cube_dir<float>(") +
                     img + ", " + dx + ", " + dy + ", " + dz + lay + ", 0.0f, 0.0f, weft_t4)")
          : msl ? ("weft::tex_sample4<float>(" + img + ", " + dx + ", " + dy + ", " + dz + lay +
                    ", weft_t4)")
                : ("weft::tex_sample4_cube<float>(" + img + ", " + dx + ", " + dy + ", " + dz + lay +
                    ", weft::sampler(" + i + ").filter, weft::sampler(" + i + ").addr, weft_t4)");
      std::vector<std::string> dsts = reg_list(args[0]);
      std::string out = guard + "{ float weft_t4[4]; " + call + ";";
      for (size_t k = 0; k < dsts.size() && k < 4; k++)
        out += " " +
               store(dsts[k], gather_absent ? std::string("0.0f") : ("weft_t4[" + std::to_string(k) + "]"),
                     "float");
      out += " }";
      return out;
    }

    // `tex`: a filtered sample at normalised coordinates.
    if (args.size() < 2) { err = "bad " + opcode; return ""; }
    std::vector<std::string> coords;
    const int idx = binding(args[1], coords);
    if (idx < 0) return "";
    const bool arr = (size_t)idx < tex_array.size() && tex_array[idx];
    const bool vol = (size_t)idx < tex_3d.size() && tex_3d[idx];
    // An `.a1d` tuple is {layer, s}: two elements, not three.
    if (coords.size() < (vol ? 3u : arr ? (d1 ? 2u : 3u) : (d1 ? 1u : 2u))) {
      err = vol ? "a 3-D texture needs u, v and w"
                : arr ? (d1 ? "a 1-D array texture needs a layer and a coordinate"
                            : "a 2-D array texture needs a layer, u and v")
                      : d1 ? "a 1-D texture needs one coordinate"
                           : "a 2-D texture needs two coordinates";
      return "";
    }
    // A texture array names its layer before the coordinates, as a surface does; a volume
    // names its plane after them.
    const std::string u = operand_as(arr ? coords[1] : coords[0], "float");
    const std::string v = d1 ? std::string("0.5f")
                             : operand_as(arr ? coords[2] : coords[1], "float");
    const std::string lay = vol ? (", " + operand_as(coords[2], "float"))
                                : arr ? (", (uint32_t)(" + operand_as(coords[0], "uint32_t") + ")")
                                      : std::string();
    const std::string i = std::to_string(idx);
    const bool gather = op == "tld4";
    const bool gather_absent = gather && gather_component(opcode) != 0;
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
      out += " " +
             store(dsts[k], gather_absent ? std::string("0.0f") : ("weft_t4[" + std::to_string(k) + "]"),
                   "float");
    out += " }";
    return out;
  }

  if (op == "nop") return guard + ";";

  if (op == "prefetch" || op == "prefetchu" || op == "discard") return guard + ";";

  if (op == "applypriority" || op == "griddepcontrol" || op == "pmevent" ||
      op == "setmaxnreg")
    return guard + ";";
  if (op == "createpolicy") return guard + store(args[0], "0ull", ty.empty() ? "uint64_t" : ty);
  if (op == "mapa" && args.size() >= 3)
    return guard + store(args[0], operand_as(args[1], ty.empty() ? "uint32_t" : ty),
                         ty.empty() ? "uint32_t" : ty);
  if (op == "barrier" && !parts.empty() && parts.size() > 1 && parts[1] == "cluster")
    return guard + ";";

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

  if (op == "nanosleep") return guard + "(void)0;";

  if (op == "call") {
    if (args.size() < 2) { err = "malformed '" + opcode + "'"; return ""; }
    size_t next = 0;
    std::string tgt = trim(args[next++]);
    std::vector<std::string> ret_dsts;
    if (!tgt.empty() && tgt[0] == '(') {
      const std::string inner = trim(tgt.substr(1, tgt.size() - 2));
      const std::vector<std::string> slots =
          inner.empty() ? std::vector<std::string>() : split_top(inner, ',');
      for (const std::string& slot : slots) ret_dsts.push_back(trim(slot));
      if (next >= args.size()) { err = "malformed '" + opcode + "'"; return ""; }
      tgt = trim(args[next++]);
    }
    {
      auto cc = callee_cycle.find(tgt);
      if (cc != callee_cycle.end()) {
        if (wctx) {
          err = "'" + opcode + " " + tgt + "': a call cycle in the warp body, whose context "
                "is not the per-lane one the cycle's frames are carried in";
          return "";
        }
        std::string list = (next < args.size()) ? trim(args[next]) : std::string();
        if (!list.empty() && list.front() == '(' && list.back() == ')')
          list = list.substr(1, list.size() - 2);
        const std::vector<std::string> avals =
            trim(list).empty() ? std::vector<std::string>() : split_top(list, ',');
        if (!ret_dsts.empty()) {
          err = "'" + opcode + " " + tgt + "' is in a return list, and a function in a "
                "call cycle returns nothing";
          return "";
        }
        auto cp = callee_cycle_params.find(tgt);
        const std::vector<std::string> ptys =
            (cp == callee_cycle_params.end()) ? std::vector<std::string>() : cp->second;
        if (avals.size() != ptys.size()) {
          err = "'" + opcode + " " + tgt + "' passes " + std::to_string(avals.size()) +
                " arguments and the function declares " + std::to_string(ptys.size());
          return "";
        }
        const bool inside = (cc->second == self_cycle && !cycle_sel.empty());
        const std::string arr = inside ? "wa" : "weft_a";
        std::string asg;
        for (size_t k = 0; k < avals.size(); k++)
          asg += arr + "[" + std::to_string(k) + "] = " + to_word(avals[k], ptys[k]) + "; ";
        const std::string at = std::to_string(callee_cycle_at[tgt]) + "u";
        if (inside)
          return guard + "{ " + asg + "*" + cycle_sel + " = " + at + "; return; }";
        const std::string cid = std::to_string(cc->second);
        const std::string cyc = "wcs_" + cid;
        const std::string cty = "weft_cs_" + cid;
        bool have_cy = false;
        for (const auto& m : callee_members)
          if (m.second == cyc) have_cy = true;
        if (!have_cy) callee_members.emplace_back(cty, cyc);
        std::string init = "s->" + cyc + ".wsel = 0x80000000u | " + at + "; ";
        for (size_t k = 0; k < avals.size(); k++)
          init += "s->" + cyc + ".wa[" + std::to_string(k) + "] = " +
                  to_word(avals[k], ptys[k]) + "; ";
        const std::string run = "weft_tr_" + cid + "(ctx, &s->" + cyc + ");";
        auto su = callee_suspends.find(tgt);
        if (su == callee_suspends.end() || !su->second) return guard + "{ " + init + run + " }";
        const int crid = next_resume++;
        const std::string cids = std::to_string(crid);
        if (!msl) resume_cases += "    case " + cids + " : goto R" + cids + ";\n";
        const std::string crlab = msl ? ("case " + cids + ":;") : ("R" + cids + ":;");
        return guard + "{ " + init + crlab + "\n  " + run +
               "\n  if (s->" + cyc + ".wsel & 0x80000000u) { s->" + cyc +
               ".wsel &= 0x7FFFFFFFu; s->pc = " + cids + "u; return; } }";
      }
    }
    if (self_deep && tgt == self_name) {
      // A self call with work after it becomes a call to the next copy of the body,
      // so each depth gets its own frame and neither column has to recurse.
      std::string list = (next < args.size()) ? trim(args[next]) : std::string();
      if (!list.empty() && list.front() == '(' && list.back() == ')')
        list = list.substr(1, list.size() - 2);
      const std::vector<std::string> avals =
          trim(list).empty() ? std::vector<std::string>() : split_top(list, ',');
      if (avals.size() != self_params.size()) {
        err = "'" + opcode + " " + tgt + "' passes " + std::to_string(avals.size()) +
              " arguments and the function declares " + std::to_string(self_params.size());
        return "";
      }
      if (ret_dsts.size() != self_rets.size()) {
        err = "'" + opcode + " " + tgt + "' names " + std::to_string(ret_dsts.size()) +
              " destinations and the function returns " + std::to_string(self_rets.size());
        return "";
      }
      const std::string tok = "WEFTSELF" + std::to_string(self_sites.size());
      std::vector<std::string> conv;
      for (size_t i = 0; i < avals.size(); i++)
        conv.push_back(operand_as(avals[i], self_params[i]));
      self_sites.push_back(conv);
      self_site_dsts.push_back(ret_dsts);
      return guard.empty() ? (tok + ";") : (guard + "{ " + tok + "; }");
    }
    if (self_tail && tgt == self_name) {
      std::string list = (next < args.size()) ? trim(args[next]) : std::string();
      if (!list.empty() && list.front() == '(' && list.back() == ')')
        list = list.substr(1, list.size() - 2);
      const std::vector<std::string> avals =
          trim(list).empty() ? std::vector<std::string>() : split_top(list, ',');
      if (!ret_dsts.empty()) {
        err = "'" + opcode + " " + tgt + "' names destinations for return values, and the "
              "call falls straight through to 'ret' with none";
        return "";
      }
      if (avals.size() != self_params.size()) {
        err = "'" + opcode + " " + tgt + "' passes " + std::to_string(avals.size()) +
              " arguments and the function declares " + std::to_string(self_params.size());
        return "";
      }
      // A tail self-call is a loop: the parameters come from temporaries, so swapped
      // arguments survive, and the body starts again with no frame.
      std::string asg;
      for (size_t i = 0; i < avals.size(); i++)
        asg += "const " + self_params[i] + " weft_t" + std::to_string(i) + " = (" +
               self_params[i] + ")(" + operand_as(avals[i], self_params[i]) + "); ";
      for (size_t i = 0; i < avals.size(); i++)
        asg += "weft_a" + std::to_string(i) + " = weft_t" + std::to_string(i) + "; ";
      return guard + "{ " + asg + "continue; }";
    }
    auto f = callees.find(tgt);
    if (f == callees.end()) {
      auto bad = callee_errs.find(tgt);
      if (bad != callee_errs.end())
        err = "'" + opcode + " " + tgt + "': the callee did not translate: " + bad->second;
      else if (entry_names.count(tgt))
        err = "'" + opcode + " " + tgt + "' names an .entry, and an entry is launched, not "
              "called: a call has to name a .func in this module";
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
    auto rt = callee_rets.find(tgt);
    const std::vector<std::string> rts =
        (rt == callee_rets.end()) ? std::vector<std::string>() : rt->second;
    if (ret_dsts.empty() && !rts.empty()) {
      err = "'" + opcode + " " + tgt + "' returns " +
            (rts.size() == 1 ? std::string("a value")
                             : std::to_string(rts.size()) + " values") +
            " and the call has no list to put it in";
      return "";
    }
    if (!ret_dsts.empty() && rts.empty()) {
      err = "'" + opcode + " " + tgt + "' is in a return list, and that function returns nothing";
      return "";
    }
    if (ret_dsts.size() != rts.size()) {
      err = "'" + opcode + " " + tgt + "' names " + std::to_string(ret_dsts.size()) +
            " destinations and the function returns " + std::to_string(rts.size());
      return "";
    }
    const bool tdeep = callee_deep.count(tgt) && callee_deep[tgt];
    const std::string mem = "wcf_" + tgt;
    // The frames of a chain of copies are the caller's, so a stopped call keeps them.
    const std::string cmem = tdeep ? ("s->" + mem + "[0]") : ("s->" + mem);
    std::string assign;
    for (size_t k = 0; k < ret_dsts.size(); k++) {
      if (!assign.empty()) assign += " ";
      assign += store(ret_dsts[k], cmem + ".weft_ret" + std::to_string(k), rts[k]);
    }
    const std::string mt = tdeep ? (mem + "[8]") : mem;
    bool have = false;
    for (const auto& m : callee_members)
      if (m.second == mt) have = true;
    if (!have) callee_members.emplace_back("weft_cf_" + tgt, mt);
    vec_ok = false;
    std::string call = "weft_f_" + tgt + (tdeep ? "_0" : "") + "(ctx, " +
                       (tdeep ? ("s->" + mem) : ("&s->" + mem));
    for (size_t i = 0; i < avals.size(); i++)
      call += ", (" + f->second[i] + ")(" + operand_as(avals[i], f->second[i]) + ")";
    call += ");";
    auto sp = callee_suspends.find(tgt);
    const bool suspends = (sp != callee_suspends.end() && sp->second);
    if (!suspends) {
      if (!assign.empty()) call += " " + assign;
      return guard + call;
    }
    const int rid = next_resume++;
    const std::string rids = std::to_string(rid);
    // A cycle member tells the driver it stopped, so a jump cannot look like a stop.
    const std::string stopf =
        cycle_sel.empty() ? std::string() : ("*" + cycle_sel + " |= 0x80000000u; ");
    // Metal has no labelled statements, so there the resume point is the case itself,
    // nested in the body's switch, and the switch dispatches straight to it.
    if (!msl) resume_cases += "    case " + rids + ": goto R" + rids + ";\n";
    const std::string rlab = msl ? ("case " + rids + ":;") : ("R" + rids + ":;");
    if (!guard.empty()) {
      // A predicated call to a callee that stops.
      const std::string cond = guard.substr(4, guard.size() - 6);
      const std::string cg = "cg" + rids;
      callee_members.emplace_back("uint32_t", cg);
      std::string out = "s->" + cg + " = (" + cond + ") ? 1u : 0u;\n  if (s->" + cg +
                        ") " + cmem + ".pc = 0u;\n  " + rlab + "\n  if (s->" + cg +
                        ") {\n  " + call;
      out += "\n  if (" + cmem + ".pc != 0xFFFFFFFFu) { " + stopf + "s->pc = " + rids +
             "u; return; }";
      if (!assign.empty()) out += "\n  " + assign;
      out += "\n  }";
      return out;
    }
    std::string out = cmem + ".pc = 0u;\n  " + rlab + "\n  " + call;
    out += "\n  if (" + cmem + ".pc != 0xFFFFFFFFu) { " + stopf + "s->pc = " +
           std::to_string(rid) + "u; return; }";
    if (!assign.empty()) out += " " + assign;
    return out;
  }

  if (op == "trap") {
    if (wctx) {
      err = "'" + opcode + "' inside a body that runs 32 lanes in one thread: the stop is "
            "per lane, and that body keeps one program counter per lane, not the one this "
            "stop would end";
      return "";
    }
    // Hardware stops the lane here. The stop is written to the word the host reads, and
    // the lane ends, so the run is reported as a trap rather than finishing quietly.
    return guard + "{ if (ctx.fault) { ctx.fault[0] = 2u; ctx.fault[1] = "
                    "(uint32_t)ctx.gid; } s->pc = 0xFFFFFFFFu; return; }";
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
  // The module prologue: everything before the first `.entry`.
  const size_t first_entry = src.find(".entry");
  const std::string module_head =
      src.substr(0, first_entry == std::string::npos ? src.size() : first_entry);

  std::map<std::string, int> tex_index;
  std::vector<std::string> tex_names;  // index -> the declared name
  int ntex = 0;
  {
    const std::string& head = module_head;
    size_t at = 0;
    while (at < head.size()) {
      size_t nl = head.find('\n', at);
      if (nl == std::string::npos) nl = head.size();
      std::string s = trim(head.substr(at, nl - at));
      at = nl + 1;
      if (!s.empty() && s.back() == ';') s.pop_back();
      s = trim(s);
      // Three spellings name a binding.
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

  std::map<std::string, std::vector<std::string>> callee_types;
  std::map<std::string, std::vector<std::string>> callee_rets;
  std::map<std::string, bool> callee_suspends;
  std::map<std::string, bool> callee_deep;
  std::map<std::string, bool> entry_names;
  for (size_t at = 0; (at = src.find(".entry", at)) != std::string::npos; at += 6) {
    size_t ei = at + 6;
    while (ei < src.size() && std::isspace((unsigned char)src[ei])) ei++;
    const size_t eb = ei;
    while (ei < src.size() && is_word(src[ei])) ei++;
    if (ei > eb) entry_names[src.substr(eb, ei - eb)] = true;
  }
  std::map<std::string, int> cycle_of, callee_cycle_at;
  std::map<std::string, std::vector<std::string>> callee_cycle_params;
  std::map<std::string, std::string> callee_errs;
  {
    struct FuncDef {
      std::string name;
      std::vector<std::string> pnames, ptypes;
      std::vector<std::string> ret_names, ret_types;  // the return slots, in list order
      std::vector<std::string> deps;
      std::string body;  // the text between the braces
      bool called = false;
    };
    std::vector<FuncDef> defs;
    std::map<std::string, size_t> def_at;

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
      if (!trim(ret_decl).empty()) {
        bool bad_slot = false;
        for (const std::string& slot : split_top(trim(ret_decl), ',')) {
          const std::vector<std::string> toks = split_top(trim(slot), ' ');
          std::string nm = toks.empty() ? std::string() : toks.back();
          if (!nm.empty() && nm[0] == '%') nm = nm.substr(1);
          std::string ty;
          for (const auto& t : toks) {
            if (t.empty() || t[0] != '.') continue;
            const std::string ct = cpp_type(t);
            if (!ct.empty() && cpp_size(ct)) { ty = ct; break; }
          }
          if (nm.empty()) { bad_slot = true; break; }
          d.ret_names.push_back(nm);
          d.ret_types.push_back(ty.empty() ? "uint32_t" : ty);
        }
        if (bad_slot) {
          callee_errs[fname] = "a return slot of the function is not a name";
          continue;
        }
      }
      if (def_at.count(fname)) continue;  // one definition per name
      def_at[fname] = defs.size();
      defs.push_back(d);
    }

    // A function is translated when some call names it, wherever that call sits.
    {
      std::vector<std::string> calls;
      call_targets(src, calls);
      for (FuncDef& d : defs) {
        call_targets(d.body, d.deps);
        for (const std::string& c : calls)
          if (c == d.name) { d.called = true; break; }
      }
    }

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

    // A self-call that is the last thing a function does needs no frame: the body can
    // start again with the new arguments. Anything else that comes back to its own name
    // still needs a struct that would contain itself.
    std::map<int, std::vector<std::string>> cycle_members;
    std::map<int, int> cycle_left;
    std::vector<bool> self_tail(defs.size(), false);
    std::map<std::string, std::string> self_note;
    {
      std::map<std::string, size_t> seen;
      for (const FuncDef& d : defs)
        for (const std::string& dep : d.deps)
          if (dep == d.name) seen[d.name]++;
      auto only_ret_after = [](const std::string& body, size_t semi) {
        for (size_t k = semi + 1; k < body.size();) {
          while (k < body.size() && std::isspace((unsigned char)body[k])) k++;
          if (k >= body.size()) break;
          if (body[k] == '{' || body[k] == '}' || body[k] == ';') { k++; continue; }
          if (body.compare(k, 3, "ret") == 0) { k += 3; continue; }
          return false;
        }
        return true;
      };
      auto last_call = [](const std::string& body, std::string& tgt, size_t& semi) {
        size_t q = 0;
        bool found = false;
        while ((q = body.find("call", q)) != std::string::npos) {
          if (q > 0 && (is_word(body[q - 1]) || body[q - 1] == '.' || body[q - 1] == '%')) {
            q += 4;
            continue;
          }
          size_t p = q + 4;
          while (p < body.size() && (is_word(body[p]) || body[p] == '.' || body[p] == ':')) p++;
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
          if (p > ns) {
            tgt = body.substr(ns, p - ns);
            const size_t sc = body.find(';', p);
            semi = (sc == std::string::npos) ? body.size() : sc;
            found = true;
          }
          q = p > q + 4 ? p : q + 4;
        }
        return found;
      };
      for (size_t i = 0; i < defs.size(); i++) {
        std::string tgt;
        size_t semi = 0;
        if (!last_call(defs[i].body, tgt, semi) || tgt != defs[i].name) continue;
        if (seen[defs[i].name] != 1 || !only_ret_after(defs[i].body, semi)) continue;
        if (!defs[i].ret_names.empty()) {
          self_note[defs[i].name] = "'" + defs[i].name +
              "' returns " + std::to_string(defs[i].ret_names.size()) +
              " value" + (defs[i].ret_names.size() == 1 ? "" : "s") +
              " and calls itself in tail position: a loop would drop " +
              (defs[i].ret_names.size() == 1 ? "it" : "them");
          continue;
        }
        self_tail[i] = true;
      }

      // A cycle of functions that only tail-call each other runs through one driver: the
      // frames are the driver's locals, and a cross-call is a jump to the next member, so
      // neither column nests a call. Anything else that comes back keeps its refusal.
      std::map<std::string, std::map<std::string, bool>> reach;
      for (const FuncDef& d : defs)
        if (needed.count(d.name))
          for (const std::string& dp : d.deps)
            if (def_at.count(dp)) reach[d.name][dp] = true;
      for (bool more = true; more;) {
        more = false;
        for (auto& kv : reach) {
          std::vector<std::string> add;
          for (const auto& xi : kv.second) {
            auto it = reach.find(xi.first);
            if (it == reach.end()) continue;
            for (const auto& yi : it->second)
              if (!kv.second.count(yi.first)) add.push_back(yi.first);
          }
          for (const std::string& y : add)
            if (!kv.second.count(y)) { kv.second[y] = true; more = true; }
        }
      }
      std::map<std::string, std::string> par;
      for (const FuncDef& d : defs)
        if (needed.count(d.name)) par[d.name] = d.name;
      for (size_t a2 = 0; a2 < defs.size(); a2++) {
        for (size_t b2 = a2 + 1; b2 < defs.size(); b2++) {
          const std::string& a = defs[a2].name;
          const std::string& b = defs[b2].name;
          if (!par.count(a) || !par.count(b)) continue;
          if (!reach[a].count(b) || !reach[b].count(a)) continue;
          std::string ra = a;
          while (par[ra] != ra) ra = par[ra];
          std::string rb = b;
          while (par[rb] != rb) rb = par[rb];
          if (ra != rb) par[ra] = rb;
        }
      }
      std::map<std::string, std::vector<size_t>> groups;
      for (size_t a2 = 0; a2 < defs.size(); a2++) {
        if (!needed.count(defs[a2].name)) continue;
        std::string r = defs[a2].name;
        while (par[r] != r) r = par[r];
        groups[r].push_back(a2);
      }
      for (const auto& g : groups) {
        if (g.second.size() < 2) continue;
        bool ok = true;
        for (size_t gi : g.second) {
          const FuncDef& d = defs[gi];
          if (!d.ret_names.empty() || d.ptypes.size() > 8) { ok = false; break; }
          size_t inside = 0;
          for (const std::string& dp : d.deps)
            for (size_t gj : g.second)
              if (dp == defs[gj].name) inside++;
          if (inside != 1) { ok = false; break; }
          std::string tgt;
          size_t semi = 0;
          if (!last_call(d.body, tgt, semi) || !only_ret_after(d.body, semi)) {
            ok = false;
            break;
          }
          bool in_group = false;
          for (size_t gj : g.second)
            if (tgt == defs[gj].name) in_group = true;
          if (!in_group) { ok = false; break; }
        }
        if (!ok) continue;
        std::vector<std::string> mem;
        for (size_t gi : g.second) mem.push_back(defs[gi].name);
        const int cid = (int)cycle_members.size();
        for (size_t k = 0; k < mem.size(); k++) {
          cycle_of[mem[k]] = cid;
          callee_cycle_at[mem[k]] = (int)k;
          callee_cycle_params[mem[k]] = defs[def_at[mem[k]]].ptypes;
        }
        cycle_members[cid] = mem;
        cycle_left[cid] = (int)mem.size();
      }
    }

    std::map<std::string, int> state;  // 1 translated, 2 refused: both are answers
    std::string callee_out;

    auto translate = [&](size_t i) -> bool {
      const std::string& fname = defs[i].name;
      const bool in_cycle = cycle_of.count(fname) != 0;
      Emitter ef;
      ef.msl = gpu;
      ef.callees = callee_types;
      ef.callee_errs = callee_errs;
      ef.callee_rets = callee_rets;
      ef.callee_suspends = callee_suspends;
      ef.callee_ret = defs[i].ret_names;
      ef.self_name = fname;
      ef.self_tail = self_tail[i];
      ef.self_params = defs[i].ptypes;
      ef.callee_cycle = cycle_of;
      ef.callee_cycle_at = callee_cycle_at;
      ef.callee_cycle_params = callee_cycle_params;
      ef.self_cycle = in_cycle ? cycle_of[fname] : -1;
      ef.cycle_sel = in_cycle ? std::string("wsel") : std::string();
      ef.reg_defs.clear();
      ef.scan_defs(defs[i].body);
      ef.self_name = defs[i].name;
      ef.self_params = defs[i].ptypes;
      for (const std::string& t : defs[i].ret_types)
        ef.self_rets.push_back((gpu && t == "double") ? std::string("uint64_t") : t);
      ef.self_deep =
          (self_note.count(defs[i].name) && self_note[defs[i].name] == "@deep");
      ef.entry_names = entry_names;
      for (size_t k = 0; k < defs[i].pnames.size(); k++)
        ef.callee_args[defs[i].pnames[k]] =
            in_cycle ? ef.word_read(defs[i].ptypes[k], k) : ("weft_a" + std::to_string(k));
      std::string decls, code;
      bool ok = true;
      size_t q = 0;
      while (q < defs[i].body.size()) {
        // One statement, up to its semicolon.
        size_t semi = defs[i].body.find(';', q);
        if (semi == std::string::npos) semi = defs[i].body.size();
        const std::string line = trim(defs[i].body.substr(q, semi - q));
        q = semi + 1;
        if (line.empty()) continue;
        if (line.rfind(".reg", 0) == 0) {
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
        // A label inside a function.
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
      std::vector<std::string> rtys;
      for (const std::string& t : defs[i].ret_types)
        rtys.push_back((gpu && t == "double") ? std::string("uint64_t") : t);
      const bool suspends = !ef.resume_cases.empty();
      const bool deep = (self_note.count(fname) && self_note[fname] == "@deep");
      if (self_tail[i] && suspends) {
        callee_errs[fname] = "'" + fname +
            "' calls itself in tail position and also stops (a barrier or an atomic): "
            "the loop has to keep its position across the stop, and there is one fixed "
            "state per lane and no stack to keep it in";
        state[fname] = 2;
        return false;
      }
      std::string members = decls;
      for (const auto& m : ef.callee_members)
        members += "  " + m.first + " " + m.second + ";\n";
      for (size_t k = 0; k < rtys.size(); k++)
        members += "  " + rtys[k] + " weft_ret" + std::to_string(k) + ";\n";
      std::string sig;
      if (in_cycle) {
        sig = gpu ? ", thread uint64_t* wa, thread uint32_t* wsel"
                  : ", uint64_t* wa, uint32_t* wsel";
      } else {
        for (size_t k = 0; k < defs[i].pnames.size(); k++)
          sig += ", " + defs[i].ptypes[k] + " weft_a" + std::to_string(k);
      }
      callee_out += "struct weft_cf_" + fname + " {\n  uint32_t pc;\n" + members + "};\n";
      auto make_body = [&](const std::string& cd) {
        std::string b = "  (void)ctx; (void)s;\n";
        if (in_cycle) b += "  (void)wa; (void)wsel;\n";
        for (size_t k = 0; k < rtys.size(); k++)
          b += std::string("  ") + (suspends ? "if (s->pc == 0u) " : "") + "s->weft_ret" +
               std::to_string(k) + " = " + rtys[k] + "();\n";
        if (gpu) {
          if (!suspends) b += "  s->pc = 0u;\n";
          b += "  for (;;) {\n    switch (s->pc) {\n";
          b += ef.resume_cases;
          b += "      case 0u:;\n";
          b += cd;
          b += "      default: return;\n    }\n  }\n";
        } else {
          if (suspends)
            b += "  if (s->pc != 0u) {\n    switch (s->pc) {\n" + ef.resume_cases +
                 "      default: break;\n    }\n  }\n";
          if (self_tail[i]) b += "  for (;;) {\n" + cd + "  }\n";
          else b += cd;
        }
        return b;
      };
      const std::string ctx_decl = gpu ? "thread weft::Ctx& ctx" : "weft::Ctx& ctx";
      const std::string tq = gpu ? "thread " : "";
      if (deep) {
        const int kDepth = 8;
        const bool stops = !ef.resume_cases.empty();
        const std::string keep = ef.resume_cases;
        for (int k = kDepth - 1; k >= 0; k--) {
          std::string cd = code;
          std::string add;
          for (size_t s = 0; s < ef.self_sites.size(); s++) {
            const std::string tok = "WEFTSELF" + std::to_string(s);
            std::string repl;
            if (k + 1 < kDepth) {
              // The next copy's frame is the caller's too, so its place survives a stop.
              const std::string in = "base[" + std::to_string(k + 1) + "]";
              std::string args;
              const std::vector<std::string>& av = ef.self_sites[s];
              for (size_t a = 0; a < av.size(); a++)
                args += ", (" + defs[i].ptypes[a] + ")(" + av[a] + ")";
              std::string body = in + ".pc = 0u; " + "weft_f_" + fname + "_" +
                                 std::to_string(k + 1) + "(ctx, base" + args + ");";
              if (stops) {
                const std::string rid = std::to_string(ef.next_resume++);
                if (!ef.msl) add += "    case " + rid + " : goto R" + rid + ";\n";
                const std::string rlab = ef.msl ? ("case " + rid + ":;") : ("R" + rid + ":;");
                body = in + ".pc = 0u;\n  " + rlab + "\n  weft_f_" + fname + "_" +
                       std::to_string(k + 1) + "(ctx, base" + args + ");\n  if (" + in +
                       ".pc != 0xFFFFFFFFu) { s->pc = " + rid + "u; return; }";
              }
              const std::vector<std::string>& ds = ef.self_site_dsts[s];
              for (size_t a = 0; a < ds.size(); a++)
                body += " " + ef.store(ds[a], in + ".weft_ret" + std::to_string(a),
                                        ef.self_rets[a]);
              repl = "{ " + body + " }";
            } else {
              repl = "{ if (ctx.fault) { ctx.fault[0] = 1u; ctx.fault[1] = "
                     "(uint32_t)ctx.gid; } s->pc = 0xFFFFFFFFu; return; }";
            }
            const size_t at = cd.find(tok);
            if (at != std::string::npos) cd.replace(at, tok.size(), repl);
          }
          ef.resume_cases = keep + add;
          callee_out += "static void weft_f_" + fname + "_" + std::to_string(k) + "(" +
                        ctx_decl + ", " + tq + "weft_cf_" + fname + "* base" + sig +
                        ") {\n  " + tq + "weft_cf_" + fname + "* s = base + " +
                        std::to_string(k) + ";\n" + make_body(cd) + "}\n";
        }
      } else {
        callee_out += "static void weft_f_" + fname + "(" + ctx_decl + ", " + tq +
                      "weft_cf_" + fname + "* s" + sig + ") {\n" + make_body(code) + "}\n";
      }
      if (in_cycle) {
        const int cid = cycle_of[fname];
        cycle_left[cid]--;
        if (cycle_left[cid] == 0) {
          const std::vector<std::string>& mem = cycle_members[cid];
          const std::string id = std::to_string(cid);
          callee_out += "struct weft_cs_" + id + " {\n  uint64_t wa[8];\n  uint32_t wsel;\n";
          for (size_t k = 0; k < mem.size(); k++)
            callee_out += "  weft_cf_" + mem[k] + " f" + std::to_string(k) + ";\n";
          callee_out += "};\n";
          callee_out += "static void weft_tr_" + id + "(" +
                        std::string(gpu ? "thread weft::Ctx& ctx" : "weft::Ctx& ctx") +
                        ", " + tq + "weft_cs_" + id + "* st) {\n";
          callee_out += "  " + tq + "uint64_t* wa = st->wa;\n  for (;;) {\n";
          callee_out += "    if (st->wsel & 0x80000000u) {\n";
          for (size_t k = 0; k < mem.size(); k++)
            callee_out += "      st->f" + std::to_string(k) + ".pc = 0u;\n";
          callee_out += "      st->wsel &= 0x7FFFFFFFu;\n    } else ";
          for (size_t k = 0; k < mem.size(); k++)
            callee_out += std::string(k ? "else if" : "if") + " (st->wsel == " +
                          std::to_string(k) + "u) { weft_f_" + mem[k] + "(ctx, &st->f" +
                          std::to_string(k) + ", wa, &st->wsel); }\n    ";
          callee_out += "else break;\n    if (st->wsel == 0xFFFFFFFFu) break;\n";
          callee_out += "    if (st->wsel & 0x80000000u) return;\n  }\n}\n";
        }
      }
      callee_types[fname] = defs[i].ptypes;
      callee_rets[fname] = rtys;
      callee_suspends[fname] = suspends;
      callee_deep[fname] = deep;
      state[fname] = 1;
      return true;
    };

    // A callee is translated only after the functions it calls.
    for (bool progress = true; progress;) {
      progress = false;
      for (size_t i = 0; i < defs.size(); i++) {
        const std::string& nm = defs[i].name;
        if (!needed.count(nm) || state.count(nm)) continue;
        bool waiting = false;
        std::string that;
        for (const std::string& dep : defs[i].deps) {
          if (cycle_of.count(nm) && cycle_of.count(dep) && cycle_of[nm] == cycle_of[dep])
            continue;
          if (dep == nm) {
            if (self_tail[i]) continue;
            self_note[nm] = "@deep";
            continue;
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
      callee_errs[nm] = "it calls a function that calls it back and has work left "
                        "after that call, which needs one frame per depth: a lane has "
                        "one fixed state and no stack, and this device has no way to "
                        "report a depth that ran out, so the calls are refused by name "
                        "rather than given a silent limit";
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
    em.callee_deep = callee_deep;
    em.entry_names = entry_names;
    em.callee_cycle = cycle_of;
    em.callee_cycle_at = callee_cycle_at;
    em.callee_cycle_params = callee_cycle_params;
    em.tex_index = tex_index;
    em.ntex = ntex;
    em.tex_write.assign((size_t)ntex, false);
    em.tex_array.assign((size_t)ntex, false);
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
        // `.a1d`, `.a2d` and `.acube` all bind an array, and all name the layer first.
        size_t hit = std::string::npos;
        bool cube_hit = false;
        for (const char* m : {".a1d", ".a2d", ".acube"}) {
          const size_t h = src.find(m, at);
          if (h != std::string::npos && h < end && (hit == std::string::npos || h < hit)) {
            hit = h;
            cube_hit = m[2] == 'c';
          }
        }
        if (hit == std::string::npos) break;
        const size_t semi = src.find(';', hit);
        const size_t lb = src.find('[', hit);
        if (lb != std::string::npos && lb < end && (semi == std::string::npos || lb < semi)) {
          size_t e = lb + 1;
          while (e < src.size() && std::isspace((unsigned char)src[e])) e++;
          const size_t b = e;
          while (e < src.size() && is_word(src[e])) e++;
          auto t2 = tex_index.find(src.substr(b, e - b));
          if (t2 != tex_index.end()) {
            const int k = t2->second;
            if ((size_t)k < em.tex_array.size()) em.tex_array[k] = true;
            if (cube_hit && (size_t)k < em.tex_cube.size()) em.tex_cube[k] = true;
          }
        }
        at = hit + 5;
      }
    }
    // A pointer the body never names is not bound.
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

    // How often each register is named in the body, its declaration aside.
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

    body = fuse_ptx_products(body);

    for (const auto& pd : pdecl) {
      if (!pd.ptr) {
        // Scalars keep their declared positions.
        em.scalars[pd.name] = (int)em.scalars.size();
      } else if (mentions(body, pd.name)) {
        em.params[pd.name] = (int)em.params.size();
      } else {
        em.unused.push_back(pd.name);
      }
    }
    std::string shared_decl;

    {
      uint64_t off = 0;
      bool any = false;
      std::vector<std::pair<std::string, bool>> externs;
      const std::string scan = module_head + "\n" + body;
      for (size_t p = 0; (p = scan.find(".shared", p)) != std::string::npos; p += 7) {
        size_t q = p;
        while (q > 0 && (scan[q - 1] == ' ' || scan[q - 1] == '\t')) q--;
        bool decl_start = (q == 0 || scan[q - 1] == '\n' || scan[q - 1] == ';' || scan[q - 1] == '{');
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
        // The bracket is a count of elements, not of bytes.
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
      em.shared_extern = !externs.empty();
      em.shared_bytes = (int)(externs.empty() ? off : off + 48u * 1024u);
    }
    em.shared_size = &shared_decl;

    // Walk the body once per backend.
    auto walk = [&](Emitter& e, std::string& out, std::string& struct_decls) -> bool {
      e.reg_defs.clear();
      e.scan_defs(body);
      std::string stmt, lane_batch;
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

        // A label glued to its instruction.
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
        if (e.vec && !e.vec_control && !e.vec_side && !e.vec_pred.empty())
          line = "if (" + e.vec_pred + ") { " + line + " }";
        if (e.vec) {
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
      for (const auto& m : e.callee_members)
        struct_decls += "  " + m.first + " " + m.second + ";\n";
      for (const auto& m : e.cc_members) struct_decls += "  " + m + "\n";
      return true;
    };

    em.msl = gpu;
    std::string code, struct_decls;
    const bool lane_ok = walk(em, code, struct_decls);

    Emitter emv;
    emv.vec = true;
    emv.callees = callee_types;
    emv.callee_errs = callee_errs;
    emv.entry_names = entry_names;
    emv.callee_rets = callee_rets;
    emv.callee_suspends = callee_suspends;
    emv.callee_cycle = cycle_of;
    emv.callee_cycle_at = callee_cycle_at;
    emv.callee_cycle_params = callee_cycle_params;
    emv.msl = gpu;
    emv.wctx = true;
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

    if (have_vec) {
      for (const auto& pb : emv.param_bytes) {
        auto it = em.param_bytes.find(pb.first);
        if (it == em.param_bytes.end()) em.param_bytes[pb.first] = pb.second;
        else if (it->second != pb.second) it->second = 0;
      }
    }
    const bool warp_only = !lane_ok && em.warp_only && have_vec;
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
      if (have_vec)
        r.error += "; the warp body could not be used either: " +
                   (emv.err.empty() ? std::string("it translated to nothing") : emv.err);
      else if (!emv.err.empty() && emv.err != em.err) r.error += " (warp body: " + emv.err + ")";
      else if (emv.err.empty()) r.error += " (the warp body translated to nothing)";
      return r;
    }

    if (gpu && ((!lane_ok && warp_only) || wide_gpu)) {
      gpu_warp_kernel = true;
      const std::string wst = "weft_ws_" + name;
      const std::string wfn = "weft_gw_" + name;
      const int wnbufs = (int)em.params.size();
      // Threadgroup memory is a `threadgroup` array here, sized from the module.
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
      body_out += "static uint32_t " + wfn + "(thread weft::WCtx& ctx, thread " + wst + "* s,\n";
      body_out += "    thread uint32_t* am, thread uint64_t* am64, thread uint32_t& weft_full,\n";
      body_out += "    threadgroup uint8_t* sh) {\n";
      body_out += "  (void)ctx; (void)sh; (void)am; (void)am64; (void)weft_full;\n";
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
      // The launch the caller asked for, as two numbers.
      body_out += "device uint64_t* weft_geom [[buffer(" + std::to_string(wnbufs + 1) +
                   ")]],\n                 ";
      body_out += "device uint32_t* weft_locks [[buffer(" + std::to_string(wnbufs + 2) +
                   ")]],\n                 ";
      body_out += "uint weft_lid [[thread_index_in_threadgroup]],\n                 ";
      body_out += "uint3 weft_tgid [[threadgroup_position_in_grid]],\n                 ";
      body_out += "uint3 weft_tgsz [[threads_per_threadgroup]],\n                 ";
      body_out += "uint3 weft_grid [[threads_per_grid]]) {\n";
      body_out += "  thread " + wst + " st;\n  thread " + wst + "* s = &st;\n";
      body_out += "  for (uint32_t weft_i = 0; weft_i < (uint32_t)sizeof(st); weft_i++)\n";
      body_out += "    ((thread uint8_t*)&st)[weft_i] = 0u;\n";
      body_out += "  for (uint32_t l = 0; l < weft::kWarpLanes; l++) s->pc[l] = 0u;\n";
      body_out += "  weft::WCtx ctx;\n";
      // One thread is one warp.
      body_out += "  ctx.lid0 = weft_lid * weft::kWarpLanes;\n";
      body_out += "  ctx.tgid = weft_tgid.x;\n";
      body_out += "  ctx.tg_size = (uint32_t)weft_geom[1];\n";
      body_out += "  ctx.grid = (uint32_t)weft_geom[0];\n";
      body_out += "  for (uint32_t l = 0; l < weft::kWarpLanes; l++)\n";
      body_out += "    s->pc[l] = ((ctx.lid0 + l) < "
                  "weft::live_threads(ctx.tgid, ctx.tg_size, ctx.grid)) ? 0u : "
                  "0xFFFFFFFFu;\n";

      for (int i = 0; i < wnbufs; i++)
        body_out += "  ctx.b[" + std::to_string(i) + "] = weft_b" + std::to_string(i) + ";\n";
      body_out += "  ctx.params = weft_params;\n";
      body_out += "  ctx.locks = weft_locks;\n";
      body_out += "  ctx.fault = weft_locks;\n";
      body_out += "  threadgroup uint8_t weft_sh[" +
                   std::to_string(arena > 0 ? arena : 1) + "];\n";
      body_out += "  threadgroup uint32_t weft_red_slots[1024];\n";
      body_out += "  if (weft::live_threads(ctx.tgid, ctx.tg_size, ctx.grid) < ctx.tg_size)\n";
      body_out += "    for (uint32_t weft_i = 0; weft_i < (uint32_t)sizeof(weft_sh); weft_i++)\n";
      body_out += "      weft_sh[weft_i] = 0u;\n";
      body_out += "  threadgroup uint8_t* sh = weft_sh;\n  (void)sh;\n";
      body_out += "  ctx.red = weft_red_slots;\n";
      if (emv.uses_red)
        body_out += "  for (uint32_t weft_i = 0; weft_i < 1024u; weft_i++) weft_red_slots[weft_i] = 0u;\n";
      body_out += "  uint32_t am[32];\n  uint64_t am64[32];\n  uint32_t weft_full = 0;\n";
      body_out += "  (void)am; (void)am64; (void)weft_full;\n";
        body_out += "  for (;;) {\n";
        body_out += "    if (" + wfn +
                     "(ctx, s, am, am64, weft_full, weft_sh) == weft::kWarpDone) return;\n";
        body_out += "  }\n}\n\n";
    } else if (lane_ok && gpu) {
      const std::string st = "weft_st_" + name;
      body_out += "struct " + st + " {\n  uint32_t pc;\n" + struct_decls + "};\n";
      const int nbufs = (int)em.params.size();
      body_out += "kernel void " + name + "(";
      for (int i = 0; i < nbufs; i++)
        body_out += "device uint8_t* weft_b" + std::to_string(i) + " [[buffer(" +
                    std::to_string(i) + ")]],\n                 ";
      body_out += "device uint64_t* weft_params [[buffer(" + std::to_string(nbufs) +
                   ")]],\n                 ";
      body_out += "device uint32_t* weft_fault [[buffer(" + std::to_string(nbufs + 1) +
                   ")]],\n                 ";

      // An image is a binding, and on the GPU a binding is a kernel parameter.
      for (int i = 0; i < em.ntex; i++) {
        const bool rw = (size_t)i < em.tex_write.size() && em.tex_write[i];
        const bool arr = (size_t)i < em.tex_array.size() && em.tex_array[i];
        const bool vol = (size_t)i < em.tex_3d.size() && em.tex_3d[i];
        const bool cb = (size_t)i < em.tex_cube.size() && em.tex_cube[i];
        const std::string shape =
            cb ? (arr ? "texturecube_array<float, access::" : "texturecube<float, access::")
            : vol ? "texture3d<float, access::"
                 : arr ? "texture2d_array<float, access::" : "texture2d<float, access::";
        const std::string ni = std::to_string(i);
        body_out += shape + (rw ? "read_write" : "sample") + "> weft_t" + ni + " [[texture(" +
                    ni + ")]],\n                 ";
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
      body_out += "  ctx.params = weft_params;\n";
      body_out += "  ctx.fault = weft_fault;\n  (void)ctx;\n";
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
      head += "static uint32_t weft_g_" + name +
              "(weft::Ctx* ctxs, unsigned char* states, size_t stride, uint32_t live,\n"
              " uint32_t weft_pass, void* const* bufs, unsigned char* sh) {\n";
      head += "  uint32_t weft_left = 0;\n";
      head += "  for (uint32_t i = 0; i < live; i++) {\n";
      head += "    " + st + "* s = (" + st + "*)(states + (size_t)i * stride);\n";
      head += "    if (s->pc == weft::kThreadDone) continue;\n";
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
      head += "static void weft_k_" + name +
              "(weft::Ctx& ctx, void* stp, void* const* bufs, unsigned char* sh) {\n";
      head += "  weft_g_" + name + "(&ctx, (unsigned char*)stp, 0, 1, ctx.pass, bufs, sh);\n}\n\n";
      body_out += head;
    }

    if (have_vec && !gpu) {
      const std::string ws = "weft_ws_" + name;
      body_out += "struct " + ws + " {\n  uint32_t pc[32];\n" + decls_v + "};\n";
      body_out += "static uint32_t weft_w_" + name + "(weft::WCtx& ctx, void* stp) {\n";
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
    for (const auto& pd : pdecl) {
      TranslatedKernel::Param gp;
      gp.type = pd.type;
      auto b = em.params.find(pd.name);
      if (b != em.params.end()) gp.buf = b->second;
      auto sl = em.scalars.find(pd.name);
      if (sl != em.scalars.end()) gp.scalar = sl->second;
      tk.params.push_back(gp);
    }
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
    // What each binding is, by name.
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
    std::string g = "// generated by weft (ptx front end, metal target), do not edit\n";
    g += kWWeftMslPreamble;
    g += kWWeftWarpMsl;
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
      "using half = _Float16;\n"
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
      "static inline uint32_t weft_bitrev32(uint32_t v) {\n"
      "  v = ((v >> 1) & 0x55555555u) | ((v & 0x55555555u) << 1);\n"
      "  v = ((v >> 2) & 0x33333333u) | ((v & 0x33333333u) << 2);\n"
      "  v = ((v >> 4) & 0x0F0F0F0Fu) | ((v & 0x0F0F0F0Fu) << 4);\n"
      "  v = ((v >> 8) & 0x00FF00FFu) | ((v & 0x00FF00FFu) << 8);\n"
      "  return (v >> 16) | (v << 16);\n"
      "}\n"
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
      "static inline uint32_t weft_f16x2(float hi, float lo) {\n"
      "  return (uint32_t)weft_h16_bits(lo) | ((uint32_t)weft_h16_bits(hi) << 16);\n"
      "}\n"
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
      // A carry out of the mantissa is a carry into the exponent.
      "  const uint32_t out = ((uint32_t)E << 3) + r;\n"
      "  return (uint8_t)(sign | (uint8_t)(out >= 0x7Fu ? 0x7Eu : out));\n"
      "}\n"
      "static inline uint32_t weft_e4m3x2(float hi, float lo) {\n"
      "  return (uint32_t)weft_e4m3_bits(lo) | ((uint32_t)weft_e4m3_bits(hi) << 8);\n"
      "}\n"
      "static inline float weft_e4m3(uint32_t b, int i) {\n"
      "  const uint32_t v = (b >> (8 * i)) & 0xFFu;\n"
      "  const uint32_t mag = v & 7u, E = (v >> 3) & 0xFu;\n"
      "  const uint32_t sb = (E + 117u) << 23;\n"
      "  float sc; std::memcpy(&sc, &sb, 4);\n"
      "  const float d = (E == 0) ? (float)mag / 512.0f : (float)(8u + mag) * sc;\n"
      "  return (float)((v & 0x80u) ? -d : d);\n"
      "}\n"
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
      "static inline uint32_t weft_mx_bits(float x, int eb, int mb) {\n"
      "  const uint32_t maxbits = (uint32_t)((((1u << eb) - 1u) << mb) | ((1u << mb) - 1u));\n"
      "  uint32_t u; std::memcpy(&u, &x, 4);\n"
      "  const uint32_t sign = u >> 31;\n"
      "  if (!std::isfinite(x)) return (x != x) ? maxbits : ((sign << (eb + mb)) | maxbits);\n"
      "  const uint32_t mag = u & 0x7FFFFFFFu;\n"
      "  if (mag == 0u) return sign << (eb + mb);\n"
      "  const int bias = (1 << (eb - 1)) - 1;\n"
      "  const int E = (int)((mag >> 23) & 0xFFu) - 127 + bias;\n"
      "  if (E >= (1 << eb)) return (sign << (eb + mb)) | maxbits;\n"
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
      "  if (out >= maxbits) out = maxbits;\n"
      "  return (sign << (eb + mb)) | out;\n"
      "}\n"
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
      "static inline float weft_mx_decode(uint32_t v, int eb, int mb);\n"
      "static inline float weft_e5m2(uint32_t b, int i);\n"
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
      "static inline uint32_t weft_pack2(uint32_t hi, uint32_t lo, int bits) {\n"
      "  return lo | (hi << bits);\n"
      "}\n"
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
      // The same bytes with an integer accumulator.
      "static inline int32_t weft_s8_i(uint32_t b, int i) {\n"
      "  return (int32_t)(int8_t)(uint8_t)((b >> (8 * i)) & 0xFFu);\n"
      "}\n"
      "static inline int32_t weft_u8_i(uint32_t b, int i) {\n"
      "  return (int32_t)((b >> (8 * i)) & 0xFFu);\n"
      "}\n"
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
      "static inline float weft_mx_decode(uint32_t v, int eb, int mb) {\n"
      "  const int bias = (1 << (eb - 1)) - 1;\n"
      "  const uint32_t mant = v & ((1u << mb) - 1u);\n"
      "  const uint32_t E = (v >> mb) & ((1u << eb) - 1u);\n"
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
      "static inline float weft_mx_get(uint32_t w, int i, int code) {\n"
      "  if (code == 0) return weft_mx_decode((w >> (4 * i)) & 0xFu, 2, 1);\n"
      "  if (code == 1) return weft_mx_decode((w >> (8 * i)) & 0x3Fu, 2, 3);\n"
      "  if (code == 2) return weft_mx_decode((w >> (8 * i)) & 0x3Fu, 3, 2);\n"
      "  if (code == 3) return weft_e4m3(w, i);\n"
      "  if (code == 4) return weft_e5m2(w, i);\n"
      "  if (code == 6) return weft_mx_decode((w >> (8 * i)) & 0xFu, 2, 1);\n"
      "  const uint32_t v = (w >> (8 * i)) & 0xFFu;\n"
      "  if (v == 0xFFu) return NAN;\n"
      "  const uint32_t sb = v == 0u ? 0x00400000u : (v << 23);\n"
      "  float f; std::memcpy(&f, &sb, 4); return f;\n"
      "}\n"
      "static inline uint32_t weft_f8_bits(float x, int code, int satfinite) {\n"
      "  if (code == 3) {\n"
      "    if (!satfinite && std::isfinite(x) && std::fabs(x) > 448.0f) return 0x7Fu;\n"
      "    return weft_e4m3_bits(x);\n"
      "  }\n"
      "  return weft_e5m2_bits(x);\n"
      "}\n"
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
      "static inline uint32_t weft_shf_l(uint32_t a, uint32_t b, uint32_t c, int clamp) {\n"
      "  if (clamp) { if (c >= 32u) return a; } else c &= 31u;\n"
      "  return c ? ((b << c) | (a >> (32u - c))) : b;\n"
      "}\n"
      "static inline uint32_t weft_shf_r(uint32_t a, uint32_t b, uint32_t c, int clamp) {\n"
      "  if (clamp) { if (c >= 32u) return b; } else c &= 31u;\n"
      "  return c ? ((a >> c) | (b << (32u - c))) : a;\n"
      "}\n"
      "static inline uint32_t weft_lop3(uint32_t a, uint32_t b, uint32_t c, uint32_t lut) {\n"
      "  uint32_t out = 0;\n"
      "  for (uint32_t i = 0; i < 8u; i++) {\n"
      "    const uint32_t m = ((i & 4u) ? a : ~a) & ((i & 2u) ? b : ~b) & ((i & 1u) ? c : ~c);\n"
      "    if (lut & (1u << i)) out |= m;\n"
      "  }\n"
      "  return out;\n"
      "}\n"
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
      // mul.hi's high word, from four 32x32 multiplies.
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
  // A width the caller states wins over anything this front end could infer.
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
