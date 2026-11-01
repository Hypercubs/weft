#include "translate.hpp"
#include "soft_math.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <utility>

namespace {

bool is_ident(char c) { return std::isalnum((unsigned char)c) || c == '_'; }

std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace((unsigned char)s[a])) a++;
  while (b > a && std::isspace((unsigned char)s[b - 1])) b--;
  return s.substr(a, b - a);
}

bool starts_with(const std::string& s, const char* p) {
  return s.rfind(p, 0) == 0;
}

size_t find_matching(const std::string& s, size_t open, char o, char c) {
  int depth = 0;
  for (size_t i = open; i < s.size(); i++) {
    if (s[i] == o) depth++;
    else if (s[i] == c) {
      depth--;
      if (depth == 0) return i;
    }
  }
  return std::string::npos;
}

// Remove a construct's attribute list `[[...]]` from a declaration.
std::string strip_attributes(const std::string& decl, std::vector<std::string>& attrs) {
  std::string out;
  size_t i = 0;
  while (i < decl.size()) {
    if (i + 1 < decl.size() && decl[i] == '[' && decl[i + 1] == '[') {
      size_t close = decl.find("]]", i + 2);
      if (close == std::string::npos) break;
      attrs.push_back(decl.substr(i + 2, close - i - 2));
      i = close + 2;
      continue;
    }
    out.push_back(decl[i]);
    i++;
  }
  return trim(out);
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
  if (from.empty()) return s;
  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.size(), to);
    pos += to.size();
  }
  return s;
}

bool contains_word(const std::string& s, const std::string& word) {
  size_t i = 0;
  while ((i = s.find(word, i)) != std::string::npos) {
    bool left = (i == 0) || !is_ident(s[i - 1]);
    bool right = (i + word.size() >= s.size()) || !is_ident(s[i + word.size()]);
    if (left && right) return true;
    i += word.size();
  }
  return false;
}

// Comments are dropped before anything else: a kernel that mentions
// `return` in a comment should not lose its loop form.
std::string strip_comments(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    if (s[i] == '"') {
      out.push_back(s[i++]);
      while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\' && i + 1 < s.size()) out.push_back(s[i++]);
        out.push_back(s[i++]);
      }
      if (i < s.size()) out.push_back(s[i++]);
      continue;
    }
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

// Split a translation unit into top-level items: a `struct { ... };`, a function
// `f(...) { ... }`, a `#define` line, a lone declaration. Everything a kernel
// file defines for itself lives in one of these, and none of it belongs to a
// kernel — it has to be carried over to the generated C++ too, or a kernel that
// names its own struct type compiles on the GPU and not here.
std::vector<std::string> top_level_items(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0, start = 0;
  int depth = 0;
  bool line_start = true;
  auto push = [&](size_t end) {
    std::string item = trim(s.substr(start, end - start));
    if (!item.empty()) out.push_back(item);
  };
  while (i < s.size()) {
    const char c = s[i];
    // A preprocessor line is an item of its own, terminated by the newline.
    if (c == '#' && depth == 0 && line_start) {
      size_t e = s.find('\n', i);
      if (e == std::string::npos) e = s.size();
      push(i);
      out.push_back(trim(s.substr(i, e - i)));
      i = e + 1;
      start = i;
      line_start = true;
      continue;
    }
    if (c == '{') depth++;
    else if (c == '}') {
      depth--;
      if (depth == 0) {
        // A `};` belongs to the item it closes, so a struct keeps its semicolon.
        size_t e = i + 1;
        size_t k = e;
        while (k < s.size() && std::isspace((unsigned char)s[k])) k++;
        if (k < s.size() && s[k] == ';') e = k + 1;
        push(e);
        i = e;
        start = i;
        line_start = true;
        continue;
      }
    } else if (c == ';' && depth == 0) {
      push(i + 1);
      i++;
      start = i;
      line_start = true;
      continue;
    }
    if (c == '\n') line_start = true;
    else if (!std::isspace((unsigned char)c)) line_start = false;
    i++;
  }
  push(s.size());
  return out;
}

std::vector<std::string> split_top_level(const std::string& s) {
  std::vector<std::string> out;
  int paren = 0, angle = 0, bracket = 0;
  size_t start = 0;
  for (size_t i = 0; i < s.size(); i++) {
    char c = s[i];
    if (c == '(') paren++;
    else if (c == ')') paren--;
    // `>` only closes a template argument list, and `a >> b` is a shift: without
    // the guard the shift makes the angle depth negative and no comma after it
    // splits any more.
    else if (c == '<') angle++;
    else if (c == '>' && angle > 0) angle--;
    else if (c == '[') bracket++;
    else if (c == ']') bracket--;
    else if (c == ',' && paren == 0 && angle == 0 && bracket == 0) {
      out.push_back(trim(s.substr(start, i - start)));
      start = i + 1;
    }
  }
  std::string last = trim(s.substr(start));
  if (!last.empty()) out.push_back(last);
  return out;
}

// `threadgroup float tile[256]` -> a handle onto this threadgroup's shared block.
std::string rewrite_shared_decls(const std::string& body, bool& found, std::string& err) {
  std::string out;
  size_t i = 0;
  while (i < body.size()) {
    if (body.compare(i, 11, "threadgroup") == 0 &&
        (i == 0 || !is_ident(body[i - 1])) &&
        (i + 11 >= body.size() || !is_ident(body[i + 11]))) {
      size_t j = i + 11;
      bool ok = true;

      auto skip_ws = [&](size_t& k) { while (k < body.size() && std::isspace((unsigned char)body[k])) k++; };
      auto read_ident = [&](size_t& k) {
        size_t s = k;
        while (k < body.size() && is_ident(body[k])) k++;
        return body.substr(s, k - s);
      };

      skip_ws(j);
      std::string type = read_ident(j);
      if (type.empty()) ok = false;
      skip_ws(j);
      std::string name = read_ident(j);
      if (name.empty()) ok = false;
      skip_ws(j);
      if (j >= body.size() || body[j] != '[') ok = false;
      size_t count = 0;
      if (ok) {
        size_t close = body.find(']', j);
        if (close == std::string::npos) ok = false;
        else {
          std::string num = trim(body.substr(j + 1, close - j - 1));
          if (num.empty() || num.find_first_not_of("0123456789") != std::string::npos) {
            err = "threadgroup array '" + name + "' needs a numeric size, got '" + num + "'";
            return body;
          }
          count = (size_t)std::strtoull(num.c_str(), nullptr, 10);
          j = close + 1;
        }
      }
      if (!ok) { out.push_back(body[i]); i++; continue; }

      skip_ws(j);
      if (j < body.size() && body[j] == ';') j++;
      out += type + "* " + name + " = weft::shared_mem<" + type + ", " + std::to_string(count) +
             ">(ctx);";
      found = true;
      i = j;
      continue;
    }
    out.push_back(body[i]);
    i++;
  }
  return out;
}

std::string rewrite_barriers(const std::string& body, bool& found, std::string& err) {
  std::string out;
  size_t i = 0;
  while (i < body.size()) {
    // simdgroup_barrier waits for one warp; a threadgroup barrier is a superset
    // of that, so waiting for the whole group is sound (it is what the hardware
    // does for a workgroup barrier) and costs the same here.
    static const char* kNames[] = {"threadgroup_barrier", "simdgroup_barrier"};
    const char* bar = nullptr;
    size_t blen = 0;
    for (const char* n : kNames) {
      const size_t len = std::strlen(n);
      if (body.compare(i, len, n) == 0 && (i == 0 || !is_ident(body[i - 1]))) {
        bar = n;
        blen = len;
        break;
      }
    }
    if (bar) {
      size_t j = i + blen;
      while (j < body.size() && std::isspace((unsigned char)body[j])) j++;
      if (j >= body.size() || body[j] != '(') { out.push_back(body[i]); i++; continue; }
      size_t close = find_matching(body, j, '(', ')');
      if (close == std::string::npos) { err = "unbalanced threadgroup_barrier"; return body; }
      j = close + 1;
      while (j < body.size() && std::isspace((unsigned char)body[j])) j++;
      if (j < body.size() && body[j] == ';') j++;
      out += "weft::barrier(ctx);";
      found = true;
      i = j;
      continue;
    }
    out.push_back(body[i]);
    i++;
  }
  return out;
}

// simd_shuffle_*(value, delta) -> an awaitable, for the same reason a barrier is
// one: a warp shuffle is a rendezvous, and in a coroutine a rendezvous is a
// suspension. The lane's value is published before suspending and the peer's
// slot is read when it is resumed.
// simd_ballot / simd_all / simd_any: one predicate in, one warp-wide answer
// out, so they are the shuffle rendezvous with a one-bit payload.
// as_type<T>(x): Metal's bitcast. The conversion is a memcpy of the same bits,
// which is exactly what weft_bitcast does, so the whole call is rewritten into
// one rather than needing a type system here.
std::string rewrite_as_type(const std::string& body, std::string& err) {
  std::string out = body;
  size_t pos = 0;
  while ((pos = out.find("as_type", pos)) != std::string::npos) {
    if (pos > 0 && is_ident(out[pos - 1])) { pos += 7; continue; }
    size_t j = pos + 7;
    if (j >= out.size() || out[j] != '<') { pos = j; continue; }
    size_t k = j + 1;
    int depth = 1;
    while (k < out.size() && depth > 0) {
      if (out[k] == '<') depth++;
      else if (out[k] == '>') depth--;
      k++;
    }
    if (depth != 0) { err = "unbalanced as_type<>"; return body; }
    const std::string ty = trim(out.substr(j + 1, k - j - 2));
    while (k < out.size() && std::isspace((unsigned char)out[k])) k++;
    if (k >= out.size() || out[k] != '(') { pos = k; continue; }
    const size_t close = find_matching(out, k, '(', ')');
    if (close == std::string::npos) { err = "unbalanced as_type(...)"; return body; }
    const std::string repl =
        "weft_bitcast<" + ty + ">(" + out.substr(k + 1, close - k - 1) + ")";
    out = out.substr(0, pos) + repl + out.substr(close + 1);
    pos += repl.size();
  }
  return out;
}

// Rename every whole-word occurrence of `from` to `to`.
std::string rename_word(const std::string& s, const std::string& from, const std::string& to) {
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    const bool hit = s.compare(i, from.size(), from) == 0 && (i == 0 || !is_ident(s[i - 1])) &&
                     (i + from.size() >= s.size() || !is_ident(s[i + from.size()]));
    if (hit) {
      out += to;
      i += from.size();
      continue;
    }
    out.push_back(s[i]);
    i++;
  }
  return out;
}

// A helper function the kernel file defines for itself. Most of them are
// translated as written; one that waits on its warp or its threadgroup cannot be,
// because the wait is a suspension of the *caller's* coroutine — so those are
// expanded at each call site instead, as a statement expression. That keeps the
// rendezvous inside the kernel body where the resume point lives.
struct HelperFn {
  std::string name, ret, body;
  std::vector<std::pair<std::string, std::string>> params;  // type, name
  bool suspends = false;
};

// Is this the start of a function definition (`T name(...) { ... }`)?
bool looks_like_function(const std::string& item, size_t& name_start, size_t& paren) {
  if (starts_with(item, "struct") || starts_with(item, "class") || starts_with(item, "union") ||
      starts_with(item, "enum") || starts_with(item, "typedef") || starts_with(item, "using") ||
      starts_with(item, "namespace")) {
    return false;
  }
  const size_t brace = item.find('{');
  if (brace == std::string::npos) return false;
  size_t i = 0;
  int angle = 0, bracket = 0;
  for (; i < brace; i++) {
    const char c = item[i];
    if (c == '<') angle++;
    else if (c == '>' && angle > 0) angle--;
    else if (c == '[') bracket++;
    else if (c == ']' && bracket > 0) bracket--;
    else if (c == '(' && angle == 0 && bracket == 0) break;
  }
  if (i >= brace) return false;
  paren = i;
  size_t n = i;
  while (n > 0 && std::isspace((unsigned char)item[n - 1])) n--;
  size_t b = n;
  while (b > 0 && is_ident(item[b - 1])) b--;
  if (b == n) return false;
  // `if (...) {` and the like are not definitions: a keyword is not a name.
  const std::string word = item.substr(b, n - b);
  static const char* kKeywords[] = {"if",     "for",    "while", "switch", "return",
                                    "sizeof", "catch",  "do"};
  for (const char* k : kKeywords)
    if (word == k) return false;
  name_start = b;
  return true;
}

HelperFn parse_helper(const std::string& item, size_t name_start, size_t paren) {
  HelperFn f;
  f.name = item.substr(name_start, paren - name_start);
  while (name_start > 0 && std::isspace((unsigned char)item[name_start - 1])) name_start--;
  f.ret = trim(item.substr(0, name_start));
  for (const char* q : {"inline ", "static ", "constexpr ", "__attribute__((always_inline)) "}) {
    if (starts_with(f.ret, q)) f.ret = trim(f.ret.substr(std::strlen(q)));
  }
  const size_t close = find_matching(item, paren, '(', ')');
  if (close == std::string::npos) return f;
  for (const std::string& p : split_top_level(item.substr(paren + 1, close - paren - 1))) {
    if (p.empty() || p == "void") continue;
    std::string decl = p;
    std::vector<std::string> attrs;
    decl = strip_attributes(decl, attrs);
    // A parameter's type names the address space it lives in — `threadgroup
    // float* tile` — but an expanded helper is handed an argument that already
    // names the storage, so the qualifier has to go or it lands in the C++.
    for (bool again = true; again;) {
      again = false;
      for (const char* space : {"threadgroup ", "device ", "constant ", "thread "}) {
        const size_t n = std::strlen(space);
        if (decl.compare(0, n, space) == 0) {
          decl = trim(decl.substr(n));
          again = true;
        }
      }
    }
    size_t n = decl.size();
    while (n > 0 && std::isspace((unsigned char)decl[n - 1])) n--;
    size_t b = n;
    while (b > 0 && is_ident(decl[b - 1])) b--;
    f.params.emplace_back(trim(decl.substr(0, b)), decl.substr(b, n - b));
  }
  const size_t brace = item.find('{', close);
  if (brace == std::string::npos) return f;
  const size_t end = find_matching(item, brace, '{', '}');
  if (end == std::string::npos) return f;
  f.body = item.substr(brace + 1, end - brace - 1);
  for (const char* pre : {"simd_", "quad_", "threadgroup_barrier", "simdgroup_barrier"}) {
    if (f.body.find(pre) != std::string::npos) f.suspends = true;
  }
  return f;
}

// `return` -> `{ r = expr; goto end; }`, so an expanded helper can hand its
// value back through the statement expression.
std::string helper_returns(const std::string& body, const std::string& retname) {
  std::string out;
  size_t i = 0;
  while (i < body.size()) {
    const bool rel = body.compare(i, 6, "return") == 0 && (i == 0 || !is_ident(body[i - 1])) &&
                     (i + 6 >= body.size() || !is_ident(body[i + 6]));
    if (!rel) { out.push_back(body[i]); i++; continue; }
    size_t j = i + 6;
    int paren = 0, bracket = 0;
    while (j < body.size()) {
      const char c = body[j];
      if (c == '(') paren++;
      else if (c == ')') paren--;
      else if (c == '[') bracket++;
      else if (c == ']') bracket--;
      else if (c == ';' && paren == 0 && bracket == 0) break;
      j++;
    }
    const std::string expr = trim(body.substr(i + 6, j - i - 6));
    if (expr.empty()) out += "goto __weft_end;";
    else out += "{ " + retname + " = (" + expr + "); goto __weft_end; }";
    i = (j < body.size()) ? j + 1 : body.size();
  }
  return out;
}

// Replace `name(args)` with the helper's body, bound to fresh locals.
std::string expand_helper_calls(const std::string& in, const HelperFn& f, bool& hit,
                                std::string& err) {
  std::string out;
  size_t i = 0;
  while (i < in.size()) {
    if (in.compare(i, f.name.size(), f.name) != 0 || (i > 0 && is_ident(in[i - 1])) ||
        (i + f.name.size() < in.size() && is_ident(in[i + f.name.size()]))) {
      out.push_back(in[i]);
      i++;
      continue;
    }
    size_t j = i + f.name.size();
    while (j < in.size() && std::isspace((unsigned char)in[j])) j++;
    if (j >= in.size() || in[j] != '(') { out.push_back(in[i]); i++; continue; }
    const size_t close = find_matching(in, j, '(', ')');
    if (close == std::string::npos) { err = "unbalanced call to " + f.name; return in; }
    std::vector<std::string> args = split_top_level(in.substr(j + 1, close - j - 1));
    if (args.size() != f.params.size()) {
      err = "helper '" + f.name + "' takes " + std::to_string(f.params.size()) +
            " arguments, called with " + std::to_string(args.size());
      return in;
    }
    std::string body = f.body;
    for (size_t k = 0; k < f.params.size(); k++) body = rename_word(body, f.params[k].second, "__weft_a" + std::to_string(k));
    body = helper_returns(body, "__weft_r");

    std::string exp = "({ ";
    if (f.ret != "void") exp += f.ret + " __weft_r{}; ";
    for (size_t k = 0; k < f.params.size(); k++) {
      exp += f.params[k].first + (f.params[k].first.empty() ? "" : " ") + "__weft_a" +
             std::to_string(k) + " = (" + trim(args[k]) + "); ";
    }
    exp += "{ " + body + " } __weft_end: ";
    exp += (f.ret == "void") ? "(void)0; }) " : "__weft_r; }) ";
    out += exp;
    hit = true;
    i = close + 1;
  }
  return out;
}

// Everything a kernel file declares for itself. Plain declarations and functions
// are carried over verbatim; a function that waits on its warp is kept aside so
// its calls can be expanded into the kernels that make them.
struct CarriedDecls {
  std::string text;
  std::vector<HelperFn> helpers;
};

CarriedDecls carry_decls(const std::string& src) {
  CarriedDecls c;
  for (const std::string& item : top_level_items(src)) {
    if (item.empty()) continue;
    if (starts_with(item, "kernel")) continue;    // rewritten as a kernel body
    if (starts_with(item, "fragment")) continue;  // rewritten as a per-pixel body
    if (starts_with(item, "#include") || starts_with(item, "#pragma")) continue;
    // The other pipeline stages are not compute: the vertex function is supplied
    // by the launcher, and a tile function never gets this far — a render pass
    // that has one is refused before any kernel is translated.
    if (starts_with(item, "vertex")) continue;
    if (item.compare(0, 8, "[[tile]]") == 0) continue;
    size_t ns = 0, par = 0;
    if (looks_like_function(item, ns, par)) {
      HelperFn f = parse_helper(item, ns, par);
      // A function that names a threadgroup waits on it, owns storage in it, or
      // both — and either way the block belongs to the kernel, so the body has to
      // be inside the kernel body, where the shared window and the resume point
      // are. Those are expanded at each call site. That makes the body an
      // expression, so it needs a value to hand back.
      const bool shares = f.body.find("threadgroup") != std::string::npos;
      if ((f.suspends || shares) && !f.ret.empty() && !f.body.empty()) {
        c.helpers.push_back(f);
        continue;
      }
    }
    c.text += rewrite_math_calls(item) + "\n\n";
  }
  // A suspending helper may call another: expand until the calls are gone. A
  // recursive one cannot be expanded and is left to fail as a call to an
  // undefined function, which names it.
  for (int pass = 0; pass < 8; pass++) {
    bool any = false;
    for (auto& dst : c.helpers) {
      for (const auto& src : c.helpers) {
        if (&dst == &src) continue;
        bool hit = false;
        std::string err;
        dst.body = expand_helper_calls(dst.body, src, hit, err);
        if (hit) any = true;
      }
    }
    if (!any) break;
  }
  return c;
}

// A texture argument: `texture2d<float>` or `texture2d<uint, access::read>`.
struct TexParam {
  std::string name;
  std::string value;  // "float", "uint", "int" or "half"
};

// `constexpr sampler s(filter::nearest, address::clamp_to_edge);` is a compile
// time constant on the GPU, so it is parsed and removed here and its two
// meaningful choices are baked into the sample calls.
std::string strip_samplers(const std::string& in, std::map<std::string, std::pair<int, int>>& out,
                           std::string& err) {
  std::string body = in;
  size_t p = 0;
  while ((p = body.find("sampler", p)) != std::string::npos) {
    if (p > 0 && is_ident(body[p - 1])) { p += 7; continue; }
    size_t e = p + 7;
    while (e < body.size() && std::isspace((unsigned char)body[e])) e++;
    if (e >= body.size() || !is_ident(body[e])) { p += 7; continue; }
    const size_t ns = e;
    while (e < body.size() && is_ident(body[e])) e++;
    const std::string name = body.substr(ns, e - ns);
    size_t q = e;
    while (q < body.size() && std::isspace((unsigned char)body[q])) q++;
    if (q >= body.size() || body[q] != '(') { p = e; continue; }
    const size_t close = find_matching(body, q, '(', ')');
    if (close == std::string::npos) { err = "unbalanced sampler declaration"; return in; }
    const std::string args = body.substr(q + 1, close - q - 1);
    int filter = 0, addr = 0;
    if (args.find("filter::linear") != std::string::npos) filter = 1;
    if (args.find("address::clamp_to_zero") != std::string::npos) addr = 1;
    else if (args.find("address::repeat") != std::string::npos) addr = 2;
    out[name] = {filter, addr};

    // Remove the whole declaration, qualifiers and semicolon included: its
    // start is the statement boundary before it.
    size_t start = p;
    while (start > 0 && body[start - 1] != ';' && body[start - 1] != '{' && body[start - 1] != '}')
      start--;
    size_t end = close + 1;
    while (end < body.size() && std::isspace((unsigned char)body[end])) end++;
    if (end < body.size() && body[end] == ';') end++;
    body.erase(start, end - start);
    p = start;
  }
  return body;
}

// Texture reads and samples, and the three size queries.
std::string rewrite_textures(const std::string& in, const std::vector<TexParam>& texs,
                             const std::map<std::string, std::pair<int, int>>& samplers,
                             std::string& err) {
  std::string out;
  size_t i = 0;
  while (i < in.size()) {
    bool hit = false;
    for (const auto& t : texs) {
      const size_t len = t.name.size();
      if (in.compare(i, len, t.name) != 0) continue;
      if (i > 0 && is_ident(in[i - 1])) continue;
      size_t j = i + len;
      if (j < in.size() && is_ident(in[j])) continue;
      size_t k = j;
      while (k < in.size() && std::isspace((unsigned char)in[k])) k++;
      if (k >= in.size() || in[k] != '.') continue;
      k++;
      const size_t ms = k;
      while (k < in.size() && is_ident(in[k])) k++;
      const std::string member = in.substr(ms, k - ms);
      size_t p = k;
      while (p < in.size() && std::isspace((unsigned char)in[p])) p++;
      if (p >= in.size() || in[p] != '(') continue;
      const size_t close = find_matching(in, p, '(', ')');
      if (close == std::string::npos) { err = "unbalanced texture ." + member; return in; }

      if (member == "get_width") { out += "(" + t.name + ".width)"; i = close + 1; hit = true; break; }
      if (member == "get_height") { out += "(" + t.name + ".height)"; i = close + 1; hit = true; break; }
      if (member == "get_array_size" || member == "get_num_mip_levels") {
        out += "(1u)";
        i = close + 1;
        hit = true;
        break;
      }

      auto args = split_top_level(in.substr(p + 1, close - p - 1));
      // Metal's texture reads always hand back four channels: a single-channel
      // format comes back as (value, 0, 0, 1), and kernels take `.x` of it. That
      // shape matters — `float v = img.read(...)` does not compile on the GPU
      // either.
      const std::string vec = (t.value == "float")   ? "float4"
                              : (t.value == "half") ? "half4"
                              : (t.value == "int")  ? "int4"
                                                    : "uint4";
      const std::string zero = (t.value == "float" || t.value == "half") ? "0" : "0u";
      const std::string one = (t.value == "float" || t.value == "half") ? "1" : "1u";
      if (member == "read") {
        if (args.size() != 1) { err = "texture read takes one coordinate"; return in; }
        const std::string c = trim(args[0]);
        out += vec + "{weft::tex_read<" + t.value + ">(" + t.name + ", (" + c + ").x, (" + c +
               ").y), " + zero + ", " + zero + ", " + one + "}";
        i = close + 1;
        hit = true;
        break;
      }
      if (member == "sample") {
        if (args.size() != 2) { err = "texture sample takes a sampler and a coordinate"; return in; }
        if (t.value != "float" && t.value != "half") {
          err = "sample() of a " + t.value + " texture (only float textures are filterable)";
          return in;
        }
        auto s = samplers.find(trim(args[0]));
        if (s == samplers.end()) {
          err = "sampler '" + trim(args[0]) + "' is not a constexpr sampler in this kernel";
          return in;
        }
        const std::string c = trim(args[1]);
        const std::string call = "weft::tex_sample<float>(" + t.name + ", (" + c + ").x, (" + c +
                                 ").y, " + std::to_string(s->second.first) + ", " +
                                 std::to_string(s->second.second) + ")";
        out += vec + "{" + (t.value == "half" ? "(half)" + call : call) + ", " + zero + ", " +
               zero + ", " + one + "}";
        i = close + 1;
        hit = true;
        break;
      }
      // A storage image: `img.write(float4(v), uint2(x, y))`. Metal's third
      // argument is a mip level and the launcher creates one level, so a write
      // with an explicit level is the same write.
      if (member == "write") {
        if (args.size() != 2 && args.size() != 3) {
          err = "texture write takes a value, a coordinate and an optional level";
          return in;
        }
        const std::string c = trim(args[1]);
        out += "weft::tex_write(" + t.name + ", (uint32_t)(" + c + ").x, (uint32_t)(" + c +
               ").y, " + trim(args[0]) + ")";
        i = close + 1;
        hit = true;
        break;
      }
      err = "texture " + t.name + " has no method '" + member + "' this front end knows";
      return in;
    }
    if (hit) continue;
    out.push_back(in[i]);
    i++;
  }
  return out;
}

// One table for the whole warp/quad library: the shuffles, the four scan and
// four reduction families, the votes, and the three things that are just this
// thread's own lane. Every one of them is either a rendezvous (an awaitable) or
// an expression, so they share one scanner.
std::string rewrite_group_ops(const std::string& body, bool& found, bool& lanes,
                              std::string& err) {
  struct Op {
    const char* name;
    const char* kind;  // shfl, fill, prefix, reduce, vote, expr
    const char* mode;  // weft::kShfl*, or the GrpReduce / vote mode as a digit
    int flag;          // prefix: inclusive?, product?; vote: returns bool?
    int width;         // 32 for a warp, 4 for a quad
    const char* expr;  // kind == expr
  };
  static const Op kOps[] = {
      // Warp shuffles. Metal's argument order is (value, delta).
      {"simd_shuffle_and_fill_down", "fill", "weft::kShflDown", 0, 32, nullptr},
      {"simd_shuffle_and_fill_up", "fill", "weft::kShflUp", 0, 32, nullptr},
      {"simd_shuffle_down", "shfl", "weft::kShflDown", 0, 32, nullptr},
      {"simd_shuffle_up", "shfl", "weft::kShflUp", 0, 32, nullptr},
      {"simd_shuffle_xor", "shfl", "weft::kShflBfly", 0, 32, nullptr},
      {"simd_shuffle", "shfl", "weft::kShflIdx", 0, 32, nullptr},
      {"simd_broadcast", "shfl", "weft::kShflIdx", 0, 32, nullptr},
      // Scans: flag bit 0 is inclusive, bit 1 is "multiply, not add".
      {"simd_prefix_exclusive_sum", "prefix", "0", 0, 32, nullptr},
      {"simd_prefix_inclusive_sum", "prefix", "0", 1, 32, nullptr},
      {"simd_prefix_exclusive_product", "prefix", "0", 2, 32, nullptr},
      {"simd_prefix_inclusive_product", "prefix", "0", 3, 32, nullptr},
      // Reductions.
      {"simd_sum", "reduce", "0", 0, 32, nullptr},
      {"simd_product", "reduce", "3", 0, 32, nullptr},
      {"simd_min", "reduce", "1", 0, 32, nullptr},
      {"simd_max", "reduce", "2", 0, 32, nullptr},
      // Votes. flag: 1 means the result is a bool, not a mask.
      {"simd_ballot", "vote", "0", 0, 32, nullptr},
      {"simd_all", "vote", "1", 1, 32, nullptr},
      {"simd_any", "vote", "2", 1, 32, nullptr},
      // Per-thread facts. No rendezvous, so no suspension.
      {"simd_is_first", "expr", "", 0, 32, "((ctx.lid & 31u) == 0u)"},
      {"simd_is_helper_thread", "expr", "", 0, 32, "(false)"},
      // The same functions over a quad: four consecutive lanes, so the group
      // base moves and the masks narrow, and nothing else changes.
      {"quad_shuffle_and_fill_down", "fill", "weft::kShflDown", 0, 4, nullptr},
      {"quad_shuffle_and_fill_up", "fill", "weft::kShflUp", 0, 4, nullptr},
      {"quad_shuffle_down", "shfl", "weft::kShflDown", 0, 4, nullptr},
      {"quad_shuffle_up", "shfl", "weft::kShflUp", 0, 4, nullptr},
      {"quad_shuffle_xor", "shfl", "weft::kShflBfly", 0, 4, nullptr},
      {"quad_shuffle", "shfl", "weft::kShflIdx", 0, 4, nullptr},
      {"quad_broadcast", "shfl", "weft::kShflIdx", 0, 4, nullptr},
      {"quad_prefix_exclusive_sum", "prefix", "0", 0, 4, nullptr},
      {"quad_prefix_inclusive_sum", "prefix", "0", 1, 4, nullptr},
      {"quad_prefix_exclusive_product", "prefix", "0", 2, 4, nullptr},
      {"quad_prefix_inclusive_product", "prefix", "0", 3, 4, nullptr},
      {"quad_sum", "reduce", "0", 0, 4, nullptr},
      {"quad_product", "reduce", "3", 0, 4, nullptr},
      {"quad_min", "reduce", "1", 0, 4, nullptr},
      {"quad_max", "reduce", "2", 0, 4, nullptr},
      {"quad_ballot", "vote", "0", 0, 4, nullptr},
      {"quad_all", "vote", "1", 1, 4, nullptr},
      {"quad_any", "vote", "2", 1, 4, nullptr},
  };

  std::string out;
  size_t i = 0;
  while (i < body.size()) {
    bool matched = false;
    for (const auto& op : kOps) {
      const size_t len = std::strlen(op.name);
      if (body.compare(i, len, op.name) != 0) continue;
      if (i > 0 && is_ident(body[i - 1])) continue;
      size_t j = i + len;
      if (j < body.size() && is_ident(body[j])) continue;
      while (j < body.size() && std::isspace((unsigned char)body[j])) j++;
      if (j >= body.size() || body[j] != '(') continue;
      const size_t close = find_matching(body, j, '(', ')');
      if (close == std::string::npos) {
        err = std::string("unbalanced ") + op.name;
        return body;
      }
      std::vector<std::string> a = split_top_level(body.substr(j + 1, close - j - 1));
      const std::string kind = op.kind;
      const std::string width = std::to_string(op.width);
      const std::string inc = (op.flag & 1) ? "true" : "false";
      const std::string prod = (op.flag & 2) ? "true" : "false";

      if (kind == "expr") {
        if (!a.empty() && !(a.size() == 1 && a[0].empty())) {
          err = std::string(op.name) + " takes no arguments";
          return body;
        }
        // Reading the lane id is not a rendezvous: no suspension, but it does
        // mean the thread needs its own context, so the kernel cannot take the
        // flattened loop form.
        out += op.expr;
        lanes = true;
        matched = true;
        i = close + 1;
        break;
      }
      if (kind == "shfl") {
        if (a.size() != 2) {
          err = std::string(op.name) + " takes (value, delta)";
          return body;
        }
        out += "(co_await weft::grp_shfl(ctx, (" + trim(a[0]) + "), (" + trim(a[1]) + "), " +
               op.mode + ", " + width + "))";
      } else if (kind == "fill") {
        if (a.size() != 3) {
          err = std::string(op.name) + " takes (value, fill, delta)";
          return body;
        }
        out += "(co_await weft::shfl_fill(ctx, (" + trim(a[0]) + "), (" + trim(a[2]) + "), " +
               op.mode + ", (" + trim(a[1]) + "), " + width + "))";
      } else if (kind == "prefix") {
        if (a.size() != 1) {
          err = std::string(op.name) + " takes one value";
          return body;
        }
        out += "(co_await weft::prefix(ctx, (" + trim(a[0]) + "), " + inc + ", " + prod + ", " +
               width + "))";
      } else if (kind == "reduce") {
        if (a.size() != 1) {
          err = std::string(op.name) + " takes one value";
          return body;
        }
        out += "(co_await weft::grp_reduce(ctx, (" + trim(a[0]) + "), " + op.mode + ", " +
               width + "))";
      } else {
        if (a.size() != 1) {
          err = std::string(op.name) + " takes one predicate";
          return body;
        }
        const std::string call = "(co_await weft::grp_vote(ctx, (" + trim(a[0]) + ") != 0, " +
                                 op.mode + ", " + width + "))";
        out += op.flag ? ("(" + call + " != 0u)") : call;
      }
      found = true;
      matched = true;
      i = close + 1;
      break;
    }
    if (matched) continue;
    out.push_back(body[i]);
    i++;
  }
  return out;
}

// The Metal standard library pieces a kernel actually leans on: vector types
// with the operators Metal defines, and atomics backed by std::atomic.
const char* kMslPreamble = R"WEFT(
// Vector types with Metal's sizes and alignments. The 3-element ones carry tail
// padding on purpose: float3 is 16 bytes on a GPU, so a kernel indexing a
// device float3* depends on that stride, and char3/short3 are padded to 4/8.
using half = _Float16;
// Alignment is part of the ABI too: Metal aligns a 2-element vector to twice its
// element, a 4-element one to four times, and a struct to its widest member and
// rounds its size up to that. A `struct { float4 p; float2 uv; }` is 32 bytes on
// the GPU (p at 0, uv at 16, size rounded from 24 to 16-aligned), so a C++ mirror
// without the alignments would bind the same buffer at a 24-byte stride and read
// every vertex after the first from the wrong place.
struct alignas(8) float2 {
  float x = 0, y = 0;
  float2() {}
  float2(float v) : x(v), y(v) {}
  float2(float a, float b) : x(a), y(b) {}
};
struct float4;
struct alignas(16) float3 {
  float x = 0, y = 0, z = 0, _pad = 0;
  float3() {}
  float3(float v) : x(v), y(v), z(v) {}
  float3(float a, float b, float c) : x(a), y(b), z(c) {}
  // The four-argument form keeps the tail padding, which is what the operator
  // macros pass to preserve it across arithmetic.
  float3(float a, float b, float c, float p) : x(a), y(b), z(c), _pad(p) {}
  float3(float2 v, float c) : x(v.x), y(v.y), z(c) {}
  float3(float a, float2 v) : x(a), y(v.x), z(v.y) {}
  float3(float4 v);
};
struct alignas(16) float4 {
  float x = 0, y = 0, z = 0, w = 0;
  float4() {}
  float4(float v) : x(v), y(v), z(v), w(v) {}
  float4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
  float4(float3 v, float d) : x(v.x), y(v.y), z(v.z), w(d) {}
  float4(float a, float3 v) : x(a), y(v.x), z(v.y), w(v.z) {}
  float4(float2 a, float2 b) : x(a.x), y(a.y), z(b.x), w(b.y) {}
};
inline float3::float3(float4 v) : x(v.x), y(v.y), z(v.z) {}
struct alignas(8) int2 { int32_t x, y; };
struct alignas(16) int3 { int32_t x, y, z; int32_t _pad; };
struct alignas(16) int4 { int32_t x, y, z, w; };
struct alignas(8) uint2 { uint32_t x, y; };
struct alignas(16) uint3 { uint32_t x, y, z; uint32_t _pad; };
struct alignas(16) uint4 { uint32_t x, y, z, w; };
struct alignas(4) half2 {
  half x, y;
  half2() {}
  half2(half v) : x(v), y(v) {}
  half2(half a, half b) : x(a), y(b) {}
};
struct half4;
struct alignas(8) half3 {
  half x, y, z, _pad;
  half3() {}
  half3(half v) : x(v), y(v), z(v) {}
  half3(half a, half b, half c) : x(a), y(b), z(c) {}
  half3(half a, half b, half c, half p) : x(a), y(b), z(c), _pad(p) {}
  half3(half2 v, half c) : x(v.x), y(v.y), z(c) {}
  half3(half4 v);
};
struct alignas(8) half4 {
  half x, y, z, w;
  half4() {}
  half4(half v) : x(v), y(v), z(v), w(v) {}
  half4(half a, half b, half c, half d) : x(a), y(b), z(c), w(d) {}
  half4(half3 v, half d) : x(v.x), y(v.y), z(v.z), w(d) {}
  half4(half a, half3 v) : x(a), y(v.x), z(v.y), w(v.z) {}
  half4(half2 a, half2 b) : x(a.x), y(a.y), z(b.x), w(b.y) {}
};
inline half3::half3(half4 v) : x(v.x), y(v.y), z(v.z) {}
struct alignas(2) char2 { int8_t x, y; };
struct alignas(4) char3 { int8_t x, y, z, _pad; };
struct alignas(4) char4 { int8_t x, y, z, w; };
struct alignas(2) uchar2 { uint8_t x, y; };
struct alignas(4) uchar3 { uint8_t x, y, z, _pad; };
struct alignas(4) uchar4 { uint8_t x, y, z, w; };
struct alignas(4) short2 { int16_t x, y; };
struct alignas(8) short3 { int16_t x, y, z, _pad; };
struct alignas(8) short4 { int16_t x, y, z, w; };
struct alignas(4) ushort2 { uint16_t x, y; };
struct alignas(8) ushort3 { uint16_t x, y, z, _pad; };
struct alignas(8) ushort4 { uint16_t x, y, z, w; };
// The 64-bit and boolean vectors. A bool vector is one byte per lane with no
// padding at all, and the 64-bit ones pad a 3-element vector out to 32 bytes.
struct bool2 { bool x, y; };
struct bool3 { bool x, y, z; };
struct bool4 { bool x, y, z, w; };
struct alignas(16) long2 { int64_t x, y; };
struct alignas(16) long3 { int64_t x, y, z, _pad; };
struct alignas(16) long4 { int64_t x, y, z, w; };
struct alignas(16) ulong2 { uint64_t x, y; };
struct alignas(16) ulong3 { uint64_t x, y, z, _pad; };
struct alignas(16) ulong4 { uint64_t x, y, z, w; };
struct alignas(16) double2 { double x, y; };
struct alignas(16) double3 { double x, y, z, _pad; };
struct alignas(16) double4 { double x, y, z, w; };
static_assert(sizeof(float3) == 16 && sizeof(int3) == 16 && sizeof(uint3) == 16,
              "Metal pads 3-element 32-bit vectors to 16 bytes");
static_assert(sizeof(bool2) == 2 && sizeof(bool3) == 3 && sizeof(bool4) == 4,
              "a Metal bool vector is one byte per lane");
static_assert(sizeof(long2) == 16 && sizeof(long3) == 32 && sizeof(long4) == 32,
              "Metal pads a 3-element 64-bit vector to 32 bytes");
static_assert(sizeof(double2) == 16 && sizeof(double3) == 32 && sizeof(double4) == 32,
              "Metal pads a 3-element double vector to 32 bytes");
static_assert(sizeof(half2) == 4 && sizeof(half3) == 8 && sizeof(half4) == 8,
              "Metal half vectors are 2/4/8/8 bytes");
static_assert(sizeof(char3) == 4 && sizeof(uchar3) == 4 && sizeof(short3) == 8 &&
                  sizeof(ushort3) == 8,
              "Metal pads 3-element 8- and 16-bit vectors to 4 and 8 bytes");
struct weft_layout_probe { float4 p; float2 uv; };
static_assert(sizeof(weft_layout_probe) == 32 && alignof(weft_layout_probe) == 16,
              "Metal lays this out as p at 0, uv at 16, size rounded to 32: a vertex "
              "struct is the case where alignment, not member size, decides the stride");

// The packed vectors: the same element types with no tail padding, so a
// packed_float3 is the 12 bytes a device buffer of xyz triples actually holds.
// That is the whole point of the type — it is a layout, and the alignments are
// the element's, which is why char3-packed is 3 bytes and not 4.
#define WEFT_PACKED_VECS(S, P)                                                     \
  struct packed_##P##2 { S x, y; };                                               \
  struct packed_##P##3 { S x, y, z; };                                            \
  struct packed_##P##4 { S x, y, z, w; };
WEFT_PACKED_VECS(float, float)
WEFT_PACKED_VECS(half, half)
WEFT_PACKED_VECS(int8_t, char)
WEFT_PACKED_VECS(uint8_t, uchar)
WEFT_PACKED_VECS(int16_t, short)
WEFT_PACKED_VECS(uint16_t, ushort)
WEFT_PACKED_VECS(int32_t, int)
WEFT_PACKED_VECS(uint32_t, uint)
WEFT_PACKED_VECS(bool, bool)
static_assert(sizeof(packed_float3) == 12 && sizeof(packed_half3) == 6 &&
                  sizeof(packed_char3) == 3 && sizeof(packed_short3) == 6 &&
                  sizeof(packed_int3) == 12 && sizeof(packed_uint3) == 12,
              "a packed 3-element vector is exactly three elements wide");
static_assert(sizeof(packed_float4) == 16 && sizeof(packed_char4) == 4,
              "a packed 4-element vector is four elements wide");

// Elementwise arithmetic for the vectors a kernel computes with. The 8- and
// 16-bit vectors stay data types: kernels poke at their fields, and defining
// every promoted-integer operator would be a lot of code for no use.
// Every operator keeps the destination type, so an overflow wraps the way the
// hardware's does: char2(100) + char2(100) is -56 here and on the GPU.
#define WEFT_VEC2(T, S)                                                            \
  static inline T operator+(T a, T b) { return T{(S)(a.x + b.x), (S)(a.y + b.y)}; } \
  static inline T operator-(T a, T b) { return T{(S)(a.x - b.x), (S)(a.y - b.y)}; } \
  static inline T operator*(T a, T b) { return T{(S)(a.x * b.x), (S)(a.y * b.y)}; } \
  static inline T operator/(T a, T b) { return T{(S)(a.x / b.x), (S)(a.y / b.y)}; } \
  static inline T operator*(T a, S s) { return T{(S)(a.x * s), (S)(a.y * s)}; }     \
  static inline T operator+(T a, S s) { return T{(S)(a.x + s), (S)(a.y + s)}; }     \
  static inline T operator-(T a, S s) { return T{(S)(a.x - s), (S)(a.y - s)}; }     \
  static inline T operator/(T a, S s) { return T{(S)(a.x / s), (S)(a.y / s)}; }     \
  static inline T operator-(T a) { return T{(S)(-a.x), (S)(-a.y)}; }
#define WEFT_VEC3(T, S)                                                             \
  static inline T operator+(T a, T b) {                                            \
    return T{(S)(a.x + b.x), (S)(a.y + b.y), (S)(a.z + b.z), a._pad};              \
  }                                                                                \
  static inline T operator-(T a, T b) {                                            \
    return T{(S)(a.x - b.x), (S)(a.y - b.y), (S)(a.z - b.z), a._pad};              \
  }                                                                                \
  static inline T operator*(T a, T b) {                                            \
    return T{(S)(a.x * b.x), (S)(a.y * b.y), (S)(a.z * b.z), a._pad};              \
  }                                                                                \
  static inline T operator/(T a, T b) {                                            \
    return T{(S)(a.x / b.x), (S)(a.y / b.y), (S)(a.z / b.z), a._pad};              \
  }                                                                                \
  static inline T operator*(T a, S s) {                                            \
    return T{(S)(a.x * s), (S)(a.y * s), (S)(a.z * s), a._pad};                    \
  }                                                                                \
  static inline T operator+(T a, S s) {                                            \
    return T{(S)(a.x + s), (S)(a.y + s), (S)(a.z + s), a._pad};                    \
  }                                                                                \
  static inline T operator-(T a, S s) {                                            \
    return T{(S)(a.x - s), (S)(a.y - s), (S)(a.z - s), a._pad};                    \
  }                                                                                \
  static inline T operator/(T a, S s) {                                            \
    return T{(S)(a.x / s), (S)(a.y / s), (S)(a.z / s), a._pad};                    \
  }                                                                                \
  static inline T operator-(T a) { return T{(S)(-a.x), (S)(-a.y), (S)(-a.z), a._pad}; }
#define WEFT_VEC4(T, S)                                                             \
  static inline T operator+(T a, T b) {                                            \
    return T{(S)(a.x + b.x), (S)(a.y + b.y), (S)(a.z + b.z), (S)(a.w + b.w)};      \
  }                                                                                \
  static inline T operator-(T a, T b) {                                            \
    return T{(S)(a.x - b.x), (S)(a.y - b.y), (S)(a.z - b.z), (S)(a.w - b.w)};      \
  }                                                                                \
  static inline T operator*(T a, T b) {                                            \
    return T{(S)(a.x * b.x), (S)(a.y * b.y), (S)(a.z * b.z), (S)(a.w * b.w)};      \
  }                                                                                \
  static inline T operator/(T a, T b) {                                            \
    return T{(S)(a.x / b.x), (S)(a.y / b.y), (S)(a.z / b.z), (S)(a.w / b.w)};      \
  }                                                                                \
  static inline T operator*(T a, S s) {                                            \
    return T{(S)(a.x * s), (S)(a.y * s), (S)(a.z * s), (S)(a.w * s)};              \
  }                                                                                \
  static inline T operator+(T a, S s) {                                            \
    return T{(S)(a.x + s), (S)(a.y + s), (S)(a.z + s), (S)(a.w + s)};              \
  }                                                                                \
  static inline T operator-(T a, S s) {                                            \
    return T{(S)(a.x - s), (S)(a.y - s), (S)(a.z - s), (S)(a.w - s)};              \
  }                                                                                \
  static inline T operator/(T a, S s) {                                            \
    return T{(S)(a.x / s), (S)(a.y / s), (S)(a.z / s), (S)(a.w / s)};              \
  }                                                                                \
  static inline T operator-(T a) {                                                 \
    return T{(S)(-a.x), (S)(-a.y), (S)(-a.z), (S)(-a.w)};                          \
  }
WEFT_VEC2(float2, float)
WEFT_VEC3(float3, float)
WEFT_VEC4(float4, float)
WEFT_VEC2(half2, half)
WEFT_VEC3(half3, half)
WEFT_VEC4(half4, half)
WEFT_VEC2(int2, int32_t)
WEFT_VEC3(int3, int32_t)
WEFT_VEC4(int4, int32_t)
WEFT_VEC2(uint2, uint32_t)
WEFT_VEC3(uint3, uint32_t)
WEFT_VEC4(uint4, uint32_t)
// The 8- and 16-bit vectors compute too, and the result is the same narrow
// type — which is what makes an overflow wrap where the hardware wraps.
WEFT_VEC2(char2, int8_t)
WEFT_VEC3(char3, int8_t)
WEFT_VEC4(char4, int8_t)
WEFT_VEC2(uchar2, uint8_t)
WEFT_VEC3(uchar3, uint8_t)
WEFT_VEC4(uchar4, uint8_t)
WEFT_VEC2(short2, int16_t)
WEFT_VEC3(short3, int16_t)
WEFT_VEC4(short4, int16_t)
WEFT_VEC2(ushort2, uint16_t)
WEFT_VEC3(ushort3, uint16_t)
WEFT_VEC4(ushort4, uint16_t)
// Packed vectors are arithmetic types as well: the element count is all the
// operators care about, and the result is the packed type so it stays 12 bytes
// rather than growing a padding lane.
#define WEFT_PVEC2(T, S)                                                          \
  static inline T operator+(T a, T b) { return T{(S)(a.x + b.x), (S)(a.y + b.y)}; } \
  static inline T operator-(T a, T b) { return T{(S)(a.x - b.x), (S)(a.y - b.y)}; } \
  static inline T operator*(T a, T b) { return T{(S)(a.x * b.x), (S)(a.y * b.y)}; } \
  static inline T operator/(T a, T b) { return T{(S)(a.x / b.x), (S)(a.y / b.y)}; } \
  static inline T operator*(T a, S s) { return T{(S)(a.x * s), (S)(a.y * s)}; }     \
  static inline T operator+(T a, S s) { return T{(S)(a.x + s), (S)(a.y + s)}; }     \
  static inline T operator-(T a, S s) { return T{(S)(a.x - s), (S)(a.y - s)}; }
#define WEFT_PVEC3(T, S)                                                          \
  static inline T operator+(T a, T b) {                                          \
    return T{(S)(a.x + b.x), (S)(a.y + b.y), (S)(a.z + b.z)};                     \
  }                                                                              \
  static inline T operator-(T a, T b) {                                          \
    return T{(S)(a.x - b.x), (S)(a.y - b.y), (S)(a.z - b.z)};                     \
  }                                                                              \
  static inline T operator*(T a, T b) {                                          \
    return T{(S)(a.x * b.x), (S)(a.y * b.y), (S)(a.z * b.z)};                     \
  }                                                                              \
  static inline T operator/(T a, T b) {                                          \
    return T{(S)(a.x / b.x), (S)(a.y / b.y), (S)(a.z / b.z)};                     \
  }                                                                              \
  static inline T operator*(T a, S s) {                                          \
    return T{(S)(a.x * s), (S)(a.y * s), (S)(a.z * s)};                           \
  }                                                                              \
  static inline T operator-(T a) { return T{(S)(-a.x), (S)(-a.y), (S)(-a.z)}; }
#define WEFT_PVEC4(T, S)                                                          \
  static inline T operator+(T a, T b) {                                          \
    return T{(S)(a.x + b.x), (S)(a.y + b.y), (S)(a.z + b.z), (S)(a.w + b.w)};     \
  }                                                                              \
  static inline T operator-(T a, T b) {                                          \
    return T{(S)(a.x - b.x), (S)(a.y - b.y), (S)(a.z - b.z), (S)(a.w - b.w)};     \
  }                                                                              \
  static inline T operator*(T a, T b) {                                          \
    return T{(S)(a.x * b.x), (S)(a.y * b.y), (S)(a.z * b.z), (S)(a.w * b.w)};     \
  }                                                                              \
  static inline T operator/(T a, T b) {                                          \
    return T{(S)(a.x / b.x), (S)(a.y / b.y), (S)(a.z / b.z), (S)(a.w / b.w)};     \
  }                                                                              \
  static inline T operator*(T a, S s) {                                          \
    return T{(S)(a.x * s), (S)(a.y * s), (S)(a.z * s), (S)(a.w * s)};             \
  }                                                                              \
  static inline T operator-(T a) {                                               \
    return T{(S)(-a.x), (S)(-a.y), (S)(-a.z), (S)(-a.w)};                         \
  }
#define WEFT_PVECS(S, P)                                                          \
  WEFT_PVEC2(packed_##P##2, S) WEFT_PVEC3(packed_##P##3, S) WEFT_PVEC4(packed_##P##4, S)
WEFT_PVECS(float, float)
WEFT_PVECS(half, half)
WEFT_PVECS(int8_t, char)
WEFT_PVECS(uint8_t, uchar)
WEFT_PVECS(int16_t, short)
WEFT_PVECS(uint16_t, ushort)
WEFT_PVECS(int32_t, int)
WEFT_PVECS(uint32_t, uint)
WEFT_VEC2(long2, int64_t)
WEFT_VEC3(long3, int64_t)
WEFT_VEC4(long4, int64_t)
WEFT_VEC2(ulong2, uint64_t)
WEFT_VEC3(ulong3, uint64_t)
WEFT_VEC4(ulong4, uint64_t)
WEFT_VEC2(double2, double)
WEFT_VEC3(double3, double)
WEFT_VEC4(double4, double)
// Boolean vectors are elementwise logic, not arithmetic, so they get their own
// set of operators.
#define WEFT_BVEC(T, N)                                                              \
  static inline T operator&&(T a, T b) {                                            \
    T r{};                                                                          \
    for (int i = 0; i < N; i++) ((bool*)&r)[i] = ((bool*)&a)[i] && ((bool*)&b)[i];  \
    return r;                                                                       \
  }                                                                                 \
  static inline T operator||(T a, T b) {                                            \
    T r{};                                                                          \
    for (int i = 0; i < N; i++) ((bool*)&r)[i] = ((bool*)&a)[i] || ((bool*)&b)[i];  \
    return r;                                                                       \
  }                                                                                 \
  static inline T operator!(T a) {                                                  \
    T r{};                                                                          \
    for (int i = 0; i < N; i++) ((bool*)&r)[i] = !((bool*)&a)[i];                   \
    return r;                                                                       \
  }                                                                                 \
  static inline bool all(T a) {                                                     \
    for (int i = 0; i < N; i++)                                                     \
      if (!((bool*)&a)[i]) return false;                                            \
    return true;                                                                    \
  }                                                                                 \
  static inline bool any(T a) {                                                     \
    for (int i = 0; i < N; i++)                                                     \
      if (((bool*)&a)[i]) return true;                                              \
    return false;                                                                   \
  }
WEFT_BVEC(bool2, 2)
WEFT_BVEC(bool3, 3)
WEFT_BVEC(bool4, 4)
static inline float2 make_float2(float x, float y) { return float2{x, y}; }
static inline float3 make_float3(float x, float y, float z) { return float3{x, y, z, 0.0f}; }
static inline half2 make_half2(half x, half y) { return half2{x, y}; }
static inline int3 make_int3(int32_t x, int32_t y, int32_t z) { return int3{x, y, z, 0}; }
static inline uint3 make_uint3(uint32_t x, uint32_t y, uint32_t z) { return uint3{x, y, z, 0}; }
static inline float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }
static inline uint4 make_uint4(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
  return uint4{x, y, z, w};
}
// Comparisons return a bool vector, element by element, for every vector type
// Metal defines them on. A bool vector gets only equality: `<` of two bools is
// not something MSL offers either.
#define WEFT_VCMP2(T, B)                                                                  \
  static inline B operator==(T a, T b) { return B{a.x == b.x, a.y == b.y}; }              \
  static inline B operator!=(T a, T b) { return B{a.x != b.x, a.y != b.y}; }              \
  static inline B operator<(T a, T b) { return B{a.x < b.x, a.y < b.y}; }                 \
  static inline B operator<=(T a, T b) { return B{a.x <= b.x, a.y <= b.y}; }              \
  static inline B operator>(T a, T b) { return B{a.x > b.x, a.y > b.y}; }                 \
  static inline B operator>=(T a, T b) { return B{a.x >= b.x, a.y >= b.y}; }
#define WEFT_VCMP3(T, B)                                                                  \
  static inline B operator==(T a, T b) { return B{a.x == b.x, a.y == b.y, a.z == b.z}; }  \
  static inline B operator!=(T a, T b) { return B{a.x != b.x, a.y != b.y, a.z != b.z}; }  \
  static inline B operator<(T a, T b) { return B{a.x < b.x, a.y < b.y, a.z < b.z}; }      \
  static inline B operator<=(T a, T b) { return B{a.x <= b.x, a.y <= b.y, a.z <= b.z}; }  \
  static inline B operator>(T a, T b) { return B{a.x > b.x, a.y > b.y, a.z > b.z}; }      \
  static inline B operator>=(T a, T b) { return B{a.x >= b.x, a.y >= b.y, a.z >= b.z}; }
#define WEFT_VCMP4(T, B)                                                                  \
  static inline B operator==(T a, T b) {                                                  \
    return B{a.x == b.x, a.y == b.y, a.z == b.z, a.w == b.w};                             \
  }                                                                                       \
  static inline B operator!=(T a, T b) {                                                  \
    return B{a.x != b.x, a.y != b.y, a.z != b.z, a.w != b.w};                             \
  }                                                                                       \
  static inline B operator<(T a, T b) { return B{a.x < b.x, a.y < b.y, a.z < b.z, a.w < b.w}; } \
  static inline B operator<=(T a, T b) {                                                  \
    return B{a.x <= b.x, a.y <= b.y, a.z <= b.z, a.w <= b.w};                             \
  }                                                                                       \
  static inline B operator>(T a, T b) { return B{a.x > b.x, a.y > b.y, a.z > b.z, a.w > b.w}; } \
  static inline B operator>=(T a, T b) {                                                  \
    return B{a.x >= b.x, a.y >= b.y, a.z >= b.z, a.w >= b.w};                             \
  }
#define WEFT_VCMP123(T, B2, B3, B4) WEFT_VCMP2(T##2, B2) WEFT_VCMP3(T##3, B3) WEFT_VCMP4(T##4, B4)
WEFT_VCMP123(float, bool2, bool3, bool4)
WEFT_VCMP123(half, bool2, bool3, bool4)
WEFT_VCMP123(int, bool2, bool3, bool4)
WEFT_VCMP123(uint, bool2, bool3, bool4)
WEFT_VCMP123(char, bool2, bool3, bool4)
WEFT_VCMP123(uchar, bool2, bool3, bool4)
WEFT_VCMP123(short, bool2, bool3, bool4)
WEFT_VCMP123(ushort, bool2, bool3, bool4)
WEFT_VCMP123(long, bool2, bool3, bool4)
WEFT_VCMP123(ulong, bool2, bool3, bool4)
WEFT_VCMP123(double, bool2, bool3, bool4)
static inline bool2 operator==(bool2 a, bool2 b) { return bool2{a.x == b.x, a.y == b.y}; }
static inline bool3 operator==(bool3 a, bool3 b) { return bool3{a.x == b.x, a.y == b.y, a.z == b.z}; }
static inline bool4 operator==(bool4 a, bool4 b) {
  return bool4{a.x == b.x, a.y == b.y, a.z == b.z, a.w == b.w};
}
static inline bool2 operator!=(bool2 a, bool2 b) { return !(a == b); }
static inline bool3 operator!=(bool3 a, bool3 b) { return !(a == b); }
static inline bool4 operator!=(bool4 a, bool4 b) { return !(a == b); }
static inline long2 make_long2(int64_t x, int64_t y) { return long2{x, y}; }
static inline long3 make_long3(int64_t x, int64_t y, int64_t z) { return long3{x, y, z, 0}; }
static inline long4 make_long4(int64_t x, int64_t y, int64_t z, int64_t w) {
  return long4{x, y, z, w};
}
static inline ulong2 make_ulong2(uint64_t x, uint64_t y) { return ulong2{x, y}; }
static inline ulong3 make_ulong3(uint64_t x, uint64_t y, uint64_t z) {
  return ulong3{x, y, z, 0};
}
static inline ulong4 make_ulong4(uint64_t x, uint64_t y, uint64_t z, uint64_t w) {
  return ulong4{x, y, z, w};
}
static inline double2 make_double2(double x, double y) { return double2{x, y}; }
static inline double3 make_double3(double x, double y, double z) { return double3{x, y, z, 0.0}; }
static inline double4 make_double4(double x, double y, double z, double w) {
  return double4{x, y, z, w};
}
static inline bool2 make_bool2(bool x, bool y) { return bool2{x, y}; }
static inline bool3 make_bool3(bool x, bool y, bool z) { return bool3{x, y, z}; }
static inline bool4 make_bool4(bool x, bool y, bool z, bool w) { return bool4{x, y, z, w}; }
static inline float4 operator*(float b, float4 a) { return a * b; }
static inline float3 operator*(float b, float3 a) { return a * b; }
static inline half4 operator*(half b, half4 a) { return a * b; }
static inline float4& operator+=(float4& a, float4 b) { a = a + b; return a; }
static inline float4& operator*=(float4& a, float4 b) { a = a * b; return a; }
static inline float4& operator*=(float4& a, float b) { a = a * b; return a; }
static inline float3& operator+=(float3& a, float3 b) { a = a + b; return a; }
static inline float3& operator*=(float3& a, float b) { a = a * b; return a; }
// ---- simdgroup matrices ----
//
// On a GPU an 8x8 simdgroup matrix is distributed across the 32 lanes, and the
// three operations are collective: every lane passes the same pointers and gets
// its own fragment. Here the whole matrix is the value every lane holds, and the
// operations do the whole load, store and product. That is the same answer for
// every program the hardware accepts, because a collective call by definition
// has the same arguments in all 32 lanes — only where the pieces live differs.
// The element order of a load/store is Metal's: row-major in memory with
// `elements_per_row`, with an optional uint2 origin and transpose.
template <class T, int Cols, int Rows> struct simdgroup_matrix {
  T e[Cols * Rows];
  T& at(int r, int c) { return e[r * Cols + c]; }
  const T& at(int r, int c) const { return e[r * Cols + c]; }
};
using simdgroup_float8x8 = simdgroup_matrix<float, 8, 8>;
using simdgroup_half8x8 = simdgroup_matrix<half, 8, 8>;

template <class T, int Cols, int Rows>
static inline void simdgroup_load(simdgroup_matrix<T, Cols, Rows>& m, const T* src,
                                  unsigned long long elements_per_row = (unsigned long long)Cols,
                                  ulong2 origin = ulong2{0, 0}, bool transpose = false) {
  for (int r = 0; r < Rows; r++)
    for (int c = 0; c < Cols; c++) {
      const size_t y = (size_t)origin.y + (transpose ? c : r);
      const size_t x = (size_t)origin.x + (transpose ? r : c);
      m.at(r, c) = src[y * elements_per_row + x];
    }
}

template <class T, int Cols, int Rows>
static inline void simdgroup_store(simdgroup_matrix<T, Cols, Rows> m, T* dst,
                                   unsigned long long elements_per_row = (unsigned long long)Cols,
                                   ulong2 origin = ulong2{0, 0}, bool transpose = false) {
  for (int r = 0; r < Rows; r++)
    for (int c = 0; c < Cols; c++) {
      const size_t y = (size_t)origin.y + (transpose ? c : r);
      const size_t x = (size_t)origin.x + (transpose ? r : c);
      dst[y * elements_per_row + x] = m.at(r, c);
    }
}

// d = a * b + c, with a Rows x K, b K x Cols and c/d Rows x Cols. The accumulate
// is a plain sum in program order, which is what the GPU's dot product does for
// the fp32 case; half products accumulate in half here, as they do there.
template <class T, int Cols, int Rows>
static inline void simdgroup_multiply_accumulate(simdgroup_matrix<T, Cols, Rows>& d,
                                                 simdgroup_matrix<T, Cols, Rows> a,
                                                 simdgroup_matrix<T, Cols, Rows> b,
                                                 simdgroup_matrix<T, Cols, Rows> c) {
  for (int r = 0; r < Rows; r++)
    for (int k = 0; k < Cols; k++) {
      T acc = c.at(r, k);
      for (int j = 0; j < Rows; j++) acc += a.at(r, j) * b.at(j, k);
      d.at(r, k) = acc;
    }
}

template <class T, int Cols, int Rows>
static inline simdgroup_matrix<T, Cols, Rows> simdgroup_multiply(simdgroup_matrix<T, Cols, Rows> a,
                                                                simdgroup_matrix<T, Cols, Rows> b) {
  simdgroup_matrix<T, Cols, Rows> d{};
  for (int i = 0; i < Cols * Rows; i++) d.e[i] = T(0);
  simdgroup_multiply_accumulate(d, a, b, d);
  return d;
}

template <class T, int Cols, int Rows>
static inline simdgroup_matrix<T, Cols, Rows> make_filled_simdgroup_matrix(T value) {
  simdgroup_matrix<T, Cols, Rows> m;
  for (int i = 0; i < Cols * Rows; i++) m.e[i] = value;
  return m;
}

using atomic_uint = std::atomic<uint32_t>;
using atomic_int = std::atomic<int32_t>;
enum weft_mem_order { memory_order_relaxed = 0, memory_order_acquire = 1, memory_order_release = 2,
                     memory_order_acq_rel = 3, memory_order_seq_cst = 4 };
template <class T> static inline T atomic_load_explicit(const std::atomic<T>* p, weft_mem_order) {
  return p->load();
}
template <class T> static inline void atomic_store_explicit(std::atomic<T>* p, T v, weft_mem_order) {
  p->store(v);
}
template <class T> static inline T atomic_fetch_add_explicit(std::atomic<T>* p, T v, weft_mem_order) {
  return p->fetch_add(v);
}
template <class T> static inline T atomic_fetch_sub_explicit(std::atomic<T>* p, T v, weft_mem_order) {
  return p->fetch_sub(v);
}
template <class T> static inline T atomic_fetch_and_explicit(std::atomic<T>* p, T v, weft_mem_order) {
  return p->fetch_and(v);
}
template <class T> static inline T atomic_fetch_or_explicit(std::atomic<T>* p, T v, weft_mem_order) {
  return p->fetch_or(v);
}
template <class T> static inline T atomic_fetch_xor_explicit(std::atomic<T>* p, T v, weft_mem_order) {
  return p->fetch_xor(v);
}
template <class T> static inline T atomic_exchange_explicit(std::atomic<T>* p, T v, weft_mem_order) {
  return p->exchange(v);
}
template <class T>
static inline bool atomic_compare_exchange_weak_explicit(std::atomic<T>* p, T* expected, T desired,
                                                        weft_mem_order, weft_mem_order) {
  return p->compare_exchange_weak(*expected, desired);
}
)WEFT";

}  // namespace

// `blk.c` is Metal's tile-memory member access: the value of this pixel's
// slice. Lowered, it is a member of one element of an array, so it becomes
// `blk->c` for the CPU (a pointer at this pixel) and `blk[pixel].c` for the GPU
// (the same array indexed by the pixel) — one piece of memory, two spellings.
// `.slice(n)` is the n-th slice at this pixel, which is an index.
// `indexed` (optional) is used when the access carries a subscript: it is a
// template whose `@` becomes that subscript, so `imblk[p].c` can become
// `blk[p - gid].c` instead of collapsing to `blk->c`. Dropping the subscript
// would silently turn a read of a neighbouring pixel into a read of your own,
// which is not the same program — see the tile-memory note in translate_msl.
std::string rewrite_member_access(const std::string& body, const std::string& name,
                                  const std::string& prefix, std::string& err,
                                  const std::string& indexed = std::string()) {
  std::string out;
  size_t i = 0;
  while (i < body.size()) {
    if (body.compare(i, name.size(), name) != 0 || (i > 0 && is_ident(body[i - 1])) ||
        (i + name.size() < body.size() && is_ident(body[i + name.size()]))) {
      out.push_back(body[i]);
      i++;
      continue;
    }
    size_t j = i + name.size();
    while (j < body.size() && std::isspace((unsigned char)body[j])) j++;
    const bool subscript = !indexed.empty() && j < body.size() && body[j] == '[';
    if (j >= body.size() || (body[j] != '.' && !subscript)) {
      out += name;
      i = j;
      continue;
    }
    if (subscript) {
      const size_t close = find_matching(body, j, '[', ']');
      if (close == std::string::npos) {
        err = "unbalanced " + name + "[...]";
        return body;
      }
      std::string fmt = indexed;
      const size_t at = fmt.find('@');
      if (at == std::string::npos) {
        err = "internal: indexed rewrite without a placeholder";
        return body;
      }
      fmt.replace(at, 1, trim(body.substr(j + 1, close - j - 1)));
      out += fmt;
      i = close + 1;
      while (i < body.size() && std::isspace((unsigned char)body[i])) i++;
      if (i < body.size() && body[i] == '.') i++;
      continue;
    }
    size_t s = j + 1;
    while (s < body.size() && std::isspace((unsigned char)body[s])) s++;
    if (body.compare(s, 6, "slice(") == 0) {
      const size_t open = s + 5;
      const size_t close = find_matching(body, open, '(', ')');
      if (close == std::string::npos) {
        err = "unbalanced " + name + ".slice(...)";
        return body;
      }
      out += name + "[" + trim(body.substr(open + 1, close - open - 1)) + "]";
      i = close + 1;
      continue;
    }
    out += prefix;
    i = j + 1;
  }
  return out;
}

// A fragment function's `return colour;` is the write to the colour attachment:
// the target is the kernel's last buffer, one float per pixel, and the pixel is
// this invocation's grid index.
std::string rewrite_frag_return(const std::string& body, const std::string& ret,
                                  std::string& err) {
  std::string out;
  size_t i = 0;
  while (i < body.size()) {
    const bool rel = body.compare(i, 6, "return") == 0 && (i == 0 || !is_ident(body[i - 1])) &&
                     (i + 6 >= body.size() || !is_ident(body[i + 6]));
    if (!rel) { out.push_back(body[i]); i++; continue; }
    size_t j = i + 6;
    int paren = 0, bracket = 0;
    while (j < body.size()) {
      const char c = body[j];
      if (c == '(') paren++;
      else if (c == ')') paren--;
      else if (c == '[') bracket++;
      else if (c == ']') bracket--;
      else if (c == ';' && paren == 0 && bracket == 0) break;
      j++;
    }
    const std::string expr = trim(body.substr(i + 6, j - i - 6));
    if (expr.empty()) {
      err = "a fragment function has to return a colour";
      return body;
    }
    // The target is RGBA32Float: the four channels the fragment returns, at this
    // invocation's pixel. A narrower return fills what it has and leaves the rest
    // at the value the GPU gives a channel the shader never wrote — alpha 1,
    // colour 0 — so a float3 colour is an opaque pixel rather than a black one.
    // The width comes from the declared return type, because an expression's type
    // is not known here and guessing it would be the translator inventing a
    // layout out of the spelling of a variable.
    std::string fill;
    if (ret.find("4") != std::string::npos) {
      fill = "weft_px[0] = (float)weft_c.x; weft_px[1] = (float)weft_c.y;\n"
             "      weft_px[2] = (float)weft_c.z; weft_px[3] = (float)weft_c.w;";
    } else if (ret.find("3") != std::string::npos) {
      fill = "weft_px[0] = (float)weft_c.x; weft_px[1] = (float)weft_c.y;\n"
             "      weft_px[2] = (float)weft_c.z; weft_px[3] = 1.0f;";
    } else if (ret.find("2") != std::string::npos) {
      fill = "weft_px[0] = (float)weft_c.x; weft_px[1] = (float)weft_c.y;\n"
             "      weft_px[2] = 0.0f; weft_px[3] = 1.0f;";
    } else {
      fill = "weft_px[0] = weft_px[1] = weft_px[2] = (float)weft_c; weft_px[3] = 1.0f;";
    }
    out += "{ float* weft_px = weft_colour + (size_t)ctx.gid * 4u; auto weft_c = (" + expr +
           ");\n      " + fill + " return; }";
    i = (j < body.size()) ? j + 1 : body.size();
  }
  return out;
}

// The source Metal compiles: the file as written, with tile memory lowered. Tile
// memory is an imageblock, and this host's Metal compiler cannot compile one in
// any spelling, so the parameter becomes a device buffer of one element per pixel
// and `blk.c` becomes `blk[pixel].c` — the same memory the CPU binds, at the same
// index. This runs for every translation, because a compute run hands Metal the
// whole file too and would otherwise fail on a tile stage it never runs.
std::string lower_imageblocks(std::string src, uint32_t width) {
  // Tile memory is a 32x32 block per tile — the unit the hardware hands a tile
  // stage — not one slot per pixel of the target. A pixel's slot is its tile's
  // block plus its position inside it, and the same expression is emitted for the
  // GPU and for the CPU translation, so both address the same slot.
  const std::string wpix = std::to_string(width) + "u";
  const std::string wtiles = std::to_string((width + 31) / 32) + "u";
  bool any_imageblock = false;
  src = replace_all(src, "[[tile]]", "kernel");

  std::string out;
  size_t pos = 0;
  while (pos < src.size()) {
    size_t k = std::string::npos;
    std::string kw;
    for (const char* cand : {"kernel", "fragment"}) {
      const size_t kk = src.find(cand, pos);
      if (kk != std::string::npos && (k == std::string::npos || kk < k)) { k = kk; kw = cand; }
    }
    if (k == std::string::npos) break;
    if (k > 0 && is_ident(src[k - 1])) { pos = k + kw.size(); continue; }
    if (k + kw.size() < src.size() && is_ident(src[k + kw.size()])) { pos = k + kw.size(); continue; }
    const size_t paren = src.find('(', k);
    if (paren == std::string::npos) break;
    const size_t close = find_matching(src, paren, '(', ')');
    if (close == std::string::npos) break;
    const size_t brace = src.find('{', close);
    if (brace == std::string::npos) break;
    const size_t end = find_matching(src, brace, '{', '}');
    if (end == std::string::npos) break;

    const std::string params = src.substr(paren + 1, close - paren - 1);
    const std::string body = src.substr(brace + 1, end - brace - 1);
    const std::string head = trim(src.substr(k + kw.size(), paren - k - kw.size()));
    // A `kernel` with an imageblock is the pass's tile stage: a compute kernel
    // cannot have one, and every fragment is spelled `fragment`.
    const bool is_tile = (kw == "kernel");

    std::string ib_name, ib_elem, pos_name, gid_name;
    int nbuf = 0;
    for (const std::string& rawp : split_top_level(params)) {
      if (rawp.empty() || rawp == "void") continue;
      std::vector<std::string> attrs;
      const std::string d = strip_attributes(rawp, attrs);
      const std::string at = attrs.empty() ? "" : trim(attrs[0]);
      size_t e = d.size();
      while (e > 0 && std::isspace((unsigned char)d[e - 1])) e--;
      size_t b = e;
      while (b > 0 && is_ident(d[b - 1])) b--;
      const std::string pname = d.substr(b, e - b);
      if (at.rfind("buffer", 0) == 0) nbuf++;
      else if (at == "thread_position_in_grid") gid_name = pname;
      else if (at == "position") pos_name = pname;
      else if (at == "imageblock_data") {
        const size_t lt = d.find('<'), gt = d.rfind('>');
        ib_elem = (lt != std::string::npos && gt != std::string::npos && gt > lt)
                      ? trim(d.substr(lt + 1, gt - lt - 1))
                      : std::string("float4");
        ib_name = pname;
      }
    }
    if (ib_name.empty()) {
      out += src.substr(pos, end + 1 - pos);
      pos = end + 1;
      continue;
    }
    out += src.substr(pos, k - pos);
    out += kw + " " + head + "(";
    std::string gp;
    bool first = true;
    for (const std::string& rawp : split_top_level(params)) {
      if (rawp.empty() || rawp == "void") continue;
      std::vector<std::string> attrs;
      const std::string d = strip_attributes(rawp, attrs);
      const std::string at = attrs.empty() ? "" : trim(attrs[0]);
      if (!first) gp += ", ";
      first = false;
      if (at == "imageblock_data") {
        gp += "device " + ib_elem + "* " + ib_name + " [[buffer(" + std::to_string(nbuf) + ")]]";
        continue;
      }
      gp += rawp;
    }
    if (is_tile && gid_name.empty()) {
      if (!gp.empty()) gp += ", ";
      gp += "uint __weft_gid [[thread_position_in_grid]]";
      gid_name = "__weft_gid";
    }
    out += gp + "){";
    any_imageblock = true;
    const std::string own_px =
        is_tile ? "(uint)" + gid_name
                : ("((uint)" + pos_name + ".y * " + wpix + " + (uint)" + pos_name + ".x)");
    const std::string px = "weft_tile_slot(" + own_px + ", " + wpix + ", " + wtiles + ")";
    std::string err;
    // An indexed access keeps its own pixel: `blk[q].c` is a read of pixel q,
    // which is legal when q is in the same tile, so it must not collapse to the
    // invocation's own pixel here either.
    out += rewrite_member_access(body, ib_name, ib_name + "[" + px + "].", err,
                                 ib_name + "[weft_tile_slot(@, " + wpix + ", " + wtiles + ")].");
    out += "}";
    pos = end + 1;
  }
  out += src.substr(pos);
  if (any_imageblock) {
    out = "// Tile memory as the hardware gives it to a tile stage: a 32x32 block per\n"
          "// tile. A pixel's slot is its tile's block plus its position in that block.\n"
          "static inline uint weft_tile_slot(uint p, uint w, uint tiles_x) {\n"
          "    const uint x = p % w, y = p / w;\n"
          "    return (((y >> 5u) * tiles_x) + (x >> 5u)) * 1024u + ((y & 31u) << 5u) + "
          "(x & 31u);\n"
          "}\n\n" + out;
  }
  return out;
}

// The module ABI (include/weft/weft.h), emitted after the kernels. Element widths
// are `sizeof` of each buffer's declared C++ type, evaluated by the compiler that
// builds this file — so a `float4*` is 16 bytes and a `packed_float3*` is 12, with
// no layout engine here. A front end that cannot type its pointers (PTX binds
// addresses, SPIR-V pointer values) reports nothing, and 4 bytes is assumed.
static const char* kMslMath = R"WEFT(
// ---- the vector half of Metal's math library --------------------------------
//
// `using std::` above covers the scalar functions; std has nothing for float3, so
// these are the component-wise definitions Metal's specification gives, in the
// order it gives them. That order is not cosmetic: `mix` is x + (y - x) * a and
// not (1 - a) * x + a * y, and the two round differently.
inline float saturate(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }

// ---- fp16, the operations that have no rounding freedom --------------------
//
// Metal defines these for `half` and std does not, so without the exact overload
// `abs(h)` is ambiguous between the int and float ones. Each is a comparison or a
// round-to-integer, which gives the same bits however it is compiled; the
// functions that *do* have rounding freedom (sqrt, pow, sin, fma, smoothstep,
// mix) are in the portable library instead, so both engines run the same text.
inline half abs(half x) { return x < half(0) ? (half)(-x) : x; }
inline half fabs(half x) { return abs(x); }
inline half min(half a, half b) { return b < a ? b : a; }
inline half max(half a, half b) { return a < b ? b : a; }
inline half clamp(half x, half lo, half hi) { return min(max(x, lo), hi); }
inline half saturate(half x) { return clamp(x, half(0), half(1)); }
inline half step(half edge, half x) { return x < edge ? half(0) : half(1); }
inline half sign(half x) { return x > half(0) ? half(1) : (x < half(0) ? half(-1) : x); }
inline half floor(half x) { return (half)std::floor((float)x); }
inline half ceil(half x) { return (half)std::ceil((float)x); }
inline half round(half x) { return (half)std::round((float)x); }
inline half rint(half x) { return (half)std::rint((float)x); }
inline half trunc(half x) { return (half)std::trunc((float)x); }
inline half fract(half x) { return x - floor(x); }
#define WEFT_HALFV(NAME, SCALAR)                                                  \
  inline half2 NAME(half2 v) { return half2(SCALAR(v.x), SCALAR(v.y)); }           \
  inline half3 NAME(half3 v) {                                                    \
    return half3(SCALAR(v.x), SCALAR(v.y), SCALAR(v.z));                           \
  }                                                                               \
  inline half4 NAME(half4 v) {                                                    \
    return half4(SCALAR(v.x), SCALAR(v.y), SCALAR(v.z), SCALAR(v.w));              \
  }
#define WEFT_HALFVV(NAME, SCALAR)                                                 \
  inline half2 NAME(half2 a, half2 b) { return half2(SCALAR(a.x, b.x), SCALAR(a.y, b.y)); } \
  inline half3 NAME(half3 a, half3 b) {                                            \
    return half3(SCALAR(a.x, b.x), SCALAR(a.y, b.y), SCALAR(a.z, b.z));             \
  }                                                                                \
  inline half4 NAME(half4 a, half4 b) {                                            \
    return half4(SCALAR(a.x, b.x), SCALAR(a.y, b.y), SCALAR(a.z, b.z), SCALAR(a.w, b.w)); \
  }
WEFT_HALFV(abs, abs)
WEFT_HALFV(fabs, fabs)
WEFT_HALFV(saturate, saturate)
WEFT_HALFV(sign, sign)
WEFT_HALFV(floor, floor)
WEFT_HALFV(ceil, ceil)
WEFT_HALFV(round, round)
WEFT_HALFV(rint, rint)
WEFT_HALFV(trunc, trunc)
WEFT_HALFV(fract, fract)
WEFT_HALFVV(min, min)
WEFT_HALFVV(max, max)
#undef WEFT_HALFV
#undef WEFT_HALFVV
inline half3 clamp(half3 x, half3 lo, half3 hi) {
  return half3(clamp(x.x, lo.x, hi.x), clamp(x.y, lo.y, hi.y), clamp(x.z, lo.z, hi.z));
}

inline float dot(float a, float b) { return a * b; }
inline float dot(float2 a, float2 b) { return std::fma(a.y, b.y, a.x * b.x); }
inline float dot(float3 a, float3 b) {
  return std::fma(a.z, b.z, std::fma(a.y, b.y, a.x * b.x));
}
inline float dot(float4 a, float4 b) {
  return std::fma(a.w, b.w, std::fma(a.z, b.z, std::fma(a.y, b.y, a.x * b.x)));
}

inline float3 cross(float3 a, float3 b) {
  return float3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline float length_squared(float v) { return v * v; }
inline float length_squared(float2 v) { return dot(v, v); }
inline float length_squared(float3 v) { return dot(v, v); }
inline float length_squared(float4 v) { return dot(v, v); }

inline float length(float v) { return std::abs(v); }
inline float length(float2 v) { return std::sqrt(dot(v, v)); }
inline float length(float3 v) { return std::sqrt(dot(v, v)); }
inline float length(float4 v) { return std::sqrt(dot(v, v)); }

inline float distance(float a, float b) { return std::abs(a - b); }
inline float distance(float2 a, float2 b) { return length(a - b); }
inline float distance(float3 a, float3 b) { return length(a - b); }
inline float distance(float4 a, float4 b) { return length(a - b); }

// One definition per width from one name: the expression forms below are written
// out, because passing an expression where this expects a function would call it.
#define WEFT_VEC1(NAME, FN)                                                        \
  inline float2 NAME(float2 v) { return float2{FN(v.x), FN(v.y)}; }                 \
  inline float3 NAME(float3 v) { return float3{FN(v.x), FN(v.y), FN(v.z)}; }        \
  inline float4 NAME(float4 v) {                                                   \
    return float4{FN(v.x), FN(v.y), FN(v.z), FN(v.w)};                              \
  }
#define WEFT_VECS(NAME, FN)                                                        \
  inline float2 NAME(float2 v, float s) { return float2{FN(v.x, s), FN(v.y, s)}; }  \
  inline float3 NAME(float3 v, float s) {                                          \
    return float3{FN(v.x, s), FN(v.y, s), FN(v.z, s)};                              \
  }                                                                                \
  inline float4 NAME(float4 v, float s) {                                          \
    return float4{FN(v.x, s), FN(v.y, s), FN(v.z, s), FN(v.w, s)};                   \
  }
#define WEFT_MATH2(NAME, FN)                                                        \
  inline float2 NAME(float2 a, float2 b) {                                         \
    return float2{FN(a.x, b.x), FN(a.y, b.y)};                                      \
  }                                                                                \
  inline float3 NAME(float3 a, float3 b) {                                         \
    return float3{FN(a.x, b.x), FN(a.y, b.y), FN(a.z, b.z)};                        \
  }                                                                                \
  inline float4 NAME(float4 a, float4 b) {                                         \
    return float4{FN(a.x, b.x), FN(a.y, b.y), FN(a.z, b.z), FN(a.w, b.w)};           \
  }

WEFT_VEC1(saturate, saturate)
WEFT_VEC1(abs, std::abs)
WEFT_VEC1(floor, std::floor)
WEFT_VEC1(ceil, std::ceil)
WEFT_VEC1(sqrt, std::sqrt)
WEFT_VEC1(exp, std::exp)
WEFT_VEC1(log, std::log)
WEFT_VECS(pow, std::pow)
WEFT_VECS(min, std::min)
WEFT_VECS(max, std::max)
WEFT_MATH2(min, std::min)
WEFT_MATH2(max, std::max)
WEFT_MATH2(pow, std::pow)
WEFT_VECS(fmod, std::fmod)
WEFT_MATH2(fmod, std::fmod)
#undef WEFT_VEC1
#undef WEFT_VECS
#undef WEFT_VEC2

inline float fract(float x) { return x - std::floor(x); }
inline float2 fract(float2 v) { return float2{fract(v.x), fract(v.y)}; }
inline float3 fract(float3 v) { return float3{fract(v.x), fract(v.y), fract(v.z)}; }
inline float4 fract(float4 v) {
  return float4{fract(v.x), fract(v.y), fract(v.z), fract(v.w)};
}

inline float sign(float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f); }
inline float2 sign(float2 v) { return float2{sign(v.x), sign(v.y)}; }
inline float3 sign(float3 v) { return float3{sign(v.x), sign(v.y), sign(v.z)}; }
inline float4 sign(float4 v) { return float4{sign(v.x), sign(v.y), sign(v.z), sign(v.w)}; }

inline float rsqrt(float x) { return 1.0f / std::sqrt(x); }
inline float2 rsqrt(float2 v) { return float2{rsqrt(v.x), rsqrt(v.y)}; }
inline float3 rsqrt(float3 v) { return float3{rsqrt(v.x), rsqrt(v.y), rsqrt(v.z)}; }
inline float4 rsqrt(float4 v) {
  return float4{rsqrt(v.x), rsqrt(v.y), rsqrt(v.z), rsqrt(v.w)};
}

// step(edge, x): zero below the edge, one at or above it.
inline float step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }
inline float2 step(float edge, float2 v) {
  return float2{step(edge, v.x), step(edge, v.y)};
}
inline float3 step(float edge, float3 v) {
  return float3{step(edge, v.x), step(edge, v.y), step(edge, v.z)};
}
inline float4 step(float edge, float4 v) {
  return float4{step(edge, v.x), step(edge, v.y), step(edge, v.z), step(edge, v.w)};
}

// Metal has fmod (a truncated remainder), not mod: the floored remainder is a
// different function and a kernel that used it would not compile on the GPU.

inline float2 normalize(float2 v) { return v / length(v); }
inline float3 normalize(float3 v) { return v / length(v); }
inline float4 normalize(float4 v) { return v / length(v); }

inline float clamp(float x, float lo, float hi) { return std::min(std::max(x, lo), hi); }
inline float2 clamp(float2 v, float lo, float hi) {
  return float2{clamp(v.x, lo, hi), clamp(v.y, lo, hi)};
}
inline float3 clamp(float3 v, float lo, float hi) {
  return float3{clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi)};
}
inline float4 clamp(float4 v, float lo, float hi) {
  return float4{clamp(v.x, lo, hi), clamp(v.y, lo, hi), clamp(v.z, lo, hi),
                clamp(v.w, lo, hi)};
}
inline float3 clamp(float3 v, float3 lo, float3 hi) {
  return float3{clamp(v.x, lo.x, hi.x), clamp(v.y, lo.y, hi.y), clamp(v.z, lo.z, hi.z)};
}

inline float mix(float x, float y, float a) { return x + (y - x) * a; }
inline float2 mix(float2 x, float2 y, float a) {
  return float2{mix(x.x, y.x, a), mix(x.y, y.y, a)};
}
inline float3 mix(float3 x, float3 y, float a) {
  return float3{mix(x.x, y.x, a), mix(x.y, y.y, a), mix(x.z, y.z, a)};
}
inline float4 mix(float4 x, float4 y, float a) {
  return float4{mix(x.x, y.x, a), mix(x.y, y.y, a), mix(x.z, y.z, a), mix(x.w, y.w, a)};
}
inline float3 mix(float3 x, float3 y, float3 a) {
  return float3{mix(x.x, y.x, a.x), mix(x.y, y.y, a.y), mix(x.z, y.z, a.z)};
}

// smoothstep: the Hermite curve between two edges, clamped outside them.
inline float smoothstep(float e0, float e1, float x) {
  const float t = clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}
inline float2 smoothstep(float e0, float e1, float2 v) {
  return float2{smoothstep(e0, e1, v.x), smoothstep(e0, e1, v.y)};
}
inline float3 smoothstep(float e0, float e1, float3 v) {
  return float3{smoothstep(e0, e1, v.x), smoothstep(e0, e1, v.y),
                smoothstep(e0, e1, v.z)};
}
inline float4 smoothstep(float e0, float e1, float4 v) {
  return float4{smoothstep(e0, e1, v.x), smoothstep(e0, e1, v.y),
                smoothstep(e0, e1, v.z), smoothstep(e0, e1, v.w)};
}
)WEFT";

// ---- mesh and task stages, as compute, for the GPU ----
//
// A mesh pipeline is hardware this device may not implement (MTLGPUFamilyApple9,
// M3 and later). What a mesh *stage* produces is not a pipeline feature though:
// it is two buffers — vertices and triangles — written by the lanes of a
// threadgroup, and a task stage produces a payload and a meshgrid. Both are
// ordinary memory, so the GPU side is lowered to a compute kernel that writes the
// same buffers the CPU writes, and the two are compared like any other kernel.
// This is the trade tile memory already makes: the missing hardware feature is
// spelled as the memory the launcher binds.
//
// The shims below are what the body's own calls go to, so `set_vertex`, `vertices`,
// `get_vertex`, `set_primitive`, `mgp.set_threadgroups_per_grid` and the payload
// reference keep their spelling while addressing the same slots the CPU does.
const char* kGpuStageShims = R"WEFT(

template <class V, int NV, int NP>
struct weft_gpu_mesh {
  struct vrow {
    device V* b;
    uint o;
    device V& operator[](uint i) { return b[o + i]; }
  };
  struct prow {
    // A primitive is a uint3, and Metal pads it to 16 bytes. Storing a uint4
    // whose fourth word is zero makes the padding defined — the CPU mirror writes
    // the same zero — so the two primitive blocks compare byte for byte instead
    // of agreeing only in their first three words.
    device uint4* b;
    uint o;
    device uint4& operator[](uint i) { return b[o + i]; }
  };
  device V* v;
  device uint4* p;
  uint g;
  vrow vertices;
  prow primitives;
  weft_gpu_mesh(device V* mv, device uint4* mp, uint mg)
      : v(mv), p(mp), g(mg), vertices{mv, mg * (uint)NV}, primitives{mp, mg * (uint)NP} {}
  void set_vertex(uint i, V x) { v[g * (uint)NV + i] = x; }
  V get_vertex(uint i) { return v[g * (uint)NV + i]; }
  template <class P>
  void set_primitive(uint i, P t) { p[g * (uint)NP + i] = uint4(t.x, t.y, t.z, 0u); }
};

// A task stage's dispatch handle, writing the same three words the launcher reads.
// Two overloads rather than one templated on `requires`: this Metal compiler has
// no concepts, and `uint` and `uint3` are the only two forms the language takes.
struct weft_gpu_mgp {
  device uint* o;
  void set_threadgroups_per_grid(uint x) {
    o[0] = x;
    o[1] = 1u;
    o[2] = 1u;
  }
  void set_threadgroups_per_grid(uint3 p) {
    o[0] = p.x;
    o[1] = p.y;
    o[2] = p.z;
  }
};
)WEFT";

// Rewrite every `[[mesh]]`/`[[object]]` stage in the source handed to Metal into
// a compute kernel over the same buffers. The stages' own members keep their
// names; what changes is that the stage is a kernel and its outputs are bindings.
std::string lower_gpu_stages(std::string src) {
  bool used = false;
  // The parameter list that contains `at` (its own `(` and the matching `)`), and
  // the highest `[[buffer(N)]]` index in it — the stage's outputs bind above it,
  // exactly where the CPU translation puts them.
  auto enclosing = [&](size_t at) -> size_t {
    int depth = 0;
    for (size_t i = at; i-- > 0;) {
      if (src[i] == ')') depth++;
      else if (src[i] == '(') {
        if (depth == 0) return i;
        depth--;
      }
    }
    return std::string::npos;
  };
  auto closing = [&](size_t open) -> size_t {
    int depth = 0;
    for (size_t i = open; i < src.size(); i++) {
      if (src[i] == '(') depth++;
      else if (src[i] == ')' && --depth == 0) return i;
    }
    return std::string::npos;
  };
  auto max_buffer = [&](size_t par, size_t par_end) -> int {
    int m = -1;
    const std::string ps = src.substr(par, par_end - par);
    for (size_t p = 0; (p = ps.find("buffer(", p)) != std::string::npos; p += 7)
      m = std::max(m, std::atoi(ps.c_str() + p + 7));
    return m;
  };
  // Every stage needs the threadgroup index: it is what a mesh's slice and a
  // task's payload element are addressed by. A stage that already declares
  // [[threadgroup_position_in_grid]] uses that parameter — declaring a second one
  // is a redeclaration error — and one that does not gets a fresh one.
  auto tg_expr_for = [&](size_t par, size_t par_end) -> std::string {
    const std::string ps = src.substr(par, par_end - par);
    const size_t a = ps.find("threadgroup_position_in_grid");
    if (a == std::string::npos) {
      src.insert(par_end, ", uint __weft_tg [[threadgroup_position_in_grid]]");
      return "__weft_tg";
    }
    size_t e = a;  // back over the attribute's own brackets and the gap
    while (e > 0 && (ps[e - 1] == ' ' || ps[e - 1] == '\t' || ps[e - 1] == '\n' || ps[e - 1] == '['))
      e--;
    size_t b = e;
    while (b > 0 && (std::isalnum((unsigned char)ps[b - 1]) || ps[b - 1] == '_')) b--;
    return ps.substr(b, e - b);
  };
  auto words_of = [](const std::string& s) {
    std::vector<std::string> w;
    std::string cur;
    for (char c : s) {
      if (std::isspace((unsigned char)c)) {
        if (!cur.empty()) { w.push_back(cur); cur.clear(); }
      } else {
        cur.push_back(c);
      }
    }
    if (!cur.empty()) w.push_back(cur);
    return w;
  };

  // ---- mesh stages ----
  for (size_t at = 0; (at = src.find("[[mesh]]", at)) != std::string::npos;) {
    const size_t open = src.rfind("mesh<", at);
    const size_t close = open == std::string::npos ? std::string::npos : src.find('>', open);
    if (close == std::string::npos || close > at) break;
    const std::vector<std::string> parts = split_top_level(src.substr(open + 5, close - open - 5));
    const size_t par = enclosing(at);
    const size_t par_end = par == std::string::npos ? std::string::npos : closing(par);
    if (parts.size() < 3 || par_end == std::string::npos) break;
    const std::string vt = trim(parts[0]), nv = trim(parts[1]), np = trim(parts[2]);
    const std::string rest = trim(src.substr(close + 1, at - close - 1));
    const std::string mname = rest.substr(0, rest.find_first_of(" \t\n"));
    int maxb = -1;
    {
      const std::string ps = src.substr(par, par_end - par);
      for (size_t p = 0; (p = ps.find("buffer(", p)) != std::string::npos; p += 7)
        maxb = std::max(maxb, std::atoi(ps.c_str() + p + 7));
    }
    const int vb = maxb + 1, pb = maxb + 2;
    const std::string tgex = tg_expr_for(par, par_end);
    const size_t brace = src.find('{', par_end);
    if (brace == std::string::npos) break;
    src.insert(brace + 1, "\n  weft_gpu_mesh<" + vt + ", " + nv + ", " + np + "> " + mname +
                            "(__weft_mv, __weft_mp, " + tgex + ");\n");
    const std::string repl = "device " + vt + "* __weft_mv [[buffer(" + std::to_string(vb) +
                             ")]], device uint4* __weft_mp [[buffer(" + std::to_string(pb) + ")]]";
    src.replace(open, at + 8 - open, repl);
    at = open + repl.size();
    used = true;
  }
  // ---- task stages ----
  for (size_t at = 0; (at = src.find("[[object]]", at)) != std::string::npos;) {
    const size_t par = src.find('(', at);
    const size_t par_end = par == std::string::npos ? std::string::npos : closing(par);
    if (par_end == std::string::npos) break;
    // The stage's own threadgroup index, which the payload is indexed by.
    tg_expr_for(par, par_end);
    src.replace(at, 10, "kernel");
    at += 6;
    used = true;
  }

  // ---- the payload, in either stage ----
  // `object_data T& p [[payload]]` becomes a binding plus a reference to this
  // group's element — the same element the CPU binds.
  for (size_t at = 0; (at = src.find("[[payload]]", at)) != std::string::npos;) {
    const size_t par = enclosing(at);
    const size_t par_end = par == std::string::npos ? std::string::npos : closing(par);
    const size_t open = par == std::string::npos ? std::string::npos : src.find("object_data", par);
    if (par_end == std::string::npos || open == std::string::npos || open > at) break;
    const std::vector<std::string> w = words_of(src.substr(open, at - open));
    if (w.size() < 3) break;
    std::string pt = w[1], pn = w[2];
    pt.erase(std::remove(pt.begin(), pt.end(), '&'), pt.end());
    pn.erase(std::remove(pn.begin(), pn.end(), '&'), pn.end());
    const int k = max_buffer(par, par_end) + 1;
    const std::string tgex = tg_expr_for(par, par_end);
    const size_t brace = src.find('{', par_end);
    if (brace == std::string::npos) break;
    src.insert(brace + 1, "\n  device " + pt + "& " + pn + " = __weft_pl[" + tgex + "];\n");
    const std::string repl =
        "device " + pt + "* __weft_pl [[buffer(" + std::to_string(k) + ")]]";
    src.replace(open, at + 11 - open, repl);
    at = open + repl.size();
    used = true;
  }

  // ---- the meshgrid handle, in a task stage ----
  for (size_t at = 0; (at = src.find("mesh_grid_properties", at)) != std::string::npos;) {
    const size_t par = enclosing(at);
    const size_t par_end = par == std::string::npos ? std::string::npos : closing(par);
    if (par_end == std::string::npos) break;
    size_t end = src.find(',', at);
    if (end == std::string::npos || end > par_end) end = par_end;
    const std::string name = trim(src.substr(at + 20, end - (at + 20)));
    const int k = max_buffer(par, par_end) + 1;
    tg_expr_for(par, par_end);
    const size_t brace = src.find('{', par_end);
    if (brace == std::string::npos) break;
    src.insert(brace + 1, "\n  weft_gpu_mgp " + name + "{ __weft_mgp };\n");
    const std::string repl =
        "device uint* __weft_mgp [[buffer(" + std::to_string(k) + ")]]";
    src.replace(at, end - at, repl);
    at += repl.size();
    used = true;
  }
  if (!used) return src;
  // The shims need metal_stdlib and the vector types, so they go after the
  // file's includes rather than before them.
  size_t last_inc = 0;
  for (size_t p = 0; (p = src.find("#include", p)) != std::string::npos;) {
    p = src.find('\n', p);
    if (p == std::string::npos) break;
    last_inc = p + 1;
  }
  src.insert(last_inc, kGpuStageShims);
  return src;
}

std::string emit_module_abi(const std::vector<TranslatedKernel>& kernels) {
  std::string code;
  for (size_t i = 0; i < kernels.size(); i++) {
    const TranslatedKernel& k = kernels[i];
    if (k.buf_type.empty()) continue;
    code += "static const uint32_t weft_bytes_" + std::to_string(i) + "[] = {";
    for (size_t b = 0; b < k.buf_type.size(); b++) {
      // A width is either a C++ type (the Metal front end, where `sizeof` is
      // the compiler's answer and needs no layout engine here) or a byte count
      // the front end derived itself (SPIR-V, whose layout is explicit).
      const std::string& w = k.buf_type[b];
      const bool literal = !w.empty() && w.find_first_not_of("0123456789") == std::string::npos;
      code += (b ? ", " : "") + std::string("(uint32_t)") +
              (literal ? w : ("sizeof(" + w + ")"));
    }
    code += "};\n";
  }
  code += "static const weft_kernel_info weft_info[] = {\n";
  for (size_t i = 0; i < kernels.size(); i++) {
    const TranslatedKernel& k = kernels[i];
    code += "  {\"" + k.name + "\", " + std::to_string(k.nbufs) + ", " + std::to_string(k.ntex) +
            ", " + (k.buf_type.empty() ? std::string("nullptr")
                                       : ("weft_bytes_" + std::to_string(i))) +
            ", " + (k.sync ? "1" : "0") + ", " + std::to_string(k.tg_hint) + "},\n";
  }
  code +=
      "};\n"
      "extern \"C\" int weft_kernel_count() { return (int)(sizeof(weft_info) / sizeof(weft_info[0])); }\n"
      "extern \"C\" const weft_kernel_info* weft_kernel_at(int i) { return &weft_info[i]; }\n"
      "extern \"C\" int weft_kernel_find(const char* n) {\n"
      "  for (int i = 0; i < weft_kernel_count(); i++)\n"
      "    if (std::strcmp(weft_info[i].name, n) == 0) return i;\n"
      "  return -1;\n"
      "}\n"
      "extern \"C\" void weft_set_sampler(int i, int f, int a) { weft::set_sampler(i, f, a); }\n"
      // Tile memory's one undefined case, made visible: a kernel that reads a
      // pixel outside its own tile has no answer the GPU defines, so the
      // launcher can ask whether any such access happened.
      "extern \"C\" void weft_tile_reset() { weft::tile_reset(); }\n"
      "extern \"C\" uint64_t weft_tile_violations() { return weft::tile_violations(); }\n";
  return code;
}

// The math calls a shader makes that are hardware approximations on the GPU.
// Both sides are rewritten to the portable implementation, so the comparison is
// between two runs of the same code rather than of two different libraries.
// A warp prefix scan is the CPU's tree, on the GPU as well.
//
// `simd_prefix_*` there is the hardware's own fold, and for floats a different
// order of additions is a different answer — 6.1e-05 apart over 65,536 lanes. The
// CPU runtime already runs the doubling (Kogge-Stone) tree, so the shader's calls
// are pointed at that tree too (`weft::warp_prefix` in src/weft_msl.hpp) and both
// engines then perform the identical sequence of additions.
//
// This rewrites the file the author wrote, not the runtime preamble emitted in
// front of it: `warp_prefix` itself asks the hardware for an integer prefix (it is
// exact, and it is how a helper learns its lane id), and rewriting that would
// make the helper call itself.
static std::string rewrite_warp_scans(const std::string& text) {
  struct Scan {
    const char* name;
    const char* inc;
    const char* prod;
    const char* width;
  };
  // Longest name first: `..._product` contains `..._`, and the match requires the
  // '(' straight after the name anyway.
  static const Scan kScans[] = {
      {"simd_prefix_inclusive_product", "true", "true", "32u"},
      {"simd_prefix_exclusive_product", "false", "true", "32u"},
      {"simd_prefix_inclusive_sum", "true", "false", "32u"},
      {"simd_prefix_exclusive_sum", "false", "false", "32u"},
      {"quad_prefix_inclusive_product", "true", "true", "4u"},
      {"quad_prefix_exclusive_product", "false", "true", "4u"},
      {"quad_prefix_inclusive_sum", "true", "false", "4u"},
      {"quad_prefix_exclusive_sum", "false", "false", "4u"},
  };
  auto matching = [](const std::string& t, size_t open) -> size_t {
    int depth = 0;
    for (size_t i = open; i < t.size(); i++) {
      if (t[i] == '(') depth++;
      else if (t[i] == ')' && --depth == 0) return i;
    }
    return std::string::npos;
  };
  std::string out = text;
  for (const Scan& s : kScans) {
    const std::string pat = std::string(s.name) + "(";
    size_t at = 0;
    while ((at = out.find(pat, at)) != std::string::npos) {
      if (at > 0 && (std::isalnum((unsigned char)out[at - 1]) || out[at - 1] == '_')) {
        at += pat.size();
        continue;
      }
      const size_t close = matching(out, at + pat.size() - 1);
      if (close == std::string::npos) break;
      const std::string arg = out.substr(at + pat.size(), close - (at + pat.size()));
      const std::string rep = "weft_warp_prefix(" + arg + ", " + s.inc + ", " + s.prod +
                             ", " + s.width + ")";
      out.replace(at, close + 1 - at, rep);
      at += rep.size();
    }
  }
  return out;
}

std::string rewrite_math_calls(const std::string& text) {
  static const char* kNames[][2] = {
      // Longest first where one name contains another; the match also requires a
      // '(' straight after the name, so `sinh` and `sin` cannot shadow each other.
      {"inverseSqrt", "weft_rsqrt"}, {"normalize", "weft_normalize"},
      {"distance", "weft_distance"}, {"length", "weft_length"},
      {"rsqrt", "weft_rsqrt"},       {"sqrt", "weft_sqrt"},
      {"log10", "weft_log10"},       {"log2", "weft_log2"},
      {"log", "weft_log"},           {"exp10", "weft_exp10"},
      {"exp2", "weft_exp2"},         {"exp", "weft_exp"},
      {"powr", "weft_powr"},         {"pow", "weft_pow"},
      {"atan2", "weft_atan2"},       {"atan", "weft_atan"},
      {"asinh", "weft_asinh"},       {"acosh", "weft_acosh"},
      {"atanh", "weft_atanh"},       {"asin", "weft_asin"},
      {"acos", "weft_acos"},         {"sinh", "weft_sinh"},
      {"cosh", "weft_cosh"},         {"tanh", "weft_tanh"},
      {"sinpi", "weft_sinpi"},       {"cospi", "weft_cospi"},
      {"tanpi", "weft_tanpi"},       {"sin", "weft_sin"},
      {"cos", "weft_cos"},           {"tan", "weft_tan"},
      {"faceforward", "weft_faceforward"}, {"refract", "weft_refract"},
      {"dot", "weft_dot"},             {"sincos", "weft_sincos"},
      {"reflect", "weft_reflect"},   {"cross", "weft_cross"},
      {"mix", "weft_mix"},           {"fma", "weft_fma"},
      {"smoothstep", "weft_smoothstep"},
  };
  std::string out = text;
  for (const auto& nm : kNames) {
    const std::string from = nm[0], to = nm[1];
    size_t p = 0;
    while ((p = out.find(from, p)) != std::string::npos) {
      const bool left_ok = p == 0 || !is_ident(out[p - 1]);
      const size_t e = p + from.size();
      if (!left_ok || e >= out.size() || out[e] != '(') { p = e; continue; }
      // `metal::sin(x)`, `fast::pow(...)`: the qualifier names a namespace whose
      // functions are exactly the ones being replaced, so it is dropped rather
      // than left in front of a function that namespace does not define.
      size_t begin = p;
      if (p >= 2 && out[p - 1] == ':' && out[p - 2] == ':') {
        size_t k = p - 2;
        while (k > 0 && is_ident(out[k - 1])) k--;
        begin = k;
      }
      out.replace(begin, e - begin, to);
      p = begin + to.size();
    }
  }
  return out;
}

TranslateResult translate_msl(const std::string& raw, const RenderMode* render) {
  TranslateResult r;
  std::string src = strip_comments(raw);

  // A task stage (`[[object]]`) is a kernel too: it runs first, writes a payload
  // and calls `set_threadgroups_per_grid`, and the mesh stage then runs over that
  // meshgrid. Its name is recorded before the rewrite so the kernel loop knows
  // which functions are tasks.
  std::vector<std::string> task_fns;
  for (size_t at = 0; (at = src.find("[[object]]", at)) != std::string::npos;) {
    size_t after = at + 10;
    while (after < src.size() && std::isspace((unsigned char)src[after])) after++;
    if (src.compare(after, 4, "void") == 0) {
      const size_t par = src.find('(', after);
      if (par == std::string::npos) break;
      task_fns.push_back(trim(src.substr(after + 4, par - (after + 4))));
      src.replace(at, 10, "kernel");
      at += 6;
    } else {
      at += 10;
    }
  }
  // An object stage is a kernel over one mesh's slots — vertices and primitives
  // written by the lanes of the group together — so it is spelled `kernel` from
  // here on and translated like one. Only the *stage* attribute is rewritten:
  // the `[[mesh]]` on a `mesh<V, NV, NP>` parameter is a binding, and the
  // parameter loop below binds the two blocks it names.
  for (size_t at = 0; (at = src.find("[[mesh]]", at)) != std::string::npos;) {
    size_t after = at + 8;
    while (after < src.size() && std::isspace((unsigned char)src[after])) after++;
    if (src.compare(after, 4, "void") == 0) {
      src.replace(at, 8, "kernel");
      at += 6;
    } else {
      at += 8;
    }
  }

  // The Metal preamble belongs to the GPU's compilation, not to the generated
  // C++, whose runtime header supplies the same names. It is dropped from the
  // text the carried declarations are read from and kept in `src`, because `src`
  // is also what Metal is handed back (see the GPU source below).
  std::string decl_src = src;
  for (const char* inc : {"#include <metal_stdlib>\n", "#include <metal_stdlib>",
                          "#include <metal_math>\n", "#include <metal_math>",
                          "#include <metal_compute>\n", "#include <metal_compute>"}) {
    decl_src = replace_all(decl_src, inc, "");
  }
  decl_src = replace_all(decl_src, "using namespace metal;", "");

  // Everything the file defines for itself — structs, enums, constants, helper
  // functions — is carried over as written. Only a kernel is rewritten.
  const CarriedDecls carried = carry_decls(decl_src);

  std::string body_out;
  int kcount = 0;

  // A compute pass translates `kernel` functions, a render pass `fragment` ones:
  // a fragment's grid is the target's pixels, so it is the same shape of
  // translation with a different way of naming the thread.
  //
  // A `[[tile]]` function is a render pass's first stage — it fills tile memory
  // before any fragment of that tile runs — and it is shaped exactly like a
  // kernel over the same pixels, so it is translated as one and run first.
  std::vector<std::string> tile_fns;
  if (render) {
    size_t t = 0;
    while ((t = src.find("[[tile]]", t)) != std::string::npos) {
      const size_t par = src.find('(', t);
      if (par == std::string::npos) break;
      std::string thead = trim(src.substr(t + 8, par - t - 8));
      const size_t sp = thead.find_last_of(" \t\n");
      if (sp != std::string::npos) tile_fns.push_back(trim(thead.substr(sp + 1)));
      t = par;
    }
    src = replace_all(src, "[[tile]]", "kernel");
  }
  // A render pass reads `fragment` functions and the tile stage, which is
  // spelled `kernel` after the rewrite above; a compute pass only `kernel`.
  size_t pos = 0;
  while (true) {
    std::string kw;
    size_t k = std::string::npos;
    if (!render) {
      k = src.find("kernel", pos);
      kw = "kernel";
    } else {
      const size_t kf = src.find("fragment", pos), kk = src.find("kernel", pos);
      if (kf != std::string::npos && (kk == std::string::npos || kf < kk)) {
        k = kf;
        kw = "fragment";
      } else {
        k = kk;
        kw = "kernel";
      }
    }
    if (k == std::string::npos) break;
    if (k > 0 && is_ident(src[k - 1])) { pos = k + kw.size(); continue; }
    if (k + kw.size() < src.size() && is_ident(src[k + kw.size()])) {
      pos = k + kw.size();
      continue;
    }
    // Signature: `kernel void name(params)` or `fragment float4 name(params)`
    size_t paren = src.find('(', k);
    if (paren == std::string::npos) break;
    size_t semi = src.find(';', k);
    if (semi != std::string::npos && semi < paren) { pos = k + kw.size(); continue; }

    std::string head = trim(src.substr(k + kw.size(), paren - k - kw.size()));
    size_t name_start = head.find_last_of(" \t\n");
    std::string name = (name_start == std::string::npos) ? head : trim(head.substr(name_start + 1));
    if (name.empty()) { r.error = "could not read kernel name"; return r; }
    // The declared return type: a fragment's width, and `void` for a kernel.
    const std::string ret_type =
        (name_start == std::string::npos) ? std::string() : trim(head.substr(0, name_start));

    size_t close_paren = find_matching(src, paren, '(', ')');
    if (close_paren == std::string::npos) { r.error = "unbalanced parameter list in '" + name + "'"; return r; }

    size_t brace = src.find('{', close_paren);
    if (brace == std::string::npos) { r.error = "kernel '" + name + "' has no body"; return r; }
    size_t close_brace = find_matching(src, brace, '{', '}');
    if (close_brace == std::string::npos) { r.error = "unbalanced body in '" + name + "'"; return r; }

    std::string params = src.substr(paren + 1, close_paren - paren - 1);
    std::string body = src.substr(brace + 1, close_brace - brace - 1);
    body = rewrite_math_calls(body);
    const bool is_tile = std::find(tile_fns.begin(), tile_fns.end(), name) != tile_fns.end();
    const bool is_task = std::find(task_fns.begin(), task_fns.end(), name) != task_fns.end();
    // A render pass translates its fragment stages and its tile stage. A compute
    // kernel that happens to live in the same file is not part of the pass, so
    // it is left exactly as written.
    if (render && kw == "kernel" && !is_tile) {
      pos = close_brace + 1;
      continue;
    }

    // ---- parameters -> locals bound to buffers and thread indices ----
    std::string locals;      // everything, for the per-thread form
    std::string buf_locals;  // buffers only, for the loop form
    std::vector<TexParam> textures;
    std::vector<std::pair<std::string, std::string>> gid_params;
    bool only_gid = true;
    int nargs = 0;
    bool unsupported = false;
    std::string why;
    std::string ib_type, ib_name;  // tile memory: its element type and name
    // Element types by binding number, not by declaration order: a mesh stage's
    // outputs bind after every `[[buffer(N)]]` the kernel wrote, so the two can
    // never land on the same slot. `buf_bytes[i]` has to mean buffer i.
    std::vector<std::pair<int, std::string>> buf_types_at;
    // A mesh stage's outputs are bound but are not buffers the caller passes, so
    // they are counted here and subtracted from the count `--buffers` matches.
    bool is_mesh = false;
    int mesh_nv = 0, mesh_np = 0;
    std::string mesh_vtype;
    // Buffers the launcher creates rather than the caller — the mesh blocks, a
    // task payload, the meshgrid handle — and the binding each one was given.
    int extra_bufs = 0;
    int payload_idx = -1, props_idx = -1;
    std::string payload_type;

    // The highest binding the kernel declares. Mesh outputs bind above it, so
    // they can never collide with an index the shader itself wrote.
    int max_buf_idx = -1;
    for (const std::string& rp : split_top_level(params)) {
      const size_t ba = rp.find("buffer(");
      if (ba == std::string::npos) continue;
      const int bi = std::atoi(rp.c_str() + ba + 7);
      if (bi > max_buf_idx) max_buf_idx = bi;
    }
    int next_idx = max_buf_idx + 1;  // the first binding the launcher supplies

    for (const std::string& rawp : split_top_level(params)) {
      if (rawp.empty() || rawp == "void") continue;
      std::vector<std::string> attrs;
      std::string decl = strip_attributes(rawp, attrs);
      // `mesh_grid_properties mgp` is a plain parameter, not a binding: it is the
      // task stage's own output about the size of the stage after it. It is bound
      // above everything the shader declared, like a mesh block.
      if (decl.find("mesh_grid_properties") != std::string::npos) {
        const std::string rest = trim(decl.substr(decl.find("mesh_grid_properties") + 20));
        std::string gname = rest.substr(0, rest.find_first_of(" \t,)"));
        gname = trim(gname);
        if (gname.empty()) {
          unsupported = true;
          why = "mesh_grid_properties needs a name: " + rawp;
          break;
        }
        props_idx = next_idx++;
        if (nargs <= props_idx) nargs = props_idx + 1;
        extra_bufs += 1;
        buf_types_at.emplace_back(props_idx, "uint32_t");
        const std::string gline =
            "  weft::mesh_grid_properties " + gname +
            " = weft::mesh_grid_properties{ (uint32_t*)weft::buffer(ctx, " +
            std::to_string(props_idx) + ") };\n";
        locals += gline;
        buf_locals += gline;
        continue;
      }
      if (attrs.empty()) {
        unsupported = true;
        why = "parameter without a binding: " + rawp;
        break;
      }
      std::string attr = trim(attrs[0]);
      std::string name_field, type_field;

      // A mesh stage's output: `mesh<V, NV, NP> out [[mesh]]`. It is not a buffer
      // the caller passes — a mesh is a threadgroup, so the launcher creates one
      // vertex block and one triangle block per group and the view below is this
      // invocation's slice of them. `out.vertices[i]`, `out.set_vertex(i, v)`,
      // `out.set_primitive(i, uint3)` and `out.get_vertex(i)` are the members a
      // mesh object stage writes through.
      if (attr == "mesh") {
        const size_t lt = decl.find('<'), gt = decl.rfind('>');
        if (lt == std::string::npos || gt == std::string::npos || gt < lt) {
          unsupported = true;
          why = "mesh<> needs a vertex type, a vertex count and a primitive count: " + rawp;
          break;
        }
        const std::vector<std::string> ms = split_top_level(decl.substr(lt + 1, gt - lt - 1));
        if (ms.size() < 3 || !std::isdigit((unsigned char)trim(ms[1])[0]) ||
            !std::isdigit((unsigned char)trim(ms[2])[0])) {
          unsupported = true;
          why = "mesh<> wants `mesh<V, max_vertices, max_primitives>` with integer counts: " + rawp;
          break;
        }
        const std::string vtype = trim(ms[0]);
        // `mesh<V, 16, 16> out` and `mesh<V, 16, 16> out[kind::triangle]` are the
        // two spellings; the kind only picks the primitive type, and both write
        // triangles.
        std::string rest = trim(decl.substr(gt + 1));
        const std::string mname = trim(rest.substr(0, rest.find('[')));
        if (mname.empty() || vtype.empty()) {
          unsupported = true;
          why = "mesh<> needs a name and a vertex type: " + rawp;
          break;
        }
        const int vb = next_idx++, pb = next_idx++;
        if (nargs <= pb) nargs = pb + 1;
        extra_bufs += 2;
        is_mesh = true;
        mesh_vtype = vtype;
        mesh_nv = std::atoi(trim(ms[1]).c_str());
        mesh_np = std::atoi(trim(ms[2]).c_str());
        buf_types_at.emplace_back(vb, vtype);
        buf_types_at.emplace_back(pb, "weft::MeshPrim");
        const std::string mt =
            "weft::mesh_view<" + vtype + ", " + std::to_string(mesh_nv) + ", " +
            std::to_string(mesh_np) + ">";
        // The view is this group's slice: `buffer` is the whole block, and a
        // group's part of it starts NV (or NP) elements in. That offset is the
        // only thing that stops one mesh from writing another's vertices.
        const std::string mline =
            "  " + mt + " " + mname + " = " + mt + "{ (" + vtype +
            "*)weft::buffer(ctx, " + std::to_string(vb) + ") + (size_t)ctx.tgid * " +
            std::to_string(mesh_nv) + "u, (weft::MeshPrim*)weft::buffer(ctx, " +
            std::to_string(pb) + ") + (size_t)ctx.tgid * " + std::to_string(mesh_np) + "u };\n";
        locals += mline;
        buf_locals += mline;
        only_gid = false;  // the slice depends on the group, not the thread
        continue;
      }

      // The task payload: `object_data T& p [[payload]]`. One element per task
      // threadgroup, written by the task stage and read by the mesh stage that
      // group spawned — which is why the module indexes it by `tgid % groups`
      // rather than by the mesh index alone.
      if (attr == "payload") {
        std::vector<std::string> pw;
        std::string cur;
        for (char c : decl) {
          if (std::isspace((unsigned char)c)) {
            if (!cur.empty()) { pw.push_back(cur); cur.clear(); }
          } else {
            cur.push_back(c);
          }
        }
        if (!cur.empty()) pw.push_back(cur);
        if (pw.size() < 3 || pw[0] != "object_data") {
          unsupported = true;
          why = "a payload is `object_data T& name [[payload]]`: " + rawp;
          break;
        }
        std::string ptype = pw[1], pname = pw[2];
        ptype.erase(std::remove(ptype.begin(), ptype.end(), '&'), ptype.end());
        pname.erase(std::remove(pname.begin(), pname.end(), '&'), pname.end());
        payload_idx = next_idx++;
        if (nargs <= payload_idx) nargs = payload_idx + 1;
        extra_bufs += 1;
        payload_type = ptype;
        buf_types_at.emplace_back(payload_idx, ptype);
        // The payload is indexed by this invocation's own threadgroup: the
        // launcher gives a task group one element and a mesh exactly the element
        // of the task group that owns it, so neither stage borrows an index it
        // has to compute.
        const std::string pline = "  " + ptype + "& " + pname + " = ((" + ptype +
                                  "*)weft::buffer(ctx, " + std::to_string(payload_idx) +
                                  "))[ctx.tgid];\n";
        locals += pline;
        buf_locals += pline;
        only_gid = false;
        continue;
      }

      // A texture is not a buffer: it has its own binding space, its own size
      // query and a filtering sampler, so it is bound to a weft::Texture.
      if (decl.find("texture") != std::string::npos) {
        if (attr.rfind("texture", 0) != 0) {
          unsupported = true;
          why = "a texture parameter needs [[texture(N)]]: " + rawp;
          break;
        }
        const std::string tf = trim(decl);
        size_t ne = tf.size();
        while (ne > 0 && !is_ident(tf[ne - 1])) ne--;
        size_t nb = ne;
        while (nb > 0 && is_ident(tf[nb - 1])) nb--;
        const std::string tname = tf.substr(nb, ne - nb);
        std::string tval = "float";
        const size_t lt = tf.find('<'), gt = tf.rfind('>');
        if (lt != std::string::npos && gt != std::string::npos && gt > lt) {
          const std::string inner = tf.substr(lt + 1, gt - lt - 1);
          tval = trim(inner.substr(0, inner.find(',')));
        }
        const size_t a = attr.find('('), b = attr.find(')');
        const std::string idx = (a != std::string::npos && b > a) ? attr.substr(a + 1, b - a - 1) : "";
        if (idx.empty()) { unsupported = true; why = "texture() without an index"; break; }
        std::string line = "  const weft::Texture " + tname + " = weft::texture(" + idx + ");\n";
        locals += line;
        buf_locals += line;  // a texture does not depend on the thread
        textures.push_back(TexParam{tname, tval});
        continue;
      }
      {
        std::vector<std::string> words;
        std::string cur;
        for (char c : decl) {
          if (std::isspace((unsigned char)c)) {
            if (!cur.empty()) { words.push_back(cur); cur.clear(); }
          } else cur.push_back(c);
        }
        if (!cur.empty()) words.push_back(cur);
        if (words.empty()) { unsupported = true; why = "empty parameter"; break; }
        int stars = 0;
        for (char c : decl) if (c == '*') stars++;
        name_field = words.back();
        name_field.erase(std::remove(name_field.begin(), name_field.end(), '*'), name_field.end());
        if (words.size() >= 2) {
          type_field = words[words.size() - 2];
          type_field.erase(std::remove(type_field.begin(), type_field.end(), '*'), type_field.end());
        }
        bool is_const = false;
        for (size_t wi = 0; wi + 1 < words.size(); wi++) {
          std::string w = words[wi];
          if (w == "const") is_const = true;
        }
        // Tile memory: `imageblock<T> blk`. It is lowered to a buffer of T, one
        // element per pixel, so it binds after the kernel's own buffers and is
        // addressed by the pixel — the same memory the GPU source below hands
        // Metal.
        if (attr == "imageblock_data") {
          if (!render) {
            unsupported = true;
            why = "an imageblock is tile memory, which only a render pass has (" + rawp + ")";
            break;
          }
          const size_t lt = type_field.find('<'), gt = type_field.rfind('>');
          if (lt == std::string::npos || gt == std::string::npos || gt < lt) {
            unsupported = true;
            why = "imageblock needs an element type: " + rawp;
            break;
          }
          ib_type = trim(type_field.substr(lt + 1, gt - lt - 1));
          ib_name = name_field;
          continue;
        }
        if (stars == 0 && attr.rfind("buffer", 0) == 0) {
          unsupported = true;
          why = "only pointer parameters can bind a buffer (" + rawp + ")";
          break;
        }
        if (attr.rfind("buffer", 0) == 0) {
          size_t a = attr.find('('), b = attr.find(')');
          std::string idx = (a != std::string::npos && b > a) ? attr.substr(a + 1, b - a - 1) : "";
          if (idx.empty()) { unsupported = true; why = "buffer() without an index"; break; }
          std::string line = "  " + std::string(is_const ? "const " : "") + type_field + "* " +
                             name_field + " = (" + std::string(is_const ? "const " : "") +
                             type_field + "*)weft::buffer(ctx, " + idx + ");\n";
          const int bidx = std::atoi(trim(idx).c_str());
          locals += line;
          buf_locals += line;
          buf_types_at.emplace_back(bidx, type_field);
          if (nargs <= bidx) nargs = bidx + 1;
          continue;
        }
        // `[[position]]` in a fragment is the pixel centre: (x + 0.5, y + 0.5),
        // with x taken modulo the target's width. That is the same coordinate a
        // GPU hands a fragment, so a body that uses it computes the same thing.
        if (render && attr == "position") {
          locals += "  float4 " + name_field + " = float4((float)(ctx.gid % " +
                     std::to_string(render->width) + "u) + 0.5f, (float)(ctx.gid / " +
                     std::to_string(render->width) + "u) + 0.5f, 0.0f, 1.0f);\n";
          only_gid = false;
          continue;
        }
        if (attr == "thread_position_in_grid") {
          locals += "  " + type_field + " " + name_field + " = (" + type_field + ")ctx.gid;\n";
          gid_params.emplace_back(type_field, name_field);
          continue;
        }
        // Group-relative indices depend on the thread, so they rule out the
        // loop form. Counts do not, so they are hoisted instead.
        if (attr == "thread_position_in_threadgroup" || attr == "thread_index_in_threadgroup") {
          locals += "  " + type_field + " " + name_field + " = (" + type_field + ")ctx.lid;\n";
          only_gid = false;
          continue;
        }
        if (attr == "threadgroup_position_in_grid") {
          locals += "  " + type_field + " " + name_field + " = (" + type_field + ")ctx.tgid;\n";
          only_gid = false;
          continue;
        }
        // The sub-warp indices: a simdgroup is 32 lanes of a threadgroup, a quad
        // is 4. Both are functions of the lane, so they rule out the loop form.
        if (attr == "thread_index_in_simdgroup") {
          locals += "  " + type_field + " " + name_field + " = (" + type_field +
                    ")(ctx.lid & 31u);\n";
          only_gid = false;
          continue;
        }
        if (attr == "simdgroup_index_in_threadgroup") {
          locals += "  " + type_field + " " + name_field + " = (" + type_field +
                    ")(ctx.lid >> 5);\n";
          only_gid = false;
          continue;
        }
        if (attr == "thread_index_in_quadgroup") {
          locals += "  " + type_field + " " + name_field + " = (" + type_field +
                    ")(ctx.lid & 3u);\n";
          only_gid = false;
          continue;
        }
        if (attr == "quadgroup_index_in_threadgroup") {
          locals += "  " + type_field + " " + name_field + " = (" + type_field +
                    ")(ctx.lid >> 2);\n";
          only_gid = false;
          continue;
        }
        if (attr == "threads_per_simdgroup") {
          std::string line = "  " + type_field + " " + name_field + " = (" + type_field + ")32u;\n";
          locals += line;
          buf_locals += line;
          continue;
        }
        if (attr == "threads_per_threadgroup") {
          std::string line =
              "  " + type_field + " " + name_field + " = (" + type_field + ")ctx.tg_size;\n";
          locals += line;
          buf_locals += line;
          continue;
        }
        if (attr == "threads_per_grid") {
          std::string line = "  " + type_field + " " + name_field + " = (" + type_field + ")ctx.grid;\n";
          locals += line;
          buf_locals += line;
          continue;
        }
        // The counts, which are quotients of the two sizes the launcher knows.
        // A last threadgroup or last group of lanes can be short, and a GPU
        // still reports a whole one, so these round up.
        if (attr == "threadgroups_per_grid") {
          std::string line = "  " + type_field + " " + name_field + " = (" + type_field +
                             ")((ctx.grid + ctx.tg_size - 1u) / ctx.tg_size);\n";
          locals += line;
          buf_locals += line;
          continue;
        }

      }
      unsupported = true;
      why = "unsupported parameter binding [[" + attr + "]]";
      break;
    }
    if (unsupported) { r.error = "kernel '" + name + "': " + why; return r; }

    // ---- body rewrites ----
    bool shared_found = false, barrier_found = false;
    std::string berr;
    int tile_idx = -1, colour_idx = -1;
    const int user_nbufs = nargs;
    if (render) {
      // Tile memory is an imageblock, and this host's Metal compiler cannot
      // compile one in any spelling. Both sides therefore lower it to the same
      // thing: one element per pixel of the target, addressed by the pixel's
      // index. The tile stage writes it and the fragments of that tile read it,
      // which is exactly what tile memory is for a pixel that only touches its
      // own slot.
      if (!ib_name.empty()) {
        const std::string idx = std::to_string(nargs);
        tile_idx = nargs;
        nargs += 1;
        // The base of all tile memory: the slot is computed per access, because a
        // tile is a block and the invocation's own pixel is not offset 0 of it.
        locals += "  " + ib_type + "* " + ib_name + " = (" + ib_type + "*)weft::buffer(ctx, " +
                  idx + ");\n";
        // An indexed access keeps its subscript and goes through `tile_px`,
        // which counts any pixel outside this invocation's 32x32 tile: the GPU
        // leaves that read undefined, so a kernel doing it has no answer that
        // two runs could agree on and the launcher refuses it.
        const std::string wpix = std::to_string(render->width) + "u";
        const std::string wtiles = std::to_string((render->width + 31) / 32) + "u";
        const std::string at_own =
            "(" + ib_name + " + weft::tile_slot(ctx, ctx.gid, " + wpix + ", " + wtiles + "))->";
        const std::string own = ib_name + "[weft::tile_slot(ctx, (@), " + wpix + ", " + wtiles + ")].";
        body = rewrite_member_access(body, ib_name, at_own, berr, own);
        if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }
      }
      if (!is_tile) {
        // The colour attachment, four floats per pixel (RGBA32Float), which is
        // where `return` writes.
        const std::string colour = std::to_string(nargs);
        locals += "  float* weft_colour = (float*)weft::buffer(ctx, " + colour + ");\n";
        colour_idx = nargs;
        nargs += 1;
        body = rewrite_frag_return(body, ret_type, berr);
        if (!berr.empty()) { r.error = "fragment '" + name + "': " + berr; return r; }
      }
    }
    // A helper that waits on its warp is expanded here, so the suspension it
    // needs is this kernel's.
    for (const HelperFn& h : carried.helpers) {
      bool hit = false;
      body = expand_helper_calls(body, h, hit, berr);
      if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }
    }
    body = rewrite_shared_decls(body, shared_found, berr);
    if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }
    body = rewrite_barriers(body, barrier_found, berr);
    if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }

    std::map<std::string, std::pair<int, int>> samplers;
    if (!textures.empty() || body.find("sampler") != std::string::npos) {
      body = strip_samplers(body, samplers, berr);
      if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }
    }
    if (!textures.empty()) {
      body = rewrite_textures(body, textures, samplers, berr);
      if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }
    }

    body = replace_all(body, "metal::", "");
    body = replace_all(body, "fast::", "");
    body = replace_all(body, "precise::", "");
    // The ballot's return type. A warp is 32 lanes and a quad is 4, so the mask
    // is a 32-bit word whatever width the GPU's vote type happens to have (it is
    // 64 bits on Apple silicon), and the vote rewrite hands back one of those.
    body = replace_all(body, "simd_vote::vote_t", "uint32_t");
    body = replace_all(body, "quad_vote::vote_t", "uint32_t");
    body = replace_all(body, "simd_vote", "uint32_t");
    body = replace_all(body, "quad_vote", "uint32_t");

    body = rewrite_as_type(body, berr);
    if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }

    bool shfl_found = false, lane_expr = false;
    body = rewrite_group_ops(body, shfl_found, lane_expr, berr);
    if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }
    // Anything from the warp/quad library the table above does not know: name it
    // rather than emit C++ that will not compile.
    for (const char* pre : {"simd_", "quad_"}) {
      const size_t sp = body.find(pre);
      if (sp == std::string::npos) continue;
      size_t e = sp + 5;
      while (e < body.size() && is_ident(body[e])) e++;
      r.error = "kernel '" + name + "': " + body.substr(sp, e - sp) + " is not supported";
      return r;
    }

    const bool is_sync = shared_found || barrier_found || shfl_found;
    if (is_sync) {
      // Emit a coroutine: the compiler owns the frame, so the body's locals
      // survive a barrier without switching stacks. This is what makes a Metal
      // barrier cost a resume instead of a context switch.
      std::string cbody = replace_all(body, "weft::barrier(ctx);", "co_await weft::barrier_async();");
      cbody = replace_all(cbody, "return;", "co_return;");
      body_out += "static weft::Task weft_k_" + name + "(weft::Ctx& ctx) {\n" + locals + "\n" + cbody +
                  "\n  co_return;\n}\n\n";
    } else {
      body_out += "static void weft_k_" + name + "(weft::Ctx& ctx) {\n" + locals + "\n" + body +
                  "\n}\n\n";
    }

    // A `return` (or break/continue) inside the body would leave the loop form
    // early, so kernels with one keep the per-thread form.
    const bool has_jump = contains_word(body, "return") || contains_word(body, "break") ||
                          contains_word(body, "continue");
    // Tile memory is a pointer at this pixel rather than a buffer the loop form
    // can hoist out of it, so a kernel that has one keeps the per-thread form.
    const bool can_flat = !(shared_found || barrier_found || shfl_found) && only_gid &&
                          !has_jump && !lane_expr && ib_name.empty();
    if (can_flat) {
      std::string gid_decls;
      for (const auto& g : gid_params)
        gid_decls += "    " + g.first + " " + g.second + " = (" + g.first + ")weft_i;\n";
      // The induction variable is named for the runtime, not for the kernel: a
      // kernel local called `i` would otherwise be redeclared inside the loop.
      body_out += "static void weft_k_" + name + "_loop(weft::Ctx& ctx, uint32_t begin, uint32_t end) {\n" +
                  buf_locals + "  for (uint32_t weft_i = begin; weft_i < end; weft_i++) {\n" + gid_decls +
                  "    " + body + "\n  }\n}\n\n";
    }

    TranslatedKernel tk;
    tk.name = name;
    tk.nargs = nargs;
    // Mesh blocks, payloads, the meshgrid handle and tile memory are created by
    // the launcher, so they are bound but are not part of `--buffers`: only the
    // caller's own bindings count.
    tk.nbufs = user_nbufs - extra_bufs;
    // One width per binding, in binding order. A slot nothing declared reports
    // 0 (unknown) rather than a guessed 4, so `weft_arg_bytes` keeps its fallback.
    std::vector<std::string> buf_types((size_t)nargs, std::string("0"));
    for (const auto& bt : buf_types_at)
      if (bt.first >= 0 && (size_t)bt.first < buf_types.size())
        buf_types[(size_t)bt.first] = bt.second;
    tk.buf_type = buf_types;
    tk.tile_idx = tile_idx;
    tk.colour_idx = colour_idx;
    tk.tile = is_tile;
    tk.mesh = is_mesh;
    tk.mesh_nv = mesh_nv;
    tk.mesh_np = mesh_np;
    tk.task = is_task;
    tk.payload_idx = payload_idx;
    tk.props_idx = props_idx;
    tk.ntex = (int)textures.size();
    tk.sync = shared_found || barrier_found || shfl_found;
    tk.flat = can_flat;
    r.kernels.push_back(tk);
    kcount++;


    pos = close_brace + 1;
  }
  r.gpu_src = lower_gpu_stages(
      rewrite_warp_scans(lower_imageblocks(src, render ? render->width : 0)));

  if (render) {
    int tile_stages = 0;
    const TranslatedKernel* chosen = nullptr;
    for (const auto& k : r.kernels) {
      if (k.tile) {
        tile_stages++;
      } else if (!chosen && (render->fragment.empty() || k.name == render->fragment)) {
        chosen = &k;
      }
    }
    // One tile stage per pass: two would leave the launcher guessing which one
    // fills tile memory, and guessing is how a wrong answer ships.
    if (tile_stages > 1) {
      r.error = "a render pass has one `[[tile]]` stage; this file has " +
                std::to_string(tile_stages);
      return r;
    }
    // A fragment that reads tile memory needs a stage that fills it. Without one
    // it reads whatever the launcher left there, and both sides would agree on a
    // number neither of them computed.
    if (chosen && chosen->tile_idx >= 0 && tile_stages == 0) {
      r.error = "fragment '" + chosen->name +
                "' reads tile memory, but the pass has no `[[tile]]` stage to fill it";
      return r;
    }
  }

  if (kcount == 0) {
    r.error = "no kernels found (expected `kernel void name(...)`)";
    return r;
  }

  std::string code =
      "// generated by weft — do not edit\n"
      "#include \"weft_runtime.hpp\"\n"
      "#include <algorithm>\n"
      "#include <atomic>\n"
      "#include <cmath>\n"
      "#include <cstdint>\n"
      "#include <cstring>\n";
  code += kMslPreamble;
  // Metal's short integer names are part of the language, so they have to exist
  // before the file's own declarations: a `struct Payload { uint n; }` is written
  // above the kernel that uses it and must compile there, not only inside a body.
  code += "using uint = unsigned int;\n"
      "using ushort = unsigned short;\n"
      "using uchar = unsigned char;\n";
  // glibc and the BSDs declare `ulong` themselves, in <sys/types.h> and as
  // `unsigned long`. Declaring it again as `unsigned long long` is a conflicting
  // declaration rather than the same alias twice, so those hosts keep the
  // library's name — under LP64 it is the same 64-bit unsigned integer Metal
  // means — and the assertion turns a 32-bit target into a message instead of a
  // silently narrow `ulong`.
#if defined(__linux__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
  code += "static_assert(sizeof(ulong) == 8, \"weft: ulong is not 64 bits on this target\");\n";
#else
  code += "using ulong = unsigned long long;\n";
#endif
  code += kMslMath;  // vector math, which std does not have
  // The file's own declarations, after the library they may use.
  code += carried.text;
  // as_type<T> becomes this: a bit-for-bit reinterpretation.
  code += "template <class T, class U>\n"
      "static inline T weft_bitcast(U u) { T t; std::memcpy(&t, &u, sizeof(T)); return t; }\n";
  code += "using std::min;\n"
      "using std::max;\n"
      "using std::abs;\n"
      "using std::sqrt;\n"
      "using std::exp;\n"
      "using std::log;\n"
      "using std::pow;\n"
      "using std::fma;\n"
      "\n" +
      body_out + "static const weft::KernelDef weft_table[] = {\n";
  for (const auto& k : r.kernels) {
    code += "  {\"" + k.name + "\", " +
            (k.sync ? std::string("nullptr") : ("weft_k_" + k.name)) + ", " +
            std::to_string(k.nargs) + ", " + (k.sync ? "true" : "false") + ", " +
            (k.flat ? ("weft_k_" + k.name + "_loop") : std::string("nullptr")) +
            ", nullptr, 0, " + (k.sync ? ("weft_k_" + k.name) : std::string("nullptr")) +
            ", nullptr, 0},\n";
  }
  code +=
      "};\n"
      "extern \"C\" void weft_set_param(int i, uint64_t v) { weft::set_param(i, v); }\n"
      "extern \"C\" void weft_set_texture(int i, void* d, uint32_t w, uint32_t h, uint32_t e) {\n"
      "  weft::set_texture(i, d, w, h, e);\n"
      "}\n"
      "extern \"C\" void weft_run(int i, void** bufs, uint32_t grid, uint32_t tg) {\n"
      "  weft::dispatch(weft_table[i], bufs, grid, tg);\n"
      "}\n";
  code += emit_module_abi(r.kernels);

  r.code = code;
  r.ok = true;
  // Portable math goes in after the vector types and before the library that
  // uses it, so the translated code and the GPU run the same implementation.
  const size_t soft_at = r.code.find("inline float dot(float a, float b)");
  if (soft_at != std::string::npos) r.code.insert(soft_at, kWSoftMath);
  return r;
}
