#include "translate.hpp"
#include "soft_math.hpp"
#include "weft_f64.hpp"

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
  int slot_n = 0;  // the ordinal of the threadgroup declaration being rewritten
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
             ">(ctx, " + std::to_string(slot_n++) + ");";
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

std::string rewrite_swizzles(const std::string& body, std::string& err) {
  static const auto comp = [](char c) -> int {
    return c == 'x' || c == 'r' ? 0
           : c == 'y' || c == 'g' ? 1
           : c == 'z' || c == 'b' ? 2
           : c == 'w' || c == 'a' ? 3
                                 : -1;
  };
  // Walk back over a name, and through the `.` that joins one to the next.
  const auto name_start = [&](size_t q) {
    while (q > 0 && (is_ident(body[q - 1]) || body[q - 1] == '.')) q--;
    return q;
  };
  std::string out;
  size_t copied = 0, i = 0;
  std::vector<std::pair<size_t, size_t>> map{{0, 0}};
  while (i < body.size()) {
    if (body[i] != '.' || comp(i + 1 < body.size() ? body[i + 1] : '\0') < 0) {
      i++;
      continue;
    }
    size_t j = i + 1;
    while (j < body.size() && comp(body[j]) >= 0) j++;
    const size_t len = j - i - 1;
    const char before = i ? body[i - 1] : '\0';
    const bool operand_before = is_ident(before) || before == ')' || before == ']' || before == '}';
    // A whole token.
    const char after_name = j < body.size() ? body[j] : '\0';
    if (len < 2 || len > 4 || !operand_before || is_ident(after_name)) {
      i++;
      continue;
    }
    // Two or more components, one spelling. `xg` is not a swizzle.
    bool xyzw = true, rgba = true;
    for (size_t k = i + 1; k < j; k++) {
      if (body[k] != 'x' && body[k] != 'y' && body[k] != 'z' && body[k] != 'w') xyzw = false;
      if (body[k] != 'r' && body[k] != 'g' && body[k] != 'b' && body[k] != 'a') rgba = false;
    }
    if (!xyzw && !rgba) {
      err = "mixed component names in '." + body.substr(i + 1, len) + "'";
      return body;
    }
    // The operand, and its first character.
    size_t p = i;
    while (p > 0 && std::isspace((unsigned char)body[p - 1])) p--;
    size_t start = p;
    const char end = p ? body[p - 1] : '\0';
    if (end == ')' || end == ']' || end == '}') {
      const char open = end == ')' ? '(' : (end == ']' ? '[' : '{');
      int depth = 0;
      size_t q = p;
      while (q > 0) {
        q--;
        if (body[q] == end) depth++;
        else if (body[q] == open) {
          depth--;
          if (!depth) break;
        }
      }
      start = name_start(q);
      size_t t = q;
      while (t > 0 && std::isspace((unsigned char)body[t - 1])) t--;
      if (t > 0 && body[t - 1] == '>') {
        int d = 0;
        size_t r = t;
        while (r > 0) {
          r--;
          if (body[r] == '>') d++;
          else if (body[r] == '<') {
            d--;
            if (!d) break;
          }
        }
        start = name_start(r);
      }
    } else {
      start = name_start(p);
      if (start == p) {
        i++;
        continue;
      }
    }
    size_t after = j;
    while (after < body.size() && std::isspace((unsigned char)body[after])) after++;
    const char op_ch = after < body.size() ? body[after] : '\0';
    const bool compound = after + 1 < body.size() && body[after + 1] == '=' &&
                          std::string("+-*/%&|^<>").find(op_ch) != std::string::npos;
    const bool is_write = compound ||
                          (after < body.size() && body[after] == '=' &&
                           (after + 1 >= body.size() || body[after + 1] != '='));
    out.append(body, copied, p - copied);
    size_t operand_out = out.size();
    if (start < copied) {
      size_t at = 0;
      for (size_t m = map.size(); m > 0; m--) {
        if (map[m - 1].first <= start) {
          at = map[m - 1].second + (start - map[m - 1].first);
          break;
        }
      }
      operand_out = at;
    } else {
      operand_out = out.size() - (p - start);
    }
    const std::string operand_text = out.substr(operand_out);
    out.resize(operand_out);
    std::string idxs;
    for (size_t k = i + 1; k < j; k++)
      idxs += (idxs.empty() ? "" : ", ") + std::to_string(comp(body[k]));
    std::string call =
        (is_write ? "weft_swz_w" + std::to_string(len) + "(" : "weft_swz(") + operand_text;
    call += ", " + idxs;
    size_t rhs_end = j;
    if (is_write) {
      rhs_end = after + (compound ? 2 : 1);
      int depth = 0;
      for (size_t q = rhs_end; q < body.size(); q++) {
        const char ch = body[q];
        if (ch == '(' || ch == '[' || ch == '{') depth++;
        else if (ch == ')' || ch == ']' || ch == '}') {
          depth--;
          if (depth < 0) { rhs_end = q; break; }
        } else if (depth == 0 && (ch == ';' || ch == ',')) {
          rhs_end = q;
          break;
        }
        rhs_end = q + 1;
      }
      const size_t rhs_from = after + (compound ? 2 : 1);
      std::string rhs = body.substr(rhs_from, rhs_end - rhs_from);
      std::string rerr;
      rhs = rewrite_swizzles(rhs, rerr);
      if (!rerr.empty()) {
        err = rerr;
        return body;
      }
      call += ", " + (compound ? "weft_swz(" + operand_text + ", " + idxs + ") " +
                                     std::string(1, op_ch) + " " + rhs
                               : rhs);
    }
    call += ")";
    out += call;
    map.push_back({is_write ? rhs_end : j, out.size()});
    copied = is_write ? rhs_end : j;
    i = copied;
  }
  out.append(body, copied, std::string::npos);
  return out;
}

// as_type<T>(x).
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

size_t statement_start(const std::string& s, size_t at) {
  size_t start = 0;
  int paren = 0;
  for (size_t i = 0; i < at && i < s.size(); i++) {
    const char c = s[i];
    if (c == '(') paren++;
    else if (c == ')') paren--;
    else if (paren == 0 && (c == ';' || c == '{' || c == '}')) start = i + 1;
  }
  return start;
}

bool has_word(const std::string& s, const char* w) {
  const size_t n = std::strlen(w);
  for (size_t p = s.find(w); p != std::string::npos; p = s.find(w, p + 1)) {
    if ((p == 0 || !is_ident(s[p - 1])) && (p + n >= s.size() || !is_ident(s[p + n]))) return true;
  }
  return false;
}

// `return expr;` -> `hv = (expr);`, with no jump.
std::string helper_returns_flat(const std::string& body, const std::string& retname,
                                const std::string& endlbl, bool& jumped) {
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
    std::string rep;
    if (!expr.empty()) rep += retname + " = (" + expr + "); ";
    bool trailing = false;
    for (size_t k = j + 1; k < body.size(); k++) {
      const char c = body[k];
      if (std::isspace((unsigned char)c) || c == '}') continue;
      trailing = true;
      break;
    }
    if (trailing) {
      rep += "goto " + endlbl + "; ";
      jumped = true;
    }
    if (!rep.empty()) out += "{ " + rep + "} ";
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

    std::string repl;
    bool hoisted = false;
    if (f.suspends) {
      const size_t at = statement_start(out, out.size());
      const std::string seg = out.substr(at);
      bool once = !has_word(seg, "if") && !has_word(seg, "else") && !has_word(seg, "for") &&
                  !has_word(seg, "while") && !has_word(seg, "do") &&
                  !has_word(seg, "switch") && !has_word(seg, "case");
      for (const char* bad : {"?", "&&", "||", "++", "--", "({"})
        if (seg.find(bad) != std::string::npos) once = false;
      if (once) {
        const std::string tag = std::to_string(i) + "_" + std::to_string(out.size());
        const std::string hv = "__weft_hv" + tag;
        const std::string he = "__weft_he" + tag;
        bool jumped = false;
        const std::string flat_body = helper_returns_flat(body, hv, he, jumped);
        std::string blk = (f.ret == "void") ? std::string() : (f.ret + " " + hv + "; ");
        blk += "{ ";
        for (size_t k = 0; k < f.params.size(); k++) {
          blk += f.params[k].first + (f.params[k].first.empty() ? "" : " ") + "__weft_a" +
                 std::to_string(k) + " = (" + trim(args[k]) + "); ";
        }
        blk += "{ " + flat_body + "} ";
        if (jumped) blk += he + ":; ";
        blk += "} ";
        out.insert(at, blk);
        repl = (f.ret == "void") ? std::string("((void)0)") : hv;
        hoisted = true;
      }
    }
    if (!hoisted) {
      body = helper_returns(body, "__weft_r");
      repl = "({ ";
      if (f.ret != "void") repl += f.ret + " __weft_r{}; ";
      for (size_t k = 0; k < f.params.size(); k++) {
        repl += f.params[k].first + (f.params[k].first.empty() ? "" : " ") + "__weft_a" +
                std::to_string(k) + " = (" + trim(args[k]) + "); ";
      }
      repl += "{ " + body + " } __weft_end: ";
      repl += (f.ret == "void") ? "(void)0; }) " : "__weft_r; }) ";
    }
    out += repl;
    hit = true;
    i = close + 1;
  }
  return out;
}

// Everything a kernel file declares for itself.
struct CarriedDecls {
  std::string text;
  std::vector<HelperFn> helpers;
};

CarriedDecls carry_decls(const std::string& src) {
  CarriedDecls c;
  for (const std::string& item : top_level_items(src)) {
    if (item.empty()) continue;
    std::string bare = item;
    bool stage_attr = false;
    while (bare.compare(0, 2, "[[") == 0) {
      const size_t end = bare.find("]]");
      if (end == std::string::npos) break;
      const std::string attr = bare.substr(2, end - 2);
      if (attr == "fragment" || attr == "kernel" || attr == "vertex" || attr == "tile")
        stage_attr = true;
      bare.erase(0, end + 2);
      const size_t nz = bare.find_first_not_of(" \t\r\n");
      bare.erase(0, nz == std::string::npos ? bare.size() : nz);
    }
    if (stage_attr) continue;                      // rewritten as a stage body
    if (starts_with(bare, "kernel")) continue;    // rewritten as a kernel body
    if (starts_with(bare, "fragment")) continue;  // rewritten as a per-pixel body
    if (starts_with(bare, "vertex")) continue;    // the launcher supplies it
    if (starts_with(item, "#include") || starts_with(item, "#pragma")) continue;
    if (starts_with(item, "vertex")) continue;
    if (item.compare(0, 8, "[[tile]]") == 0) continue;
    size_t ns = 0, par = 0;
    if (looks_like_function(item, ns, par)) {
      HelperFn f = parse_helper(item, ns, par);
      const bool shares = f.body.find("threadgroup") != std::string::npos;
      if ((f.suspends || shares) && !f.ret.empty() && !f.body.empty()) {
        c.helpers.push_back(f);
        continue;
      }
    }
    std::string swz_err;
    c.text += rewrite_swizzles(rewrite_math_calls(replace_all(item, "constant ", "const ")),
                               swz_err) + "\n\n";
  }
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

// A texture argument.
struct TexParam {
  std::string name;
  std::string value;  // "float", "uint", "int" or "half"
  bool array = false;
  bool is3d = false;   // `texture3d`: every read and sample names a plane
  bool cube = false;   // `texturecube`: a sample takes a direction
  bool is1d = false;
};

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
      if (member == "get_depth") { out += "(" + t.name + ".layers)"; i = close + 1; hit = true; break; }
      if (member == "get_array_size") {
        out += (t.array || t.cube) ? ("(" + t.name + ".layers)") : std::string("(1u)");
        i = close + 1;
        hit = true;
        break;
      }
      if (member == "get_num_mip_levels") {
        out += "(1u)";
        i = close + 1;
        hit = true;
        break;
      }

      auto args = split_top_level(in.substr(p + 1, close - p - 1));
      const std::string vec = (t.value == "float")   ? "float4"
                              : (t.value == "half") ? "half4"
                              : (t.value == "int")  ? "int4"
                                                    : "uint4";
      const std::string zero = (t.value == "float" || t.value == "half") ? "0" : "0u";
      const std::string one = (t.value == "float" || t.value == "half") ? "1" : "1u";
      if (member == "read") {
        if (t.cube) {
          err = "texture '" + t.name + "' is a cube, so there is no texel to read: sample "
                "it with a direction";
          return in;
        }
        if (t.is1d) {
          // One row, so a texel is an x and there is no y to name.
          if (args.size() != 1) {
            err = "a 1D texture read takes one coordinate (x)";
            return in;
          }
          out += vec + "{weft::tex_read<" + t.value + ">(" + t.name + ", (uint32_t)(" +
                 trim(args[0]) + "), 0u), " + zero + ", " + zero + ", " + one + "}";
          i = close + 1;
          hit = true;
          break;
        }
        if (t.is3d) {
          // A plane, not a slice: the coordinate's third component is the depth.
          if (args.size() != 1) {
            err = "a 3D texture read takes one coordinate (x, y, z)";
            return in;
          }
          const std::string c = trim(args[0]);
          out += vec + "{weft::tex_read<" + t.value + ">(" + t.name + ", (" + c + ").x, (" + c +
                 ").y, (" + c + ").z), " + zero + ", " + zero + ", " + one + "}";
          i = close + 1;
          hit = true;
          break;
        }
        const bool layered = args.size() == 2;
        if (args.size() != 1 && !layered) {
          err = "texture read takes a coordinate, and a layer on an array texture";
          return in;
        }
        if (layered && !t.array) {
          err = "texture '" + t.name + "' is not an array, so read takes one coordinate";
          return in;
        }
        const std::string c = trim(args[0]);
        const std::string lay =
            layered ? (", (uint32_t)(" + trim(args[1]) + ")") : std::string();
        out += vec + "{weft::tex_read<" + t.value + ">(" + t.name + ", (" + c + ").x, (" + c +
               ").y" + lay + "), " + zero + ", " + zero + ", " + one + "}";
        i = close + 1;
        hit = true;
        break;
      }
      if (member == "sample") {
        if (t.cube) {
          if (args.size() != 2) {
            err = "a cube sample takes a sampler and a direction";
            return in;
          }
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
          const std::string call =
              "weft::tex_sample_cube<float>(" + t.name + ", (" + c + ").x, (" + c + ").y, (" + c +
              ").z, " + std::to_string(s->second.first) + ", " + std::to_string(s->second.second) +
              ")";
          out += vec + "{" + (t.value == "half" ? "(half)" + call : call) + ", " + zero + ", " +
                 zero + ", " + one + "}";
          i = close + 1;
          hit = true;
          break;
        }
        if (t.is1d) {
          if (args.size() != 2) {
            err = "a 1D texture sample takes a sampler and one coordinate";
            return in;
          }
          if (t.value != "float" && t.value != "half") {
            err = "sample() of a " + t.value + " texture (only float textures are filterable)";
            return in;
          }
          auto s = samplers.find(trim(args[0]));
          if (s == samplers.end()) {
            err = "sampler '" + trim(args[0]) + "' is not a constexpr sampler in this kernel";
            return in;
          }
          const std::string call =
              "weft::tex_sample<float>(" + t.name + ", (float)(" + trim(args[1]) +
              "), 0.5f, " + std::to_string(s->second.first) + ", " +
              std::to_string(s->second.second) + ")";
          out += vec + "{" + (t.value == "half" ? "(half)" + call : call) + ", " + zero + ", " +
                 zero + ", " + one + "}";
          i = close + 1;
          hit = true;
          break;
        }
        if (t.is3d) {
          if (args.size() != 2) {
            err = "a 3D texture sample takes a sampler and a coordinate (x, y, z)";
            return in;
          }
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
          const std::string call =
              "weft::tex_sample<float>(" + t.name + ", (" + c + ").x, (" + c + ").y, (" + c +
              ").z, " + std::to_string(s->second.first) + ", " + std::to_string(s->second.second) +
              ")";
          out += vec + "{" + (t.value == "half" ? "(half)" + call : call) + ", " + zero + ", " +
                 zero + ", " + one + "}";
          i = close + 1;
          hit = true;
          break;
        }
        const bool layered = args.size() == 3;
        if (args.size() != 2 && !layered) {
          err = "texture sample takes a sampler and a coordinate, and a layer on an array";
          return in;
        }
        if (layered && !t.array) {
          err = "texture '" + t.name + "' is not an array, so sample takes no layer";
          return in;
        }
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
        const std::string lay =
            layered ? (", (uint32_t)(" + trim(args[2]) + ")") : std::string();
        const std::string call = "weft::tex_sample<float>(" + t.name + ", (" + c + ").x, (" + c +
                                 ").y" + lay + ", " + std::to_string(s->second.first) + ", " +
                                 std::to_string(s->second.second) + ")";
        out += vec + "{" + (t.value == "half" ? "(half)" + call : call) + ", " + zero + ", " +
               zero + ", " + one + "}";
        i = close + 1;
        hit = true;
        break;
      }
      // A storage image.
      if (member == "write") {
        if (t.cube) {
          err = "texture '" + t.name + "' is a cube, which cannot be written: a face has no "
                "texel address without a direction";
          return in;
        }
        if (args.size() != 2 && args.size() != 3) {
          err = "texture write takes a value, a coordinate and an optional level";
          return in;
        }
        const std::string c = trim(args[1]);
        if (t.is3d)
          out += "weft::tex_write(" + t.name + ", (uint32_t)(" + c + ").x, (uint32_t)(" + c +
                 ").y, (uint32_t)(" + c + ").z, " + trim(args[0]) + ")";
        else
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
               width + ", " + std::to_string((unsigned long long)i + 1ull) + "u))";
      } else if (kind == "reduce") {
        if (a.size() != 1) {
          err = std::string(op.name) + " takes one value";
          return body;
        }
        out += "(co_await weft::grp_reduce(ctx, (" + trim(a[0]) + "), " + op.mode + ", " +
               width + ", " + std::to_string((unsigned long long)i + 1ull) + "u))";
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

const char* kMslPreamble = R"WEFT(
// Vector types with Metal's sizes and alignments.
using half = _Float16;
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
  // A vector followed by scalars, which Metal's constructors accept.
  float4(float2 v, float a, float b) : x(v.x), y(v.y), z(a), w(b) {}
  float4(float a, float2 v, float b) : x(a), y(v.x), z(v.y), w(b) {}
  float4(float a, float b, float2 v) : x(a), y(b), z(v.x), w(v.y) {}
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
// The 64-bit and boolean vectors.
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

// packed_float3 is the 12 bytes a device buffer of xyz triples holds.
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

// every promoted-integer operator would be a lot of code for no use.
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

// column, which is why a float3x3 is 48 bytes and not 36.
struct alignas(8) float2x2 {
  float2 c[2];
  float2x2() {}
  float2x2(float d) : c{float2(d, 0.0f), float2(0.0f, d)} {}
  float2x2(float2 a, float2 b) : c{a, b} {}
  float2x2(float a0, float a1, float b0, float b1) : c{float2(a0, a1), float2(b0, b1)} {}
  float2& operator[](int i) { return c[i]; }
  const float2& operator[](int i) const { return c[i]; }
};
struct alignas(16) float3x3 {
  float3 c[3];
  float3x3() {}
  float3x3(float d) : c{float3(d, 0.0f, 0.0f), float3(0.0f, d, 0.0f), float3(0.0f, 0.0f, d)} {}
  float3x3(float3 a, float3 b, float3 d) : c{a, b, d} {}
  float3x3(float a0, float a1, float a2, float b0, float b1, float b2, float d0, float d1,
           float d2)
      : c{float3(a0, a1, a2), float3(b0, b1, b2), float3(d0, d1, d2)} {}
  float3& operator[](int i) { return c[i]; }
  const float3& operator[](int i) const { return c[i]; }
};
struct alignas(16) float4x4 {
  float4 c[4];
  float4x4() {}
  float4x4(float d)
      : c{float4(d, 0, 0, 0), float4(0, d, 0, 0), float4(0, 0, d, 0), float4(0, 0, 0, d)} {}
  float4x4(float4 a, float4 b, float4 d, float4 e) : c{a, b, d, e} {}
  float4x4(float a0, float a1, float a2, float a3, float b0, float b1, float b2, float b3,
           float d0, float d1, float d2, float d3, float e0, float e1, float e2, float e3)
      : c{float4(a0, a1, a2, a3), float4(b0, b1, b2, b3), float4(d0, d1, d2, d3),
          float4(e0, e1, e2, e3)} {}
  float4& operator[](int i) { return c[i]; }
  const float4& operator[](int i) const { return c[i]; }
};
struct alignas(4) half2x2 {
  half2 c[2];
  half2x2() {}
  half2x2(half d) : c{half2(d, half(0)), half2(half(0), d)} {}
  half2x2(half2 a, half2 b) : c{a, b} {}
  half2& operator[](int i) { return c[i]; }
  const half2& operator[](int i) const { return c[i]; }
};
struct alignas(8) half3x3 {
  half3 c[3];
  half3x3() {}
  half3x3(half d)
      : c{half3(d, half(0), half(0)), half3(half(0), d, half(0)), half3(half(0), half(0), d)} {}
  half3x3(half3 a, half3 b, half3 d) : c{a, b, d} {}
  half3& operator[](int i) { return c[i]; }
  const half3& operator[](int i) const { return c[i]; }
};
struct alignas(8) half4x4 {
  half4 c[4];
  half4x4() {}
  half4x4(half d) : c{half4(d, half(0), half(0), half(0)), half4(half(0), d, half(0), half(0)),
                       half4(half(0), half(0), d, half(0)), half4(half(0), half(0), half(0), d)} {}
  half4x4(half4 a, half4 b, half4 d, half4 e) : c{a, b, d, e} {}
  half4& operator[](int i) { return c[i]; }
  const half4& operator[](int i) const { return c[i]; }
};
static_assert(sizeof(float2x2) == 16 && sizeof(float3x3) == 48 && sizeof(float4x4) == 64,
              "a float matrix is its columns: a float3x3 keeps the padded 16-byte column");
static_assert(sizeof(half2x2) == 8 && sizeof(half3x3) == 24 && sizeof(half4x4) == 32,
              "a half matrix is its columns: 4, 8 and 8 bytes each");
static inline float weft_lane(float2 v, int i) { return i == 0 ? v.x : v.y; }
static inline float weft_lane(float3 v, int i) { return i == 0 ? v.x : i == 1 ? v.y : v.z; }
static inline float weft_lane(float4 v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}
static inline half weft_lane(half2 v, int i) { return i == 0 ? v.x : v.y; }
static inline half weft_lane(half3 v, int i) { return i == 0 ? v.x : i == 1 ? v.y : v.z; }
static inline half weft_lane(half4 v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}
#define WEFT_MAT(MT, VT, S, N)                                                     \
  static inline VT operator*(MT m, VT v) {                                         \
    VT r = m.c[0] * weft_lane(v, 0);                                               \
    for (int i = 1; i < N; i++) r = r + m.c[i] * weft_lane(v, i);                  \
    return r;                                                                      \
  }                                                                                \
  static inline MT operator*(MT a, MT b) {                                         \
    MT r;                                                                          \
    for (int i = 0; i < N; i++) r.c[i] = a * b.c[i];                               \
    return r;                                                                      \
  }                                                                                \
  static inline MT operator*(MT a, S s) {                                          \
    MT r;                                                                          \
    for (int i = 0; i < N; i++) r.c[i] = a.c[i] * s;                               \
    return r;                                                                      \
  }                                                                                \
  static inline MT operator*(S s, MT a) { return a * s; }                          \
  static inline MT operator+(MT a, MT b) {                                         \
    MT r;                                                                          \
    for (int i = 0; i < N; i++) r.c[i] = a.c[i] + b.c[i];                          \
    return r;                                                                      \
  }                                                                                \
  static inline MT operator-(MT a, MT b) {                                         \
    MT r;                                                                          \
    for (int i = 0; i < N; i++) r.c[i] = a.c[i] - b.c[i];                          \
    return r;                                                                      \
  }                                                                                \
  static inline MT operator-(MT a) {                                               \
    MT r;                                                                          \
    for (int i = 0; i < N; i++) r.c[i] = -a.c[i];                                  \
    return r;                                                                      \
  }
WEFT_MAT(float2x2, float2, float, 2)
WEFT_MAT(float3x3, float3, float, 3)
WEFT_MAT(float4x4, float4, float, 4)
WEFT_MAT(half2x2, half2, half, 2)
WEFT_MAT(half3x3, half3, half, 3)
WEFT_MAT(half4x4, half4, half, 4)
WEFT_VEC2(int2, int32_t)
WEFT_VEC3(int3, int32_t)
WEFT_VEC4(int4, int32_t)
WEFT_VEC2(uint2, uint32_t)
WEFT_VEC3(uint3, uint32_t)
WEFT_VEC4(uint4, uint32_t)
// type, which is what makes an overflow wrap where the hardware wraps.
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
// and `.zyx` is a float3 in the other order.
#define WEFT_SWZ(T2, T3, T4, S)                                                    \
  static inline S weft_pick(T2 v, int i) { return i == 0 ? v.x : v.y; }             \
  static inline S weft_pick(T3 v, int i) {                                          \
    return i == 0 ? v.x : i == 1 ? v.y : (S)v.z;                                    \
  }                                                                                 \
  static inline S weft_pick(T4 v, int i) {                                          \
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? (S)v.z : (S)v.w;                  \
  }                                                                                 \
  static inline T2 weft_swz(T2 v, int a, int b) {                                   \
    return T2{weft_pick(v, a), weft_pick(v, b)};                                    \
  }                                                                                 \
  static inline T3 weft_swz(T2 v, int a, int b, int c) {                            \
    return T3{weft_pick(v, a), weft_pick(v, b), weft_pick(v, c)};                   \
  }                                                                                 \
  static inline T2 weft_swz(T3 v, int a, int b) {                                   \
    return T2{weft_pick(v, a), weft_pick(v, b)};                                    \
  }                                                                                 \
  static inline T3 weft_swz(T3 v, int a, int b, int c) {                            \
    return T3{weft_pick(v, a), weft_pick(v, b), weft_pick(v, c)};                   \
  }                                                                                 \
  static inline T4 weft_swz(T3 v, int a, int b, int c, int d) {                     \
    return T4{weft_pick(v, a), weft_pick(v, b), weft_pick(v, c), weft_pick(v, d)};  \
  }                                                                                 \
  static inline T2 weft_swz(T4 v, int a, int b) {                                   \
    return T2{weft_pick(v, a), weft_pick(v, b)};                                    \
  }                                                                                 \
  static inline T3 weft_swz(T4 v, int a, int b, int c) {                            \
    return T3{weft_pick(v, a), weft_pick(v, b), weft_pick(v, c)};                   \
  }                                                                                 \
  static inline T4 weft_swz(T4 v, int a, int b, int c, int d) {                     \
    return T4{weft_pick(v, a), weft_pick(v, b), weft_pick(v, c), weft_pick(v, d)};  \
  }
WEFT_SWZ(float2, float3, float4, float)
WEFT_SWZ(half2, half3, half4, half)
WEFT_SWZ(int2, int3, int4, int32_t)
WEFT_SWZ(uint2, uint3, uint4, uint32_t)

// locals first, so `a.yx = a.xy` swaps rather than reading a member it wrote.
#define WEFT_PUT(T2, T3, T4, S)                                                        \
  static inline void weft_put(T2& v, int i, S x) { if (i == 0) v.x = x; else v.y = x; } \
  static inline void weft_put(T3& v, int i, S x) {                                     \
    if (i == 0) v.x = x; else if (i == 1) v.y = x; else v.z = x;                        \
  }                                                                                    \
  static inline void weft_put(T4& v, int i, S x) {                                     \
    if (i == 0) v.x = x; else if (i == 1) v.y = x; else if (i == 2) v.z = x; else v.w = x; \
  }
WEFT_PUT(float2, float3, float4, float)
WEFT_PUT(half2, half3, half4, half)
WEFT_PUT(int2, int3, int4, int32_t)
WEFT_PUT(uint2, uint3, uint4, uint32_t)

template <class D, class V>
static inline void weft_swz_w2(D& d, int a, int b, V v) {
  const auto t0 = weft_pick(v, 0);
  const auto t1 = weft_pick(v, 1);
  weft_put(d, a, t0);
  weft_put(d, b, t1);
}

template <class D, class V>
static inline void weft_swz_w3(D& d, int a, int b, int c, V v) {
  const auto t0 = weft_pick(v, 0);
  const auto t1 = weft_pick(v, 1);
  const auto t2 = weft_pick(v, 2);
  weft_put(d, a, t0);
  weft_put(d, b, t1);
  weft_put(d, c, t2);
}

template <class D, class V>
static inline void weft_swz_w4(D& d, int a, int b, int c, int e, V v) {
  const auto t0 = weft_pick(v, 0);
  const auto t1 = weft_pick(v, 1);
  const auto t2 = weft_pick(v, 2);
  const auto t3 = weft_pick(v, 3);
  weft_put(d, a, t0);
  weft_put(d, b, t1);
  weft_put(d, c, t2);
  weft_put(d, e, t3);
}

static inline long2 make_long2(int64_t x, int64_t y) { return long2{x, y}; }
static inline long3 make_long3(int64_t x, int64_t y, int64_t z) { return long3{x, y, z, 0}; }
static inline ulong2 make_ulong2(uint64_t x, uint64_t y) { return ulong2{x, y}; }
static inline double2 make_double2(double x, double y) { return double2{x, y}; }
static inline double3 make_double3(double x, double y, double z) { return double3{x, y, z, 0.0}; }
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

// kernel that writes `q |= m` or `q <<= 2` is ordinary Metal.
#define WEFT_BITS2(T)                                                                  \
  static inline T operator&(T a, T b) {                                                \
    T r{};                                                                             \
    weft_put(r, 0, weft_pick(a, 0) & weft_pick(b, 0));                                 \
    weft_put(r, 1, weft_pick(a, 1) & weft_pick(b, 1));                                 \
    return r;                                                                          \
  }                                                                                    \
  static inline T operator|(T a, T b) {                                                \
    T r{};                                                                             \
    weft_put(r, 0, weft_pick(a, 0) | weft_pick(b, 0));                                 \
    weft_put(r, 1, weft_pick(a, 1) | weft_pick(b, 1));                                 \
    return r;                                                                          \
  }                                                                                    \
  static inline T operator^(T a, T b) {                                                \
    T r{};                                                                             \
    weft_put(r, 0, weft_pick(a, 0) ^ weft_pick(b, 0));                                 \
    weft_put(r, 1, weft_pick(a, 1) ^ weft_pick(b, 1));                                 \
    return r;                                                                          \
  }                                                                                    \
  static inline T operator<<(T a, int s) {                                             \
    T r{};                                                                             \
    weft_put(r, 0, weft_pick(a, 0) << s);                                              \
    weft_put(r, 1, weft_pick(a, 1) << s);                                              \
    return r;                                                                          \
  }                                                                                    \
  static inline T operator>>(T a, int s) {                                             \
    T r{};                                                                             \
    weft_put(r, 0, weft_pick(a, 0) >> s);                                              \
    weft_put(r, 1, weft_pick(a, 1) >> s);                                              \
    return r;                                                                          \
  }                                                                                    \
  static inline T operator~(T a) {                                                     \
    T r{};                                                                             \
    weft_put(r, 0, ~weft_pick(a, 0));                                                  \
    weft_put(r, 1, ~weft_pick(a, 1));                                                  \
    return r;                                                                          \
  }                                                                                    \
  static inline T& operator&=(T& a, T b) { a = a & b; return a; }                      \
  static inline T& operator|=(T& a, T b) { a = a | b; return a; }                      \
  static inline T& operator^=(T& a, T b) { a = a ^ b; return a; }                      \
  static inline T& operator<<=(T& a, int s) { a = a << s; return a; }                  \
  static inline T& operator>>=(T& a, int s) { a = a >> s; return a; }

#define WEFT_BITS3(T)                                                                  \
  static inline T operator&(T a, T b) {                                                \
    T r{};                                                                             \
    weft_put(r, 0, weft_pick(a, 0) & weft_pick(b, 0));                                 \
    weft_put(r, 1, weft_pick(a, 1) & weft_pick(b, 1));                                 \
    weft_put(r, 2, weft_pick(a, 2) & weft_pick(b, 2));                                 \
    return r;                                                                          \
  }                                                                                    \
  static inline T operator|(T a, T b) {                                                \
    T r{};                                                                             \
    weft_put(r, 0, weft_pick(a, 0) | weft_pick(b, 0));                                 \
    weft_put(r, 1, weft_pick(a, 1) | weft_pick(b, 1));                                 \
    weft_put(r, 2, weft_pick(a, 2) | weft_pick(b, 2));                                 \
    return r;                                                                          \
  }                                                                                    \
  static inline T operator^(T a, T b) {                                                \
    T r{};                                                                             \
    weft_put(r, 0, weft_pick(a, 0) ^ weft_pick(b, 0));                                 \
    weft_put(r, 1, weft_pick(a, 1) ^ weft_pick(b, 1));                                 \
    weft_put(r, 2, weft_pick(a, 2) ^ weft_pick(b, 2));                                 \
    return r;                                                                          \
  }                                                                                    \
  static inline T operator<<(T a, int s) {                                             \
    T r{};                                                                             \
    weft_put(r, 0, weft_pick(a, 0) << s);                                              \
    weft_put(r, 1, weft_pick(a, 1) << s);                                              \
    weft_put(r, 2, weft_pick(a, 2) << s);                                              \
    return r;                                                                          \
  }                                                                                    \
  static inline T operator>>(T a, int s) {                                             \
    T r{};                                                                             \
    weft_put(r, 0, weft_pick(a, 0) >> s);                                              \
    weft_put(r, 1, weft_pick(a, 1) >> s);                                              \
    weft_put(r, 2, weft_pick(a, 2) >> s);                                              \
    return r;                                                                          \
  }                                                                                    \
  static inline T operator~(T a) {                                                     \
    T r{};                                                                             \
    weft_put(r, 0, ~weft_pick(a, 0));                                                  \
    weft_put(r, 1, ~weft_pick(a, 1));                                                  \
    weft_put(r, 2, ~weft_pick(a, 2));                                                  \
    return r;                                                                          \
  }                                                                                    \
  static inline T& operator&=(T& a, T b) { a = a & b; return a; }                      \
  static inline T& operator|=(T& a, T b) { a = a | b; return a; }                      \
  static inline T& operator^=(T& a, T b) { a = a ^ b; return a; }                      \
  static inline T& operator<<=(T& a, int s) { a = a << s; return a; }                  \
  static inline T& operator>>=(T& a, int s) { a = a >> s; return a; }

#define WEFT_BITS4(T)                                                                  \
  static inline T operator&(T a, T b) {                                                \
    T r{};                                                                             \
    weft_put(r, 0, weft_pick(a, 0) & weft_pick(b, 0));                                 \
    weft_put(r, 1, weft_pick(a, 1) & weft_pick(b, 1));                                 \
    weft_put(r, 2, weft_pick(a, 2) & weft_pick(b, 2));                                 \
    weft_put(r, 3, weft_pick(a, 3) & weft_pick(b, 3));                                 \
    return r;                                                                          \
  }                                                                                    \
  static inline T operator|(T a, T b) {                                                \
    T r{};                                                                             \
    weft_put(r, 0, weft_pick(a, 0) | weft_pick(b, 0));                                 \
    weft_put(r, 1, weft_pick(a, 1) | weft_pick(b, 1));                                 \
    weft_put(r, 2, weft_pick(a, 2) | weft_pick(b, 2));                                 \
    weft_put(r, 3, weft_pick(a, 3) | weft_pick(b, 3));                                 \
    return r;                                                                          \
  }                                                                                    \
  static inline T operator^(T a, T b) {                                                \
    T r{};                                                                             \
    weft_put(r, 0, weft_pick(a, 0) ^ weft_pick(b, 0));                                 \
    weft_put(r, 1, weft_pick(a, 1) ^ weft_pick(b, 1));                                 \
    weft_put(r, 2, weft_pick(a, 2) ^ weft_pick(b, 2));                                 \
    weft_put(r, 3, weft_pick(a, 3) ^ weft_pick(b, 3));                                 \
    return r;                                                                          \
  }                                                                                    \
  static inline T operator<<(T a, int s) {                                             \
    T r{};                                                                             \
    weft_put(r, 0, weft_pick(a, 0) << s);                                              \
    weft_put(r, 1, weft_pick(a, 1) << s);                                              \
    weft_put(r, 2, weft_pick(a, 2) << s);                                              \
    weft_put(r, 3, weft_pick(a, 3) << s);                                              \
    return r;                                                                          \
  }                                                                                    \
  static inline T operator>>(T a, int s) {                                             \
    T r{};                                                                             \
    weft_put(r, 0, weft_pick(a, 0) >> s);                                              \
    weft_put(r, 1, weft_pick(a, 1) >> s);                                              \
    weft_put(r, 2, weft_pick(a, 2) >> s);                                              \
    weft_put(r, 3, weft_pick(a, 3) >> s);                                              \
    return r;                                                                          \
  }                                                                                    \
  static inline T operator~(T a) {                                                     \
    T r{};                                                                             \
    weft_put(r, 0, ~weft_pick(a, 0));                                                  \
    weft_put(r, 1, ~weft_pick(a, 1));                                                  \
    weft_put(r, 2, ~weft_pick(a, 2));                                                  \
    weft_put(r, 3, ~weft_pick(a, 3));                                                  \
    return r;                                                                          \
  }                                                                                    \
  static inline T& operator&=(T& a, T b) { a = a & b; return a; }                      \
  static inline T& operator|=(T& a, T b) { a = a | b; return a; }                      \
  static inline T& operator^=(T& a, T b) { a = a ^ b; return a; }                      \
  static inline T& operator<<=(T& a, int s) { a = a << s; return a; }                  \
  static inline T& operator>>=(T& a, int s) { a = a >> s; return a; }

WEFT_BITS2(int2)
WEFT_BITS3(int3)
WEFT_BITS4(int4)
WEFT_BITS2(uint2)
WEFT_BITS3(uint3)
WEFT_BITS4(uint4)
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

// d = a * b + c, with a Rows x K, b K x Cols and c/d Rows x Cols.
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

struct FragTarget {
  std::string field;
  std::string type;
  int index = 0;
  int width = 4;
};

// What a fragment's return type says about where its values go.
struct FragLayout {
  int width = 0;  // scalar or vector return: 1 to 4 channels, 0 for a struct
  std::string type_name;  // the return type as the shader spells it
  std::vector<FragTarget> colours;
  bool depth = false;
  std::string depth_field;
  std::string depth_qual;
};

std::string depth_test_src(const std::string& qual, const std::string& value) {
  const std::string cur = "weft_depth[ctx.gid]";
  if (qual == "less") return value + " < " + cur;
  if (qual == "less_equal") return value + " <= " + cur;
  if (qual == "greater") return value + " > " + cur;
  if (qual == "greater_equal") return value + " >= " + cur;
  if (qual == "equal") return value + " == " + cur;
  if (qual == "not_equal") return value + " != " + cur;
  if (qual == "never") return "false";
  return std::string();  // any: the value is stored whatever it is
}

bool parse_frag_layout(const std::string& src, const std::string& ret, FragLayout& lay,
                       std::string& err) {
  std::string t = trim(ret);
  for (;;) {
    if (t.rfind("[[", 0) == 0) {
      const size_t rb = t.find("]]");
      if (rb == std::string::npos) break;
      t = trim(t.substr(rb + 2));
      continue;
    }
    if (!t.empty() && t[0] == ']') {
      t = trim(t.substr(1));
      continue;
    }
    break;
  }
  std::string base;
  size_t i = 0;
  while (i < t.size() && (std::isalpha((unsigned char)t[i]) || t[i] == '_')) base += t[i++];
  const bool vec = base == "float" || base == "half";
  lay.type_name = base;
  if (vec && i == t.size()) { lay.width = 1; return true; }
  if (vec && i + 1 == t.size() && t[i] >= '1' && t[i] <= '4') {
    lay.width = t[i] - '0';
    return true;
  }
  if (base.empty()) {
    err = "could not read the fragment's return type";
    return false;
  }
  const std::string key = "struct " + base;
  size_t at = src.find(key);
  while (at != std::string::npos) {
    const size_t after = at + key.size();
    const bool whole = after >= src.size() || !is_ident(src[after]);
    if (whole) {
      const size_t b = src.find('{', after);
      if (b != std::string::npos && trim(src.substr(after, b - after)).empty()) {
        const size_t e = find_matching(src, b, '{', '}');
        if (e == std::string::npos) break;
        const std::string fields = src.substr(b + 1, e - b - 1);
        size_t s = 0;
        while (s < fields.size()) {
          const size_t semi = fields.find(';', s);
          if (semi == std::string::npos) break;
          const std::string decl = trim(fields.substr(s, semi - s));
          s = semi + 1;
          const size_t ca = decl.find("[[color(");
          const size_t da = decl.find("[[depth(");
          if (ca == std::string::npos && da == std::string::npos) continue;
          const size_t att = ca != std::string::npos ? ca : da;
          const std::string head = trim(decl.substr(0, att));
          const size_t sp = head.find_last_of(" \t");
          const std::string fname = sp == std::string::npos ? head : head.substr(sp + 1);
          if (da != std::string::npos) {
            lay.depth = true;
            lay.depth_field = fname;
            const size_t q = da + 8, qe = decl.find(')', q);
            lay.depth_qual = (qe != std::string::npos) ? trim(decl.substr(q, qe - q)) : "any";
            if (lay.depth_qual.empty()) lay.depth_qual = "any";
            continue;
          }
          const std::string ftype = sp == std::string::npos ? std::string() : trim(head.substr(0, sp));
          const size_t a = decl.find('(', ca), b2 = decl.find(')', ca);
          FragTarget tg;
          tg.field = fname;
          tg.index = (a != std::string::npos && b2 > a)
                         ? std::atoi(trim(decl.substr(a + 1, b2 - a - 1)).c_str())
                         : 0;
          tg.type = ftype;
          tg.width = 1;
          for (const char c : ftype)
            if (c >= '1' && c <= '4') tg.width = c - '0';
          lay.colours.push_back(tg);
        }
        if (lay.colours.empty() && !lay.depth) {
          err = "the fragment returns '" + base + "', which names no [[color(n)]] target";
          return false;
        }
        for (size_t x = 0; x < lay.colours.size(); x++)
          for (size_t y = x + 1; y < lay.colours.size(); y++)
            if (lay.colours[y].index < lay.colours[x].index) {
              const FragTarget tmp = lay.colours[x];
              lay.colours[x] = lay.colours[y];
              lay.colours[y] = tmp;
            }
        return true;
      }
    }
    at = src.find(key, after);
  }
  err = "the fragment returns '" + base + "', which is not a struct in this file";
  return false;
}

std::string pad_struct_src(const FragLayout& lay) {
  std::string s = "struct weft_pad_" + lay.type_name + " {\n";
  for (const FragTarget& tg : lay.colours)
    s += "  float4 " + tg.field + " [[color(" + std::to_string(tg.index) + ")]];\n";
  if (lay.depth)
    s += "  float " + lay.depth_field + " [[depth(" + lay.depth_qual + ")]];\n";
  s += "};\n";
  return s;
}

std::string widen_returns(const std::string& body, const FragLayout& lay,
                          std::string& err) {
  const std::string pad_t = "weft_pad_" + lay.type_name;
  std::string out;
  size_t i = 0;
  while (i < body.size()) {
    const bool rel = body.compare(i, 6, "return") == 0 && (i == 0 || !is_ident(body[i - 1])) &&
                     (i + 6 >= body.size() || !is_ident(body[i + 6]));
    if (!rel) {
      out.push_back(body[i]);
      i++;
      continue;
    }
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
    if (lay.width != 0) {
      if (lay.width == 3)
        out += "{ auto weft_v = (" + expr + "); return float4(weft_v, 1.0); }";
      else if (lay.width == 2)
        out += "{ auto weft_v = (" + expr +
               "); return float4(weft_v.x, weft_v.y, 0.0, 1.0); }";
      else
        out += "{ auto weft_v = (" + expr +
               "); return float4(weft_v, weft_v, weft_v, 1.0); }";
    } else {
      std::string stmt = "{ auto weft_o = (" + expr + "); " + pad_t + " weft_p;";
      for (const FragTarget& tg : lay.colours) {
        const std::string f = "weft_o." + tg.field;
        if (tg.width == 4)
          stmt += " weft_p." + tg.field + " = " + f + ";";
        else if (tg.width == 3)
          stmt += " weft_p." + tg.field + " = float4(" + f + ", 1.0);";
        else if (tg.width == 2)
          stmt += " weft_p." + tg.field + " = float4(" + f + ".x, " + f + ".y, 0.0, 1.0);";
        else
          stmt += " weft_p." + tg.field + " = float4(" + f + ", " + f + ", " + f + ", 1.0);";
      }
      if (lay.depth)
        stmt += " weft_p." + lay.depth_field + " = weft_o." + lay.depth_field + ";";
      out += stmt + " return weft_p; }";
    }
    i = (j < body.size()) ? j + 1 : body.size();
  }
  return out;
}

// One fragment whose colour output is narrower than its attachment.
struct GpuPad {
  std::string name;
  std::string ret_type;
  FragLayout lay;
};

// Rewrites the GPU copy of each such fragment.
bool apply_gpu_pads(std::string& src, const std::vector<GpuPad>& pads, std::string& err) {
  struct Edit {
    size_t at;
    size_t end;
    std::string text;
  };
  std::vector<Edit> edits;
  for (const GpuPad& gp : pads) {
    const std::string sig = gp.ret_type + " " + gp.name + "(";
    const size_t at = src.find(sig);
    if (at == std::string::npos) {
      err = "could not find '" + gp.name + "' to widen its colour output";
      return false;
    }
    const size_t paren = at + sig.size() - 1;
    const size_t close = find_matching(src, paren, '(', ')');
    if (close == std::string::npos) {
      err = "unbalanced parameter list in '" + gp.name + "'";
      return false;
    }
    const size_t brace = src.find('{', close);
    if (brace == std::string::npos) {
      err = "fragment '" + gp.name + "' has no body";
      return false;
    }
    const size_t cbrace = find_matching(src, brace, '{', '}');
    if (cbrace == std::string::npos) {
      err = "unbalanced body in '" + gp.name + "'";
      return false;
    }
    edits.push_back({at, at + gp.ret_type.size(),
                     gp.lay.width != 0 ? std::string("float4") : ("weft_pad_" + gp.lay.type_name)});
    std::string berr;
    const std::string body = src.substr(brace + 1, cbrace - brace - 1);
    const std::string wide = widen_returns(body, gp.lay, berr);
    if (!berr.empty()) {
      err = "fragment '" + gp.name + "': " + berr;
      return false;
    }
    edits.push_back({brace + 1, cbrace, wide});
    if (gp.lay.width == 0) {
      size_t ls = src.rfind('\n', at);
      ls = (ls == std::string::npos) ? 0 : ls + 1;
      edits.push_back({ls, ls, pad_struct_src(gp.lay)});
    }
  }
  for (size_t x = 0; x < edits.size(); x++)
    for (size_t y = x + 1; y < edits.size(); y++)
      if (edits[y].at > edits[x].at) {
        const Edit tmp = edits[x];
        edits[x] = edits[y];
        edits[y] = tmp;
      }
  for (const Edit& e : edits) src.replace(e.at, e.end - e.at, e.text);
  return true;
}

std::string rewrite_frag_return(const std::string& body, const std::string& ret,
                                const FragLayout& lay, std::string& err) {
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
    if (lay.width != 0) {
      std::string fill;
      if (lay.width == 4) {
        fill = "weft_px[0] = (float)weft_c.x; weft_px[1] = (float)weft_c.y;\n"
               "      weft_px[2] = (float)weft_c.z; weft_px[3] = (float)weft_c.w;";
      } else if (lay.width == 3) {
        fill = "weft_px[0] = (float)weft_c.x; weft_px[1] = (float)weft_c.y;\n"
               "      weft_px[2] = (float)weft_c.z; weft_px[3] = 1.0f;";
      } else if (lay.width == 2) {
        fill = "weft_px[0] = (float)weft_c.x; weft_px[1] = (float)weft_c.y;\n"
               "      weft_px[2] = 0.0f; weft_px[3] = 1.0f;";
      } else {
        fill = "weft_px[0] = weft_px[1] = weft_px[2] = (float)weft_c; weft_px[3] = 1.0f;";
      }
      out += "{ float* weft_px = weft_colour + (size_t)ctx.gid * 4u; auto weft_c = (" + expr +
             ");\n      " + fill + " return; }";
    } else {
      std::string stmt = "{\n      auto weft_r = (" + expr + ");";
      std::string weft_depth_val;
      if (lay.depth) {
        weft_depth_val = "(float)weft_r." + lay.depth_field;
        const std::string test = depth_test_src(lay.depth_qual, weft_depth_val);
        if (!test.empty()) stmt += "\n      if (!(" + test + ")) return;";
      }
      for (const FragTarget& tg : lay.colours) {
        const std::string px = "weft_px_" + std::to_string(tg.index);
        const std::string f = "weft_r." + tg.field;
        stmt += "\n      float* " + px + " = weft_colour_" + std::to_string(tg.index) +
                " + (size_t)ctx.gid * 4u;";
        if (tg.width == 4) {
          stmt += "\n      " + px + "[0] = (float)" + f + ".x; " + px + "[1] = (float)" + f +
                  ".y;\n      " + px + "[2] = (float)" + f + ".z; " + px +
                  "[3] = (float)" + f + ".w;";
        } else if (tg.width == 3) {
          stmt += "\n      " + px + "[0] = (float)" + f + ".x; " + px + "[1] = (float)" + f +
                  ".y;\n      " + px + "[2] = (float)" + f + ".z; " + px + "[3] = 1.0f;";
        } else if (tg.width == 2) {
          stmt += "\n      " + px + "[0] = (float)" + f + ".x; " + px + "[1] = (float)" + f +
                  ".y;\n      " + px + "[2] = 0.0f; " + px + "[3] = 1.0f;";
        } else {
          stmt += "\n      " + px + "[0] = " + px + "[1] = " + px + "[2] = (float)" + f +
                  "; " + px + "[3] = 1.0f;";
        }
      }
      if (lay.depth) stmt += "\n      weft_depth[ctx.gid] = " + weft_depth_val + ";";
      stmt += " return; }";
      out += stmt;
    }
    i = (j < body.size()) ? j + 1 : body.size();
  }
  return out;
}

std::string lower_imageblocks(std::string src, uint32_t width) {
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
    // An indexed access keeps its own pixel.
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

static const char* kMslMathText = R"WEFT(
// the order it gives them.
inline float saturate(float x) { return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x); }

// `abs(h)` is ambiguous between the int and float ones.
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

// The matrix helpers: transpose, determinant and inverse for the float matrices.
inline float2x2 transpose(float2x2 m) {
  return float2x2{float2{m.c[0].x, m.c[1].x}, float2{m.c[0].y, m.c[1].y}};
}
inline float3x3 transpose(float3x3 m) {
  return float3x3{float3{m.c[0].x, m.c[1].x, m.c[2].x}, float3{m.c[0].y, m.c[1].y, m.c[2].y},
                  float3{m.c[0].z, m.c[1].z, m.c[2].z}};
}
inline float4x4 transpose(float4x4 m) {
  return float4x4{float4{m.c[0].x, m.c[1].x, m.c[2].x, m.c[3].x},
                  float4{m.c[0].y, m.c[1].y, m.c[2].y, m.c[3].y},
                  float4{m.c[0].z, m.c[1].z, m.c[2].z, m.c[3].z},
                  float4{m.c[0].w, m.c[1].w, m.c[2].w, m.c[3].w}};
}
inline float determinant(float2x2 m) { return m.c[0].x * m.c[1].y - m.c[1].x * m.c[0].y; }
inline float determinant(float3x3 m) { return dot(m.c[0], cross(m.c[1], m.c[2])); }
inline float determinant(float4x4 m) {
  // Expansion along the first column, with the three remaining columns as rows.
  const float3 r0{m.c[1].x, m.c[2].x, m.c[3].x};
  const float3 r1{m.c[1].y, m.c[2].y, m.c[3].y};
  const float3 r2{m.c[1].z, m.c[2].z, m.c[3].z};
  const float3 r3{m.c[1].w, m.c[2].w, m.c[3].w};
  return m.c[0].x * dot(r1, cross(r2, r3)) - m.c[0].y * dot(r0, cross(r2, r3)) +
         m.c[0].z * dot(r0, cross(r1, r3)) - m.c[0].w * dot(r0, cross(r1, r2));
}
inline float2x2 inverse(float2x2 m) {
  return float2x2{float2{m.c[1].y, -m.c[0].y}, float2{-m.c[1].x, m.c[0].x}} *
         (1.0f / determinant(m));
}
inline float3x3 inverse(float3x3 m) {
  // The adjugate: each column is the cross product of the other two columns.
  return float3x3{cross(m.c[1], m.c[2]), cross(m.c[2], m.c[0]), cross(m.c[0], m.c[1])} *
         (1.0f / determinant(m));
}
inline float4x4 inverse(float4x4 m) {
  // remaining element chosen so a nearly singular matrix still divides once.
  const float cols[4][4] = {{m.c[0].x, m.c[0].y, m.c[0].z, m.c[0].w},
                            {m.c[1].x, m.c[1].y, m.c[1].z, m.c[1].w},
                            {m.c[2].x, m.c[2].y, m.c[2].z, m.c[2].w},
                            {m.c[3].x, m.c[3].y, m.c[3].z, m.c[3].w}};
  float a[4][8];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      a[i][j] = cols[j][i];
      a[i][j + 4] = (i == j) ? 1.0f : 0.0f;
    }
  for (int col = 0; col < 4; col++) {
    int piv = col;
    for (int r = col + 1; r < 4; r++)
      if (std::abs(a[r][col]) > std::abs(a[piv][col])) piv = r;
    if (a[piv][col] == 0.0f) return float4x4{0.0f};
    for (int k = 0; k < 8; k++) {
      const float t = a[col][k];
      a[col][k] = a[piv][k];
      a[piv][k] = t;
    }
    const float s = 1.0f / a[col][col];
    for (int k = 0; k < 8; k++) a[col][k] *= s;
    for (int r = 0; r < 4; r++) {
      if (r == col) continue;
      const float f = a[r][col];
      for (int k = 0; k < 8; k++) a[r][k] -= f * a[col][k];
    }
  }
  return float4x4{float4{a[0][4], a[1][4], a[2][4], a[3][4]},
                  float4{a[0][5], a[1][5], a[2][5], a[3][5]},
                  float4{a[0][6], a[1][6], a[2][6], a[3][6]},
                  float4{a[0][7], a[1][7], a[2][7], a[3][7]}};
}
// select(a, b, c) is b where c holds and a elsewhere, which is Metal's order.
inline float select(float a, float b, bool c) { return c ? b : a; }
inline float2 select(float2 a, float2 b, bool2 c) {
  return float2{c.x ? b.x : a.x, c.y ? b.y : a.y};
}
inline float3 select(float3 a, float3 b, bool3 c) {
  return float3{c.x ? b.x : a.x, c.y ? b.y : a.y, c.z ? b.z : a.z};
}
inline float4 select(float4 a, float4 b, bool4 c) {
  return float4{c.x ? b.x : a.x, c.y ? b.y : a.y, c.z ? b.z : a.z, c.w ? b.w : a.w};
}
inline int select(int a, int b, bool c) { return c ? b : a; }
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

const char* kGpuStageShims = R"WEFT(

template <class V, int NV, int NP>
struct weft_gpu_mesh {
  struct vrow {
    device V* b;
    uint o;
    device V& operator[](uint i) { return b[o + i]; }
  };
  struct prow {
    // A primitive is a uint3, and Metal pads it to 16 bytes.
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

std::string lower_gpu_stages(std::string src) {
  bool used = false;
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
  for (size_t at = 0; (at = src.find("[[object]]", at)) != std::string::npos;) {
    const size_t par = src.find('(', at);
    const size_t par_end = par == std::string::npos ? std::string::npos : closing(par);
    if (par_end == std::string::npos) break;
    tg_expr_for(par, par_end);
    src.replace(at, 10, "kernel");
    at += 6;
    used = true;
  }

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
  size_t last_inc = 0;
  for (size_t p = 0; (p = src.find("#include", p)) != std::string::npos;) {
    p = src.find('\n', p);
    if (p == std::string::npos) break;
    last_inc = p + 1;
  }
  src.insert(last_inc, kGpuStageShims);
  return src;
}

std::string emit_ir(const std::vector<TranslatedKernel>& kernels) {
  std::string s = "weft-ir 1\n";
  for (const TranslatedKernel& k : kernels) {
    s += "kernel " + k.name + "\n";
    s += "  bindings " + std::to_string(k.nargs) + "\n";
    s += "  buffers " + std::to_string(k.nbufs) + "\n";
    for (size_t b = 0; b < k.buf_type.size(); b++)
      s += "  buffer " + std::to_string(b) + " " + k.buf_type[b] + "\n";
    for (size_t p = 0; p < k.params.size(); p++) {
      const TranslatedKernel::Param& pr = k.params[p];
      s += "  param " + std::to_string(p) + " " + pr.type;
      if (pr.buf >= 0) s += " buffer " + std::to_string(pr.buf);
      if (pr.scalar >= 0) s += " scalar " + std::to_string(pr.scalar);
      s += "\n";
    }
    size_t ntex = k.tex_name.size();
    if (k.tex_shape.size() > ntex) ntex = k.tex_shape.size();
    if (k.tex_rw.size() > ntex) ntex = k.tex_rw.size();
    for (size_t t = 0; t < ntex; t++) {
      s += "  texture " + std::to_string(t);
      if (t < k.tex_name.size() && !k.tex_name[t].empty()) s += " " + k.tex_name[t];
      if (t < k.tex_shape.size() && !k.tex_shape[t].empty()) s += " " + k.tex_shape[t];
      if (t < k.tex_rw.size() && k.tex_rw[t]) s += " read_write";
      s += "\n";
    }
    if (k.ntex) s += "  textures " + std::to_string(k.ntex) + "\n";
    if (k.nsamp) s += "  samplers " + std::to_string(k.nsamp) + "\n";
    if (k.tg_hint) s += "  threads " + std::to_string(k.tg_hint) + "\n";
    if (k.fixed_threads) s += "  threads_fixed " + std::to_string(k.fixed_threads) + "\n";
    if (k.def_threads) s += "  threads_default " + std::to_string(k.def_threads) + "\n";
    if (k.slot_words) s += "  slot_words " + std::to_string(k.slot_words) + "\n";
    if (k.slot_per) s += "  slot_per " + std::to_string(k.slot_per) + "\n";
    if (k.tile_idx >= 0) s += "  tile_buffer " + std::to_string(k.tile_idx) + "\n";
    if (k.colour_idx >= 0) s += "  colour_buffer " + std::to_string(k.colour_idx) + "\n";
    if (k.payload_idx >= 0) s += "  payload_buffer " + std::to_string(k.payload_idx) + "\n";
    if (k.props_idx >= 0) s += "  props_buffer " + std::to_string(k.props_idx) + "\n";
    if (k.mesh) s += "  mesh " + std::to_string(k.mesh_nv) + " " + std::to_string(k.mesh_np) + "\n";
    static const char* const kFlags[] = {"sync", "flat", "step", "warp", "warp_only",
                                         "gpu_warp", "tile", "task", "wide"};
    const bool vals[] = {k.sync,  k.flat,  k.step,     k.warp, k.warp_only,
                         k.gpu_warp, k.tile, k.task, k.wide};
    for (size_t f = 0; f < sizeof(vals) / sizeof(vals[0]); f++)
      if (vals[f]) s += std::string("  flag ") + kFlags[f] + "\n";
    for (const std::string& u : k.unused) s += "  unused " + u + "\n";
  }
  return s;
}

static std::string ir_quote(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    o += c;
  }
  return o + "\"";
}

std::string emit_ir_json(const std::vector<TranslatedKernel>& kernels) {
  std::string s = "{\"ir\":\"weft-ir\",\"version\":1,\"kernels\":[";
  for (size_t i = 0; i < kernels.size(); i++) {
    const TranslatedKernel& k = kernels[i];
    if (i) s += ",";
    s += "{\"name\":" + ir_quote(k.name);
    s += ",\"bindings\":" + std::to_string(k.nargs);
    s += ",\"buffer_count\":" + std::to_string(k.nbufs);
    s += ",\"buffers\":[";
    for (size_t b = 0; b < k.buf_type.size(); b++)
      s += (b ? "," : "") + ir_quote(k.buf_type[b]);
    s += "],\"params\":[";
    for (size_t p = 0; p < k.params.size(); p++) {
      if (p) s += ",";
      s += "{\"type\":" + ir_quote(k.params[p].type) +
           ",\"buffer\":" + std::to_string(k.params[p].buf) +
           ",\"scalar\":" + std::to_string(k.params[p].scalar) + "}";
    }
    size_t ntex = k.tex_name.size();
    if (k.tex_shape.size() > ntex) ntex = k.tex_shape.size();
    if (k.tex_rw.size() > ntex) ntex = k.tex_rw.size();
    s += "],\"textures\":[";
    for (size_t t = 0; t < ntex; t++) {
      if (t) s += ",";
      s += "{\"name\":" + ir_quote(t < k.tex_name.size() ? k.tex_name[t] : std::string());
      s += ",\"shape\":" + ir_quote(t < k.tex_shape.size() ? k.tex_shape[t] : std::string());
      s += std::string(",\"read_write\":") + (t < k.tex_rw.size() && k.tex_rw[t] ? "true" : "false");
      s += "}";
    }
    s += "],\"texture_count\":" + std::to_string(k.ntex);
    s += ",\"sampler_count\":" + std::to_string(k.nsamp);
    s += ",\"threads\":{\"declared\":" + std::to_string(k.tg_hint) +
         ",\"fixed\":" + std::to_string(k.fixed_threads) +
         ",\"default\":" + std::to_string(k.def_threads) + "}";
    s += ",\"warp\":{\"slot_words\":" + std::to_string(k.slot_words) +
         ",\"slot_per\":" + std::to_string(k.slot_per) + "}";
    s += ",\"binding_indices\":{\"tile\":" + std::to_string(k.tile_idx) +
         ",\"colour\":" + std::to_string(k.colour_idx) +
         ",\"payload\":" + std::to_string(k.payload_idx) +
         ",\"props\":" + std::to_string(k.props_idx) + "}";
    if (k.mesh)
      s += ",\"mesh\":{\"vertices\":" + std::to_string(k.mesh_nv) +
           ",\"primitives\":" + std::to_string(k.mesh_np) + "}";
    else
      s += ",\"mesh\":null";
    static const char* const kFlagNames[] = {"sync",  "flat",      "step", "warp",
                                             "warp_only", "gpu_warp", "tile", "task", "wide"};
    const bool flagVals[] = {k.sync,  k.flat,      k.step,     k.warp, k.warp_only,
                             k.gpu_warp, k.tile,     k.task,     k.wide};
    s += ",\"flags\":[";
    bool first = true;
    for (size_t f = 0; f < sizeof(flagVals) / sizeof(flagVals[0]); f++) {
      if (!flagVals[f]) continue;
      if (!first) s += ",";
      s += ir_quote(kFlagNames[f]);
      first = false;
    }
    s += "],\"unused\":[";
    for (size_t u = 0; u < k.unused.size(); u++)
      s += (u ? "," : "") + ir_quote(k.unused[u]);
    s += "]}";
  }
  return s + "]}";
}

std::string emit_module_abi(const std::vector<TranslatedKernel>& kernels) {
  std::string code;
  for (size_t i = 0; i < kernels.size(); i++) {
    const TranslatedKernel& k = kernels[i];
    if (k.buf_type.empty()) continue;
    code += "static const uint32_t weft_bytes_" + std::to_string(i) + "[] = {";
    for (size_t b = 0; b < k.buf_type.size(); b++) {
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
      "extern \"C\" void weft_tile_reset() { weft::tile_reset(); }\n"
      "extern \"C\" uint64_t weft_tile_violations() { return weft::tile_violations(); }\n";
  return code;
}

// The math calls a shader makes that are hardware approximations on the GPU.
static std::string rewrite_warp_scans(const std::string& text) {
  struct Scan {
    const char* name;
    const char* inc;
    const char* prod;
    const char* width;
  };
  // Longest name first.
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

std::string weft_math_preamble() {
  return std::string(kMslPreamble) +
         "using uint = unsigned int;\n"
         "using ushort = unsigned short;\n"
         "using uchar = unsigned char;\n" +
         kMslMathText + kWSoftMath + weft_f64_cpp_text();
}

// The other place the same choice is made.
std::string fuse_mul_add(const std::string& src) {
  auto skip_ws = [&](size_t& p) {
    while (p < src.size() && (src[p] == ' ' || src[p] == '\t' || src[p] == '\n')) p++;
  };
  auto operand = [&](size_t& p) {
    skip_ws(p);
    const size_t from = p;
    if (p < src.size() && src[p] == '-') {
      p++;
      skip_ws(p);
    }
    bool any = false;
    while (p < src.size()) {
      const char c = src[p];
      if (is_ident(c) || c == '.') {
        p++;
        any = true;
      } else if (c == '(' || c == '[') {
        const char close = c == '(' ? ')' : ']';
        int depth = 0;
        for (; p < src.size(); p++) {
          if (src[p] == c) depth++;
          else if (src[p] == close) {
            if (--depth == 0) {
              p++;
              break;
            }
          }
        }
        if (depth != 0) return std::string();
        any = true;
      } else {
        break;
      }
    }
    if (!any) return std::string();
    const std::string e = trim(src.substr(from, p - from));
    if (e.empty() || e.find("++") != std::string::npos || e.find("--") != std::string::npos ||
        e.find('=') != std::string::npos || e.find(',') != std::string::npos)
      return std::string();
    return e;
  };
  std::string out;
  size_t i = 0;
  while (i < src.size()) {
    const char before = i == 0 ? ' ' : src[i - 1];
    // The last character that is not a space, before the operand at `i`.
    size_t pb = i;
    while (pb > 0 && (src[pb - 1] == ' ' || src[pb - 1] == '\t' || src[pb - 1] == '\n')) pb--;
    const char pre = pb == 0 ? ' ' : src[pb - 1];
    const bool bound = pre != '*' && pre != '/' && pre != '%';
    const bool opens = before == '(' || before == ',' || before == '=' || before == '+' ||
                       before == '-' || before == '*' || before == '/' || before == ';' ||
                       before == '{' || before == '?' || before == ':' || before == '\n' ||
                       before == ' ' || before == '\t';
    std::vector<std::string> f;   // the product's factors, in source order
    std::string add, repl;
    bool neg_prod = false;  // `c - a * b` negates the product, not the addend
    size_t end = i;
    if (opens) {
      size_t q = i;
      std::string g = operand(q);
      size_t r = q;
      skip_ws(r);
      if (!g.empty() && r < src.size() && (src[r] == '+' || src[r] == '-')) {
        // The addend comes first: `c + a * b`, `c - a * b`.
        const bool neg = src[r] == '-';
        r++;
        std::string h = operand(r);
        if (!h.empty() && bound) {
          f.push_back(h);
          for (;;) {
            size_t t = r;
            skip_ws(t);
            if (t >= src.size() || src[t] != '*') break;
            t++;
            std::string k = operand(t);
            if (k.empty()) break;
            f.push_back(k);
            r = t;
          }
          if (f.size() >= 2) {
            add = g;
            neg_prod = neg;
            end = r;
          }
        }
      } else if (!g.empty() && bound) {
        // The product comes first: `a * b - c`, `a * b * d + c`.
        f.push_back(g);
        for (;;) {
          size_t t = q;
          skip_ws(t);
          if (t >= src.size() || src[t] != '*') break;
          t++;
          std::string k = operand(t);
          if (k.empty()) break;
          f.push_back(k);
          q = t;
        }
        size_t t = q;
        skip_ws(t);
        if (f.size() >= 2 && t < src.size() && (src[t] == '+' || src[t] == '-')) {
          const bool neg = src[t] == '-';
          t++;
          add = operand(t);
          if (add.empty()) {
            f.clear();
          } else {
            // The addend can be a product of its own.
            std::string tail;
            for (;;) {
              size_t look = t;
              skip_ws(look);
              if (look + 1 >= src.size() || src[look] != '*') break;
              size_t after = look + 1;
              std::string k = operand(after);
              if (k.empty()) break;
              tail += " * " + k;
              t = after;
            }
            if (!tail.empty()) add = "(" + add + tail + ")";
            if (neg) add = "-(" + add + ")";
            end = t;
          }
        } else {
          f.clear();
        }
      }
    }
    if (f.size() < 2) {
      out += src[i];
      i++;
      continue;
    }
    std::string prod = f[0];
    for (size_t k = 1; k + 1 < f.size(); k++) prod += " * " + f[k];
    if (neg_prod) prod = "-(" + prod + ")";
    repl = "fma(" + prod + ", " + f.back() + ", " + add + ")";
    auto once = [](const std::string& e) {
      return e.empty() || e.find("++") != std::string::npos ||
             e.find("--") != std::string::npos || e.find('=') != std::string::npos ||
             e.find(',') != std::string::npos;
    };
    // A type name only counts as itself.
    auto looks_float = [](const std::string& e) {
      if (e.find("sizeof") != std::string::npos) return false;
      for (const char* ty : {"float", "half"}) {
        const size_t len = std::strlen(ty);
        for (size_t at = e.find(ty); at != std::string::npos; at = e.find(ty, at + 1)) {
          const bool left = at == 0 || !(std::isalnum((unsigned char)e[at - 1]) || e[at - 1] == '_');
          const bool right = at + len >= e.size() ||
                             !(std::isalpha((unsigned char)e[at + len]) || e[at + len] == '_');
          if (left && right) return true;
        }
      }
      for (size_t k = 1; k < e.size(); k++)
        if ((e[k] == 'f' || e[k] == 'h') &&
            (std::isdigit((unsigned char)e[k - 1]) || e[k - 1] == '.'))
          return true;
      return false;
    };
    bool floaty = looks_float(add), clean = !once(add);
    for (const auto& g : f) {
      floaty = floaty || looks_float(g);
      clean = clean && !once(g);
    }
    if (!floaty || !clean) {
      out += src[i];
      i++;
      continue;
    }
    size_t after = end;
    skip_ws(after);
    if (after < src.size() && (src[after] == '*' || src[after] == '/' || src[after] == '(' ||
                              src[after] == '{')) {
      out += src[i];
      i++;
      continue;
    }
    out += " " + repl;
    i = end;
  }
  return out;
}

std::string fuse_accum_products(const std::string& src) {
  std::string out;
  size_t i = 0;
  while (i < src.size()) {
    const size_t at = src.find(" += ", i);
    if (at == std::string::npos) break;
    size_t lb = at;
    while (lb > 0 && (is_ident(src[lb - 1]) || src[lb - 1] == '.' || src[lb - 1] == '[' ||
                      src[lb - 1] == ']'))
      lb--;
    const std::string lhs = src.substr(lb, at - lb);
    size_t j = at + 4;
    int depth = 0;
    size_t star = std::string::npos;
    bool ok = !lhs.empty();
    for (; j < src.size() && ok; j++) {
      const char c = src[j];
      if (c == '(' || c == '[') {
        depth++;
      } else if (c == ')' || c == ']') {
        if (--depth < 0) ok = false;
      } else if (depth == 0 && c == ';') {
        break;
      } else if (depth == 0 && c == '*') {
        star = j;
      } else if (depth == 0 && (c == '=' || c == ',' || c == '+' || c == '-')) {
        ok = false;  // a sum, not one product: the other pass owns that shape
      }
    }
    const std::string x = ok && star != std::string::npos ? trim(src.substr(at + 4, star - at - 4))
                                                          : std::string();
    const std::string y = ok && star != std::string::npos && j < src.size()
                              ? trim(src.substr(star + 1, j - star - 1))
                              : std::string();
    auto once = [](const std::string& e) {
      return e.empty() || e.find("++") != std::string::npos ||
             e.find("--") != std::string::npos;
    };
    auto twice = [&](const std::string& e) {
      return once(e) || e.find('(') != std::string::npos ||
             e.find(',') != std::string::npos;
    };
    auto looks_float = [](const std::string& e) {
      // A type name only counts as itself: a variable called `floaty` is not a float.
      for (const char* ty : {"float", "half"}) {
        const size_t len = std::strlen(ty);
        for (size_t at = e.find(ty); at != std::string::npos; at = e.find(ty, at + 1)) {
          const bool left = at == 0 || !(std::isalnum((unsigned char)e[at - 1]) || e[at - 1] == '_');
          const bool right = at + len >= e.size() ||
                             !(std::isalpha((unsigned char)e[at + len]) || e[at + len] == '_');
          if (left && right) return true;
        }
      }
      for (size_t k = 1; k < e.size(); k++)
        if ((e[k] == 'f' || e[k] == 'h') &&
            (std::isdigit((unsigned char)e[k - 1]) || e[k - 1] == '.'))
          return true;
      return false;
    };
    if (!looks_float(lhs) && !looks_float(x) && !looks_float(y)) {
      out += src.substr(i, at + 4 - i);
      i = at + 4;
      continue;
    }
    if (j >= src.size() || star == std::string::npos || twice(lhs) || once(x) || once(y)) {
      out += src.substr(i, at + 4 - i);
      i = at + 4;
      continue;
    }
    out += src.substr(i, lb - i) + lhs + " = fma(" + x + ", " + y + ", " + lhs + ")";
    i = j;
  }
  out += src.substr(i);
  return out;
}

std::string rewrite_math_calls(const std::string& text) {
  static const char* kNames[][2] = {
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
      {"select", "weft_select"}, {"any", "weft_any"}, {"all", "weft_all"},
      {"pack_float_to_unorm4x8", "weft_pack_unorm4x8"},
      {"pack_float_to_snorm4x8", "weft_pack_snorm4x8"},
      {"pack_float_to_unorm2x16", "weft_pack_unorm2x16"},
      {"pack_float_to_snorm2x16", "weft_pack_snorm2x16"},
      {"unpack_unorm4x8_to_float", "weft_unpack_unorm4x8"},
      {"unpack_snorm4x8_to_float", "weft_unpack_snorm4x8"},
      {"unpack_unorm2x16_to_float", "weft_unpack_unorm2x16"},
      {"unpack_snorm2x16_to_float", "weft_unpack_snorm2x16"},
  };
  std::string out = text;
  for (const auto& nm : kNames) {
    const std::string from = nm[0], to = nm[1];
    size_t p = 0;
    while ((p = out.find(from, p)) != std::string::npos) {
      const bool left_ok = p == 0 || !is_ident(out[p - 1]);
      const size_t e = p + from.size();
      if (!left_ok || e >= out.size() || out[e] != '(') { p = e; continue; }
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

namespace {

bool suspends(const std::string& s) {
  return s.find("co_await") != std::string::npos ||
         s.find("weft::barrier(ctx);") != std::string::npos;
}

// Split on `;` at bracket depth zero: a `for` header's three parts.
std::vector<std::string> split_semis(const std::string& s) {
  std::vector<std::string> out;
  int depth = 0;
  std::string cur;
  for (size_t i = 0; i < s.size(); i++) {
    const char ch = s[i];
    if (ch == '(' || ch == '[' || ch == '{') depth++;
    else if (ch == ')' || ch == ']' || ch == '}') depth--;
    if (ch == ';' && depth == 0) {
      out.push_back(cur);
      cur.clear();
      continue;
    }
    cur.push_back(ch);
  }
  if (!trim(cur).empty()) out.push_back(cur);
  return out;
}

// Split a suspension's arguments on top-level commas.
std::vector<std::string> split_args(const std::string& s) {
  std::vector<std::string> out;
  int depth = 0;
  size_t start = 0;
  for (size_t i = 0; i < s.size(); i++) {
    const char ch = s[i];
    if (ch == '(' || ch == '[' || ch == '{') depth++;
    else if (ch == ')' || ch == ']' || ch == '}') depth--;
    else if (ch == ',' && depth == 0) {
      out.push_back(trim(s.substr(start, i - start)));
      start = i + 1;
    }
  }
  std::string last = trim(s.substr(start));
  if (!last.empty()) out.push_back(last);
  return out;
}

struct Nd {
  enum Kind {
    Seq,
    If,
    For,
    While,
    DoWhile,
    Switch,
    Case,
    Label,
    Block,
    Simple,
    Brk,
    Cont,
    Ret
  } k = Simple;
  std::string text;     // the source this node covers
  std::string a, b, c;  // init / condition / step of the control forms
  std::vector<Nd> kids;
};

bool node_has_jump(const Nd& n) {
  if (n.k == Nd::Brk || n.k == Nd::Cont) return true;
  for (const Nd& k : n.kids)
    if (node_has_jump(k)) return true;
  return false;
}

bool node_suspends(const Nd& n) {
  if (n.k == Nd::Simple || n.k == Nd::Ret) return suspends(n.text);
  if (n.k == Nd::For && (suspends(n.a) || suspends(n.c))) return true;
  for (const Nd& k : n.kids)
    if (node_suspends(k)) return true;
  return false;
}

struct BodyParser {
  const std::string& t;
  size_t pos = 0;
  std::string why;
  bool ok = true;

  explicit BodyParser(const std::string& src) : t(src) {}

  bool bad(const std::string& w) {
    if (ok) why = w;
    ok = false;
    return false;
  }
  void skip_ws() {
    while (pos < t.size() && std::isspace((unsigned char)t[pos])) pos++;
  }
  bool word(const char* w) const {
    const size_t n = std::strlen(w);
    if (t.compare(pos, n, w) != 0) return false;
    return pos + n >= t.size() || !is_ident(t[pos + n]);
  }
  bool keyword() const {
    for (const char* w : {"if", "for", "while", "do", "switch", "return", "break", "continue"})
      if (word(w)) return true;
    return false;
  }

  bool seq(Nd& out, bool top) {
    out.k = Nd::Seq;
    const size_t first = pos;
    while (true) {
      skip_ws();
      if (pos >= t.size() || t[pos] == '}') break;
      Nd one;
      if (!stmt(one)) return false;
      out.kids.push_back(std::move(one));
    }
    out.text = t.substr(first, pos - first);
    if (!top && (pos >= t.size() || t[pos] != '}')) return bad("a block that is not closed");
    return ok;
  }

  bool stmt(Nd& n) {
    skip_ws();
    const size_t first = pos;
    if (pos >= t.size()) return bad("a statement that runs off the end");
    if (word("if")) {
      pos += 2;
      skip_ws();
      if (pos >= t.size() || t[pos] != '(') return bad("`if` without a condition");
      const size_t cl = find_matching(t, pos, '(', ')');
      if (cl == std::string::npos) return bad("unbalanced `if` condition");
      n.k = Nd::If;
      n.b = trim(t.substr(pos + 1, cl - pos - 1));
      pos = cl + 1;
      Nd then_part;
      if (!stmt(then_part)) return false;
      n.kids.push_back(std::move(then_part));
      skip_ws();
      if (word("else")) {
        pos += 4;
        Nd else_part;
        if (!stmt(else_part)) return false;
        n.kids.push_back(std::move(else_part));
      }
      n.text = t.substr(first, pos - first);
      return true;
    }
    if (word("for")) {
      pos += 3;
      skip_ws();
      if (pos >= t.size() || t[pos] != '(') return bad("`for` without a header");
      const size_t cl = find_matching(t, pos, '(', ')');
      if (cl == std::string::npos) return bad("unbalanced `for` header");
      const auto parts = split_semis(t.substr(pos + 1, cl - pos - 1));
      if (parts.size() > 3) return bad("a `for` header with more than three parts");
      n.k = Nd::For;
      n.a = parts.size() > 0 ? trim(parts[0]) : std::string();
      n.b = parts.size() > 1 ? trim(parts[1]) : std::string();
      n.c = parts.size() > 2 ? trim(parts[2]) : std::string();
      pos = cl + 1;
      Nd body_part;
      if (!stmt(body_part)) return false;
      n.kids.push_back(std::move(body_part));
      n.text = t.substr(first, pos - first);
      return true;
    }
    if (word("while") || word("do")) {
      const bool is_do = word("do");
      pos += is_do ? 2 : 5;
      n.k = is_do ? Nd::DoWhile : Nd::While;
      if (is_do) {
        Nd body_part;
        if (!stmt(body_part)) return false;
        n.kids.push_back(std::move(body_part));
        skip_ws();
        if (!word("while")) return bad("`do` without a `while`");
        pos += 5;
        skip_ws();
      }
      if (pos >= t.size() || t[pos] != '(') return bad("`while` without a condition");
      const size_t cl = find_matching(t, pos, '(', ')');
      if (cl == std::string::npos) return bad("unbalanced `while` condition");
      n.b = trim(t.substr(pos + 1, cl - pos - 1));
      pos = cl + 1;
      if (is_do) {
        skip_ws();
        if (pos >= t.size() || t[pos] != ';') return bad("`do` without a `;`");
        pos++;
      } else {
        Nd body_part;
        if (!stmt(body_part)) return false;
        n.kids.insert(n.kids.begin(), std::move(body_part));
      }
      n.text = t.substr(first, pos - first);
      return true;
    }
    if (word("switch")) {
      pos += 6;
      skip_ws();
      if (pos >= t.size() || t[pos] != '(') return bad("`switch` without a subject");
      const size_t cl = find_matching(t, pos, '(', ')');
      if (cl == std::string::npos) return bad("unbalanced `switch` subject");
      n.k = Nd::Switch;
      n.b = trim(t.substr(pos + 1, cl - pos - 1));
      pos = cl + 1;
      skip_ws();
      if (pos >= t.size() || t[pos] != '{') return bad("`switch` without a body");
      pos++;
      for (;;) {
        skip_ws();
        if (pos >= t.size()) return bad("a `switch` that is not closed");
        if (t[pos] == '}') {
          pos++;
          break;
        }
        if (word("case")) {
          pos += 4;
          skip_ws();
          int depth = 0;
          size_t i = pos;
          for (; i < t.size(); i++) {
            const char ch = t[i];
            if (ch == '(' || ch == '[' || ch == '{') depth++;
            else if (ch == ')' || ch == ']' || ch == '}') {
              if (depth == 0) break;
              depth--;
            } else if ((ch == ':' || ch == ';') && depth == 0) {
              break;
            }
          }
          if (i >= t.size() || t[i] != ':') return bad("a `case` without a `:`");
          Nd lab;
          lab.k = Nd::Case;
          lab.a = trim(t.substr(pos, i - pos));
          if (lab.a.empty()) return bad("a `case` with no value");
          pos = i + 1;
          n.kids.push_back(std::move(lab));
          continue;
        }
        if (word("default")) {
          pos += 7;
          skip_ws();
          if (pos >= t.size() || t[pos] != ':') return bad("`default` without a `:`");
          pos++;
          Nd lab;
          lab.k = Nd::Case;
          n.kids.push_back(std::move(lab));
          continue;
        }
        Nd one;
        if (!stmt(one)) return false;
        n.kids.push_back(std::move(one));
      }
      n.text = t.substr(first, pos - first);
      return true;
    }
    if (word("return")) {
      const size_t end = t.find(';', pos);
      if (end == std::string::npos) return bad("a `return` without a `;`");
      if (!trim(t.substr(pos + 6, end - pos - 6)).empty()) return bad("a `return` with a value");
      n.k = Nd::Ret;
      pos = end + 1;
      n.text = t.substr(first, pos - first);
      return true;
    }
    if (word("break") || word("continue")) {
      const bool brk = word("break");
      const size_t end = t.find(';', pos);
      if (end == std::string::npos) return bad("a jump without a `;`");
      n.k = brk ? Nd::Brk : Nd::Cont;
      pos = end + 1;
      n.text = t.substr(first, pos - first);
      return true;
    }
    // A label.
    {
      size_t j = pos;
      while (j < t.size() && is_ident(t[j])) j++;
      size_t k = j;
      while (k < t.size() && std::isspace((unsigned char)t[k])) k++;
      if (j > pos && k < t.size() && t[k] == ':' && (k + 1 >= t.size() || t[k + 1] != ':')) {
        n.k = Nd::Label;
        n.a = t.substr(pos, j - pos);
        pos = k + 1;
        n.text = t.substr(first, pos - first);
        return true;
      }
    }
    if (t[pos] == '{') {
      pos++;
      n.k = Nd::Block;
      Nd body_part;
      if (!seq(body_part, false)) return false;
      pos++;  // the `}` seq stopped on
      n.kids.push_back(std::move(body_part));
      n.text = t.substr(first, pos - first);
      return true;
    }
    int depth = 0;
    size_t i = pos;
    for (; i < t.size(); i++) {
      const char ch = t[i];
      if (ch == '(' || ch == '[' || ch == '{') depth++;
      else if (ch == ')' || ch == ']' || ch == '}') {
        if (depth == 0) break;
        depth--;
      } else if (ch == ';' && depth == 0) break;
    }
    if (i >= t.size() || t[i] != ';') return bad("a statement with no `;`");
    n.k = Nd::Simple;
    n.text = t.substr(first, i + 1 - first);
    pos = i + 1;
    return true;
  }
};

struct StepEmitter {
  std::string out, cases, why;
  bool ok = true;
  int site = 1, label = 0;
  std::vector<std::pair<std::string, std::string>> loops;  // break, continue

  bool bad(const std::string& w) {
    if (ok) why = w;
    ok = false;
    return false;
  }
  std::string fresh() { return "weft_l" + std::to_string(label++); }

  void park() {
    const std::string n = std::to_string(site);
    out += "  weft_s->pc = " + n + "u; return;\n";
    cases += "      case " + n + "u: goto R" + n + ";\n";
    out += "  R" + n + ":;\n";
    site++;
  }

  bool halves(const std::string& fn, const std::vector<std::string>& a, std::string& pub,
              std::string& take) {
    const std::string ty = "std::decay_t<decltype((" + trim(a[1]) + "))>";
    const std::string v = trim(a[1]);
    if (fn == "grp_shfl") {
      if (a.size() != 5) return bad("a shuffle with an unexpected argument count");
      pub = "weft::shfl_publish(ctx, (" + v + "));";
      take = "weft::shfl_take<" + ty + ">(ctx, " + trim(a[2]) + ", " + trim(a[3]) + ", " +
             trim(a[4]) + ")";
      return true;
    }
    if (fn == "shfl_fill") {
      if (a.size() != 6) return bad("a filling shuffle with an unexpected argument count");
      pub = "weft::shfl_publish(ctx, (" + v + "));";
      take = "weft::shfl_take_fill<" + ty + ">(ctx, " + trim(a[2]) + ", " + trim(a[3]) + ", (" +
             trim(a[4]) + "), " + trim(a[5]) + ")";
      return true;
    }
    if (fn == "prefix") {
      if (a.size() != 6) return bad("a scan with an unexpected argument count");
      pub = "weft::shfl_publish(ctx, (" + v + "));";
      take = "weft::prefix_take<" + ty + ">(ctx, " + trim(a[2]) + ", " + trim(a[3]) + ", " +
             trim(a[4]) + ", " + trim(a[5]) + ")";
      return true;
    }
    if (fn == "grp_reduce") {
      if (a.size() != 5) return bad("a reduction with an unexpected argument count");
      pub = "weft::shfl_publish(ctx, (" + v + "));";
      take = "weft::reduce_take<" + ty + ">(ctx, " + trim(a[2]) + ", " + trim(a[3]) + ", " +
             trim(a[4]) + ")";
      return true;
    }
    if (fn == "grp_vote") {
      if (a.size() != 4)
        return bad("a vote with " + std::to_string(a.size()) + " arguments");
      pub = "weft::shfl_publish(ctx, (" + v + ") ? 1u : 0u);";
      take = "weft::vote_take(ctx, " + trim(a[2]) + ", " + trim(a[3]) + ")";
      return true;
    }
    return bad("a suspension '" + fn + "' this does not split");
  }

  bool simple(const std::string& text) {
    const std::string one = trim(text);
    if (one == "weft::barrier(ctx);") {
      park();
      return true;
    }
    if (one.find("weft::barrier(ctx);") != std::string::npos)
      return bad("a barrier inside an expression");
    std::string pub, code;
    bool any = false;
    size_t i = 0;
    while (i < text.size()) {
      const size_t p = text.find("(co_await weft::", i);
      if (p == std::string::npos) {
        code += text.substr(i);
        break;
      }
      code += text.substr(i, p - i);
      size_t j = p + 1 + 9;  // past `(co_await `
      if (text.compare(p + 1, 9, "co_await ") != 0 || text.compare(j, 6, "weft::") != 0)
        return bad("a suspension that is not the runtime's");
      j += 6;
      const size_t fn0 = j;
      while (j < text.size() && is_ident(text[j])) j++;
      const std::string fn = text.substr(fn0, j - fn0);
      while (j < text.size() && std::isspace((unsigned char)text[j])) j++;
      if (j >= text.size() || text[j] != '(') return bad("a suspension without a call");
      const size_t close = find_matching(text, j, '(', ')');
      if (close == std::string::npos || close + 1 >= text.size() || text[close + 1] != ')')
        return bad("a suspension that is not one expression");
      const std::string inner = text.substr(j + 1, close - j - 1);
      const auto a = split_args(inner);
      if (a.size() < 2 || trim(a[0]) != "ctx") return bad("a suspension without the lane's context");
      if (suspends(inner)) return bad("a suspension whose operand is another suspension");
      std::string one_pub, one_take;
      if (!halves(fn, a, one_pub, one_take)) return false;
      pub += one_pub;
      code += one_take;
      any = true;
      i = close + 2;
    }
    if (suspends(code)) return bad("a suspension this does not split");
    if (any) {
      out += "  " + pub + "\n";
      park();
    }
    out += "  " + trim(code) + "\n";
    return true;
  }

  void emit(const Nd& n) {
    if (!ok) return;
    switch (n.k) {
      case Nd::Seq:
        for (const Nd& k : n.kids) emit(k);
        return;
      case Nd::Ret:
        out += "  weft_s->pc = weft::kThreadDone;\n  return;\n";
        return;
      case Nd::Switch: {
        const std::string end = fresh();
        std::vector<std::string> arms;
        std::string def;
        for (const Nd& k : n.kids) {
          if (k.k != Nd::Case) continue;
          arms.push_back(fresh());
          if (k.a.empty()) def = arms.back();
        }
        out += "  {\n  auto weft_sw = (" + n.b + ");\n  (void)weft_sw;\n";
        size_t arm = 0;
        for (const Nd& k : n.kids) {
          if (k.k != Nd::Case) continue;
          if (!k.a.empty()) out += "  if (weft_sw == (" + k.a + ")) goto " + arms[arm] + ";\n";
          arm++;
        }
        out += "  goto " + (def.empty() ? end : def) + ";\n";
        loops.push_back({end, std::string()});
        arm = 0;
        for (const Nd& k : n.kids) {
          if (k.k == Nd::Case) {
            out += "  " + arms[arm++] + ":;\n";
            continue;
          }
          emit(k);
        }
        loops.pop_back();
        out += "  " + end + ":;\n  }\n";
        return;
      }
      case Nd::Case:
        return;
      case Nd::Label:
        out += "  " + n.a + ":;\n";
        return;
      case Nd::Brk:
      case Nd::Cont:
        if (loops.empty()) {
          bad("a `break` or `continue` outside a loop this lowered");
          return;
        }
        if (n.k == Nd::Brk) {
          out += "  goto " + loops.back().first + ";\n";
          return;
        }
        for (size_t f = loops.size(); f-- > 0;) {
          if (loops[f].second.empty()) continue;
          out += "  goto " + loops[f].second + ";\n";
          return;
        }
        bad("a `continue` with no loop around it this lowered");
        return;
      case Nd::Simple:
        simple(n.text);
        return;
      case Nd::Block:
        if (!node_suspends(n) && (loops.empty() || !node_has_jump(n))) {
          out += "  " + trim(n.text) + "\n";
          return;
        }
        out += "  {\n";
        for (const Nd& k : n.kids) emit(k);
        out += "  }\n";
        return;
      case Nd::If: {
        if (!node_suspends(n) && (loops.empty() || !node_has_jump(n))) {
          out += "  " + trim(n.text) + "\n";
          return;
        }
        const std::string l_else = fresh(), l_end = fresh();
        out += "  if (!(" + n.b + ")) goto " + l_else + ";\n";
        emit(n.kids[0]);
        out += "  goto " + l_end + ";\n";
        out += "  " + l_else + ":;\n";
        if (n.kids.size() > 1) emit(n.kids[1]);
        out += "  " + l_end + ":;\n";
        return;
      }
      case Nd::For: {
        if (!node_suspends(n)) {
          out += "  " + trim(n.text) + "\n";
          return;
        }
        const std::string l_cond = fresh(), l_inc = fresh(), l_end = fresh();
        if (!n.a.empty()) out += "  " + n.a + ";\n";
        out += "  " + l_cond + ":;\n";
        if (!n.b.empty()) out += "  if (!(" + n.b + ")) goto " + l_end + ";\n";
        loops.push_back({l_end, l_inc});
        emit(n.kids[0]);
        loops.pop_back();
        out += "  " + l_inc + ":;\n";
        if (!n.c.empty()) out += "  " + n.c + ";\n";
        out += "  goto " + l_cond + ";\n  " + l_end + ":;\n";
        return;
      }
      case Nd::While: {
        if (!node_suspends(n)) {
          out += "  " + trim(n.text) + "\n";
          return;
        }
        const std::string l_cond = fresh(), l_end = fresh();
        out += "  " + l_cond + ":;\n  if (!(" + n.b + ")) goto " + l_end + ";\n";
        loops.push_back({l_end, l_cond});
        emit(n.kids[0]);
        loops.pop_back();
        out += "  goto " + l_cond + ";\n  " + l_end + ":;\n";
        return;
      }
      case Nd::DoWhile: {
        if (!node_suspends(n)) {
          out += "  " + trim(n.text) + "\n";
          return;
        }
        const std::string l_body = fresh(), l_cont = fresh(), l_end = fresh();
        out += "  " + l_body + ":;\n";
        loops.push_back({l_end, l_cont});
        emit(n.kids[0]);
        loops.pop_back();
        out += "  " + l_cont + ":;\n";
        out += "  if (" + n.b + ") goto " + l_body + ";\n  " + l_end + ":;\n";
        return;
      }
    }
  }
};

// Point a local's uses at the blob.
std::string rename_blob_local(const std::string& s, const std::string& name) {
  const std::string to = "weft_s->" + name;
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    const bool hit = s.compare(i, name.size(), name) == 0 &&
                     (i == 0 || (!is_ident(s[i - 1]) && s[i - 1] != '.' && s[i - 1] != '>')) &&
                     (i + name.size() >= s.size() || !is_ident(s[i + name.size()]));
    if (hit) {
      out += to;
      i += name.size();
      continue;
    }
    out.push_back(s[i++]);
  }
  return out;
}

bool freshen_lambda_params(std::string& body, const std::vector<std::string>& names,
                           std::string& why) {
  auto hoisted = [&](const std::string& n) {
    for (const std::string& h : names)
      if (h == n) return true;
    return false;
  };
  size_t pos = 0;
  for (;;) {
    const size_t br = body.find('[', pos);
    if (br == std::string::npos) return true;
    pos = br + 1;
    // `a[i]`, `p->at[i]` and `[[attribute]]` are not capture lists.
    if (br > 0 && (is_ident(body[br - 1]) || body[br - 1] == ']' || body[br - 1] == ')')) continue;
    const size_t cap = find_matching(body, br, '[', ']');
    if (cap == std::string::npos) continue;
    const std::string caps = body.substr(br + 1, cap - br - 1);
    bool lambda = true;
    for (char c : caps) {
      if (!(std::isalnum((unsigned char)c) || std::isspace((unsigned char)c) || c == '&' ||
            c == '=' || c == '_' || c == ','))
        lambda = false;
    }
    if (!lambda) continue;
    size_t k = cap + 1;
    while (k < body.size() && std::isspace((unsigned char)body[k])) k++;
    if (k >= body.size() || body[k] != '(') continue;
    const size_t par = find_matching(body, k, '(', ')');
    if (par == std::string::npos) continue;
    // The body, past an optional `mutable`, `noexcept` and a trailing return type.
    size_t brace = std::string::npos;
    for (size_t b = par + 1; b < body.size(); b++) {
      if (body[b] == ';') break;
      if (body[b] == '{') {
        brace = b;
        break;
      }
    }
    if (brace == std::string::npos) continue;
    const size_t close = find_matching(body, brace, '{', '}');
    if (close == std::string::npos) {
      why = "a lambda whose body this cannot read";
      return false;
    }
    std::vector<std::string> params;
    const std::string list = body.substr(k + 1, par - k - 1);
    size_t st = 0;
    int depth = 0;
    for (size_t i = 0; i <= list.size(); i++) {
      const char c = i < list.size() ? list[i] : ',';
      if (c == '(' || c == '<' || c == '[') depth++;
      else if (c == ')' || c == '>' || c == ']') depth--;
      else if (c == ',' && depth == 0) {
        const std::string one = trim(list.substr(st, i - st));
        st = i + 1;
        size_t e = one.size();
        while (e > 0 && (std::isspace((unsigned char)one[e - 1]) || one[e - 1] == '&' ||
                         one[e - 1] == '*'))
          e--;
        size_t s = e;
        while (s > 0 && is_ident(one[s - 1])) s--;
        if (e > s) params.push_back(one.substr(s, e - s));
      }
    }
    std::string seg = body.substr(br, close + 1 - br);
    for (const std::string& p : params)
      if (hoisted(p)) seg = rename_word(seg, p, "weft_lp_" + p);
    body = body.substr(0, br) + seg + body.substr(close + 1);
    pos = br + seg.size();
  }
}

bool rebuild_captures(std::string& body, const std::vector<std::string>& names,
                      std::string& why, const std::string& src = "weft_s->") {
  auto swap = [](const std::string& s, const std::string& from, const std::string& to) {
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
      if (s.compare(i, from.size(), from) == 0 &&
          (i + from.size() >= s.size() || !is_ident(s[i + from.size()]))) {
        out += to;
        i += from.size();
        continue;
      }
      out.push_back(s[i++]);
    }
    return out;
  };
  auto mentions = [](const std::string& s, const std::string& n) {
    const std::string pat = "weft_s->" + n;
    for (size_t i = s.find(pat); i != std::string::npos; i = s.find(pat, i + 1))
      if (i + pat.size() >= s.size() || !is_ident(s[i + pat.size()])) return true;
    return false;
  };
  auto capture_list = [](const std::string& caps) {
    bool plain = true;
    for (char c : caps)
      if (!(std::isalnum((unsigned char)c) || std::isspace((unsigned char)c) || c == '&' ||
            c == '=' || c == '_' || c == ',' || c == '-' || c == '>'))
        plain = false;
    return plain;
  };
  size_t pos = 0;
  for (;;) {
    const size_t br = body.find('[', pos);
    if (br == std::string::npos) return true;
    pos = br + 1;
    if (br > 0 && (is_ident(body[br - 1]) || body[br - 1] == ']' || body[br - 1] == ')')) continue;
    const size_t cap = find_matching(body, br, '[', ']');
    if (cap == std::string::npos) continue;
    const std::string caps = body.substr(br + 1, cap - br - 1);
    if (!capture_list(caps)) continue;
    bool has_ident = false;
    for (char c : caps)
      if (std::isalpha((unsigned char)c) || c == '_') has_ident = true;
    if (caps.find('=') == std::string::npos && !has_ident) continue;
    // Where the lambda's body starts: a parameter list is optional.
    size_t k = cap + 1;
    while (k < body.size() && std::isspace((unsigned char)body[k])) k++;
    size_t scan = k;
    if (k < body.size() && body[k] == '(') {
      const size_t par = find_matching(body, k, '(', ')');
      if (par == std::string::npos) continue;
      scan = par + 1;
    }
    size_t brace = std::string::npos;
    for (size_t b = scan; b < body.size(); b++) {
      if (body[b] == ';') break;
      if (body[b] == '{') {
        brace = b;
        break;
      }
    }
    if (brace == std::string::npos) continue;
    const size_t close = find_matching(body, brace, '{', '}');
    if (close == std::string::npos) {
      why = "a lambda whose body this cannot read";
      return false;
    }
    const std::string inner = body.substr(brace + 1, close - brace - 1);
    for (size_t b = inner.find('['); b != std::string::npos; b = inner.find('[', b + 1)) {
      if (b > 0 && (is_ident(inner[b - 1]) || inner[b - 1] == ']' || inner[b - 1] == ')')) continue;
      const size_t c2 = find_matching(inner, b, '[', ']');
      if (c2 == std::string::npos) continue;
      if (!capture_list(inner.substr(b + 1, c2 - b - 1))) continue;
      size_t z = c2 + 1;
      while (z < inner.size() && std::isspace((unsigned char)inner[z])) z++;
      if (z < inner.size() && (inner[z] == '(' || inner[z] == '{')) {
        why = "a lambda inside a lambda that captures a local";
        return false;
      }
    }
    // What the list asks for: a default, and any name given one explicitly.
    std::vector<std::string> keep, by_ref, by_copy;
    bool def_ref = false;
    size_t st = 0;
    int depth = 0;
    for (size_t i = 0; i <= caps.size(); i++) {
      const char c = i < caps.size() ? caps[i] : ',';
      if (c == '(' || c == '[' || c == '{') depth++;
      else if (c == ')' || c == ']' || c == '}') depth--;
      else if (c == ',' && depth == 0) {
        const std::string one = trim(caps.substr(st, i - st));
        st = i + 1;
        if (one.empty()) continue;
        if (one == "=") continue;
        if (one == "&") {
          def_ref = true;
          continue;
        }
        if (one == "this") {
          keep.push_back(one);
          continue;
        }
        const bool ref = one[0] == '&';
        std::string nm = ref ? trim(one.substr(1)) : one;
        const std::string blob = "weft_s->";
        if (nm.compare(0, blob.size(), blob) == 0) nm = trim(nm.substr(blob.size()));
        bool ident = !nm.empty();
        for (char c2 : nm)
          if (!is_ident(c2)) ident = false;
        if (ident) (ref ? by_ref : by_copy).push_back(nm);
        else keep.push_back(one);  // an init-capture the body wrote: left as it is
      }
    }
    std::string list;
    std::string replaced = inner;
    for (const std::string& p : keep) list += (list.empty() ? "" : ", ") + p;
    for (const std::string& n : names) {
      if (!mentions(inner, n)) continue;
      bool ref = def_ref;
      for (const std::string& f : by_ref)
        if (f == n) ref = true;
      for (const std::string& f : by_copy)
        if (f == n) ref = false;
      list += (list.empty() ? "" : ", ") + std::string(ref ? "&" : "") + "weft_c_" + n +
              " = " + src + n;
      replaced = swap(replaced, "weft_s->" + n, "weft_c_" + n);
    }
    const std::string head = "[" + list + "]" + body.substr(cap + 1, brace - cap - 1);
    body = body.substr(0, br) + head + "{" + replaced + "}" + body.substr(close + 1);
    pos = br + head.size() + 1 + replaced.size() + 1;
  }
}

struct Start {
  size_t at, end;
};

// Every statement start in a body, and where each brace pair closes.
void scan_starts(const std::string& body, std::vector<Start>& starts,
                 std::map<size_t, size_t>& closes) {
  starts.clear();
  closes.clear();
  std::vector<size_t> stack;
  for (size_t i = 0; i < body.size(); i++) {
    if (body[i] == '{') {
      stack.push_back(i);
    } else if (body[i] == '}' && !stack.empty()) {
      closes[stack.back()] = i;
      stack.pop_back();
    }
  }
  stack.clear();
  auto scope_end = [&]() -> size_t {
    return stack.empty() ? body.size() : closes[stack.back()];
  };
  auto note = [&](size_t at, size_t end) { starts.push_back(Start{at, end}); };
  note(0, body.size());
  for (size_t i = 0; i < body.size(); i++) {
    const char ch = body[i];
    if (ch == '{') {
      stack.push_back(i);
      note(i + 1, closes.count(i) ? closes[i] : body.size());
    } else if (ch == '}') {
      if (!stack.empty()) stack.pop_back();
      note(i + 1, scope_end());
    } else if (ch == ';') {
      note(i + 1, scope_end());
    } else if (ch == ':') {
      note(i + 1, scope_end());
    }
    if (ch == 'f' && i + 3 <= body.size() && body.compare(i, 3, "for") == 0 &&
        (i == 0 || !is_ident(body[i - 1]))) {
      size_t j = i + 3;
      while (j < body.size() && std::isspace((unsigned char)body[j])) j++;
      if (j < body.size() && body[j] == '(') {
        const size_t cl = find_matching(body, j, '(', ')');
        size_t e = cl == std::string::npos ? body.size() : cl + 1;
        while (e < body.size() && std::isspace((unsigned char)body[e])) e++;
        if (e < body.size() && body[e] == '{') {
          const size_t cb = find_matching(body, e, '{', '}');
          e = cb == std::string::npos ? body.size() : cb + 1;
        } else {
          int depth = 0;
          size_t k = e;
          for (; k < body.size(); k++) {
            const char c2 = body[k];
            if (c2 == '(' || c2 == '[' || c2 == '{') depth++;
            else if (c2 == ')' || c2 == ']' || c2 == '}') {
              if (depth == 0) break;
              depth--;
            } else if (c2 == ';' && depth == 0) break;
          }
          e = k < body.size() ? k + 1 : body.size();
        }
        note(j + 1, e);
      }
    }
  }
  std::sort(starts.begin(), starts.end(),
            [](const Start& x, const Start& y) { return x.at < y.at; });
  starts.erase(std::unique(starts.begin(), starts.end(),
                           [](const Start& x, const Start& y) { return x.at == y.at; }),
               starts.end());
}

bool place_braces(const std::string& text, const std::vector<long>& ext, size_t level,
                  const std::string& prefix,
                  std::vector<std::pair<std::string, std::string>>& cells) {
  std::vector<std::string> parts;
  size_t st = 0;
  int dep = 0;
  for (size_t q = 0; q <= text.size(); q++) {
    const char c = q < text.size() ? text[q] : ',';
    if (c == '(' || c == '[' || c == '{') dep++;
    else if (c == ')' || c == ']' || c == '}') dep--;
    else if (c == ',' && dep == 0) {
      const std::string one = trim(text.substr(st, q - st));
      st = q + 1;
      if (!one.empty()) parts.push_back(one);
    }
  }
  if (level >= ext.size()) return parts.empty();
  bool braced = false, plain = false;
  for (const std::string& p : parts) {
    if (!p.empty() && p.front() == '{') braced = true;
    else plain = true;
  }
  if (braced && plain) return false;
  for (size_t q = 0; q < parts.size(); q++) {
    if ((long)q >= ext[level]) return false;
    const std::string at = prefix + "[" + std::to_string(q) + "]";
    if (!braced) {
      cells.push_back({at, parts[q]});
      continue;
    }
    const std::string& p = parts[q];
    if (p.size() < 2 || p.front() != '{' || p.back() != '}') return false;
    if (!place_braces(p.substr(1, p.size() - 2), ext, level + 1, at, cells)) return false;
  }
  return true;
}

void unshadow(std::string& body) {
  for (int round = 0; round < 256; round++) {
    std::vector<Start> starts;
    std::map<size_t, size_t> closes;
    scan_starts(body, starts, closes);
    std::vector<std::pair<std::string, size_t>> live;  // name, where its scope ends
    bool renamed = false;
    for (const Start& st0 : starts) {
      size_t i = st0.at;
      auto ws = [&](size_t& q) {
        while (q < body.size() && std::isspace((unsigned char)body[q])) q++;
      };
      ws(i);
      if (i >= body.size() || !is_ident(body[i])) continue;
      // A keyword that begins a statement is not a type: `return x;`, `case 0:`.
      {
        size_t q = i;
        while (q < body.size() && is_ident(body[q])) q++;
        const std::string first = body.substr(i, q - i);
        bool jump = false;
        for (const char* w : {"case", "default", "goto", "return", "break", "continue",
                              "throw", "delete", "else", "do", "if", "while", "for",
                              "switch", "co_return", "co_await"})
          if (first == w) jump = true;
        if (jump) continue;
      }
      if (body.compare(i, 5, "const") == 0 && i + 5 < body.size() &&
          std::isspace((unsigned char)body[i + 5])) {
        i += 5;
        ws(i);
        if (i >= body.size() || !is_ident(body[i])) continue;
      }
      while (i < body.size() && is_ident(body[i])) i++;  // the type
      ws(i);
      while (i < body.size() && (body[i] == '*' || body[i] == '&')) {
        i++;
        ws(i);
      }
      if (i >= body.size() || !is_ident(body[i])) continue;
      const size_t n0 = i;
      while (i < body.size() && is_ident(body[i])) i++;
      const std::string dname = body.substr(n0, i - n0);
      ws(i);
      if (i >= body.size() ||
          (body[i] != '=' && body[i] != ';' && body[i] != ',' && body[i] != '['))
        continue;
      // Whatever scope closed before this statement is not live any more.
      size_t keep = 0;
      for (size_t q = 0; q < live.size(); q++)
        if (live[q].second > st0.at) live[keep++] = live[q];
      live.resize(keep);
      bool shadow = false;
      for (const auto& lv : live)
        if (lv.first == dname) shadow = true;
      if (!shadow) {
        live.push_back({dname, st0.end});
        continue;
      }
      std::string fresh;
      for (int n2 = 1; n2 < 1024; n2++) {
        fresh = dname + "_" + std::to_string(n2);
        if (!contains_word(body, fresh)) break;
      }
      body = body.substr(0, st0.at) +
             rename_word(body.substr(st0.at, st0.end - st0.at), dname, fresh) +
             body.substr(st0.end);
      renamed = true;
      break;
    }
    if (!renamed) return;
  }
}

bool hoist_locals(std::string& body, const std::string& kname, const std::string& reserved,
                  std::string& fields, std::string& types, std::string& why) {
  struct Edit {
    size_t from, to;
    std::string text;
  };
  std::vector<Edit> edits;
  std::vector<std::string> names;
  // Structs a `decltype` field is read against, written ahead of the blob.
  std::vector<std::string> type_helpers;
  std::vector<std::string> kparams;
  std::map<std::string, std::string> kparam_types;
  for (size_t eq = reserved.find('='); eq != std::string::npos; eq = reserved.find('=', eq + 1)) {
    if (eq + 1 < reserved.size() && reserved[eq + 1] == '=') continue;
    size_t q = eq;
    while (q > 0 && std::isspace((unsigned char)reserved[q - 1])) q--;
    const size_t qe = q;
    while (q > 0 && is_ident(reserved[q - 1])) q--;
    if (qe <= q) continue;
    const std::string pname = reserved.substr(q, qe - q);
    kparams.push_back(pname);
    size_t line0 = reserved.rfind(';', q);
    line0 = line0 == std::string::npos ? 0 : line0 + 1;
    size_t line_end = reserved.find(';', eq);
    const std::string decl = trim(reserved.substr(
        line0, (line_end == std::string::npos ? reserved.size() : line_end) - line0));
    // The occurrence that is the declaration: a whole word the `=` follows.
    size_t cut = std::string::npos;
    for (size_t w = decl.find(pname); w != std::string::npos; w = decl.find(pname, w + 1)) {
      if (w > 0 && is_ident(decl[w - 1])) continue;
      size_t z = w + pname.size();
      while (z < decl.size() && std::isspace((unsigned char)decl[z])) z++;
      if (z < decl.size() && decl[z] == '=' && (z + 1 >= decl.size() || decl[z + 1] != '=')) {
        cut = w;
        break;
      }
    }
    if (cut == std::string::npos) continue;
    const std::string ty = trim(decl.substr(0, cut));
    bool ok = !ty.empty();
    for (char c : ty)
      if (!(std::isalnum((unsigned char)c) || c == '_' || c == '*' || c == '&' || c == ':' ||
            c == ' ' || c == ',' || c == '<' || c == '>'))
        ok = false;
    if (ok && (is_ident(ty.back()) || ty.back() == '*' || ty.back() == '&'))
      kparam_types[pname] = ty;
  }
  unshadow(body);
  std::vector<Start> starts;
  std::map<size_t, size_t> closes;
  scan_starts(body, starts, closes);
  struct TypeDecl {
    size_t from, to;
    std::string name, text;
  };
  std::vector<TypeDecl> types_decl;
  for (const Start& st0 : starts) {
    size_t k = st0.at;
    while (k < body.size() && std::isspace((unsigned char)body[k])) k++;
    std::string kw;
    for (const char* w : {"struct", "class", "union", "enum"}) {
      const size_t n = std::strlen(w);
      if (body.compare(k, n, w) == 0 && (k + n >= body.size() || !is_ident(body[k + n]))) kw = w;
    }
    if (kw.empty()) continue;
    bool inside = false;
    for (const TypeDecl& t : types_decl)
      if (k >= t.from && k < t.to) inside = true;
    if (inside) continue;
    size_t nm = k + kw.size();
    while (nm < body.size() && std::isspace((unsigned char)body[nm])) nm++;
    const size_t n0 = nm;
    while (nm < body.size() && is_ident(body[nm])) nm++;
    if (nm == n0) {
      why = "a " + kw + " declaration the body does not name";
      return false;
    }
    const std::string tname = body.substr(n0, nm - n0);
    const size_t brace = body.find('{', nm);
    const size_t sc = body.find(';', nm);
    const size_t close =
        brace == std::string::npos ? std::string::npos : find_matching(body, brace, '{', '}');
    if (sc != std::string::npos && (brace == std::string::npos || sc < brace)) {
      why = "a " + kw + " declaration with no body";
      return false;
    }
    size_t semi = close == std::string::npos ? std::string::npos : close + 1;
    while (semi != std::string::npos && semi < body.size() &&
           std::isspace((unsigned char)body[semi]))
      semi++;
    if (semi == std::string::npos || semi >= body.size() || body[semi] != ';') {
      why = "a " + kw + " declaration this cannot read";
      return false;
    }
    // A method is not a problem.
    for (const TypeDecl& t : types_decl) {
      if (t.name == tname) {
        why = "'" + tname + "', which the body declares twice";
        return false;
      }
    }
    types_decl.push_back({st0.at, semi + 1, tname, body.substr(st0.at, semi + 1 - st0.at)});
  }
  struct LambdaLocal {
    size_t from, to;
    std::string name, text;
    bool copies;  // the list takes a copy of a local: `[=]` or `[v]`
  };
  std::vector<LambdaLocal> lam_locals;
  auto list_copies = [](const std::string& caps) {
    size_t st = 0;
    for (size_t i = 0; i <= caps.size(); i++) {
      if (i < caps.size() && caps[i] != ',') continue;
      const std::string one = trim(caps.substr(st, i - st));
      st = i + 1;
      if (one.empty()) continue;
      if (one == "=") return true;
      if (one[0] != '&' && one != "this" && one.find('=') == std::string::npos) return true;
    }
    return false;
  };
  for (const Start& st0 : starts) {
    size_t k = st0.at;
    auto skip_ws_k = [&](size_t& q) {
      while (q < body.size() && std::isspace((unsigned char)body[q])) q++;
    };
    bool nested = false;
    for (const LambdaLocal& t : lam_locals)
      if (st0.at > t.from && st0.at < t.to) nested = true;
    if (nested) continue;
    skip_ws_k(k);
    if (body.compare(k, 5, "const") == 0) {
      k += 5;
      skip_ws_k(k);
    }
    if (body.compare(k, 4, "auto") != 0) continue;
    k += 4;
    skip_ws_k(k);
    const size_t n0 = k;
    while (k < body.size() && is_ident(body[k])) k++;
    if (k == n0) continue;
    const std::string lname = body.substr(n0, k - n0);
    skip_ws_k(k);
    if (k >= body.size() || body[k] != '=') continue;
    k++;
    skip_ws_k(k);
    if (k >= body.size() || body[k] != '[') continue;
    size_t back = st0.at;
    while (back > 0 && std::isspace((unsigned char)body[back - 1])) back--;
    if (back > 0 && body[back - 1] != '{' && body[back - 1] != ';' && body[back - 1] != ':' &&
        body[back - 1] != '}')
      continue;
    const size_t cap = find_matching(body, k, '[', ']');
    if (cap == std::string::npos) continue;
    const std::string caps = body.substr(k + 1, cap - k - 1);
    bool dup = false;
    for (const LambdaLocal& t : lam_locals)
      if (t.name == lname) dup = true;
    if (dup) continue;
    int depth = 0;
    size_t e = k;
    for (; e < body.size(); e++) {
      const char c = body[e];
      if (c == '(' || c == '[' || c == '{') depth++;
      else if (c == ')' || c == ']' || c == '}') depth--;
      else if (c == ';' && depth == 0) break;
    }
    if (e >= body.size()) continue;
    bool reassigned = false;
    for (size_t w = 0; w + lname.size() <= body.size(); w++) {
      if (body.compare(w, lname.size(), lname) != 0) continue;
      if (w > 0 && is_ident(body[w - 1])) continue;
      if (w >= st0.at && w <= e) continue;  // its own declaration
      size_t z = w + lname.size();
      while (z < body.size() && std::isspace((unsigned char)body[z])) z++;
      if (z < body.size() && body[z] == '=' && (z + 1 >= body.size() || body[z + 1] != '='))
        reassigned = true;
    }
    if (reassigned) continue;
    lam_locals.push_back(
        {st0.at, e + 1, lname, body.substr(st0.at, e + 1 - st0.at), list_copies(caps)});
    edits.push_back(
        {st0.at, e + 1, "weft_lam_" + std::to_string(lam_locals.size() - 1) + ";"});
  }
  // name -> the scopes it was declared in so far
  std::map<std::string, std::vector<std::pair<size_t, size_t>>> seen_at;
  std::map<std::string, std::string> seen_type;
  std::map<std::string, std::string> array_extent;  // an array local and its `[N]`
  auto skip_ws = [&](size_t& k) {
    while (k < body.size() && std::isspace((unsigned char)body[k])) k++;
  };
  for (const Start& st0 : starts) {
    size_t s0 = st0.at;
    size_t i = s0;
    skip_ws(i);
    bool in_type = false;
    const TypeDecl* lifted = nullptr;
    for (const TypeDecl& t : types_decl) {
      if (s0 == t.from) lifted = &t;
      if (s0 >= t.from && s0 < t.to) in_type = true;
    }
    if (lifted) {
      edits.push_back({lifted->from, lifted->to, ""});
      continue;
    }
    if (in_type) continue;
    bool in_lam = false;
    const LambdaLocal* lam = nullptr;
    for (const LambdaLocal& t : lam_locals) {
      if (s0 == t.from) lam = &t;
      if (s0 >= t.from && s0 < t.to) in_lam = true;
    }
    if (lam || in_lam) continue;
    for (const char* kw : {"case", "default", "goto", "return", "break", "continue",
                           "co_return", "co_await", "throw", "delete", "else", "do", "if",
                           "while", "for", "switch"}) {
      const size_t n = std::strlen(kw);
      if (body.compare(i, n, kw) == 0 && (i + n >= body.size() || !is_ident(body[i + n]))) {
        i = body.size();
      }
    }
    if (i >= body.size()) continue;
    const size_t decl0 = i;  // the declaration starts at its first qualifier
    bool q_const = false;
    while (body.compare(i, 5, "const") == 0 && i + 5 < body.size() &&
           std::isspace((unsigned char)body[i + 5])) {
      q_const = true;
      i += 5;
      skip_ws(i);
    }
    const size_t ty0 = i;
    while (i < body.size() && is_ident(body[i])) i++;
    const std::string ty = body.substr(ty0, i - ty0);
    if (ty.empty()) continue;
    skip_ws(i);
    std::string stars;
    while (i < body.size() && (body[i] == '*' || body[i] == '&')) {
      stars.push_back(body[i++]);
      skip_ws(i);
    }
    // A type with no name after it is not a declaration.
    if (i >= body.size() || !is_ident(body[i])) continue;
    struct Decl {
      std::string name, init, suffix;  // suffix is an array's `[N]`
    };
    std::vector<Decl> decls;
    size_t semi = std::string::npos;
    for (;;) {
      const size_t name0 = i;
      while (i < body.size() && is_ident(body[i])) i++;
      if (i == name0) {
        why = "a declaration this cannot read";
        return false;
      }
      Decl d;
      d.name = body.substr(name0, i - name0);
      skip_ws(i);
      while (i < body.size() && body[i] == '[') {
        const size_t close = find_matching(body, i, '[', ']');
        if (close == std::string::npos) {
          why = "an array declaration of '" + d.name + "' that is not closed";
          return false;
        }
        d.suffix += body.substr(i, close + 1 - i);
        i = close + 1;
        skip_ws(i);
      }
      if (i < body.size() && body[i] == '=') {
        int depth = 0;
        size_t k = i + 1;
        for (; k < body.size(); k++) {
          const char ch = body[k];
          if (ch == '(' || ch == '[' || ch == '{') depth++;
          else if (ch == ')' || ch == ']' || ch == '}') {
            if (depth == 0) break;
            depth--;
          } else if (ch == ';' && depth == 0) {
            break;
          } else if (ch == ',' && depth == 0) {
            size_t la = k + 1;
            while (la < body.size() && std::isspace((unsigned char)body[la])) la++;
            if (la < body.size() && is_ident(body[la]) &&
                !std::isdigit((unsigned char)body[la]))
              break;
          }
        }
        if (k >= body.size() || (body[k] != ';' && body[k] != ',')) {
          why = "a declaration of '" + d.name + "' this cannot split";
          return false;
        }
        d.init = trim(body.substr(i + 1, k - i - 1));
        i = k;
      } else if (i >= body.size() || (body[i] != ';' && body[i] != ',')) {
        why = "a declaration of '" + d.name + "' this cannot split";
        return false;
      }
      decls.push_back(d);
      if (body[i] == ',') {
        i++;
        skip_ws(i);
        continue;
      }
      semi = i;
      break;
    }
    std::string repl;
    for (const Decl& d : decls) {
      std::string field_ty = ty;
      for (const TypeDecl& t : types_decl)
        if (t.name == ty) field_ty = "weft_t_" + t.name + "_" + kname;
      if (q_const && !stars.empty()) field_ty = "const " + field_ty;
      std::string full = field_ty + stars;
      if (full == "auto") {
        if (d.init.empty()) {
          why = "'" + d.name + "', an `auto` local with no initialiser to take a type from";
          return false;
        }
        auto used_in = [](const std::string& s, const std::string& n) {
          for (size_t w = 0; w + n.size() <= s.size(); w++) {
            if (s.compare(w, n.size(), n) != 0) continue;
            if (w > 0 && (is_ident(s[w - 1]) || s[w - 1] == '.')) continue;
            if (w + n.size() < s.size() && is_ident(s[w + n.size()])) continue;
            return true;
          }
          return false;
        };
        const std::string hname = "weft_cd_" + kname + "_" + d.name;
        std::string hfields, hty = d.init;
        for (const auto& st2 : seen_type) {
          if (!used_in(d.init, st2.first)) continue;
          const auto ae = array_extent.find(st2.first);
          hfields += "  " + st2.second + " " + st2.first +
                     (ae == array_extent.end() ? "" : ae->second) + ";\n";
          hty = rename_word(hty, st2.first, "(((" + hname + "*)0)->" + st2.first + ")");
        }
        for (const std::string& pn : kparams) {
          if (!used_in(d.init, pn)) continue;
          const auto kt = kparam_types.find(pn);
          if (kt == kparam_types.end()) {
            why = "'" + d.name + "', an `auto` local whose type comes from the parameter '" +
                  pn + "', declared with a shape this cannot read";
            return false;
          }
          hfields += "  " + kt->second + " " + pn + ";\n";
          hty = rename_word(hty, pn, "(((" + hname + "*)0)->" + pn + ")");
        }
        type_helpers.push_back("struct " + hname + " {\n" + hfields + "};\n");
        field_ty = "std::decay_t<decltype(" + hty + ")>";
        full = field_ty;
      }
      const bool first_decl = !seen_type.count(d.name);
      if (seen_type.count(d.name) && seen_type[d.name] != full) {
        why = "'" + d.name + "', which the body declares as two types";
        return false;
      }
      for (const auto& other : seen_at[d.name]) {
        // Two scopes overlap when neither ends before the other starts.
        if (st0.at < other.second && other.first < st0.end) {
          why = "'" + d.name + "', declared again where the first is still in scope";
          return false;
        }
      }
      seen_at[d.name].push_back({st0.at, st0.end});
      seen_type[d.name] = full;
      if (!d.suffix.empty()) array_extent[d.name] = d.suffix;
      if (contains_word(reserved, d.name)) {
        why = "'" + d.name + "', which the kernel also binds as a parameter";
        return false;
      }
      if (d.name == "ctx" || d.name.rfind("weft", 0) == 0) {
        why = "'" + d.name + "', which the runtime's own code uses";
        return false;
      }
      names.push_back(d.name);
      if (first_decl) fields += "  " + field_ty + stars + " " + d.name + d.suffix + ";\n";
      if (!repl.empty()) repl += " ";
      if (!d.suffix.empty() && !d.init.empty()) {
        if (d.init.front() != '{' || d.init.back() != '}') {
          why = "'" + d.name + "', an array initialised with something other than a brace list";
          return false;
        }
        std::vector<long> ext;
        for (size_t q = 0; q < d.suffix.size(); q++) {
          if (d.suffix[q] != '[') continue;
          long v = 0;
          size_t r = q + 1;
          bool digit = false;
          for (; r < d.suffix.size() && std::isdigit((unsigned char)d.suffix[r]); r++) {
            v = v * 10 + (d.suffix[r] - '0');
            digit = true;
          }
          if (!digit || v <= 0) {
            why = "'" + d.name + "', an array whose extent this cannot read";
            return false;
          }
          ext.push_back(v);
        }
        long total = 1;
        for (long v : ext) total *= v;
        if (total <= 0 || total > 1024) {
          why = "'" + d.name + "', an array too large to fill in one field of the blob";
          return false;
        }
        std::vector<std::pair<std::string, std::string>> cells;
        if (!place_braces(d.init.substr(1, d.init.size() - 2), ext, 0, "", cells)) {
          why = "'" + d.name +
                "', an array initialiser this cannot place: a group must hold sub-groups "
                "or values, never both, and no more than its extent";
          return false;
        }
        std::map<std::string, std::string> named(cells.begin(), cells.end());
        std::vector<long> at(ext.size(), 0);
        for (long n2 = 0; n2 < total; n2++) {
          std::string ix;
          for (size_t d2 = 0; d2 < ext.size(); d2++) ix += "[" + std::to_string(at[d2]) + "]";
          if (n2) repl += " ";
          const auto it2 = named.find(ix);
          repl += d.name + ix + " = " +
                  (it2 == named.end() ? std::string("0") : it2->second) + ";";
          for (size_t d2 = ext.size(); d2-- > 0;) {
            if (++at[d2] < ext[d2]) break;
            at[d2] = 0;
          }
        }
      } else {
        repl += d.init.empty() ? std::string(";") : (d.name + " = " + d.init + ";");
      }
    }
    edits.push_back({decl0, semi + 1, repl});
  }
  // A lambda kept in a local.
  std::vector<std::string> lam_real(lam_locals.size(), std::string()), lam_structs;
  auto param_names_of = [](const std::string& params) {
    std::vector<std::string> out;
    const std::string list = params.size() >= 2 ? params.substr(1, params.size() - 2) : "";
    size_t st = 0;
    int depth = 0;
    for (size_t i = 0; i <= list.size(); i++) {
      const char c = i < list.size() ? list[i] : ',';
      if (c == '(' || c == '<' || c == '[') depth++;
      else if (c == ')' || c == '>' || c == ']') depth--;
      else if (c == ',' && depth == 0) {
        const std::string one = trim(list.substr(st, i - st));
        st = i + 1;
        size_t e2 = one.size();
        while (e2 > 0 && (std::isspace((unsigned char)one[e2 - 1]) || one[e2 - 1] == '&' ||
                          one[e2 - 1] == '*'))
          e2--;
        size_t s2 = e2;
        while (s2 > 0 && is_ident(one[s2 - 1])) s2--;
        if (e2 > s2) out.push_back(one.substr(s2, e2 - s2));
      }
    }
    return out;
  };
  for (size_t li = 0; li < lam_locals.size(); li++) {
    const LambdaLocal& l = lam_locals[li];
    const std::string& t = l.text;
    const size_t lb = t.find('[');
    const size_t rb =
        lb == std::string::npos ? std::string::npos : find_matching(t, lb, '[', ']');
    if (rb == std::string::npos) {
      why = "'" + l.name + "', a lambda local whose capture list this cannot read";
      return false;
    }
    const std::string caps = t.substr(lb + 1, rb - lb - 1);
    size_t k = rb + 1;
    while (k < t.size() && std::isspace((unsigned char)t[k])) k++;
    std::string params = "()";
    size_t after = k;
    if (k < t.size() && t[k] == '(') {
      const size_t par = find_matching(t, k, '(', ')');
      if (par == std::string::npos) {
        why = "'" + l.name + "', a lambda local whose parameters this cannot read";
        return false;
      }
      params = t.substr(k, par - k + 1);
      after = par + 1;
    }
    const size_t brace = t.find('{', after);
    const size_t close =
        brace == std::string::npos ? std::string::npos : find_matching(t, brace, '{', '}');
    if (close == std::string::npos) {
      why = "'" + l.name + "', a lambda local whose body this cannot read";
      return false;
    }
    const std::string between = trim(t.substr(after, brace - after));
    const bool is_mut = between.find("mutable") != std::string::npos;
    std::string ret = "auto";
    const size_t arrow = between.find("->");
    if (arrow != std::string::npos) ret = trim(between.substr(arrow + 2));
    const std::vector<std::string> pnames = param_names_of(params);
    auto is_param = [&](const std::string& n) {
      for (const std::string& p : pnames)
        if (p == n) return true;
      return false;
    };
    std::string inner = t.substr(brace + 1, close - brace - 1);
    std::vector<std::pair<std::string, bool>> items;  // name, by reference
    std::vector<std::pair<std::string, std::string>> capinits;  // name, initialiser
    std::vector<std::pair<std::string, std::string>> capalias;  // name, the local it binds
    struct CapRef {
      std::string name, expr, base;
    };
    std::vector<CapRef> capref;
    bool def_ref = false;
    size_t st = 0;
    int depth = 0;
    for (size_t i = 0; i <= caps.size(); i++) {
      const char c = i < caps.size() ? caps[i] : ',';
      if (c == '(' || c == '[' || c == '{') depth++;
      else if (c == ')' || c == ']' || c == '}') depth--;
      else if (c == ',' && depth == 0) {
        const std::string one = trim(caps.substr(st, i - st));
        st = i + 1;
        if (one.empty() || one == "=") continue;
        if (one == "&") {
          def_ref = true;
          continue;
        }
        if (one == "this") {
          why = "'" + l.name + "', a lambda local capturing `this`";
          return false;
        }
        const size_t eqt = one.find('=');
        if (eqt != std::string::npos) {
          const bool by_ref = one[0] == '&';
          const std::string cname = trim(one.substr(by_ref ? 1 : 0, eqt - (by_ref ? 1 : 0)));
          const std::string cexpr = trim(one.substr(eqt + 1));
          bool cident = !cexpr.empty();
          for (char c : cexpr)
            if (!is_ident(c)) cident = false;
          if (by_ref && cident && seen_type.count(cexpr)) {
            bool have = false;
            for (const auto& it2 : items)
              if (it2.first == cexpr && it2.second) have = true;
            if (!have) items.push_back({cexpr, true});
            capalias.push_back({cname, cexpr});
            continue;
          }
          if (by_ref) {
            size_t q2 = 0;
            while (q2 < cexpr.size() && is_ident(cexpr[q2])) q2++;
            const std::string base = cexpr.substr(0, q2);
            bool chain = !base.empty() && seen_type.count(base);
            while (q2 < cexpr.size()) {
              if (cexpr[q2] != '[') {
                chain = false;
                break;
              }
              const size_t cl = find_matching(cexpr, q2, '[', ']');
              if (cl == std::string::npos) {
                chain = false;
                break;
              }
              q2 = cl + 1;
            }
            if (chain) {
              capref.push_back({cname, cexpr, base});
              continue;
            }
            why = "'" + l.name + "', a by-reference capture of '" + cexpr +
                  "': only a local or an element of one can be bound";
            return false;
          }
          capinits.push_back({cname, cexpr});
          continue;
        }
        const bool ref = one[0] == '&';
        const std::string nm = ref ? trim(one.substr(1)) : one;
        bool ident = !nm.empty();
        for (char c2 : nm)
          if (!is_ident(c2)) ident = false;
        if (!ident) {
          why = "'" + l.name + "', a lambda local with a capture this cannot read";
          return false;
        }
        items.push_back({nm, ref});
      }
    }
    if (items.empty()) {
      for (const std::string& n : names) {
        if (is_param(n)) continue;
        for (size_t w = 0; w + n.size() <= inner.size(); w++) {
          if (inner.compare(w, n.size(), n) != 0) continue;
          if (w > 0 && (is_ident(inner[w - 1]) || inner[w - 1] == '.')) continue;
          if (w + n.size() < inner.size() && is_ident(inner[w + n.size()])) continue;
          items.push_back({n, def_ref});
          break;
        }
      }
      for (const std::string& n : kparams) {
        if (is_param(n)) continue;
        for (size_t w = 0; w + n.size() <= inner.size(); w++) {
          if (inner.compare(w, n.size(), n) != 0) continue;
          if (w > 0 && (is_ident(inner[w - 1]) || inner[w - 1] == '.')) continue;
          if (w + n.size() < inner.size() && is_ident(inner[w + n.size()])) continue;
          items.push_back({n, def_ref});
          break;
        }
      }
    }
    std::string members, args, rbody = inner;
    for (const auto& al : capalias)
      rbody = rename_word(rbody, al.first, "(*weft_c_" + al.second + ")");
    auto extent_of = [](const std::string& suffix) -> long {
      if (suffix.size() < 3 || suffix.front() != '[' || suffix.back() != ']') return -1;
      long v = 0;
      for (size_t i = 1; i + 1 < suffix.size(); i++) {
        if (!std::isdigit((unsigned char)suffix[i])) return -1;
        v = v * 10 + (suffix[i] - '0');
        if (v > 256) return -1;
      }
      return v;
    };
    for (const auto& it : items) {
      const std::string& n = it.first;
      if (is_param(n)) continue;
      const auto kt = kparam_types.find(n);
      if (kt != kparam_types.end()) {
        members += "  " + kt->second + " weft_c_" + n + ";\n";
        args += (args.empty() ? "" : ", ") + n;
        rbody = rename_word(rbody, n, "weft_c_" + n);
        continue;
      }
      if (!seen_type.count(n)) {
        why = "'" + l.name + "', a lambda local that names '" + n +
              "', which nothing the blob holds is called";
        return false;
      }
      const auto ae = array_extent.find(n);
      if (ae != array_extent.end()) {
        const long ext = extent_of(ae->second);
        if (ext < 0) {
          why = "'" + l.name + "', a lambda local that captures the array '" + n +
                "', whose extent this cannot read";
          return false;
        }
        if (it.second) {
          members += "  " + seen_type[n] + " (*weft_c_" + n + ")" + ae->second + ";\n";
          args += (args.empty() ? "" : ", ") + std::string("&weft_s->") + n;
          rbody = rename_word(rbody, n, "(*weft_c_" + n + ")");
        } else {
          members += "  " + seen_type[n] + " weft_c_" + n + ae->second + ";\n";
          std::string el;
          for (long i = 0; i < ext; i++)
            el += (i ? ", " : "") + std::string("weft_s->") + n + "[" + std::to_string(i) + "]";
          args += (args.empty() ? "" : ", ") + std::string("{") + el + "}";
          rbody = rename_word(rbody, n, "weft_c_" + n);
        }
        continue;
      }
      members += "  " + seen_type[n] + (it.second ? "*" : "") + " weft_c_" + n + ";\n";
      args += (args.empty() ? "" : ", ") + std::string(it.second ? "&" : "") + "weft_s->" + n;
      rbody = rename_word(rbody, n, it.second ? "(*weft_c_" + n + ")" : "weft_c_" + n);
    }
    for (size_t ci = 0; ci < capinits.size(); ci++) {
      const std::string& iname = capinits[ci].first;
      const std::string& iexpr = capinits[ci].second;
      bool ident = !iname.empty();
      for (char c : iname)
        if (!is_ident(c)) ident = false;
      if (!ident || iexpr.empty()) {
        why = "'" + l.name + "', an initialised capture whose name this cannot read";
        return false;
      }
      auto used = [](const std::string& s, const std::string& n) {
        for (size_t w = 0; w + n.size() <= s.size(); w++) {
          if (s.compare(w, n.size(), n) != 0) continue;
          if (w > 0 && (is_ident(s[w - 1]) || s[w - 1] == '.')) continue;
          if (w + n.size() < s.size() && is_ident(s[w + n.size()])) continue;
          return true;
        }
        return false;
      };
      const std::string hname = "weft_cd_" + kname + "_" + l.name + "_" + std::to_string(ci);
      std::string hfields, hty_expr = iexpr, hval = iexpr;
      for (const std::string& n : names) {
        if (n == iname || !used(iexpr, n)) continue;
        if (!seen_type.count(n)) {
          why = "'" + l.name + "', an initialised capture that reads '" + n +
                "', which nothing the blob holds is called";
          return false;
        }
        const auto ae = array_extent.find(n);
        hfields += "  " + seen_type[n] + " " + n + (ae == array_extent.end() ? "" : ae->second) +
                   ";\n";
        hty_expr = rename_word(hty_expr, n, "(((" + hname + "*)0)->" + n + ")");
        hval = rename_word(hval, n, "weft_s->" + n);
      }
      for (const std::string& n : kparams) {
        if (n == iname || !used(iexpr, n)) continue;
        const auto kt = kparam_types.find(n);
        if (kt == kparam_types.end()) {
          why = "'" + l.name + "', an initialised capture that reads the parameter '" + n +
                "', whose type is not written where this can read it";
          return false;
        }
        hfields += "  " + kt->second + " " + n + ";\n";
        hty_expr = rename_word(hty_expr, n, "(((" + hname + "*)0)->" + n + ")");
      }
      if (!hfields.empty()) lam_structs.push_back("struct " + hname + " {\n" + hfields + "};\n");
      members += "  std::decay_t<decltype(" + hty_expr + ")> weft_c_" + iname + ";\n";
      args += (args.empty() ? "" : ", ") + hval;
      rbody = rename_word(rbody, iname, "weft_c_" + iname);
    }
    for (const CapRef& cr : capref) {
      members += "  " + seen_type[cr.base] + "* weft_c_" + cr.name + ";\n";
      std::string arg = cr.expr;
      for (const std::string& n : names) arg = rename_word(arg, n, "weft_s->" + n);
      args += (args.empty() ? "" : ", ") + std::string("&") + arg;
      rbody = rename_word(rbody, cr.name, "(*weft_c_" + cr.name + ")");
    }
    for (const TypeDecl& ty : types_decl)
      rbody = rename_word(rbody, ty.name, "weft_t_" + ty.name + "_" + kname);
    // A kernel parameter is recomputed at every entry and is not part of a closure.
    for (size_t w = 0; w < rbody.size(); w++) {
      if (!is_ident(rbody[w])) continue;
      const size_t s2 = w;
      while (w < rbody.size() && is_ident(rbody[w])) w++;
      const std::string word = rbody.substr(s2, w - s2);
      bool kernel_param = false;
      for (const std::string& pn : kparams)
        if (pn == word) kernel_param = true;
      if (kernel_param) {
        why = "'" + l.name + "', a lambda local that reads the kernel parameter '" + word +
              "': a closure has no argument list for it";
        return false;
      }
    }
    for (size_t b = rbody.find('['); b != std::string::npos; b = rbody.find('[', b + 1)) {
      if (b > 0 && (is_ident(rbody[b - 1]) || rbody[b - 1] == ']' || rbody[b - 1] == ')'))
        continue;
      const size_t c2 = find_matching(rbody, b, '[', ']');
      if (c2 == std::string::npos) continue;
      const std::string list = rbody.substr(b + 1, c2 - b - 1);
      std::string fixed, one;
      auto flush = [&]() {
        const std::string it = trim(one);
        one.clear();
        if (it.empty()) return;
        std::string out = it;
        if (it.size() > 3 && it.compare(0, 2, "(*") == 0 && it.back() == ')') {
          out = it.substr(2, it.size() - 3);
        }
        bool member = out.rfind("weft_c_", 0) == 0;
        for (char c : out)
          if (!is_ident(c)) member = false;
        if (member) out = out + " = " + out;
        if (!fixed.empty()) fixed += ", ";
        fixed += out;
      };
      for (char c : list) {
        if (c == ',') {
          flush();
          continue;
        }
        one.push_back(c);
      }
      flush();
      if (fixed != list) rbody = rbody.substr(0, b + 1) + fixed + rbody.substr(c2);
    }
    const std::string sname = "weft_lam_" + kname + "_" + l.name;
    lam_structs.push_back("struct " + sname + " {\n" + members + "  " + ret + " operator()" +
                          params + (is_mut ? "" : " const") + " {\n" + rbody + "\n  }\n};\n");
    fields += "  " + sname + " " + l.name + ";\n";
    lam_real[li] = "weft_s->" + l.name + " = " + sname + "{" + args + "};";
    // Its uses in the body follow it into the blob like any other local's.
    names.push_back(l.name);
  }
  std::sort(edits.begin(), edits.end(),
            [](const Edit& a, const Edit& b) { return a.from < b.from; });
  for (size_t e = edits.size(); e-- > 0;)
    body = body.substr(0, edits[e].from) + edits[e].text + body.substr(edits[e].to);
  // A lambda's own parameters first: the blob rename would rewrite them.
  if (!freshen_lambda_params(body, names, why)) return false;
  for (const std::string& n : names) body = rename_blob_local(body, n);
  for (const TypeDecl& t : types_decl) {
    const std::string to = "weft_t_" + t.name + "_" + kname;
    types += rename_word(t.text, t.name, to) + "\n";
    body = rename_word(body, t.name, to);
  }
  // Then the capture lists, which cannot name a blob field.
  if (!rebuild_captures(body, names, why)) return false;
  for (const std::string& s : type_helpers) types += s;
  for (const std::string& s : lam_structs) types += s;
  for (size_t li = 0; li < lam_locals.size(); li++)
    body = replace_all(body, "weft_lam_" + std::to_string(li) + ";", lam_real[li]);
  return true;
}

bool lower_step(const std::string& body_in, const std::string& kname, const std::string& reserved,
                std::string& fields, std::string& types, std::string& code, std::string& cases,
                std::string& why) {
  std::string body = body_in;
  // An address space says where a variable lives on the GPU.
  auto drop_space = [](std::string s, const std::string& w, const std::string& to) {
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
      if (s.compare(i, w.size(), w) == 0 && (i == 0 || !is_ident(s[i - 1])) &&
          (i + w.size() >= s.size() || !is_ident(s[i + w.size()]))) {
        out += to;
        i += w.size();
        continue;
      }
      out.push_back(s[i++]);
    }
    return out;
  };
  body = drop_space(body, "device", "");
  body = drop_space(body, "thread", "");
  body = drop_space(body, "constant", "const");
  body = replace_all(body, "const const", "const");
  for (const char* no : {"//", "/*", "\"", "'", "#", "template", "asm", "threadgroup"}) {
    if (body.find(no) != std::string::npos) {
      why = std::string("a body with `") + no + "` in it";
      return false;
    }
  }
  if (body.find("({") != std::string::npos) {
    why = "an inlined helper that returns before its last statement";
    return false;
  }
  if (!hoist_locals(body, kname, reserved, fields, types, why)) return false;
  BodyParser bp(body);
  Nd root;
  if (!bp.seq(root, true)) {
    why = bp.why;
    return false;
  }
  StepEmitter em;
  em.emit(root);
  if (!em.ok) {
    why = em.why;
    return false;
  }
  code = em.out;
  cases = em.cases;
  return true;
}

}  // namespace

std::string attr_field_name(const std::string& src, const std::string& type,
                            const std::string& attr) {
  const size_t def = src.find("struct " + type);
  if (def == std::string::npos) return std::string();
  const size_t open = src.find('{', def);
  if (open == std::string::npos) return std::string();
  const size_t close = find_matching(src, open, '{', '}');
  if (close == std::string::npos || close < open) return std::string();
  const std::string text = src.substr(open + 1, close - open - 1);
  size_t at = 0;
  while (true) {
    const size_t semi = text.find(';', at);
    if (semi == std::string::npos) break;
    const std::string stmt = text.substr(at, semi - at);
    at = semi + 1;
    const size_t lb = stmt.find("[["), rb = stmt.find("]]", lb == std::string::npos ? 0 : lb);
    if (lb == std::string::npos || rb == std::string::npos) continue;
    if (trim(stmt.substr(lb + 2, rb - lb - 2)).rfind(attr, 0) != 0) continue;
    const std::string head = trim(stmt.substr(0, lb));
    size_t e = head.size();
    while (e > 0 && !is_ident(head[e - 1])) e--;
    size_t b = e;
    while (b > 0 && is_ident(head[b - 1])) b--;
    return head.substr(b, e - b);
  }
  return std::string();
}

struct StructField {
  std::string type, name, attr;
};

// The fields a struct declares, in order.
std::vector<StructField> read_struct_fields(const std::string& src, const std::string& type) {
  std::vector<StructField> out;
  const size_t def = src.find("struct " + type);
  if (def == std::string::npos) return out;
  const size_t open = src.find('{', def);
  if (open == std::string::npos) return out;
  const size_t close = find_matching(src, open, '{', '}');
  if (close == std::string::npos || close < open) return out;
  const std::string text = src.substr(open + 1, close - open - 1);
  size_t at = 0;
  while (true) {
    const size_t semi = text.find(';', at);
    if (semi == std::string::npos) break;
    std::string f = text.substr(at, semi - at);
    at = semi + 1;
    StructField fd;
    const size_t lb = f.find("[[");
    if (lb != std::string::npos) {
      const size_t rb = f.find("]]", lb);
      if (rb != std::string::npos) {
        fd.attr = trim(f.substr(lb + 2, rb - lb - 2));
        f = f.substr(0, lb) + f.substr(rb + 2);
      }
    }
    std::string word;
    for (char c : f) {
      if (std::isspace((unsigned char)c)) {
        if (!word.empty()) {
          if (fd.type.empty()) {
            fd.type = word;
          } else if (word[0] == '[' && !fd.name.empty()) {
            fd.name += word;
          } else {
            fd.name = word;
          }
          word.clear();
        }
      } else {
        word.push_back(c);
      }
    }
    if (!word.empty()) {
      if (fd.type.empty()) {
        fd.type = word;
      } else if (word[0] == '[' && !fd.name.empty()) {
        fd.name += word;
      } else {
        fd.name = word;
      }
    }
    if (!fd.name.empty()) out.push_back(fd);
  }
  return out;
}

struct VertexStage {
  bool ok = false;
  std::string name, ret;
  std::string new_sig, orig_sig;  // the CPU signature and the file's own
  std::string new_all, orig_all;  // the whole function, both spellings
  std::string unsupplied;
  struct Param {
    std::string type, name;
    std::string orig_type;
    int buffer = -1;
    bool vertex_id = false;
    bool by_ref = false;
    bool pointer = false;
    bool is_const = false;
    bool texture = false;
    bool sampler = false;
    bool zero_index = false;
  };
  std::vector<Param> params;
  std::vector<Param> sampler_params;
  std::vector<TexParam> textures;                       // sampled in the body
  std::map<std::string, std::pair<int, int>> samplers;  // its constexpr ones
  std::vector<int> samp_slots;                          // its [[sampler(n)]] ones
  int ntex = 0;
};

VertexStage read_vertex_stage(std::string& src) {
  VertexStage vs;
  size_t at = 0;
  while ((at = src.find("vertex", at)) != std::string::npos) {
    if ((at > 0 && is_ident(src[at - 1])) || (at + 6 < src.size() && is_ident(src[at + 6]))) {
      at += 6;
      continue;
    }
    const size_t par = src.find('(', at);
    if (par == std::string::npos) return vs;
    const std::string head = trim(src.substr(at + 6, par - at - 6));
    const size_t sp = head.find_last_of(" \t\r\n");
    if (sp == std::string::npos) { at = par; continue; }
    VertexStage out;
    out.name = trim(head.substr(sp + 1));
    out.ret = trim(head.substr(0, sp));
    if (out.name.empty() || out.ret.empty()) { at = par; continue; }
    const size_t close = find_matching(src, par, '(', ')');
    if (close == std::string::npos) return vs;

    std::string sig;
    bool bad = false;
    for (const std::string& rawp : split_top_level(src.substr(par + 1, close - par - 1))) {
      if (trim(rawp).empty()) continue;
      std::string decl = rawp;
      VertexStage::Param p;
      const size_t lb = decl.find("[[");
      if (lb != std::string::npos) {
        const size_t rb = decl.find("]]", lb);
        if (rb == std::string::npos) { bad = true; break; }
        const std::string attr = trim(decl.substr(lb + 2, rb - lb - 2));
        decl = decl.substr(0, lb) + decl.substr(rb + 2);
        if (attr == "vertex_id") {
          p.vertex_id = true;
        } else if (attr.rfind("texture", 0) == 0) {
          const size_t a = attr.find('('), b = attr.find(')');
          if (a == std::string::npos || b <= a) { bad = true; break; }
          p.texture = true;
          p.buffer = std::atoi(trim(attr.substr(a + 1, b - a - 1)).c_str());
        } else if (attr.rfind("sampler", 0) == 0) {
          const size_t a = attr.find('('), b = attr.find(')');
          p.sampler = true;
          p.buffer = (a == std::string::npos || b <= a)
                         ? 0
                         : std::atoi(trim(attr.substr(a + 1, b - a - 1)).c_str());
        } else if (attr.rfind("buffer", 0) == 0) {
          const size_t a = attr.find('('), b = attr.find(')');
          if (a == std::string::npos || b <= a) { bad = true; break; }
          p.buffer = std::atoi(trim(attr.substr(a + 1, b - a - 1)).c_str());
        } else if (attr == "instance_id" || attr == "base_instance" ||
                   attr == "base_vertex") {
          // One instance, starting at vertex 0: the argument names a zero.
          p.zero_index = true;
        } else {
          vs.unsupplied = attr;
          bad = true;
          break;
        }
      }
      p.pointer = decl.find('*') != std::string::npos;
      p.by_ref = decl.find('&') != std::string::npos;
      decl = trim(decl);
      decl = replace_all(decl, "constant ", "const ");
      decl = replace_all(decl, "device ", "");
      decl = replace_all(decl, "thread ", "");
      p.is_const = decl.find("const ") != std::string::npos;
      const size_t cut = decl.find_last_of(" \t\r\n");
      if (cut == std::string::npos) { bad = true; break; }
      p.name = trim(decl.substr(cut + 1));
      p.type = trim(decl.substr(0, cut));
      p.name.erase(std::remove(p.name.begin(), p.name.end(), '&'), p.name.end());
      p.name.erase(std::remove(p.name.begin(), p.name.end(), '*'), p.name.end());
      if (p.type.rfind("const ", 0) == 0) p.type = trim(p.type.substr(6));
      for (const char* c : {"&", "*"})
        p.type.erase(std::remove(p.type.begin(), p.type.end(), *c), p.type.end());
      p.type = trim(p.type);
      p.orig_type = p.type;
      if (p.name.empty() || p.type.empty()) { bad = true; break; }
      if (p.sampler) {
        out.samplers[p.name] = {0, 0};
        out.samp_slots.push_back(p.buffer);
        out.sampler_params.push_back(p);
        continue;
      }
      if (p.texture) {
        TexParam tp;
        tp.name = p.name;
        tp.value = p.type.find("half") != std::string::npos ? "half" : "float";
        tp.is1d = p.type.find("texture1d") != std::string::npos;
        tp.is3d = p.type.find("texture3d") != std::string::npos;
        tp.cube = p.type.find("cube") != std::string::npos;
        tp.array = p.type.find("array") != std::string::npos;
        out.textures.push_back(tp);
        if (out.ntex <= p.buffer) out.ntex = p.buffer + 1;
        decl = "weft::Texture " + p.name;
      }
      out.params.push_back(p);
      if (!sig.empty()) sig += ", ";
      sig += decl;
    }
    if (bad) { at = close + 1; continue; }

    const std::string newsig = out.ret + " " + out.name + "(" + sig + ")";
    const size_t brace0 = src.find('{', close);
    const size_t bend0 = brace0 == std::string::npos ? std::string::npos
                                                     : find_matching(src, brace0, '{', '}');
    if (bend0 == std::string::npos) { at = close + 1; continue; }
    out.orig_sig = src.substr(at, close - at + 1);
    out.new_sig = newsig;
    out.orig_all = src.substr(at, bend0 - at + 1);

    src.replace(at, close - at + 1, newsig);
    size_t brace = src.find('{', at);
    size_t bend = brace == std::string::npos ? std::string::npos
                                             : find_matching(src, brace, '{', '}');
    if (bend == std::string::npos) { at = close + 1; continue; }
    if (!out.textures.empty()) {
      // The sample sites become the portable read, exactly as in a kernel.
      std::string berr;
      std::string body = src.substr(brace + 1, bend - brace - 1);
      body = strip_samplers(body, out.samplers, berr);
      if (berr.empty()) body = rewrite_textures(body, out.textures, out.samplers, berr);
      if (!berr.empty()) { at = close + 1; continue; }
      src.replace(brace + 1, bend - brace - 1, body);
      brace = src.find('{', at);
      bend = brace == std::string::npos ? std::string::npos
                                        : find_matching(src, brace, '{', '}');
      if (bend == std::string::npos) { at = close + 1; continue; }
    }
    out.new_all = src.substr(at, bend - at + 1);
    out.ok = true;
    vs = out;
    at += newsig.size();
  }
  return vs;
}

std::string init_const_globals(const std::string& in) {
  std::string out;
  size_t at = 0;
  while (at <= in.size()) {
    const size_t nl = in.find('\n', at);
    const std::string line = in.substr(at, (nl == std::string::npos ? in.size() : nl) - at);
    const std::string t = trim(line);
    std::string addr;
    for (const char* p : {"constant ", "const "})
      if (t.rfind(p, 0) == 0) { addr = p; break; }
    const bool plain = !addr.empty() && t.size() > addr.size() + 2 && t.back() == ';' &&
                       t.find('=') == std::string::npos &&
                       t.find('(') == std::string::npos &&
                       t.find('{') == std::string::npos &&
                       t.find('[') == std::string::npos;
    if (plain) {
      const size_t indent = line.find_first_not_of(" \t");
      const std::string lead = indent == std::string::npos ? std::string() : line.substr(0, indent);
      out += lead + "const " +
             trim(t.substr(addr.size(), t.size() - addr.size() - 1)) + " = {};";
      if (nl != std::string::npos) out += '\n';
    } else {
      out += line;
      if (nl != std::string::npos) out += '\n';
    }
    if (nl == std::string::npos) break;
    at = nl + 1;
  }
  return out;
}

std::vector<std::string> vector_decl_names(const std::string& src) {
  static const char* kBases[] = {"packed_float", "packed_half", "float", "half", "uint",
                                 "int",          "ushort",      "short", "char", "uchar",
                                 "bool"};
  std::vector<std::string> names;
  for (size_t i = 0; i < src.size();) {
    if (!std::isalpha((unsigned char)src[i])) {
      i++;
      continue;
    }
    size_t j = i;
    while (j < src.size() && is_ident(src[j])) j++;
    const std::string w = src.substr(i, j - i);
    bool wide = false;
    if (w.size() > 2 && w.back() >= '2' && w.back() <= '4') {
      const std::string base = w.substr(0, w.size() - 1);
      for (const char* b : kBases)
        if (base == b) wide = true;
    }
    if (wide) {
      size_t k = j;
      while (k < src.size() && std::isspace((unsigned char)src[k])) k++;
      if (k < src.size() && (std::isalpha((unsigned char)src[k]) || src[k] == '_')) {
        size_t e = k;
        while (e < src.size() && is_ident(src[e])) e++;
        const std::string nm = src.substr(k, e - k);
        bool seen = false;
        for (const std::string& s : names)
          if (s == nm) seen = true;
        if (!seen) names.push_back(nm);
      }
    }
    i = j;
  }
  return names;
}

struct FragLower {
  std::string name;
  std::string interp;
  std::string add;
  std::string calls;
};

bool lower_fragment(std::string& src, const FragLower& l, const std::string& prelude) {
  size_t at = 0, paren = std::string::npos;
  while ((at = src.find(l.name, at)) != std::string::npos) {
    if (at > 0 && is_ident(src[at - 1])) { at += l.name.size(); continue; }
    const size_t p = src.find('(', at + l.name.size());
    if (p == std::string::npos) return false;
    paren = p;
    break;
  }
  if (paren == std::string::npos) return false;
  const size_t close = find_matching(src, paren, '(', ')');
  if (close == std::string::npos) return false;

  auto param_name = [](const std::string& p) -> std::string {
    const size_t lb = p.find("[[");
    const std::string head = trim(p.substr(0, lb == std::string::npos ? p.size() : lb));
    size_t e = head.size();
    while (e > 0 && !is_ident(head[e - 1])) e--;
    size_t b = e;
    while (b > 0 && is_ident(head[b - 1])) b--;
    return head.substr(b, e - b);
  };

  std::string sig, extra, pos;
  for (const std::string& p : split_top_level(src.substr(paren + 1, close - paren - 1))) {
    if (p.find("[[stage_in]]") != std::string::npos) continue;
    // The point selection the pass computed is the sprite coordinate.
    if (p.find("[[point_coord]]") != std::string::npos) {
      const std::string ty = trim(p.substr(0, p.find(' ')));
      extra += "  const " + ty + " " + param_name(p) + " = " + ty +
               "(weft_pcx, weft_pcy);\n";
      continue;
    }
    // `[[barycentric_coord]]` names the same weights the interpolation computed.
    if (p.find("barycentric_coord") != std::string::npos) {
      const bool np = p.find("no_perspective") != std::string::npos;
      extra += "  const float3 " + param_name(p) + " = float3(" +
               (np ? "weft_l0, weft_l1, weft_l2"
                   : "weft_q0 * weft_iq, weft_q1 * weft_iq, weft_q2 * weft_iq") +
               ");\n";
      continue;
    }
    if (p.find("[[position]]") != std::string::npos) pos = param_name(p);
    if (!sig.empty()) sig += ", ";
    sig += p;
  }
  std::string add = l.add;
  if (pos.empty()) add = "float4 weft_frag_pos [[position]]" + (add.empty() ? "" : ", " + add);
  if (!sig.empty() && !add.empty()) sig += ", ";
  sig += add;

  // Metal's library has the same names without the namespace.
  std::string interp = replace_all(l.interp, "std::", "");
  const std::string px = (pos.empty() ? std::string("weft_frag_pos") : pos) + ".x";
  const std::string py = (pos.empty() ? std::string("weft_frag_pos") : pos) + ".y";
  interp = replace_all(interp, "@PX@", px);
  interp = replace_all(interp, "@PY@", py);
  interp = replace_all(interp, "@RET@", "discard_fragment();");

  const size_t brace = src.find('{', close);
  if (brace == std::string::npos) return false;
  src.replace(brace + 1, 0, "\n" + l.calls + interp + extra);
  src.replace(paren + 1, close - paren - 1, sig);
  if (!prelude.empty()) {
    size_t start = at;
    while (start > 0 && src[start - 1] != '}') start--;
    src.insert(start, "\n" + prelude + "\n");
  }
  return true;
}

TranslateResult translate_msl(const std::string& raw, const RenderMode* render) {
  TranslateResult r;
  std::string src = strip_comments(raw);
  src = fuse_mul_add(fuse_accum_products(src));
  // The `[[function_constant(n)]]` globals, in binding order.
  std::vector<std::pair<int, std::string>> fn_consts;
  VertexStage vstage;
  if (render) vstage = read_vertex_stage(src);
  // A fragment that reads `[[point_coord]]` makes the pass a point pass.
  const bool file_points = render && src.find("point_coord") != std::string::npos;
  const std::string vpos_name =
      vstage.ok ? attr_field_name(src, vstage.ret, "position") : std::string();
  const std::string vsize_name =
      vstage.ok ? attr_field_name(src, vstage.ret, "point_size") : std::string();
  std::string vbody;
  if (vstage.ok) {
    const size_t at = src.find(vstage.name + "(");
    const size_t brace = at == std::string::npos ? std::string::npos : src.find('{', at);
    const size_t end =
        brace == std::string::npos ? std::string::npos : find_matching(src, brace, '{', '}');
    if (brace != std::string::npos && end != std::string::npos) vbody = src.substr(brace, end - brace + 1);
  }
  const bool vheavy = vbody.size() > 240 || vbody.find("for (") != std::string::npos ||
                      vbody.find("for(") != std::string::npos ||
                      vbody.find("while (") != std::string::npos ||
                      vbody.find("while(") != std::string::npos ||
                      vbody.find("texture") != std::string::npos;
  bool vstage_pure = vstage.ok && vheavy;
  for (const auto& p : vstage.params) {
    if (p.pointer) vstage_pure = false;
    if (p.texture && p.orig_type.find("write") != std::string::npos) vstage_pure = false;
  }
  std::vector<FragLower> lowers;
  int stage_in_fns = 0;

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

  std::string decl_src = src;
  for (const char* inc : {"#include <metal_stdlib>\n", "#include <metal_stdlib>",
                          "#include <metal_math>\n", "#include <metal_math>",
                          "#include <metal_compute>\n", "#include <metal_compute>"}) {
    decl_src = replace_all(decl_src, inc, "");
  }
  decl_src = replace_all(decl_src, "using namespace metal;", "");

  for (size_t at = 0; (at = decl_src.find("[[function_constant", at)) != std::string::npos;) {
    const size_t rb = decl_src.find("]]", at);
    if (rb == std::string::npos) break;
    const size_t a = decl_src.find('(', at), b = decl_src.find(')', at);
    const int slot = (a != std::string::npos && b != std::string::npos && b > a && b < rb)
                         ? std::atoi(trim(decl_src.substr(a + 1, b - a - 1)).c_str())
                         : 0;
    size_t ls = decl_src.rfind('\n', at);
    ls = (ls == std::string::npos) ? 0 : ls + 1;
    std::string decl = trim(decl_src.substr(ls, at - ls));
    for (const char* addr : {"constant ", "const "})
      if (decl.rfind(addr, 0) == 0) { decl = trim(decl.substr(std::strlen(addr))); break; }
    const size_t sp = decl.find_first_of(" \t");
    std::string type = sp == std::string::npos ? decl : decl.substr(0, sp);
    if (type.empty()) type = "bool";
    fn_consts.emplace_back(slot, type);
    decl_src.erase(at, rb + 2 - at);
  }
  decl_src = init_const_globals(decl_src);

  const CarriedDecls carried = carry_decls(decl_src);

  std::string body_out;
  int kcount = 0;
  std::vector<int> frag_sampler_slots;  // every `[[sampler(n)]]` a fragment names
  std::vector<GpuPad> gpu_pads;  // fragments whose colour output needs widening
  bool points_run = false;       // a fragment reads `[[point_coord]]`
  bool vertex_used = false;  // the pass fills `stage_in` with a vertex stage

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
    // A render pass translates its fragment stages and its tile stage.
    if (render && kw == "kernel" && !is_tile) {
      pos = close_brace + 1;
      continue;
    }

    std::string locals;      // everything, for the per-thread form
    std::string buf_locals;  // buffers only, for the loop form
    std::vector<TexParam> textures;
    std::map<std::string, std::pair<int, int>> samplers_for_body;
    std::vector<std::pair<std::string, std::string>> gid_params;
    bool only_gid = true;
    int nargs = 0;
    bool unsupported = false;
    std::string why;
    std::string ib_type, ib_name;  // tile memory: its element type and name
    std::vector<std::pair<int, std::string>> buf_types_at;
    bool is_mesh = false;
    int mesh_nv = 0, mesh_np = 0;
    std::string mesh_vtype;
    int extra_bufs = 0;
    int payload_idx = -1, props_idx = -1;
    std::string payload_type;

    // The highest binding the kernel declares.
    int max_buf_idx = -1;
    for (const std::string& rp : split_top_level(params)) {
      const size_t ba = rp.find("buffer(");
      if (ba == std::string::npos) continue;
      const int bi = std::atoi(rp.c_str() + ba + 7);
      if (bi > max_buf_idx) max_buf_idx = bi;
    }
    int next_idx = max_buf_idx + 1;  // the first binding the launcher supplies

    bool points_prelude = false;
    // The clip distances the pass declares, expanded to one name per plane.
    std::vector<std::string> clip_fields;
    std::string points_gpu, points_params;
    auto ensure_points_prelude = [&]() -> bool {
      if (points_prelude) return true;
      if (vpos_name.empty()) {
        unsupported = true;
        why = "[[point_coord]] needs the vertex stage's `[[position]]` field";
        return false;
      }
      const std::string dw = std::to_string(render->width);
      const std::string dh = std::to_string(render->height);
      std::vector<std::string> parts;
      for (const auto& p : vstage.params) {
        if (p.texture) {
          parts.push_back(p.orig_type + " weft_stage_tex_" + std::to_string(p.buffer) +
                          " [[texture(" + std::to_string(p.buffer) + ")]]");
        } else if (p.buffer >= 0) {
          parts.push_back((p.pointer ? "device " + p.type + "* " : "constant " + p.type + "& ") +
                          "weft_stage_buf_" + std::to_string(p.buffer) + " [[buffer(" +
                          std::to_string(p.buffer) + ")]]");
          if (nargs <= p.buffer) nargs = p.buffer + 1;
        }
      }
      for (const auto& p : vstage.sampler_params) {
        parts.push_back("sampler weft_stage_smp_" + std::to_string(p.buffer) +
                        " [[sampler(" + std::to_string(p.buffer) + ")]]");
        frag_sampler_slots.push_back(p.buffer);
      }
      for (const std::string& part : parts) {
        if (!points_params.empty()) points_params += ", ";
        points_params += part;
      }
      std::string out, gpu;
      for (int i = 0; i < 3; i++) {
        std::string args, gargs;
        for (const auto& p : vstage.params) {
          if (!args.empty()) { args += ", "; gargs += ", "; }
          if (p.vertex_id) {
            args += std::to_string(i) + "u";
            gargs += std::to_string(i) + "u";
          } else if (p.zero_index) {
            args += "0u";
            gargs += "0u";
          } else if (p.texture) {
            args += "weft::texture(" + std::to_string(p.buffer) + ")";
            gargs += "weft_stage_tex_" + std::to_string(p.buffer);
          } else if (p.buffer >= 0) {
            args += p.pointer
                        ? "(" + p.type + "*)weft::buffer(ctx, " + std::to_string(p.buffer) + ")"
                        : "*(const " + p.type + "*)weft::buffer(ctx, " + std::to_string(p.buffer) +
                              ")";
            gargs += "weft_stage_buf_" + std::to_string(p.buffer);
          } else {
            unsupported = true;
            why = "vertex stage parameter '" + p.name +
                  "' is neither a vertex id nor a buffer";
            return false;
          }
        }
        for (const auto& p : vstage.sampler_params)
          gargs += (gargs.empty() ? "" : ", ") + std::string("weft_stage_smp_") +
                   std::to_string(p.buffer);
        out += "  " + vstage.ret + " weft_v" + std::to_string(i) + " = " + vstage.name + "(" +
               args + ");\n";
        gpu += "  " + vstage.ret + " weft_v" + std::to_string(i) +
               " = weft_plain_vstage(" + gargs + ");\n";
      }
      auto body = [&](const std::string& xexpr, const std::string& yexpr,
                      const std::string& ret) -> std::string {
        std::string t;
        t += "  const float weft_ppx = " + xexpr + ";\n";
        t += "  const float weft_ppy = " + yexpr + ";\n";
        t += "  int weft_which = -1;  // the point this pixel shows, if any\n";
        t += "  float weft_pcx = 0.5f, weft_pcy = 0.5f;\n";
        for (int i = 0; i < 3; i++) {
          const std::string v = "weft_v" + std::to_string(i);
          const std::string sx =
              "std::round((weft_clip.x / weft_clip.w * 0.5f + 0.5f) * " + dw +
              ".0f * 256.0f) * 0.00390625f";
          const std::string sy =
              "std::round((0.5f - weft_clip.y / weft_clip.w * 0.5f) * " + dh +
              ".0f * 256.0f) * 0.00390625f";
          t += "  {\n";
          t += "    const float4 weft_clip = " + v + "." + vpos_name + ";\n";
          t += "    if (weft_clip.w > 0.0f) {\n";
          t += "      const float weft_psx = " + sx + ";\n";
          t += "      const float weft_psy = " + sy + ";\n";
          t += "      const float weft_psz = " +
               (vsize_name.empty() ? std::string("0.0f")
                                   : "(" + v + "." + vsize_name + ")") +
               ";\n";
          std::string clip_ok;
          for (const std::string& cn : clip_fields)
            clip_ok += " && " + v + "." + cn + " >= 0.0f";
          t += "      if (weft_ppx >= weft_psx - weft_psz * 0.5f &&\n";
          t += "          weft_ppx < weft_psx + weft_psz * 0.5f &&\n";
          t += "          weft_ppy >= weft_psy - weft_psz * 0.5f &&\n";
          t += "          weft_ppy < weft_psy + weft_psz * 0.5f" + clip_ok + ") {\n";
          t += "        weft_which = " + std::to_string(i) + ";\n";
          t += "        weft_pcx = (weft_ppx - (weft_psx - weft_psz * 0.5f)) / weft_psz;\n";
          t += "        weft_pcy = (weft_ppy - (weft_psy - weft_psz * 0.5f)) / weft_psz;\n";
          t += "      }\n    }\n  }\n";
        }
        t += "  if (weft_which < 0) " + ret + "\n";
        return t;
      };
      out += body("((float)(ctx.gid % " + dw + "u) + 0.5f)",
                  "((float)(ctx.gid / " + dw + "u) + 0.5f)", "return;");
      gpu += body("@PX@", "@PY@", "@RET@");
      locals += out;
      points_gpu = gpu;
      points_prelude = true;
      return true;
    };

    clip_fields.clear();
    if (render && vstage.ok) {
      for (const StructField& vf : read_struct_fields(src, vstage.ret)) {
        if (vf.attr.find("clip_distance") == std::string::npos) continue;
        std::string nm = vf.name;
        const size_t lbk = nm.find('[');
        std::string count;
        if (lbk != std::string::npos) {
          const size_t rb = nm.find(']', lbk);
          if (rb == std::string::npos) {
            r.error = "the vertex stage's clip array '" + vf.name + "' has no bound";
            return r;
          }
          count = nm.substr(lbk + 1, rb - lbk - 1);
          nm = nm.substr(0, lbk);
        }
        if (vf.type != "float" ||
            (!count.empty() && (count.size() > 2 ||
                                count.find_first_not_of("0123456789") != std::string::npos))) {
          r.error = "[[clip_distance]] needs float planes, not " + vf.type + " " + vf.name;
          return r;
        }
        if (count.empty()) {
          clip_fields.push_back(nm);
        } else {
          for (int e = 0; e < std::stoi(count); e++)
            clip_fields.push_back(nm + "[" + std::to_string(e) + "]");
        }
      }
    }

    for (const std::string& rawp : split_top_level(params)) {
      if (rawp.empty() || rawp == "void") continue;
      std::vector<std::string> attrs;
      std::string decl = strip_attributes(rawp, attrs);
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

      // A mesh stage's output.
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
        // The view is this group's slice.
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

      // The task payload.
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
        const std::string pline = "  " + ptype + "& " + pname + " = ((" + ptype +
                                  "*)weft::buffer(ctx, " + std::to_string(payload_idx) +
                                  "))[ctx.tgid];\n";
        locals += pline;
        buf_locals += pline;
        only_gid = false;
        continue;
      }

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
        textures.push_back(TexParam{tname, tval, tf.find("texture2d_array") != std::string::npos,
                                    tf.find("texture3d") != std::string::npos,
                                    tf.find("texturecube") != std::string::npos,
                                    tf.find("texture1d") != std::string::npos &&
                                        tf.find("texture1d_array") == std::string::npos});
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
        if (render && (attr == "barycentric_coord" ||
                       attr == "barycentric_coord_no_perspective")) {
          if (file_points && vstage.ok) {
            unsupported = true;
            why = "a point primitive has no barycentric coordinates (" + rawp + ")";
            break;
          }
          if (!vertex_used) {
            unsupported = true;
            why = "[[barycentric_coord]] needs a `[[stage_in]]` parameter before it, which is "
                  "what declares the triangle (" + rawp + ")";
            break;
          }
          if (type_field != "float3") {
            unsupported = true;
            why = "[[barycentric_coord]] is a float3, not " + type_field + " (" + rawp + ")";
            break;
          }
          const bool np = attr.find("no_perspective") != std::string::npos;
          locals += "  const float3 " + name_field + " = float3(" +
                    (np ? "weft_l0, weft_l1, weft_l2"
                        : "weft_q0 * weft_iq, weft_q1 * weft_iq, weft_q2 * weft_iq") +
                    ");\n";
          continue;
        }
        if (render && attr == "point_coord") {
          points_run = true;
          if (vstage.ok) {
            if (!ensure_points_prelude()) break;
            locals += "  " + type_field + " " + name_field + " = " + type_field +
                      "(weft_pcx, weft_pcy);\n";
            only_gid = false;
            // A point pass with a vertex stage draws the points that stage writes.
            if (params.find("[[stage_in]]") == std::string::npos) {
              lowers.push_back(FragLower{name, points_gpu, points_params, std::string()});
              stage_in_fns++;
            }
            continue;
          }
          locals += "  " + type_field + " " + name_field + " = " + type_field + "(0.5);\n";
          continue;
        }
        if (render && attr.rfind("color", 0) == 0 && attr.find('(') != std::string::npos) {
          locals += "  " + type_field + " " + name_field + " = " + type_field + "(0.0);\n";
          continue;
        }
        // Tile memory.
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
        if (attr.rfind("buffer", 0) == 0) {
          size_t a = attr.find('('), b = attr.find(')');
          std::string idx = (a != std::string::npos && b > a) ? attr.substr(a + 1, b - a - 1) : "";
          if (idx.empty()) { unsupported = true; why = "buffer() without an index"; break; }
          type_field.erase(std::remove(type_field.begin(), type_field.end(), '&'),
                           type_field.end());
          const std::string cq = is_const ? "const " : "";
          std::string line;
          if (stars > 0) {
            line = "  " + cq + type_field + "* " + name_field + " = (" + cq + type_field +
                   "*)weft::buffer(ctx, " + idx + ");\n";
          } else {
            name_field.erase(std::remove(name_field.begin(), name_field.end(), '&'),
                             name_field.end());
            line = "  " + cq + type_field + "& " + name_field + " = *(" + cq + type_field +
                   "*)weft::buffer(ctx, " + idx + ");\n";
          }
          const int bidx = std::atoi(trim(idx).c_str());
          locals += line;
          buf_locals += line;
          buf_types_at.emplace_back(bidx, type_field);
          if (nargs <= bidx) nargs = bidx + 1;
          continue;
        }
        if (render && attr.rfind("sampler", 0) == 0) {
          const size_t a = attr.find('('), b = attr.find(')');
          const int slot = (a != std::string::npos && b > a)
                               ? std::atoi(trim(attr.substr(a + 1, b - a - 1)).c_str())
                               : 0;
          std::string nm = name_field;
          nm.erase(std::remove(nm.begin(), nm.end(), '&'), nm.end());
          samplers_for_body[nm] = {0, 0};
          frag_sampler_slots.push_back(slot);
          continue;
        }
        // `[[stage_in]]` is what the vertex stage wrote for this pixel.
        if (render && attr == "stage_in") {
          std::string nm = name_field;
          nm.erase(std::remove(nm.begin(), nm.end(), '&'), nm.end());
          const size_t def = src.find("struct " + type_field);
          const size_t open = def == std::string::npos ? std::string::npos : src.find('{', def);
          size_t close = std::string::npos, depth = 0;
          if (open != std::string::npos) {
            for (size_t i = open; i < src.size(); i++) {
              if (src[i] == '{') depth++;
              else if (src[i] == '}' && --depth == 0) { close = i; break; }
            }
          }
          if (close == std::string::npos) {
            unsupported = true;
            why = "stage_in struct '" + type_field + "' is not declared in this file (" + rawp +
                  ")";
            break;
          }
          struct StageField {
            std::string type, name, attr;
            bool position() const { return attr.find("position") != std::string::npos; }
            bool flat() const { return attr.find("flat") != std::string::npos; }
          };
          std::vector<StageField> fields;
          {
            const std::string text = src.substr(open + 1, close - open - 1);
            size_t at = 0;
            while (true) {
              const size_t semi = text.find(';', at);
              if (semi == std::string::npos) break;
              std::string f = text.substr(at, semi - at);
              at = semi + 1;
              StageField fd;
              const size_t lb = f.find("[["), rb = f.find("]]");
              if (lb != std::string::npos && rb > lb) {
                fd.attr = trim(f.substr(lb + 2, rb - lb - 2));
                f = f.substr(0, lb) + f.substr(rb + 2);
              }
              std::string word;
              for (char c : f) {
                if (std::isspace((unsigned char)c)) {
                  if (!word.empty()) {
                    if (fd.type.empty()) {
                      fd.type = word;
                    } else if (word[0] == '[' && !fd.name.empty()) {
                      fd.name += word;  // an array bound written after the name
                    } else {
                      fd.name = word;
                    }
                    word.clear();
                  }
                } else {
                  word.push_back(c);
                }
              }
              if (!word.empty()) {
                if (fd.type.empty()) {
                  fd.type = word;
                } else if (word[0] == '[' && !fd.name.empty()) {
                  fd.name += word;
                } else {
                  fd.name = word;
                }
              }
              if (!fd.name.empty()) fields.push_back(fd);
            }
          }
          const std::string w = std::to_string(render->width);
          const std::string h = std::to_string(render->height);
          const std::string px = "((float)(ctx.gid % " + w + "u) + 0.5f)";
          const std::string py = "((float)(ctx.gid / " + w + "u) + 0.5f)";
          std::string pos_field;
          for (const StageField& fd : fields)
            if (fd.position()) pos_field = fd.name;
          stage_in_fns++;
          bool bad = false;
          std::string decl;

          const std::vector<StructField> vfields =
              vstage.ok ? read_struct_fields(src, vstage.ret) : std::vector<StructField>();
          bool vmatch = vstage.ok && !pos_field.empty();
          for (const StageField& fd : fields) {
            if (!vmatch) break;
            const StructField* hit = nullptr;
            for (const StructField& vf : vfields)
              if (vf.name == fd.name) hit = &vf;
            if (!hit) {
              unsupported = true;
              why = "the fragment reads '" + fd.name + "', which the vertex stage '" +
                    vstage.name + "' does not write (" + rawp + ")";
              bad = true;
              vmatch = false;
            } else if (hit->type != fd.type) {
              unsupported = true;
              why = "the fragment reads '" + fd.name + "' as " + fd.type +
                    ", and the vertex stage writes " + hit->type + " (" + rawp + ")";
              bad = true;
              vmatch = false;
            }
          }
          if (bad) break;
          if (vmatch) {
            for (const auto& p : vstage.params)
              if (p.buffer >= 0 && nargs <= p.buffer) nargs = p.buffer + 1;
            if (file_points) {
              if (!ensure_points_prelude()) { bad = true; break; }
              decl += "  " + type_field + " " + nm + ";\n";
              for (const StageField& fd : fields) {
                if (fd.name == pos_field) {
                  decl += "  " + nm + "." + fd.name + " = float4(" + px + ", " + py +
                          ", 0.0f, 1.0f);\n";
                  continue;
                }
                decl += "  if (weft_which == 0) " + nm + "." + fd.name + " = weft_v0." +
                        fd.name + ";\n";
                decl += "  else if (weft_which == 1) " + nm + "." + fd.name + " = weft_v1." +
                        fd.name + ";\n";
                decl += "  else " + nm + "." + fd.name + " = weft_v2." + fd.name + ";\n";
              }
              if (!points_gpu.empty()) {
                std::string decl_gpu = replace_all(replace_all(decl, px, "@PX@"), py, "@PY@");
                lowers.push_back(
                    FragLower{name, points_gpu + decl_gpu, points_params, std::string()});
              }
              locals += decl;
              only_gid = false;
              vertex_used = true;
              continue;
            }
            if (vstage_pure) {
              decl += "  static thread_local " + vstage.ret +
                      " weft_hv0, weft_hv1, weft_hv2;\n"
                      "  static thread_local uint64_t weft_hgen = ~0ull;\n"
                      "  if (weft_hgen != weft::pass_generation()) {\n"
                      "    weft_hgen = weft::pass_generation();\n";
            }
            for (int i = 0; i < 3 && !bad; i++) {
              std::string args;
              for (const auto& p : vstage.params) {
                if (!args.empty()) args += ", ";
                if (p.vertex_id) {
                  args += std::to_string(i) + "u";
                } else if (p.zero_index) {
                  args += "0u";
                } else if (p.texture) {
                  args += "weft::texture(" + std::to_string(p.buffer) + ")";
                } else if (p.buffer >= 0) {
                  args += p.pointer
                              ? "(" + p.type + "*)weft::buffer(ctx, " + std::to_string(p.buffer) + ")"
                              : "*(const " + p.type + "*)weft::buffer(ctx, " +
                                    std::to_string(p.buffer) + ")";
                } else {
                  unsupported = true;
                  why = "vertex stage parameter '" + p.name +
                        "' is neither a vertex id nor a buffer (" + rawp + ")";
                  bad = true;
                  break;
                }
              }
              if (!bad)
                decl += vstage_pure
                            ? ("    weft_hv" + std::to_string(i) + " = " + vstage.name + "(" +
                               args + ");\n")
                            : ("  " + vstage.ret + " weft_v" + std::to_string(i) + " = " +
                               vstage.name + "(" + args + ");\n");
            }
            if (vstage_pure) {
              decl += "  }\n"
                      "  const " + vstage.ret + "& weft_v0 = weft_hv0;\n"
                      "  const " + vstage.ret + "& weft_v1 = weft_hv1;\n"
                      "  const " + vstage.ret + "& weft_v2 = weft_hv2;\n";
            }
            if (bad) break;
            decl += "  const float weft_px = " + px + ", weft_py = " + py + ";\n";
            decl += "  const float4 weft_c0 = weft_v0." + pos_field + ", weft_c1 = weft_v1." +
                    pos_field + ", weft_c2 = weft_v2." + pos_field + ";\n";
            for (int i = 0; i < 3; i++) {
              const std::string c = "weft_c" + std::to_string(i);
              const std::string sx =
                  "std::round(((" + c + ".x / " + c + ".w * 0.5f + 0.5f) * " + w +
                  ".0f) * 256.0f) * 0.00390625f";
              const std::string sy =
                  "std::round(((0.5f - " + c + ".y / " + c + ".w * 0.5f) * " + h +
                  ".0f) * 256.0f) * 0.00390625f";
              decl += "  const float2 weft_s" + std::to_string(i) + " = float2(" + sx + ", " +
                      sy + ");\n";
            }
            decl +=
                "  const float weft_d = (weft_s1.x - weft_s0.x) * (weft_s2.y - weft_s0.y) - "
                "(weft_s2.x - weft_s0.x) * (weft_s1.y - weft_s0.y);\n"
                "  const float weft_l0 = ((weft_s1.x - weft_px) * (weft_s2.y - weft_py) - "
                "(weft_s2.x - weft_px) * (weft_s1.y - weft_py)) / weft_d;\n"
                "  const float weft_l1 = ((weft_s2.x - weft_px) * (weft_s0.y - weft_py) - "
                "(weft_s0.x - weft_px) * (weft_s2.y - weft_py)) / weft_d;\n"
                "  const float weft_l2 = 1.0f - weft_l0 - weft_l1;\n";
            decl += "  if (!(weft_l0 >= 0.0f && weft_l1 >= 0.0f && weft_l2 >= 0.0f)) return;\n";
            decl += "  const float weft_q0 = weft_l0 / weft_c0.w;\n"
                    "  const float weft_q1 = weft_l1 / weft_c1.w;\n"
                    "  const float weft_q2 = weft_l2 / weft_c2.w;\n"
                    "  const float weft_iq = 1.0f / (weft_q0 + weft_q1 + weft_q2);\n";
            for (size_t ci = 0; ci < clip_fields.size(); ci++) {
              const std::string& cn = clip_fields[ci];
              const std::string t = std::to_string(ci);
              decl += "  const float weft_clip" + t + " = (weft_v0." + cn +
                      " * weft_q0 + weft_v1." + cn + " * weft_q1 + weft_v2." + cn +
                      " * weft_q2) * weft_iq;\n";
              decl += "  if (weft_clip" + t + " < 0.0f) return;\n";
            }
            decl += "  " + type_field + " " + nm + ";\n";
            std::vector<StageField> expanded;
            for (const StageField& fd : fields) {
              const size_t lbk = fd.name.find('[');
              if (lbk == std::string::npos) {
                expanded.push_back(fd);
                continue;
              }
              const size_t rb = fd.name.find(']', lbk);
              const std::string count =
                  rb == std::string::npos ? std::string() : fd.name.substr(lbk + 1, rb - lbk - 1);
              if (fd.type != "float" || count.empty() || count.size() > 2 ||
                  count.find_first_not_of("0123456789") != std::string::npos) {
                unsupported = true;
                why = "stage_in field '" + fd.name + "' is an array weft cannot blend (" + rawp +
                      ")";
                bad = true;
                break;
              }
              const std::string base = fd.name.substr(0, lbk);
              for (int e = 0; e < std::stoi(count); e++) {
                StageField el = fd;
                el.name = base + "[" + std::to_string(e) + "]";
                expanded.push_back(el);
              }
            }
            for (const StageField& fd : expanded) {
              if (fd.name == pos_field) {
                decl += "  " + nm + "." + fd.name + " = float4(" + px + ", " + py +
                        ", 0.0f, 1.0f);\n";
              } else if (fd.flat()) {
                decl += "  " + nm + "." + fd.name + " = weft_v0." + fd.name + ";\n";
              } else if (fd.type == "float" || fd.type == "float2" || fd.type == "float3" ||
                         fd.type == "float4" || fd.type == "half" || fd.type == "half2" ||
                         fd.type == "half3" || fd.type == "half4") {
                const bool np = fd.attr.find("no_perspective") != std::string::npos;
                const std::string q0 = np ? "weft_l0" : "weft_q0";
                const std::string q1 = np ? "weft_l1" : "weft_q1";
                const std::string q2 = np ? "weft_l2" : "weft_q2";
                const std::string scale = np ? std::string() : " * weft_iq";
                auto blend = [&](const std::string& comp) {
                  return "((float)weft_v0." + fd.name + comp + " * " + q0 + " + (float)weft_v1." +
                         fd.name + comp + " * " + q1 + " + (float)weft_v2." + fd.name + comp +
                         " * " + q2 + ")" + scale;
                };
                if (fd.type.rfind("half", 0) != 0) {
                  decl += "  " + nm + "." + fd.name + " = (weft_v0." + fd.name + " * " + q0 +
                          " + weft_v1." + fd.name + " * " + q1 + " + weft_v2." + fd.name + " * " +
                          q2 + ")" + scale + ";\n";
                } else if (fd.type == "half") {
                  decl += "  " + nm + "." + fd.name + " = half(" + blend("") + ");\n";
                } else {
                  static const char* const comps[] = {"x", "y", "z", "w"};
                  const int lanes = fd.type.back() - '0';
                  decl += "  " + nm + "." + fd.name + " = " + fd.type + "(";
                  for (int c = 0; c < lanes; c++) {
                    if (c) decl += ", ";
                    decl += "half(" + blend(std::string(".") + comps[c]) + ")";
                  }
                  decl += ");\n";
                }
              } else {
                unsupported = true;
                why = "stage_in field '" + fd.name + "' has type " + fd.type +
                      ", which the CPU interpolator does not blend (" + rawp + ")";
                bad = true;
                break;
              }
            }
            if (bad) break;
            const size_t interp_at = decl.find("  const float weft_px");
            if (interp_at != std::string::npos) {
              std::string decl_gpu = decl.substr(interp_at);
              decl_gpu = replace_all(decl_gpu, px, "@PX@");
              decl_gpu = replace_all(decl_gpu, py, "@PY@");
              decl_gpu = replace_all(decl_gpu, " return;\n", " @RET@\n");
              std::string add, calls;
              std::vector<std::string> parts;
              for (const auto& p : vstage.params) {
                if (p.texture) {
                  parts.push_back(p.orig_type + " weft_stage_tex_" + std::to_string(p.buffer) +
                                  " [[texture(" + std::to_string(p.buffer) + ")]]");
                } else if (p.buffer >= 0) {
                  parts.push_back((p.pointer ? "device " + p.type + "* "
                                             : "constant " + p.type + "& ") +
                                  "weft_stage_buf_" + std::to_string(p.buffer) + " [[buffer(" +
                                  std::to_string(p.buffer) + ")]]");
                }
              }
              for (const auto& p : vstage.sampler_params) {
                parts.push_back("sampler weft_stage_smp_" + std::to_string(p.buffer) +
                                " [[sampler(" + std::to_string(p.buffer) + ")]]");
                frag_sampler_slots.push_back(p.buffer);
              }
              for (const std::string& part : parts) {
                if (!add.empty()) add += ", ";
                add += part;
              }
              for (int i = 0; i < 3; i++) {
                std::string args;
                for (const auto& p : vstage.params) {
                  if (!args.empty()) args += ", ";
                  args += p.vertex_id  ? (std::to_string(i) + "u")
                        : p.zero_index ? std::string("0u")
                        : p.texture    ? ("weft_stage_tex_" + std::to_string(p.buffer))
                                       : ("weft_stage_buf_" + std::to_string(p.buffer));
                }
                for (const auto& p : vstage.sampler_params)
                  args += (args.empty() ? "" : ", ") + std::string("weft_stage_smp_") +
                          std::to_string(p.buffer);
                calls += "  const " + vstage.ret + " weft_v" + std::to_string(i) +
                         " = weft_plain_vstage(" + args + ");\n";
              }
              if (!file_points) lowers.push_back(FragLower{name, decl_gpu, add, calls});
            }
            locals += decl;
            only_gid = false;
            vertex_used = true;
            continue;
          }

          if (vstage.ok) {
            unsupported = true;
            why = "the `[[stage_in]]` struct '" + type_field +
                  "' has no `[[position]]` field, or the vertex stage does not write it (" +
                  rawp + ")";
            break;
          }
          decl = "  " + type_field + " " + nm + ";\n";
          for (const StageField& fd : fields) {
            if (fd.name == pos_field) {
              decl += "  " + nm + "." + fd.name + " = float4(" + px + ", " + py + ", 0.0f, 1.0f);\n";
            } else if (fd.type == "float2") {
              decl += "  " + nm + "." + fd.name + " = float2(" + px + " / " + w + ".0f, 1.0f - " +
                      py + " / " + h + ".0f);\n";
            } else {
              unsupported = true;
              why = "stage_in field '" + fd.name + "' has type " + fd.type +
                    ", which a full-screen pass has no value for (" + rawp + ")";
              bad = true;
              break;
            }
          }
          if (bad) break;
          locals += decl;
          only_gid = false;
          continue;
        }
        // `[[position]]` in a fragment is the pixel centre.
        if (render && attr == "position") {
          locals += "  float4 " + name_field + " = float4((float)(ctx.gid % " +
                     std::to_string(render->width) + "u) + 0.5f, (float)(ctx.gid / " +
                     std::to_string(render->width) + "u) + 0.5f, 0.0f, 1.0f);\n";
          only_gid = false;
          continue;
        }
        // The inputs the pipeline supplies itself.
        if (render && attr == "front_facing") {
          std::string line = "  bool " + name_field + " = false;\n";
          locals += line;
          buf_locals += line;
          continue;
        }
        if (render && (attr == "primitive_id" || attr == "sample_id")) {
          std::string line = "  " + type_field + " " + name_field + " = 0u;\n";
          locals += line;
          buf_locals += line;
          continue;
        }
        // One sample per pixel, so the only sample a mask can name is the first.
        if (render && attr == "sample_mask") {
          std::string line = "  " + type_field + " " + name_field + " = 1u;\n";
          locals += line;
          buf_locals += line;
          continue;
        }
        if (attr == "thread_position_in_grid") {
          locals += "  " + type_field + " " + name_field + " = (" + type_field + ")ctx.gid;\n";
          gid_params.emplace_back(type_field, name_field);
          continue;
        }
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

    bool shared_found = false, barrier_found = false;
    std::string berr;
    int tile_idx = -1, colour_idx = -1, colour_targets = 1, depth_idx = -1;
    // The qualifier a `[[depth(.)]]` output names.
    std::string depth_qual = "any";
    const int user_nbufs = nargs;
    if (render) {
      if (!is_tile && !file_points && vstage.ok && !clip_fields.empty() &&
          params.find("[[stage_in]]") == std::string::npos) {
        const std::vector<std::string>& cuts = clip_fields;
        if (!cuts.empty()) {
          if (vpos_name.empty()) {
            r.error = "the vertex stage's struct has no `[[position]]` field, so there is no "
                      "primitive for its [[clip_distance]] to cut";
            return r;
          }
          const std::string w = std::to_string(render->width);
          const std::string h = std::to_string(render->height);
          const std::string cpx = "((float)(ctx.gid % " + w + "u) + 0.5f)";
          const std::string cpy = "((float)(ctx.gid / " + w + "u) + 0.5f)";
          for (const auto& p : vstage.params)
            if (p.buffer >= 0 && nargs <= p.buffer) nargs = p.buffer + 1;
          std::string add, calls;
          std::vector<std::string> parts;
          for (const auto& p : vstage.params) {
            if (p.texture) {
              parts.push_back(p.orig_type + " weft_stage_tex_" + std::to_string(p.buffer) +
                              " [[texture(" + std::to_string(p.buffer) + ")]]");
            } else if (p.buffer >= 0) {
              parts.push_back((p.pointer ? "device " + p.type + "* "
                                         : "constant " + p.type + "& ") +
                              "weft_stage_buf_" + std::to_string(p.buffer) + " [[buffer(" +
                              std::to_string(p.buffer) + ")]]");
            }
          }
          for (const auto& p : vstage.sampler_params) {
            parts.push_back("sampler weft_stage_smp_" + std::to_string(p.buffer) +
                            " [[sampler(" + std::to_string(p.buffer) + ")]]");
            frag_sampler_slots.push_back(p.buffer);
          }
          for (const std::string& part : parts) {
            if (!add.empty()) add += ", ";
            add += part;
          }
          std::string cd;
          if (vstage_pure) {
            cd += "  static thread_local " + vstage.ret + " weft_hv0, weft_hv1, weft_hv2;\n"
                  "  static thread_local uint64_t weft_hgen = ~0ull;\n"
                  "  if (weft_hgen != weft::pass_generation()) {\n"
                  "    weft_hgen = weft::pass_generation();\n";
          }
          for (int i = 0; i < 3; i++) {
            std::string args, gargs;
            for (const auto& p : vstage.params) {
              if (!args.empty()) { args += ", "; gargs += ", "; }
              if (p.vertex_id) {
                args += std::to_string(i) + "u";
                gargs += std::to_string(i) + "u";
              } else if (p.zero_index) {
                args += "0u";
                gargs += "0u";
              } else if (p.texture) {
                args += "weft::texture(" + std::to_string(p.buffer) + ")";
                gargs += "weft_stage_tex_" + std::to_string(p.buffer);
              } else if (p.buffer >= 0) {
                args += p.pointer
                            ? ("(" + p.type + "*)weft::buffer(ctx, " + std::to_string(p.buffer) + ")")
                            : ("*(const " + p.type + "*)weft::buffer(ctx, " +
                               std::to_string(p.buffer) + ")");
                gargs += "weft_stage_buf_" + std::to_string(p.buffer);
              }
            }
            for (const auto& p : vstage.sampler_params)
              gargs += (gargs.empty() ? "" : ", ") + std::string("weft_stage_smp_") +
                       std::to_string(p.buffer);
            cd += vstage_pure
                      ? ("    weft_hv" + std::to_string(i) + " = " + vstage.name + "(" + args +
                         ");\n")
                      : ("  const " + vstage.ret + " weft_v" + std::to_string(i) + " = " +
                         vstage.name + "(" + args + ");\n");
            calls += "  const " + vstage.ret + " weft_v" + std::to_string(i) +
                     " = weft_plain_vstage(" + gargs + ");\n";
          }
          if (vstage_pure) {
            cd += "  }\n"
                  "  const " + vstage.ret + "& weft_v0 = weft_hv0;\n"
                  "  const " + vstage.ret + "& weft_v1 = weft_hv1;\n"
                  "  const " + vstage.ret + "& weft_v2 = weft_hv2;\n";
          }
          cd += "  const float weft_px = " + cpx + ", weft_py = " + cpy + ";\n";
          cd += "  const float4 weft_c0 = weft_v0." + vpos_name + ", weft_c1 = weft_v1." +
                vpos_name + ", weft_c2 = weft_v2." + vpos_name + ";\n";
          for (int i = 0; i < 3; i++) {
            const std::string c = "weft_c" + std::to_string(i);
            const std::string sx = "std::round(((" + c + ".x / " + c + ".w * 0.5f + 0.5f) * " + w +
                                   ".0f) * 256.0f) * 0.00390625f";
            const std::string sy = "std::round(((0.5f - " + c + ".y / " + c + ".w * 0.5f) * " + h +
                                   ".0f) * 256.0f) * 0.00390625f";
            cd += "  const float2 weft_s" + std::to_string(i) + " = float2(" + sx + ", " + sy +
                  ");\n";
          }
          cd += "  const float weft_d = (weft_s1.x - weft_s0.x) * (weft_s2.y - weft_s0.y) - "
                "(weft_s2.x - weft_s0.x) * (weft_s1.y - weft_s0.y);\n"
                "  const float weft_l0 = ((weft_s1.x - weft_px) * (weft_s2.y - weft_py) - "
                "(weft_s2.x - weft_px) * (weft_s1.y - weft_py)) / weft_d;\n"
                "  const float weft_l1 = ((weft_s2.x - weft_px) * (weft_s0.y - weft_py) - "
                "(weft_s0.x - weft_px) * (weft_s2.y - weft_py)) / weft_d;\n"
                "  const float weft_l2 = 1.0f - weft_l0 - weft_l1;\n";
          cd += "  if (!(weft_l0 >= 0.0f && weft_l1 >= 0.0f && weft_l2 >= 0.0f)) return;\n";
          cd += "  const float weft_q0 = weft_l0 / weft_c0.w;\n"
                "  const float weft_q1 = weft_l1 / weft_c1.w;\n"
                "  const float weft_q2 = weft_l2 / weft_c2.w;\n"
                "  const float weft_iq = 1.0f / (weft_q0 + weft_q1 + weft_q2);\n";
          for (size_t ci = 0; ci < cuts.size(); ci++) {
            const std::string t = std::to_string(ci);
            cd += "  const float weft_clip" + t + " = (weft_v0." + cuts[ci] + " * weft_q0 + weft_v1." +
                  cuts[ci] + " * weft_q1 + weft_v2." + cuts[ci] + " * weft_q2) * weft_iq;\n";
            cd += "  if (weft_clip" + t + " < 0.0f) return;\n";
          }
          const size_t at_px = cd.find("  const float weft_px");
          std::string cd_gpu = cd.substr(at_px);
          cd_gpu = replace_all(cd_gpu, cpx, "@PX@");
          cd_gpu = replace_all(cd_gpu, cpy, "@PY@");
          cd_gpu = replace_all(cd_gpu, " return;\n", " @RET@\n");
          lowers.push_back(FragLower{name, cd_gpu, add, calls});
          locals += cd;
          only_gid = false;
          vertex_used = true;
          stage_in_fns++;
        }
      }
      if (!ib_name.empty()) {
        const std::string idx = std::to_string(nargs);
        tile_idx = nargs;
        nargs += 1;
        buf_types_at.emplace_back(tile_idx, ib_type);
        locals += "  " + ib_type + "* " + ib_name + " = (" + ib_type + "*)weft::buffer(ctx, " +
                  idx + ");\n";
        const std::string wpix = std::to_string(render->width) + "u";
        const std::string wtiles = std::to_string((render->width + 31) / 32) + "u";
        const std::string at_own =
            "(" + ib_name + " + weft::tile_slot(ctx, ctx.gid, " + wpix + ", " + wtiles + "))->";
        const std::string own = ib_name + "[weft::tile_slot(ctx, (@), " + wpix + ", " + wtiles + ")].";
        body = rewrite_member_access(body, ib_name, at_own, berr, own);
        if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }
      }
      if (!is_tile) {
        FragLayout lay;
        if (!parse_frag_layout(src, ret_type, lay, berr)) {
          r.error = "fragment '" + name + "': " + berr;
          return r;
        }
        // Metal wants each colour output as wide as its attachment.
        bool narrow = lay.width != 0 && lay.width < 4;
        for (const FragTarget& ft : lay.colours)
          if (ft.width < 4) narrow = true;
        if (narrow) gpu_pads.push_back(GpuPad{name, ret_type, lay});
        colour_idx = nargs;
        if (lay.width != 0) {
          locals += "  float* weft_colour = (float*)weft::buffer(ctx, " + std::to_string(nargs) +
                    ");\n";
          nargs += 1;
        } else {
          const int n = lay.colours.back().index + 1;
          for (int i = 0; i < n; i++)
            locals += "  float* weft_colour_" + std::to_string(i) +
                      " = (float*)weft::buffer(ctx, " + std::to_string(nargs + i) + ");\n";
          nargs += n;
          colour_targets = n;
        }
        if (lay.depth) {
          depth_idx = nargs;
          depth_qual = lay.depth_qual;
          locals += "  float* weft_depth = (float*)weft::buffer(ctx, " + std::to_string(nargs) +
                    ");\n";
          nargs += 1;
        }
        body = rewrite_frag_return(body, ret_type, lay, berr);
        if (!berr.empty()) { r.error = "fragment '" + name + "': " + berr; return r; }
      }
    }
    for (const HelperFn& h : carried.helpers) {
      bool hit = false;
      body = expand_helper_calls(body, h, hit, berr);
      if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }
    }
    body = rewrite_shared_decls(body, shared_found, berr);
    if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }
    body = rewrite_barriers(body, barrier_found, berr);
    if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }

    std::map<std::string, std::pair<int, int>> samplers = samplers_for_body;
    if (!textures.empty() || !samplers.empty() || body.find("sampler") != std::string::npos) {
      body = strip_samplers(body, samplers, berr);
      if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }
    }
    if (!textures.empty()) {
      body = rewrite_textures(body, textures, samplers, berr);
      if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }
    }

    {
      const std::vector<std::string> vecs = vector_decl_names(src);
      std::string fixed;
      for (size_t k = 0; k < body.size(); k++) {
        if (body[k] != '.' || k + 1 >= body.size()) {
          fixed += body[k];
          continue;
        }
        const char c = body[k + 1];
        const char nx = (k + 2 < body.size()) ? body[k + 2] : '\0';
        if (std::string("rgba").find(c) == std::string::npos ||
            std::string("rgba").find(nx) != std::string::npos || is_ident(nx)) {
          fixed += body[k];
          continue;
        }
        size_t s = k;
        while (s > 0 && is_ident(body[s - 1])) s--;
        const std::string recv = body.substr(s, k - s);
        bool vec = false;
        for (const std::string& v : vecs)
          if (v == recv) vec = true;
        if (!vec) {
          fixed += body[k];
          continue;
        }
        fixed += '.';
        fixed += c == 'r' ? 'x' : c == 'g' ? 'y' : c == 'b' ? 'z' : 'w';
        k++;
      }
      body = fixed;
    }
    if (body.find("discard_fragment") != std::string::npos) {
      body = replace_all(body, "discard_fragment()", "return;");
      only_gid = false;
    }
    body = replace_all(body, "metal::", "");
    body = replace_all(body, "fast::", "");
    body = replace_all(body, "precise::", "");
    // The ballot's return type.
    body = replace_all(body, "simd_vote::vote_t", "uint32_t");
    body = replace_all(body, "quad_vote::vote_t", "uint32_t");
    body = replace_all(body, "simd_vote", "uint32_t");
    body = replace_all(body, "quad_vote", "uint32_t");

    body = rewrite_as_type(body, berr);
    if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }

    body = rewrite_swizzles(body, berr);
    if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }

    bool shfl_found = false, lane_expr = false;
    body = rewrite_group_ops(body, shfl_found, lane_expr, berr);
    if (!berr.empty()) { r.error = "kernel '" + name + "': " + berr; return r; }
    for (const char* pre : {"simd_", "quad_"}) {
      const size_t sp = body.find(pre);
      if (sp == std::string::npos) continue;
      size_t e = sp + 5;
      while (e < body.size() && is_ident(body[e])) e++;
      r.error = "kernel '" + name + "': " + body.substr(sp, e - sp) + " is not supported";
      return r;
    }

    const bool is_sync = shared_found || barrier_found || shfl_found;
    bool is_step = false;
    if (is_sync) {
      std::string fields, types, code, cases, why;
      const bool step_ok = lower_step(body, name, locals, fields, types, code, cases, why);
      if (!step_ok && std::getenv("WEFT_STEP_WHY"))
        std::fprintf(stderr, "weft: %s stays on its coroutine: %s\n", name.c_str(), why.c_str());
      if (step_ok) {
        is_step = true;
        body_out += types;
        body_out += "struct weft_st_" + name + " {\n  uint32_t pc;\n" + fields + "};\n";
        std::string inner;
        inner += "    weft_st_" + name + "* weft_s = (weft_st_" + name + "*)st;\n";
        inner += "    (void)bufs; (void)sh;\n";
        inner += "    ctx.slot = 0;\n";
        inner += locals;
        inner += "    if (weft_s->pc != 0u) {\n      switch (weft_s->pc) {\n" + cases +
                 "        default: goto weft_done;\n      }\n    }\n";
        inner += code;
        inner += "    weft_done:;\n    weft_s->pc = weft::kThreadDone;\n";
        body_out += "static uint32_t weft_g_" + name +
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
        body_out += "static void weft_k_" + name +
                    "(weft::Ctx& ctx, void* stp, void* const* bufs, unsigned char* sh) {\n"
                    "  weft_g_" + name + "(&ctx, (unsigned char*)stp, 0, 1, ctx.pass, bufs, sh);\n"
                    "}\n\n";
      } else {
        body_out += "// no state machine for '" + name + "'" +
                    (why.empty() ? std::string() : (": " + why)) + "\n";
        std::string cbody =
            replace_all(body, "weft::barrier(ctx);", "co_await weft::barrier_async();");
        cbody = replace_all(cbody, "return;", "co_return;");
        body_out += "static weft::Task weft_k_" + name + "(weft::Ctx& ctx) {\n" + locals + "\n" +
                    cbody + "\n  co_return;\n}\n\n";
      }
    } else {
      body_out += "static void weft_k_" + name + "(weft::Ctx& ctx) {\n" + locals + "\n" + body +
                  "\n}\n\n";
    }

    const bool has_jump = contains_word(body, "return") || contains_word(body, "break") ||
                          contains_word(body, "continue");
    const bool can_flat = !(shared_found || barrier_found || shfl_found) && only_gid &&
                          !has_jump && !lane_expr && ib_name.empty();
    if (can_flat) {
      std::string gid_decls;
      for (const auto& g : gid_params)
        gid_decls += "    " + g.first + " " + g.second + " = (" + g.first + ")weft_i;\n";
      body_out += "static void weft_k_" + name + "_loop(weft::Ctx& ctx, uint32_t begin, uint32_t end) {\n" +
                  buf_locals + "  for (uint32_t weft_i = begin; weft_i < end; weft_i++) {\n" + gid_decls +
                  "    " + body + "\n  }\n}\n\n";
    }

    TranslatedKernel tk;
    tk.name = name;
    tk.nargs = nargs;
    tk.nbufs = user_nbufs - extra_bufs;
    // One width per binding, in binding order.
    std::vector<std::string> buf_types((size_t)nargs, std::string("0"));
    for (const auto& bt : buf_types_at)
      if (bt.first >= 0 && (size_t)bt.first < buf_types.size())
        buf_types[(size_t)bt.first] = bt.second;
    tk.buf_type = buf_types;
    tk.tile_idx = tile_idx;
    tk.colour_idx = colour_idx;
    tk.colour_targets = colour_targets;
    tk.depth_idx = depth_idx;
    tk.depth_qual = depth_qual;
    tk.tile = is_tile;
    tk.mesh = is_mesh;
    tk.mesh_nv = mesh_nv;
    tk.mesh_np = mesh_np;
    tk.task = is_task;
    tk.payload_idx = payload_idx;
    tk.props_idx = props_idx;
    tk.ntex = (int)textures.size();
    // A vertex stage that samples a texture is bound the same table.
    if (vertex_used && vstage.ntex > tk.ntex) tk.ntex = vstage.ntex;
    for (const TexParam& tp : textures)
      tk.tex_shape.push_back(tp.cube ? "cube"
                                     : tp.is3d ? "3d"
                                               : tp.array ? "array"
                                                          : tp.is1d ? "1d" : "2d");
    tk.sync = is_sync;
    tk.flat = can_flat;
    tk.step = is_step;
    r.kernels.push_back(tk);
    kcount++;

    pos = close_brace + 1;
  }
  if (stage_in_fns > 0 && !vstage.unsupplied.empty()) {
    r.error = "the file's vertex stage names [[" + vstage.unsupplied +
              "]], which weft cannot supply";
    return r;
  }
  if (stage_in_fns == 0 && vstage.ok && !file_points) {
    for (const StructField& vf : read_struct_fields(src, vstage.ret)) {
      if (vf.attr.find("clip_distance") == std::string::npos) continue;
      r.error = "the vertex stage writes a [[clip_distance]] and the pass draws no triangle "
                "for the cut to run on";
      return r;
    }
  }
  if (vertex_used) {
    r.vertex_name = vstage.name;
    r.vertex_sampler_slots = vstage.samp_slots;
  }
  r.points = points_run;
  r.fragment_sampler_slots = frag_sampler_slots;
  r.function_constants = fn_consts;
  std::string pad_err;
  if (!apply_gpu_pads(src, gpu_pads, pad_err)) { r.error = pad_err; return r; }
  std::string gpu_text = src;
  gpu_text = replace_all(gpu_text, "[[early_fragment_tests]]",
                         "__attribute__((early_fragment_tests))");
  if (vstage.ok && !vstage.new_all.empty())
    gpu_text = replace_all(gpu_text, vstage.new_all, vstage.orig_all);
  if (!lowers.empty() && stage_in_fns == (int)lowers.size() && vstage.ok) {
    std::string text = gpu_text;
    std::string params_gpu;
    for (const auto& p : vstage.params) {
      if (!params_gpu.empty()) params_gpu += ", ";
      if (p.by_ref) {
        params_gpu += (p.is_const ? "constant " : "device ") + p.orig_type + "& " + p.name;
      } else if (p.pointer) {
        params_gpu += (p.is_const ? "const device " : "device ") + p.orig_type + "* " + p.name;
      } else {
        params_gpu += p.orig_type + " " + p.name;
      }
    }
    for (const auto& p : vstage.sampler_params) {
      if (!params_gpu.empty()) params_gpu += ", ";
      params_gpu += "sampler " + p.name;
    }
    const std::string plain =
        vstage.ret + " weft_plain_vstage(" + params_gpu + ")" +
        vstage.orig_all.substr(vstage.orig_sig.size());
    bool all = true;
    for (size_t i = 0; i < lowers.size(); i++)
      if (!lower_fragment(text, lowers[i], i == 0 ? plain : std::string())) all = false;
    if (all) {
      gpu_text = text;
      r.lowered = true;
    } else {
      for (const FragLower& l : lowers)
        if (l.interp.find("weft_clip") != std::string::npos) {
          r.error = "the pass clips with [[clip_distance]] but its fragment cannot use "
                    "weft's interpolation, so the cut would be lost on the gpu";
          return r;
        }
    }
  }
  r.gpu_src = lower_gpu_stages(
      rewrite_warp_scans(lower_imageblocks(gpu_text, render ? render->width : 0)));

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
    if (tile_stages > 1) {
      r.error = "a render pass has one `[[tile]]` stage; this file has " +
                std::to_string(tile_stages);
      return r;
    }
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
      "// generated by weft, do not edit\n"
      "#include \"weft_runtime.hpp\"\n"
      "#include <algorithm>\n"
      "#include <atomic>\n"
      "#include <cmath>\n"
      "#include <cstdint>\n"
      "#include <cstring>\n";
  code += kMslPreamble;
  code += weft_f64_cpp_text();
  code += "using uint = unsigned int;\n"
      "using ushort = unsigned short;\n"
      "using uchar = unsigned char;\n";
#if defined(__linux__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
  code += "static_assert(sizeof(ulong) == 8, \"weft: ulong is not 64 bits on this target\");\n";
#else
  code += "using ulong = unsigned long long;\n";
#endif
  code += kMslMathText;  // vector math, which std does not have
  // The file's own declarations, after the library they may use.
  code += carried.text;
  // as_type<T> becomes this: a bit-for-bit reinterpretation.
  code += "template <class T, class U>\n"
      "// A narrow source, an integer literal under a 64-bit destination, fills the low\n"
      "// bytes of a zeroed destination instead of reading past itself.\n"
      "static inline T weft_bitcast(U u) {\n"
      "  T t{};\n"
      "  std::memcpy(&t, &u, sizeof(U) < sizeof(T) ? sizeof(U) : sizeof(T));\n"
      "  return t;\n"
      "}\n";
  code += "using std::min;\n"
      "using std::max;\n"
      "using std::abs;\n"
      "using std::fabs;\n"
      "using std::sqrt;\n"
      "using std::exp;\n"
      "using std::log;\n"
      "using std::pow;\n"
      "using std::fmod;\n"
      "using std::fma;\n"
      "\n" +
      body_out + "static const weft::KernelDef weft_table[] = {\n";
  for (const auto& k : r.kernels) {
    code += "  {\"" + k.name + "\", " +
            (k.sync ? std::string("nullptr") : ("weft_k_" + k.name)) + ", " +
            std::to_string(k.nargs) + ", " + (k.sync ? "true" : "false") + ", " +
            (k.flat ? ("weft_k_" + k.name + "_loop") : std::string("nullptr")) + ", " +
            (k.step ? ("weft_k_" + k.name) : std::string("nullptr")) + ", " +
            (k.step ? ("sizeof(weft_st_" + k.name + ")") : std::string("0")) + ", " +
            (k.sync && !k.step ? ("weft_k_" + k.name) : std::string("nullptr")) +
            ", nullptr, 0, " + (k.step ? ("weft_g_" + k.name) : std::string("nullptr")) +
            "},\n";
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
  code += emit_module_abi(r.kernels);

  r.code = code;
  r.ok = true;
  const size_t soft_at = r.code.find("inline float dot(float a, float b)");
  if (soft_at != std::string::npos) r.code.insert(soft_at, kWSoftMath);
  return r;
}
