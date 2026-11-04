// Checkers for the chat page: JavaScript ports of the Python checkers in
// formal/, which are themselves checked against Lean 4. LeanCheck.verify(prompt,
// answer) returns true or false for a prompt a ported domain claims, and null
// for any other prompt. tests/test_web.py runs this file under Node against the
// Python checkers on references, variants and mutated answers; the two must
// agree on every one.
(function (root) {
  "use strict";

  class Reject extends Error {}
  const reject = () => { throw new Reject(); };

  // ------------------------------------------------------------------ prop
  // Port of formal/prop.py: formulas in Lean 4 notation, proof terms checked
  // bidirectionally, ¬A unfolded to A → False, and tactic proofs (`by ...`) run
  // on their goal as Lean would.
  const Prop = (() => {
    const MAX_ANSWER = 400, MAX_PROMPT = 300, MAX_NESTING = 64;
    // Lean 4.34 runs out of recursion depth (maxRecDepth 512) elaborating a
    // header about 250 connectives deep, and its `trivial` splits every
    // conjunction of the goal before trying anything else, failing past about
    // 60 nested ∧. Stay well inside both (formal/prop.py has the measurements).
    const MAX_FORMULA_DEPTH = 200, TRIVIAL_DEPTH = 32;
    const TOP = ["top"], BOT = ["bot"];
    const OPS = { "∧": "and", "∨": "or", "→": "imp", "↔": "iff" };
    const SYMBOL = { and: "∧", or: "∨", imp: "→", iff: "↔" };
    const TIGHTER = { imp: "or", or: "and", and: null };
    const CONSTANTS = { "trivial": 0, "absurd": 2, "Or.inl": 1, "Or.inr": 1, "Or.elim": 3, "False.elim": 1,
      "And.intro": 2, "And.left": 1, "And.right": 1, "Iff.intro": 2 };
    const FIELDS = new Set(["1", "2", "left", "right", "mp", "mpr"]);
    // Never binder names: the constants' heads and core Lean 4.34's keywords (see prop.py).
    const RESERVED = new Set(["And", "False", "Iff", "Not", "Or", "Prop", "Sort", "StateRefT", "True", "Type", "_", "abbrev", "absurd", "add_decl_doc", "assert_not_exists", "assert_not_imported", "at", "attribute", "axiom", "bif", "binder_predicate", "break", "builtin_cbv_simproc", "builtin_cbv_simproc_decl", "builtin_dsimproc", "builtin_dsimproc_decl", "builtin_grind_propagator", "builtin_initialize", "builtin_simproc", "builtin_simproc_decl", "by", "by_elab", "calc", "catch", "cbv_eval", "cbv_simproc", "cbv_simproc_decl", "class", "coinductive", "coinductive_fixpoint", "continue", "dbg_trace", "declare_bitwise_int_theorems", "declare_bitwise_uint_theorems", "declare_int_theorems", "declare_simp_like_tactic", "declare_syntax_cat", "declare_uint_theorems", "decreasing_by", "def", "deprecated_module", "deprecated_syntax", "deriving", "do", "docs_to_verso", "dsimproc", "dsimproc_decl", "elab", "elab_rules", "else", "end", "eval_prec", "eval_prio", "example", "exists", "export", "extends", "finally", "for", "forall", "from", "fun", "generalizing", "grind_annotated", "grind_pattern", "grind_propagator", "have", "haveI", "hiding", "idbg", "if", "import", "in", "include", "include_str", "inductive", "inductive_fixpoint", "inferInstanceAs", "infix", "infixl", "infixr", "init_grind_norm", "init_quot", "initialize", "instance", "leading_parser", "let", "letI", "let_delayed", "let_expr", "let_fun", "let_tmp", "local", "logNamedError", "logNamedErrorAt", "logNamedWarning", "logNamedWarningAt", "macro", "macro_rules", "match", "match_expr", "matches", "max_prec", "meta", "mod_cast", "mut", "mutual", "namespace", "nat_lit", "no_index", "nofun", "nomatch", "noncomputable", "nonrec", "norm_cast_add_elim", "notation", "omit", "opaque", "open", "partial", "partial_fixpoint", "postfix", "prefix", "private", "protected", "public", "recommended_spelling", "register_error_explanation", "register_sym_dsimp", "register_sym_simp", "register_tactic_tag", "renaming", "repeat", "return", "run_cmd", "run_elab", "run_meta", "scoped", "seal", "section", "set_library_suggestions", "set_option", "show", "show_term", "show_term_elab", "simproc", "simproc_decl", "sorry", "structure", "suffices", "syntax", "tactic_alt", "tactic_extension", "tactic_name", "tactic_tag", "termination_by", "then", "theorem", "throwNamedError", "throwNamedErrorAt", "trailing_parser", "trivial", "try", "unif_hint", "universe", "unless", "unlock_limits", "unsafe", "unseal", "until", "using", "variable", "where", "while", "with", "with_weak_namespace", "without_expected_type"]);
    const UNCOMPILABLE = new RegExp("run_|native|\\b(?:import|prelude|open|export|namespace|section|end|variable|universe|omit|include|set_option|in|axiom|theorem|lemma|def|abbrev|instance|example|opaque|structure|class|inductive|coinductive|mutual|attribute|notation|infix|infixl|infixr|prefix|postfix|macro|macro_rules|syntax|elab|elab_rules|declare_syntax_cat|initialize|builtin_initialize|deriving|noncomputable|unsafe|partial|private|protected|local|scoped|where|add_decl_doc|register_simp_attr|simproc|dsimproc|binder_predicate|grind_pattern|unif_hint|implemented_by|extern|sorry|admit)\\b");
    const BINDER = /^[A-Za-z_][A-Za-z0-9_']*$/;
    const bindable = (name) => BINDER.test(name) && !RESERVED.has(name) && !UNCOMPILABLE.test(name);
    // Python's `.` excludes only \n; formulas are spaced with plain spaces only.
    const HEADER = /^example(?: \(([a-z][a-z ]*) : Prop\))? : ([^\n]+) :=$/;
    const FORMULA_TOKEN = / *([()¬∧∨→↔]|[A-Za-z_][A-Za-z0-9_']*)/y;
    const NAME = /[A-Za-z_][A-Za-z0-9_']*(?:\.(?:[A-Za-z_][A-Za-z0-9_']*|[0-9]+))*/y;
    const FIELD = /\.([A-Za-z_][A-Za-z0-9_']*|[0-9]+)/y;
    const hasOwn = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

    function eq(a, b) {
      if (a.length !== b.length || a[0] !== b[0]) return false;
      for (let i = 1; i < a.length; i++) {
        if (typeof a[i] === "string" ? a[i] !== b[i] : !eq(a[i], b[i])) return false;
      }
      return true;
    }

    function parseFormula(text) {
      const toks = [];
      let pos = 0, m;
      FORMULA_TOKEN.lastIndex = 0;
      while ((m = FORMULA_TOKEN.exec(text)) !== null) {
        toks.push(m[1]);
        pos = FORMULA_TOKEN.lastIndex;
      }
      if (!/^ *$/.test(text.slice(pos))) reject();
      let i = 0;
      const peek = () => (i < toks.length ? toks[i] : "");
      function iff(depth) {
        const a = chain("imp", depth);
        if (peek() !== "↔") return a;
        i++;
        const b = chain("imp", depth);
        if (peek() === "↔") reject();
        return ["iff", a, b];
      }
      function chain(op, depth) {
        if (op === null) return unary(depth);
        const parts = [chain(TIGHTER[op], depth)];
        while (peek() === SYMBOL[op]) { i++; parts.push(chain(TIGHTER[op], depth)); }
        let f = parts.pop();
        while (parts.length) f = [op, parts.pop(), f];
        return f;
      }
      function unary(depth) {
        let nots = 0;
        while (peek() === "¬") { nots++; i++; }
        const tok = peek();
        i++;
        let f;
        if (tok === "(") {
          if (depth >= MAX_NESTING) reject();
          f = iff(depth + 1);
          if (peek() !== ")") reject();
          i++;
        } else if (tok === "True") f = TOP;
        else if (tok === "False") f = BOT;
        else if (/^[A-Za-z_]/.test(tok)) f = ["atom", tok];
        else reject();
        for (let k = 0; k < nots; k++) f = ["not", f];
        return f;
      }
      const f = iff(0);
      if (i !== toks.length) reject();
      return f;
    }

    function unfold(f) {
      if (f[0] === "not") return ["imp", unfold(f[1]), BOT];
      if (f[0] === "and" || f[0] === "or" || f[0] === "imp" || f[0] === "iff") return [f[0], unfold(f[1]), unfold(f[2])];
      return f;
    }

    // The most connectives (¬ included) on one path from the root to a leaf.
    function formulaDepth(f) {
      let best = 0;
      const stack = [[f, 0]];
      while (stack.length) {
        const [g, d] = stack.pop();
        if (g[0] === "atom" || g[0] === "top" || g[0] === "bot") best = Math.max(best, d);
        else for (let k = 1; k < g.length; k++) stack.push([g[k], d + 1]);
      }
      return best;
    }

    // How deep the conjunctions at the top of f nest: what Lean's `trivial` splits.
    const andHeight = (f) => (f[0] === "and" ? 1 + Math.max(andHeight(f[1]), andHeight(f[2])) : 0);

    function atomsOf(f, out = new Set()) {
      if (f[0] === "atom") out.add(f[1]);
      else for (let k = 1; k < f.length; k++) atomsOf(f[k], out);
      return out;
    }

    function goalOf(prompt) {
      if (typeof prompt !== "string" || prompt.length > MAX_PROMPT) return null;
      const m = HEADER.exec(prompt);
      if (!m) return null;
      const names = m[1] ? m[1].split(" ").filter(Boolean) : [];
      let goal;
      try { goal = parseFormula(m[2]); } catch (e) { if (e instanceof Reject || e instanceof RangeError) return null; throw e; }
      if (new Set(names).size !== names.length) return null;
      for (const a of atomsOf(goal)) if (!names.includes(a)) return null;
      if (names.some((name) => RESERVED.has(name))) return null;
      return goal;
    }

    // Tokens, and for each whether whitespace precedes it: Lean reads an argument
    // only after whitespace (`f(x)` is not an application). A term stays on one line.
    function lexTerm(text) {
      const toks = [], spaced = [];
      let i = 0, gap = true;
      const n = text.length;
      while (i < n) {
        const c = text[i];
        if (c === " ") { i++; gap = true; continue; }
        spaced.push(gap);
        gap = false;
        if (c === "(" || c === "⟨" || c === ",") { toks.push(["p", c]); i++; }
        else if (c === ")" || c === "⟩") {
          toks.push(["p", c]);
          i++;
          for (;;) {
            FIELD.lastIndex = i;
            const m = FIELD.exec(text);
            if (!m) break;
            toks.push([".", m[1]]);
            spaced.push(false);
            i = FIELD.lastIndex;
          }
        } else if (text.startsWith("=>", i)) { toks.push(["p", "=>"]); i += 2; }
        else {
          NAME.lastIndex = i;
          const m = NAME.exec(text);
          if (!m) reject();
          toks.push(["id", m[0]]);
          i = NAME.lastIndex;
        }
      }
      return [toks, spaced];
    }

    // `outer` holds the hypotheses already named outside the term (a tactic proof's context).
    function parseTerm(text, outer = []) {
      const [toks, spaced] = lexTerm(text);
      let i = 0, depth = 0;
      const scope = [...outer];
      const END = ["end", ""];
      const peek = () => (i < toks.length ? toks[i] : END);
      const next = () => { const t = peek(); if (t === END) reject(); i++; return t; };
      const isP = (t, s) => t[0] === "p" && t[1] === s;
      const project = (t, field) => { if (!FIELDS.has(field)) reject(); return ["proj", t, field]; };
      function startsAtom() {
        const [kind, text] = peek();
        return (kind === "id" && text !== "fun") || (kind === "p" && (text === "(" || text === "⟨"));
      }
      function term() {
        if (++depth > MAX_NESTING) reject();
        let t;
        const [k0, n0] = peek();
        if (k0 === "id" && n0 === "fun") {
          i++;
          const names = [];
          while (!isP(peek(), "=>")) {
            const [kind, name] = next();
            if (kind !== "id" || !(name === "_" || bindable(name))) reject();
            names.push(name);
          }
          if (!names.length) reject();
          i++;
          scope.push(...names);
          t = term();
          scope.length -= names.length;
          for (let k = names.length - 1; k >= 0; k--) t = ["lam", names[k], t];
        } else {
          if (k0 === "id" && hasOwn(CONSTANTS, n0) && CONSTANTS[n0] > 0) {
            i++;
            const args = [];
            for (let k = 0; k < CONSTANTS[n0]; k++) args.push(arg());
            t = ["const", n0, args];
          } else t = atom();
          const args = [];
          while (startsAtom()) args.push(arg());
          if (args.length) t = ["app", t, args];
        }
        depth--;
        return t;
      }
      function arg() {
        if (i < toks.length && !spaced[i]) reject();
        return atom();
      }
      function atom() {
        const [kind, text] = next();
        let t;
        if (kind === "id") {
          const parts = text.split(".");
          let fields;
          if (parts[0] !== "_" && scope.includes(parts[0])) {
            t = ["var", parts[0]];
            fields = parts.slice(1);
          } else {
            const width = hasOwn(CONSTANTS, parts[0]) ? 1 : 2;
            const name = parts.slice(0, width).join(".");
            if (!hasOwn(CONSTANTS, name) || CONSTANTS[name] !== 0) reject();
            t = ["const", name, []];
            fields = parts.slice(width);
          }
          for (const f of fields) t = project(t, f);
          return t;
        }
        if (kind === "p" && text === "(") {
          t = term();
          if (!isP(next(), ")")) reject();
        } else if (kind === "p" && text === "⟨") {
          const items = [term()];
          while (isP(peek(), ",")) { i++; items.push(term()); }
          if (!isP(next(), "⟩")) reject();
          if (items.length < 2) reject();
          t = ["anon", items];
        } else reject();
        while (peek()[0] === ".") t = project(t, next()[1]);
        return t;
      }
      const t = term();
      if (i !== toks.length) reject();
      return t;
    }

    // Contexts are linked lists [name, type, outer], innermost binder first.
    function lookup(ctx, name) {
      for (; ctx !== null; ctx = ctx[2]) if (ctx[0] === name) return ctx[1];
      return reject();
    }

    function check(ctx, t, ty) {
      const tag = t[0];
      if (tag === "lam") {
        if (ty[0] !== "imp") reject();
        check([t[1], ty[1], ctx], t[2], ty[2]);
      } else if (tag === "anon") checkFields(ctx, t[1], ty);
      else if (tag === "const" && t[1] !== "And.left" && t[1] !== "And.right") checkConst(ctx, t[1], t[2], ty);
      else if (!eq(infer(ctx, t), ty)) reject();
    }

    function fields(ty) {
      if (ty[0] === "and") return [ty[1], ty[2]];
      if (ty[0] === "iff") return [["imp", ty[1], ty[2]], ["imp", ty[2], ty[1]]];
      return reject();
    }

    function checkFields(ctx, items, ty) {
      const [first, rest] = fields(ty);
      check(ctx, items[0], first);
      if (items.length === 2) check(ctx, items[1], rest);
      else checkFields(ctx, items.slice(1), rest);
    }

    function checkConst(ctx, name, args, ty) {
      if (name === "Or.inl" || name === "Or.inr") {
        if (ty[0] !== "or") reject();
        check(ctx, args[0], name === "Or.inl" ? ty[1] : ty[2]);
      } else if (name === "Or.elim") {
        const d = infer(ctx, args[0]);
        if (d[0] !== "or") reject();
        check(ctx, args[1], ["imp", d[1], ty]);
        check(ctx, args[2], ["imp", d[2], ty]);
      } else if (name === "False.elim") check(ctx, args[0], BOT);
      else if (name === "absurd") check(ctx, args[1], ["imp", infer(ctx, args[0]), BOT]);
      else if (name === "And.intro" || name === "Iff.intro") {
        if (ty[0] !== (name === "And.intro" ? "and" : "iff")) reject();
        checkFields(ctx, args, ty);
      } else if (!eq(ty, TOP)) reject();
    }

    function infer(ctx, t) {
      const tag = t[0];
      if (tag === "var") return lookup(ctx, t[1]);
      if (tag === "proj") {
        const ty = infer(ctx, t[1]), field = t[2];
        if (ty[0] === "and" && ["1", "left", "2", "right"].includes(field)) return field === "1" || field === "left" ? ty[1] : ty[2];
        if (ty[0] === "iff" && ["1", "mp", "2", "mpr"].includes(field)) {
          return field === "1" || field === "mp" ? ["imp", ty[1], ty[2]] : ["imp", ty[2], ty[1]];
        }
        return reject();
      }
      if (tag === "app") {
        let ty = infer(ctx, t[1]);
        for (const arg of t[2]) {
          if (ty[0] !== "imp") reject();
          check(ctx, arg, ty[1]);
          ty = ty[2];
        }
        return ty;
      }
      if (tag === "const") {
        const [, name, args] = t;
        if (name === "trivial") return TOP;
        if (name === "And.left" || name === "And.right") {
          const ty = infer(ctx, args[0]);
          if (ty[0] !== "and") reject();
          return name === "And.left" ? ty[1] : ty[2];
        }
        if (name === "And.intro") return ["and", infer(ctx, args[0]), infer(ctx, args[1])];
        if (name === "Iff.intro") {
          const f = infer(ctx, args[0]), g = infer(ctx, args[1]);
          if (f[0] !== "imp" || !eq(g, ["imp", f[2], f[1]])) reject();
          return ["iff", f[1], f[2]];
        }
        if (name === "Or.elim") {
          const d = infer(ctx, args[0]), f = infer(ctx, args[1]);
          if (d[0] !== "or" || f[0] !== "imp" || !eq(f[1], d[1])) reject();
          check(ctx, args[2], ["imp", d[2], f[2]]);
          return f[2];
        }
      }
      return reject();
    }

    // Tactics. A proof is lexed into tokens [text, line, column, start, end, first
    // on its line]; as in lexTerm, projections written directly after `)` or `⟩`
    // stay attached to it. Any other character, a tab included, is refused.
    const TACTIC_TOKEN = /[A-Za-z_][A-Za-z0-9_']*(?:\.(?:[A-Za-z_][A-Za-z0-9_']*|[0-9]+))*|:=|=>|[()⟨⟩,;|·]/y;
    const PROJECTIONS = /(?:\.(?:[A-Za-z_][A-Za-z0-9_']*|[0-9]+))+/y;
    // What ends the term of an exact, apply, cases or obtain; a term never continues on a new line.
    const TERM_STOP = new Set([";", "|", ":=", "·", "with"]);
    // Names Lean accepts, but a hypothesis named like a tactic or an rcases keyword only invites misreadings.
    const TACTIC_WORDS = new Set(["intro", "exact", "apply", "constructor", "left", "right", "exfalso", "assumption",
      "trivial", "contradiction", "cases", "rcases", "obtain", "inl", "inr", "rfl"]);
    const MAX_BRANCHES = 64;  // goals one rcases pattern may split into
    // `apply c` for a constant whose implicit arguments Lean unifies with the goal.
    const INTRO_RULES = ["Or.inl", "Or.inr", "And.intro", "Iff.intro", "False.elim"];
    const hypName = (name) => bindable(name) && !TACTIC_WORDS.has(name);

    function lexTactics(text) {
      const toks = [];
      let i = 0, line = 0, col = 0, first = true;
      while (i < text.length) {
        const c = text[i];
        if (c === " ") { i++; col++; }
        else if (c === "\n") { i++; line++; col = 0; first = true; }
        else {
          TACTIC_TOKEN.lastIndex = i;
          const m = TACTIC_TOKEN.exec(text);
          if (!m) reject();
          let end = TACTIC_TOKEN.lastIndex;
          if (m[0] === ")" || m[0] === "⟩") {
            PROJECTIONS.lastIndex = end;
            if (PROJECTIONS.exec(text)) end = PROJECTIONS.lastIndex;
          }
          toks.push([text.slice(i, end), line, col, i, end, first]);
          col += end - i;  // every token character is in the BMP, so this counts code points as prop.py does
          i = end;
          first = false;
        }
      }
      return toks;
    }

    // Every hypothesis type of a context, the inaccessible ones (named null) included.
    function hyps(ctx) {
      const out = [];
      for (; ctx !== null; ctx = ctx[2]) out.push(ctx[1]);
      return out;
    }
    const member = (f, list) => list.some((g) => eq(f, g));

    // `ctx` after cases or rcases consumes the hypothesis `name` refers to: that
    // one is removed, and older ones it shadowed lose their name too.
    function clear(ctx, name) {
      const entries = [];
      for (; ctx !== null; ctx = ctx[2]) entries.push(ctx);
      const innermost = entries.findIndex((e) => e[0] === name);
      let out = null;
      for (let k = entries.length - 1; k >= 0; k--) {
        if (k !== innermost) out = [entries[k][0] === name ? null : entries[k][0], entries[k][1], out];
      }
      return out;
    }

    // What Lean's `contradiction` certainly closes: a hypothesis False, or ¬a (written either way) with a.
    function contradiction(ctx) {
      const hs = hyps(ctx);
      return member(BOT, hs) || hs.some((h) => h[0] === "imp" && eq(h[2], BOT) && member(h[1], hs));
    }

    // The parts of Lean's `trivial` trusted here: assumption, contradiction,
    // True.intro and `apply And.intro <;> trivial`.
    function trivial(ctx, goal) {
      if (eq(goal, TOP) || member(goal, hyps(ctx)) || contradiction(ctx)) return true;
      return goal[0] === "and" && trivial(ctx, goal[1]) && trivial(ctx, goal[2]);
    }

    function arrows(f) {
      let n = 0;
      for (; f[0] === "imp"; f = f[2]) n++;
      return n;
    }

    // The type `cases ... with | intro h1 h2` gives the fields of an iff: Lean
    // records it as `?a → ?b`, assigned but not instantiated. It equals the
    // formula it stands for; only applyGoals tells the two apart.
    class Uninstantiated extends Array {}

    // The goals Lean's `apply e` leaves for `e : ty`, or null if it fails: the
    // first i from arrows(ty) - arrows(goal) up to all of them whose conclusion
    // after i arguments is the goal, the arguments becoming the new goals. With
    // a metavariable as conclusion (`?a → ?b`, b no implication) only all of them.
    function applyGoals(ty, goal) {
      const premises = [], concl = [ty];
      while (concl[concl.length - 1][0] === "imp") {
        premises.push(concl[concl.length - 1][1]);
        concl.push(concl[concl.length - 1][2]);
      }
      const start = ty instanceof Uninstantiated && ty[2][0] !== "imp" ? premises.length
        : Math.max(0, premises.length - arrows(goal));
      for (let k = start; k <= premises.length; k++) if (eq(concl[k], goal)) return premises.slice(0, k);
      return null;
    }

    function introRule(name, goal) {
      // False.elim : {C : Sort u} → False → C. On a goal False → c, Lean's first
      // try instantiates only C, which unifies and closes the goal outright.
      if (name === "False.elim") return goal[0] === "imp" && eq(goal[1], BOT) ? [] : [BOT];
      if ((name === "Or.inl" || name === "Or.inr") && goal[0] === "or") return [name === "Or.inl" ? goal[1] : goal[2]];
      if ((name === "And.intro" || name === "Iff.intro") && goal[0] === (name === "And.intro" ? "and" : "iff")) return fields(goal);
      return null;
    }

    // The hypotheses each goal gets from an rcases pattern, goals in Lean's order:
    // ⟨p1, ..., pn⟩ splits ∧ and ↔, p1 | ... | pn splits ∨, surplus patterns nesting into the last.
    function destruct(pat, ty) {
      if (pat[0] === "name") return [[[pat[1], ty]]];
      const items = pat[1];
      const rest = items.length === 2 ? items[1] : [pat[0], items.slice(1)];
      let out;
      if (pat[0] === "tuple") {
        const [first, second] = fields(ty);
        const left = destruct(items[0], first), right = destruct(rest, second);
        out = [];
        for (const a of left) for (const b of right) out.push([...a, ...b]);
      } else if (ty[0] === "or") out = [...destruct(items[0], ty[1]), ...destruct(rest, ty[2])];
      else reject();
      if (out.length > MAX_BRANCHES) reject();
      return out;
    }

    // Runs a tactic proof on its goal as Lean would, over prop.py's fragment (see
    // _TacticChecker for the grammar). A sequence continues on a new line only at
    // the column of its first tactic, `;` binds the rest of the line into the
    // innermost sequence, and alternatives continue while each `|` is at least as
    // far right as the first; a line further right than the sequence it would
    // continue is rejected rather than guessed at. A goal is [ctx, target]; each
    // tactic acts on the first goal and puts the goals it creates in front.
    function runTactics(text, goal) {
      const toks = lexTactics(text);
      let i = 0, depth = 0;
      const peek = () => (i < toks.length ? toks[i] : null);
      const next = () => { const t = peek(); if (t === null) reject(); i++; return t; };
      // Whether the next token is on the current line (and reads `word`).
      const inline = (word) => { const t = peek(); return t !== null && !t[5] && (word === undefined || t[0] === word); };

      function seq(goals) {
        if (++depth > MAX_NESTING || peek() === null) reject();
        const col = peek()[2];
        goals = tactic(goals, col);
        for (let tok; (tok = peek()) !== null;) {
          if (tok[0] === ";" && !tok[5]) {
            i++;
            if (!inline()) reject();
          } else if (!tok[5] || tok[2] < col) break;  // the enclosing construct decides what the token is
          else if (tok[2] > col) reject();
          goals = tactic(goals, col);
        }
        depth--;
        return goals;
      }

      function names() {
        const out = [];
        while (inline() && BINDER.test(peek()[0])) {
          const name = next()[0];
          if (!hypName(name)) reject();
          out.push(name);
        }
        return out;
      }

      function termText() {
        const start = i;
        while (inline() && !TERM_STOP.has(peek()[0])) i++;
        if (i === start) reject();
        return text.slice(toks[start][3], toks[i - 1][4]);
      }

      function parse(ctx, source) {
        const scope = [];
        for (let c = ctx; c !== null; c = c[2]) if (c[0] !== null) scope.push(c[0]);
        return parseTerm(source, scope);
      }

      // The type of the term a cases or rcases destructs, and the context
      // without it: Lean clears a destructed hypothesis, but nothing else.
      function subject(ctx) {
        const term = parse(ctx, termText());
        const ty = infer(ctx, term);
        return [ty, term[0] === "var" ? clear(ctx, term[1]) : ctx];
      }

      function tactic(goals, col) {
        const word = next()[0];
        if (!goals.length) reject();  // Lean: "no goals to be solved"
        let [ctx, target] = goals[0];
        const rest = goals.slice(1);
        if (word === "·") {
          if (!inline() || seq([goals[0]]).length) reject();
          return rest;
        }
        if (word === "intro") {
          const introduced = names();
          if (!introduced.length) reject();
          for (const name of introduced) {
            if (target[0] !== "imp") reject();
            ctx = [name, target[1], ctx];
            target = target[2];
          }
          return [[ctx, target], ...rest];
        }
        if (word === "exact") {
          check(ctx, parse(ctx, termText()), target);
          return rest;
        }
        if (word === "apply") {
          const source = termText();
          const created = INTRO_RULES.includes(source) ? introRule(source, target)
            : applyGoals(infer(ctx, parse(ctx, source)), target);
          if (created === null) reject();
          return [...created.map((g) => [ctx, g]), ...rest];
        }
        if (word === "cases") {
          const [ty, cleared] = subject(ctx);
          if (!inline("with")) {
            if (!eq(ty, BOT)) reject();  // `cases h` without alternatives: only for h : False
            return rest;
          }
          i++;
          alternatives(cleared, ty, target, col);
          return rest;
        }
        if (word === "rcases" || word === "obtain") {
          let ty, pat;
          if (word === "rcases") {
            [ty, ctx] = subject(ctx);
            if (!inline("with")) reject();
            i++;
            pat = pattern();
          } else {
            pat = pattern();
            if (!inline(":=")) reject();
            i++;
            [ty, ctx] = subject(ctx);
          }
          if (pat[0] === "name") reject();
          const out = [];
          for (const branch of destruct(pat, ty)) {
            if (new Set(branch.map(([name]) => name)).size !== branch.length) reject();  // one name twice in a goal
            let c = ctx;
            for (const [name, f] of branch) c = [name, f, c];
            out.push([c, target]);
          }
          return [...out, ...rest];
        }
        if (word === "constructor") {
          if (target[0] === "and" || target[0] === "iff") return [...fields(target).map((g) => [ctx, g]), ...rest];
          if (!eq(target, TOP)) reject();
          return rest;
        }
        if (word === "left" || word === "right") {
          if (target[0] !== "or") reject();
          return [[ctx, word === "left" ? target[1] : target[2]], ...rest];
        }
        if (word === "exfalso") return [[ctx, BOT], ...rest];
        let closed;
        if (word === "assumption") closed = member(target, hyps(ctx));
        else if (word === "contradiction") closed = contradiction(ctx);
        else if (word === "trivial") closed = andHeight(target) <= TRIVIAL_DEPTH && trivial(ctx, target);
        else reject();
        if (!closed) reject();
        return rest;
      }

      // `| inl h => seq | inr h => seq` for ∨, `| intro h1 h2 => seq` for ∧ and ↔,
      // each exactly once, in any order, each closing its goal. A Map, so that
      // no prototype key (`constructor`, `__proto__`) is ever an alternative.
      function alternatives(ctx, ty, target, col) {
        const wanted = new Map();
        if (ty[0] === "or") {
          wanted.set("inl", [ty[1]]);
          wanted.set("inr", [ty[2]]);
        } else {
          const parts = fields(ty);
          wanted.set("intro", ty[0] === "iff" ? parts.map((f) => Uninstantiated.from(f)) : parts);
        }
        let tok = peek();
        if (tok === null || tok[0] !== "|" || (tok[5] && tok[2] < col)) reject();
        const anchor = tok[2];
        while ((tok = peek()) !== null && tok[0] === "|" && tok[2] >= anchor) {
          i++;
          if (!inline()) reject();
          const ctor = next()[0];
          const parts = wanted.has(ctor) ? wanted.get(ctor) : null;  // null: an unknown or repeated alternative
          wanted.delete(ctor);
          const bound = names();
          if (parts === null || bound.length !== parts.length || !inline("=>")) reject();
          i++;
          const body = peek();
          if (body === null || (body[5] && body[2] <= tok[2])) reject();
          let arm = ctx;
          bound.forEach((name, k) => { arm = [name, parts[k], arm]; });
          if (seq([[arm, target]]).length) reject();
        }
        if (wanted.size) reject();  // Lean: "alternative has not been provided"
      }

      // pattern := item ("|" item)*, item := name | "⟨" pattern ("," pattern)+ "⟩" | "(" pattern ")"
      function pattern() {
        if (++depth > MAX_NESTING) reject();
        const items = [patternItem()];
        while (inline("|")) { i++; items.push(patternItem()); }
        depth--;
        return items.length === 1 ? items[0] : ["alt", items];
      }

      function patternItem() {
        if (!inline()) reject();
        const tok = next()[0];
        if (tok === "⟨") {
          const items = [pattern()];
          while (inline(",")) { i++; items.push(pattern()); }
          if (items.length < 2 || !inline("⟩")) reject();
          i++;
          return ["tuple", items];
        }
        if (tok === "(") {
          const pat = pattern();
          if (!inline(")")) reject();
          i++;
          return pat;
        }
        if (!hypName(tok)) reject();
        return ["name", tok];
      }

      if (next()[0] !== "by" || peek() === null) reject();
      // The first line also holds the prompt, which shifts its columns, so a
      // multi-line proof keeps `by` alone on it and starts the block at column 2.
      if (text.includes("\n") && (!peek()[5] || peek()[2] !== 2)) reject();
      if (seq([[null, goal]]).length || i !== toks.length) reject();
    }

    function verify(prompt, answer) {
      if (typeof answer !== "string" || answer.length > MAX_ANSWER) return false;
      const goal = goalOf(prompt);
      if (goal === null || formulaDepth(goal) > MAX_FORMULA_DEPTH) return false;
      try {
        if (answer.startsWith("by ") || answer.startsWith("by\n")) runTactics(answer, unfold(goal));
        else check(null, parseTerm(answer), unfold(goal));
        return true;
      } catch (e) {
        if (e instanceof Reject || e instanceof RangeError) return false;
        throw e;
      }
    }

    return { name: "prop", claims: (prompt) => goalOf(prompt) !== null, verify };
  })();

  // ------------------------------------------------------------------ fol
  // Port of formal/fol.py: intuitionistic first-order logic over one type α.
  // Formulas are locally nameless (a bound variable is a de Bruijn index, its
  // name only a hint), so equality is alpha-equivalence and instantiation
  // cannot capture; proof terms are checked bidirectionally, ¬A unfolded to
  // A → False. fol.py checks term proofs only, so a `by` answer is rejected.
  const Fol = (() => {
    const MAX_ANSWER = 400, MAX_PROMPT = 300, MAX_NESTING = 64;
    const TOP = ["top"], BOT = ["bot"];
    const BINARY = new Set(["and", "or", "imp", "iff"]);
    const SYMBOL = { and: "∧", or: "∨", imp: "→", iff: "↔" };
    const TIGHTER = { imp: "or", or: "and", and: null };
    const QUANTIFIERS = { "∀": "all", "∃": "ex" };
    const ARITY = { P: 1, Q: 1, R: 1, r: 2, p: 0, q: 0 };
    const INDIVIDUALS = ["a", "b"];
    const GROUPS = ["PQR", "r", "pq", "ab"];  // the names each binder group of the header may declare
    const CONSTANTS = { "trivial": 0, "absurd": 2, "Or.inl": 1, "Or.inr": 1, "Or.elim": 3, "False.elim": 1,
      "And.intro": 2, "And.left": 1, "And.right": 1, "Iff.intro": 2, "Exists.intro": 2, "Exists.elim": 2 };
    const CHECKED = new Set(["trivial", "absurd", "Or.inl", "Or.inr", "Or.elim", "False.elim", "And.intro", "Iff.intro",
      "Exists.intro", "Exists.elim"]);
    const FIELDS = new Set(["1", "2", "left", "right", "mp", "mpr", "elim"]);
    // Lean reads only plain spaces between a formula's tokens (a tab or any other
    // Unicode space is an error), so nothing else separates them here.
    const FORMULA_TOKEN = / *([()¬∧∨→↔∀∃,]|[A-Za-z_][A-Za-z0-9_']*)/y;
    const BLANK = /^ *$/;
    const VARIABLE = /^[a-z]'*$/;
    // Python's `.` excludes only \n; the header's own spacing is plain spaces.
    const HEADER = /^example \(α : Type\)(?: \(([PQR ]+) : α → Prop\))?(?: \((r) : α → α → Prop\))?(?: \(([pq ]+) : Prop\))?(?: \(([ab ]+) : α\))? : ([^\n]+) :=$/;
    const BINDER = /^(?:[A-Za-z][0-9_']*|_)$/;
    const NAME = /[A-Za-z_][A-Za-z0-9_']*(?:\.(?:[A-Za-z_][A-Za-z0-9_']*|[0-9]+))*/y;
    const FIELD = /\.([A-Za-z_][A-Za-z0-9_']*|[0-9]+)/y;
    const hasOwn = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
    const isQuantifier = (tag) => tag === "all" || tag === "ex";

    // Formulas are arrays: ["pred", name, args] with each arg a de Bruijn index
    // (a number, 0 the innermost quantifier) or a free individual (a string),
    // ["top"], ["bot"], ["not", a], [op, a, b], ["all" | "ex", hint, body].
    // Equality ignores hints, which makes it alpha-equivalence, as fol.py's _Hint does.
    function eq(a, b) {
      if (a === b) return true;
      if (a[0] !== b[0]) return false;
      const tag = a[0];
      if (tag === "pred") return a[1] === b[1] && a[2].length === b[2].length && a[2].every((x, k) => x === b[2][k]);
      if (isQuantifier(tag)) return eq(a[2], b[2]);
      if (tag === "not") return eq(a[1], b[1]);
      if (BINARY.has(tag)) return eq(a[1], b[1]) && eq(a[2], b[2]);
      return true;  // top, bot
    }

    // The body f of a quantifier of a closed formula with its variable (index k
    // under k more binders) instantiated with the free individual t: nothing in
    // a closed formula can capture t, so no renaming is ever needed.
    function open(f, t, k = 0) {
      const tag = f[0];
      if (tag === "pred") return f[2].includes(k) ? ["pred", f[1], f[2].map((x) => (x === k ? t : x))] : f;
      if (BINARY.has(tag)) return [tag, open(f[1], t, k), open(f[2], t, k)];
      if (tag === "not") return ["not", open(f[1], t, k)];
      if (isQuantifier(tag)) return [tag, f[1], open(f[2], t, k + 1)];
      return f;
    }

    // Whether the variable with index k occurs in f: Lean infers a quantified
    // variable's type from a use, so an unused one is rejected.
    function mentions(f, k = 0) {
      if (f[0] === "pred") return f[2].includes(k);
      if (isQuantifier(f[0])) return mentions(f[2], k + 1);
      for (let j = 1; j < f.length; j++) if (mentions(f[j], k)) return true;
      return false;
    }

    function unfold(f) {
      const tag = f[0];
      if (tag === "not") return ["imp", unfold(f[1]), BOT];
      if (BINARY.has(tag)) return [tag, unfold(f[1]), unfold(f[2])];
      if (isQuantifier(tag)) return [tag, f[1], unfold(f[2])];
      return f;
    }

    // The predicate and atom names and the free individuals of f, so that the
    // header must declare exactly the symbols the formula uses.
    function symbols(f, preds, individuals) {
      if (f[0] === "pred") {
        preds.add(f[1]);
        for (const x of f[2]) if (typeof x === "string") individuals.add(x);
      } else if (isQuantifier(f[0])) symbols(f[2], preds, individuals);
      else for (let j = 1; j < f.length; j++) symbols(f[j], preds, individuals);
    }

    // Precedence climbing as in prop, plus quantifiers whose body extends as far
    // right as possible. `arity` (a Map, so no prototype key is ever a symbol)
    // gives the declared predicates and atoms; only parentheses and quantifiers
    // recurse, and their depth is bounded.
    function parseFormula(text, arity, individuals) {
      const toks = [];
      let pos = 0, m;
      FORMULA_TOKEN.lastIndex = 0;
      while ((m = FORMULA_TOKEN.exec(text)) !== null) {
        toks.push(m[1]);
        pos = FORMULA_TOKEN.lastIndex;
      }
      if (!BLANK.test(text.slice(pos))) reject();
      const scope = [];
      let i = 0;
      const peek = () => (i < toks.length ? toks[i] : "");
      function iff(depth) {
        const a = chain("imp", depth);
        if (peek() !== "↔") return a;
        i++;
        const b = chain("imp", depth);
        if (peek() === "↔") reject();  // `a ↔ b ↔ c` does not parse in Lean
        return ["iff", a, b];
      }
      function chain(op, depth) {
        if (op === null) return unary(depth);
        const parts = [chain(TIGHTER[op], depth)];
        while (peek() === SYMBOL[op]) { i++; parts.push(chain(TIGHTER[op], depth)); }
        let f = parts.pop();
        while (parts.length) f = [op, parts.pop(), f];
        return f;
      }
      function unary(depth) {
        let nots = 0;
        while (peek() === "¬") { nots++; i++; }
        const tok = peek();
        i++;
        if ((tok === "(" || tok === "∀" || tok === "∃") && depth >= MAX_NESTING) reject();
        let f;
        if (tok === "(") {
          f = iff(depth + 1);
          if (peek() !== ")") reject();
          i++;
        } else if (tok === "∀" || tok === "∃") {
          const name = peek();
          i++;
          if (!VARIABLE.test(name) || arity.has(name) || peek() !== ",") reject();
          i++;
          scope.push(name);
          const body = iff(depth + 1);
          scope.pop();
          if (!mentions(body)) reject();
          f = [QUANTIFIERS[tok], name, body];
        } else if (tok === "True") f = TOP;
        else if (tok === "False") f = BOT;
        else if (arity.has(tok)) {
          const args = [];
          for (let k = 0; k < arity.get(tok); k++) args.push(term());
          f = ["pred", tok, args];
        } else reject();
        for (let k = 0; k < nots; k++) f = ["not", f];
        return f;
      }
      // A bound variable, innermost binder first (so it shadows an individual), else a declared individual.
      function term() {
        const name = peek();
        i++;
        for (let j = scope.length - 1; j >= 0; j--) if (scope[j] === name) return scope.length - 1 - j;
        if (individuals.includes(name)) return name;
        return reject();
      }
      const f = iff(0);
      if (i !== toks.length) reject();
      return f;
    }

    // The goal and declared individuals, or null unless the header declares
    // exactly the symbols the formula uses, each once, in the fixed order.
    function goalOf(prompt) {
      if (typeof prompt !== "string" || prompt.length > MAX_PROMPT) return null;
      const m = HEADER.exec(prompt);
      if (!m) return null;
      const declared = [];
      for (let g = 0; g < GROUPS.length; g++) {
        const names = m[g + 1] ? m[g + 1].split(" ") : [];
        // One-letter names from the group's own set, strictly ascending: the
        // character class alone would also admit `(PQ : α → Prop)` or `(Q P : ...)`.
        for (let k = 0; k < names.length; k++) {
          if (names[k].length !== 1 || !GROUPS[g].includes(names[k]) || (k > 0 && !(names[k - 1] < names[k]))) return null;
        }
        declared.push(...names);
      }
      const individuals = declared.filter((n) => INDIVIDUALS.includes(n));
      const arity = new Map(declared.filter((n) => hasOwn(ARITY, n)).map((n) => [n, ARITY[n]]));
      const preds = new Set(), used = new Set();
      let goal;
      try {
        goal = parseFormula(m[5], arity, individuals);
        symbols(goal, preds, used);
      } catch (e) {
        if (e instanceof Reject || e instanceof RangeError) return null;
        throw e;
      }
      const same = (set, list) => set.size === list.length && list.every((x) => set.has(x));
      if (!same(preds, [...arity.keys()]) || !same(used, individuals)) return null;
      return [goal, individuals];
    }

    // Tokens, and for each whether a plain space precedes it: Lean reads an
    // argument only after whitespace (`f(x)`, `h.2x` do not parse), and a term
    // stays on one line. A projection written right after `)` or `⟩` is a "." token.
    function lexTerm(text) {
      const toks = [], spaced = [];
      let i = 0, gap = false;
      const n = text.length;
      while (i < n) {
        const c = text[i];
        if (c === " ") { gap = true; i++; continue; }
        spaced.push(gap);
        gap = false;
        if (c === "(" || c === "⟨" || c === ",") { toks.push(["p", c]); i++; }
        else if (c === ")" || c === "⟩") {
          toks.push(["p", c]);
          i++;
          for (;;) {
            FIELD.lastIndex = i;
            const m = FIELD.exec(text);
            if (!m) break;
            toks.push([".", m[1]]);
            spaced.push(false);
            i = FIELD.lastIndex;
          }
        } else if (text.startsWith("=>", i)) { toks.push(["p", "=>"]); i += 2; }
        else {
          NAME.lastIndex = i;
          const m = NAME.exec(text);
          if (!m) reject();
          toks.push(["id", m[0]]);
          i = NAME.lastIndex;
        }
      }
      return [toks, spaced];
    }

    function binderNames(binder, out) {
      if (typeof binder === "string") out.push(binder);
      else for (const item of binder[1]) binderNames(item, out);
    }

    // Recursive descent over fol.py's fragment (see _TermParser):
    //   term   := "fun" binder+ "=>" term | head (" " atom)*
    //   binder := name | "⟨" binder ("," binder)+ "⟩"
    //   head   := constant (" " atom)^arity | atom
    //   atom   := name ("." field)* | "(" term ")" ("." field)* | "⟨" term ("," term)+ "⟩" ("." field)*
    // Names resolve innermost binder first, then the prompt's individuals.
    function parseTerm(text, individuals) {
      const [toks, spaced] = lexTerm(text);
      let i = 0, depth = 0;
      const scope = [...individuals];
      const END = ["end", ""];
      const peek = () => (i < toks.length ? toks[i] : END);
      const next = () => { const t = peek(); if (t === END) reject(); i++; return t; };
      const isP = (t, s) => t[0] === "p" && t[1] === s;
      const enter = () => { if (++depth > MAX_NESTING) reject(); };
      const project = (t, field) => { if (!FIELDS.has(field)) reject(); return ["proj", t, field]; };
      function term() {
        enter();
        let t;
        const [k0, n0] = peek();
        if (k0 === "id" && n0 === "fun") {
          i++;
          const binders = [];
          while (!isP(peek(), "=>")) binders.push(binder());
          if (!binders.length) reject();
          i++;
          const names = [];
          for (const b of binders) binderNames(b, names);
          // Lean binds a group's pattern variables after its other binders, so
          // with a pattern in the group sequential scoping is only faithful when no name repeats.
          const bound = names.filter((n) => n !== "_");
          if (new Set(bound).size !== bound.length && !binders.every((b) => typeof b === "string")) reject();
          scope.push(...names);
          t = term();
          scope.length -= names.length;
          for (let k = binders.length - 1; k >= 0; k--) t = ["lam", binders[k], t];
        } else {
          if (k0 === "id" && hasOwn(CONSTANTS, n0) && CONSTANTS[n0] > 0) {
            i++;
            const args = [];
            for (let k = 0; k < CONSTANTS[n0]; k++) args.push(argument());
            t = ["const", n0, args];
          } else t = atom();
          const args = [];
          while (startsAtom()) args.push(argument());
          if (args.length) t = ["app", t, args];
        }
        depth--;
        return t;
      }
      function binder() {
        const [kind, text] = next();
        if (kind === "id" && BINDER.test(text)) return text;
        if (!isP([kind, text], "⟨")) reject();
        enter();
        const items = [binder()];
        while (isP(peek(), ",")) { i++; items.push(binder()); }
        if (!isP(next(), "⟩")) reject();
        depth--;
        if (items.length < 2) reject();
        return ["pat", items];
      }
      function startsAtom() {
        const [kind, text] = peek();
        return (kind === "id" && text !== "fun") || (kind === "p" && (text === "(" || text === "⟨"));
      }
      function argument() {
        if (i >= toks.length || !spaced[i]) reject();
        return atom();
      }
      function atom() {
        const [kind, text] = next();
        let t;
        if (kind === "id") {
          const parts = text.split(".");
          let fields;
          if (parts[0] !== "_" && scope.includes(parts[0])) {
            t = ["var", parts[0]];
            fields = parts.slice(1);
          } else {
            const width = hasOwn(CONSTANTS, parts[0]) ? 1 : 2;
            const name = parts.slice(0, width).join(".");
            if (!hasOwn(CONSTANTS, name) || CONSTANTS[name] !== 0) reject();  // unknown, or missing its arguments
            t = ["const", name, []];
            fields = parts.slice(width);
          }
          for (const f of fields) t = project(t, f);
          return t;
        }
        if (isP([kind, text], "(")) {
          t = term();
          if (!isP(next(), ")")) reject();
        } else if (isP([kind, text], "⟨")) {
          const items = [term()];
          while (isP(peek(), ",")) { i++; items.push(term()); }
          if (!isP(next(), "⟩")) reject();
          if (items.length < 2) reject();
          t = ["anon", items];
        } else reject();
        while (peek()[0] === ".") t = project(t, next()[1]);
        return t;
      }
      const t = term();
      if (i !== toks.length) reject();
      return t;
    }

    // Contexts are linked lists [name, kind, value, outer], so no branch of the
    // checker leaks bindings into another. A hypothesis's value is its formula;
    // an individual's is its identity, the declared name or a fresh "#n" shared
    // with no other individual, which is what enforces the eigenvariable conditions.
    const HYP = 0, IND = 1;
    let freshIds = 0;

    function lookup(ctx, name) {
      for (; ctx !== null; ctx = ctx[3]) if (ctx[0] === name) return ctx;
      return reject();
    }

    // The individual a term names; individuals are only ever variables.
    function individual(ctx, t) {
      if (t[0] !== "var") reject();
      const b = lookup(ctx, t[1]);
      if (b[1] !== IND) reject();
      return b[2];
    }

    function newIndividual(ctx, name, body) {
      const c = `#${freshIds++}`;
      return [[name, IND, c, ctx], open(body, c)];
    }

    // Binds a `fun` binder against the expected type: a name against ∀ binds a
    // fresh individual, anything against → a hypothesis (possibly a pattern).
    function bind(ctx, binder, ty) {
      if (typeof binder === "string" && ty[0] === "all") return newIndividual(ctx, binder, ty[2]);
      if (ty[0] !== "imp") reject();
      return [bindHyp(ctx, binder, ty[1]), ty[2]];
    }

    function bindHyp(ctx, binder, ty) {
      return typeof binder === "string" ? [binder, HYP, ty, ctx] : destructure(ctx, binder[1], ty);
    }

    // The pattern ⟨items⟩ against an ∧ or an ∃; surplus items nest into the last field, as in Lean.
    function destructure(ctx, items, ty) {
      let rest;
      if (ty[0] === "and") {
        ctx = bindHyp(ctx, items[0], ty[1]);
        rest = ty[2];
      } else if (ty[0] === "ex" && typeof items[0] === "string") [ctx, rest] = newIndividual(ctx, items[0], ty[2]);
      else reject();
      return items.length === 2 ? bindHyp(ctx, items[1], rest) : destructure(ctx, items.slice(1), rest);
    }

    function check(ctx, t, ty) {
      const tag = t[0];
      if (tag === "lam") {
        const [inner, body] = bind(ctx, t[1], ty);
        check(inner, t[2], body);
      } else if (tag === "anon") checkFields(ctx, t[1], ty);
      else if (tag === "const" && CHECKED.has(t[1])) checkConst(ctx, t[1], t[2], ty);
      else if (tag === "proj" && t[2] === "elim") checkElim(ctx, t[1], [], ty, null);
      else if (tag === "app" && t[1][0] === "proj" && t[1][2] === "elim") checkElim(ctx, t[1][1], t[2], ty, null);
      else if (!eq(infer(ctx, t), ty)) reject();
    }

    function fields(ty) {
      if (ty[0] === "and") return [ty[1], ty[2]];
      if (ty[0] === "iff") return [["imp", ty[1], ty[2]], ["imp", ty[2], ty[1]]];  // Iff.intro mp mpr
      return reject();
    }

    // ⟨a, b⟩ against ∧ or ↔, or ⟨w, h⟩ against ∃ with an individual witness w;
    // ⟨a, b, c⟩ nests the surplus into the last field, as in Lean.
    function checkFields(ctx, items, ty) {
      let rest;
      if (ty[0] === "ex") rest = open(ty[2], individual(ctx, items[0]));
      else {
        const [first, second] = fields(ty);
        check(ctx, items[0], first);
        rest = second;
      }
      if (items.length === 2) check(ctx, items[1], rest);
      else checkFields(ctx, items.slice(1), rest);
    }

    function checkConst(ctx, name, args, ty) {
      if (name === "Or.inl" || name === "Or.inr") {
        if (ty[0] !== "or") reject();
        check(ctx, args[0], name === "Or.inl" ? ty[1] : ty[2]);
      } else if (name === "Or.elim" || name === "Exists.elim") {
        checkElim(ctx, args[0], args.slice(1), ty, name === "Or.elim" ? "or" : "ex");
      } else if (name === "Exists.intro") {
        if (ty[0] !== "ex") reject();
        check(ctx, args[1], open(ty[2], individual(ctx, args[0])));
      } else if (name === "False.elim") check(ctx, args[0], BOT);
      // Lean elaborates absurd's arguments left to right, so the first one's type must be known on its own.
      else if (name === "absurd") check(ctx, args[1], ["imp", infer(ctx, args[0]), BOT]);
      else if (name === "And.intro" || name === "Iff.intro") {
        if (ty[0] !== (name === "And.intro" ? "and" : "iff")) reject();
        checkFields(ctx, args, ty);
      } else if (!eq(ty, TOP)) reject();  // trivial
    }

    // `subject.elim args` (or Or.elim / Exists.elim when `kind` is given) against
    // ty. Lean elaborates the subject first, so its type must be inferable; the
    // cases are then checked against the motive ty (closed, so it may sit under the ∀).
    function checkElim(ctx, subject, args, ty, kind) {
      const d = infer(ctx, subject);
      if (kind !== null && d[0] !== kind) reject();
      if (d[0] === "bot" && !args.length) return;
      if (d[0] === "or" && args.length === 2) {
        check(ctx, args[0], ["imp", d[1], ty]);
        check(ctx, args[1], ["imp", d[2], ty]);
      } else if (d[0] === "ex" && args.length === 1) check(ctx, args[0], ["all", d[1], ["imp", d[2], ty]]);
      else reject();
    }

    // The type of a term determined without an expected type: hypotheses,
    // projections, applications and some constants; everything else needs one.
    function infer(ctx, t) {
      const tag = t[0];
      if (tag === "var") {
        const b = lookup(ctx, t[1]);
        if (b[1] !== HYP) reject();
        return b[2];
      }
      if (tag === "proj") {
        const ty = infer(ctx, t[1]), field = t[2];
        if (ty[0] === "and" && ["1", "left", "2", "right"].includes(field)) return field === "1" || field === "left" ? ty[1] : ty[2];
        if (ty[0] === "iff" && ["1", "mp", "2", "mpr"].includes(field)) {
          return field === "1" || field === "mp" ? ["imp", ty[1], ty[2]] : ["imp", ty[2], ty[1]];
        }
        return reject();
      }
      if (tag === "app") {
        let ty = infer(ctx, t[1]);
        for (const arg of t[2]) {
          if (ty[0] === "imp") {
            check(ctx, arg, ty[1]);
            ty = ty[2];
          } else if (ty[0] === "all") ty = open(ty[2], individual(ctx, arg));
          else reject();
        }
        return ty;
      }
      if (tag === "const") {
        const [, name, args] = t;
        if (name === "trivial") return TOP;
        if (name === "And.left" || name === "And.right") {
          const ty = infer(ctx, args[0]);
          if (ty[0] !== "and") reject();
          return name === "And.left" ? ty[1] : ty[2];
        }
        if (name === "And.intro") return ["and", infer(ctx, args[0]), infer(ctx, args[1])];
        if (name === "Iff.intro") {
          const f = infer(ctx, args[0]), g = infer(ctx, args[1]);
          if (f[0] !== "imp" || !eq(g, ["imp", f[2], f[1]])) reject();
          return ["iff", f[1], f[2]];
        }
      }
      return reject();
    }

    function verify(prompt, answer) {
      if (typeof answer !== "string" || answer.length > MAX_ANSWER) return false;
      const parsed = goalOf(prompt);
      if (parsed === null) return false;
      const [goal, individuals] = parsed;
      let ctx = null;
      for (const name of individuals) ctx = [name, IND, name, ctx];
      try {
        check(ctx, parseTerm(answer, individuals), unfold(goal));
        return true;
      } catch (e) {
        if (e instanceof Reject || e instanceof RangeError) return false;
        throw e;
      }
    }

    return { name: "fol", claims: (prompt) => goalOf(prompt) !== null, verify };
  })();

  // ------------------------------------------------------------------ nat
  // Port of formal/nat.py: closed Nat arithmetic with Lean's semantics
  // (truncated subtraction, floor division, n / 0 = 0, n % 0 = n).
  const Nat = (() => {
    const PROOFS = ["by decide", "rfl", "by rfl", "by simp"];
    const MAX_ANSWER = 400, MAX_VALUE = 10000, MAX_PROMPT = 200, MAX_NESTING = 32;
    const HEAD = "example : ", TAIL = " =";
    const apply = (op, a, b) => op === "+" ? a + b : op === "-" ? Math.max(a - b, 0) : op === "*" ? a * b
      : op === "/" ? (b ? Math.floor(a / b) : 0) : (b ? a % b : a);

    function lex(text) {
      const out = [];
      let i = 0;
      while (i < text.length) {
        const ch = text[i];
        if (ch === " ") i++;
        else if (ch >= "0" && ch <= "9") {
          let j = i;
          while (j < text.length && text[j] >= "0" && text[j] <= "9") j++;
          if (j - i > String(MAX_VALUE).length || (ch === "0" && j - i > 1)) reject();
          out.push(Number(text.slice(i, j)));
          i = j;
        } else if ("+-*/%()".includes(ch)) { out.push(ch); i++; }
        else reject();
      }
      return out;
    }

    // expr := term (("+" | "-") term)*, term := atom (("*" | "/" | "%") atom)*,
    // atom := numeral | "(" expr ")", evaluated as it parses (None past MAX_VALUE).
    function valueOf(prompt) {
      if (typeof prompt !== "string" || prompt.length > MAX_PROMPT || !prompt.startsWith(HEAD) || !prompt.endsWith(TAIL)) {
        return null;
      }
      let toks;
      try { toks = lex(prompt.slice(HEAD.length, prompt.length - TAIL.length)); } catch (e) { if (e instanceof Reject) return null; throw e; }
      let i = 0, tooBig = false;
      const peek = () => (i < toks.length ? toks[i] : null);
      const bound = (v) => { if (v > MAX_VALUE) tooBig = true; return v; };
      function expr(depth) {
        let v = term(depth);
        while (peek() === "+" || peek() === "-") { const op = toks[i++]; v = bound(apply(op, v, term(depth))); }
        return v;
      }
      function term(depth) {
        let v = atom(depth);
        while (peek() === "*" || peek() === "/" || peek() === "%") { const op = toks[i++]; v = bound(apply(op, v, atom(depth))); }
        return v;
      }
      function atom(depth) {
        const tok = peek();
        i++;
        if (typeof tok === "number") return bound(tok);
        if (tok === "(" && depth < MAX_NESTING) {
          const v = expr(depth + 1);
          if (peek() === ")") { i++; return v; }
        }
        return reject();
      }
      try {
        const v = expr(0);
        if (i !== toks.length) return null;
        return tooBig ? null : v;
      } catch (e) {
        if (e instanceof Reject) return null;
        throw e;
      }
    }

    function verify(prompt, answer) {
      if (typeof answer !== "string" || answer.length > MAX_ANSWER) return false;
      const value = valueOf(prompt);
      return value !== null && PROOFS.some((proof) => answer === `${value} := ${proof}`);
    }

    return { name: "nat", claims: (prompt) => valueOf(prompt) !== null, verify };
  })();

  // ------------------------------------------------------------------ linear
  // Port of formal/linear.py: one linear equation in x over Int, closed by omega.
  const Linear = (() => {
    const PROOFS = ["by omega", "by revert h; omega"];
    const MAX_ANSWER = 400, MAX_PROMPT = 200, MAX_NESTING = 32, MAX_MAGNITUDE = 1e6;
    const HEADER = /^example \(x : Int\) \(h : ([0-9x+*() -]+=[0-9x+*() -]+)\) : x =$/;

    function lex(text) {
      if (text.includes("--")) reject();  // a Lean comment, not `- -`
      const out = [];
      let i = 0;
      while (i < text.length) {
        const ch = text[i];
        if (ch === " ") i++;
        else if (ch >= "0" && ch <= "9") {
          let j = i;
          while (j < text.length && text[j] >= "0" && text[j] <= "9") j++;
          if (j - i > String(MAX_MAGNITUDE).length || (ch === "0" && j - i > 1)) reject();
          out.push(Number(text.slice(i, j)));
          i = j;
        } else if ("x+-*()".includes(ch)) { out.push(ch); i++; }
        else reject();
      }
      return out;
    }

    const bounded = (k, m, hasX) => {
      if (Math.abs(k) > MAX_MAGNITUDE || Math.abs(m) > MAX_MAGNITUDE) reject();
      return [k, m, hasX];
    };

    // Lean's grammar: expr := term (("+" | "-") term)*, term := neg ("*" neg)*,
    // neg := "-" neg | atom (prefix minus binds tighter than *), computing k*x + m.
    function parseLinear(text) {
      const toks = lex(text);
      let i = 0;
      const peek = () => (i < toks.length ? toks[i] : null);
      function expr(depth) {
        let [k, m, hasX] = term(depth);
        while (peek() === "+" || peek() === "-") {
          const sign = toks[i++] === "+" ? 1 : -1;
          const [k2, m2, x2] = term(depth);
          [k, m, hasX] = bounded(k + sign * k2, m + sign * m2, hasX || x2);
        }
        return [k, m, hasX];
      }
      function term(depth) {
        let [k, m, hasX] = neg(depth);
        while (peek() === "*") {
          i++;
          const [k2, m2, x2] = neg(depth);
          if (hasX && x2) reject();
          [k, m, hasX] = bounded(k * m2 + k2 * m, m * m2, hasX || x2);
        }
        return [k, m, hasX];
      }
      function neg(depth) {
        if (peek() !== "-") return atom(depth);
        if (depth >= MAX_NESTING) reject();
        i++;
        const [k, m, hasX] = neg(depth + 1);
        return [-k, -m, hasX];
      }
      function atom(depth) {
        const tok = peek();
        i++;
        if (typeof tok === "number") return bounded(0, tok, false);
        if (tok === "x") return [1, 0, true];
        if (tok === "(" && depth < MAX_NESTING) {
          const v = expr(depth + 1);
          if (peek() === ")") { i++; return v; }
        }
        return reject();
      }
      const v = expr(0);
      if (i !== toks.length) reject();
      return v;
    }

    function solutionOf(prompt) {
      if (typeof prompt !== "string" || prompt.length > MAX_PROMPT) return null;
      const m = HEADER.exec(prompt);
      if (!m) return null;
      const sides = m[1].split("=");
      if (sides.length !== 2) return null;
      let left, right;
      try {
        left = parseLinear(sides[0]);
        right = parseLinear(sides[1]);
      } catch (e) {
        if (e instanceof Reject || e instanceof RangeError) return null;
        throw e;
      }
      const [k1, m1] = left, [k2, m2] = right;
      if (k1 === k2 || (m2 - m1) % (k1 - k2) !== 0) return null;  // no unique integer solution
      return (m2 - m1) / (k1 - k2) + 0;  // + 0 turns -0 into 0
    }

    function verify(prompt, answer) {
      if (typeof answer !== "string" || answer.length > MAX_ANSWER) return false;
      const root = solutionOf(prompt);
      return root !== null && PROOFS.some((proof) => answer === `${root} := ${proof}`);
    }

    return { name: "linear", claims: (prompt) => solutionOf(prompt) !== null, verify };
  })();

  const DOMAINS = [Prop, Fol, Nat, Linear];  // formal/__init__.py's registry order

  function verify(prompt, answer) {
    for (const d of DOMAINS) if (d.claims(prompt)) return d.verify(prompt, answer);
    return null;
  }

  const api = { verify, domains: DOMAINS.map((d) => d.name) };
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.LeanCheck = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
