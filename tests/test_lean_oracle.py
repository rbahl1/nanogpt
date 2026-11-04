"""lean_oracle.check_declarations against hand-picked declarations."""

import lean_oracle


def test_verdicts(lean):
    good = "example : 2 + 2 = 4 := by decide"
    cases = [
        (good, True),
        ("example : 2 + 2 = 5 := by decide", False),
        ("example (p : Prop) : p → p := fun h => h", True),
        ("example (p : Prop) : p → p := sorry", False),               # sorry fails
        ("example (p q : Prop) (hp : p) : p := hp", True),            # unused-variable lint is not a failure
        ("example (p : Prop) : p := by", False),                       # unfinished: blamed on itself
        ("#eval 1", False),                                            # not a plain example: never compiled
        ("example : True := trivial -- comment", False),               # comments are refused unread
    ]
    assert lean_oracle.check_declarations([d for d, _ in cases], lean) == [ok for _, ok in cases]


def test_errors_lean_places_at_the_start_of_the_file(lean):
    """Lean reports the "unsolved goals" of a `by` block broken by a lexer error
    at line 1, column 0; it must be blamed on that declaration, not the first."""
    good = "example : 2 + 2 = 4 := by decide"
    bad = "example (x : Int) (h : 8 * (x - 12) = -96) : x = 0 := by 0evert h; omega"
    assert lean_oracle.check_declarations([good, good, bad, good, good], lean) == [True, True, False, True, True]
    assert lean_oracle.check_declarations([bad], lean) == [False]
    assert lean_oracle.check_declarations([bad, bad, good], lean) == [False, False, True]
