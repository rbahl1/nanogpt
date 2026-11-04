"""Character-level tokenizer over printable ASCII plus the Lean 4 symbols the
domains use.

Four special tokens come first: <pad> pads batches, <bos> starts every
sequence, <ans> separates a declaration's header from the text the model
writes, and <eos> ends a sequence. Task sequences are

    <bos> header <ans> answer <eos>

and pretraining documents are `<bos> declaration <eos>`.
"""

SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<ans>")
PAD, BOS, EOS, ANS = range(len(SPECIAL_TOKENS))

# Order matters: token ids are fixed by it (and the web page reads it from the
# exported model). Printable ASCII, newline, then the Lean symbols.
ALPHABET = "\n" + "".join(chr(c) for c in range(32, 127)) + "¬∧∨→↔⟨⟩∀∃αβγ≠≤≥ℕℤ·▸←⊢"


class Tokenizer:
    def __init__(self, alphabet: str = ALPHABET):
        if len(set(alphabet)) != len(alphabet):
            raise ValueError("the alphabet has repeated characters")
        self.alphabet = alphabet
        self.stoi = {ch: i + len(SPECIAL_TOKENS) for i, ch in enumerate(alphabet)}
        self.vocab_size = len(SPECIAL_TOKENS) + len(alphabet)

    def encode(self, text: str) -> list:
        """Token ids for `text`; raises ValueError naming any character outside
        the alphabet (they are never dropped silently)."""
        ids, unknown = [], []
        for ch in text:
            i = self.stoi.get(ch)
            if i is None:
                if ch not in unknown:
                    unknown.append(ch)
            else:
                ids.append(i)
        if unknown:
            raise ValueError("outside the model's alphabet: " + " ".join(repr(c) for c in unknown))
        return ids

    def decode(self, ids) -> str:
        """Text for `ids`, with special tokens dropped."""
        n = len(SPECIAL_TOKENS)
        return "".join(self.alphabet[i - n] for i in ids if i >= n)

    def prompt_ids(self, prompt: str) -> list:
        return [BOS] + self.encode(prompt) + [ANS]

    def example_ids(self, prompt: str, answer: str):
        """(ids, n_prompt): the full task sequence and how many of its tokens
        (<bos> prompt <ans>) are prompt; the loss covers the rest."""
        head = self.prompt_ids(prompt)
        return head + self.encode(answer) + [EOS], len(head)

    def document_ids(self, document: str) -> list:
        return [BOS] + self.encode(document) + [EOS]
