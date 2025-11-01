# src/tokenizer.py
"""
Simple printable-ASCII tokenizer including the <PATCH> token.

Vocabulary:
 - printable ASCII characters from codepoint 32..126 (inclusive)
 - special tokens: <PAD>, <SOS>, <EOS>, <UNK>, <PATCH>

API:
 - Tokenizer.encode(text, add_sos=True, add_eos=True, strict=True, max_len=None)
 - Tokenizer.decode(ids, strip_specials=True)
 - Tokenizer.save(path) / Tokenizer.load(path)
 - Tokenizer.to_dict() / Tokenizer.from_dict(d)
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional
import json
import os


PRINTABLE_ASCII_MIN = 32
PRINTABLE_ASCII_MAX = 126


class Tokenizer:
    def __init__(self, min_char: int = PRINTABLE_ASCII_MIN, max_char: int = PRINTABLE_ASCII_MAX):
        """
        Build tokenizer for printable ASCII range [min_char, max_char].
        Adds special tokens: <PAD>, <SOS>, <EOS>, <UNK>, <PATCH>.

        Mappings:
           char -> id: 0 .. (V-1) (for printable chars)
           pad_id = V
           sos_id = V+1
           eos_id = V+2
           unk_id = V+3
           patch_id = V+4
        """
        if min_char < 0 or max_char < min_char:
            raise ValueError("invalid char range")

        self.min_char = min_char
        self.max_char = max_char

        # build base vocab (chars)
        self._id_to_token: List[str] = []
        self._token_to_id: Dict[str, int] = {}

        # printable ascii characters
        for cp in range(self.min_char, self.max_char + 1):
            ch = chr(cp)
            self._token_to_id[ch] = len(self._id_to_token)
            self._id_to_token.append(ch)

        # special tokens
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        self.patch_token = "<PATCH>"

        for tok in (self.pad_token, self.sos_token, self.eos_token, self.unk_token, self.patch_token):
            self._token_to_id[tok] = len(self._id_to_token)
            self._id_to_token.append(tok)

        # cached ids
        self.pad_id = self._token_to_id[self.pad_token]
        self.sos_id = self._token_to_id[self.sos_token]
        self.eos_id = self._token_to_id[self.eos_token]
        self.unk_id = self._token_to_id[self.unk_token]
        self.patch_id = self._token_to_id[self.patch_token]

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    def encode(
        self,
        text: str,
        add_sos: bool = True,
        add_eos: bool = True,
        strict: bool = True,
        max_len: Optional[int] = None
    ) -> List[int]:
        """
        Convert string -> list of ids.

        Non-printable characters:
          - strict=True -> raise ValueError
          - strict=False -> map to <UNK>

        NOTE: <PATCH> token is not automatically inserted by encode();
        it's available for model code to insert if needed.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a str")

        ids: List[int] = []

        if add_sos:
            ids.append(self.sos_id)

        core_ids: List[int] = []
        for ch in text:
            code = ord(ch)
            if self.min_char <= code <= self.max_char:
                core_ids.append(self._token_to_id[ch])
            else:
                if strict:
                    raise ValueError(f"Non-printable ASCII character encountered: {repr(ch)} (ord={code})")
                else:
                    core_ids.append(self.unk_id)

            if max_len is not None and len(core_ids) >= max_len:
                break

        ids.extend(core_ids)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids: List[int], strip_specials: bool = True) -> str:
        """
        Convert list of ids -> string.

        If strip_specials=True, removes PAD/SOS/EOS/UNK/PATCH tokens from output;
        otherwise special tokens are emitted as their literal names.
        """
        if not isinstance(ids, (list, tuple)):
            raise TypeError("ids must be list or tuple of ints")

        chars: List[str] = []
        for idx in ids:
            if idx < 0 or idx >= self.vocab_size:
                if strip_specials:
                    chars.append("?")
                else:
                    chars.append("<OOR>")
                continue
            tok = self._id_to_token[idx]
            if tok in (self.pad_token, self.sos_token, self.eos_token, self.unk_token, self.patch_token):
                if strip_specials:
                    continue
                else:
                    chars.append(tok)
            else:
                chars.append(tok)
        return "".join(chars)

    def save(self, path: str) -> None:
        try:
            d = self.to_dict()
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
        except (IOError, OSError) as e:
            raise ValueError(f"Error saving tokenizer to {path}: {e}")

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            return cls.from_dict(d)
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Error loading tokenizer from {path}: {e}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "min_char": self.min_char,
            "max_char": self.max_char,
            "id_to_token": list(self._id_to_token)
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Tokenizer":
        if "id_to_token" not in d:
            raise ValueError("dict must contain 'id_to_token' key")
        id_to_token = d["id_to_token"]
        if not isinstance(id_to_token, list):
            raise TypeError("id_to_token must be a list")

        # Create blank tokenizer then override mapping to preserve exact ordering
        t = cls()
        t._id_to_token = list(id_to_token)
        t._token_to_id = {tok: i for i, tok in enumerate(t._id_to_token)}
        # reassign special ids (if present)
        t.pad_id = t._token_to_id.get(t.pad_token)
        t.sos_id = t._token_to_id.get(t.sos_token)
        t.eos_id = t._token_to_id.get(t.eos_token)
        t.unk_id = t._token_to_id.get(t.unk_token)
        t.patch_id = t._token_to_id.get(t.patch_token)
        return t


# -------------------------
# Small sanity / demo test
# -------------------------
def _basic_tests():
    print("Running Tokenizer basic tests...")

    tok = Tokenizer()
    s = "Hello, world!"
    enc = tok.encode(s, add_sos=True, add_eos=True)
    dec = tok.decode(enc, strip_specials=True)
    assert dec == s, f"Roundtrip failed: got {dec!r} expected {s!r}"

    enc_no_specials = tok.encode(s, add_sos=False, add_eos=False)
    assert enc_no_specials[0] != tok.sos_id, "Unexpected sos in encoding"

    # non-printable handling (strict -> raise)
    try:
        tok.encode("Hello\nWorld", strict=True)
        raise AssertionError("Expected ValueError for non-printable char in strict mode")
    except ValueError:
        pass

    # non-strict maps to UNK
    enc2 = tok.encode("Hi\u2603", strict=False, add_sos=False, add_eos=False)
    assert enc2[-1] == tok.unk_id

    # patch token exists
    assert "<PATCH>" in tok._token_to_id
    assert tok.patch_id == tok._token_to_id["<PATCH>"]

    # save/load roundtrip
    tmp_path = "tmp_tokenizer.json"
    tok.save(tmp_path)
    tok2 = Tokenizer.load(tmp_path)
    os.remove(tmp_path)
    assert tok.encode("abc", add_sos=False, add_eos=False) == tok2.encode("abc", add_sos=False, add_eos=False)

    print("Tokenizer basic tests passed.")


if __name__ == "__main__":
    _basic_tests()
    print("Tokenizer module self-test completed.")
