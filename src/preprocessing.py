"""
(B) Classical NLP - regex normalization of noisy chat text + tokenization.

The SAME ``normalize`` function is applied in front of every model (rule
baseline, TF-IDF, word2vec-FFN and BERT), both during training and inside the
app, so all models see identical input.

Normalization pipeline (order matters):
  1. Unicode NFKC + lowercase        "ＨＥＬＬＯ"        -> "hello"
  2. URLs           -> <url>         "http://t.co/x"     -> "<url>"
  3. @mentions      -> <user>        "@john"             -> "<user>"
     repeated <user> tokens are collapsed into one
  4. hashtags       -> word          "#idiot"            -> "idiot"
  5. censoring stars inside words    "f*ck"              -> "fck"   (edit distance fixes the rest)
  6. leetspeak                       "$h1t", "1d10t"     -> "shit", "idiot"
  7. repeated letters (3+ -> 2)      "stuuuupid"         -> "stuupid"
  8. strip emojis / symbols          "you 😡 !!"         -> "you !!"
  9. collapse whitespace

Run ``python -m src.preprocessing`` to see the steps on example messages.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Callable

USER_TOKEN = "<user>"
URL_TOKEN = "<url>"

# --- compiled regular expressions ------------------------------------------ #
_RE_URL = re.compile(r"https?://\S+|www\.\S+|\burl\b")
_RE_MENTION = re.compile(r"(?<!\w)@\w+")              # "@john" but not "b@stard"
_RE_MULTI_USER = re.compile(r"(?:<user>\s*){2,}")      # "<user> <user> <user>"
_RE_HASHTAG = re.compile(r"#(\w+)")
_CENSOR_WORDS = [
    (re.compile(r"\bf\*+c?k\b"), "fuck"),
    (re.compile(r"\bsh\*+t\b"), "shit"),
    (re.compile(r"\bb\*+tch\b"), "bitch"),
    (re.compile(r"\ba\*+s\b"), "ass"),
]
_RE_CENSOR = re.compile(r"(?<=[a-z])\*+(?=[a-z])")     # star(s) between letters
# A candidate leetspeak token is made of letters, digits, '@' and '$'.
_RE_LEET_TOKEN = re.compile(r"(?<![\w@$])[a-z0-9@$]+(?![\w@$])")
# ...but "number + letters" ("10am", "4th") and "letters + number" ("b4",
# "covid19") are normal words, so they are NOT translated. Pure numbers
# ("2020") contain no letter and are left alone too.
_RE_NOT_LEET = re.compile(r"\d+[a-z]+|[a-z]+\d+")
_LEET_CHARS = set("013457@$")
_LEET_TABLE = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a",
                             "5": "s", "7": "t", "@": "a", "$": "s"})
_RE_REPEAT = re.compile(r"(.)\1{2,}")                   # 3+ identical chars
_RE_ALLOWED = re.compile(r"[^a-z0-9<>'!?.,\s]")         # everything else -> space
_RE_SPACES = re.compile(r"\s+")
_RE_TOKEN = re.compile(r"<user>|<url>|[a-z0-9]+(?:'[a-z]+)?")


def _deleet(match: re.Match) -> str:
    """Translate a token only if it contains a letter AND a leetspeak char."""
    tok = match.group(0)
    if (any(c.isalpha() for c in tok)
            and any(c in _LEET_CHARS for c in tok)
            and not _RE_NOT_LEET.fullmatch(tok)):
        return tok.translate(_LEET_TABLE)
    return tok


def _uncensor(text: str) -> str:
    for pat, repl in _CENSOR_WORDS:
        text = pat.sub(repl, text)
    return _RE_CENSOR.sub("", text)


# Each step is (human-readable name, function). Keeping them in a list lets
# the app show every intermediate result (explain_normalization).
STEPS: list[tuple[str, Callable[[str], str]]] = [
    ("unicode + lowercase", lambda t: unicodedata.normalize("NFKC", t).lower()),
    ("urls -> <url>", lambda t: _RE_URL.sub(f" {URL_TOKEN} ", t)),
    ("mentions -> <user>", lambda t: _RE_MULTI_USER.sub(f"{USER_TOKEN} ",
                                                        _RE_MENTION.sub(f" {USER_TOKEN} ", t))),
    ("hashtags -> words", lambda t: _RE_HASHTAG.sub(r"\1", t)),
    ("remove censoring *", _uncensor),
    ("leetspeak -> letters", lambda t: _RE_LEET_TOKEN.sub(_deleet, t)),
    ("repeated letters", lambda t: _RE_REPEAT.sub(r"\1\1", t)),
    ("strip emojis/symbols", lambda t: _RE_ALLOWED.sub(" ", t)),
    ("collapse spaces", lambda t: _RE_SPACES.sub(" ", t).strip()),
]


def normalize(text: str) -> str:
    """Apply the full regex normalization pipeline to one message."""
    for _, step in STEPS:
        text = step(text)
    return text


def explain_normalization(text: str) -> list[tuple[str, str]]:
    """Return (step name, text after the step) for every step - used in the app."""
    out = []
    for name, step in STEPS:
        text = step(text)
        out.append((name, text))
    return out


def tokenize(text: str) -> list[str]:
    """Simple regex word tokenizer for already-normalized text."""
    return _RE_TOKEN.findall(text)


if __name__ == "__main__":
    examples = [
        "@USER @USER @USER You are such an 1d10t!!! #loser http://t.co/abc",
        "stuuuuupid $h1t, f*ck off 😡😡",
        "Meet me at 10am on the 4th, see you!",
    ]
    for ex in examples:
        print("RAW :", ex)
        for name, res in explain_normalization(ex):
            print(f"  {name:<22} {res}")
        print("TOKENS:", tokenize(normalize(ex)), "\n")
