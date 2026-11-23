"""Convert Grace / xmgrace escape sequences to publication-grade Unicode / Matplotlib mathtext."""

from __future__ import annotations

import re

# Standard PostScript Symbol font character mapping used by GROMACS / xmgrace
_SYMBOL_MAP: dict[str, str] = {
    "A": "Α", "B": "Β", "G": "Γ", "D": "Δ", "E": "Ε", "Z": "Ζ", "H": "Η",
    "Q": "Θ", "I": "Ι", "K": "Κ", "L": "Λ", "M": "Μ", "N": "Ν", "X": "Ξ",
    "O": "Ο", "P": "Π", "R": "Ρ", "S": "Σ", "T": "Τ", "U": "Υ", "F": "Φ",
    "C": "Χ", "Y": "Ψ", "W": "Ω",
    "a": "α", "b": "β", "g": "γ", "d": "δ", "e": "ε", "z": "ζ", "h": "η",
    "q": "θ", "i": "ι", "k": "κ", "l": "λ", "m": "μ", "n": "ν", "x": "ξ",
    "o": "ο", "p": "π", "r": "ρ", "s": "σ", "t": "τ", "u": "υ", "f": "φ",
    "c": "χ", "y": "ψ", "w": "ω",
    "j": "φ", "v": "ϖ", "V": "ς",
    "2": "′", "3": "≤", "4": "⁄", "5": "∞", "6": "ƒ",
    "±": "±", "+": "+", "-": "−", "=": "=",
}

_SUPERSCRIPT_CHARS: dict[str, str] = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽", ")": "⁾",
    "n": "ⁿ", "i": "ⁱ", "a": "ᵃ", "b": "ᵇ", "c": "ᶜ",
    "d": "ᵈ", "e": "ᵉ", "g": "ᵍ", "h": "ʰ", "k": "ᵏ",
    "m": "ᵐ", "o": "ᵒ", "p": "ᵖ", "r": "ʳ", "s": "ˢ",
    "t": "ᵗ", "x": "ˣ", "y": "ʸ", "z": "ᶻ",
}

_SUBSCRIPT_CHARS: dict[str, str] = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "=": "₌", "(": "₍", ")": "₎",
    "a": "ₐ", "e": "ₑ", "h": "ₕ", "i": "ᵢ", "j": "ⱼ",
    "k": "ₖ", "l": "ₗ", "m": "ₘ", "n": "ₙ", "o": "ₒ",
    "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ", "u": "ᵤ",
    "v": "ᵥ", "x": "ₓ", "g": "g",
}

_SPECIAL_CHAR_MAP: dict[str, str] = {
    # Grace \c<char>\C extended ASCII (ISO-8859-1 + 128)
    "E": "Å",  # 0x45 + 0x80 = 0xC5 (Å)
    "0": "°",  # 0x30 + 0x80 = 0xB0 (°)
    "1": "±",  # 0x31 + 0x80 = 0xB1 (±)
    "2": "²",  # 0x32 + 0x80 = 0xB2 (²)
    "3": "³",  # 0x33 + 0x80 = 0xB3 (³)
    "5": "μ",  # 0x35 + 0x80 = 0xB5 (µ)
    "W": "×",  # 0x57 + 0x80 = 0xD7 (×)
}


def _to_symbol(text: str) -> str:
    return "".join(_SYMBOL_MAP.get(ch, ch) for ch in text)


def _format_script(text: str, is_super: bool, use_mathtext: bool) -> str:
    """Render subscript or superscript cleanly for Matplotlib."""
    if not text:
        return ""
    table = _SUPERSCRIPT_CHARS if is_super else _SUBSCRIPT_CHARS
    if all(ch in table for ch in text) and (is_super or "g" not in text):
        return "".join(table[ch] for ch in text)
    if use_mathtext:
        op = "^" if is_super else "_"
        safe = text.replace("{", "").replace("}", "").replace(" ", r"\ ")
        return f"${{{op}{{\\mathrm{{{safe}}}}}}}$"
    return f"^{text}" if is_super else f"_{text}"


def grace_to_matplotlib(text: str, use_mathtext: bool = True) -> str:
    """Translate Grace/xmgrace formatting escapes into clean Matplotlib labels.

    Handles common GROMACS Grace tags:
    - ``\\s...\\N`` (subscript) and ``\\S...\\N`` (superscript)
    - ``\\f{Symbol}...\\f{}`` and ``\\x...\\f{}`` (Greek / Symbol font)
    - ``\\cE\\C`` (Ångström), ``\\c0\\C`` (degree), ``\\c1\\C`` (±)
    - ``\\-``, ``\\+``, ``\\N`` (normal font reset)
    - ``\\q`` / ``\\Q`` (oblique/normal)
    """
    if not text:
        return ""

    s = text.strip()
    # Unescape doubled backslashes occasionally emitted by GROMACS tools
    s = s.replace("\\\\", "\\")

    # Handle \c<X>\C extended Latin-1 escapes (e.g. \cE\C -> Å)
    def _repl_c(match: re.Match[str]) -> str:
        code = match.group(1)
        if code in _SPECIAL_CHAR_MAP:
            return _SPECIAL_CHAR_MAP[code]
        if len(code) == 1:
            return chr((ord(code) + 128) & 0xFF)
        return code

    s = re.sub(r"\\c(.)\\C", _repl_c, s)

    # Handle \f{Symbol}...\f{} or \f{Symbol}...\f{...} or \f{12}...\f{}
    def _repl_sym(match: re.Match[str]) -> str:
        return _to_symbol(match.group(1))

    s = re.sub(r"\\f\{(?:Symbol|12)\}(.*?)(?:\\f\{[^}]*\}|$)", _repl_sym, s)

    # Handle shorthand \x...\f{} (xmgrace font 12 = Symbol)
    s = re.sub(r"\\x(.*?)(?:\\f\{[^}]*\}|\\N|$)", _repl_sym, s)

    # Handle explicit GROMACS Rg pattern like R\sg\N or R\s\f{...}g\N
    # Strip any remaining font switches like \f{Helvetica} or \f{}
    s = re.sub(r"\\f\{[^}]*\}", "", s)

    # Handle superscripts \S...\N (or up to next \s, \S, or end of string)
    def _repl_sup(match: re.Match[str]) -> str:
        return _format_script(match.group(1), is_super=True, use_mathtext=use_mathtext)

    def _repl_sub(match: re.Match[str]) -> str:
        return _format_script(match.group(1), is_super=False, use_mathtext=use_mathtext)

    # Process paired \S...\N and \s...\N
    s = re.sub(r"\\S([^\\]+?)\\N", _repl_sup, s)
    s = re.sub(r"\\s([^\\]+?)\\N", _repl_sub, s)
    # Process unterminated trailing \S... or \s...
    s = re.sub(r"\\S([^\s\\]+)", _repl_sup, s)
    s = re.sub(r"\\s([^\s\\]+)", _repl_sub, s)

    # Clean up remaining Grace control tokens
    s = re.sub(r"\\[NnqQbBuUoO+-]", "", s)

    # Normalize common GROMACS notation like "nm\S2\N" already turned to "nm²",
    # "kJ mol⁻¹", "R_g"
    s = s.replace("R\\sg", "R$_g$")
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def strip_grace_to_plain(text: str) -> str:
    """Convert Grace text to plain Unicode string without Matplotlib $...$ mathtext."""
    return grace_to_matplotlib(text, use_mathtext=False)
