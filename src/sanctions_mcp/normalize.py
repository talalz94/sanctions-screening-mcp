"""Name normalisation for screening, with Arabic script and Arabic name structure handled.

A name becomes a list of tokens. Each token keeps:

    folded     lowercase ASCII with common spellings mapped to one form ("mohammed" -> "muhammad")
    skeletons  consonant keys: vowels dropped, sounds that are spelled several ways merged
               ("Hussein", "Husayn" and Arabic script "حسين" all give "hsn")
    shapes     for Latin words, the letters behind each key with every run of vowels kept as
               one slot ("karim" -> "kVrVm"), so Karim and Akram share a key but not a shape
    initials   every letter the word could be shortened to ("Osama" -> "o" or "u")

The skeleton is what lets a Latin name meet the same name written in Arabic script, where
short vowels are not written at all. Some spellings can be read two ways, so a word can carry
more than one key:

    Latin th, sh, kh, dh   one Arabic letter (Muthanna, ث) or two (Fathi, ت then ح)
    Latin dj               the j of Djamel, or the d with stroke of Djordjevic (Dordevic)
    Arabic ض               "d" in Arabic romanisation (Ramadan), "z" in Persian and Urdu (Ramzan)
    a glued Latin "al"     the article (Alhashimi) or part of the name (Alberto)
    a leading ال           the article (الرحمن) or part of the name (الياس, Ilyas)

Latin text has accents removed and letters with no accent to strip (ı ł đ ø æ ə ß) spelled
out; Cyrillic is transliterated; a name that mixes scripts is read one script run at a time.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache

from rapidfuzz.distance import Levenshtein

# Arabic, Persian and Urdu letters, readable form. Used for the folded token and for display.
ARABIC_READABLE = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "a", "ٱ": "a", "ء": "",
    "ؤ": "u", "ئ": "i", "ب": "b", "ت": "t", "ث": "th", "ج": "j",
    "ح": "h", "خ": "kh", "د": "d", "ذ": "dh", "ر": "r", "ز": "z",
    "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "t", "ظ": "z",
    "ع": "a", "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "ل": "l",
    "م": "m", "ن": "n", "ه": "h", "ة": "a", "و": "w", "ي": "y",
    "ى": "a", "پ": "p", "چ": "ch", "ژ": "zh", "گ": "g", "ک": "k",
    "ی": "y", "ے": "e", "ھ": "h", "ہ": "h", "ٹ": "t", "ڈ": "d",
    "ڑ": "r", "ں": "n", "ۃ": "a",
}  # fmt: skip

# The same letters mapped straight to skeleton letters, one letter each, so two Arabic
# letters can never fuse into a Latin digraph by accident (ت + ح is not "th").
ARABIC_SKELETON = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "a", "ٱ": "a", "ء": "",
    "ؤ": "u", "ئ": "i", "ب": "b", "ت": "t", "ث": "t", "ج": "j",
    "ح": "h", "خ": "h", "د": "d", "ذ": "z", "ر": "r", "ز": "z",
    "س": "s", "ش": "s", "ص": "s", "ض": "d", "ط": "t", "ظ": "z",
    "ع": "a", "غ": "j", "ف": "f", "ق": "k", "ك": "k", "ل": "l",
    "م": "m", "ن": "n", "ه": "h", "ة": "a", "و": "w", "ي": "y",
    "ى": "a", "پ": "p", "چ": "s", "ژ": "j", "گ": "j", "ک": "k",
    "ی": "y", "ے": "e", "ھ": "h", "ہ": "h", "ٹ": "t", "ڈ": "d",
    "ڑ": "r", "ں": "n", "ۃ": "a",
}  # fmt: skip
# Second readings, added as a second key: ض is "z" in Persian, Urdu and Turkish names.
ARABIC_SKELETON_ALT = {"ض": "z"}
_ARABIC_SKELETON_WITH_ALT = ARABIC_SKELETON | ARABIC_SKELETON_ALT

# The letters a Latin initial can stand for, by the first Arabic letter of a word.
_ARABIC_INITIALS = {
    "ا": "aeiou", "أ": "aeiou", "إ": "aeiou", "آ": "aeiou", "ٱ": "aeiou", "ع": "aeiou",
    "و": "ouvw", "ي": "eiy", "ی": "eiy", "ج": "dgj", "ق": "gkq", "خ": "hk", "ث": "st",
    "ذ": "dz", "ض": "dz", "ظ": "dz", "ش": "cs", "غ": "g", "چ": "c", "گ": "g", "ژ": "jz",
}  # fmt: skip

_ARABIC_MARKS = re.compile("[ً-ٰٟۖ-ۭـ]")  # short vowels, tatweel
_ARABIC_BLOCKS = re.compile("[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻾]")
# Format characters that carry no letters: zero width spaces and joiners, direction marks
# (including the Arabic letter mark), word joiners, soft hyphens and the byte order mark.
_INVISIBLE = re.compile("[­؜᠎​-‏‪-‮⁠-⁤⁦-⁩﻿]")

CYRILLIC_TO_LATIN = {
    # Russian, Ukrainian and Belarusian
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z",
    "и": "i", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh",
    "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya", "і": "i",
    "ї": "yi", "є": "ye", "ґ": "g", "ў": "u",
    # Serbian and Macedonian, spelled so they meet the Latin forms (ћ like ć, џ like dž)
    "ђ": "dj", "ј": "j", "љ": "lj", "њ": "nj", "ћ": "c", "џ": "dz", "ѓ": "g", "ќ": "k",
    "ѕ": "dz",
    # Kazakh, Kyrgyz, Tajik, Uzbek, Azerbaijani, Tatar, Bashkir and Chechen
    "ә": "a", "ғ": "gh", "қ": "k", "ң": "ng", "ө": "o", "ұ": "u", "ү": "u", "һ": "h",
    "ҳ": "kh", "ҷ": "j", "җ": "zh", "ҹ": "j", "ҝ": "g", "ӏ": "",
}  # fmt: skip
_HAS_CYRILLIC = re.compile("[Ѐ-ԯ]")

# Letters with no accent to strip. English language lists write them like this.
_LATIN_FOLD = str.maketrans(
    {
        "ı": "i", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ð": "d", "Ð": "D", "ø": "o",
        "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "þ": "th", "Þ": "TH", "ħ": "h",
        "Ħ": "H", "ə": "a", "Ə": "A", "ß": "ss", "ẞ": "SS", "ŋ": "ng", "Ŋ": "NG", "ŧ": "t",
        "Ŧ": "T", "ſ": "s",
    }
)  # fmt: skip
# Apostrophes, ayn and hamza marks (ʿ ʾ), prime and soft sign marks (ʹ ʺ), and accents typed on
# their own. All are removed so "Sa'id", "Saʿid" and "Sa´id" stay one word.
_APOSTROPHES = re.compile("['‘’‛ʹʺʻʼʽʾʿˈ´`′＇｀]")
_DASHES = re.compile("[‐-―−﹘﹣－]")
# "ad-Din", "as-Salam", "ash-Shami": the article written as it sounds before a sun letter.
_ASSIMILATED_HYPHEN = re.compile(r"\b[aeiu](sh|th|dh|zh|[tdrszn])-(?=\1)")
_SON_OF = re.compile(r"\b[sdw]\s*/\s*o\b", re.IGNORECASE)  # South Asian "s/o", "d/o", "w/o"

TITLES = frozenset(
    [
        "sheikh",
        "shaikh",
        "shaykh",
        "sheik",
        "haji",
        "hajji",
        "hajj",
        "alhaj",
        "alhajj",
        "mullah",
        "mulla",
        "maulana",
        "mawlana",
        "maulvi",
        "molvi",
        "moulvi",
        "mufti",
        "qari",
        "dr",
        "doctor",
        "prof",
        "professor",
        "mr",
        "mrs",
        "ms",
        "miss",
        "general",
        "major",
        "colonel",
        "lieutenant",
        "brigadier",
        "captain",
        "commander",
        "admiral",
        "sergeant",
        "marshal",
    ]
)
# Titles that are also given names. Dropped only in front of at least two name words, so
# "Hafiz Muhammad Saeed" loses the title but "Hafiz Saeed" and "Abdul Hafiz" keep the name.
FIRST_WORD_TITLES = frozenset(["hafiz"])
FRENCH_TITLE_WORDS = frozenset(["general", "de", "brigade", "division", "corps", "armee"])
CONNECTORS = frozenset(["bin", "ibn", "bint", "binti", "ben", "ould", "walad"])
ARTICLES = frozenset(["al", "el", "ul", "il"])
# The article spelled as it sounds before a sun letter. Only the forms starting with "u" are
# dropped wherever they stand, because "Ed", "An" or "Er" can be names in their own right.
_SUN_SOUNDS = frozenset(["t", "th", "d", "dh", "r", "z", "zh", "s", "sh", "n"])
_U_ARTICLES = frozenset("u" + s for s in _SUN_SOUNDS)
_DIN_ARTICLES = frozenset(["ad", "ed", "id", "ud"])
_DIN_WORDS = frozenset(["din", "deen", "dine"])
ABD_FORMS = frozenset(
    ["abd", "abdul", "abdel", "abdal", "abdol", "abdu", "abdur", "abdo", "abdoul", "abdil", "abdi"]
)
# Names that follow "Abd" (servant of ...), used to split glued forms like "abdelaziz": the
# divine names, plus the names Shia compounds use (Abd al-Husayn, Abd al-Zahra, Abd al-Nabi).
DIVINE_NAMES = frozenset(
    [
        "rahman",
        "rahim",
        "malik",
        "quddus",
        "salam",
        "mumin",
        "muhaymin",
        "aziz",
        "jabbar",
        "karim",
        "ghaffar",
        "qahhar",
        "wahhab",
        "wahab",
        "vahab",
        "vahap",
        "wahap",
        "razzaq",
        "razaq",
        "fattah",
        "alim",
        "basit",
        "hafiz",
        "rafi",
        "muizz",
        "samii",
        "basir",
        "hakam",
        "adl",
        "latif",
        "khabir",
        "halim",
        "azim",
        "ghafur",
        "shakur",
        "ali",
        "aali",
        "kabir",
        "hafeez",
        "muqit",
        "hasib",
        "jalil",
        "raqib",
        "mujib",
        "wasi",
        "hakim",
        "wadud",
        "majid",
        "shahid",
        "haqq",
        "haq",
        "wakil",
        "qawi",
        "matin",
        "wali",
        "hamid",
        "muhsi",
        "muhyi",
        "hayy",
        "hai",
        "hay",
        "qayyum",
        "wajid",
        "wahid",
        "ahad",
        "samad",
        "qadir",
        "qader",
        "kadir",
        "muqtadir",
        "awwal",
        "zahir",
        "batin",
        "barr",
        "tawwab",
        "rauf",
        "ghani",
        "mughni",
        "nur",
        "hadi",
        "badi",
        "baqi",
        "rashid",
        "sabur",
        "salaam",
        "jawad",
        "nasir",
        "nasser",
        "muttalib",
        "mutalib",
        "ilah",
        "majeed",
        "kareem",
        "raheem",
        "lateef",
        "azeez",
        "haleem",
        "hameed",
        "qadeer",
        "sattar",
        "khaliq",
        "khaleq",
        "mannan",
        "hannan",
        "muin",
        "muhsin",
        "husain",
        "hussein",
        "husayn",
        "hasan",
        "nabi",
        "rasul",
        "rasool",
        "amir",
        "mahdi",
        "zahra",
        "masih",
        "reza",
        "rida",
    ]
)
ULLAH = ("ullah", "ulla", "allah", "ollah")
DIN_SUFFIXES = (
    "uddin", "uddeen", "eddine", "eddin", "addin", "iddin", "oddin", "aldin", "uldin", "udin",
    "edine", "edin",
)  # fmt: skip

# Arabic script words handled before transliteration, compared in the normalised spelling of
# `_ar_norm` (one alef, one yeh, one kaf, ta marbuta as ha).
AR_ARTICLE = "ال"
AR_ALLAH = "الله"
AR_ABD = "عبد"
AR_DIN = "الدين"
AR_HAFIZ = "حافظ"
AR_CONNECTORS = frozenset(["بن", "ابن", "بنت", "ولد"])
AR_FAMILY = frozenset(["ال"])  # آل (Al Saud, Al Thani) once normalised, or a lone ال
AR_ALLAH_NORM = "الله"  # already in _ar_norm form
AR_DIN_NORM = "الدين"
AR_TITLES = frozenset(
    ["شيخ", "حاج", "حاجي", "ملا", "مولانا", "مولوي", "مفتي", "قاري", "دكتور", "دكتر"]
)
_AR_NORM = str.maketrans(
    {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ی": "ي", "ک": "ك", "ة": "ه", "ۃ": "ه",
     "ہ": "ه", "ھ": "ه"}
)  # fmt: skip

# Spellings of frequent given names that a consonant key alone does not unite.
VARIANT_GROUPS: dict[str, tuple[str, ...]] = {
    "muhammad": tuple(
        [
            "mohammad",
            "mohammed",
            "mohamed",
            "mohamad",
            "muhammed",
            "muhamad",
            "muhammet",
            "mohmad",
            "muhamed",
            "mohmed",
            "mehmet",
            "mehmed",
            "mohd",
            "muhammadu",
            "mouhamed",
            "mouhammad",
            "mahomed",
        ]
    ),
    "ahmad": ("ahmed", "ahmet", "ahamed", "ahmd", "ahmadu"),
    "mahmud": ("mahmoud", "mahmood", "mahmut", "mehmood"),
    "husain": tuple(
        [
            "hussein",
            "hussain",
            "husein",
            "hosein",
            "hossein",
            "husayn",
            "hussin",
            "houssein",
            "huseyin",
            "hossain",
        ]
    ),
    "hasan": ("hassan", "hassen", "hasen", "hasane", "hasson"),
    "yusuf": tuple(
        ["yousef", "youssef", "yousuf", "yusef", "yousif", "yusif", "youssouf", "yousaf", "yusof"]
    ),
    "umar": ("omar", "omer", "oumar", "umer"),
    "uthman": ("othman", "osman", "usman", "othmane", "outhman", "usmon"),
    "usama": ("osama", "oussama", "usamah", "osamah"),
    "khalid": ("khaled", "halid", "kaled"),
    "salih": ("saleh",),
    "said": ("saeed", "saied", "saeid", "sayeed"),
    "sayyid": ("sayed", "sayid", "syed", "seyed", "seyyed", "sayyed"),
    "abdullah": tuple(
        [
            "abdallah",
            "abdulla",
            "abdalla",
            "abdollah",
            "abdulah",
            "abdellah",
            "abdoullah",
            "abdullahi",
            "abdulahi",
            "abdellahi",
            "abdallahi",
            "abdillahi",
            "abdollahi",
            "abdoullahi",
            "abdoulahi",
            "abdoulaye",
            "abdoulay",
            "abdulaye",
            "abdulai",
            "abdulle",
        ]
    ),
    "ibrahim": ("ebrahim", "ibraheem", "ebrahem", "ibragim", "ibraim", "brahim"),
    "ismail": ("ismael", "esmail", "ismayil", "ismaeel"),
    "yahya": ("yahia", "yehia", "yahiya", "yehya"),
    "mustafa": ("mostafa", "moustafa", "mustapha", "mostapha", "mustafah"),
    "jamal": ("gamal", "djamel", "jamel", "djamal"),
    "jamil": ("jameel", "djamil", "gamil"),
    "mahdi": ("mehdi",),
    "jafar": ("jaafar", "jafer", "jaffar", "djafar"),
    "murtada": ("mortada", "murtaza", "mortaza", "murtadha"),
    "muhsin": ("mohsen", "mohsin", "muhsen"),
    "amin": ("ameen", "emin"),
    "amir": ("ameer", "emir"),
    "rashid": ("rasheed", "rachid"),
    "nasir": ("nasser", "naser", "nassir", "naseer"),
    "faisal": ("faysal", "feisal", "fayssal"),
    "tariq": ("tarek", "tareq", "tarik"),
    "zaid": ("zayd", "zeid", "zayed"),
    "ayman": ("aiman", "eiman"),
    "sulaiman": ("suleiman", "sulayman", "suleyman", "soliman", "solaiman", "sulaymon"),
    "fathi": ("fethi", "fatehi"),
    "rajab": ("ragab", "radjab"),
    "hamza": ("hamzah", "hamzeh"),
    "hamid": ("hamed", "hameed"),
    "karim": ("kareem", "kerim"),
    "hashim": ("hashem", "hachem"),
    "hisham": ("hesham", "hicham"),
    "bashir": ("basheer", "bachir", "beshir"),
    "faris": ("fares",),
    "samir": ("sameer",),
    "sharif": ("sherif", "cherif", "shareef"),
    "abbas": ("abas",),
    "habib": ("habeeb",),
    "majid": ("majeed", "maged"),
    "walid": ("waleed", "oualid"),
    "jasim": ("jassim", "jasem", "jassem", "gassim"),
    "salim": ("saleem", "salem", "selim"),
    "ali": ("aly",),
    "abu": ("abou", "abo"),
    "isa": ("issa", "eissa", "esa"),
    "musa": ("moussa", "mousa", "mussa"),
}
CANONICAL: dict[str, str] = {
    variant: canon for canon, variants in VARIANT_GROUPS.items() for variant in variants
}
# Different names that share a consonant key, and often a vowel pattern too (Hasan and Husain,
# Hamid and Hamad). Two of these never count as the same name on the strength of a key alone.
KNOWN_GIVEN_NAMES = frozenset(VARIANT_GROUPS) | frozenset(
    [
        "hamad",
        "hammad",
        "karam",
        "akram",
        "ikram",
        "aslam",
        "islam",
        "salam",
        "saad",
        "saud",
        "iman",
        "rashad",
        "samar",
        "sharaf",
        "firas",
        "bashar",
        "ammar",
        "shakir",
        "shakur",
    ]
)
# Arabic script spellings of those names (in `_ar_norm` form), so an Arabic script word can be
# told apart from a different name with the same consonants (محمد Muhammad, محمود Mahmud).
ARABIC_KNOWN = {
    "محمد": "muhammad", "محمود": "mahmud", "احمد": "ahmad", "حامد": "hamid", "حمد": "hamad",
    "حماد": "hammad", "حسن": "hasan", "حسين": "husain", "كريم": "karim", "كرم": "karam",
    "اكرم": "akram", "سليم": "salim", "سالم": "salim", "سلام": "salam", "اسلام": "islam",
    "اسلم": "aslam", "جمال": "jamal", "جميل": "jamil", "امين": "amin", "ايمان": "iman",
    "ايمن": "ayman", "رشيد": "rashid", "رشاد": "rashad", "سمير": "samir", "سمر": "samar",
    "سعيد": "said", "سعد": "saad", "سعود": "saud", "سيد": "sayyid", "شريف": "sharif",
    "شرف": "sharaf", "فارس": "faris", "فراس": "firas", "بشير": "bashir", "بشار": "bashar",
    "عمر": "umar", "امير": "amir", "عمار": "ammar", "هاشم": "hashim", "هشام": "hisham",
    "شاكر": "shakir", "شكور": "shakur", "يوسف": "yusuf", "ابراهيم": "ibrahim",
    "عثمان": "uthman", "علي": "ali", "خالد": "khalid", "صالح": "salih", "ناصر": "nasir",
    "مصطفي": "mustafa", "اسامه": "usama", "يحيي": "yahya", "اسماعيل": "ismail",
    "جعفر": "jafar", "مهدي": "mahdi", "محسن": "muhsin", "فيصل": "faisal", "طارق": "tariq",
    "زيد": "zaid", "سليمان": "sulaiman", "فتحي": "fathi", "رجب": "rajab", "حمزه": "hamza",
    "عباس": "abbas", "حبيب": "habib", "ماجد": "majid", "وليد": "walid", "جاسم": "jasim",
    "مرتضي": "murtada", "عيسي": "isa", "موسي": "musa",
}  # fmt: skip
# Known names whose Arabic script spelling is in ARABIC_KNOWN: an Arabic word that is not
# that spelling is not that name.
ARABIC_SPELLED_NAMES = frozenset(ARABIC_KNOWN.values())

_DIGRAPHS = [
    ("tch", "s"), ("sch", "s"), ("ph", "f"), ("kh", "h"), ("gh", "j"), ("sh", "s"), ("ch", "s"),
    ("zh", "j"), ("dj", "j"), ("dh", "z"), ("th", "t"), ("ck", "k"), ("q", "k"), ("c", "k"),
    ("x", "ks"), ("g", "j"),
]  # fmt: skip
# Read as two letters in the alternative skeleton (see module docstring).
_AMBIGUOUS = frozenset(["th", "sh", "kh", "dh"])
_VOWELS = frozenset("aeiou")


@dataclass(frozen=True, slots=True)
class Token:
    folded: str
    skeletons: tuple[str, ...]
    from_arabic: bool = False
    shapes: tuple[str, ...] = ()  # Latin words: the vowel pattern behind each key, same order
    glued: tuple[bool, ...] = ()  # Latin words: key read with a glued "al" taken off
    initials: str = ""  # letters this word could be shortened to

    def __post_init__(self) -> None:
        if not self.initials and self.folded:
            object.__setattr__(self, "initials", self.folded[:1])

    @property
    def skeleton(self) -> str:
        return self.skeletons[0]


def _is_arabic_letter(ch: str) -> bool:
    return ch != "ـ" and unicodedata.category(ch).startswith("L")


def has_arabic(text: str) -> bool:
    """True when the text holds an Arabic script letter. Marks, digits and punctuation from the
    Arabic blocks (such as the Arabic letter mark) do not count."""
    return any(_is_arabic_letter(ch) for ch in _ARABIC_BLOCKS.findall(text))


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def fold_latin(text: str) -> str:
    """Accents off, and letters with no accent to strip (ı ł đ ø æ ə ß) spelled out."""
    return strip_accents(text).translate(_LATIN_FOLD)


def _strip_marks(text: str) -> str:
    """Drop apostrophes and accents typed on their own, before NFKC turns "Sa´id" into
    "Sa ́id" and splits the word."""
    text = _APOSTROPHES.sub("", text)
    if text.isascii():
        return text.replace("^", "")
    return "".join(ch for ch in text if unicodedata.category(ch) != "Sk")


def _clean(text: str) -> str:
    return _INVISIBLE.sub("", unicodedata.normalize("NFKC", _strip_marks(text)))


def _ar_norm(word: str) -> str:
    return word.translate(_AR_NORM)


def is_arabic_particle(word: str) -> bool:
    """An Arabic script title, connector or family article (شيخ, بن, آل), which the
    tokenizer drops."""
    norm = _ar_norm(word)
    return norm in AR_TITLES or norm in AR_CONNECTORS or norm in AR_FAMILY


def _key_shape(letters: str) -> tuple[str, str]:
    """Consonant key and vowel pattern of mapped letters.

    Key: drop vowels, drop y and w unless they start the word, collapse a consonant that
    repeats the one before it. Shape: collapse doubled letters and write each run of vowels
    as one "V" ("muhammad" gives key "mhmd" and shape "mVhVmVd")."""
    key: list[str] = []
    shape: list[str] = []
    for i, ch in enumerate(letters):
        if ch in _VOWELS or (ch in "yw" and i > 0):
            if not shape or shape[-1] != "V":
                shape.append("V")
            continue
        if not key or key[-1] != ch:
            key.append(ch)
        if not shape or shape[-1] != ch:
            shape.append(ch)
    return "".join(key), "".join(shape)


def _mapped(word: str, *, ambiguous_as_two: bool = False) -> str:
    for src, dst in _DIGRAPHS:
        if ambiguous_as_two and src in _AMBIGUOUS:
            continue
        word = word.replace(src, dst)
    return word


@lru_cache(maxsize=200_000)
def latin_readings(word: str) -> tuple[tuple[str, str], ...]:
    """(key, shape) readings of a Latin word: the main one, then one that reads th, sh, kh
    and dh as two letters, then one that reads dj as d, when the word has them."""
    w = re.sub(r"[^a-z]", "", word.lower())
    readings = [_key_shape(_mapped(w))]
    alternatives = []
    if any(d in w for d in _AMBIGUOUS):
        alternatives.append(_mapped(w, ambiguous_as_two=True))
    if "dj" in w:
        alternatives.append(_mapped(w.replace("dj", "d")))
    if "gh" in w:  # Persian writes ق as gh (Ghasem, Bagher)
        alternatives.append(_mapped(w.replace("gh", "q")))
    seen = {readings[0][0]}
    for alt in alternatives:
        key, shape = _key_shape(alt)
        if key not in seen:
            readings.append((key, shape))
            seen.add(key)
    return tuple(readings)


def latin_skeletons(word: str) -> tuple[str, ...]:
    return tuple(key for key, _ in latin_readings(word))


def skeleton(word: str) -> str:
    return latin_readings(word)[0][0]


def transliterate_arabic(text: str) -> str:
    """Readable letter by letter transliteration ("محمد" -> "mhmd")."""
    text = _ARABIC_MARKS.sub("", text)
    return "".join(ARABIC_READABLE.get(ch, ch) for ch in text)


def _arabic_key(word: str, table: dict[str, str]) -> str:
    letters = "".join(table.get(ch, ch) for ch in word)
    return _key_shape(re.sub(r"[^a-z]", "", strip_accents(letters).lower()))[0]


@lru_cache(maxsize=100_000)
def _arabic_token(word: str) -> Token:
    # Hamza on or under alef is not always written, so it does not change the folded form.
    readable = "".join(ARABIC_READABLE.get(ch, ch) for ch in word.replace("إ", "ا"))
    folded = ARABIC_KNOWN.get(_ar_norm(word)) or readable
    keys = [_arabic_key(word, ARABIC_SKELETON)]
    if any(ch in ARABIC_SKELETON_ALT for ch in word):
        alt = _arabic_key(word, _ARABIC_SKELETON_WITH_ALT)
        if alt != keys[0]:
            keys.append(alt)
    # Persian and Urdu write و as v where it is a consonant: first in the word, or before
    # alef (وحيد Vahid, جواد Javad).
    if "و" in word:
        letters = [
            "v" if ch == "و" and (i == 0 or word[i + 1 : i + 2] == "ا") else ch
            for i, ch in enumerate(word)
        ]
        alt = _arabic_key("".join(letters), ARABIC_SKELETON)
        if alt not in keys:
            keys.append(alt)
    first = word[:1]
    letters = set(_ARABIC_INITIALS.get(first, ARABIC_READABLE.get(first, first)[:1]))
    if folded in KNOWN_GIVEN_NAMES:
        letters.add(folded[0])
        letters.update(v[0] for v in VARIANT_GROUPS.get(folded, ()))
    letters.discard("")
    return Token(folded, tuple(keys), True, (), (), "".join(sorted(letters)))


@lru_cache(maxsize=200_000)
def _latin_token(word: str, written: str = "") -> Token:
    folded = CANONICAL.get(word, word)
    if folded in ULLAH:
        folded = "ullah"
    elif folded in ("deen", "dine"):
        folded = "din"
    readings = list(latin_readings(folded))
    glued = [False] * len(readings)
    letters = {(written or word)[:1], folded[:1]}
    letters.update(v[:1] for v in VARIANT_GROUPS.get(folded, ()))
    # A glued "al" may be the article or part of the name: keep the whole word's keys first,
    # whatever the word's position, and add the keys of the rest as weaker alternatives.
    if folded.startswith("al") and len(folded) >= 5:
        seen = {key for key, _ in readings}
        for key, shape in latin_readings(folded[2:]):
            if len(key) >= 2 and key not in seen:
                readings.append((key, shape))
                glued.append(True)
                seen.add(key)
        if folded[2] not in _VOWELS:
            letters.add(folded[2])
    letters.discard("")
    return Token(
        folded,
        tuple(key for key, _ in readings),
        False,
        tuple(shape for _, shape in readings),
        tuple(glued),
        "".join(sorted(letters)),
    )


@lru_cache(maxsize=100_000)
def latin_token(word: str) -> Token:
    """The token for one lowercase Latin word, as the tokenizer builds it."""
    return _latin_token(word)


@lru_cache(maxsize=100_000)
def _arabic_word(word: str) -> Token:
    """One Arabic script word. A leading ال is read as the article, and the whole word's key
    is kept as a second key for names that only look like it (الياس, Ilyas)."""
    if not (word.startswith(AR_ARTICLE) and len(word) > 3):
        return _arabic_token(word)
    if _ar_norm(word[2:]) == "دين":  # what is left of الدين
        return _latin_token("din")
    core = _arabic_token(word[2:])
    full = _arabic_token(word)
    extra = tuple(k for k in full.skeletons if k and k not in core.skeletons)
    initials = "".join(sorted(set(core.initials) | set(full.initials)))
    return replace(core, skeletons=core.skeletons + extra, initials=initials)


def _is_arabic_title(word: str) -> bool:
    norm = _ar_norm(word)
    return norm in AR_TITLES or (norm.startswith(AR_ARTICLE) and norm[2:] in AR_TITLES)


def _tokenize_arabic(text: str, *, has_prior: bool = False) -> list[Token]:
    text = _ARABIC_MARKS.sub("", text)
    words = [w for w in re.split(r"[\W_]+", text) if w]  # any punctuation splits words
    words = [w for w in words if not _is_arabic_title(w)]
    if not has_prior and len(words) >= 3 and _ar_norm(words[0]) == AR_HAFIZ:
        words = words[1:]  # Hafiz as a title, as on the Latin side
    tokens: list[Token] = []
    i = 0
    while i < len(words):
        w = words[i]
        nxt = words[i + 1] if i + 1 < len(words) else None
        norm = _ar_norm(w)
        if norm in AR_FAMILY and nxt is not None:
            i += 1
            continue
        if norm in AR_CONNECTORS and (tokens or has_prior) and nxt is not None:
            i += 1
            continue
        if norm == AR_ABD and nxt is not None:
            if _ar_norm(nxt) == AR_ALLAH_NORM:
                tokens.append(_latin_token("abdullah"))
            else:
                target = nxt[2:] if nxt.startswith(AR_ARTICLE) and len(nxt) > 3 else nxt
                tokens.extend([_latin_token("abd"), _arabic_token(target)])
            i += 2
            continue
        # _ar_norm maps one letter to one letter, so slicing `w` by `norm` lengths is safe.
        if norm.startswith(AR_ARTICLE + AR_ABD) and len(w) > 5:  # العبدالله, as Al Abdullah
            w, norm = w[2:], norm[2:]
        rest = norm[3:] if norm.startswith(AR_ABD) else ""
        if rest == AR_ALLAH_NORM:  # glued عبدالله
            tokens.append(_latin_token("abdullah"))
        elif rest.startswith(AR_ARTICLE) and len(rest) > 3:  # glued عبدالرحمن
            tokens.extend([_latin_token("abd"), _arabic_token(w[5:])])
        elif norm == AR_ALLAH_NORM:
            tokens.append(_latin_token("ullah"))
        elif norm.endswith(AR_DIN_NORM) and len(w) > len(AR_DIN) + 1:  # glued: نورالدين
            tokens.extend([_arabic_word(w[: -len(AR_DIN)]), _latin_token("din")])
        elif norm.endswith(AR_ALLAH_NORM) and len(w) > len(AR_ALLAH) + 1:  # glued: نصرالله
            tokens.extend([_arabic_word(w[: -len(AR_ALLAH)]), _latin_token("ullah")])
        else:
            tokens.append(_arabic_word(w))
        i += 1
    return [t for t in tokens if t.skeleton or t.folded]


def _reorder_comma(text: str) -> str:
    """'SURNAME, Given Names' -> 'Given Names SURNAME'."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) == 2 and all(parts):
        return f"{parts[1]} {parts[0]}"
    return text.replace(",", " ")


def _plain_vowels(word: str) -> str:
    return word.replace("ee", "i").replace("oo", "u").replace("ou", "u")


_DIVINE_SKELETONS = frozenset(k for k in (skeleton(n) for n in DIVINE_NAMES) if len(k) >= 3)
_DIVINE_BY_SHORT_KEY: dict[str, list[str]] = {}
for _name in sorted(DIVINE_NAMES):
    if len(skeleton(_name)) < 3:
        _DIVINE_BY_SHORT_KEY.setdefault(skeleton(_name), []).append(_plain_vowels(_name))
_ABD_ARTICLES = (
    "oul", "our", "ous", "ul", "el", "al", "ol", "il", "ur", "ar", "us", "as", "ou", "u", "e",
    "o", "a", "i",
)  # fmt: skip


def _is_abd_name(word: str) -> bool:
    """Whether a word can be the name in an Abd compound. A key of one or two letters says
    little (Abdullahi would split into "abd" + "lahi"), so a short key needs a spelling within
    one edit of a known name that starts with the same sound (nour for nur, kawi for qawi)."""
    if len(word) < 2:
        return False
    if word in DIVINE_NAMES:
        return True
    key = skeleton(word)
    if len(key) >= 3:
        return key in _DIVINE_SKELETONS
    if len(word) < 3:
        return False
    plain, first = _plain_vowels(word), _mapped(word)[:1]
    return any(
        Levenshtein.distance(plain, n) <= 1 and _mapped(n)[:1] == first
        for n in _DIVINE_BY_SHORT_KEY.get(key, ())
    )


def _split_abd(word: str) -> list[str] | None:
    """'abdulrahman', 'abdurrahman', 'abdelaziz' -> ['abd', 'rahman'] when the rest is a
    name that follows Abd; 'abdullah' and its spellings (Abdullahi, Abdoulaye) stay one."""
    if not word.startswith("abd") or len(word) <= 5:
        return None
    if CANONICAL.get(word, word) == "abdullah":
        return ["abdullah"]
    rest = word[3:]
    options = []
    for article in _ABD_ARTICLES:
        if rest.startswith(article):
            candidate = rest[len(article) :]
            # sun letter assimilation doubles the next letter: abdur-rahman, abdus-salam
            if len(candidate) > 3 and candidate[0] == candidate[1]:
                candidate = candidate[1:]
            options.append(candidate)
    options.append(rest)
    for candidate in options:  # an exact name first, then a close spelling
        if candidate in DIVINE_NAMES:
            return ["abd", candidate]
    for candidate in options:
        if _is_abd_name(candidate):
            return ["abd", candidate]
    return None


def _split_suffixes(word: str) -> list[str]:
    for suffix in ULLAH:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return [word[: -len(suffix)], "ullah"]
    for suffix in DIN_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return [word[: -len(suffix)], "din"]
    return [word]


def _assimilated(article: str, word: str | None) -> bool:
    """'ar' before 'rahman', 'as' before 'salam', 'ash' before 'shakur'."""
    if word is None or not 2 <= len(article) <= 3 or article[0] not in "aeiu":
        return False
    sound = article[1:]
    return sound in _SUN_SOUNDS and word.startswith(sound) and len(word) > len(sound) + 1


def _drop_particle(word: str, nxt: str | None) -> bool:
    if nxt is None:
        return False
    if word in _DIN_ARTICLES and nxt in _DIN_WORDS:  # Noor ud Din, Nasr ed Dine
        return True
    return word in _U_ARTICLES and _assimilated(word, nxt)  # Habib ur Rahman


def _name_words(words: list[str]) -> int:
    return sum(w not in ARTICLES and w not in CONNECTORS and w not in TITLES for w in words)


def _tokenize_latin(text: str, *, has_prior: bool = False) -> list[Token]:
    text = _SON_OF.sub(" ", text)
    text = _reorder_comma(text)
    text = fold_latin(text).casefold()
    text = _APOSTROPHES.sub("", text)
    text = _ASSIMILATED_HYPHEN.sub("", _DASHES.sub("-", text))
    words = re.sub(r"[^a-z0-9]+", " ", text).split()
    if not words:
        return []
    kept = [w for w in words if w not in TITLES]
    while len(kept) > 1 and kept[0] in FRENCH_TITLE_WORDS:
        kept.pop(0)
    if not has_prior and kept and kept[0] in FIRST_WORD_TITLES and _name_words(kept[1:]) >= 2:
        kept.pop(0)
    words = kept or words

    out: list[tuple[str, str]] = []  # (word, as written)
    i = 0
    n = len(words)
    while i < n:
        w = words[i]
        nxt = words[i + 1] if i + 1 < n else None
        if w in ARTICLES and nxt is not None:
            i += 1
            continue
        if w in CONNECTORS and (out or has_prior) and nxt is not None:
            i += 1
            continue
        if _drop_particle(w, nxt):
            i += 1
            continue
        if w in ABD_FORMS and nxt is not None:
            j = i + 1
            after = words[j + 1] if j + 1 < n else None
            if words[j] in ARTICLES or _assimilated(words[j], after):
                j += 1
            if j >= n:  # "Abd al" with nothing after the article
                out.append(("abd", w))
            elif words[j] in ULLAH:
                out.append(("abdullah", w))
            else:
                out.extend([("abd", w), (words[j], words[j])])
            i = j + 1
            continue
        if w.startswith("alabd"):  # Alabdullah, as Al Abdullah
            parts = _split_abd(w[2:])
            if parts:
                out.extend((p, p) for p in parts)
                i += 1
                continue
        parts = _split_abd(w) or _split_suffixes(w)
        out.extend((p, w if k == 0 else p) for k, p in enumerate(parts))
        i += 1
    tokens = [_latin_token(word, written) for word, written in out]
    return [t for t in tokens if t.skeleton or t.folded]


def _translit_cyrillic(text: str, *, h_for_g: bool = False) -> str:
    """Cyrillic to Latin, letter by letter. A letter missing from the table falls back to its
    base letter (ӣ to и), and an unknown Cyrillic letter is dropped rather than splitting
    the word. With `h_for_g`, г is written h, as Ukrainian and Belarusian romanise it
    (Ihor, Oleh), except after a letter that would turn "h" into a digraph."""
    out: list[str] = []
    for ch in text.lower():
        if h_for_g and ch == "г" and not (out and out[-1][-1:] in "zskctdgp"):
            out.append("h")
            continue
        latin = CYRILLIC_TO_LATIN.get(ch)
        if latin is None:
            latin = CYRILLIC_TO_LATIN.get(unicodedata.normalize("NFD", ch)[0])
        if latin is None:
            latin = "" if _HAS_CYRILLIC.match(ch) else ch
        out.append(latin)
    return "".join(out)


def _script_runs(text: str) -> list[tuple[bool, str]]:
    """Split a name into runs of Arabic script words and other words, in order."""
    if not has_arabic(text):
        return [(False, text)]
    runs: list[tuple[bool, str]] = []
    for chunk in text.split():
        arabic = has_arabic(chunk)
        if runs and runs[-1][0] == arabic:
            runs[-1] = (arabic, f"{runs[-1][1]} {chunk}")
        else:
            runs.append((arabic, chunk))
    return runs


def tokenize(name: str) -> list[Token]:
    """Normalise a name into matching tokens. Titles, articles and connectors are dropped."""
    tokens: list[Token] = []
    for arabic, run in _script_runs(_clean(name)):
        if arabic:
            tokens.extend(_tokenize_arabic(run, has_prior=bool(tokens)))
            continue
        if not _HAS_CYRILLIC.search(run):
            tokens.extend(_tokenize_latin(run, has_prior=bool(tokens)))
            continue
        main = _tokenize_latin(_translit_cyrillic(run), has_prior=bool(tokens))
        if "г" in run.lower():
            alt = _tokenize_latin(_translit_cyrillic(run, h_for_g=True), has_prior=bool(tokens))
            if len(alt) == len(main):
                main = [_with_keys_of(a, b) for a, b in zip(main, alt, strict=True)]
        tokens.extend(main)
    return tokens


def _with_keys_of(token: Token, other: Token) -> Token:
    """`token` with the keys of `other` it lacks added after its own (a second reading)."""
    keys, shapes, glued = list(token.skeletons), list(token.shapes), list(token.glued)
    for i, key in enumerate(other.skeletons):
        if key and key not in keys:
            keys.append(key)
            shapes.append(other.shapes[i] if i < len(other.shapes) else key)
            glued.append(other.glued[i] if i < len(other.glued) else False)
    initials = "".join(sorted(set(token.initials) | set(other.initials)))
    return replace(
        token, skeletons=tuple(keys), shapes=tuple(shapes), glued=tuple(glued), initials=initials
    )


def neutral_key(name: str) -> str:
    """A plain normalisation used only to check whether two names are the same string:
    accents, case, punctuation and token order ignored. No name knowledge at all."""
    text = fold_latin(_clean(name)).casefold()
    text = _APOSTROPHES.sub("", text)
    return " ".join(sorted(re.sub(r"[^\w]+", " ", text).split()))
