import pytest

from sanctions_mcp.normalize import latin_skeletons, neutral_key, tokenize


def folded(name: str) -> list[str]:
    return [t.folded for t in tokenize(name)]


def skeletons(name: str) -> list[str]:
    return [t.skeleton for t in tokenize(name)]


def meet(a: str, b: str) -> bool:
    """Same number of words, and every word pair shares a key."""
    ta, tb = tokenize(a), tokenize(b)
    return len(ta) == len(tb) and all(
        set(x.skeletons) & set(y.skeletons) for x, y in zip(ta, tb, strict=True)
    )


@pytest.mark.parametrize(
    ("latin", "arabic"),
    [
        ("Karim Fadel Al-Nouri", "كريم فاضل النوري"),
        ("Yusuf Abdul Rahman Haddad", "يوسف عبد الرحمن حداد"),
        ("Nur al-Din Hamdani", "نورالدين حمداني"),
        ("Abdullah Fawzi", "عبدالله فوزي"),
        ("Fathi Suhail Taha", "فتحي سهيل طه"),
        ("Issa Ali Omar", "عيسى علي عمر"),
        ("Ali Reza", "علي رضا"),  # ض read as z
        ("Fadhil Ramzan", "فاضل رمضان"),
        ("Muhammad Abduh", "محمد عبده"),  # عبده is a name, not Abd + h
        ("Ilyas Khan", "الياس خان"),  # ال that belongs to the name
        ("Tamim bin Hamad Al Thani", "تميم بن حمد آل ثاني"),  # آل is the family article
        ("Mohamed Ould Abdel Aziz", "محمد ولد عبد العزيز"),  # ولد is a connector
        ("Nur al-Din", "نورالدین"),  # Persian yeh
        ("Abdullah", "عبداللہ"),  # Urdu heh goal
        ("Nasrullah", "نصراللہ"),
        ("Ghasem Soleimani", "قاسم سليماني"),  # Persian gh for ق
        ("Vahid", "وحيد"),  # Persian v for و
        ("Javad", "جواد"),
    ],
)
def test_arabic_script_meets_latin_on_every_word(latin, arabic):
    lat, ara = tokenize(latin), tokenize(arabic)
    assert len(lat) == len(ara)
    for a, b in zip(lat, ara, strict=True):
        assert set(a.skeletons) & set(b.skeletons), (a, b)
    assert all(t.from_arabic for t in ara if t.folded not in ("abd", "abdullah", "din", "ullah"))


def test_common_spellings_share_one_form():
    assert folded("Mohammed Hussein") == folded("Muhammad Husayn") == ["muhammad", "husain"]
    assert folded("Kareem") == folded("Karim") == ["karim"]


def test_abd_compounds_split_the_same_way():
    expected = ["abd", "rahman"]
    for spelling in ["Abdul Rahman", "Abdulrahman", "Abd al-Rahman", "Abdurrahman", "Abdel Rahman"]:
        assert folded(spelling) == expected, spelling
    assert folded("Abd Allah") == folded("Abdallah") == ["abdullah"]
    for spelling in ["Abdul Hafiz", "Abdulhafiz", "Abd al-Hafiz", "Abdul Hafeez", "عبد الحفيظ"]:
        assert skeletons(spelling) == ["bd", "hfz"], spelling
    assert folded("Abd al") == ["abd"]  # an article is never the name


def test_abdullah_spellings_stay_whole_and_short_names_still_split():
    for spelling in ["Abdullahi", "Abdoulaye", "Abdulai", "Abdellahi", "Abdulle"]:
        assert folded(spelling) == ["abdullah"], spelling
    assert folded("Abdulhai") == ["abd", "hai"]
    assert folded("Abdelkader") == ["abd", "kader"]
    assert folded("Abdelnour") == ["abd", "nour"]
    assert folded("Abdulraouf") == ["abd", "raouf"]
    assert folded("Abdelli") == ["abdelli"]  # a family name, not Abd + li


def test_hafiz_is_a_title_only_before_two_name_words():
    assert folded("Hafiz Muhammad Saeed") == ["muhammad", "said"]
    assert folded("Hafiz Saeed") == ["hafiz", "said"]
    assert len(tokenize("حافظ محمد سعيد")) == 2
    assert folded("Emir Hodzic") == ["amir", "hodzic"]  # Emir is a given name


def test_assimilated_articles_are_dropped():
    assert folded("Noor ud Din") == folded("Nooruddin")
    assert folded("Nasr ed Dine") == folded("Nasreddine")
    assert folded("Salah ad-Din") == folded("Salahuddin")
    assert folded("Abd ar-Rahman") == folded("Abd ar Rahman") == folded("Abdurrahman")
    assert folded("Ayman az-Zawahiri") == folded("Ayman al-Zawahiri")
    assert folded("Habib ur Rahman") == ["habib", "rahman"]
    assert folded("Ed Davis") == ["ed", "davis"]


def test_ullah_and_din_suffixes():
    assert folded("Saifullah") == folded("Saif Ullah") == ["saif", "ullah"]
    assert skeletons("Nooruddin") == skeletons("Nur al-Din")


def test_titles_articles_and_connectors_are_dropped():
    assert folded("Sheikh Ahmad bin Salim Al-Qasimi") == ["ahmad", "salim", "qasimi"]
    assert folded("Major General Ali Durgham") == ["ali", "durgham"]


def test_surname_comma_order_is_restored():
    assert folded("NOURI, Karim Fadel") == ["karim", "fadel", "nouri"]


@pytest.mark.parametrize(
    ("cyrillic", "latin"),
    [
        ("Иван Петрович", "Ivan Petrovich"),
        ("Ратко Младић", "Ratko Mladic"),  # Serbian
        ("Радован Караџић", "Radovan Karadzic"),
        ("Ђорђе", "Djordje"),
        ("Нұрсұлтан", "Nursultan"),  # Kazakh
        ("Ольга", "Olʹga"),  # soft sign written as a prime
    ],
)
def test_cyrillic_is_transliterated_without_splitting_words(cyrillic, latin):
    assert meet(cyrillic, latin)


@pytest.mark.parametrize(
    ("special", "plain"),
    [
        ("Yılmaz", "YILMAZ"),
        ("Kılıç", "Kilic"),
        ("Michał", "Michal"),
        ("Đorđević", "Dordevic"),
        ("Djordjevic", "Dordevic"),  # dj for the d with stroke
        ("Əliyev", "Aliyev"),
        ("Strauß", "Strauss"),
    ],
)
def test_latin_letters_without_accents_are_folded(special, plain):
    assert len(tokenize(special)) == 1
    assert meet(special, plain)


def test_apostrophes_and_lone_accents_never_split_a_word():
    assert folded("Sa´id") == folded("Saʿid") == folded("Sa'id") == ["said"]
    assert folded("Muʿammar") == ["muammar"]
    assert neutral_key("Sa´id") == neutral_key("Said")


def test_mixed_script_names_are_read_one_script_at_a_time():
    tokens = tokenize("Qasim al-Rimi الريمي")
    assert [t.skeleton for t in tokens] == ["ksm", "rm", "rm"]
    assert [t.from_arabic for t in tokens] == [False, False, True]
    assert tokenize("؜Qasim Soleimani") == tokenize("Qasim Soleimani")
    assert tokenize("Qasim­Soleimani") == tokenize("QasimSoleimani")
    assert folded("Sheikh Ahmad bin Salim أحمد") == ["ahmad", "salim", "ahmad"]


def test_glued_arabic_abd_splits_only_real_compounds():
    assert len(tokenize("عبده")) == 1
    assert [t.folded for t in tokenize("عبدالله")] == ["abdullah"]
    assert tokenize("عبدالرحمن")[0].folded == "abd"


def test_glued_al_gives_the_same_keys_in_any_position():
    a, b = tokenize("Juan Alvarez"), tokenize("Alvarez Juan")
    assert {t.folded: t.skeletons for t in a} == {t.folded: t.skeletons for t in b}
    almatyouti = tokenize("Ali Almatyouti")[1]
    assert almatyouti.skeletons == ("lmt", "mt") and almatyouti.glued == (False, True)
    assert "v" not in tokenize("Ilham Aliyev")[1].skeletons  # no one letter extra key
    assert "" not in tokenize("Ahmed Alaoui")[1].skeletons
    assert folded("Alabdullah") == folded("Al-Abdullah") == folded("العبدالله") == ["abdullah"]


def test_initials_cover_every_spelling():
    assert set("ou") <= set(tokenize("Osama")[0].initials)
    assert "o" in tokenize("عمر")[0].initials
    assert "g" in tokenize("جمال")[0].initials


def test_ambiguous_digraph_gets_a_second_skeleton():
    assert latin_skeletons("fathi") == ("ft", "fth")
    assert latin_skeletons("ahmad") == ("hmd",)


def test_shapes_keep_vowel_patterns_apart():
    karim, akram, kareem = (tokenize(n)[0] for n in ("Karim", "Akram", "Kareem"))
    assert karim.skeleton == akram.skeleton == "krm"
    assert karim.shapes[0] != akram.shapes[0]
    assert karim == kareem


def test_neutral_key_ignores_case_accents_punctuation_and_order():
    assert neutral_key("KOÇAK, Abdül-Vahap") == neutral_key("abdul vahap kocak")
    assert neutral_key("Mohammed") != neutral_key("Muhammad")
    assert neutral_key("Yılmaz") == neutral_key("yilmaz")


def test_ukrainian_g_is_also_read_as_h():
    assert meet("Ігор", "Ihor") and meet("Олег", "Oleh") and meet("Олег", "Oleg")
    assert meet("Григорій Гриценко", "Hryhoriy Hrytsenko")


def test_persian_and_urdu_titles_are_dropped():
    assert len(tokenize("حاجی محمد ابراهیم")) == len(tokenize("Haji Muhammad Ibrahim")) == 2
    assert len(tokenize("مولوی عبدالکبیر")) == 2  # [abd, kabir], as Maulvi Abdul Kabir
    assert len(tokenize("دکتر علی")) == len(tokenize("مفتي علي")) == 1


def test_punctuation_never_sticks_to_arabic_script_words():
    plain = tokenize("محمد احمد")
    for name in ["(محمد) احمد", "«محمد» احمد", "محمد. احمد", "محمد؛ احمد"]:
        assert tokenize(name) == plain, name
