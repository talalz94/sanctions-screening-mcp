import random

import pytest

from sanctions_mcp.variants import GENERATORS, make_variant

NAME = "Yusuf Abdul Rahman Al-Haddad"


@pytest.mark.parametrize("kind", sorted(GENERATORS))
def test_every_generator_changes_the_name_or_declines(kind):
    rng = random.Random(1)
    for _ in range(20):
        out = make_variant(NAME, kind, rng)
        assert out is None or out != NAME


def test_generation_is_deterministic_for_a_seed():
    a = [make_variant(NAME, k, random.Random(7)) for k in sorted(GENERATORS)]
    b = [make_variant(NAME, k, random.Random(7)) for k in sorted(GENERATORS)]
    assert a == b


def test_drop_middle_and_initials_need_three_words():
    rng = random.Random(0)
    assert make_variant("Ali Hassan", "drop_middle", rng) is None
    assert make_variant("Ali Hassan", "initials", rng) is None


def test_honorific_is_prepended():
    out = make_variant("Ali Hassan", "honorific", random.Random(0))
    assert out.endswith("Ali Hassan") and out != "Ali Hassan"


def test_particles_never_cut_al_or_el_off_a_name():
    import random

    from sanctions_mcp.variants import particles

    for seed in range(60):
        out = particles(["Aleksandr", "Elena", "Petrova"], random.Random(seed))
        assert "Aleksandr" in out and "Elena" in out, out


def test_initials_never_abbreviate_titles_or_connectors():
    import random

    from sanctions_mcp.variants import initials

    for seed in range(60):
        out = initials(["Sheikh", "Muhammad", "bin", "Salman", "Saud"], random.Random(seed))
        assert out[0] == "Sheikh" and out[2] == "bin", out
