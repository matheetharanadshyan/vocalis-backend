from __future__ import annotations

from types import ModuleType

from conftest import patched_modules


def build_phoneme_estimates_module(
    load_backend_module,
    *,
    download_outcomes: dict[str, tuple[bool, tuple[str, ...]]],
):
    available_resources: set[str] = set()
    download_calls: list[str] = []
    g2p_calls: list[str] = []

    nltk_module = ModuleType("nltk")
    nltk_data_module = ModuleType("nltk.data")

    def find(resource_path: str) -> str:
        if resource_path in available_resources:
            return resource_path
        raise LookupError(resource_path)

    def download(resource_name: str, quiet: bool = True) -> bool:
        download_calls.append(resource_name)
        succeeded, resources_to_add = download_outcomes.get(resource_name, (False, ()))
        if succeeded:
            available_resources.update(resources_to_add)
        return succeeded

    nltk_data_module.find = find
    nltk_module.data = nltk_data_module
    nltk_module.download = download

    g2p_en_module = ModuleType("g2p_en")

    class FakeG2p:
        def __call__(self, word: str) -> list[str]:
            g2p_calls.append(word)
            pronunciations = {
                "warmup": ["W", "AO1", "R", "M", "AH0", "P"],
                "alpha": ["AE1", "L", "F", "AH0"],
            }
            return pronunciations.get(word, ["T", "EH1", "S", "T"])

    g2p_en_module.G2p = FakeG2p

    module = load_backend_module(
        "phoneme_estimates",
        {
            "g2p_en": g2p_en_module,
        },
    )
    return module, download_calls, g2p_calls, nltk_module


def test_get_g2p_downloads_the_new_english_tagger_when_missing(load_backend_module) -> None:
    phoneme_estimates, download_calls, g2p_calls, nltk_module = build_phoneme_estimates_module(
        load_backend_module,
        download_outcomes={
            "cmudict": (True, ("corpora/cmudict",)),
            "averaged_perceptron_tagger_eng": (True, ("taggers/averaged_perceptron_tagger_eng",)),
        },
    )

    with patched_modules({"nltk": nltk_module}):
        assert phoneme_estimates.phonemize_text("alpha") == [["AE", "L", "F", "AH"]]
    assert download_calls == ["cmudict", "averaged_perceptron_tagger_eng"]
    assert g2p_calls == ["warmup", "alpha"]


def test_get_g2p_falls_back_to_the_legacy_tagger_download(load_backend_module) -> None:
    phoneme_estimates, download_calls, g2p_calls, nltk_module = build_phoneme_estimates_module(
        load_backend_module,
        download_outcomes={
            "cmudict": (True, ("corpora/cmudict",)),
            "averaged_perceptron_tagger_eng": (False, ()),
            "averaged_perceptron_tagger": (True, ("taggers/averaged_perceptron_tagger",)),
        },
    )

    with patched_modules({"nltk": nltk_module}):
        phoneme_estimates.get_g2p()

    assert download_calls == [
        "cmudict",
        "averaged_perceptron_tagger_eng",
        "averaged_perceptron_tagger",
    ]
    assert g2p_calls == ["warmup"]
