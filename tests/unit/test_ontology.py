import pytest

from fixgraph.core.ontology import Ontology, load_ontology, parse_os_version


@pytest.fixture(scope="module")
def onto() -> Ontology:
    return load_ontology()


def test_families_word_boundaries(onto: Ontology) -> None:
    assert onto.families_in("Pair AirPods Pro with your iPhone") == ["AirPods", "iPhone"]
    assert onto.families_in("macOS on a MacBook Air") == ["Mac"]
    assert onto.families_in("Apple Watch Ultra 2") == ["Apple Watch"]
    assert onto.families_in("machine learning") == []


def test_products(onto: Ontology) -> None:
    found = onto.products_in("On iPhone 16 Pro Max and Apple Watch Series 10 with AirPods Pro 2")
    assert ("iPhone", "iPhone 16 Pro Max") in found
    assert ("Apple Watch", "Apple Watch Series 10") in found
    assert ("AirPods", "AirPods Pro 2") in found


def test_os_versions(onto: Ontology) -> None:
    text = "Requires iOS 26.1 or iPadOS 18, watchOS 26, and macOS Tahoe or macOS 15.2."
    assert onto.os_versions_in(text) == [
        "iOS 26.1",
        "iPadOS 18",
        "macOS 15.2",
        "macOS 26",
        "watchOS 26",
    ]


def test_macos_names_with_and_without_numbers(onto: Ontology) -> None:
    assert onto.os_versions_in("a Mac with macOS Catalina or later") == ["macOS 10.15"]
    assert onto.os_versions_in("update to macOS Ventura 13.5 or later") == ["macOS 13.5"]
    assert onto.os_versions_in("macOS High Sierra") == ["macOS 10.13"]


def test_parse_os_version() -> None:
    assert parse_os_version("iOS 26.1") == ("iOS", 26, 1, 0)
    assert parse_os_version("macOS 15.2.3") == ("macOS", 15, 2, 3)


def test_components_and_features(onto: Ontology) -> None:
    text = "Turn Wi-Fi and Bluetooth off, then check Find My and iCloud Photos."
    assert onto.components_in(text) == ["Bluetooth", "Wi-Fi"]
    assert {"Find My", "iCloud", "iCloud Photos"} <= set(onto.features_in(text))
