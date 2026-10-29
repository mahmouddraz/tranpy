from __future__ import annotations

from tranpy.orthophotos import ORTHOPHOTO_SERVICES, _coverage, _licence


def test_all_states_have_a_service() -> None:
    states = {s.state for s in ORTHOPHOTO_SERVICES}
    assert len(states) == 16
    assert all(s.url.startswith("https://") for s in ORTHOPHOTO_SERVICES)


def test_coverage_from_capabilities() -> None:
    wms13 = (
        "<EX_GeographicBoundingBox><westBoundLongitude>8.9</westBoundLongitude>"
        "<eastBoundLongitude>13.9</eastBoundLongitude><southBoundLatitude>47.2"
        "</southBoundLatitude><northBoundLatitude>50.6</northBoundLatitude>"
    )
    assert _coverage(wms13) == (8.9, 13.9, 47.2, 50.6)
    wms11 = '<LatLonBoundingBox minx="5.8" miny="50.3" maxx="9.5" maxy="52.6"/>'
    assert _coverage(wms11) == (5.8, 9.5, 50.3, 52.6)
    assert _coverage("<WMS/>") is None


def test_licence_text_as_published() -> None:
    xml = "<AccessConstraints>CC BY 4.0\n   vgl. https://x</AccessConstraints>"
    assert _licence(xml) == "CC BY 4.0 vgl. https://x"
    assert _licence("<WMS/>") == "not stated by the service"
