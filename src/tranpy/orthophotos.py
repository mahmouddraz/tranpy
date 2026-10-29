"""Official aerial photographs (digital orthophotos) of the German states.

Each state survey office publishes its orthophotos (20-40 cm ground resolution) through
a WMS. :func:`fetch_orthophoto` picks the service whose coverage contains a location
(as each service declares it), downloads an image around it and returns it with the
service's own licence statement, which must be followed when the image is shown.

Used to verify grid data on the ground: substation positions, separate switchyards,
tee points and line routes.
"""

from __future__ import annotations

import io
import math
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from tranpy_core._optional import require
from tranpy_core.data.download import cache_dir
from tranpy_core.errors import DataError

__all__ = ["ORTHOPHOTO_SERVICES", "Orthophoto", "OrthophotoService", "fetch_orthophoto"]


@dataclass(frozen=True)
class OrthophotoService:
    """A state's orthophoto WMS (``version`` 1.3.0 uses latitude-longitude order)."""

    state: str
    url: str
    layer: str
    version: str = "1.3.0"


ORTHOPHOTO_SERVICES: tuple[OrthophotoService, ...] = (
    OrthophotoService(
        "BB", "https://isk.geobasis-bb.de/mapproxy/dop20c/service/wms", "bebb_dop20c"
    ),
    OrthophotoService("BE", "https://gdi.berlin.de/services/wms/truedop_2024", "truedop_2024"),
    OrthophotoService(
        "BW",
        "https://owsproxy.lgl-bw.de/owsproxy/ows/WMS_LGL-BW_ATKIS_DOP_20_C",
        "IMAGES_DOP_20_RGB",
    ),
    OrthophotoService(
        "BY", "https://geoservices.bayern.de/od/wms/dop/v1/dop40", "by_dop40c", "1.1.1"
    ),
    OrthophotoService("HB", "https://geodienste.bremen.de/wms_dop20_2023", "DOP20_2023_HB"),
    OrthophotoService(
        "HE",
        "https://www.gds-srv.hessen.de/cgi-bin/lika-services/ogc-free-images.ows",
        "he_dop20_rgb",
    ),
    OrthophotoService(
        "HH", "https://geodienste.hamburg.de/wms_dop_zeitreihe_unbelaubt", "dop_zeitreihe_unbelaubt"
    ),
    OrthophotoService("MV", "https://www.geodaten-mv.de/dienste/adv_dop", "mv_dop"),
    OrthophotoService(
        "NI", "https://opendata.lgln.niedersachsen.de/doorman/noauth/dop_wms", "ni_dop20"
    ),
    OrthophotoService("NW", "https://www.wms.nrw.de/geobasis/wms_nw_dop", "nw_dop_rgb"),
    OrthophotoService("RP", "https://geo4.service24.rlp.de/wms/rp_dop20.fcgi", "rp_dop20"),
    OrthophotoService("SH", "https://dienste.gdi-sh.de/WMS_SH_DOP20col_OpenGBD", "sh_dop20_rgb"),
    OrthophotoService("SL", "https://geoportal.saarland.de/freewms/dop2023", "sl_dop20_rgb"),
    OrthophotoService("SN", "https://geodienste.sachsen.de/wms_geosn_dop-rgb/guest", "sn_dop_020"),
    OrthophotoService(
        "ST",
        "https://www.geodatenportal.sachsen-anhalt.de/wss/service/ST_LVermGeo_DOP_WMS_OpenData/guest",
        "lsa_lvermgeo_dop20_2",
    ),
    OrthophotoService(
        "TH", "https://www.geoproxy.geoportal-th.de/geoproxy/services/DOP", "th_dop20rgb"
    ),
)
"""Orthophoto services of the 16 German states (verified October 2026)."""


@dataclass(frozen=True)
class Orthophoto:
    """An image with its geographic extent and provenance.

    Attributes:
        image: ``(height, width, 3)`` RGB array.
        extent: ``(west, east, south, north)`` in degrees.
        service: the state service it came from.
        licence: the service's access constraints, as published.
    """

    image: npt.NDArray[np.uint8]
    extent: tuple[float, float, float, float]
    service: OrthophotoService
    licence: str


def _get(url: str, timeout_s: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "tranpy"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        data: bytes = response.read()
    return data


def _capabilities(service: OrthophotoService, directory: Path) -> str:
    path = directory / f"capabilities_{service.state}.xml"
    if not path.exists():
        query = urllib.parse.urlencode({"SERVICE": "WMS", "REQUEST": "GetCapabilities"})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(f"{service.url}?{query}", 60.0))
    return path.read_text(encoding="utf-8", errors="replace")


def _coverage(xml: str) -> tuple[float, float, float, float] | None:
    """Declared extent (west, east, south, north) of the service."""
    found = re.search(
        r"<westBoundLongitude>([-\d.]+)</westBoundLongitude>\s*"
        r"<eastBoundLongitude>([-\d.]+)</eastBoundLongitude>\s*"
        r"<southBoundLatitude>([-\d.]+)</southBoundLatitude>\s*"
        r"<northBoundLatitude>([-\d.]+)</northBoundLatitude>",
        xml,
    )
    if found:
        west, east, south, north = (float(v) for v in found.groups())
        return west, east, south, north
    box = re.search(
        r'<LatLonBoundingBox minx="([-\d.]+)" miny="([-\d.]+)" maxx="([-\d.]+)" maxy="([-\d.]+)"',
        xml,
    )
    if box:
        minx, miny, maxx, maxy = (float(v) for v in box.groups())
        return minx, maxx, miny, maxy
    return None


def _licence(xml: str) -> str:
    found = re.search(r"<AccessConstraints>(.*?)</AccessConstraints>", xml, re.S)
    text = found.group(1) if found else ""
    return re.sub(r"\s+", " ", text).strip() or "not stated by the service"


def fetch_orthophoto(
    lon: float,
    lat: float,
    *,
    half_width_km: float = 0.6,
    pixels: int = 1000,
    directory: str | Path | None = None,
    timeout_s: float = 60.0,
) -> Orthophoto:
    """Orthophoto centred on (``lon``, ``lat``), ``2 * half_width_km`` wide.

    The services whose declared coverage contains the point are tried from the smallest
    coverage up; an empty (uniform) image means the point is outside the state.
    Images and capabilities are cached in ``directory`` (default: the cache).
    """
    pil = require("PIL.Image", "plot")
    folder = Path(directory) if directory is not None else cache_dir() / "orthophotos"
    dlat = half_width_km / 111.2
    dlon = half_width_km / (111.2 * math.cos(math.radians(lat)))
    west, east, south, north = lon - dlon, lon + dlon, lat - dlat, lat + dlat
    candidates = []
    for service in ORTHOPHOTO_SERVICES:
        try:
            xml = _capabilities(service, folder)
        except OSError:
            continue
        cover = _coverage(xml)
        if cover and cover[0] <= lon <= cover[1] and cover[2] <= lat <= cover[3]:
            area = (cover[1] - cover[0]) * (cover[3] - cover[2])
            candidates.append((area, service, _licence(xml)))
    partial: list[tuple[float, Orthophoto]] = []
    for _, service, licence in sorted(candidates, key=lambda c: c[0]):
        bbox = (
            f"{south},{west},{north},{east}"
            if service.version == "1.3.0"
            else f"{west},{south},{east},{north}"
        )
        params = {
            "SERVICE": "WMS",
            "VERSION": service.version,
            "REQUEST": "GetMap",
            "LAYERS": service.layer,
            "STYLES": "",
            ("CRS" if service.version == "1.3.0" else "SRS"): "EPSG:4326",
            "BBOX": bbox,
            "WIDTH": str(pixels),
            "HEIGHT": str(pixels),
            "FORMAT": "image/jpeg",
        }
        name = f"{service.state}_{lon:.5f}_{lat:.5f}_{half_width_km:g}_{pixels}.jpg"
        path = folder / name
        if not path.exists():
            try:
                data = _get(f"{service.url}?{urllib.parse.urlencode(params)}", timeout_s)
            except OSError:
                continue
            if not data.startswith(b"\xff\xd8"):
                continue  # an error document instead of an image
            path.write_bytes(data)
        image = np.asarray(pil.open(io.BytesIO(path.read_bytes())).convert("RGB"))
        blank = float((image.min(axis=2) > 245).mean())
        photo = Orthophoto(image, (west, east, south, north), service, licence)
        if float(image.std()) >= 3.0 and blank <= 0.3:
            return photo
        if float(image.std()) >= 3.0 and blank < 0.6:
            partial.append((blank, photo))  # at a state border: part of the image is blank
    if partial:
        return min(partial, key=lambda p: p[0])[1]
    raise DataError(f"no orthophoto service covers ({lon:.4f}, {lat:.4f})")
