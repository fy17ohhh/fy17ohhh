from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time
from astropy.utils import iers

# Keep the workflow fully self-contained. Astropy will use bundled Earth-rotation
# data instead of attempting a network download during every GitHub Actions run.
iers.conf.auto_download = False
iers.conf.auto_max_age = None

README = Path(__file__).resolve().parents[1] / "README.md"
START_MARKER = "<!-- DAILY_SKY_START -->"
END_MARKER = "<!-- DAILY_SKY_END -->"

TZ = ZoneInfo("Asia/Shanghai")
LOCATION = EarthLocation(lat=39.9042 * u.deg, lon=116.4074 * u.deg, height=44 * u.m)

CATEGORY_ORDER = (
    "star",
    "planet",
    "constellation",
    "galaxy",
    "cluster_nebula",
    "moon",
)

CATEGORY_LABEL = {
    "star": "Star",
    "planet": "Planet",
    "constellation": "Constellation",
    "galaxy": "Galaxy",
    "cluster_nebula": "Cluster / Nebula",
    "moon": "Moon",
}

SYMBOL = {
    "star": "✦",
    "planet": "●",
    "constellation": "✧",
    "galaxy": "◇",
    "cluster_nebula": "✺",
    "moon": "☾",
}

# Objects need to be high enough to be pleasant to observe, not merely above
# the mathematical horizon. Deep-sky objects also require a darker sky.
MIN_ALTITUDE = {
    "star": 15.0,
    "planet": 12.0,
    "constellation": 20.0,
    "galaxy": 25.0,
    "cluster_nebula": 20.0,
    "moon": 10.0,
}

MAX_SUN_ALTITUDE = {
    "star": -6.0,
    "planet": -6.0,
    "constellation": -8.0,
    "galaxy": -12.0,
    "cluster_nebula": -12.0,
    "moon": -6.0,
}


@dataclass(frozen=True)
class FixedObject:
    name: str
    category: str
    ra: str
    dec: str
    priority: float
    magnitude: float | None = None
    distance: str | None = None

    @property
    def coord(self) -> SkyCoord:
        return SkyCoord(
            ra=self.ra,
            dec=self.dec,
            unit=(u.hourangle, u.deg),
            frame="icrs",
        )


@dataclass
class Candidate:
    name: str
    category: str
    score: float
    altitude: float
    best_index: int
    distance: str | None = None
    illumination: float | None = None


FIXED_OBJECTS = [
    # Bright stars visible from Beijing during different seasons.
    FixedObject("Sirius", "star", "06h45m08.9s", "-16d42m58s", 10, -1.46, "8.6 ly away"),
    FixedObject("Arcturus", "star", "14h15m39.7s", "+19d10m57s", 9, -0.05, "36.7 ly away"),
    FixedObject("Vega", "star", "18h36m56.3s", "+38d47m01s", 10, 0.03, "25 ly away"),
    FixedObject("Capella", "star", "05h16m41.4s", "+45d59m53s", 9, 0.08, "42.9 ly away"),
    FixedObject("Rigel", "star", "05h14m32.3s", "-08d12m06s", 8, 0.13, "~860 ly away"),
    FixedObject("Procyon", "star", "07h39m18.1s", "+05d13m30s", 8, 0.34, "11.5 ly away"),
    FixedObject("Betelgeuse", "star", "05h55m10.3s", "+07d24m25s", 9, 0.42, "~640 ly away"),
    FixedObject("Altair", "star", "19h50m47.0s", "+08d52m06s", 9, 0.76, "16.7 ly away"),
    FixedObject("Aldebaran", "star", "04h35m55.2s", "+16d30m33s", 8, 0.85, "65 ly away"),
    FixedObject("Spica", "star", "13h25m11.6s", "-11d09m41s", 8, 0.98, "250 ly away"),
    FixedObject("Antares", "star", "16h29m24.5s", "-26d25m55s", 8, 1.06, "~550 ly away"),
    FixedObject("Pollux", "star", "07h45m18.9s", "+28d01m34s", 7, 1.14, "33.7 ly away"),
    FixedObject("Deneb", "star", "20h41m25.9s", "+45d16m49s", 9, 1.25, "~2,600 ly away"),
    FixedObject("Regulus", "star", "10h08m22.3s", "+11d58m02s", 7, 1.35, "79 ly away"),

    # Constellation anchors are representative central/recognizable positions.
    FixedObject("Orion", "constellation", "05h35m00s", "+00d00m00s", 10),
    FixedObject("Ursa Major", "constellation", "11h00m00s", "+55d00m00s", 9),
    FixedObject("Cassiopeia", "constellation", "01h00m00s", "+60d00m00s", 9),
    FixedObject("Cygnus", "constellation", "20h30m00s", "+42d00m00s", 10),
    FixedObject("Lyra", "constellation", "18h50m00s", "+36d00m00s", 9),
    FixedObject("Scorpius", "constellation", "16h50m00s", "-30d00m00s", 9),
    FixedObject("Sagittarius", "constellation", "19h00m00s", "-25d00m00s", 9),
    FixedObject("Gemini", "constellation", "07h00m00s", "+22d00m00s", 8),
    FixedObject("Leo", "constellation", "10h40m00s", "+15d00m00s", 8),
    FixedObject("Taurus", "constellation", "04h30m00s", "+18d00m00s", 9),
    FixedObject("Pegasus", "constellation", "22h40m00s", "+20d00m00s", 8),
    FixedObject("Aquila", "constellation", "19h40m00s", "+05d00m00s", 8),

    # Galaxies: a compact visual catalogue rather than an exhaustive database.
    FixedObject("Andromeda Galaxy · M31", "galaxy", "00h42m44.3s", "+41d16m09s", 10, 3.44, "2.54 Mly away"),
    FixedObject("Triangulum Galaxy · M33", "galaxy", "01h33m50.9s", "+30d39m36s", 8, 5.72, "2.73 Mly away"),
    FixedObject("Bode's Galaxy · M81", "galaxy", "09h55m33.2s", "+69d03m55s", 9, 6.94, "11.8 Mly away"),
    FixedObject("Cigar Galaxy · M82", "galaxy", "09h55m52.2s", "+69d40m47s", 9, 8.4, "12 Mly away"),
    FixedObject("Whirlpool Galaxy · M51", "galaxy", "13h29m52.7s", "+47d11m43s", 10, 8.4, "~23 Mly away"),
    FixedObject("Pinwheel Galaxy · M101", "galaxy", "14h03m12.6s", "+54d20m57s", 8, 7.9, "~22 Mly away"),
    FixedObject("Sombrero Galaxy · M104", "galaxy", "12h39m59.4s", "-11d37m23s", 9, 8.0, "31 Mly away"),
    FixedObject("Black Eye Galaxy · M64", "galaxy", "12h56m43.7s", "+21d40m58s", 8, 8.5, "17 Mly away"),

    # Open/globular clusters and bright nebulae.
    FixedObject("Orion Nebula · M42", "cluster_nebula", "05h35m17.3s", "-05d23m28s", 10, 4.0, "~1,340 ly away"),
    FixedObject("Pleiades · M45", "cluster_nebula", "03h47m24s", "+24d07m00s", 10, 1.6, "444 ly away"),
    FixedObject("Beehive Cluster · M44", "cluster_nebula", "08h40m24s", "+19d41m00s", 9, 3.1, "~580 ly away"),
    FixedObject("Hercules Cluster · M13", "cluster_nebula", "16h41m41.2s", "+36d27m36s", 10, 5.8, "22,200 ly away"),
    FixedObject("Double Cluster", "cluster_nebula", "02h20m00s", "+57d08m00s", 9, 4.3, "~7,500 ly away"),
    FixedObject("Ring Nebula · M57", "cluster_nebula", "18h53m35s", "+33d01m45s", 9, 8.8, "~2,300 ly away"),
    FixedObject("Dumbbell Nebula · M27", "cluster_nebula", "19h59m36.3s", "+22d43m16s", 9, 7.5, "~1,360 ly away"),
    FixedObject("Lagoon Nebula · M8", "cluster_nebula", "18h03m37s", "-24d23m12s", 9, 6.0, "~4,100 ly away"),
    FixedObject("Trifid Nebula · M20", "cluster_nebula", "18h02m23s", "-23d01m48s", 8, 6.3, "~5,200 ly away"),
    FixedObject("Wild Duck Cluster · M11", "cluster_nebula", "18h51m05s", "-06d16m12s", 8, 5.8, "~6,100 ly away"),
]

PLANETS = {
    "venus": ("Venus", 10),
    "jupiter": ("Jupiter", 10),
    "saturn": ("Saturn", 9),
    "mars": ("Mars", 8),
    "mercury": ("Mercury", 5),
}


def build_time_grid(local_date):
    start = datetime.combine(local_date, time(18, 0), tzinfo=TZ)
    # 18:00 through 00:30 in 30-minute increments.
    datetimes = [start + timedelta(minutes=30 * i) for i in range(14)]
    return Time(datetimes)


def stable_choice(items: list[Candidate], date_key: str) -> Candidate:
    """Choose among near-equal top candidates without sacrificing visibility.

    Seasonal sky geometry often makes the same object #1 for weeks. We keep all
    candidates within 8 score points of the best and use a deterministic hash of
    the Beijing date to add gentle variety while remaining astronomically sane.
    """
    items = sorted(items, key=lambda item: item.score, reverse=True)
    best_score = items[0].score
    near_top = [item for item in items if item.score >= best_score - 8.0][:3]
    digest = hashlib.sha256(date_key.encode("utf-8")).digest()
    return near_top[int.from_bytes(digest[:4], "big") % len(near_top)]


def planet_distance(coord: SkyCoord) -> str:
    au = coord.distance.to_value(u.au)
    return f"{au:.2f} AU from Earth"


def moon_distance(coord: SkyCoord) -> str:
    km = coord.distance.to_value(u.km)
    return f"{km / 1000:.0f},000 km away"


def evaluate_fixed(obj: FixedObject, times: Time, sun_alt, moon_coords, moon_alt, moon_illum) -> Candidate | None:
    altaz = obj.coord.transform_to(AltAz(obstime=times, location=LOCATION))
    alt = altaz.alt.to_value(u.deg)

    valid = (
        (sun_alt <= MAX_SUN_ALTITUDE[obj.category])
        & (alt >= MIN_ALTITUDE[obj.category])
    )
    if not np.any(valid):
        return None

    masked_alt = np.where(valid, alt, -999.0)
    idx = int(np.argmax(masked_alt))
    max_alt = float(alt[idx])
    visible_slots = int(np.count_nonzero(valid))

    score = max_alt + visible_slots * 1.4 + obj.priority * 3.0
    if obj.magnitude is not None:
        score += max(0.0, 8.0 - obj.magnitude) * 1.4

    # Bright Moon can noticeably degrade deep-sky viewing. Penalize objects that
    # are close to it while the Moon is above the horizon and strongly lit.
    if obj.category in {"galaxy", "cluster_nebula"} and moon_alt[idx] > 0:
        separation = obj.coord.separation(moon_coords[idx].icrs).deg
        proximity = max(0.0, (75.0 - separation) / 75.0)
        score -= float(moon_illum[idx]) * proximity * 24.0

    return Candidate(
        name=obj.name,
        category=obj.category,
        score=score,
        altitude=max_alt,
        best_index=idx,
        distance=obj.distance,
    )


def evaluate_planet(body: str, display_name: str, priority: float, times: Time, sun_alt) -> Candidate | None:
    coords = get_body(body, times, location=LOCATION)
    alt = coords.transform_to(AltAz(obstime=times, location=LOCATION)).alt.to_value(u.deg)

    valid = (
        (sun_alt <= MAX_SUN_ALTITUDE["planet"])
        & (alt >= MIN_ALTITUDE["planet"])
    )
    if not np.any(valid):
        return None

    idx = int(np.argmax(np.where(valid, alt, -999.0)))
    score = float(alt[idx]) + int(np.count_nonzero(valid)) * 1.5 + priority * 3.0

    return Candidate(
        name=display_name,
        category="planet",
        score=score,
        altitude=float(alt[idx]),
        best_index=idx,
        distance=planet_distance(coords[idx]),
    )


def evaluate_moon(times: Time, sun_alt, moon_coords, moon_alt, moon_illum) -> Candidate | None:
    valid = (
        (sun_alt <= MAX_SUN_ALTITUDE["moon"])
        & (moon_alt >= MIN_ALTITUDE["moon"])
    )
    if not np.any(valid):
        return None

    idx = int(np.argmax(np.where(valid, moon_alt, -999.0)))
    illumination = float(moon_illum[idx])
    score = float(moon_alt[idx]) + int(np.count_nonzero(valid)) * 1.5 + 20.0

    return Candidate(
        name="Moon",
        category="moon",
        score=score,
        altitude=float(moon_alt[idx]),
        best_index=idx,
        distance=moon_distance(moon_coords[idx]),
        illumination=illumination,
    )


def render(candidate: Candidate, times: Time, local_date) -> str:
    best_dt = times[candidate.best_index].to_datetime(timezone=TZ)
    date_text = local_date.strftime("%d %b %Y")
    time_text = best_dt.strftime("%H:%M")

    parts = [
        f"{SYMBOL[candidate.category]} {date_text}",
        f"Tonight: {candidate.name} — {CATEGORY_LABEL[candidate.category]}",
        f"best around {time_text} UTC+8",
        f"{candidate.altitude:.0f}° high",
    ]

    if candidate.category == "moon" and candidate.illumination is not None:
        parts.append(f"{candidate.illumination * 100:.0f}% illuminated")

    if candidate.distance:
        parts.append(candidate.distance)

    return " · ".join(parts)


def replace_readme_line(line: str) -> None:
    text = README.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}",
        flags=re.DOTALL,
    )
    replacement = (
        f'{START_MARKER}\n'
        f'<p align="center"><sub>{line}</sub></p>\n'
        f'{END_MARKER}'
    )

    if not pattern.search(text):
        raise RuntimeError("README markers are missing; refusing to overwrite README.md")

    README.write_text(pattern.sub(replacement, text), encoding="utf-8")


def main() -> None:
    local_date = datetime.now(TZ).date()
    times = build_time_grid(local_date)
    frame = AltAz(obstime=times, location=LOCATION)

    sun_coords = get_sun(times)
    sun_alt = sun_coords.transform_to(frame).alt.to_value(u.deg)

    moon_coords = get_body("moon", times, location=LOCATION)
    moon_alt = moon_coords.transform_to(frame).alt.to_value(u.deg)
    elongation = moon_coords.separation(sun_coords).to_value(u.deg)
    moon_illum = (1.0 - np.cos(np.deg2rad(elongation))) / 2.0

    candidates: list[Candidate] = []

    for obj in FIXED_OBJECTS:
        candidate = evaluate_fixed(obj, times, sun_alt, moon_coords, moon_alt, moon_illum)
        if candidate:
            candidates.append(candidate)

    for body, (display_name, priority) in PLANETS.items():
        candidate = evaluate_planet(body, display_name, priority, times, sun_alt)
        if candidate:
            candidates.append(candidate)

    moon_candidate = evaluate_moon(times, sun_alt, moon_coords, moon_alt, moon_illum)
    if moon_candidate:
        candidates.append(moon_candidate)

    if not candidates:
        raise RuntimeError("No observable candidates found for tonight")

    category = CATEGORY_ORDER[local_date.toordinal() % len(CATEGORY_ORDER)]
    category_candidates = [c for c in candidates if c.category == category]

    # If today's rotating category has nothing reasonably observable tonight,
    # select the best object across all remaining categories.
    pool = category_candidates if category_candidates else candidates
    selected = stable_choice(pool, f"{local_date.isoformat()}:{category}")

    line = render(selected, times, local_date)
    replace_readme_line(line)
    print(line)


if __name__ == "__main__":
    main()
