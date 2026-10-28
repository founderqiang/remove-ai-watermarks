"""Verify TC260 ``SecurityData`` label signatures (TC260-PG-202511A).

The guide lets a label producer record a ``SecurityData`` JSON object in the
label's ``ReservedCode1``: SM3withSM2 signatures (``PubSD`` entries of type
``DS``), the public key (``PubKey``), and content bindings (``Bindings``). Only
the producer side is read here. MiniMax writes it on every video in the local
corpus. Honor YOYO JPEGs carry nested RSA/content signatures with ``PubSd``
spelling in both reserved slots; those signatures are reported as unsupported.
Other measured producers use opaque vendor strings rather than ``SecurityData``.

Only the label signature is checked. MiniMax spells its type ``LabelMataData``
where the guide says ``Md``; both are accepted. MiniMax's second signature, over
its ``Bindings`` entry, also verifies (docs/research-sweep-2026-09.md), but no
verdict reads it, and the content hash inside that entry is not recomputed
because MiniMax names no content-selection method. The key travels inside the
file it signs, so a signature proves only integrity unless the key matches a
pinned signer in ``TC260_SIGNING_KEYS``.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from remove_ai_watermarks._internal import sm2
from remove_ai_watermarks._internal.tc260_producers import TC260_PRODUCERS

if TYPE_CHECKING:
    from collections.abc import Mapping

Status = Literal["verified", "failed", "absent", "unsupported"]

SM3_WITH_SM2 = "1.2.156.10197.1.501"


def _pinned(x: int) -> sm2.Point:
    point = sm2.lift_x(x, odd=False)
    if point is None:
        raise ValueError(f"pinned TC260 key x={x:#x} is not on the SM2 curve")
    return point


# Derived from the same producer rows used by metadata and visible-mark routing.
TC260_SIGNING_KEYS: dict[str, sm2.Point] = {
    row.signer: _pinned(row.signing_key_x)
    for row in TC260_PRODUCERS
    if row.signer is not None and row.signing_key_x is not None
}

_LABEL_TBS_TYPES = frozenset({"Md", "LabelMataData"})
# The producer's signed fields, in GB 45438 appendix E order ("Label" to value3).
_PRODUCER_FIELDS = ("Label", "ContentProducer", "ProduceID")


@dataclass(frozen=True)
class Tc260Signature:
    """Outcome of checking one label's ``SecurityData`` label signature."""

    label: Status
    signer: str | None

    def describe(self) -> str:
        if self.label == "verified":
            if self.signer:
                return f"TC260 label signature verified (pinned {self.signer} key)"
            return "TC260 label signature verified with its own embedded key (no pinned signer)"
        if self.label == "failed":
            return "TC260 label signature does not verify"
        if self.label == "unsupported":
            return (
                "TC260 SecurityData signature algorithm or signed-content selection "
                "is not supported; integrity unverified"
            )
        return "TC260 SecurityData without a label signature"


def _candidate_keys(key_hex: str) -> tuple[list[sm2.Point], str | None]:
    """The points a ``KeyValue`` can denote, and the pinned signer when it matches one."""
    try:
        raw = bytes.fromhex(key_hex)
    except ValueError:
        return [], None
    if len(raw) == 65 and raw[0] == 4:
        raw = raw[1:]
    if len(raw) == 33 and raw[0] == 0:
        raw = raw[1:]  # MiniMax pads a bare x coordinate with 00
    if len(raw) == 64:
        points = [(int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))]
    elif len(raw) == 33 and raw[0] in (2, 3):
        points = [p for p in [sm2.lift_x(int.from_bytes(raw[1:], "big"), odd=raw[0] == 3)] if p]
    elif len(raw) == 32:
        x = int.from_bytes(raw, "big")  # a bare x coordinate: either y may be meant
        points = [p for p in (sm2.lift_x(x, odd=False), sm2.lift_x(x, odd=True)) if p]
    else:
        return [], None
    for signer, pinned in TC260_SIGNING_KEYS.items():
        if any(p[0] == pinned[0] for p in points):
            return [pinned], signer
    return points, None


def _label_message(label: Mapping[str, str]) -> bytes:
    """Appendix A.1: ``"Label":"1","ContentProducer":"...","ProduceID":"..."``."""
    return ",".join(
        f"{json.dumps(field)}:{json.dumps(label.get(field, ''), ensure_ascii=False)}" for field in _PRODUCER_FIELDS
    ).encode()


def _object(value: object) -> dict[str, Any]:
    """``value`` as a JSON object, or empty when it is anything else."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _objects(value: object) -> list[dict[str, Any]]:
    """The JSON objects in a JSON array, or none when ``value`` is not an array."""
    items = cast("list[object]", value) if isinstance(value, list) else []
    return [cast("dict[str, Any]", item) for item in items if isinstance(item, dict)]


@functools.lru_cache(maxsize=256)
def _verified(point: sm2.Point, message: bytes, signature: bytes) -> bool:
    # get_ai_metadata and the identify verdict both check the same label, and
    # removal re-reads metadata; the pure-Python curve math runs once per triple.
    return sm2.verify(point, message, signature)


def check_tc260_signature(label: Mapping[str, str]) -> Tc260Signature | None:
    """Check the producer's ``SecurityData`` in ``ReservedCode1``; None when it has none."""
    try:
        parsed: object = json.loads(label.get("ReservedCode1", ""))
    except ValueError:
        return None
    top = _object(parsed)
    security = _object(top.get("SecurityData")) or top
    security_type = security.get("Type")
    if not isinstance(security_type, str) or security_type.strip() != "TC260PG":
        return None
    public = _objects(security.get("PubSD", security.get("PubSd")))
    keys = {
        str(entry.get("KeyID", 0)): _candidate_keys(str(entry.get("KeyValue", "")))
        for entry in reversed(public)  # reversed: the first entry per KeyID wins
        if entry.get("Type") == "PubKey" and entry.get("AlgID") == SM3_WITH_SM2
    }
    message = _label_message(label)
    results: list[bool] = []
    unsupported = False
    signer: str | None = None
    for entry in public:
        if entry.get("Type") != "DS":
            continue
        if entry.get("AlgID") != SM3_WITH_SM2 or _object(entry.get("TBSData")).get("Type") not in _LABEL_TBS_TYPES:
            unsupported = True
            continue
        points, key_signer = keys.get(str(entry.get("KeyID", 0)), ([], None))
        try:
            signature = bytes.fromhex(str(entry.get("Signature", "")))
        except ValueError:
            signature = b""
        verified = any(_verified(point, message, signature) for point in points)
        results.append(verified)
        if verified:
            signer = signer or key_signer
    if results:
        return Tc260Signature(label="verified" if any(results) else "failed", signer=signer)
    return Tc260Signature(label="unsupported" if unsupported else "absent", signer=None)
