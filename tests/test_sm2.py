"""SM3/SM2 verification and TC260 SecurityData label signatures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from remove_ai_watermarks._internal import sm2, tc260_signature
from remove_ai_watermarks._internal.tc260_signature import check_tc260_signature
from remove_ai_watermarks.metadata import aigc_label

ROOT = Path(__file__).resolve().parents[1]
MINIMAX_VIDEOS = (
    ROOT / "data" / "fixtures" / "provenance" / "higgsfield-hailuo-2-3.mp4",
    ROOT / "data" / "fixtures" / "provenance" / "higgsfield-minimax-h3.mp4",
)

# OpenSSL test/recipes/30-test_evp_data/evppkey_sm2.txt, key SM2_key1: the public
# point of its PKCS#8 key and two "Verify" rows (digest e, DER signature r, s).
OPENSSL_KEY = (
    0x393A2AD9E27E43ACAAAB85A19D3B85591E14406157175ADDD37BF77FF0CAF9ED,
    0x37FCF3A8B4F55C038523261FCA6636AFE6A7AFA2A1482A606F5CB3A18B2E2A1E,
)
OPENSSL_VERIFY = (
    (
        "D7AD397F6FFA5D4F7F11E7217F241607DC30618C236D2C09C1B9EA8FDADEE2E8",
        0xAB1DB64DE7C40EDBDE6651C9B8EBDB804673DB836E5D5C7FE15DCF9ED2725037,
        0xEBA714451FF69B0BB930B379E192E7CD5FA6E3C41C7FBD8303B799AB54A54621,
    ),
    (
        "B1139602C6ECC9E15E2F3F9C635A1AFE737058BC15387479C1EA0D0B3D90E9E5",
        0xE6E0414EBD3A656C35602AF14AB20287DBF30D57AF75C49A188ED4B42391F224,
        0x2F54F277C606F4605E1CE9514947FFDDF94C67A539804A4ED17F852288BDBE2E,
    ),
)


def _minimax_label() -> dict[str, str]:
    label = aigc_label(MINIMAX_VIDEOS[0])
    assert label is not None
    return label


def _with_security(label: dict[str, str], edit: object) -> dict[str, str]:
    security = json.loads(label["ReservedCode1"])
    edit(security["SecurityData"])  # type: ignore[operator]
    return {**label, "ReservedCode1": json.dumps(security, separators=(",", ":"))}


class TestSm3:
    @pytest.mark.parametrize(
        ("message", "digest"),
        [
            (b"abc", "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0"),
            (b"abcd" * 16, "debe9ff92275b8a138604889c18e5a4d6fdb70e5387e5765293dcba39c0c5732"),
        ],
    )
    def test_standard_vectors(self, message: bytes, digest: str) -> None:
        # GB/T 32905-2016 appendix A examples 1 and 2.
        assert sm2.sm3(message).hex() == digest


class TestSm2:
    @pytest.mark.parametrize(("digest", "r", "s"), OPENSSL_VERIFY)
    def test_openssl_vectors_verify(self, digest: str, r: int, s: int) -> None:
        assert sm2.verify_digest(OPENSSL_KEY, bytes.fromhex(digest), r, s)

    @pytest.mark.parametrize(("digest", "r", "s"), OPENSSL_VERIFY)
    def test_a_changed_digest_or_signature_fails(self, digest: str, r: int, s: int) -> None:
        changed = bytes.fromhex(digest)[:-1] + b"\x00"
        assert not sm2.verify_digest(OPENSSL_KEY, changed, r, s)
        assert not sm2.verify_digest(OPENSSL_KEY, bytes.fromhex(digest), r, s ^ 1)

    def test_lift_x_returns_the_requested_parity(self) -> None:
        for odd in (False, True):
            point = sm2.lift_x(OPENSSL_KEY[0], odd=odd)
            assert point is not None
            assert sm2.on_curve(point)
            assert (point[1] & 1) == odd
        assert OPENSSL_KEY in {sm2.lift_x(OPENSSL_KEY[0], odd=False), sm2.lift_x(OPENSSL_KEY[0], odd=True)}

    def test_an_off_curve_key_is_rejected(self) -> None:
        digest, r, s = OPENSSL_VERIFY[0]
        assert not sm2.verify_digest((OPENSSL_KEY[0], OPENSSL_KEY[1] + 1), bytes.fromhex(digest), r, s)
        # Congruent to the real key mod P, so the point arithmetic alone would accept it.
        assert not sm2.verify_digest((OPENSSL_KEY[0] + sm2.P, OPENSSL_KEY[1]), bytes.fromhex(digest), r, s)


class TestTc260Signature:
    @pytest.mark.parametrize("path", MINIMAX_VIDEOS, ids=lambda p: p.name)
    def test_real_minimax_labels_verify_against_the_pinned_key(self, path: Path) -> None:
        label = aigc_label(path)
        assert label is not None
        result = check_tc260_signature(label)
        assert result == tc260_signature.Tc260Signature(label="verified", signer="MiniMax")
        assert result.describe() == "TC260 label signature verified (pinned MiniMax key)"

    def test_a_changed_produce_id_fails_the_label_signature(self) -> None:
        label = _minimax_label()
        produce_id = label["ProduceID"]
        altered = {**label, "ProduceID": produce_id[:-1] + ("0" if produce_id[-1] != "0" else "1")}
        result = check_tc260_signature(altered)
        assert result is not None
        assert (result.label, result.signer) == ("failed", None)

    def test_a_repeated_check_verifies_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[bytes] = []
        real_verify = sm2.verify

        def counting(point: sm2.Point, message: bytes, signature: bytes) -> bool:
            calls.append(message)
            return real_verify(point, message, signature)

        tc260_signature._verified.cache_clear()
        monkeypatch.setattr(sm2, "verify", counting)
        label = _minimax_label()
        first, second = check_tc260_signature(label), check_tc260_signature(dict(label))
        assert first == second
        assert first is not None
        assert first.label == "verified"
        assert len(calls) == 1

    def test_an_unpinned_key_proves_integrity_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(tc260_signature, "TC260_SIGNING_KEYS", {})
        result = check_tc260_signature(_minimax_label())
        assert result is not None
        assert (result.label, result.signer) == ("verified", None)
        assert "no pinned signer" in result.describe()

    def test_a_pinned_signer_is_checked_with_the_pinned_point(self) -> None:
        # The file's own parity claim is untrusted: an odd-y spelling of MiniMax's
        # x coordinate still verifies, because the pinned even-y point is used.
        def edit(security: dict[str, object]) -> None:
            for entry in security["PubSD"]:  # type: ignore[union-attr]
                if entry["Type"] == "PubKey":
                    entry["KeyValue"] = "03" + entry["KeyValue"][-64:]

        result = check_tc260_signature(_with_security(_minimax_label(), edit))
        assert result is not None
        assert (result.label, result.signer) == ("verified", "MiniMax")

    def test_the_first_key_with_an_id_wins(self) -> None:
        def edit(security: dict[str, object]) -> None:
            foreign = {
                "Type": "PubKey",
                "AlgID": tc260_signature.SM3_WITH_SM2,
                "KeyValue": "04" + f"{OPENSSL_KEY[0]:064x}{OPENSSL_KEY[1]:064x}",
            }
            security["PubSD"].append(foreign)  # type: ignore[union-attr]

        result = check_tc260_signature(_with_security(_minimax_label(), edit))
        assert result is not None
        assert (result.label, result.signer) == ("verified", "MiniMax")

    def test_a_foreign_key_does_not_verify_the_label(self) -> None:
        def edit(security: dict[str, object]) -> None:
            for entry in security["PubSD"]:  # type: ignore[union-attr]
                if entry["Type"] == "PubKey":
                    entry["KeyValue"] = "04" + f"{OPENSSL_KEY[0]:064x}{OPENSSL_KEY[1]:064x}"

        result = check_tc260_signature(_with_security(_minimax_label(), edit))
        assert result is not None
        assert (result.label, result.signer) == ("failed", None)

    @pytest.mark.parametrize(
        "reserved",
        [
            "",
            "S7Wr/Evf+Mq+T3SWymUoO7naAs0WPNLrahOqG9WzR+o=",
            '{"SecurityData":{"Type":"OTHER","Version":1}}',
            "[1, 2]",
        ],
    )
    def test_reserved_codes_without_security_data_are_not_signatures(self, reserved: str) -> None:
        assert check_tc260_signature({"Label": "1", "ReservedCode1": reserved}) is None

    def test_direct_tc260pg_object_with_padded_type_verifies(self) -> None:
        label = _minimax_label()
        security = json.loads(label["ReservedCode1"])["SecurityData"]
        security["Type"] = " TC260PG"
        direct_label = {**label, "ReservedCode1": json.dumps(security, separators=(",", ":"))}
        result = check_tc260_signature(direct_label)
        assert result == tc260_signature.Tc260Signature(label="verified", signer="MiniMax")


def _labeled_png(path: Path, label: dict[str, str], text: dict[str, str] | None = None) -> Path:
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text("AIGC", json.dumps(label, separators=(",", ":")))
    for key, value in (text or {}).items():
        info.add_text(key, value)
    Image.new("RGB", (64, 64), (90, 120, 150)).save(path, pnginfo=info)
    return path


class TestIdentifyIntegration:
    def test_a_verified_pinned_signature_is_a_high_confidence_signal(self, tmp_path: Path) -> None:
        from remove_ai_watermarks.identify import identify

        report = identify(
            _labeled_png(tmp_path / "signed.png", _minimax_label()), check_visible=False, check_invisible=False
        )
        signatures = [signal for signal in report.signals if signal.name == "aigc_signature"]
        assert [(signal.detail, signal.confidence) for signal in signatures] == [
            ("TC260 label signature verified (pinned MiniMax key)", "high")
        ]
        assert not any("does not verify" in caveat for caveat in report.caveats)

    def test_pinned_signer_supplies_vendor_when_the_producer_lookup_is_unavailable(self, tmp_path, monkeypatch):
        import remove_ai_watermarks.identify as identification

        path = _labeled_png(
            tmp_path / "mixed.png",
            _minimax_label(),
            {"XML:com.adobe.xmp": "<x:Iptc4xmpExt:AISystemUsed>OpenAI</x:Iptc4xmpExt:AISystemUsed>"},
        )
        monkeypatch.setattr(identification, "producer_for_code", lambda code: None)
        report = identification.identify(path, check_visible=False, check_invisible=False)
        assert any(s.name == "aigc_signature" for s in report.signals)
        assert any("MiniMax" in clash and "OpenAI" in clash for clash in report.integrity_clashes)

        # The same real signature under its embedded key cannot name a vendor.
        monkeypatch.setattr(tc260_signature, "TC260_SIGNING_KEYS", {})
        unpinned = identification.identify(path, check_visible=False, check_invisible=False)
        assert not any(s.name == "aigc_signature" for s in unpinned.signals)
        assert not unpinned.integrity_clashes

    def test_a_failed_signature_is_a_caveat_not_a_signal(self, tmp_path: Path) -> None:
        from remove_ai_watermarks.identify import identify

        label = _minimax_label()
        altered = {**label, "ProduceID": "0" * len(label["ProduceID"])}
        report = identify(_labeled_png(tmp_path / "altered.png", altered), check_visible=False, check_invisible=False)
        assert not any(signal.name == "aigc_signature" for signal in report.signals)
        assert any("does not verify" in caveat for caveat in report.caveats)

    def test_the_metadata_record_path_matches_the_file_path(self, tmp_path: Path) -> None:
        from remove_ai_watermarks.identify import identify, identify_metadata_record
        from remove_ai_watermarks.metadata import get_ai_metadata
        from remove_ai_watermarks.metadata_record import collect_metadata_record

        path = _labeled_png(tmp_path / "signed.png", _minimax_label())
        record = json.loads(json.dumps(collect_metadata_record(path)))
        from_record = identify_metadata_record(record, path=path)
        from_file = identify(path, check_visible=False, check_invisible=False)
        assert from_record.signals == from_file.signals
        assert from_record.caveats == from_file.caveats
        assert get_ai_metadata(path)["aigc_signature"] == "TC260 label signature verified (pinned MiniMax key)"

    def test_a_minimax_video_carries_the_signature_marker(self) -> None:
        from remove_ai_watermarks.video import identify_video

        report = identify_video(MINIMAX_VIDEOS[0], check_visible=False)
        assert report.metadata_markers["aigc_signature"] == "TC260 label signature verified (pinned MiniMax key)"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
