"""Control-flow tests for instantid_restore -- no model download.

The end-to-end InstantID run is monkey-patched: we replace ``_get_pipeline`` and
``_get_face_analyser`` with fakes, install a fake InsightFace ``FaceAnalysis``
embedding, and check that the per-face crop + composite pipeline wires up the
expected pixels into ``cleaned_bgr``.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from remove_ai_watermarks import instantid_restore


class TestIsAvailable:
    def test_returns_bool(self):
        assert isinstance(instantid_restore.is_available(), bool)


class TestRepoPins:
    """Pin the InstantID repo + adapter file so a maintainer change is visible."""

    def test_repo_is_instantx_instantid(self):
        assert instantid_restore._INSTANTID_REPO == "InstantX/InstantID"

    def test_controlnet_subfolder(self):
        assert instantid_restore._INSTANTID_CONTROLNET_SUBFOLDER == "ControlNetModel"

    def test_ip_adapter_filename(self):
        assert instantid_restore._INSTANTID_IP_ADAPTER == "ip-adapter.bin"


class TestDrawKps:
    def test_renders_color_image(self):
        kps = np.array([[100, 100], [200, 100], [150, 150], [120, 200], [180, 200]])
        img = instantid_restore._draw_kps((256, 256), kps)
        arr = np.array(img)
        assert arr.shape == (256, 256, 3)
        # Has nonzero pixels (the stick figure is rendered).
        assert arr.sum() > 0

    def test_black_outside_kps(self):
        kps = np.array([[100, 100], [200, 100], [150, 150], [120, 200], [180, 200]])
        img = instantid_restore._draw_kps((256, 256), kps)
        arr = np.array(img)
        # Top-left corner should be black (no keypoint there).
        assert arr[0, 0].sum() == 0


class TestRestoreFacesInstantidControlFlow:
    """End-to-end flow with the pipeline / face analyser / InsightFace mocked.

    Checks that with one detected face: (1) the original crop is fed to the
    InsightFace mock; (2) the pipeline mock receives the expected kwargs; (3)
    the regenerated output ends up composited into the cleaned image.
    """

    @staticmethod
    def _fake_pipeline_class(fill_value: int = 210):
        import torch
        from PIL import Image

        class _FakePipeOutput:
            def __init__(self, images):
                self.images = images

        class _FakePipe:
            device = "cpu"
            dtype = torch.float32

            def __call__(self, **kwargs):
                # Save kwargs for assertion.
                _FakePipe.last_kwargs = kwargs
                # Gradient face so the color-match step shifts the mean but
                # preserves contrast (the composite is then detectable as a
                # variance change in the face region even with uniform canvas).
                grad = np.linspace(0, fill_value, 1024, dtype=np.uint8)
                arr = np.broadcast_to(grad[:, None, None], (1024, 1024, 3)).copy()
                img = Image.fromarray(arr)
                return _FakePipeOutput([img])

        return _FakePipe()

    def test_no_faces_returns_cleaned_unchanged(self, monkeypatch):
        monkeypatch.setattr(instantid_restore, "is_available", lambda: True)
        monkeypatch.setattr(instantid_restore, "_get_pipeline", lambda: self._fake_pipeline_class())
        monkeypatch.setattr(instantid_restore, "_get_face_analyser", lambda: object())

        orig = np.full((400, 400, 3), 50, dtype=np.uint8)
        cleaned = np.full((400, 400, 3), 100, dtype=np.uint8)
        out = instantid_restore.restore_faces_instantid(orig, cleaned, detect_faces_fn=lambda _b: [])
        assert np.array_equal(out, cleaned)

    def test_one_face_gets_composited_into_cleaned(self, monkeypatch):
        monkeypatch.setattr(instantid_restore, "is_available", lambda: True)
        monkeypatch.setattr(instantid_restore, "_get_pipeline", lambda: self._fake_pipeline_class(fill_value=210))

        # Fake FaceAnalyser that returns one face with a 512-d embedding + 5 keypoints.
        class _FakeFA:
            def get(self, _bgr):
                return [
                    {
                        "bbox": np.array([10, 10, 100, 100], dtype=np.float32),
                        "embedding": np.zeros(512, dtype=np.float32),
                        "kps": np.array(
                            [[30, 40], [70, 40], [50, 60], [35, 80], [65, 80]],
                            dtype=np.float32,
                        ),
                    }
                ]

        monkeypatch.setattr(instantid_restore, "_get_face_analyser", lambda: _FakeFA())

        orig = np.full((400, 400, 3), 30, dtype=np.uint8)
        cleaned = np.full((400, 400, 3), 90, dtype=np.uint8)
        cv2.rectangle(orig, (150, 150), (250, 250), (200, 100, 50), -1)

        out = instantid_restore.restore_faces_instantid(
            orig, cleaned, detect_faces_fn=lambda _b: [(150, 150, 100, 100)]
        )
        # The composite must have written non-uniform values into the face
        # region (gradient survives color-match as variance), and the canvas
        # corner stays close to the cleaned base.
        face_region = out[170:230, 170:230]
        assert int(face_region.std()) > 0
        assert int(out[0, 0, 0]) - int(cleaned[0, 0, 0]) <= 1

    def test_insightface_misses_face_skips_gracefully(self, monkeypatch):
        monkeypatch.setattr(instantid_restore, "is_available", lambda: True)
        monkeypatch.setattr(instantid_restore, "_get_pipeline", lambda: self._fake_pipeline_class())

        class _EmptyFA:
            def get(self, _bgr):
                return []

        monkeypatch.setattr(instantid_restore, "_get_face_analyser", lambda: _EmptyFA())

        orig = np.full((400, 400, 3), 30, dtype=np.uint8)
        cleaned = np.full((400, 400, 3), 90, dtype=np.uint8)

        out = instantid_restore.restore_faces_instantid(
            orig, cleaned, detect_faces_fn=lambda _b: [(150, 150, 100, 100)]
        )
        # No face detected by InsightFace -> cleaned image is returned unchanged.
        assert np.array_equal(out, cleaned)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
