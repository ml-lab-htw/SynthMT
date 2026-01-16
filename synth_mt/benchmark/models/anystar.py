import shutil
from pathlib import Path
from typing import Optional

from .stardist import StarDist


class AnyStar(StarDist):
    """
    AnyStar wrapper built on top of the generic StarDist (3D) class.

    Extended possible behavior:
    - If `model_dir` is provided, that folder is used.
    - Else, if `pretrained` (case-insensitive) == "anystar", this class ensures
      a local folder exists at 'models/AnyStar/anystar-mix/' and uses it.
    - Else, falls back to StarDist built-in 3D pretrained names.

    Notes:
    - Requires `gdown` to be installed if it needs to download the weights.

    Parameters
    ----------
    prob_thresh : float
        Softmax detection threshold. Lower detects more and vice versa.
        Default: 0.5
    nms_thresh : Optional[float]
        Non-max suppression threshold. Lower value suppresses more.
        Default: 0.3
    """

    # Google Drive folder for AnyStar release
    _ANYSTAR_DRIVE_ID = "1yiY_vBR2GQW9zJzgUPRWeIecN4ZnCi3c"

    def __init__(
        self,
        model_name="StarDist",
        pretrained="AnyStar",  # trigger auto-managed weights
        prob_thresh: float = 0.5,
        nms_thresh: float = 0.3,
        **kwargs,
    ):
        self._model_name_on_disk = "anystar-mix"
        super().__init__(
            model_name=model_name,
            pretrained=pretrained,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            **kwargs,
        )

    # ---- hook implementation ----
    def _prepare_model_dir(self) -> Optional[Path]:
        # If user provided a folder, use it
        if self._model_dir is not None:
            return self._model_dir

        # If the user asked for "AnyStar", ensure/download the local weights
        if (self.pretrained or "").lower() == "anystar":
            return self._ensure_anystar_weights()

        # Otherwise, fall back to built-in pretrained StarDist3D (handled by base)
        return None

    # ---- internal helpers ----
    def _ensure_anystar_weights(self) -> Path:
        dest_root = Path(self._save_dir)
        model_dir = dest_root / self._model_name_on_disk
        cfg = model_dir / "config.json"

        if not cfg.exists():
            # Download the whole Drive folder to a temp and move the inner model folder in place
            import gdown  # assumed installed

            dest_root.mkdir(parents=True, exist_ok=True)
            tmp = dest_root / f"__tmp_{self._model_name_on_disk}"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True, exist_ok=True)

            url = f"https://drive.google.com/drive/folders/{self._ANYSTAR_DRIVE_ID}"
            gdown.download_folder(url=url, output=str(tmp), quiet=False, use_cookies=False)

            candidates = list(tmp.rglob("config.json"))
            if not candidates:
                raise RuntimeError(
                    "AnyStar download did not contain a StarDist model (missing config.json)."
                )

            src = candidates[0].parent
            if model_dir.exists():
                shutil.rmtree(model_dir)
            shutil.move(str(src), str(model_dir))
            shutil.rmtree(tmp, ignore_errors=True)

        return model_dir
