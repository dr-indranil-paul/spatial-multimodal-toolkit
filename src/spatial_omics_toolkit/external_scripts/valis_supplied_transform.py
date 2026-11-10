from __future__ import annotations
from typing import Any
import numpy as np
from valis import registration

try:
    from valis import serial_rigid
except ImportError:
    serial_rigid = None
try:
    from valis import valtils
except ImportError:
    valtils = None


class SuppliedTransformerValis(registration.Valis):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        supplied = kwargs.get("do_rigid", {})
        if valtils is not None:
            self._supplied_matrices = {
                valtils.get_name(k): v["M"] for k, v in supplied.items() if "M" in v
            }
        else:
            self._supplied_matrices = {
                str(k): v["M"] for k, v in supplied.items() if "M" in v
            }
        if self._supplied_matrices and "matcher" not in kwargs and "matcher_for_sorting" not in kwargs:
            from valis import feature_detectors, feature_matcher

            # Supplied MI matrices do not use descriptor-derived transforms.
            # Keep the existing bounded sorting matcher for both VALIS stages:
            # its native same-class/detector policy avoids an unnecessary
            # rotation rematch whose estimated canvas can exceed max_image_dim_px.
            kwargs["matcher_for_sorting"] = feature_matcher.Matcher(
                match_filter_method=feature_matcher.DEFAULT_RANSAC_NAME,
                feature_detector=feature_detectors.VggFD(),
            )
            kwargs["matcher"] = feature_matcher.Matcher(
                match_filter_method=feature_matcher.DEFAULT_RANSAC_NAME,
                feature_detector=feature_detectors.VggFD(),
            )
        super().__init__(*args, **kwargs)

    def rigid_register_partial(self, tform_dict: Any = None) -> Any:
        rigid_registrar = super().rigid_register_partial(tform_dict=tform_dict)
        if rigid_registrar is not False and rigid_registrar is not None:
            if (
                hasattr(self, "rigid_reg_kwargs")
                and "transformer" in self.rigid_reg_kwargs
            ):
                rigid_registrar.transformer = self.rigid_reg_kwargs["transformer"]
        return rigid_registrar

    def extract_rigid_transforms_from_serial_rigid(
        self, rigid_registrar: Any
    ) -> tuple[
        dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray,
        dict[str, dict[str, np.ndarray]],
    ]:
        rigid_tforms, shape, cropped, cropped_shape, matches = (
            super().extract_rigid_transforms_from_serial_rigid(rigid_registrar)
        )
        for name, M in self._supplied_matrices.items():
            if name in rigid_tforms:
                if name in rigid_registrar.img_obj_dict:
                    supplied_M = rigid_registrar.img_obj_dict[name].M
                    rigid_tforms[name] = supplied_M
                    cropped[name] = supplied_M
        return rigid_tforms, shape, cropped, cropped_shape, matches


if serial_rigid is not None:
    _ORIGINAL_REMATCH = serial_rigid.SerialRigidRegistrar.rematch

    def _rematch_keeping_sorting_matches(
        self: Any,
        matcher_obj: Any,
        valis_obj: Any = None,
        keep_unfiltered: bool = False,
    ) -> Any:
        """Keep sorting matches if an explicitly requested optional rematch degenerates."""
        try:
            return _ORIGINAL_REMATCH(
                self, matcher_obj, valis_obj=valis_obj, keep_unfiltered=keep_unfiltered
            )
        except np.linalg.LinAlgError as error:
            print(
                f"WARNING: optional VALIS re-match was degenerate ({error}); keeping earlier matches"
            )
            return None

    serial_rigid.SerialRigidRegistrar.rematch = _rematch_keeping_sorting_matches
