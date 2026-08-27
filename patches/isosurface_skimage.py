from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# SnapMesh patch: the upstream repo depends on `torchmcubes`, a C++/CUDA extension that
# must be compiled against the local torch + CUDA toolkit — the single hardest install
# step on Windows. scikit-image ships a prebuilt, pure-wheel marching cubes that is more
# than fast enough for a 256^3 grid (well under a second), so we use it instead.
# Convention note: TripoSR builds its density grid from meshgrid(x, y, z, indexing="ij")
# flattened in C order, i.e. volume axis 0 = x, 1 = y, 2 = z. skimage returns vertices
# in (axis0, axis1, axis2) order, which is therefore already (x, y, z) — unlike
# torchmcubes, no [2, 1, 0] column swap is needed.
from skimage import measure


class IsosurfaceHelper(nn.Module):
    points_range: Tuple[float, float] = (0, 1)

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        raise NotImplementedError


class MarchingCubeHelper(IsosurfaceHelper):
    def __init__(self, resolution: int) -> None:
        super().__init__()
        self.resolution = resolution
        self._grid_vertices: Optional[torch.FloatTensor] = None

    @property
    def grid_vertices(self) -> torch.FloatTensor:
        if self._grid_vertices is None:
            # keep the vertices on CPU so that we can support very large resolution
            x, y, z = (
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
                torch.linspace(*self.points_range, self.resolution),
            )
            x, y, z = torch.meshgrid(x, y, z, indexing="ij")
            verts = torch.cat(
                [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1
            ).reshape(-1, 3)
            self._grid_vertices = verts
        return self._grid_vertices

    def forward(
        self,
        level: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        level = -level.view(self.resolution, self.resolution, self.resolution)
        volume = level.detach().cpu().numpy()

        # Guard: marching_cubes raises if the level never crosses the isosurface.
        lo, hi = float(volume.min()), float(volume.max())
        if not (lo < 0.0 < hi):
            raise ValueError(
                f"Isosurface level 0.0 outside volume range [{lo:.4f}, {hi:.4f}] — "
                "the model produced no crossable surface for this input."
            )

        verts, faces, _normals, _values = measure.marching_cubes(volume, level=0.0)
        v_pos = torch.from_numpy(verts.copy()).float() / (self.resolution - 1.0)
        t_pos_idx = torch.from_numpy(faces.astype(np.int64))
        return v_pos.to(level.device), t_pos_idx.to(level.device)
