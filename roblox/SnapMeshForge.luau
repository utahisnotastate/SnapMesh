--!strict
-- SnapMeshForge — builds a SnapMesh export (.luau data module) into a live MeshPart
-- using AssetService:CreateEditableMesh(). No asset uploads, no moderation queue:
-- the mesh is constructed from pure code, with per-vertex colors and smooth
-- area-weighted normals, at server boot or in Studio.
--
-- Usage:
--   local SnapMeshForge = require(path.to.SnapMeshForge)
--   local data = require(path.to.Server_Rack)     -- file produced by snapmesh.py
--   local part = SnapMeshForge.Build(data)
--   part.CFrame = CFrame.new(0, 0, 0)
--   part.Parent = workspace
--
-- Notes:
--   * The exported mesh has its feet at local Y=0 and is centered on X/Z; the
--     returned MeshPart's pivot is the part's bounding-box center (Roblox
--     convention), so place it with `SnapMeshForge.PlaceOnGround(part, position)`
--     if you want floor-accurate placement.
--   * Clones of the returned part share the same mesh content — scatter freely.

local AssetService = game:GetService("AssetService")

export type SnapMeshData = {
	Name: string,
	Verts: { number },   -- flat x,y,z triplets
	Colors: { number },  -- flat r,g,b triplets (0..1), one per vertex
	Tris: { number },    -- flat 1-based vertex index triplets
}

local SnapMeshForge = {}

function SnapMeshForge.Build(data: SnapMeshData): MeshPart
	local vertexCount = #data.Verts / 3
	local em = AssetService:CreateEditableMesh()

	-- 1. Vertices
	local vertexIds = table.create(vertexCount)
	local positions = table.create(vertexCount)
	for i = 1, vertexCount do
		local base = (i - 1) * 3
		local pos = Vector3.new(data.Verts[base + 1], data.Verts[base + 2], data.Verts[base + 3])
		positions[i] = pos
		vertexIds[i] = em:AddVertex(pos)
	end

	-- 2. Per-vertex colors
	local colorIds = table.create(vertexCount)
	for i = 1, vertexCount do
		local base = (i - 1) * 3
		colorIds[i] = em:AddColor(
			Color3.new(data.Colors[base + 1], data.Colors[base + 2], data.Colors[base + 3]),
			1
		)
	end

	-- 3. Smooth area-weighted vertex normals (accumulate face cross products)
	local accum = table.create(vertexCount)
	for i = 1, vertexCount do
		accum[i] = Vector3.zero
	end
	for t = 1, #data.Tris, 3 do
		local i1, i2, i3 = data.Tris[t], data.Tris[t + 1], data.Tris[t + 2]
		local p1, p2, p3 = positions[i1], positions[i2], positions[i3]
		local cross = (p2 - p1):Cross(p3 - p1) -- magnitude = 2*area (weighting)
		accum[i1] += cross
		accum[i2] += cross
		accum[i3] += cross
	end
	local normalIds = table.create(vertexCount)
	for i = 1, vertexCount do
		local n = accum[i]
		local unit = (n.Magnitude > 1e-8) and n.Unit or Vector3.yAxis
		normalIds[i] = em:AddNormal(unit)
	end

	-- 4. Triangles, binding colors + normals per face corner
	for t = 1, #data.Tris, 3 do
		local i1, i2, i3 = data.Tris[t], data.Tris[t + 1], data.Tris[t + 2]
		local faceId = em:AddTriangle(vertexIds[i1], vertexIds[i2], vertexIds[i3])
		em:SetFaceColors(faceId, { colorIds[i1], colorIds[i2], colorIds[i3] })
		em:SetFaceNormals(faceId, { normalIds[i1], normalIds[i2], normalIds[i3] })
	end

	-- 5. Materialize
	local part = AssetService:CreateMeshPartAsync(Content.fromObject(em))
	part.Name = data.Name
	part.Anchored = true
	part.DoubleSided = true -- robust against mixed winding from marching cubes
	part.Material = Enum.Material.SmoothPlastic
	part.Color = Color3.new(1, 1, 1) -- white so vertex colors read true

	return part
end

-- Places the part so the mesh's lowest point sits exactly at groundPosition.Y.
function SnapMeshForge.PlaceOnGround(part: MeshPart, groundPosition: Vector3): ()
	part.CFrame = CFrame.new(
		groundPosition.X,
		groundPosition.Y + part.Size.Y / 2,
		groundPosition.Z
	)
end

return SnapMeshForge
