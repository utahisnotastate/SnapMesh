--!strict
-- SnapMeshForge — builds a SnapMesh / Hi3D export (a .luau data module or a JSON
-- payload) into a live MeshPart using AssetService:CreateEditableMesh().
-- No mesh uploads, no moderation queue: geometry is constructed from pure code.
--
-- Two appearance paths, picked automatically from what the data contains:
--   * UVs + a texture asset id  -> proper UV-mapped texture (high quality; this is
--     what makes a generated mesh look like a real game asset)
--   * Colors (per-vertex)       -> vertex-colored fallback, no texture needed
--
-- Usage:
--   local SnapMeshForge = require(ReplicatedStorage.SnapMeshForge)
--   local data = require(ReplicatedStorage.Meshes.AI_Matrix)
--   local part = SnapMeshForge.Build(data, { TextureId = "rbxassetid://123456789" })
--   SnapMeshForge.PlaceOnGround(part, Vector3.new(0, 0, 0))
--   part.Parent = workspace
--
-- Notes:
--   * Exports are normalized: feet at local Y=0, centered on X/Z. Use PlaceOnGround
--     for floor-accurate placement (the part's own pivot is its bbox center).
--   * Clones of the returned part share mesh content — scatter hundreds cheaply.

local AssetService = game:GetService("AssetService")

export type SnapMeshData = {
	Name: string,
	Verts: { number },           -- flat x,y,z triplets
	Tris: { number },            -- flat 1-based vertex index triplets
	UVs: { number }?,            -- flat u,v pairs, one per vertex (preferred)
	Colors: { number }?,         -- flat r,g,b triplets 0..1, one per vertex (fallback)
	CornerNormals: { number }?,  -- 9 floats per triangle (3 corners x xyz).
	                             -- Crease-angle normals baked by the exporter: hard
	                             -- panel edges stay sharp instead of melting. Strongly
	                             -- preferred over the smooth fallback below.
}

export type BuildOptions = {
	TextureId: string?,        -- rbxassetid:// for the diffuse map
	Material: Enum.Material?,
	DoubleSided: boolean?,
	Anchored: boolean?,
}

local SnapMeshForge = {}

local function isFinite(n: number): boolean
	return n == n and n ~= math.huge and n ~= -math.huge
end

function SnapMeshForge.Build(data: SnapMeshData, options: BuildOptions?): MeshPart
	local opts: BuildOptions = options or {}
	local vertexCount = #data.Verts // 3
	assert(vertexCount > 0, "SnapMeshForge: empty Verts")
	assert(#data.Tris % 3 == 0, "SnapMeshForge: Tris length must be a multiple of 3")

	local hasUVs = data.UVs ~= nil and #data.UVs == vertexCount * 2
	local hasColors = data.Colors ~= nil and #data.Colors == vertexCount * 3

	local em = AssetService:CreateEditableMesh()

	-- 1. Vertices (guarding against non-finite coords, which hard-crash the engine)
	local vertexIds = table.create(vertexCount)
	local positions = table.create(vertexCount)
	for i = 1, vertexCount do
		local b = (i - 1) * 3
		local x, y, z = data.Verts[b + 1], data.Verts[b + 2], data.Verts[b + 3]
		assert(isFinite(x) and isFinite(y) and isFinite(z),
			`SnapMeshForge: non-finite vertex at index {i}`)
		local pos = Vector3.new(x, y, z)
		positions[i] = pos
		vertexIds[i] = em:AddVertex(pos)
	end

	-- 2. Appearance attributes
	local uvIds: { number }? = nil
	if hasUVs then
		uvIds = table.create(vertexCount)
		local uvs = (data.UVs :: { number })
		for i = 1, vertexCount do
			local b = (i - 1) * 2
			-- OBJ/glTF V axis is bottom-up; Roblox samples top-down.
			(uvIds :: { number })[i] = em:AddUV(Vector2.new(uvs[b + 1], 1 - uvs[b + 2]))
		end
	end

	local colorIds: { number }? = nil
	if hasColors and not hasUVs then
		colorIds = table.create(vertexCount)
		local cols = (data.Colors :: { number })
		for i = 1, vertexCount do
			local b = (i - 1) * 3
			(colorIds :: { number })[i] =
				em:AddColor(Color3.new(cols[b + 1], cols[b + 2], cols[b + 3]), 1)
		end
	end

	local tris = data.Tris
	local triCount = #tris // 3
	local cn = data.CornerNormals
	local hasCornerNormals = cn ~= nil and #cn == triCount * 9

	-- 3. Normals. Preferred: per-corner crease normals baked by the exporter (hard
	--    edges stay hard). Fallback: smooth area-weighted per-vertex normals, which
	--    suit organic shapes but visibly soften hard-surface bevels.
	local smoothNormalIds: { number }? = nil
	if not hasCornerNormals then
		local accum = table.create(vertexCount, Vector3.zero)
		for t = 1, #tris, 3 do
			local i1, i2, i3 = tris[t], tris[t + 1], tris[t + 2]
			local p1, p2, p3 = positions[i1], positions[i2], positions[i3]
			if p1 and p2 and p3 then
				local cross = (p2 - p1):Cross(p3 - p1) -- |cross| == 2*area (weighting)
				accum[i1] += cross
				accum[i2] += cross
				accum[i3] += cross
			end
		end
		local ids = table.create(vertexCount)
		for i = 1, vertexCount do
			local n = accum[i]
			ids[i] = em:AddNormal(n.Magnitude > 1e-8 and n.Unit or Vector3.yAxis)
		end
		smoothNormalIds = ids
	end

	-- 4. Faces
	for t = 1, #tris, 3 do
		local i1, i2, i3 = tris[t], tris[t + 1], tris[t + 2]
		if vertexIds[i1] and vertexIds[i2] and vertexIds[i3] then
			local faceId = em:AddTriangle(vertexIds[i1], vertexIds[i2], vertexIds[i3])

			if hasCornerNormals then
				local b = (t - 1) * 3 -- 9 floats per triangle
				local nrm = (cn :: { number })
				em:SetFaceNormals(faceId, {
					em:AddNormal(Vector3.new(nrm[b + 1], nrm[b + 2], nrm[b + 3])),
					em:AddNormal(Vector3.new(nrm[b + 4], nrm[b + 5], nrm[b + 6])),
					em:AddNormal(Vector3.new(nrm[b + 7], nrm[b + 8], nrm[b + 9])),
				})
			elseif smoothNormalIds then
				local ids = (smoothNormalIds :: { number })
				em:SetFaceNormals(faceId, { ids[i1], ids[i2], ids[i3] })
			end

			if uvIds then
				em:SetFaceUVs(faceId, { uvIds[i1], uvIds[i2], uvIds[i3] })
			elseif colorIds then
				em:SetFaceColors(faceId, { colorIds[i1], colorIds[i2], colorIds[i3] })
			end
		end
	end

	-- 5. Materialize
	local part = AssetService:CreateMeshPartAsync(Content.fromObject(em))
	part.Name = data.Name
	part.Anchored = if opts.Anchored ~= nil then opts.Anchored else true
	part.DoubleSided = if opts.DoubleSided ~= nil then opts.DoubleSided else false
	part.Material = opts.Material or Enum.Material.SmoothPlastic
	part.Color = Color3.new(1, 1, 1) -- white base so texture/vertex colors read true

	if opts.TextureId and hasUVs then
		part.TextureID = opts.TextureId
	end

	return part
end

-- Places the part so the mesh's lowest point sits exactly at groundPosition.Y.
--
-- Note on why this is just a plain CFrame.new: a MeshPart produced by
-- CreateMeshPartAsync keeps the EditableMesh's own local coordinates — the engine
-- does NOT recenter geometry on the part's pivot. Since SnapMesh exports are
-- normalized with feet at local Y=0 and centered on X/Z, the part's origin already
-- IS the ground contact point. (Offsetting by Size.Y/2, as you would for a normal
-- Part, floats the model exactly half its height off the ground.)
function SnapMeshForge.PlaceOnGround(part: MeshPart, groundPosition: Vector3): ()
	part.CFrame = CFrame.new(groundPosition)
end

return SnapMeshForge
