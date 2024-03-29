// File generated by "Obj2Header.exe" [Version 1.7]
// Data is converted to left-handed coordinate system and UV data is V flipped for Direct3D/Vulkan.
/************************************************/
/*  This section contains the model's size data */
/************************************************/
#ifndef _purple_vertexcount_
const unsigned purple_vertexcount = 64;
#define _purple_vertexcount_
#endif
#ifndef _purple_indexcount_
const unsigned purple_indexcount = 180;
#define _purple_indexcount_
#endif
#ifndef _purple_materialcount_
const unsigned purple_materialcount = 2;
#define _purple_materialcount_
#endif
#ifndef _purple_meshcount_
const unsigned purple_meshcount = 2;
#define _purple_meshcount_
#endif
/************************************************/
/*  This section contains the raw data to load  */
/************************************************/
#ifndef __OBJ_VERT__
typedef struct _OBJ_VERT_
{
	float pos[3]; // Left-handed +Z forward coordinate w not provided, assumed to be 1.
	float uvw[3]; // D3D/Vulkan style top left 0,0 coordinate.
	float nrm[3]; // Provided direct from obj file, may or may not be normalized.
}OBJ_VERT;
#define __OBJ_VERT__
#endif
#ifndef _purple_data_
// Raw Vertex Data follows: Positions, Texture Coordinates and Normals.
const OBJ_VERT purple_data[64] =
{
	{	{ -0.386424f, -4.413216f, -0.072448f },	{ 0.449721f, 0.087105f, 0.000000f },	{ -0.499700f, 0.032900f, 0.865600f }	},
	{	{ -0.360989f, -4.413216f, -0.057763f },	{ 0.448969f, 0.087105f, 0.000000f },	{ -0.499700f, 0.032900f, 0.865600f }	},
	{	{ 0.079247f, -0.708501f, 0.055772f },	{ 0.448445f, 0.053594f, 0.000000f },	{ -0.499700f, 0.032900f, 0.865600f }	},
	{	{ -0.015223f, -0.708501f, 0.001230f },	{ 0.450245f, 0.053594f, 0.000000f },	{ -0.499700f, 0.032900f, 0.865600f }	},
	{	{ -0.360989f, -4.413216f, -0.057763f },	{ 0.448969f, 0.087105f, 0.000000f },	{ 0.499700f, -0.032900f, -0.865600f }	},
	{	{ -0.386424f, -4.413216f, -0.072448f },	{ 0.449721f, 0.087105f, 0.000000f },	{ 0.499700f, -0.032900f, -0.865600f }	},
	{	{ -0.015223f, -0.708501f, 0.001230f },	{ 0.450245f, 0.053594f, 0.000000f },	{ 0.499700f, -0.032900f, -0.865600f }	},
	{	{ 0.079247f, -0.708501f, 0.055772f },	{ 0.448445f, 0.053594f, 0.000000f },	{ 0.499700f, -0.032900f, -0.865600f }	},
	{	{ 0.523942f, 0.349174f, -0.271315f },	{ 0.440146f, 0.080698f, 0.000000f },	{ 0.776300f, 0.443300f, -0.448200f }	},
	{	{ 0.523942f, 0.349174f, 0.281075f },	{ 0.443550f, 0.085257f, 0.000000f },	{ 0.776300f, 0.443300f, 0.448200f }	},
	{	{ 0.620923f, -0.141558f, 0.337066f },	{ 0.436243f, 0.087105f, 0.000000f },	{ 0.859100f, -0.126100f, 0.496000f }	},
	{	{ 0.620923f, -0.141558f, -0.327306f },	{ 0.432149f, 0.081622f, 0.000000f },	{ 0.859100f, -0.126100f, -0.496000f }	},
	{	{ -0.432824f, 0.349174f, -0.271315f },	{ 0.443550f, 0.067024f, 0.000000f },	{ -0.776300f, 0.443300f, -0.448200f }	},
	{	{ 0.045559f, 0.349174f, -0.547509f },	{ 0.440146f, 0.071582f, 0.000000f },	{ 0.000000f, 0.443300f, -0.896400f }	},
	{	{ 0.045559f, -0.141558f, -0.659492f },	{ 0.432149f, 0.070658f, 0.000000f },	{ 0.000000f, -0.126100f, -0.992000f }	},
	{	{ -0.529805f, -0.141558f, -0.327306f },	{ 0.436243f, 0.065176f, 0.000000f },	{ -0.859100f, -0.126100f, -0.496000f }	},
	{	{ 0.045559f, 0.349174f, 0.557270f },	{ 0.421149f, 0.068954f, 0.000000f },	{ 0.000000f, 0.443300f, 0.896400f }	},
	{	{ -0.432824f, 0.349174f, 0.281075f },	{ 0.425974f, 0.076689f, 0.000000f },	{ -0.776300f, 0.443300f, 0.448200f }	},
	{	{ -0.529805f, -0.141558f, 0.337066f },	{ 0.419726f, 0.081675f, 0.000000f },	{ -0.859100f, -0.126100f, 0.496000f }	},
	{	{ 0.045559f, -0.141558f, 0.669254f },	{ 0.413923f, 0.072372f, 0.000000f },	{ 0.000000f, -0.126100f, 0.992000f }	},
	{	{ 0.523942f, 0.349174f, 0.281075f },	{ 0.418186f, 0.065430f, 0.000000f },	{ 0.776300f, 0.443300f, 0.448200f }	},
	{	{ 0.620923f, -0.141558f, 0.337066f },	{ 0.410360f, 0.068134f, 0.000000f },	{ 0.859100f, -0.126100f, 0.496000f }	},
	{	{ -0.432824f, 0.349174f, -0.271315f },	{ 0.427836f, 0.080900f, 0.000000f },	{ -0.776300f, 0.443300f, -0.448200f }	},
	{	{ -0.529805f, -0.141558f, -0.327306f },	{ 0.421966f, 0.086740f, 0.000000f },	{ -0.859100f, -0.126100f, -0.496000f }	},
	{	{ -0.255295f, 0.561406f, -0.168819f },	{ 0.435731f, 0.050326f, 0.000000f },	{ -0.423300f, 0.872400f, -0.244400f }	},
	{	{ 0.045559f, 0.561406f, -0.342516f },	{ 0.430766f, 0.053193f, 0.000000f },	{ 0.000000f, 0.872400f, -0.488800f }	},
	{	{ 0.045559f, 0.349174f, -0.547509f },	{ 0.427836f, 0.051501f, 0.000000f },	{ 0.000000f, 0.443300f, -0.896400f }	},
	{	{ -0.432824f, 0.349174f, -0.271315f },	{ 0.435731f, 0.046943f, 0.000000f },	{ -0.776300f, 0.443300f, -0.448200f }	},
	{	{ 0.045559f, 0.561406f, 0.352277f },	{ 0.424802f, 0.068669f, 0.000000f },	{ 0.000000f, 0.872400f, 0.488800f }	},
	{	{ -0.255295f, 0.561406f, 0.178579f },	{ 0.427836f, 0.073534f, 0.000000f },	{ -0.423300f, 0.872400f, 0.244400f }	},
	{	{ 0.346414f, 0.561406f, 0.178579f },	{ 0.435731f, 0.061792f, 0.000000f },	{ 0.423300f, 0.872400f, 0.244400f }	},
	{	{ 0.045559f, 0.561406f, 0.352277f },	{ 0.440696f, 0.058926f, 0.000000f },	{ 0.000000f, 0.872400f, 0.488800f }	},
	{	{ 0.045559f, 0.349174f, 0.557270f },	{ 0.443626f, 0.060617f, 0.000000f },	{ 0.000000f, 0.443300f, 0.896400f }	},
	{	{ 0.523942f, 0.349174f, 0.281075f },	{ 0.435731f, 0.065176f, 0.000000f },	{ 0.776300f, 0.443300f, 0.448200f }	},
	{	{ 0.045559f, 0.561406f, -0.342516f },	{ 0.444569f, 0.073273f, 0.000000f },	{ 0.000000f, 0.872400f, -0.488800f }	},
	{	{ 0.346414f, 0.561406f, -0.168819f },	{ 0.444569f, 0.079007f, 0.000000f },	{ 0.423300f, 0.872400f, -0.244400f }	},
	{	{ -0.255295f, 0.561406f, 0.178579f },	{ 0.440696f, 0.053193f, 0.000000f },	{ -0.423300f, 0.872400f, 0.244400f }	},
	{	{ -0.432824f, 0.349174f, 0.281075f },	{ 0.443626f, 0.051501f, 0.000000f },	{ -0.776300f, 0.443300f, 0.448200f }	},
	{	{ 0.346414f, 0.561406f, -0.168819f },	{ 0.430766f, 0.058926f, 0.000000f },	{ 0.423300f, 0.872400f, -0.244400f }	},
	{	{ 0.523942f, 0.349174f, -0.271315f },	{ 0.427836f, 0.060617f, 0.000000f },	{ 0.776300f, 0.443300f, -0.448200f }	},
	{	{ 0.441375f, -0.516979f, 0.233404f },	{ 0.412170f, 0.049404f, 0.000000f },	{ 0.725900f, -0.545400f, 0.419100f }	},
	{	{ 0.441375f, -0.516979f, -0.223643f },	{ 0.409870f, 0.056587f, 0.000000f },	{ 0.725900f, -0.545400f, -0.419100f }	},
	{	{ 0.620923f, -0.141558f, -0.327306f },	{ 0.406526f, 0.057313f, 0.000000f },	{ 0.859100f, -0.126100f, -0.496000f }	},
	{	{ 0.620923f, -0.141558f, 0.337066f },	{ 0.409870f, 0.046871f, 0.000000f },	{ 0.859100f, -0.126100f, 0.496000f }	},
	{	{ 0.045559f, -0.516979f, -0.452167f },	{ 0.414941f, 0.062171f, 0.000000f },	{ 0.000000f, -0.545400f, -0.838200f }	},
	{	{ -0.350255f, -0.516979f, -0.223643f },	{ 0.422312f, 0.060572f, 0.000000f },	{ -0.725900f, -0.545400f, -0.419100f }	},
	{	{ -0.529805f, -0.141558f, -0.327306f },	{ 0.424612f, 0.063105f, 0.000000f },	{ -0.859100f, -0.126100f, -0.496000f }	},
	{	{ 0.045559f, -0.141558f, -0.659492f },	{ 0.413897f, 0.065430f, 0.000000f },	{ 0.000000f, -0.126100f, -0.992000f }	},
	{	{ -0.350255f, -0.516979f, 0.233404f },	{ 0.413375f, 0.083621f, 0.000000f },	{ -0.725900f, -0.545400f, 0.419100f }	},
	{	{ 0.045559f, -0.516979f, 0.461929f },	{ 0.409383f, 0.077221f, 0.000000f },	{ 0.000000f, -0.545400f, 0.838200f }	},
	{	{ 0.441375f, -0.516979f, 0.233404f },	{ 0.406932f, 0.074305f, 0.000000f },	{ 0.725900f, -0.545400f, 0.419100f }	},
	{	{ 0.441375f, -0.516979f, -0.223643f },	{ 0.427836f, 0.079912f, 0.000000f },	{ 0.725900f, -0.545400f, -0.419100f }	},
	{	{ 0.045559f, -0.516979f, -0.452167f },	{ 0.427836f, 0.072369f, 0.000000f },	{ 0.000000f, -0.545400f, -0.838200f }	},
	{	{ -0.350255f, -0.516979f, -0.223643f },	{ 0.414916f, 0.087105f, 0.000000f },	{ -0.725900f, -0.545400f, -0.419100f }	},
	{	{ -0.172309f, -0.771345f, 0.130666f },	{ 0.408723f, 0.084524f, 0.000000f },	{ -0.461400f, -0.846200f, 0.266400f }	},
	{	{ 0.045559f, -0.771345f, 0.256453f },	{ 0.406526f, 0.081001f, 0.000000f },	{ 0.000000f, -0.846200f, 0.532800f }	},
	{	{ 0.045559f, -0.771345f, 0.256453f },	{ 0.418507f, 0.051034f, 0.000000f },	{ 0.000000f, -0.846200f, 0.532800f }	},
	{	{ 0.263427f, -0.771345f, 0.130666f },	{ 0.414450f, 0.051914f, 0.000000f },	{ 0.461400f, -0.846200f, 0.266400f }	},
	{	{ 0.045559f, -0.516979f, 0.461929f },	{ 0.419541f, 0.047804f, 0.000000f },	{ 0.000000f, -0.545400f, 0.838200f }	},
	{	{ 0.263427f, -0.771345f, -0.120906f },	{ 0.413184f, 0.055868f, 0.000000f },	{ 0.461400f, -0.846200f, -0.266400f }	},
	{	{ 0.045559f, -0.771345f, -0.246692f },	{ 0.415975f, 0.058942f, 0.000000f },	{ 0.000000f, -0.846200f, -0.532800f }	},
	{	{ -0.172309f, -0.771345f, -0.120906f },	{ 0.420032f, 0.058061f, 0.000000f },	{ -0.461400f, -0.846200f, -0.266400f }	},
	{	{ -0.172309f, -0.771345f, 0.130666f },	{ 0.421298f, 0.054107f, 0.000000f },	{ -0.461400f, -0.846200f, 0.266400f }	},
	{	{ -0.350255f, -0.516979f, 0.233404f },	{ 0.424612f, 0.053388f, 0.000000f },	{ -0.725900f, -0.545400f, 0.419100f }	},
};
#define _purple_data_
#endif
#ifndef _purple_indicies_
// Index Data follows: Sequential by mesh, Every Three Indicies == One Triangle.
const unsigned int purple_indicies[180] =
{
	 0, 1, 2,
	 3, 0, 2,
	 4, 5, 6,
	 7, 4, 6,
	 8, 9, 10,
	 11, 8, 10,
	 12, 13, 14,
	 15, 12, 14,
	 16, 17, 18,
	 19, 16, 18,
	 20, 16, 19,
	 21, 20, 19,
	 13, 8, 11,
	 14, 13, 11,
	 17, 22, 23,
	 18, 17, 23,
	 24, 25, 26,
	 27, 24, 26,
	 28, 29, 17,
	 16, 28, 17,
	 30, 31, 32,
	 33, 30, 32,
	 34, 35, 8,
	 13, 34, 8,
	 36, 24, 27,
	 37, 36, 27,
	 38, 30, 33,
	 39, 38, 33,
	 30, 38, 25,
	 31, 30, 25,
	 36, 31, 25,
	 24, 36, 25,
	 40, 41, 42,
	 43, 40, 42,
	 44, 45, 46,
	 47, 44, 46,
	 48, 49, 19,
	 18, 48, 19,
	 49, 50, 21,
	 19, 49, 21,
	 51, 52, 14,
	 11, 51, 14,
	 53, 48, 18,
	 23, 53, 18,
	 54, 55, 49,
	 48, 54, 49,
	 56, 57, 40,
	 58, 56, 40,
	 59, 60, 44,
	 41, 59, 44,
	 61, 62, 63,
	 45, 61, 63,
	 57, 59, 41,
	 40, 57, 41,
	 60, 61, 45,
	 44, 60, 45,
	 61, 60, 59,
	 62, 61, 59,
	 56, 62, 59,
	 57, 56, 59,
};
#define _purple_indicies_
#endif
// Part of an OBJ_MATERIAL, the struct is 16 byte aligned so it is GPU register friendly.
#ifndef __OBJ_ATTRIBUTES__
typedef struct _OBJ_ATTRIBUTES_
{
	float       Kd[3]; // diffuse reflectivity
	float	    d; // dissolve (transparency) 
	float       Ks[3]; // specular reflectivity
	float       Ns; // specular exponent
	float       Ka[3]; // ambient reflectivity
	float       sharpness; // local reflection map sharpness
	float       Tf[3]; // transmission filter
	float       Ni; // optical density (index of refraction)
	float       Ke[3]; // emissive reflectivity
	unsigned    illum; // illumination model
}OBJ_ATTRIBUTES;
#define __OBJ_ATTRIBUTES__
#endif
// This structure also has been made GPU register friendly so it can be transfered directly if desired.
// Note: Total struct size will vary depenedening on CPU processor architecture (string pointers).
#ifndef __OBJ_MATERIAL__
typedef struct _OBJ_MATERIAL_
{
	// The following items are typically used in a pixel/fragment shader, they are packed for GPU registers.
	OBJ_ATTRIBUTES attrib; // Surface shading characteristics & illumination model
	// All items below this line are not needed on the GPU and are generally only used during load time.
	const char* name; // the name of this material
	// If the model's materials contain any specific texture data it will be located below.
	const char* map_Kd; // diffuse texture
	const char* map_Ks; // specular texture
	const char* map_Ka; // ambient texture
	const char* map_Ke; // emissive texture
	const char* map_Ns; // specular exponent texture
	const char* map_d; // transparency texture
	const char* disp; // roughness map (displacement)
	const char* decal; // decal texture (lerps texture & material colors)
	const char* bump; // normal/bumpmap texture
	void* padding[2]; // 16 byte alignment on 32bit or 64bit
}OBJ_MATERIAL;
#define __OBJ_MATERIAL__
#endif
#ifndef _purple_materials_
// Material Data follows: pulled from a .mtl file of the same name if present.
const OBJ_MATERIAL purple_materials[2] =
{
	{
		{{ 0.600000f, 0.600000f, 0.600000f },
		1.000000f,
		{ 0.200000f, 0.200000f, 0.200000f },
		500.000000f,
		{ 1.000000f, 1.000000f, 1.000000f },
		60.000000f,
		{ 1.000000f, 1.000000f, 1.000000f },
		1.500000f,
		{ 0.000000f, 0.000000f, 0.000000f },
		2},
		"default",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"",
	},
	{
		{{ 0.800000f, 0.800000f, 0.800000f },
		1.000000f,
		{ 0.500000f, 0.500000f, 0.500000f },
		225.000000f,
		{ 1.000000f, 1.000000f, 1.000000f },
		60.000000f,
		{ 1.000000f, 1.000000f, 1.000000f },
		1.450000f,
		{ 0.000000f, 0.000000f, 0.000000f },
		2},
		"Material",
		"a.png",
		"",
		"",
		"",
		"",
		"a.png",
		"",
		"",
		"",
	},
};
#define _purple_materials_
#endif
/************************************************/
/*  This section contains the model's structure */
/************************************************/
#ifndef _purple_batches_
// Use this conveinence array to batch render all geometry by like material.
// Each entry corresponds to the same entry in the materials array above.
// The two numbers provided are the IndexCount and the IndexOffset into the indicies array.
// If you need more fine grained control(ex: for transformations) use the OBJ_MESH data below.
const unsigned int purple_batches[2][2] =
{
	{ 0, 0 },
	{ 180, 0 },
};
#define _purple_batches_
#endif
#ifndef __OBJ_MESH__
typedef struct _OBJ_MESH_
{
	const char* name;
	unsigned    indexCount;
	unsigned    indexOffset;
	unsigned    materialIndex;
}OBJ_MESH;
#define __OBJ_MESH__
#endif
#ifndef _purple_meshes_
// Mesh Data follows: Meshes are .obj groups sorted & split by material.
// Meshes are provided in sequential order, sorted by material first and name second.
const OBJ_MESH purple_meshes[2] =
{
	{
		"default",
		0,
		0,
		0,
	},
	{
		"default",
		180,
		0,
		1,
	},
};
#define _purple_meshes_
#endif
