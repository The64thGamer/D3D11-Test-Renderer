// minimalistic code to draw a single triangle, this is not part of the API.
// required for compiling shaders on the fly, consider pre-compiling instead
#include <d3dcompiler.h>
#include "test_pyramid.h"
#include "dev4.h"
#include "DDSTextureLoader.h"
#pragma comment(lib, "d3dcompiler.lib")
// Simple Vertex Shader

//BE SURE TO USE THE PRAGMA PACK MATRIX ON SHADERS!!!
const char* vertexShaderSource = R"(
#pragma pack_matrix(row_major)

cbuffer SHDR_VARS // constant buggers are 16byte aligned
{
	matrix w, v, p;
	float4 lightDir; //adding 1 float for padding
	float4 lightColor;
	float4 ambientColor;
	float4 pointLightPos;
	float4 pointLightColor;
	float4 camPos;
	float4 spotLightPos;
	float4 spotLightDir;
	float4 spotLightColor;
	float4 innerConeRatio;
	float4 outerConeRatio;
};

struct VOUT
{
	float4 posH : SV_POSITION;
	float3 uvw	: TEXCOORD0;
	float3 nrm	: NORMAL;
	float3 plp : TEXCOORD1;
	float3 cam : TEXCOORD2;
	float3 slp : TEXCOORD3;
};

// an ultra simple hlsl vertex shader
VOUT main(float3 posL : POSITION, float3 uvw : TEXCOORD0, float3 nrm : NORMAL)
{
	VOUT output;
	float4 vert = float4(posL, 1);
	float4 worldPos = mul(vert,w);
	vert = mul(vert, w);
	vert = mul(vert, v);
	vert = mul(vert, p);
	output.posH = vert;
	output.uvw = uvw;
	output.nrm = mul(nrm, w);
	output.plp = normalize(pointLightPos.xyz-worldPos.xyz);
	output.slp = normalize(spotLightPos.xyz-worldPos.xyz);
	output.cam = normalize(camPos.xyz - worldPos.xyz);
	return output;
}
)";
// Simple Pixel Shader
const char* pixelShaderSource = R"(
// an ultra simple hlsl pixel shader
cbuffer SHDR_VARS // constant buggers are 16byte aligned
{
	matrix w, v, p;
	float4 lightDir; //adding 1 float for padding
	float4 lightColor;
	float4 ambientColor;
	float4 pointLightPos;
	float4 pointLightColor;
	float4 camPos;
	float4 spotLightPos;
	float4 spotLightDir;
	float4 spotLightColor;
	float4 innerConeRatio;
	float4 outerConeRatio;
};


Texture2D mytexture;
SamplerState mysampler;

struct VOUT
{
	float4 posH : SV_POSITION;
	float3 uvw : TEXCOORD0;
	float3 nrm : NORMAL;
	float3 plp : TEXCOORD1;
	float3 cam : TEXCOORD2;
	float3 slp : TEXCOORD3;
};

float4 main(VOUT input) : SV_TARGET 
{	
	float4 diffuse = mytexture.Sample(mysampler, input.uvw.xy);
	float4 light = float4(0,0,0,0);

	//Point Light
	float distance = length(input.plp.xyz);
	distance *= distance;
	light +=saturate(dot(input.nrm, input.plp.xyz)) * pointLightColor / distance;

	//Spot Light
	distance = length(input.slp.xyz);
	distance *= distance;
	float surfaceRatio = saturate(dot(-input.slp.xyz,spotLightDir.xyz));
	float lightRatio = saturate(dot(input.slp.xyz,input.nrm.xyz));
	float spotAtten = 1.0 - saturate((innerConeRatio.x - surfaceRatio)/(innerConeRatio.x - outerConeRatio.x));
	spotAtten *= spotAtten;
	light += lightRatio * spotLightColor * spotAtten * distance;

	//Ambient Light
	light += ambientColor;

	//Directional Light
	light += saturate(dot(-lightDir.xyz,input.nrm)) * lightColor;

	//Phong
	light += max(pow(saturate(dot(input.nrm, normalize(-input.plp.xyz + input.cam.xyz))), 30.0f),0);	
	light += max(pow(saturate(dot(input.nrm, normalize(-input.slp.xyz + input.cam.xyz))), 30.0f),0);	
	
	//Final
	light *= diffuse;
	
	//Final Saturation
	light = saturate(light);

	light.w = diffuse.w;

	//Return
	return light;
}
)";
// Creation, Rendering & Cleanup
class Renderer
{
	//Variables
	std::chrono::system_clock::time_point oldTime = std::chrono::system_clock::now();
	double timeDeltaTime = 0;
	double timeSinceStart = 0;
	float fovSpeed = 0;
	float fov = 75;
	float nearFarSpeed = 0;
	float nearPlane = 0.1f;
	float farPlane = 100.0f;
	float playerVelX = 0;
	float playerVelZ = 0;
	float playerVelY = 0;

	//Keyboard
	struct InputKeyboard
	{
		bool up;
		bool down;
		bool left;
		bool right;
		bool space;
		bool shift;
		bool plus;
		bool minus;
		bool bracketR;
		bool bracketL;
		bool one;
		float mouseX;
		float mouseY;
	};
	InputKeyboard keys;

	//Camera
	GW::MATH::GMATRIXF viewWorldM;
	GW::MATH::GMATRIXF viewLocalM;

	//MESHES
	struct MESH
	{
		unsigned index_count;
		GW::MATH::GMATRIXF w;
		Microsoft::WRL::ComPtr<ID3D11Buffer>		vertexBuffer;
		Microsoft::WRL::ComPtr<ID3D11Buffer>		indexBuffer;
		Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> texture;
	};
	MESH meshes[3];



	//SETUP
	GW::INPUT::GInput ginput;
	// proxy handles
	GW::SYSTEM::GWindow win;
	GW::GRAPHICS::GDirectX11Surface d3d;
	// what we need at a minimum to draw a triangle
	Microsoft::WRL::ComPtr<ID3D11VertexShader>	vertexShader;
	Microsoft::WRL::ComPtr<ID3D11PixelShader>	pixelShader;
	Microsoft::WRL::ComPtr<ID3D11InputLayout>	vertexFormat;
	ID3D11BlendState* g_pBlendStateNoBlend;
	// Shader Variables
	Microsoft::WRL::ComPtr<ID3D11Buffer>		constantBuffer;
	struct SHDR_VARS
	{
		GW::MATH::GMATRIXF w, v, p;
		GW::MATH::GVECTORF lightDir; //adding 1 float for padding
		GW::MATH::GVECTORF lightColor;
		GW::MATH::GVECTORF ambientColor;
		GW::MATH::GVECTORF pointLightPos;
		GW::MATH::GVECTORF pointLightColor;
		GW::MATH::GVECTORF camPos;
		GW::MATH::GVECTORF spotLightPos;
		GW::MATH::GVECTORF spotLightDir;
		GW::MATH::GVECTORF spotLightColor;
		GW::MATH::GVECTORF innerConeRatio;
		GW::MATH::GVECTORF outerConeRatio;
	}svars;
	//math lib
	GW::MATH::GMatrix m;

public:
	bool FillMesh(MESH& fill, const OBJ_VERT* verts, unsigned num_vert, const unsigned* indices, unsigned num_index, const wchar_t* tex_file)
	{
		ID3D11Device* creator;
		d3d.GetDevice((void**)&creator);

		D3D11_SUBRESOURCE_DATA bData = { verts , 0, 0 };
		CD3D11_BUFFER_DESC bDesc(sizeof(OBJ_VERT) * num_vert, D3D11_BIND_VERTEX_BUFFER);
		creator->CreateBuffer(&bDesc, &bData, fill.vertexBuffer.GetAddressOf());
		//index buffer
		D3D11_SUBRESOURCE_DATA iData = { indices , 0, 0 };
		CD3D11_BUFFER_DESC iDesc(sizeof(unsigned) * num_index, D3D11_BIND_INDEX_BUFFER);
		creator->CreateBuffer(&iDesc, &iData, fill.indexBuffer.GetAddressOf());
		fill.index_count = num_index;

		//Try to load texture from disk
		HRESULT hr = CreateDDSTextureFromFile(creator, tex_file, nullptr, fill.texture.GetAddressOf());

		fill.w = GW::MATH::GIdentityMatrixF;

		creator->Release();

		return true;
	}
	void DrawMesh(const MESH& draw)
	{
		ID3D11DeviceContext* con;
		ID3D11DepthStencilView* depth;
		d3d.GetImmediateContext((void**)&con);
		d3d.GetDepthStencilView((void**)&depth);
		const UINT strides[] = { sizeof(OBJ_VERT) };
		const UINT offsets[] = { 0 };

		ID3D11Buffer* const buffs[] = { draw.vertexBuffer.Get() };
		con->IASetVertexBuffers(0, ARRAYSIZE(buffs), buffs, strides, offsets);
		con->IASetIndexBuffer(draw.indexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);

		//set out texture
		ID3D11ShaderResourceView* const srvs[] = { draw.texture.Get() };
		con->PSSetShaderResources(0, 1, srvs);

		svars.w = draw.w; //make sure our world matrix is used at ttime of drawing
		con->UpdateSubresource(constantBuffer.Get(), 0, nullptr, &svars, sizeof(SHDR_VARS), 0);

		con->DrawIndexed(draw.index_count, 0, 0);

		con->Release();
	}

	Renderer(GW::SYSTEM::GWindow _win, GW::GRAPHICS::GDirectX11Surface _d3d)
	{
		win = _win;
		d3d = _d3d;
		//Input
		GW::GReturn g = ginput.Create(win);
		ID3D11Device* creator;
		d3d.GetDevice((void**)&creator);
		// Create Vertex Shader
		UINT compilerFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#if _DEBUG
		compilerFlags |= D3DCOMPILE_DEBUG;
#endif
		Microsoft::WRL::ComPtr<ID3DBlob> vsBlob, errors;
		if (SUCCEEDED(D3DCompile(vertexShaderSource, strlen(vertexShaderSource),
			nullptr, nullptr, nullptr, "main", "vs_4_0", compilerFlags, 0,
			vsBlob.GetAddressOf(), errors.GetAddressOf())))
		{
			creator->CreateVertexShader(vsBlob->GetBufferPointer(),
				vsBlob->GetBufferSize(), nullptr, vertexShader.GetAddressOf());
		}
		else
			std::cout << (char*)errors->GetBufferPointer() << std::endl;
		// Create Pixel Shader
		Microsoft::WRL::ComPtr<ID3DBlob> psBlob; errors.Reset();
		if (SUCCEEDED(D3DCompile(pixelShaderSource, strlen(pixelShaderSource),
			nullptr, nullptr, nullptr, "main", "ps_4_0", compilerFlags, 0,
			psBlob.GetAddressOf(), errors.GetAddressOf())))
		{
			creator->CreatePixelShader(psBlob->GetBufferPointer(),
				psBlob->GetBufferSize(), nullptr, pixelShader.GetAddressOf());
		}
		else
			std::cout << (char*)errors->GetBufferPointer() << std::endl;
		// Create Input Layout
		D3D11_INPUT_ELEMENT_DESC format[] = {
			{
				"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,
				D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0
			},{
				"TEXCOORD", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,
				D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0
			},{
				"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,
				D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0
			}
		};
		creator->CreateInputLayout(format, ARRAYSIZE(format),
			vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(),
			vertexFormat.GetAddressOf());
		//setup matricies
		m.Create();
		svars.w = GW::MATH::GIdentityMatrixF;
		viewWorldM = GW::MATH::GIdentityMatrixF;
		viewWorldM.data[14] = 1;
		float ar;
		d3d.GetAspectRatio(ar);
		m.ProjectionDirectXLHF(G_DEGREE_TO_RADIAN(fov), ar, 0.1f, 100.0f, svars.p);
		m.LookAtLHF(GW::MATH::GVECTORF{ 0, 1, 0 }, GW::MATH::GVECTORF{ 0, 1, 0 + 1 }, GW::MATH::GVECTORF{ 0,1,0 }, viewLocalM);
		// init light data
		svars.lightColor = GW::MATH::GVECTORF{ .6,.6,.7,1 };
		svars.lightDir = GW::MATH::GVECTORF{ -1,-1,1,0 };
		svars.pointLightColor = GW::MATH::GVECTORF{ 255.0 / 255.0 / 2.0,	 179.0 / 255.0 / 2.0,	 15.0 / 255.0 / 2.0	,0 };
		svars.pointLightPos = GW::MATH::GVECTORF{ 1,-3,1,1 };
		svars.ambientColor = GW::MATH::GVECTORF{ 39.0 / 255.0,	18.0 / 255.0,		53.0 / 255.0,0 };
		svars.spotLightColor = GW::MATH::GVECTORF{ 75.0 / 255.0,	 255 / 255.0,	 105.0 / 255.0,		0 };
		svars.spotLightPos = GW::MATH::GVECTORF{ 4,-5,4,1 };
		svars.spotLightDir = GW::MATH::GVECTORF{ 1,-1,1,0 };
		svars.innerConeRatio.x = .2;
		svars.outerConeRatio.x = .2;
		GW::MATH::GVector::NormalizeF(svars.lightDir, svars.lightDir);

		//Blend State
		D3D11_BLEND_DESC blendDesc;

		//Set Settings
		blendDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
		blendDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
		blendDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
		blendDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
		blendDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
		blendDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
		blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

		//Bool Settings
		blendDesc.AlphaToCoverageEnable = false;
		blendDesc.IndependentBlendEnable = false;
		blendDesc.RenderTarget[0].BlendEnable = true;

		//Create Blend State
		creator->CreateBlendState(&blendDesc, &g_pBlendStateNoBlend);


		//Constant buffer crearte
		D3D11_SUBRESOURCE_DATA cData = { &svars, 0, 0 };
		CD3D11_BUFFER_DESC cDesc(sizeof(SHDR_VARS), D3D11_BIND_CONSTANT_BUFFER);
		creator->CreateBuffer(&cDesc, &cData, constantBuffer.GetAddressOf());

		//Set Up Models
		ModelSetUp();

		// free temporary handle
		creator->Release();
	}
	void Render()
	{
		//Input
		SetKeyboardInput();
		//Time
		SetDeltaTime();
		//Physics
		Physics();
		//Update
		Update();
		// grab the context & render target
		ID3D11DeviceContext* con;
		ID3D11RenderTargetView* view;
		ID3D11DepthStencilView* depth;
		d3d.GetImmediateContext((void**)&con);
		d3d.GetRenderTargetView((void**)&view);
		d3d.GetDepthStencilView((void**)&depth);
		// setup the pipeline
		ID3D11RenderTargetView* const views[] = { view };
		con->OMSetRenderTargets(ARRAYSIZE(views), views, depth);
		con->VSSetShader(vertexShader.Get(), nullptr, 0);
		con->PSSetShader(pixelShader.Get(), nullptr, 0);
		con->IASetInputLayout(vertexFormat.Get());
		ID3D11Buffer* const cbuffs[] = { constantBuffer.Get() };
		con->VSSetConstantBuffers(0, 1, cbuffs);
		con->PSSetConstantBuffers(0, 1, cbuffs);
		// now we can draw
		con->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		//Update Resource
		con->UpdateSubresource(constantBuffer.Get(), 0, nullptr, &svars, sizeof(SHDR_VARS), 0);

		//Update Blend State
		float bf[] = { 1.0f, 1.0f, 1.0f, 1.0f };
		UINT sampleMask = 0xffffffff;
		con->OMSetBlendState(g_pBlendStateNoBlend, bf, sampleMask);

		//Draw Mesh
		for (size_t i = 0; i < sizeof(meshes) / sizeof(meshes[0]); i++)
		{
			DrawMesh(meshes[i]);
		}
		// release temp handles
		con->OMSetBlendState(nullptr, 0,0);
		depth->Release();
		view->Release();
		con->Release();
	}
	~Renderer()
	{
		// ComPtr will auto release so nothing to do here 
	}

	void ModelSetUp()
	{
		FillMesh(meshes[0], test_pyramid_data, test_pyramid_vertexcount, test_pyramid_indicies, test_pyramid_indexcount, L"../Rock.dds");
		FillMesh(meshes[1], dev4_data, dev4_vertexcount, dev4_indicies, dev4_indexcount, L"../dev4.dds");
		FillMesh(meshes[2], test_pyramid_data, test_pyramid_vertexcount, test_pyramid_indicies, test_pyramid_indexcount, L"../Rock.dds");
	}

	void SetDeltaTime()
	{
		std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
		timeDeltaTime = std::chrono::duration_cast<std::chrono::microseconds>(now - oldTime).count();
		timeDeltaTime /= 600000.0;
		oldTime = now;
		timeSinceStart += timeDeltaTime;
	}

	void SetKeyboardInput()
	{
		float w = 0;
		float a = 0;
		float s = 0;
		float d = 0;
		float up = 0;
		float left = 0;
		float down = 0;
		float right = 0;
		float shift = 0;
		float space = 0;
		float plus = 0;
		float minus = 0;
		float bracketR = 0;
		float bracketL = 0;
		float one = 0;
		float mouseX = 0;
		float mouseY = 0;
		ginput.GetState(60, w);
		ginput.GetState(38, a);
		ginput.GetState(56, s);
		ginput.GetState(41, d);
		ginput.GetState(29, up);
		ginput.GetState(31, left);
		ginput.GetState(34, down);
		ginput.GetState(32, right);
		ginput.GetState(14, shift);
		ginput.GetState(23, space);
		ginput.GetState(3, plus);
		ginput.GetState(2, minus);
		ginput.GetState(6, bracketR);
		ginput.GetState(7, bracketL);
		ginput.GetState(65, one);
		GW::GReturn result = ginput.GetMouseDelta(mouseX, mouseY);
		if (result == GW::GReturn::REDUNDANT)
		{
			mouseX = 0;
			mouseY = 0;
		}
		keys.up = (w + up);
		keys.left = (a + left);
		keys.down = (s + down);
		keys.right = (d + right);
		keys.space = (space);
		keys.shift = (shift);
		keys.plus = (plus);
		keys.minus = (minus);
		keys.bracketR = (bracketR);
		keys.bracketL = (bracketL);
		keys.one = (one);
		keys.mouseX = mouseX;
		keys.mouseY = mouseY;
	}

	void Physics()
	{
		//FOV
		if (keys.minus)
		{
			fovSpeed += timeDeltaTime;
			fov += fovSpeed / 20.0f;
		}
		else if (keys.plus)
		{
			fovSpeed += timeDeltaTime;
			fov -= fovSpeed / 20.0f;
		}
		else
		{
			fovSpeed = 0;
		}
		if (fov > 120)
		{
			fovSpeed = 0;
			fov = 120;
		}
		else if (fov < 10)
		{
			fovSpeed = 0;
			fov = 10;
		}
		//NearFar
		if (keys.bracketR)
		{
			nearFarSpeed += timeDeltaTime;
			if (keys.shift)
			{
				farPlane += nearFarSpeed / 20.0f;
			}
			else
			{
				nearPlane += nearFarSpeed / 20.0f;
			}
		}
		else if (keys.bracketL)
		{
			nearFarSpeed += timeDeltaTime;
			if (keys.shift)
			{
				farPlane -= nearFarSpeed / 20.0f;
			}
			else
			{
				nearPlane -= nearFarSpeed / 20.0f;
			}
		}
		else
		{
			nearFarSpeed = 0;
		}
		if (farPlane > 120)
		{
			nearFarSpeed = 0;
			farPlane = 120;
		}
		else if (farPlane < 0.01)
		{
			nearFarSpeed = 0;
			farPlane = 0.01;
		}
		if (nearPlane > 120)
		{
			nearFarSpeed = 0;
			nearPlane = 120;
		}
		else if (nearPlane < 0.01)
		{
			nearFarSpeed = 0;
			nearPlane = 0.01;
		}
		//Body
		float maxed = .001f;
		if (keys.shift)
		{
			maxed *= 2;
		}
		if (keys.up)
		{
			playerVelZ += timeDeltaTime / 50.0f;
			playerVelZ = max(playerVelZ, -maxed);
			playerVelZ = min(playerVelZ, maxed);
		}
		else if (keys.down)
		{
			playerVelZ -= timeDeltaTime / 50.0f;
			playerVelZ = max(playerVelZ, -maxed);
			playerVelZ = min(playerVelZ, maxed);
		}
		else if (playerVelZ > 0)
		{
			playerVelZ -= timeDeltaTime / 50.0f;
			playerVelZ = max(playerVelZ, 0);
		}
		else if (playerVelZ < 0)
		{
			playerVelZ += timeDeltaTime / 50.0f;
			playerVelZ = min(playerVelZ, 0);
		}

		if (keys.right)
		{
			playerVelX += timeDeltaTime / 50.0f;
			playerVelX = max(playerVelX, -maxed);
			playerVelX = min(playerVelX, maxed);
		}
		else if (keys.left)
		{
			playerVelX -= timeDeltaTime / 50.0f;
			playerVelX = max(playerVelX, -maxed);
			playerVelX = min(playerVelX, maxed);
		}
		else if (playerVelX > 0)
		{
			playerVelX -= timeDeltaTime / 50.0f;
			playerVelX = max(playerVelX, 0);
		}
		else if (playerVelX < 0)
		{
			playerVelX += timeDeltaTime / 50.0f;
			playerVelX = min(playerVelX, 0);
		}
		//Apply Camera Data
		float ar;
		d3d.GetAspectRatio(ar);
		m.ProjectionDirectXLHF(G_DEGREE_TO_RADIAN(fov), ar, nearPlane, farPlane, svars.p);

		//Get Radians
		float radianX = G_DEGREE_TO_RADIAN(keys.mouseX / 10.0f);
		float radianY = G_DEGREE_TO_RADIAN(keys.mouseY / 10.0f);

		//Get Rotation Matrices
		GW::MATH::GMATRIXF rotatedX;
		GW::MATH::GMATRIXF rotatedY;
		m.RotationYawPitchRollF(radianX, 0, 0, rotatedX);
		m.RotationYawPitchRollF(0, radianY, 0, rotatedY);

		//Edit View Local
		m.MultiplyMatrixF(viewLocalM, rotatedY, viewLocalM);

		//Edit Character Local
		m.MultiplyMatrixF(viewWorldM, rotatedX, viewWorldM);
		m.TranslateLocalF(viewWorldM, GW::MATH::GVECTORF{ -playerVelX,playerVelY,-playerVelZ }, viewWorldM);

		//Multiply views
		m.MultiplyMatrixF(viewWorldM, viewLocalM, svars.v);
	}

	void Update()
	{

		svars.camPos = GW::MATH::GVECTORF{ viewLocalM.data[12],viewLocalM.data[13],viewLocalM.data[14] };
		svars.lightDir.x = sin(timeSinceStart / 2.0);
		svars.lightDir.y = (sin(timeSinceStart / 2.0) / 2) - 1;
		svars.pointLightPos = GW::MATH::GVECTORF{ (float)cos(timeSinceStart),((float)sin(timeSinceStart) + 2),(float)cos(timeSinceStart),1 };
		svars.spotLightPos = GW::MATH::GVECTORF{ (float)cos(timeSinceStart) + 10,3,(float)cos(timeSinceStart),1 };
		svars.innerConeRatio.x = 0.4f - min(sin(timeSinceStart), 0.0f);
		svars.outerConeRatio.x = 0.4f;
		svars.spotLightDir.x = sin(timeSinceStart / 2.0);
		meshes[0].w.row4 = svars.pointLightPos;
		meshes[2].w.data[13] = 2;
	}
};