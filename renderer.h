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
};

struct VOUT
{
	float4 posH : SV_POSITION;
	float3 uvw	: TEXCOORD;
	float3 nrm	: NORMAL;
};

// an ultra simple hlsl vertex shader
VOUT main(float3 posL : POSITION, float3 uvw : TEXCOORD, float3 nrm : NORMAL)
{
	VOUT output;
	float4 vert = float4(posL, 1);
	vert = mul(vert, w);
	vert = mul(vert, v);
	vert = mul(vert, p);
	output.posH = vert;
	output.uvw = uvw;
	output.nrm = mul(nrm, w);
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
};


Texture2D mytexture;
SamplerState mysampler;

struct VOUT
{
	float4 posH : SV_POSITION;
	float3 uvw : TEXCOORD;
	float3 nrm : NORMAL;
};

float4 main(VOUT input) : SV_TARGET 
{	
	float4 diffuse = mytexture.Sample(mysampler, input.uvw.xy);
	return saturate(dot(-lightDir.xyz,input.nrm)) * diffuse * lightColor;
}
)";
// Creation, Rendering & Cleanup
class Renderer
{
	//Variables
	std::chrono::system_clock::time_point oldTime = std::chrono::system_clock::now();
	double timeDeltaTime = 0;
	float fovSpeed = 0;
	float fov = 75;
	float nearFarSpeed = 0;
	float nearPlane = 0.1f;
	float farPlane = 100.0f;
	float playerVelX = 0;
	float playerVelZ = 0;
	float playerVelY = 0;

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

	GW::MATH::GMATRIXF viewWorldM;
	GW::MATH::GMATRIXF viewLocalM;

	//input
	GW::INPUT::GInput ginput;
	// proxy handles
	GW::SYSTEM::GWindow win;
	GW::GRAPHICS::GDirectX11Surface d3d;
	// what we need at a minimum to draw a triangle
	Microsoft::WRL::ComPtr<ID3D11Buffer>		vertexBuffer;
	Microsoft::WRL::ComPtr<ID3D11Buffer>		indexBuffer;
	Microsoft::WRL::ComPtr<ID3D11VertexShader>	vertexShader;
	Microsoft::WRL::ComPtr<ID3D11PixelShader>	pixelShader;
	Microsoft::WRL::ComPtr<ID3D11InputLayout>	vertexFormat;
	//Texture Variables
	Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> texture;
	// Shader Variables
	Microsoft::WRL::ComPtr<ID3D11Buffer>		constantBuffer;
	struct SHDR_VARS
	{
		GW::MATH::GMATRIXF w, v, p;
		GW::MATH::GVECTORF lightDir; //adding 1 float for padding
		GW::MATH::GVECTORF lightColor;
	}svars;
	//math lib
	GW::MATH::GMatrix m;

public:
	Renderer(GW::SYSTEM::GWindow _win, GW::GRAPHICS::GDirectX11Surface _d3d)
	{
		win = _win;
		d3d = _d3d;

		//Input
		GW::GReturn g = ginput.Create(win);

		ID3D11Device* creator;
		d3d.GetDevice((void**)&creator);
		D3D11_SUBRESOURCE_DATA bData = { dev4_data, 0, 0 };
		CD3D11_BUFFER_DESC bDesc(sizeof(dev4_data), D3D11_BIND_VERTEX_BUFFER);
		creator->CreateBuffer(&bDesc, &bData, vertexBuffer.GetAddressOf());
		//index buffer
		D3D11_SUBRESOURCE_DATA iData = { dev4_indicies, 0, 0 };
		CD3D11_BUFFER_DESC iDesc(sizeof(dev4_indicies), D3D11_BIND_INDEX_BUFFER);
		creator->CreateBuffer(&iDesc, &iData, indexBuffer.GetAddressOf());

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
		svars.lightColor = GW::MATH::GVECTORF{ 1,1,1,1 };
		svars.lightDir = GW::MATH::GVECTORF{ -1,-1,1,0 };
		GW::MATH::GVector::NormalizeF(svars.lightDir, svars.lightDir);

		//Constant buffer crearte
		D3D11_SUBRESOURCE_DATA cData = { &svars, 0, 0 };
		CD3D11_BUFFER_DESC cDesc(sizeof(SHDR_VARS), D3D11_BIND_CONSTANT_BUFFER);
		creator->CreateBuffer(&cDesc, &cData, constantBuffer.GetAddressOf());
		//Try to load texture from disk
		HRESULT hr = CreateDDSTextureFromFile(creator, L"../Rock.dds", nullptr, texture.GetAddressOf());
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
		const UINT strides[] = { sizeof(OBJ_VERT) };
		const UINT offsets[] = { 0 };
		ID3D11Buffer* const buffs[] = { vertexBuffer.Get() };
		con->IASetVertexBuffers(0, ARRAYSIZE(buffs), buffs, strides, offsets);
		con->IASetIndexBuffer(indexBuffer.Get(), DXGI_FORMAT_R32_UINT, 0);
		con->VSSetShader(vertexShader.Get(), nullptr, 0);
		con->PSSetShader(pixelShader.Get(), nullptr, 0);
		con->IASetInputLayout(vertexFormat.Get());
		ID3D11Buffer* const cbuffs[] = { constantBuffer.Get() };
		con->VSSetConstantBuffers(0, 1, cbuffs);
		con->PSSetConstantBuffers(0, 1, cbuffs);
		// now we can draw
		con->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		//set out texture
		ID3D11ShaderResourceView* const srvs[] = { texture.Get() };
		con->PSSetShaderResources(0, 1, srvs);
		// update the matricies
		//m.RotateYLocalF(svars.w, 1 * timeDeltaTime, svars.w);


		//Update Resource
		con->UpdateSubresource(constantBuffer.Get(), 0, nullptr, &svars, sizeof(SHDR_VARS), 0);
		con->DrawIndexed(dev4_indexcount, 0, 0);
		// release temp handles
		depth->Release();
		view->Release();
		con->Release();
	}
	~Renderer()
	{
		// ComPtr will auto release so nothing to do here 
	}

	void SetDeltaTime()
	{
		std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
		timeDeltaTime = std::chrono::duration_cast<std::chrono::microseconds>(now - oldTime).count();
		timeDeltaTime /= 600000.0;
		oldTime = now;
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
		//Apply Camera
		float ar;
		d3d.GetAspectRatio(ar);
		m.ProjectionDirectXLHF(G_DEGREE_TO_RADIAN(fov), ar, nearPlane, farPlane, svars.p);
		GW::MATH::GMATRIXF rotatedX;
		GW::MATH::GMATRIXF rotatedY;
		float radianX = G_DEGREE_TO_RADIAN(keys.mouseX / 10.0f);
		float radianY = G_DEGREE_TO_RADIAN(keys.mouseY / 10.0f);
		m.RotationYawPitchRollF(radianX, 0, 0, rotatedX);
		m.RotationYawPitchRollF(0, radianY, 0, rotatedY);
		m.MultiplyMatrixF(rotatedY, viewLocalM, viewLocalM);
		m.TranslateGlobalF(viewWorldM, GW::MATH::GVECTORF{ -playerVelX,playerVelY,-playerVelZ}, viewWorldM);
		m.RotateYLocalF(viewWorldM, radianX, viewWorldM);
		m.MultiplyMatrixF(viewWorldM, viewLocalM, svars.v);
	}
};