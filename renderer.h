// minimalistic code to draw a single triangle, this is not part of the API.
// required for compiling shaders on the fly, consider pre-compiling instead
#include <d3dcompiler.h>
#include "test_pyramid.h"
#pragma comment(lib, "d3dcompiler.lib")
// Simple Vertex Shader

//BE SURE TO USE THE PRAGMA PACK MATRIX ON SHADERS!!!
const char* vertexShaderSource = R"(
#pragma pack_matrix(row_major)
cbuffer SHDR_VARS
{
	matrix w, v, p;
};

// an ultra simple hlsl vertex shader
float4 main(float3 posL : POSITION, float3 uvw : TEXCOORD, float3 nrm : NORMAL) : SV_POSITION
{
	float4 vert = float4(posL, 1);
	vert = mul(vert, w);
	vert = mul(vert, v);
	vert = mul(vert, p);
	return vert;
}
)";
// Simple Pixel Shader
const char* pixelShaderSource = R"(
// an ultra simple hlsl pixel shader
float4 main() : SV_TARGET 
{	
	return float4(0.25f,0.0f,1.0f,0); 
}
)";
// Creation, Rendering & Cleanup
class Renderer
{
	//Variables
	float oldTime = 0;


	// proxy handles
	GW::SYSTEM::GWindow win;
	GW::GRAPHICS::GDirectX11Surface d3d;
	// what we need at a minimum to draw a triangle
	Microsoft::WRL::ComPtr<ID3D11Buffer>		vertexBuffer;
	Microsoft::WRL::ComPtr<ID3D11Buffer>		indexBuffer;
	Microsoft::WRL::ComPtr<ID3D11VertexShader>	vertexShader;
	Microsoft::WRL::ComPtr<ID3D11PixelShader>	pixelShader;
	Microsoft::WRL::ComPtr<ID3D11InputLayout>	vertexFormat;
	// Shader Variables
	Microsoft::WRL::ComPtr<ID3D11Buffer>		constantBuffer;
	struct SHDR_VARS
	{
		GW::MATH::GMATRIXF w, v, p;
	}svars;
	//math lib
	GW::MATH::GMatrix m;

public:
	Renderer(GW::SYSTEM::GWindow _win, GW::GRAPHICS::GDirectX11Surface _d3d)
	{
		win = _win;
		d3d = _d3d;
		ID3D11Device* creator;
		d3d.GetDevice((void**)&creator);
		D3D11_SUBRESOURCE_DATA bData = { test_pyramid_data, 0, 0 };
		CD3D11_BUFFER_DESC bDesc(sizeof(test_pyramid_data), D3D11_BIND_VERTEX_BUFFER);
		creator->CreateBuffer(&bDesc, &bData, vertexBuffer.GetAddressOf());
		//index buffer
		D3D11_SUBRESOURCE_DATA iData = { test_pyramid_indicies, 0, 0 };
		CD3D11_BUFFER_DESC iDesc(sizeof(test_pyramid_indicies), D3D11_BIND_INDEX_BUFFER);
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
		m.LookAtLHF(GW::MATH::GVECTORF{ 1, 1,-2}, GW::MATH::GVECTORF{ 0, 0.5f, 0 }, GW::MATH::GVECTORF{ 0,1,0 },svars.v);
		float ar;
		d3d.GetAspectRatio(ar);
		m.ProjectionDirectXLHF(G_DEGREE_TO_RADIAN(75),ar,0.1f,100.0f,svars.p);
		//Constant buffer crearte
		D3D11_SUBRESOURCE_DATA cData = { &svars, 0, 0 };
		CD3D11_BUFFER_DESC cDesc(sizeof(SHDR_VARS), D3D11_BIND_CONSTANT_BUFFER);
		creator->CreateBuffer(&cDesc, &cData, constantBuffer.GetAddressOf());

		// free temporary handle
		creator->Release();
	}
	void Render()
	{
		// grab the context & render target
		ID3D11DeviceContext* con;
		ID3D11RenderTargetView* view;
		ID3D11DepthStencilView* depth;
		d3d.GetImmediateContext((void**)&con);
		d3d.GetRenderTargetView((void**)&view);
		d3d.GetDepthStencilView((void**)&depth);
		// setup the pipeline
		ID3D11RenderTargetView *const views[] = { view };
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
		con->VSSetConstantBuffers(0,1, cbuffs);
		// now we can draw
		con->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		// update the matricies
		m.RotateYLocalF(svars.w, 0.01, svars.w);
		con->UpdateSubresource(constantBuffer.Get(),0,nullptr,&svars,sizeof(SHDR_VARS),0);
		oldTime = std::chrono::system_clock::now();
		con->DrawIndexed(test_pyramid_indexcount, 0, 0);
		// release temp handles
		depth->Release();
		view->Release();
		con->Release();
	}
	~Renderer()
	{
		// ComPtr will auto release so nothing to do here 
	}
};
