
static const float4 vertex_pos[3] = { float4(-1.0, -1.0, 0.0, 1.0),
                                      float4(3.0, -1.0, 0.0, 1.0),
                                      float4(-1.0, 3.0, 0.0, 1.0) };

struct OggaUint {
  StructuredBuffer<uint> sssss = sb[10];

  [vk_binding(0, 0)]
  private StructuredBuffer<uint> sb[];
}

struct Pogga {
  StructuredBuffer<int> sssss = sb[10];

  [vk_binding(0, 0)]
  private StructuredBuffer<int> sb[];
}

[shader("vertex")]
float4 main(uint vertex_id: SV_VertexID) : SV_Position {
  return vertex_pos[vertex_id];
}
