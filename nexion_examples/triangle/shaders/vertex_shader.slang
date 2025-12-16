#include "nexion.slang"

struct VertexInput {
  [[vk::location(0)]]
  float2 pos;
  [[vk::location(1)]]
  float2 uv;
};

struct VertexOutput {
  float4 pos : SV_POSITION;
  [[vk::location(0)]]
  float4 color;
  [[vk::location(1)]]
  float2 uv;
};

struct TriangleColor {
  float3 color;
};

[shader("vertex")]
VertexOutput main(VertexInput vsin) {
  VertexOutput output;
  output.pos = float4(vsin.pos, 0, 1);
  output.uv = vsin.uv;
  let color_buffer = nexion::ReadOnlyBuffer<TriangleColor>::get_buffer(0);
  output.color = float4(color_buffer[0].color, 0.0);
  return output;
}
