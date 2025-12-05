#include "vulcany.slang"

struct VertexInput {
  [[vk::location(0)]]
  float2 pos;
  [[vk::location(1)]]
  float3 color;
};

struct VertexOutput {
  float4 pos : SV_POSITION;
  [[vk::location(0)]]
  float3 color;
};

struct TriangleColor {
  float3 color;
};

[shader("vertex")]
VertexOutput main(VertexInput vsin) {
  VertexOutput output;
  output.pos = float4(vsin.pos, 0, 1);
  let color_buffer = vulcanny::ReadOnlyBuffer<TriangleColor>::get_buffer(0);
  output.color = color_buffer[0].color;
  return output;
}
