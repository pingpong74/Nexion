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
  float2 uv;
};

[shader("vertex")]
VertexOutput main(VertexInput vsin) {
  VertexOutput output;
  output.pos = float4(vsin.pos, 0, 1);
  output.uv = vsin.uv;
  return output;
}
