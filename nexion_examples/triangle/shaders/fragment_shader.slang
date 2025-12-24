#include "nexion.slang"

struct VertexOutput {
  [[vk::location(0)]]
  float4 color;
  [[vk::location(1)]]
  float2 uv;
};

[shader("fragment")]
float4 main(VertexOutput input) {
  let img = GET_SAMPLED_IMG_2D(float4, 3);
  let sampler = nexion::get_sampler(0);
  let c = img.Sample(sampler, input.uv);
  return c;
}
