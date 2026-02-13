#include "nexion.slang"

struct VertexOutput {
  [[vk::location(0)]]
  float2 uv;
};

[[vk_push_constant]]
float3 *color;

[shader("fragment")]
float4 main(VertexOutput input) {
  let img = GET_SAMPLED_IMG_2D(float4, 3);
  let sampler = nexion::get_sampler(0);
  let c1 = img.Sample(sampler, input.uv);
  let c2 = float4(color[0], 0.0);
  return lerp(c1, c2, 0.5);
}
