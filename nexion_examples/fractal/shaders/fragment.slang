
struct PushConstants {
  matrix<float32_t, 4, 4> inv_view_proj_mat;
  float3 pos;
  uint width;
  uint height;
  float32_t time;
};

[[vk_push_constant]]
uniform PushConstants pc;

float3 ray_direction(float2 fragCoord, float2 resolution) {
  float2 uv = (fragCoord / resolution) * 2.0 - 1.0;
  uv.y = -uv.y; // Vulkan Y is flipped
  float4 ray_clip = float4(uv, 1.0, 1.0);

  // Transform from clip space to world space
  float4 ray_world = mul(pc.inv_view_proj_mat, ray_clip);
  ray_world.xyz /= ray_world.w;

  float3 cam_pos = (mul(pc.inv_view_proj_mat, float4(0, 0, 0, 1))).xyz;
  return normalize(ray_world.xyz - cam_pos);
}

float mandelbulbDE(float3 p) {
  float Power = 7.0+ 1.5 * sin(pc.time * 0.75 + p.y * 0.75);
  float dr = 1.0;
  float r = length(p);

  float3 z = p;
  for (int i = 0; i < 8; i++) {

    r = length(z);
    if (r > 4.0)
      break;

    float theta = acos(z.z / r);
    float phi = atan2(z.y, z.x);
    dr = pow(r, Power - 1.0) * Power * dr + 1.0;

    float zr = pow(r, Power);
    theta *= Power;
    phi *= Power;

    z = zr * float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    z += p;
  }

  return 0.5 * log(r) * r / dr;
}

float trace(float3 ro, float3 rd) {
  float totalDist = 0.0;
  const float MAX_DIST = 500.0;
  const int MAX_STEPS = 256;
  const float SURF_DIST = 0.001;

  for (int i = 0; i < MAX_STEPS; i++) {
    float3 p = ro + rd * totalDist;
    float d = mandelbulbDE(p);

    if (d < SURF_DIST)
      return totalDist;
    totalDist += d;
    if (totalDist > MAX_DIST)
      break;
  }
  return -1.0;
}

float3 getNormal(float3 p) {
  const float eps = 0.001;
  float3 e = float3(eps, 0.0, 0.0);
  float dx = mandelbulbDE(p + e.xyy) - mandelbulbDE(p - e.xyy);
  float dy = mandelbulbDE(p + e.yxy) - mandelbulbDE(p - e.yxy);
  float dz = mandelbulbDE(p + e.yyx) - mandelbulbDE(p - e.yyx);
  return normalize(float3(dx, dy, dz));
}

float3 shade(float3 ro, float3 rd, float t) {
  float3 p = ro + rd * t;
  float3 n = getNormal(p);
  float3 lightDir = normalize(float3(0.6, 0.7, 0.5));

  float diff = max(dot(n, lightDir), 0.0);
  float3 col = float3(0.9, 0.6, 0.8) * diff + float3(0.1, 0.1, 0.1);
  return col;
}

[shader("fragment")]
float4 main(float4 fragCoord: SV_Position) : SV_Target {
  float2 frag = fragCoord.xy;
  float2 res = float2(pc.width, pc.height);

  float3 ro = pc.pos;
  float3 rd = ray_direction(frag, res);

  float t = trace(ro, rd);
  float3 color;

  if (t > 0.0)
    color = shade(ro, rd, t);
  else
    color = float3(0.00, 0.00, 0.00);

  return float4(color, 1.0);
}
