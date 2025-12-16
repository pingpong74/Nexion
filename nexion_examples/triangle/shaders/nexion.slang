#pragma once

// Buffers
namespace nexion {

// -----------------------------------------
// Read only Buffers
// -----------------------------------------
[[vk::binding(0, 0)]]
ByteAddressBuffer ro_buffers[];

/// This is read only buffer type
public struct ReadOnlyBuffer<T> {
  public ByteAddressBuffer raw_buffer;

  public static inline const ReadOnlyBuffer<T> get_buffer(uint index) {
    return { ro_buffers[index] };
  }
  public __subscript(uint index)->T {
    get { return raw_buffer.Load<T>(index * sizeof(T)); }
  }
}

// -----------------------------------------
// Read write buffers
// -----------------------------------------
[[vk::binding(0, 0)]]
RWByteAddressBuffer rw_buffers[];

[[vk::binding(0, 0)]]
coherent RWByteAddressBuffer rw_coherent_buffers[];

public struct ReadWriteBuffer<T> {
  public RWByteAddressBuffer raw_buffer;

  public static inline ReadWriteBuffer<T> get_buffer(uint index) {
    return { rw_buffers[index] };
  }

  public static inline ReadWriteBuffer<T> get_coherent_buffer(uint index) {
    return { rw_coherent_buffers[index] };
  }

  public const inline T load(uint id) {
    return raw_buffer.Load<T>(id * sizeof(T));
  }

  public inline void store(uint id, T new_value) {
    raw_buffer.Store<T>(id * sizeof(T), new_value);
  }

  public inline void atomic_add_f32(uint id, float val) {
    raw_buffer.InterlockedAddF32(id * 4, val);
  }
}

//----------------
// SAMPLER
// ---------------
[[vk::binding(3, 0)]]
SamplerState samplers[];

public SamplerState get_sampler(uint index) { return nexion::samplers[index]; }

#define DECLARE_SAMPLED_IMGS_1D(TYPE)                                          \
  [[vk::binding(1, 0)]]                                                        \
  Texture1D<TYPE> sampled_images_1d_##TYPE[];

#define DECLARE_SAMPLED_IMGS_2D(TYPE)                                          \
  [[vk::binding(1, 0)]]                                                        \
  Texture2D<TYPE> sampled_images_2d_##TYPE[];

#define DECLARE_SAMPLED_IMGS_3D(TYPE)                                          \
  [[vk::binding(1, 0)]]                                                        \
  Texture3D<TYPE> sampled_images_3d_##TYPE[];

#define DECLARE_SAMPLED_IMGS_CUBE(TYPE)                                        \
  [[vk::binding(1, 0)]]                                                        \
  TextureCube<TYPE> sampled_images_cube_##TYPE[];

#define DECLARE_RW_IMGS_1D(TYPE)                                               \
  [[vk::binding(2, 0)]]                                                        \
  RWTexture1D<TYPE> rw_images_1d_##TYPE[];

#define DECLARE_RW_IMGS_2D(TYPE)                                               \
  [[vk::binding(1, 0)]]                                                        \
  RWTexture2D<TYPE> rw_images_2d_##TYPE[];

#define DECLARE_RW_IMGS_3D(TYPE)                                               \
  [[vk::binding(1, 0)]]                                                        \
  RWTexture3D<TYPE> rw_images_3d_##TYPE[];

#define GET_SAMPLED_IMG_1D(TYPE, index) nexion::sampled_images_1d_##TYPE[index]
#define GET_SAMPLED_IMG_2D(TYPE, index) nexion::sampled_images_2d_##TYPE[index]
#define GET_SAMPLED_IMG_3D(TYPE, index) nexion::sampled_images_3d_##TYPE[index]
#define GET_SAMPLED_IMG_CUBE(TYPE, index)                                      \
  nexion::sampled_images_cube_##TYPE[index]

// -----------------------------------------
// Sampled images
// -----------------------------------------
// 1D
DECLARE_SAMPLED_IMGS_1D(float)
DECLARE_SAMPLED_IMGS_1D(float2)
DECLARE_SAMPLED_IMGS_1D(float3)
DECLARE_SAMPLED_IMGS_1D(float4)
DECLARE_SAMPLED_IMGS_1D(half)
DECLARE_SAMPLED_IMGS_1D(half2)
DECLARE_SAMPLED_IMGS_1D(half3)
DECLARE_SAMPLED_IMGS_1D(half4)
DECLARE_SAMPLED_IMGS_1D(uint)
DECLARE_SAMPLED_IMGS_1D(uint2)
DECLARE_SAMPLED_IMGS_1D(uint3)
DECLARE_SAMPLED_IMGS_1D(uint4)
DECLARE_SAMPLED_IMGS_1D(int)
DECLARE_SAMPLED_IMGS_1D(int2)
DECLARE_SAMPLED_IMGS_1D(int3)
DECLARE_SAMPLED_IMGS_1D(int4)

// 2D
DECLARE_SAMPLED_IMGS_2D(float)
DECLARE_SAMPLED_IMGS_2D(float2)
DECLARE_SAMPLED_IMGS_2D(float3)
DECLARE_SAMPLED_IMGS_2D(float4)
DECLARE_SAMPLED_IMGS_2D(half)
DECLARE_SAMPLED_IMGS_2D(half2)
DECLARE_SAMPLED_IMGS_2D(half3)
DECLARE_SAMPLED_IMGS_2D(half4)
DECLARE_SAMPLED_IMGS_2D(uint)
DECLARE_SAMPLED_IMGS_2D(uint2)
DECLARE_SAMPLED_IMGS_2D(uint3)
DECLARE_SAMPLED_IMGS_2D(uint4)
DECLARE_SAMPLED_IMGS_2D(int)
DECLARE_SAMPLED_IMGS_2D(int2)
DECLARE_SAMPLED_IMGS_2D(int3)
DECLARE_SAMPLED_IMGS_2D(int4)

// 3D
DECLARE_SAMPLED_IMGS_3D(float)
DECLARE_SAMPLED_IMGS_3D(float2)
DECLARE_SAMPLED_IMGS_3D(float3)
DECLARE_SAMPLED_IMGS_3D(float4)
DECLARE_SAMPLED_IMGS_3D(half)
DECLARE_SAMPLED_IMGS_3D(half2)
DECLARE_SAMPLED_IMGS_3D(half3)
DECLARE_SAMPLED_IMGS_3D(half4)
DECLARE_SAMPLED_IMGS_3D(uint)
DECLARE_SAMPLED_IMGS_3D(uint2)
DECLARE_SAMPLED_IMGS_3D(uint3)
DECLARE_SAMPLED_IMGS_3D(uint4)
DECLARE_SAMPLED_IMGS_3D(int)
DECLARE_SAMPLED_IMGS_3D(int2)
DECLARE_SAMPLED_IMGS_3D(int3)
DECLARE_SAMPLED_IMGS_3D(int4)

// Cube
DECLARE_SAMPLED_IMGS_CUBE(float)
DECLARE_SAMPLED_IMGS_CUBE(float2)
DECLARE_SAMPLED_IMGS_CUBE(float3)
DECLARE_SAMPLED_IMGS_CUBE(float4)
DECLARE_SAMPLED_IMGS_CUBE(half)
DECLARE_SAMPLED_IMGS_CUBE(half2)
DECLARE_SAMPLED_IMGS_CUBE(half3)
DECLARE_SAMPLED_IMGS_CUBE(half4)
DECLARE_SAMPLED_IMGS_CUBE(uint)
DECLARE_SAMPLED_IMGS_CUBE(uint2)
DECLARE_SAMPLED_IMGS_CUBE(uint3)
DECLARE_SAMPLED_IMGS_CUBE(uint4)
DECLARE_SAMPLED_IMGS_CUBE(int)
DECLARE_SAMPLED_IMGS_CUBE(int2)
DECLARE_SAMPLED_IMGS_CUBE(int3)
DECLARE_SAMPLED_IMGS_CUBE(int4)

// -----------------------------------------
// Storage images
// -----------------------------------------
// 1D
DECLARE_RW_IMGS_1D(float)
DECLARE_RW_IMGS_1D(float2)
DECLARE_RW_IMGS_1D(float3)
DECLARE_RW_IMGS_1D(float4)
DECLARE_RW_IMGS_1D(half)
DECLARE_RW_IMGS_1D(half2)
DECLARE_RW_IMGS_1D(half3)
DECLARE_RW_IMGS_1D(half4)
DECLARE_RW_IMGS_1D(uint)
DECLARE_RW_IMGS_1D(uint2)
DECLARE_RW_IMGS_1D(uint3)
DECLARE_RW_IMGS_1D(uint4)
DECLARE_RW_IMGS_1D(int)
DECLARE_RW_IMGS_1D(int2)
DECLARE_RW_IMGS_1D(int3)
DECLARE_RW_IMGS_1D(int4)

// 2D
DECLARE_RW_IMGS_2D(float)
DECLARE_RW_IMGS_2D(float2)
DECLARE_RW_IMGS_2D(float3)
DECLARE_RW_IMGS_2D(float4)
DECLARE_RW_IMGS_2D(half)
DECLARE_RW_IMGS_2D(half2)
DECLARE_RW_IMGS_2D(half3)
DECLARE_RW_IMGS_2D(half4)
DECLARE_RW_IMGS_2D(uint)
DECLARE_RW_IMGS_2D(uint2)
DECLARE_RW_IMGS_2D(uint3)
DECLARE_RW_IMGS_2D(uint4)
DECLARE_RW_IMGS_2D(int)
DECLARE_RW_IMGS_2D(int2)
DECLARE_RW_IMGS_2D(int3)
DECLARE_RW_IMGS_2D(int4)

// 3D
DECLARE_RW_IMGS_3D(float)
DECLARE_RW_IMGS_3D(float2)
DECLARE_RW_IMGS_3D(float3)
DECLARE_RW_IMGS_3D(float4)
DECLARE_RW_IMGS_3D(half)
DECLARE_RW_IMGS_3D(half2)
DECLARE_RW_IMGS_3D(half3)
DECLARE_RW_IMGS_3D(half4)
DECLARE_RW_IMGS_3D(uint)
DECLARE_RW_IMGS_3D(uint2)
DECLARE_RW_IMGS_3D(uint3)
DECLARE_RW_IMGS_3D(uint4)
DECLARE_RW_IMGS_3D(int)
DECLARE_RW_IMGS_3D(int2)
DECLARE_RW_IMGS_3D(int3)
DECLARE_RW_IMGS_3D(int4)

}
