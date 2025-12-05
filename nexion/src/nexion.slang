// module vulkany;

#pragma once

// Buffer stuff
namespace vulcany {

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

/// This manages Read write buffers
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
}

}

// Sampled images
namespace vulcany {
[[vk::binding(1, 0)]]

public enum ImageDim { Dim1D, Dim2D, Dim3D, CubeMap }

Texture2D<float4> sampled_images_f4[];

public struct SampledImage<T = float4, let dim : ImageDim = ImageDim::Dim2D> {
  public void get_texture() {
    switch (dim) {
    case ImageDim::Dim1D:
      break;
    case ImageDim::Dim2D:
      break;
    case ImageDim::Dim3D:
      break;
    case ImageDim::CubeMap:
      break;
    }
  }
}

// Storage Images
[[vk::binding(2, 0)]]
RWTexture2D<float4> storage_images_f4[];

public struct StorageImage<T = float4, let dim : ImageDim = ImageDim::Dim2D> {}
}
