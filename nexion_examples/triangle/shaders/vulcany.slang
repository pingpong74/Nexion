// module vulkany;

#pragma once

namespace vulcanny {

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

  public inline __subscript(uint index)->T {
    get { return raw_buffer.Load<T>(index * sizeof(T)); }
    set { raw_buffer.Store<T>(index * sizeof(T), newValue); }
  }
}

}
