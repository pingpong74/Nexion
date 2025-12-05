use crate::backend::{
    device::InnerDevice,
    gpu_resources::{GpuBindlessDescriptorPool, GpuResourcePool},
    instance::InnerInstance,
};
use std::sync::{Arc, RwLock};

use super::device::Device;

use crate::{DeviceDescription, InstanceDescription};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[derive(Clone)]
pub struct Instance {
    pub(crate) inner: Arc<InnerInstance>,
}

impl Instance {
    pub fn new<W: HasDisplayHandle + HasWindowHandle>(instance_desc: &InstanceDescription<W>) -> Instance {
        let inner_instance = InnerInstance::new(instance_desc);
        return Instance { inner: Arc::new(inner_instance) };
    }

    pub fn create_device(&self, device_desc: &DeviceDescription) -> Device {
        let (device, physical_device, allocator) = self.inner.create_device_data(device_desc);
        let (graphics_queue, transfer_queue, compute_queue) = InnerInstance::create_queues(&device, &physical_device);
        let bindless_desc = GpuBindlessDescriptorPool::new(&device, 100, 100, 100, 100);

        return Device {
            inner: Arc::new(InnerDevice {
                handle: device,
                physical_device: physical_device,
                allocator: allocator,
                instance: self.inner.clone(),

                //Resource Pools
                bindless_descriptors: bindless_desc,
                buffer_pool: RwLock::new(GpuResourcePool::new()),
                image_pool: RwLock::new(GpuResourcePool::new()),
                image_view_pool: RwLock::new(GpuResourcePool::new()),
                sampler_pool: RwLock::new(GpuResourcePool::new()),

                //Queues
                graphics_queue: graphics_queue,
                transfer_queue: transfer_queue,
                compute_queue: compute_queue,

                rt: None,
            }),
        };
    }
}
