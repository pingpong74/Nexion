use crate::backend::{device::InnerDevice, instance::InnerInstance, pipelines::InnerPipelineManager};
use std::sync::Arc;

use super::device::Device;

use crate::{DeviceDescription, InstanceDescription};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[derive(Clone)]
pub struct Instance {
    pub(crate) inner: Arc<InnerInstance>,
}

impl Instance {
    pub fn new<W: HasDisplayHandle + HasWindowHandle>(window: &W, instance_desc: &InstanceDescription) -> Instance {
        let inner_instance = InnerInstance::new(window, instance_desc);
        return Instance { inner: Arc::new(inner_instance) };
    }

    pub fn create_device(&self, device_desc: &DeviceDescription) -> Device {
        let inner_device = Arc::new(InnerDevice::new(device_desc, self.inner.clone()));
        let pipeline_manager = Arc::new(InnerPipelineManager::new(inner_device.clone()));
        return Device {
            inner_device: inner_device,
            pipeline_manager: pipeline_manager,
        };
    }
}
