use ahash::{HashMap, HashMapExt};
use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use smallvec::smallvec;

use crate::{
    BinarySemaphore, BufferDescription, BufferID, BufferWriteInfo, CommandRecorder, Fence, ImageDescription, ImageID, ImageViewDescription, ImageViewID, ImageWriteInfo, PipelineManager,
    QueueSubmitInfo, QueueType, SamplerDescription, SamplerID, SamplerWriteInfo, Semaphore, Swapchain, SwapchainDescription, TimelineSemaphore,
    backend::{device::InnerDevice, pipelines::InnerPipelineManager, swapchain::InnerSwapchain},
    utils::texture::Texture,
};
use std::sync::Arc;

#[derive(Clone)]
pub struct Device {
    pub(crate) inner: Arc<InnerDevice>,
}

//Swapchain Impl//
impl Device {
    pub fn create_swapchain<W: HasDisplayHandle + HasWindowHandle>(&self, window: &W, swapchain_desc: &SwapchainDescription) -> Swapchain {
        let surface = unsafe { InnerSwapchain::create_surface(&self.inner, window) };
        let inner_swapchain = InnerSwapchain::new(self.inner.clone(), &surface, swapchain_desc, None);

        return Swapchain {
            inner: Arc::new(inner_swapchain),
            surface: Arc::new(surface),
        };
    }
}

// Buffer //
impl Device {
    pub fn create_buffer(&self, buffer_desc: &BufferDescription) -> BufferID {
        return self.inner.create_buffer(buffer_desc);
    }

    pub fn destroy_buffer(&self, id: BufferID) {
        self.inner.destroy_buffer(id);
    }

    pub fn write_data_to_buffer<T: Copy>(&self, buffer_id: BufferID, data: &[T]) {
        self.inner.write_data_to_buffer(buffer_id, data);
    }

    pub fn get_raw_ptr(&self, buffer_id: BufferID) -> *mut u8 {
        return self.inner.get_raw_ptr(buffer_id);
    }

    pub fn get_buffer_address(&self, buffer_id: BufferID) -> vk::DeviceAddress {
        return self.inner.get_device_address(buffer_id);
    }
}

// Image //
impl Device {
    pub fn create_image(&self, image_desc: &ImageDescription) -> ImageID {
        return self.inner.create_image(image_desc);
    }

    pub fn destroy_image(&self, image_id: ImageID) {
        self.inner.destroy_image(image_id);
    }
}

// Image View //
impl Device {
    pub fn create_image_view(&self, image_id: ImageID, image_view_desc: &ImageViewDescription) -> ImageViewID {
        return self.inner.create_image_view(image_id, image_view_desc);
    }

    pub fn destroy_image_view(&self, image_view_id: ImageViewID) {
        self.inner.destroy_image_view(image_view_id);
    }
}

// Sampler //
impl Device {
    pub fn create_sampler(&self, sampler_desc: &SamplerDescription) -> SamplerID {
        return self.inner.create_sampler(sampler_desc);
    }

    pub fn destroy_sampler(&self, sampler_id: SamplerID) {
        self.inner.destroy_sampler(sampler_id);
    }
}

// texture //
impl Device {
    pub fn create_texture(&self, image_desc: &ImageDescription, image_view_desc: &ImageViewDescription, index: u32) -> Texture {
        let img = self.create_image(image_desc);
        let img_view = self.create_image_view(img, image_view_desc);

        self.write_image(&ImageWriteInfo {
            view: img_view,
            image_descriptor_type: crate::ImageDescriptorType::SampledImage,
            index: index,
        });

        return Texture { image: img, image_view: img_view };
    }

    pub fn destory_texture(&self, texture: Texture) {
        self.destroy_image(texture.image);
        self.destroy_image_view(texture.image_view);
    }
}

// Descriptors //
impl Device {
    pub fn write_buffer(&self, buffer_write_info: &BufferWriteInfo) {
        self.inner.write_buffer(buffer_write_info);
    }

    pub fn write_image(&self, image_write_info: &ImageWriteInfo) {
        self.inner.write_image(image_write_info);
    }

    pub fn write_sampler(&self, sampler_write_info: &SamplerWriteInfo) {
        self.inner.write_sampler(sampler_write_info);
    }
}

// Pipeline Manager //
impl Device {
    pub fn create_pipeline_manager(&self) -> PipelineManager {
        return PipelineManager {
            inner: Arc::new(InnerPipelineManager::new(self.inner.clone())),
        };
    }
}

// Command buffer //
impl Device {
    pub fn create_command_recorder(&self, queue_type: QueueType) -> CommandRecorder {
        return CommandRecorder {
            handle: self.inner.create_cmd_recorder_data(queue_type),
            commad_buffers: smallvec![],
            exec_command_buffers: smallvec![],
            current_commad_buffer: vk::CommandBuffer::null(),
            queue_type: queue_type,
            remembered_image_ids: HashMap::new(),
            remembered_buffer_ids: HashMap::new(),
            remembered_image_view_ids: HashMap::new(),
            device: self.inner.clone(),
        };
    }
}

// Sync //
impl Device {
    pub fn create_fence(&self, signaled: bool) -> Fence {
        return Fence {
            handle: self.inner.create_fence(signaled),
        };
    }

    pub fn create_binary_semaphore(&self) -> Semaphore {
        return Semaphore::Binary(BinarySemaphore {
            handle: self.inner.create_binary_semaphore(),
        });
    }

    pub fn create_timeline_semaphore(&self) -> Semaphore {
        return Semaphore::Timeline(TimelineSemaphore {
            handle: self.inner.create_timeline_semaphore(),
        });
    }

    pub fn wait_fence(&self, fence: Fence) {
        self.inner.wait_fence(fence);
    }

    pub fn reset_fence(&self, fence: Fence) {
        self.inner.reset_fence(fence);
    }

    pub fn destroy_fence(&self, fence: Fence) {
        self.inner.destroy_fence(fence);
    }

    pub fn destroy_semaphore(&self, semaphore: Semaphore) {
        self.inner.destroy_semaphore(semaphore);
    }
}

// Queue submissions
impl Device {
    pub fn submit(&self, submit_info: &QueueSubmitInfo) {
        self.inner.submit(submit_info);
    }

    pub fn wait_idle(&self) {
        self.inner.wait_idle();
    }

    pub fn wait_queue(&self, queue_type: QueueType) {
        self.inner.wait_queue(queue_type);
    }
}
