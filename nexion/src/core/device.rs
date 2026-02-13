use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use smallvec::smallvec;

use crate::{
    backend::{device::InnerDevice, pipelines::InnerPipelineManager, swapchain::InnerSwapchain},
    utils::texture::Texture,
    *,
};
use std::sync::Arc;

#[derive(Clone)]
pub struct Device {
    pub(crate) inner_device: Arc<InnerDevice>,
    pub(crate) pipeline_manager: Arc<InnerPipelineManager>,
}

//Swapchain Impl//
impl Device {
    pub fn create_swapchain<W: HasDisplayHandle + HasWindowHandle>(&self, window: &W, swapchain_desc: &SwapchainDescription) -> Swapchain {
        let surface = unsafe { InnerSwapchain::create_surface(&self.inner_device, window) };
        let inner_swapchain = InnerSwapchain::new(self.inner_device.clone(), &surface, swapchain_desc, None);

        return Swapchain {
            inner: Arc::new(inner_swapchain),
            surface: Arc::new(surface),
        };
    }
}

// Buffer //
impl Device {
    pub fn create_buffer(&self, buffer_desc: &BufferDescription) -> BufferId {
        return self.inner_device.create_buffer(buffer_desc);
    }

    pub fn destroy_buffer(&self, id: BufferId) {
        self.inner_device.destroy_buffer(id);
    }

    pub fn write_data_to_buffer<T: Copy>(&self, buffer_id: BufferId, data: &[T]) {
        self.inner_device.write_data_to_buffer(buffer_id, data);
    }

    pub fn get_raw_ptr(&self, buffer_id: BufferId) -> *mut u8 {
        return self.inner_device.get_raw_ptr(buffer_id);
    }

    pub fn get_buffer_address(&self, buffer_id: BufferId) -> u64 {
        return self.inner_device.get_device_address(buffer_id);
    }
}

// Image //
impl Device {
    pub fn create_image(&self, image_desc: &ImageDescription) -> ImageId {
        return self.inner_device.create_image(image_desc);
    }

    pub fn destroy_image(&self, image_id: ImageId) {
        self.inner_device.destroy_image(image_id);
    }
}

// Image View //
impl Device {
    pub fn create_image_view(&self, image_id: ImageId, image_view_desc: &ImageViewDescription) -> ImageViewId {
        return self.inner_device.create_image_view(image_id, image_view_desc);
    }

    pub fn destroy_image_view(&self, image_view_id: ImageViewId) {
        self.inner_device.destroy_image_view(image_view_id);
    }
}

// Sampler //
impl Device {
    pub fn create_sampler(&self, sampler_desc: &SamplerDescription) -> SamplerId {
        return self.inner_device.create_sampler(sampler_desc);
    }

    pub fn destroy_sampler(&self, sampler_id: SamplerId) {
        self.inner_device.destroy_sampler(sampler_id);
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

impl Device {
    pub fn create_rasterization_pipeline(&self, raster_pipeline_desc: &RasterizationPipelineDescription) -> Pipeline {
        return self.pipeline_manager.create_raster_pipeline_data(raster_pipeline_desc);
    }

    pub fn create_compute_pipeline(&self, compute_pipeline_desc: &ComputePipelineDescription) -> Pipeline {
        return self.pipeline_manager.create_compute_pipeline(compute_pipeline_desc);
    }

    pub fn destroy_pipeline(&self, pipeline: Pipeline) {
        self.pipeline_manager.destroy_pipeline(pipeline);
    }
}

// Descriptors //
impl Device {
    pub fn write_buffer(&self, buffer_write_info: &BufferWriteInfo) {
        self.inner_device.write_buffer(buffer_write_info);
    }

    pub fn write_image(&self, image_write_info: &ImageWriteInfo) {
        self.inner_device.write_image(image_write_info);
    }

    pub fn write_sampler(&self, sampler_write_info: &SamplerWriteInfo) {
        self.inner_device.write_sampler(sampler_write_info);
    }

    pub fn upload_descriptors(&self) {
        let mut rec = self.create_command_recorder(QueueType::Transfer);

        rec.begin_recording(CommandBufferUsage::OneTimeSubmit);
        rec.flush_descriptors();
        let exec = rec.end_recording();

        self.inner_device.submit(&QueueSubmitInfo {
            fence: None,
            command_buffers: &[exec],
            wait_semaphores: &[],
            signal_semaphores: &[],
        });

        self.inner_device.wait_idle();
    }
}

// Command buffer //
impl Device {
    pub fn create_command_recorder(&self, queue_type: QueueType) -> CommandRecorder {
        return CommandRecorder {
            handle: self.inner_device.create_cmd_recorder_data(queue_type),
            commad_buffers: smallvec![],
            exec_command_buffers: smallvec![],
            current_commad_buffer: vk::CommandBuffer::null(),
            pipeline_manager: self.pipeline_manager.clone(),
            queue_type: queue_type,
            device: self.inner_device.clone(),
        };
    }
}

// Sync //
impl Device {
    pub fn create_fence(&self, signaled: bool) -> Fence {
        return Fence {
            handle: self.inner_device.create_fence(signaled),
        };
    }

    pub fn create_binary_semaphore(&self) -> Semaphore {
        return Semaphore::Binary(BinarySemaphore {
            handle: self.inner_device.create_binary_semaphore(),
        });
    }

    pub fn create_timeline_semaphore(&self) -> Semaphore {
        return Semaphore::Timeline(TimelineSemaphore {
            handle: self.inner_device.create_timeline_semaphore(),
        });
    }

    pub fn wait_fence(&self, fence: Fence) {
        self.inner_device.wait_fence(fence);
    }

    pub fn reset_fence(&self, fence: Fence) {
        self.inner_device.reset_fence(fence);
    }

    pub fn destroy_fence(&self, fence: Fence) {
        self.inner_device.destroy_fence(fence);
    }

    pub fn destroy_semaphore(&self, semaphore: Semaphore) {
        self.inner_device.destroy_semaphore(semaphore);
    }
}

// Queue submissions
impl Device {
    pub fn submit(&self, submit_info: &QueueSubmitInfo) {
        self.inner_device.submit(submit_info);
    }

    pub fn wait_idle(&self) {
        self.inner_device.wait_idle();
    }

    pub fn wait_queue(&self, queue_type: QueueType) {
        self.inner_device.wait_queue(queue_type);
    }
}
