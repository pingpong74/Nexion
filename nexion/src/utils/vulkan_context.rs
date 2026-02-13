use crate::{utils::texture::Texture, *};
use delegate::delegate;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[allow(unused)]
#[derive(Clone)]
pub struct VulkanContext {
    pub instance: Instance,
    pub device: Device,
    pub swapchain: Swapchain,
    swapchain_description: SwapchainDescription,
}

impl VulkanContext {
    pub fn new<W: HasDisplayHandle + HasWindowHandle>(window: &W, instance_desc: &InstanceDescription, device_desc: &DeviceDescription, swapchain_desc: &SwapchainDescription) -> VulkanContext {
        let instance = Instance::new(window, instance_desc);
        let device = instance.create_device(device_desc);
        let swapchain = device.create_swapchain(window, swapchain_desc);

        return VulkanContext {
            instance: instance,
            device: device,
            swapchain: swapchain,
            swapchain_description: swapchain_desc.clone(),
        };
    }
}

impl VulkanContext {
    pub fn resize(&mut self, width: u32, height: u32) {
        self.device.wait_idle();
        self.swapchain.recreate_swapchain(width, height);
    }
}

impl VulkanContext {
    delegate! {
        to self.device {
            //Buffer
            pub fn create_buffer(&self, buffer_desc: &BufferDescription) -> BufferId;
            pub fn destroy_buffer(&self, id: BufferId);
            pub fn write_data_to_buffer<T: Copy>(&self, buffer_id: BufferId, data: &[T]);
            pub fn get_raw_ptr(&self, buffer_id: BufferId) -> *mut u8;
            //Image
            pub fn create_image(&self, image_desc: &ImageDescription) -> ImageId;
            pub fn destroy_image(&self, image_id: ImageId);
            //Image view
            pub fn create_image_view(&self, image_id: ImageId, image_view_desc: &ImageViewDescription) -> ImageViewId;
            pub fn destroy_image_view(&self, image_view_id: ImageViewId);
            //Sampler
            pub fn create_sampler(&self, sampler_desc: &SamplerDescription) -> SamplerId;
            pub fn destroy_sampler(&self, sampler_id: SamplerId);
            //Texture
            pub fn create_texture(&self, image_desc: &ImageDescription, image_view_desc: &ImageViewDescription, index: u32) -> Texture;
            pub fn destory_texture(&self, texture: Texture);
            // Pipeline
            pub fn create_rasterization_pipeline(&self, raster_pipeline_desc: &RasterizationPipelineDescription) -> Pipeline;
            pub fn create_compute_pipeline(&self, compute_pipeline_desc: &ComputePipelineDescription) -> Pipeline;
            pub fn destroy_pipeline(&self, pipeline: Pipeline);
            // Descriptors
            pub fn write_buffer(&self, buffer_write_info: &BufferWriteInfo);
            pub fn write_image(&self, image_write_info: &ImageWriteInfo);
            pub fn write_sampler(&self, sampler_write_info: &SamplerWriteInfo);
            // Command buffer
            pub fn create_command_recorder(&self, queue_type: QueueType) -> CommandRecorder;
            // Sync
            pub fn create_fence(&self, signaled: bool) -> Fence;
            pub fn create_binary_semaphore(&self) -> Semaphore;
            pub fn create_timeline_semaphore(&self) -> Semaphore;
            pub fn wait_fence(&self, fence: Fence);
            pub fn reset_fence(&self, fence: Fence);
            pub fn destroy_fence(&self, fence: Fence);
            pub fn destroy_semaphore(&self, semaphore: Semaphore);
            // Queue submissions
            pub fn submit(&self, submit_info: &QueueSubmitInfo);
            pub fn wait_idle(&self);
            pub fn wait_queue(&self, queue_type: QueueType);
        }
        to self.swapchain {
            pub fn acquire_image(&self) -> AcquiredImage;
            pub fn present(&self);
        }
    }
}
