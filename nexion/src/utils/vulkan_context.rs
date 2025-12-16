use crate::*;
use delegate::delegate;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[allow(unused)]
#[derive(Clone)]
pub struct VulkanContext {
    instance: Instance,
    device: Device,
    swapchain: Swapchain,
    pipeline_manager: PipelineManager,
    swapchain_description: SwapchainDescription,
}

impl VulkanContext {
    pub fn new<W: HasDisplayHandle + HasWindowHandle>(window: &W, instance_desc: &InstanceDescription, device_desc: &DeviceDescription, swapchain_desc: &SwapchainDescription) -> VulkanContext {
        let instance = Instance::new(window, instance_desc);
        let device = instance.create_device(device_desc);
        let swapchain = device.create_swapchain(window, swapchain_desc);
        let pipeline_manager = device.create_pipeline_manager();

        return VulkanContext {
            instance: instance,
            device: device,
            swapchain: swapchain,
            pipeline_manager: pipeline_manager,
            swapchain_description: swapchain_desc.clone(),
        };
    }
}

impl VulkanContext {
    pub fn resize(&mut self, width: u32, height: u32) {
        self.swapchain.recreate_swapchain(width, height);
    }
}

impl VulkanContext {
    delegate! {
        to self.device {
            //Buffer
            pub fn create_buffer(&self, buffer_desc: &BufferDescription) -> BufferID;
            pub fn destroy_buffer(&self, id: BufferID);
            pub fn write_data_to_buffer<T: Copy>(&self, buffer_id: BufferID, data: &[T]);
            pub fn get_raw_ptr(&self, buffer_id: BufferID) -> *mut u8;
            //Image
            pub fn create_image(&self, image_desc: &ImageDescription) -> ImageID;
            pub fn destroy_image(&self, image_id: ImageID);
            //Image view
            pub fn create_image_view(&self, image_id: ImageID, image_view_desc: &ImageViewDescription) -> ImageViewID;
            pub fn destroy_image_view(&self, image_view_id: ImageViewID);
            //Sampler
            pub fn create_sampler(&self, sampler_desc: &SamplerDescription) -> SamplerID;
            pub fn destroy_sampler(&self, sampler_id: SamplerID);
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
            pub fn acquire_image(&self) -> (ImageID, ImageViewID, Semaphore, Semaphore);
            pub fn present(&self);
        }
        to self.pipeline_manager {
            pub fn create_rasterization_pipeline(&self, raster_pipeline_desc: &RasterizationPipelineDescription) -> RasterizationPipeline;
            pub fn create_compute_pipeline(&self, compute_pipeline_desc: &ComputePipelineDescription) -> ComputePipeline;
        }
    }
}
