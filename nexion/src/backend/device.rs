use crate::{
    BufferDescription, BufferID, BufferWriteInfo, Fence, ImageDescription, ImageDescriptorType, ImageID, ImageViewDescription, ImageViewID, ImageWriteInfo, QueueSubmitInfo, QueueType,
    SamplerDescription, SamplerID, SamplerWriteInfo, Semaphore, SwapchainDescription,
    backend::{
        gpu_resources::{BufferSlot, GpuBindlessDescriptorPool, GpuResourcePool, ImageSlot, ImageViewSlot, SamplerSlot},
        instance::InnerInstance,
    },
};

use super::instance::PhysicalDevice;
use ash::vk::{self};
use std::{
    ptr::null_mut,
    sync::{Arc, RwLock},
    u64,
};
use vk_mem::*;

pub(crate) struct InnerDevice {
    pub(crate) allocator: Allocator,
    pub(crate) handle: ash::Device,
    pub(crate) physical_device: PhysicalDevice,
    pub(crate) instance: Arc<InnerInstance>,

    //Pools for various gpu resources
    pub(crate) bindless_descriptors: GpuBindlessDescriptorPool,
    pub(crate) buffer_pool: RwLock<GpuResourcePool<BufferSlot>>,
    pub(crate) image_pool: RwLock<GpuResourcePool<ImageSlot>>,
    pub(crate) image_view_pool: RwLock<GpuResourcePool<ImageViewSlot>>,
    pub(crate) sampler_pool: RwLock<GpuResourcePool<SamplerSlot>>,

    //Queues
    pub(crate) graphics_queue: vk::Queue,
    pub(crate) transfer_queue: vk::Queue,
    pub(crate) compute_queue: vk::Queue,

    // Extensions
    pub(crate) rt: Option<ash::khr::ray_tracing_pipeline::Device>,
}

// Swapchain Creation //
impl InnerDevice {
    fn choose_surface_format(available_formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
        available_formats
            .iter()
            .cloned()
            .find(|f| f.format == vk::Format::R16G16B16A16_SFLOAT && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .unwrap_or_else(|| available_formats[0])
    }

    fn choose_present_mode(available_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
        if available_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        }
    }

    fn choose_extent(capabilities: &vk::SurfaceCapabilitiesKHR, width: u32, height: u32) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::MAX {
            capabilities.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
                height: height.clamp(capabilities.min_image_extent.height, capabilities.max_image_extent.height),
            }
        }
    }

    pub(crate) fn create_swapchain_data(
        &self,
        swapchain_description: &SwapchainDescription,
        old_swapchain: vk::SwapchainKHR,
    ) -> (ash::khr::swapchain::Device, vk::SwapchainKHR, Vec<ImageID>, Vec<ImageViewID>) {
        let swapchain_loader = ash::khr::swapchain::Device::new(&self.instance.handle, &self.handle);

        let support = &self.physical_device.swapchain_support;

        let extent = InnerDevice::choose_extent(&support.capabilities, swapchain_description.width, swapchain_description.height);
        let present_mode = InnerDevice::choose_present_mode(&support.present_modes);
        let surface_format = InnerDevice::choose_surface_format(&support.formats);

        let graphics_family = self.physical_device.queue_families.graphics_family.expect("This shouldnt be possible lol");
        let present_family = self.physical_device.queue_families.presetation_family.expect("This shouldnt be possible lol");

        let mut create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.instance.surface.handle)
            .min_image_count(swapchain_description.image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT);

        let queue_family_indices = [graphics_family, present_family];

        if graphics_family != present_family {
            create_info = create_info.image_sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&queue_family_indices);
        } else {
            create_info = create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE);
        }

        create_info = create_info
            .pre_transform(support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(old_swapchain);

        let swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None).expect("Failed to create swapchain") };

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain).expect("Failed to get swapchain images") };

        let image_ids: Vec<ImageID> = images
            .iter()
            .map(|&image| {
                let id = self.image_pool.write().unwrap().add(ImageSlot {
                    handle: image,
                    allocation: vk_mem::Allocation(std::ptr::null_mut()),
                    alloc_info: vk_mem::AllocationInfo {
                        memory_type: 0,
                        device_memory: vk::DeviceMemory::null(),
                        user_data: 0,
                        mapped_data: null_mut(),
                        offset: 0,
                        size: 0,
                    },
                    format: surface_format.format,
                });

                ImageID { id: id }
            })
            .collect();

        let image_views: Vec<ImageViewID> = image_ids.iter().map(|&image_id| self.create_image_view(image_id, &ImageViewDescription::default())).collect();

        return (swapchain_loader, swapchain, image_ids, image_views);
    }
}

// Buffer //
impl InnerDevice {
    pub(crate) fn create_buffer(&self, buffer_desc: &BufferDescription) -> BufferID {
        let indices = [
            self.physical_device.queue_families.compute_family.unwrap(),
            self.physical_device.queue_families.graphics_family.unwrap(),
            self.physical_device.queue_families.transfer_family.unwrap(),
        ];

        let buffer_create_info = vk::BufferCreateInfo::default()
            .usage(buffer_desc.usage.to_vk_flag() | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
            .size(buffer_desc.size)
            .sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(&indices);

        let mut allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: buffer_desc.memory_type.to_vk_flag(),
            ..Default::default()
        };

        if buffer_desc.create_mapped {
            allocation_create_info.flags = AllocationCreateFlags::MAPPED | AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE;
        }

        let (buffer, allocation) = unsafe { self.allocator.create_buffer(&buffer_create_info, &allocation_create_info).expect("Failed to create buffer") };

        let alloc_info = self.allocator.get_allocation_info(&allocation);

        let buffer_address = unsafe { self.handle.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer)) };

        let id = self.buffer_pool.write().unwrap().add(BufferSlot {
            handle: buffer,
            address: buffer_address,
            allocation: allocation,
            alloc_info: alloc_info,
        });

        return BufferID { id: id };
    }

    pub(crate) fn destroy_buffer(&self, id: BufferID) {
        let mut res = self.buffer_pool.write().unwrap().delete(id.id);

        unsafe {
            self.allocator.destroy_buffer(res.handle, &mut res.allocation);
        }
    }

    pub(crate) fn write_data_to_buffer<T: Copy>(&self, buffer_id: BufferID, data: &[T]) {
        let buffer_pool = self.buffer_pool.read().unwrap();
        let buffer = buffer_pool.get_ref(buffer_id.id);

        unsafe {
            let ptr = buffer.alloc_info.mapped_data as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }
}

// Image //
impl InnerDevice {
    pub(crate) fn create_image(&self, image_desc: &ImageDescription) -> ImageID {
        let image_create_info = vk::ImageCreateInfo::default()
            .usage(image_desc.usage.to_vk_flag())
            .extent(vk::Extent3D {
                height: image_desc.height,
                width: image_desc.width,
                depth: image_desc.depth,
            })
            .format(image_desc.format.to_vk_format())
            .array_layers(image_desc.array_layers)
            .mip_levels(image_desc.mip_levels)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .image_type(image_desc.image_type.to_vk())
            .samples(image_desc.samples.to_vk_flags())
            .tiling(vk::ImageTiling::OPTIMAL);

        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: image_desc.memory_type.to_vk_flag(),
            ..Default::default()
        };

        let (image, allocation) = unsafe { self.allocator.create_image(&image_create_info, &allocation_create_info).expect("Failed to create image") };

        let alloc_info = self.allocator.get_allocation_info(&allocation);

        let id = self.image_pool.write().unwrap().add(ImageSlot {
            handle: image,
            allocation: allocation,
            alloc_info: alloc_info,
            format: image_desc.format.to_vk_format(),
        });

        return ImageID { id: id };
    }

    pub(crate) fn destroy_image(&self, id: ImageID) {
        let mut img = self.image_pool.write().unwrap().delete(id.id);

        unsafe {
            self.allocator.destroy_image(img.handle, &mut img.allocation);
        };
    }
}

// Image View //
impl InnerDevice {
    pub(crate) fn create_image_view(&self, image_id: ImageID, image_view_description: &ImageViewDescription) -> ImageViewID {
        let pool = self.image_pool.read().unwrap();
        let img = pool.get_ref(image_id.id);

        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(img.handle)
            .view_type(image_view_description.view_type.to_vk_type())
            .format(img.format)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(image_view_description.aspect.to_vk_aspect())
                    .base_mip_level(image_view_description.base_mip_level)
                    .level_count(image_view_description.layer_count)
                    .base_array_layer(image_view_description.base_array_layer)
                    .layer_count(image_view_description.layer_count),
            );

        let image_view = unsafe { self.handle.create_image_view(&image_view_create_info, None).expect("Failed to create Image view") };

        let id = self.image_view_pool.write().unwrap().add(ImageViewSlot {
            handle: image_view,
            parent_image: img.handle,
        });

        return ImageViewID { id: id };
    }

    pub(crate) fn destroy_image_view(&self, image_view_id: ImageViewID) {
        let img_view = self.image_view_pool.write().unwrap().delete(image_view_id.id);

        unsafe {
            self.handle.destroy_image_view(img_view.handle, None);
        }
    }
}

// Sampler //
impl InnerDevice {
    pub(crate) fn create_sampler(&self, sampler_desc: &SamplerDescription) -> SamplerID {
        let create_info = vk::SamplerCreateInfo::default()
            .mag_filter(sampler_desc.mag_filter.to_vk())
            .min_filter(sampler_desc.min_filter.to_vk())
            .mipmap_mode(sampler_desc.mipmap_mode.to_vk())
            .address_mode_u(sampler_desc.address_mode_u.to_vk())
            .address_mode_v(sampler_desc.address_mode_v.to_vk())
            .address_mode_w(sampler_desc.address_mode_w.to_vk())
            .mip_lod_bias(sampler_desc.mip_lod_bias)
            .anisotropy_enable(sampler_desc.max_anisotropy.is_some())
            .max_anisotropy(sampler_desc.max_anisotropy.unwrap_or(1.0))
            .compare_enable(sampler_desc.compare_op.is_some())
            .compare_op(sampler_desc.compare_op.map(|c| c.to_vk()).unwrap_or(vk::CompareOp::ALWAYS))
            .min_lod(sampler_desc.min_lod)
            .max_lod(sampler_desc.max_lod)
            .border_color(sampler_desc.border_color.to_vk())
            .unnormalized_coordinates(sampler_desc.unnormalized_coordinates);

        let sampler = unsafe { self.handle.create_sampler(&create_info, None).expect("Failed to create sampler") };

        let id = self.sampler_pool.write().unwrap().add(SamplerSlot { handle: sampler });

        return SamplerID { id: id };
    }

    pub(crate) fn destroy_sampler(&self, sampler_id: SamplerID) {
        let sampler = self.sampler_pool.write().unwrap().delete(sampler_id.id);

        unsafe {
            self.handle.destroy_sampler(sampler.handle, None);
        };
    }
}

// Descriptor //
impl InnerDevice {
    pub(crate) fn write_buffer(&self, buffer_write_info: &BufferWriteInfo) {
        let buffer_pool = self.buffer_pool.read().unwrap();
        let buffer = buffer_pool.get_ref(buffer_write_info.buffer.id);

        self.bindless_descriptors.write_buffer(&self.handle, buffer.handle, buffer_write_info.index);
    }

    pub(crate) fn write_image(&self, image_write_info: &ImageWriteInfo) {
        let img_view_pool = self.image_view_pool.read().unwrap();
        let img_view = img_view_pool.get_ref(image_write_info.view.id);

        match image_write_info.image_descriptor_type {
            ImageDescriptorType::SampledImage => self.bindless_descriptors.write_sampled_image(&self.handle, img_view.handle, image_write_info.index),
            ImageDescriptorType::StorageImage => self.bindless_descriptors.write_storage_image(&self.handle, img_view.handle, image_write_info.index),
        }
    }

    pub(crate) fn write_sampler(&self, sampler_write_info: &SamplerWriteInfo) {
        let sampler_pool = self.sampler_pool.read().unwrap();
        let sampler = sampler_pool.get_ref(sampler_write_info.sampler.id);

        self.bindless_descriptors.write_sampler(&self.handle, sampler.handle, sampler_write_info.index);
    }
}

//// Command buffers ////
impl InnerDevice {
    pub(crate) fn createcmd_recorder_data(&self, queue_type: QueueType) -> vk::CommandPool {
        let cmd_pool_info = vk::CommandPoolCreateInfo::default().flags(vk::CommandPoolCreateFlags::empty()).queue_family_index(match queue_type {
            QueueType::Compute => self.physical_device.queue_families.compute_family.unwrap(),
            QueueType::Transfer => self.physical_device.queue_families.transfer_family.unwrap(),
            QueueType::Graphics => self.physical_device.queue_families.graphics_family.unwrap(),
            QueueType::None => panic!("Please dont pass a None queue for command pool"),
        });

        let pool = unsafe { self.handle.create_command_pool(&cmd_pool_info, None).expect("Failed to create command pool") };

        return pool;
    }
}

//// Sync ////
impl InnerDevice {
    pub(crate) fn create_fence(&self, signaled: bool) -> vk::Fence {
        let create_info = vk::FenceCreateInfo::default().flags(if signaled { vk::FenceCreateFlags::SIGNALED } else { vk::FenceCreateFlags::empty() });

        return unsafe { self.handle.create_fence(&create_info, None).expect("Failed to create Fence") };
    }

    pub(crate) fn create_binary_semaphore(&self) -> vk::Semaphore {
        let create_info = vk::SemaphoreCreateInfo::default().flags(vk::SemaphoreCreateFlags::empty());

        return unsafe { self.handle.create_semaphore(&create_info, None).expect("Failed to create semaphore") };
    }

    pub(crate) fn create_timeline_semaphore(&self) -> vk::Semaphore {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE).initial_value(0);

        let create_info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);

        return unsafe { self.handle.create_semaphore(&create_info, None).expect("Failed to create timeline semaphore") };
    }

    pub(crate) fn destroy_fence(&self, fence: Fence) {
        unsafe {
            self.handle.destroy_fence(fence.handle, None);
        }
    }

    pub(crate) fn destroy_semaphore(&self, semaphore: Semaphore) {
        unsafe {
            self.handle.destroy_semaphore(semaphore.handle(), None);
        }
    }

    pub(crate) fn wait_fence(&self, fence: Fence) {
        unsafe {
            self.handle.wait_for_fences(&[fence.handle], true, u64::MAX).expect("Failed to wait for fence");
        }
    }

    pub(crate) fn reset_fence(&self, fence: Fence) {
        unsafe {
            self.handle.reset_fences(&[fence.handle]).expect("Failed to reset fence");
        }
    }
}

//// Queue submission ////
impl InnerDevice {
    // We need to take an array as an input
    pub(crate) fn submit(&self, submit_info: &QueueSubmitInfo) {
        let signal_infos: Vec<vk::SemaphoreSubmitInfo> = submit_info
            .signal_semaphores
            .iter()
            .map(|s| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(s.semaphore.handle())
                    .stage_mask(s.pipeline_stage.to_vk())
                    .value(s.value.unwrap_or(0))
            })
            .collect();

        let wait_infos: Vec<vk::SemaphoreSubmitInfo> = submit_info
            .wait_semaphores
            .iter()
            .map(|s| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(s.semaphore.handle())
                    .stage_mask(s.pipeline_stage.to_vk())
                    .value(s.value.unwrap_or(0))
            })
            .collect();

        let cmd_type = submit_info.command_buffers[0].queue_type;

        let cmd_infos: Vec<vk::CommandBufferSubmitInfo> = submit_info
            .command_buffers
            .iter()
            .map(|cb| {
                assert!(cb.queue_type == cmd_type);

                vk::CommandBufferSubmitInfo::default().command_buffer(cb.handle).device_mask(0)
            })
            .collect();

        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(wait_infos.as_slice())
            .command_buffer_infos(cmd_infos.as_slice())
            .signal_semaphore_infos(signal_infos.as_slice())
            .flags(vk::SubmitFlags::empty());

        let fence_handle = match &submit_info.fence {
            Some(f) => f.handle,
            None => vk::Fence::null(),
        };

        let queue = match cmd_type {
            QueueType::Graphics => self.graphics_queue,
            QueueType::Compute => self.compute_queue,
            QueueType::Transfer => self.transfer_queue,
            _ => panic!("WHY ARE U PASSING NONE QUEUE"),
        };

        unsafe {
            self.handle.queue_submit2(queue, &[submit], fence_handle).expect("Queue submit failed");
        }
    }

    pub(crate) fn wait_idle(&self) {
        unsafe {
            self.handle.device_wait_idle().expect("Failed to wait device idle");
        }
    }

    pub(crate) fn wait_queue(&self, queue_type: QueueType) {
        let queue = match queue_type {
            QueueType::Graphics => self.graphics_queue,
            QueueType::Compute => self.compute_queue,
            QueueType::Transfer => self.transfer_queue,
            _ => panic!("WHY ARE U PASSING NONE QUEUE"),
        };

        unsafe {
            self.handle.queue_wait_idle(queue).expect("Failed to wait for queue");
        }
    }
}

impl Drop for InnerDevice {
    fn drop(&mut self) {
        self.bindless_descriptors.cleanup(&self.handle);

        unsafe {
            std::ptr::drop_in_place(&mut self.allocator);
            self.handle.destroy_device(None);
        }
    }
}
