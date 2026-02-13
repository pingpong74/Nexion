use crate::{
    backend::{gpu_resources::*, instance::InnerInstance},
    *,
};

use ash::vk;
use gpu_allocator::{vulkan::*, *};
use std::{cell::UnsafeCell, sync::Arc};

pub(crate) struct QueueFamilyIndices {
    pub graphics_family: Option<u32>,
    pub transfer_family: Option<u32>,
    pub compute_family: Option<u32>,
}

impl QueueFamilyIndices {
    fn is_complete(&self) -> bool {
        return self.graphics_family.is_some() && self.compute_family.is_some() && self.transfer_family.is_some();
    }
}

pub(crate) struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub queue_families: QueueFamilyIndices,
}

// TODO: Should i use an unsafe cell instead of RwLock?

pub(crate) struct InnerDevice {
    pub(crate) allocator: UnsafeCell<Allocator>,
    pub(crate) handle: ash::Device,
    pub(crate) physical_device: PhysicalDevice,
    pub(crate) instance: Arc<InnerInstance>,

    //Pools for various gpu resources
    pub(crate) bindless_descriptors: GpuBindlessDescriptorPool,
    pub(crate) buffer_pool: UnsafeCell<ResourcePool<BufferSlot>>,
    pub(crate) image_pool: UnsafeCell<ResourcePool<ImageSlot>>,
    pub(crate) image_view_pool: UnsafeCell<ResourcePool<ImageViewSlot>>,
    pub(crate) sampler_pool: UnsafeCell<ResourcePool<SamplerSlot>>,

    //Queues
    pub(crate) graphics_queue: vk::Queue,
    pub(crate) transfer_queue: vk::Queue,
    pub(crate) compute_queue: vk::Queue,
}

impl InnerDevice {
    pub(crate) fn new(device_desc: &DeviceDescription, instance: Arc<InnerInstance>) -> InnerDevice {
        // Required device extensions (swapchain needed for presentation)
        let mut device_extensions = vec![ash::khr::swapchain::NAME.as_ptr(), ash::khr::synchronization2::NAME.as_ptr()];

        if device_desc.ray_tracing {
            device_extensions.push(ash::khr::acceleration_structure::NAME.as_ptr());
            device_extensions.push(ash::khr::ray_tracing_pipeline::NAME.as_ptr());
            device_extensions.push(ash::khr::deferred_host_operations::NAME.as_ptr());
            if !device_extensions.contains(&ash::khr::spirv_1_4::NAME.as_ptr()) {
                device_extensions.push(ash::khr::spirv_1_4::NAME.as_ptr());
            }
        }

        if device_desc.atomic_float_operations {
            device_extensions.push(ash::ext::shader_atomic_float::NAME.as_ptr());
        }

        if device_desc.mesh_shaders {
            device_extensions.push(ash::ext::mesh_shader::NAME.as_ptr());
            device_extensions.push(ash::khr::shader_float_controls::NAME.as_ptr());
            if !device_extensions.contains(&ash::khr::spirv_1_4::NAME.as_ptr()) {
                device_extensions.push(ash::khr::spirv_1_4::NAME.as_ptr());
            }
        }

        let physical_device = {
            let dev = Self::select_physical_device(&instance, &device_extensions);
            if dev.is_none() {
                panic!("Failed to find vulkan compatible device")
            }

            dev.unwrap()
        };

        let unique_families: Vec<u32> = {
            let mut v = vec![
                physical_device.queue_families.graphics_family.unwrap(),
                physical_device.queue_families.transfer_family.unwrap(),
                physical_device.queue_families.compute_family.unwrap(),
            ];
            v.sort();
            v.dedup();
            v
        };

        // Queue priorities (all same)
        let priorities = [1.0_f32];
        let queue_infos: Vec<_> = unique_families.iter().map(|&family| vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)).collect();

        // Existing common features
        let features = vk::PhysicalDeviceFeatures::default().shader_int64(true).multi_draw_indirect(true).sampler_anisotropy(true);
        let mut float_atomic_features = vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT::default().shader_buffer_float32_atomic_add(true);

        let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

        let mut indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default()
            .shader_sampled_image_array_non_uniform_indexing(true)
            .descriptor_binding_partially_bound(true)
            .runtime_descriptor_array(true)
            .descriptor_binding_variable_descriptor_count(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_storage_image_update_after_bind(true)
            .descriptor_binding_storage_texel_buffer_update_after_bind(true)
            .descriptor_binding_uniform_buffer_update_after_bind(true)
            .descriptor_binding_uniform_texel_buffer_update_after_bind(true);

        let mut sync2 = vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut timeline_sem = vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
        let mut buffer_device_address = vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut vk_features_11 = vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true).variable_pointers(true).variable_pointers_storage_buffer(true);

        // Ray tracing
        let mut accel_struct_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut rt_pipeline_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
        let mut ray_query_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default();

        if device_desc.ray_tracing {
            accel_struct_features = accel_struct_features.acceleration_structure(true);
            rt_pipeline_features = rt_pipeline_features.ray_tracing_pipeline(true);
            ray_query_features = ray_query_features.ray_query(true);
        }

        // mesh shaders
        let mut mesh_shader_features = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();

        if device_desc.mesh_shaders {
            mesh_shader_features = mesh_shader_features.mesh_shader(true).task_shader(true);
        }

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut indexing_features)
            .push_next(&mut dynamic_rendering_features)
            .push_next(&mut sync2)
            .push_next(&mut timeline_sem)
            .push_next(&mut buffer_device_address)
            .push_next(&mut vk_features_11)
            .push_next(&mut mesh_shader_features)
            .features(features);

        if device_desc.ray_tracing {
            features2 = features2.push_next(&mut accel_struct_features).push_next(&mut rt_pipeline_features).push_next(&mut ray_query_features);
        }

        if device_desc.atomic_float_operations {
            features2 = features2.push_next(&mut float_atomic_features);
        }

        let create_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_infos).enabled_extension_names(&device_extensions).push_next(&mut features2);

        let dev = unsafe { instance.handle.create_device(physical_device.handle, &create_info, None).expect("Failed to create logical device") };

        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.handle.clone(),
            device: dev.clone(),
            physical_device: physical_device.handle,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: true,
            allocation_sizes: AllocationSizes::default(),
        })
        .expect("Failed to create allocator");

        let graphics_queue = unsafe { dev.get_device_queue(physical_device.queue_families.graphics_family.unwrap(), 0) };
        let compute_queue = unsafe { dev.get_device_queue(physical_device.queue_families.compute_family.unwrap(), 0) };
        let transfer_queue = unsafe { dev.get_device_queue(physical_device.queue_families.transfer_family.unwrap(), 0) };

        let device_address_buffer = {
            let indices = [
                physical_device.queue_families.compute_family.unwrap(),
                physical_device.queue_families.graphics_family.unwrap(),
                physical_device.queue_families.transfer_family.unwrap(),
            ];

            let buffer_create_info = vk::BufferCreateInfo::default()
                .usage(vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER)
                .size(100 * 64)
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&indices);

            let buffer = unsafe { dev.create_buffer(&buffer_create_info, None).expect("Failed to create buffer ") };
            let memory_requirements = unsafe { dev.get_buffer_memory_requirements(buffer) };

            let allocation_create_info = AllocationCreateDesc {
                name: "o",
                requirements: memory_requirements,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::DedicatedBuffer(buffer),
            };

            let allocation = allocator.allocate(&allocation_create_info).expect("Failed to allocate memory on device");

            unsafe {
                dev.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()).expect("Failed to bind buffer memory");
            }
            let buffer_address = unsafe { dev.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer)) };

            BufferSlot {
                handle: buffer,
                allocation: allocation,
                address: buffer_address,
            }
        };

        let bindless_desc = GpuBindlessDescriptorPool::new(&dev, device_address_buffer, 100, 100, 100);

        return InnerDevice {
            handle: dev,
            physical_device: physical_device,
            allocator: UnsafeCell::new(allocator),
            instance: instance,

            //Resource Pools
            bindless_descriptors: bindless_desc,
            buffer_pool: UnsafeCell::new(ResourcePool::new()),
            image_pool: UnsafeCell::new(ResourcePool::new()),
            image_view_pool: UnsafeCell::new(ResourcePool::new()),
            sampler_pool: UnsafeCell::new(ResourcePool::new()),

            //Queues
            graphics_queue: graphics_queue,
            transfer_queue: transfer_queue,
            compute_queue: compute_queue,
        };
    }

    fn get_queue_families(instance: &Arc<InnerInstance>, physical_device: ash::vk::PhysicalDevice) -> Option<QueueFamilyIndices> {
        let queue_families = unsafe { instance.handle.get_physical_device_queue_family_properties(physical_device) };

        let mut indices = QueueFamilyIndices {
            graphics_family: None,
            transfer_family: None,
            compute_family: None,
        };

        for (i, family) in queue_families.iter().enumerate() {
            // Graphics
            if family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS) && indices.graphics_family.is_none() {
                indices.graphics_family = Some(i as u32);
            }

            // Compute (dedicated if possible)
            if family.queue_flags.contains(ash::vk::QueueFlags::COMPUTE) && !family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS) && indices.compute_family.is_none() {
                indices.compute_family = Some(i as u32);
            }

            // Transfer (dedicated if possible)
            if family.queue_flags.contains(ash::vk::QueueFlags::TRANSFER) && !family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS) && !family.queue_flags.contains(ash::vk::QueueFlags::COMPUTE) && indices.transfer_family.is_none() {
                indices.transfer_family = Some(i as u32);
            }

            if indices.is_complete() {
                break;
            }
        }

        if indices.is_complete() {
            return Some(indices);
        } else {
            return None;
        }
    }

    fn check_device_extension_support(instance: &Arc<InnerInstance>, device: ash::vk::PhysicalDevice, required_extensions: &Vec<*const i8>) -> bool {
        let available_extensions = unsafe { instance.handle.enumerate_device_extension_properties(device).expect("Failed to enumerate device extensions") };

        return required_extensions.iter().all(|&required_ptr| {
            let required_str = unsafe { std::ffi::CStr::from_ptr(required_ptr) };

            available_extensions.iter().any(|avail| {
                let avail_str = unsafe { std::ffi::CStr::from_ptr(avail.extension_name.as_ptr()) };

                avail_str == required_str
            })
        });
    }

    fn select_physical_device(instance: &Arc<InnerInstance>, required_extensions: &Vec<*const i8>) -> Option<PhysicalDevice> {
        let devices = unsafe { instance.handle.enumerate_physical_devices().expect("Failed to enumerate physical devices") };

        let mut best_device: Option<(i32, PhysicalDevice)> = None;

        for device in devices {
            let mut props: vk::PhysicalDeviceProperties2 = vk::PhysicalDeviceProperties2::default();
            unsafe {
                instance.handle.get_physical_device_properties2(device, &mut props);
            };

            if let Some(qf) = Self::get_queue_families(instance, device) {
                if !Self::check_device_extension_support(instance, device, required_extensions) {
                    continue;
                }

                // Score device: discrete = 1000, integrated = 100, others = 10
                let score = match props.properties.device_type {
                    ash::vk::PhysicalDeviceType::DISCRETE_GPU => 1000,
                    ash::vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
                    _ => 10,
                };

                // Prefer larger max image dimension as tiebreaker
                let score = score + props.properties.limits.max_image_dimension2_d as i32;

                let candidate = PhysicalDevice { handle: device, queue_families: qf };

                if let Some((best_score, _)) = &best_device {
                    if score > *best_score {
                        best_device = Some((score, candidate));
                    }
                } else {
                    best_device = Some((score, candidate));
                }
            }
        }

        return best_device.map(|(_, dev)| dev);
    }
}

// Buffer //
impl InnerDevice {
    pub(crate) fn create_buffer(&self, buffer_desc: &BufferDescription) -> BufferId {
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

        let buffer = unsafe { self.handle.create_buffer(&buffer_create_info, None).expect("Failed to create buffer ") };
        let memory_requirements = unsafe { self.handle.get_buffer_memory_requirements(buffer) };

        let allocation_create_info = AllocationCreateDesc {
            name: "o",
            requirements: memory_requirements,
            location: buffer_desc.memory_type.to_vk_flag(),
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = unsafe { self.allocator.get().as_mut().unwrap().allocate(&allocation_create_info).expect("Failed to allocate memory on device") };

        unsafe {
            self.handle.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()).expect("Failed to bind buffer memory");
        }
        let buffer_address = unsafe { self.handle.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer)) };

        let raw_id = unsafe {
            (&mut *self.buffer_pool.get()).add(BufferSlot {
                handle: buffer,
                address: buffer_address,
                allocation: allocation,
            })
        };

        return BufferId { id: raw_id };
    }

    pub(crate) fn destroy_buffer(&self, id: BufferId) {
        let res = unsafe { (&mut *self.buffer_pool.get()).delete(id.id) };

        unsafe {
            self.allocator.get().as_mut().unwrap().free(res.allocation).expect("Failed to deallocate buffer");
            self.handle.destroy_buffer(res.handle, None);
        }
    }

    pub(crate) fn write_data_to_buffer<T: Copy>(&self, buffer_id: BufferId, data: &[T]) {
        let buffer = unsafe { (&mut *self.buffer_pool.get()).get_ref(buffer_id.id) };

        unsafe {
            let ptr = buffer.allocation.mapped_ptr().expect("Tried to write to an unmapped buffer").as_ptr() as *mut T;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    pub(crate) fn get_raw_ptr(&self, buffer_id: BufferId) -> *mut u8 {
        let buffer = unsafe { (&mut *self.buffer_pool.get()).get_ref(buffer_id.id) };

        return buffer.allocation.mapped_ptr().expect("Tried to write to an unmapped buffer").as_ptr() as *mut u8;
    }

    pub(crate) fn get_device_address(&self, buffer_id: BufferId) -> vk::DeviceAddress {
        let buffer = unsafe { (&mut *self.buffer_pool.get()).get_ref(buffer_id.id) };

        return buffer.address;
    }
}

// Image //
impl InnerDevice {
    pub(crate) fn create_image(&self, image_desc: &ImageDescription) -> ImageId {
        let image_create_info = vk::ImageCreateInfo::default()
            .usage(image_desc.usage.to_vk_flag())
            .extent(image_desc.extent.to_vk())
            .format(image_desc.format.to_vk_format())
            .array_layers(image_desc.array_layers)
            .mip_levels(image_desc.mip_levels)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .image_type(image_desc.image_type.to_vk())
            .samples(image_desc.samples.to_vk_flags())
            .tiling(vk::ImageTiling::OPTIMAL);

        let image = unsafe { self.handle.create_image(&image_create_info, None).expect("Failed to create Image") };

        let memory_requirements = unsafe { self.handle.get_image_memory_requirements(image) };

        let allocation_create_info = AllocationCreateDesc {
            name: "o",
            requirements: memory_requirements,
            location: image_desc.memory_type.to_vk_flag(),
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = unsafe { self.allocator.get().as_mut().unwrap().allocate(&allocation_create_info).expect("Failed to allocate memory on device") };

        unsafe {
            self.handle.bind_image_memory(image, allocation.memory(), allocation.offset()).expect("Failed to bind image memory");
        }

        let id = unsafe {
            (&mut *self.image_pool.get()).add(ImageSlot {
                handle: image,
                allocation: allocation,
                format: image_desc.format.to_vk_format(),
            })
        };

        return ImageId { id: id };
    }

    pub(crate) fn destroy_image(&self, id: ImageId) {
        let img = unsafe { (&mut *self.image_pool.get()).delete(id.id) };

        unsafe {
            self.allocator.get().as_mut().unwrap().free(img.allocation).expect("Failed to deallocate image");
            self.handle.destroy_image(img.handle, None);
        };
    }
}

// Image View //
impl InnerDevice {
    pub(crate) fn create_image_view(&self, image_id: ImageId, image_view_description: &ImageViewDescription) -> ImageViewId {
        let img = unsafe { (&mut *self.image_pool.get()).get_ref(image_id.id) };

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
            .subresource_range(image_view_description.subresources.to_vk_subresource_range());

        let image_view = unsafe { self.handle.create_image_view(&image_view_create_info, None).expect("Failed to create Image view") };

        let id = unsafe { (&mut *self.image_view_pool.get()).add(ImageViewSlot { handle: image_view }) };

        return ImageViewId { id: id };
    }

    pub(crate) fn destroy_image_view(&self, image_view_id: ImageViewId) {
        let img_view = unsafe { (&mut *self.image_view_pool.get()).delete(image_view_id.id) };

        unsafe {
            self.handle.destroy_image_view(img_view.handle, None);
        }
    }
}

// Sampler //
impl InnerDevice {
    pub(crate) fn create_sampler(&self, sampler_desc: &SamplerDescription) -> SamplerId {
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

        let id = unsafe { (&mut *self.sampler_pool.get()).add(SamplerSlot { handle: sampler }) };

        return SamplerId { id: id };
    }

    pub(crate) fn destroy_sampler(&self, sampler_id: SamplerId) {
        let sampler = unsafe { (&mut *self.sampler_pool.get()).delete(sampler_id.id) };

        unsafe {
            self.handle.destroy_sampler(sampler.handle, None);
        };
    }
}

// Descriptor //
impl InnerDevice {
    pub(crate) fn write_buffer(&self, buffer_write_info: &BufferWriteInfo) {
        let buffer = unsafe { (&mut *self.buffer_pool.get()).get_ref(buffer_write_info.buffer.id) };

        self.bindless_descriptors.write_buffer(buffer.address, buffer_write_info.index);
    }

    pub(crate) fn write_image(&self, image_write_info: &ImageWriteInfo) {
        let img_view = unsafe { (&mut *self.image_view_pool.get()).get_ref(image_write_info.view.id) };

        match image_write_info.image_descriptor_type {
            ImageDescriptorType::SampledImage => self.bindless_descriptors.write_sampled_image(&self.handle, img_view.handle, image_write_info.index),
            ImageDescriptorType::StorageImage => self.bindless_descriptors.write_storage_image(&self.handle, img_view.handle, image_write_info.index),
        }
    }

    pub(crate) fn write_sampler(&self, sampler_write_info: &SamplerWriteInfo) {
        let sampler = unsafe { (&mut *self.sampler_pool.get()).get_ref(sampler_write_info.sampler.id) };

        self.bindless_descriptors.write_sampler(&self.handle, sampler.handle, sampler_write_info.index);
    }
}

//// Command buffers ////
impl InnerDevice {
    pub(crate) fn create_cmd_recorder_data(&self, queue_type: QueueType) -> vk::CommandPool {
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
            .map(|s| vk::SemaphoreSubmitInfo::default().semaphore(s.semaphore.handle()).stage_mask(s.pipeline_stage.to_vk()).value(s.value.unwrap_or(0)))
            .collect();

        let wait_infos: Vec<vk::SemaphoreSubmitInfo> = submit_info
            .wait_semaphores
            .iter()
            .map(|s| vk::SemaphoreSubmitInfo::default().semaphore(s.semaphore.handle()).stage_mask(s.pipeline_stage.to_vk()).value(s.value.unwrap_or(0)))
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
        let buffer_pool = unsafe { &mut (*self.buffer_pool.get()) };
        let image_pool = unsafe { &mut (*self.image_pool.get()) };
        let image_view_pool = unsafe { &mut (*self.image_view_pool.get()) };
        let sampler_pool = unsafe { &mut (*self.sampler_pool.get()) };

        unsafe {
            self.bindless_descriptors.cleanup(&self.handle, &mut (*self.allocator.get()));
            std::ptr::drop_in_place(&mut self.allocator);
            self.handle.destroy_device(None);
        }
    }
}

unsafe impl Send for InnerDevice {}
unsafe impl Sync for InnerDevice {}
