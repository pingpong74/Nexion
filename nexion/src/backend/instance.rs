use crate::{ApiVersion, DeviceDescription, InstanceDescription};

use ash::vk;
//use image::imageops::FilterType::Triangle;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use std::{ffi::CStr, sync::Arc};

pub(crate) struct Surface {
    pub(crate) handle: vk::SurfaceKHR,
    pub(crate) loader: ash::khr::surface::Instance,
}

pub(crate) struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

pub(crate) struct QueueFamilyIndices {
    pub graphics_family: Option<u32>,
    pub presetation_family: Option<u32>,
    pub transfer_family: Option<u32>,
    pub compute_family: Option<u32>,
}

pub(crate) struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub swapchain_support: SwapchainSupport,
    pub queue_families: QueueFamilyIndices,
    pub properties: vk::PhysicalDeviceProperties2<'static>,
    //pub rt_props: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'a>,
}

pub(crate) struct InnerInstance {
    entry: ash::Entry,
    pub(crate) handle: ash::Instance,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    debug_loader: Option<ash::ext::debug_utils::Instance>,
    pub(crate) surface: Surface,
    physical_device_extensions: Vec<&'static CStr>,
    api_version: ApiVersion,
}

impl InnerInstance {
    pub(crate) fn new<W: HasDisplayHandle + HasWindowHandle>(instance_create_info: &InstanceDescription<W>) -> InnerInstance {
        let entry = ash::Entry::linked();

        let mut required_extensions = vec![ash::khr::surface::NAME.as_ptr()];

        let raw_window_handle = instance_create_info.window.window_handle().expect("Failed to accuqire raw window handle").as_raw();

        match raw_window_handle {
            //Windows
            raw_window_handle::RawWindowHandle::Win32(_) => {
                required_extensions.push(ash::khr::win32_surface::NAME.as_ptr());
            }

            //Wayland
            raw_window_handle::RawWindowHandle::Wayland(_) => {
                required_extensions.push(ash::khr::wayland_surface::NAME.as_ptr());
            }

            //Xcb
            raw_window_handle::RawWindowHandle::Xcb(_) => {
                required_extensions.push(ash::khr::xcb_surface::NAME.as_ptr());
            }

            //Apple
            raw_window_handle::RawWindowHandle::AppKit(_) => {
                required_extensions.push(ash::ext::metal_surface::NAME.as_ptr());
            }

            //Panic if none found :(
            _ => {}
        };

        if instance_create_info.enable_validation_layers {
            required_extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let app_info = vk::ApplicationInfo {
            api_version: instance_create_info.api_version.clone() as u32,
            ..Default::default()
        };

        let mut create_info = vk::InstanceCreateInfo::default().application_info(&app_info).enabled_extension_names(&required_extensions);

        let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
            .message_type(vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION)
            .pfn_user_callback(Some(InnerInstance::vulkan_debug_callback));

        if instance_create_info.enable_validation_layers {
            create_info = create_info.push_next(&mut debug_create_info);
        }

        let instance = unsafe { entry.create_instance(&create_info, None).expect("Failed to create instance") };

        let mut debug_messenger: Option<vk::DebugUtilsMessengerEXT> = None;
        let mut debug_loader: Option<ash::ext::debug_utils::Instance> = None;

        if instance_create_info.enable_validation_layers {
            let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);

            debug_messenger = Some(unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_create_info, None) }.expect("Debug Utils Messenger creation failed"));

            debug_loader = Some(debug_utils_loader);
        }

        let surface = unsafe { InnerInstance::create_surface(&entry, &instance, &instance_create_info.window) };

        return InnerInstance {
            entry: entry,
            handle: instance,
            debug_messenger: debug_messenger,
            debug_loader: debug_loader,
            surface: surface,
            physical_device_extensions: vec![ash::khr::swapchain::NAME],
            api_version: instance_create_info.api_version.clone(),
        };
    }

    pub(crate) fn create_device_data(&self, device_desc: &DeviceDescription) -> (ash::Device, PhysicalDevice, vk_mem::Allocator) {
        let physical_device = {
            let dev = self.select_physical_device();
            if dev.is_none() {
                panic!("Failed to find vulkan compatible device")
            }

            dev.unwrap()
        };

        let unique_families: Vec<u32> = {
            let mut v = vec![
                physical_device.queue_families.graphics_family.unwrap(),
                physical_device.queue_families.presetation_family.unwrap(),
                physical_device.queue_families.transfer_family.unwrap(),
                physical_device.queue_families.compute_family.unwrap(),
            ];
            v.sort();
            v.dedup();
            v
        };

        // Queue priorities (all same)
        let priorities = [1.0_f32];
        let queue_infos: Vec<_> = unique_families
            .iter()
            .map(|&family| vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities))
            .collect();

        // Required device extensions (swapchain needed for presentation)
        let mut device_extensions = vec![ash::khr::swapchain::NAME.as_ptr(), ash::khr::synchronization2::NAME.as_ptr()];

        // Existing common features
        let features = vk::PhysicalDeviceFeatures::default().shader_int64(true);
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
        let mut vk_features_11 = vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);

        // ----> CONDITIONAL RAY TRACING ADDITIONS <----
        let mut accel_struct_features = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default();
        let mut rt_pipeline_features = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default();
        let mut ray_query_features = vk::PhysicalDeviceRayQueryFeaturesKHR::default();

        if device_desc.ray_tracing {
            // Add RT extensions
            device_extensions.push(ash::khr::acceleration_structure::NAME.as_ptr());
            device_extensions.push(ash::khr::ray_tracing_pipeline::NAME.as_ptr());
            device_extensions.push(ash::khr::deferred_host_operations::NAME.as_ptr());

            // Enable the Vulkan features
            accel_struct_features = accel_struct_features.acceleration_structure(true);
            rt_pipeline_features = rt_pipeline_features.ray_tracing_pipeline(true);

            // Optional: RT queries inside shaders
            ray_query_features = ray_query_features.ray_query(true);
        }

        if device_desc.atomic_float_operations {
            device_extensions.push(ash::ext::shader_atomic_float::NAME.as_ptr());
        }

        // ----> Build final feature2 chain <----
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut indexing_features)
            .push_next(&mut dynamic_rendering_features)
            .push_next(&mut sync2)
            .push_next(&mut timeline_sem)
            .push_next(&mut buffer_device_address)
            .push_next(&mut vk_features_11)
            .features(features);

        // Add ray tracing feature structs *only if* enabled
        if device_desc.ray_tracing {
            features2 = features2.push_next(&mut accel_struct_features).push_next(&mut rt_pipeline_features).push_next(&mut ray_query_features);
        }

        if device_desc.atomic_float_operations {
            features2 = features2.push_next(&mut float_atomic_features);
        }

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut features2);

        let dev = unsafe { self.handle.create_device(physical_device.handle, &create_info, None).expect("Failed to create logical device") };

        let mut allocator_create_info = vk_mem::AllocatorCreateInfo::new(&self.handle, &dev, physical_device.handle);
        allocator_create_info.vulkan_api_version = self.api_version.clone() as u32;
        allocator_create_info.flags = vk_mem::AllocatorCreateFlags::EXTERNALLY_SYNCHRONIZED | vk_mem::AllocatorCreateFlags::BUFFER_DEVICE_ADDRESS;

        let allocator = unsafe { vk_mem::Allocator::new(allocator_create_info).expect("Failed to create vma allocator") };

        return (dev, physical_device, allocator);
    }

    pub(crate) fn create_queues(device: &ash::Device, physical_device: &PhysicalDevice) -> (vk::Queue, vk::Queue, vk::Queue) {
        return unsafe {
            (
                device.get_device_queue(physical_device.queue_families.graphics_family.unwrap(), 0),
                device.get_device_queue(physical_device.queue_families.transfer_family.unwrap(), 0),
                device.get_device_queue(physical_device.queue_families.compute_family.unwrap(), 0),
            )
        };
    }
}

//////Private functions//////

//Surface creation
impl InnerInstance {
    unsafe fn create_surface<W: HasDisplayHandle + HasWindowHandle>(entry: &ash::Entry, instance: &ash::Instance, window: &W) -> Surface {
        let raw_window = window.window_handle().unwrap().as_raw();
        let raw_display = window.display_handle().unwrap().as_raw();

        let surface_handle = match (raw_window, raw_display) {
            // ---------------- Windows ----------------
            (RawWindowHandle::Win32(w), RawDisplayHandle::Windows(_)) => {
                let info = ash::vk::Win32SurfaceCreateInfoKHR::default().hinstance(w.hinstance.unwrap().get()).hwnd(w.hwnd.get());
                let loader = ash::khr::win32_surface::Instance::new(entry, instance);
                unsafe { loader.create_win32_surface(&info, None).expect("Failed to create surface") }
            }

            // ---------------- XCB ----------------
            (RawWindowHandle::Xcb(w), RawDisplayHandle::Xcb(d)) => {
                let info = ash::vk::XcbSurfaceCreateInfoKHR::default().connection(d.connection.unwrap().as_ptr()).window(w.window.get());
                let loader = ash::khr::xcb_surface::Instance::new(entry, instance);
                unsafe { loader.create_xcb_surface(&info, None).expect("Failed to create surface") }
            }

            // ---------------- Wayland ----------------
            (RawWindowHandle::Wayland(w), RawDisplayHandle::Wayland(d)) => {
                let info = ash::vk::WaylandSurfaceCreateInfoKHR::default().display(d.display.as_ptr()).surface(w.surface.as_ptr());
                let loader = ash::khr::wayland_surface::Instance::new(entry, instance);
                unsafe { loader.create_wayland_surface(&info, None).expect("Failed to create surface") }
            }

            // ---------------- macOS ----------------
            (RawWindowHandle::AppKit(w), RawDisplayHandle::AppKit(_)) => {
                let info = ash::vk::MetalSurfaceCreateInfoEXT::default().layer(w.ns_view.as_ptr());
                let loader = ash::ext::metal_surface::Instance::new(entry, instance);
                unsafe { loader.create_metal_surface(&info, None).expect("Failed to create surface") }
            }

            // ---------------- Unsupported ----------------
            _ => panic!("Unsupported platform or mismatched window/display handle"),
        };

        return Surface {
            handle: surface_handle,
            loader: ash::khr::surface::Instance::new(entry, instance),
        };
    }
}

//Physical device selection
impl InnerInstance {
    fn get_queue_families(&self, physical_device: ash::vk::PhysicalDevice) -> Option<QueueFamilyIndices> {
        let queue_families = unsafe { self.handle.get_physical_device_queue_family_properties(physical_device) };

        let mut indices = QueueFamilyIndices {
            graphics_family: None,
            transfer_family: None,
            compute_family: None,
            presetation_family: None,
        };

        for (i, family) in queue_families.iter().enumerate() {
            // Graphics
            if family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS) && indices.graphics_family.is_none() {
                indices.graphics_family = Some(i as u32);
            }

            // Compute (dedicated if possible)
            if family.queue_flags.contains(ash::vk::QueueFlags::COMPUTE) && indices.compute_family.is_none() {
                if !family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS) {
                    indices.compute_family = Some(i as u32);
                }
            }

            // Transfer (dedicated if possible)
            if family.queue_flags.contains(ash::vk::QueueFlags::TRANSFER) && indices.transfer_family.is_none() {
                if !family.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS) && !family.queue_flags.contains(ash::vk::QueueFlags::COMPUTE) {
                    indices.transfer_family = Some(i as u32);
                }
            }

            // Presentation
            let present_support = unsafe { self.surface.loader.get_physical_device_surface_support(physical_device, i as u32, self.surface.handle).unwrap_or(false) };
            if present_support && indices.presetation_family.is_none() {
                indices.presetation_family = Some(i as u32);
            }
        }

        if indices.graphics_family.is_some() && indices.presetation_family.is_some() {
            Some(indices)
        } else {
            None
        }
    }

    fn get_swapchain_support(&self, physical_device: ash::vk::PhysicalDevice) -> Option<SwapchainSupport> {
        unsafe {
            let capabilities = self.surface.loader.get_physical_device_surface_capabilities(physical_device, self.surface.handle).ok()?;

            let formats = self.surface.loader.get_physical_device_surface_formats(physical_device, self.surface.handle).ok()?;

            let present_modes = self.surface.loader.get_physical_device_surface_present_modes(physical_device, self.surface.handle).ok()?;

            if formats.is_empty() || present_modes.is_empty() {
                return None;
            } else {
                return Some(SwapchainSupport { capabilities, formats, present_modes });
            }
        }
    }

    fn check_device_extension_support(&self, device: ash::vk::PhysicalDevice) -> bool {
        let available_extensions = unsafe { self.handle.enumerate_device_extension_properties(device).expect("Failed to enumerate device extensions") };

        let available_extension_names: Vec<&std::ffi::CStr> = available_extensions
            .iter()
            .map(|ext| {
                // Convert raw `extension_name` to CStr
                let raw_name = unsafe { std::ffi::CStr::from_ptr(ext.extension_name.as_ptr()) };
                raw_name
            })
            .collect();

        // Check all required extensions are present
        self.physical_device_extensions.iter().all(|&required| available_extension_names.iter().any(|&avail| avail == required))
    }

    fn select_physical_device(&self) -> Option<PhysicalDevice> {
        let devices = unsafe { self.handle.enumerate_physical_devices().expect("Failed to enumerate physical devices") };

        let mut best_device: Option<(i32, PhysicalDevice)> = None;

        for device in devices {
            let mut props: vk::PhysicalDeviceProperties2 = vk::PhysicalDeviceProperties2::default();
            unsafe {
                self.handle.get_physical_device_properties2(device, &mut props);
            };

            if let (Some(sc), Some(qf)) = (self.get_swapchain_support(device), self.get_queue_families(device)) {
                if !self.check_device_extension_support(device) {
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

                let owned_props = props; // Copy, no pNext
                //let owned_rt_props = rt_props;

                let candidate = PhysicalDevice {
                    handle: device,
                    swapchain_support: sc,
                    queue_families: qf,
                    properties: owned_props,
                    //rt_props: owned_rt_props,
                };

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

//Debug Messenger
impl InnerInstance {
    #[allow(unused)]
    unsafe extern "system" fn vulkan_debug_callback(
        severity: ash::vk::DebugUtilsMessageSeverityFlagsEXT,
        types: ash::vk::DebugUtilsMessageTypeFlagsEXT,
        data: *const ash::vk::DebugUtilsMessengerCallbackDataEXT,
        _user: *mut std::ffi::c_void,
    ) -> ash::vk::Bool32 {
        let message = unsafe { std::ffi::CStr::from_ptr((*data).p_message).to_string_lossy().into_owned() };
        println!("[VULKAN, {:?} {:?}]: {}", severity, types, message);

        ash::vk::FALSE
    }
}

//Drop implementation
impl Drop for InnerInstance {
    fn drop(&mut self) {
        unsafe {
            self.surface.loader.destroy_surface(self.surface.handle, None);

            if !self.debug_messenger.is_none() {
                if self.debug_loader.is_none() {
                    panic!("Created debug utils but not debug loader")
                }

                self.debug_loader.as_mut().unwrap().destroy_debug_utils_messenger(self.debug_messenger.unwrap(), None);
            }

            self.handle.destroy_instance(None);
        };
    }
}
