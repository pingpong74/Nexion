use crate::InstanceDescription;

use ash::vk;
//use image::imageops::FilterType::Triangle;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub(crate) struct InnerInstance {
    pub(crate) entry: ash::Entry,
    pub(crate) handle: ash::Instance,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    debug_loader: Option<ash::ext::debug_utils::Instance>,
}

impl InnerInstance {
    pub(crate) fn new<W: HasDisplayHandle + HasWindowHandle>(window: &W, instance_create_info: &InstanceDescription) -> InnerInstance {
        let entry = ash::Entry::linked();

        let mut required_extensions = vec![ash::khr::surface::NAME.as_ptr()];
        let supported_exts = unsafe { entry.enumerate_instance_extension_properties(None).unwrap() };
        let supported_names: Vec<&std::ffi::CStr> = supported_exts.iter().map(|e| unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) }).collect();

        let mut push_if_supported = |ext_name: &std::ffi::CStr| {
            if supported_names.contains(&ext_name) {
                required_extensions.push(ext_name.as_ptr());
                return true;
            }
            false
        };

        let raw_window_handle = window.window_handle().unwrap().as_raw();

        match raw_window_handle {
            raw_window_handle::RawWindowHandle::Win32(_) => {
                push_if_supported(ash::khr::win32_surface::NAME);
            }
            raw_window_handle::RawWindowHandle::Wayland(_) => {
                // If RenderDoc doesn't support Wayland, try to fall back or at least don't crash here
                if !push_if_supported(ash::khr::wayland_surface::NAME) {
                    // If we are on Wayland but the extension isn't supported (RenderDoc),
                    // you might need to force X11 via env vars or handle the error gracefully.
                    println!("Warning: Wayland surface extension not supported by Vulkan driver/layer");
                }
            }
            raw_window_handle::RawWindowHandle::Xcb(_) => {
                push_if_supported(ash::khr::xcb_surface::NAME);
            }
            raw_window_handle::RawWindowHandle::Xlib(_) => {
                required_extensions.push(ash::khr::xlib_surface::NAME.as_ptr());
            }
            raw_window_handle::RawWindowHandle::AppKit(_) => {
                push_if_supported(ash::ext::metal_surface::NAME);
            }
            _ => {}
        }

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

        return InnerInstance {
            entry: entry,
            handle: instance,
            debug_messenger: debug_messenger,
            debug_loader: debug_loader,
        };
    }
}
//////Private functions//////

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
