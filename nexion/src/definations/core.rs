/// Represents the Vulkan API version used by the application.
/// Basically useless as only Vulkan 1.3 is used. Kept for future proofing
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ApiVersion {
    VkApi1_3 = ash::vk::API_VERSION_1_3,
}

/// High level abstraction for instance creation
/// Surface gets created along with the instance
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InstanceDescription {
    pub api_version: ApiVersion,
    pub enable_validation_layers: bool,
}

/// Very high level abstraction for device creation
/// Need to add more options
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceDescription {
    pub use_compute_queue: bool,
    pub use_transfer_queue: bool,
    pub mesh_shaders: bool,
    pub atomic_float_operations: bool,
    pub ray_tracing: bool,
}

impl Default for DeviceDescription {
    fn default() -> Self {
        return DeviceDescription {
            use_compute_queue: true,
            use_transfer_queue: true,
            mesh_shaders: false,
            atomic_float_operations: false,
            ray_tracing: false,
        };
    }
}

/// High level swapchain description
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwapchainDescription {
    pub image_count: u32,
    pub frames_in_flight: usize,
    pub width: u32,
    pub height: u32,
}

/// Wrapper for vk::Extent3D
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Extent3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl Extent3D {
    pub(crate) fn to_vk(&self) -> ash::vk::Extent3D {
        return ash::vk::Extent3D {
            width: self.width,
            height: self.height,
            depth: self.depth,
        };
    }
}

/// Wrapper for vk::Extent2D
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Extent2D {
    pub width: u32,
    pub height: u32,
}

impl Extent2D {
    pub(crate) fn to_vk(&self) -> ash::vk::Extent2D {
        return ash::vk::Extent2D { width: self.width, height: self.height };
    }
}

/// Wrapper for vk::Offset3D
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Offset3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl Offset3D {
    pub(crate) fn to_vk(&self) -> ash::vk::Offset3D {
        return ash::vk::Offset3D { x: self.x, y: self.y, z: self.z };
    }
}

/// Wrapper for vk::Offset2D
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Offset2D {
    pub x: i32,
    pub y: i32,
}

impl Offset2D {
    pub(crate) fn to_vk(&self) -> ash::vk::Offset2D {
        return ash::vk::Offset2D { x: self.x, y: self.y };
    }
}
