use ash::vk;
use std::ops::BitOr;

/// Specifies the desired properties for a memory allocation.
///
/// These variants typically correspond to strategies for choosing
/// the best available Vulkan memory type on the device.
/// Maps to vulkan memory allocator
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryType {
    /// Memory that resides on the **GPU** (Device).
    ///
    /// This memory is typically the fastest for the GPU to access for operations
    /// like rendering or compute, but it is often **inaccessible to the CPU**.
    /// Used for resources like render targets and high-performance buffers.
    DeviceLocal,

    /// Memory that prefers properties making it efficiently **accessible by the Host (CPU)**,
    /// while still allowing the Device (GPU) to use it.
    PreferHost,

    /// Allows the allocator to **automatically select** the most appropriate
    /// memory type based on the resource's usage flags and desired properties.
    Auto,
}

impl MemoryType {
    pub(crate) const fn to_vk_flag(&self) -> vk_mem::MemoryUsage {
        match self {
            Self::DeviceLocal => vk_mem::MemoryUsage::AutoPreferDevice,
            Self::PreferHost => vk_mem::MemoryUsage::AutoPreferHost,
            Self::Auto => vk_mem::MemoryUsage::Auto,
        }
    }
}

/// A wrapper struct for Vulkan's buffer usage flags (`vk::BufferUsageFlags`).
///
/// Can be combined using Bitwise Or (|)
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub struct BufferUsage {
    pub(crate) flags: vk::BufferUsageFlags,
}

impl BufferUsage {
    /// Specifies that the buffer is used as a **storage buffer** in shaders.
    pub const STORAGE: Self = Self {
        flags: vk::BufferUsageFlags::STORAGE_BUFFER,
    };

    /// Specifies that the buffer is used as a **vertex buffer** for drawing commands.
    pub const VERTEX: Self = Self {
        flags: vk::BufferUsageFlags::VERTEX_BUFFER,
    };

    /// Specifies that the buffer is used as an **index buffer** for indexed drawing commands.
    pub const INDEX: Self = Self {
        flags: vk::BufferUsageFlags::INDEX_BUFFER,
    };

    /// Specifies that the buffer is used as a **uniform buffer** in shaders.
    pub const UNIFORM: Self = Self {
        flags: vk::BufferUsageFlags::UNIFORM_BUFFER,
    };

    /// Specifies that the buffer contains **indirect dispatch or drawing parameters**.
    pub const INDIRECT: Self = Self {
        flags: vk::BufferUsageFlags::INDIRECT_BUFFER,
    };

    /// Specifies that the buffer can be used as the **source** in a transfer operation
    pub const TRANSFER_SRC: Self = Self {
        flags: vk::BufferUsageFlags::TRANSFER_SRC,
    };

    /// Specifies that the buffer can be used as the **destination** in a transfer operation
    pub const TRANSFER_DST: Self = Self {
        flags: vk::BufferUsageFlags::TRANSFER_DST,
    };

    // --- Implementation Methods ---

    /// Converts the custom usage struct into the raw Vulkan buffer usage flags.
    pub(crate) fn to_vk_flag(&self) -> vk::BufferUsageFlags {
        self.flags
    }
}

impl BitOr for BufferUsage {
    type Output = Self;
    fn bitor(self, other: Self) -> Self::Output {
        Self { flags: self.flags | other.flags }
    }
}

impl BitOr<BufferUsage> for &BufferUsage {
    type Output = BufferUsage;
    fn bitor(self, other: BufferUsage) -> Self::Output {
        BufferUsage { flags: self.flags | other.flags }
    }
}

impl BitOr<&BufferUsage> for BufferUsage {
    type Output = BufferUsage;
    fn bitor(self, other: &BufferUsage) -> Self::Output {
        BufferUsage { flags: self.flags | other.flags }
    }
}

/// Buffer descriptions, create mapped works only for perfer host memory type
pub struct BufferDescription {
    pub usage: BufferUsage,
    pub size: vk::DeviceSize,
    pub memory_type: MemoryType,
    pub create_mapped: bool,
}

impl Default for BufferDescription {
    fn default() -> Self {
        return BufferDescription {
            usage: BufferUsage::STORAGE,
            size: 10,
            memory_type: MemoryType::Auto,
            create_mapped: false,
        };
    }
}

//// IMAGE DESCRIPTION ////
#[derive(Clone, Copy, Debug)]
pub enum ImageType {
    Type1D,
    Type2D,
    Type3D,
}

impl ImageType {
    pub(crate) fn to_vk(&self) -> vk::ImageType {
        match self {
            Self::Type1D => vk::ImageType::TYPE_1D,
            Self::Type2D => vk::ImageType::TYPE_2D,
            Self::Type3D => vk::ImageType::TYPE_3D,
        }
    }
}

#[derive(Clone)]
pub enum ImageUsage {
    TransferSrc,
    TransferDst,
    Sampled,
    Storage,
    ColorAttachment,
    DepthStencilAttachment,
}

impl ImageUsage {
    pub(crate) const fn to_vk_flag(&self) -> vk::ImageUsageFlags {
        return match self {
            Self::TransferSrc => vk::ImageUsageFlags::TRANSFER_SRC,
            Self::TransferDst => vk::ImageUsageFlags::TRANSFER_DST,
            Self::Sampled => vk::ImageUsageFlags::SAMPLED,
            Self::Storage => vk::ImageUsageFlags::STORAGE,
            Self::ColorAttachment => vk::ImageUsageFlags::COLOR_ATTACHMENT,
            Self::DepthStencilAttachment => vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        };
    }
}
#[derive(Clone, Copy)]
pub enum Format {
    // --- Unsigned Normalized (UNORM) Formats - Standard Color & Textures ---
    Rgba8Unorm,
    Bgra8Unorm,
    Rgb565Unorm,

    // --- Signed/Unsigned Integers (SINT/UINT) ---
    Rgba8Uint,
    Rgba32Sint,

    // --- Float Formats (SFLOAT) - High Precision & Data ---
    Rgba16Float,
    Rg32Float,
    Rgb32Float,
    Rgba32Float,
    R32Float,

    // --- Depth and Stencil Formats ---
    D32Float,
    D24UnormS8Uint,
    D16Unorm,

    // --- Block Compressed (Slightly less common but essential for assets) ---
    BC1RgbaUnorm,
    BC7Unorm,
}

impl Format {
    pub(crate) const fn to_vk_format(&self) -> vk::Format {
        return match self {
            // Unsigned Normalized (UNORM)
            Self::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            Self::Bgra8Unorm => vk::Format::B8G8R8A8_UNORM,
            Self::Rgb565Unorm => vk::Format::R5G6B5_UNORM_PACK16,

            // Signed/Unsigned Integers (SINT/UINT)
            Self::Rgba8Uint => vk::Format::R8G8B8A8_UINT,
            Self::Rgba32Sint => vk::Format::R32G32B32A32_SINT,

            // Float Formats (SFLOAT)
            Self::Rgba16Float => vk::Format::R16G16B16A16_SFLOAT,
            Self::Rg32Float => vk::Format::R32G32_SFLOAT,
            Self::Rgb32Float => vk::Format::R32G32B32_SFLOAT,
            Self::Rgba32Float => vk::Format::R32G32B32A32_SFLOAT,
            Self::R32Float => vk::Format::R32_SFLOAT,

            // Depth and Stencil
            Self::D32Float => vk::Format::D32_SFLOAT,
            Self::D24UnormS8Uint => vk::Format::D24_UNORM_S8_UINT,
            Self::D16Unorm => vk::Format::D16_UNORM,

            // Block Compressed
            Self::BC1RgbaUnorm => vk::Format::BC1_RGBA_UNORM_BLOCK,
            Self::BC7Unorm => vk::Format::BC7_UNORM_BLOCK,
        };
    }
}

#[repr(u32)]
pub enum SampleCount {
    Type1,
    Type2,
    Type4,
    Type8,
    Type16,
    Type32,
    Type64,
}

impl SampleCount {
    pub(crate) const fn to_vk_flags(&self) -> vk::SampleCountFlags {
        return match self {
            Self::Type1 => vk::SampleCountFlags::TYPE_1,
            Self::Type2 => vk::SampleCountFlags::TYPE_2,
            Self::Type4 => vk::SampleCountFlags::TYPE_4,
            Self::Type8 => vk::SampleCountFlags::TYPE_8,
            Self::Type16 => vk::SampleCountFlags::TYPE_16,
            Self::Type32 => vk::SampleCountFlags::TYPE_32,
            Self::Type64 => vk::SampleCountFlags::TYPE_64,
        };
    }
}
#[derive(Copy, Clone, PartialEq)]
pub enum ImageLayout {
    Undefined,
    General,
    ColorAttachment,
    DepthStencilAttachment,
    DepthStencilReadOnly,
    ShaderReadOnly,
    TransferSrc,
    TransferDst,
    PresentSrc,
}

impl ImageLayout {
    #[inline]
    pub(crate) const fn to_vk_layout(self) -> vk::ImageLayout {
        match self {
            ImageLayout::Undefined => vk::ImageLayout::UNDEFINED,
            ImageLayout::General => vk::ImageLayout::GENERAL,
            ImageLayout::ColorAttachment => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ImageLayout::DepthStencilAttachment => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ImageLayout::DepthStencilReadOnly => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            ImageLayout::ShaderReadOnly => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            ImageLayout::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ImageLayout::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ImageLayout::PresentSrc => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }
}

pub struct ImageDescription {
    pub usage: ImageUsage,
    pub format: Format,
    pub image_type: ImageType,
    pub height: u32,
    pub width: u32,
    pub depth: u32,
    pub memory_type: MemoryType,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: SampleCount,
}

impl Default for ImageDescription {
    fn default() -> Self {
        return Self {
            usage: ImageUsage::Sampled,
            format: Format::Rgba16Float,
            image_type: ImageType::Type2D,
            height: 1,
            width: 1,
            depth: 1,
            memory_type: MemoryType::Auto,
            mip_levels: 1,
            array_layers: 1,
            samples: SampleCount::Type1,
        };
    }
}

#[derive(Clone, Copy)]
pub struct ImageSubresourceLayers {
    pub aspect: ImageAspect,
    pub mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

//// Image View Description ////
#[derive(Clone, Copy)]
pub enum ImageViewType {
    Type1D,
    Type2D,
    Type3D,
    Cube,
    Type1DArray,
    Type2DArray,
    CubeArray,
}

impl ImageViewType {
    pub(crate) const fn to_vk_type(&self) -> vk::ImageViewType {
        match self {
            Self::Type1D => vk::ImageViewType::TYPE_1D,
            Self::Type2D => vk::ImageViewType::TYPE_2D,
            Self::Type3D => vk::ImageViewType::TYPE_3D,
            Self::Cube => vk::ImageViewType::CUBE,
            Self::Type1DArray => vk::ImageViewType::TYPE_1D_ARRAY,
            Self::Type2DArray => vk::ImageViewType::TYPE_2D_ARRAY,
            Self::CubeArray => vk::ImageViewType::CUBE_ARRAY,
        }
    }
}

#[derive(Clone, Copy)]
pub enum ImageAspect {
    Color,
    Depth,
    Stencil,
    DepthStencil,
}

impl ImageAspect {
    pub(crate) fn to_vk_aspect(&self) -> vk::ImageAspectFlags {
        match self {
            Self::Color => vk::ImageAspectFlags::COLOR,
            Self::Depth => vk::ImageAspectFlags::DEPTH,
            Self::Stencil => vk::ImageAspectFlags::STENCIL,
            Self::DepthStencil => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        }
    }
}

pub struct ImageViewDescription {
    pub view_type: ImageViewType,
    pub aspect: ImageAspect,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

impl Default for ImageViewDescription {
    fn default() -> Self {
        return Self {
            view_type: ImageViewType::Type2D,
            aspect: ImageAspect::Color,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
    }
}

//// SAMPLER DESCRIPTION ////
#[derive(Clone, Copy, Debug)]
pub enum Filter {
    Nearest,
    Linear,
}

impl Filter {
    pub(crate) fn to_vk(self) -> vk::Filter {
        match self {
            Filter::Nearest => vk::Filter::NEAREST,
            Filter::Linear => vk::Filter::LINEAR,
        }
    }
}

/// Mipmap filter mode
#[derive(Clone, Copy, Debug)]
pub enum SamplerMipmapMode {
    Nearest,
    Linear,
}

impl SamplerMipmapMode {
    pub(crate) fn to_vk(self) -> vk::SamplerMipmapMode {
        match self {
            SamplerMipmapMode::Nearest => vk::SamplerMipmapMode::NEAREST,
            SamplerMipmapMode::Linear => vk::SamplerMipmapMode::LINEAR,
        }
    }
}

/// Addressing (wrap/clamp modes)
#[derive(Clone, Copy, Debug)]
pub enum SamplerAddressMode {
    Repeat,
    MirroredRepeat,
    ClampToEdge,
    ClampToBorder,
}
impl SamplerAddressMode {
    pub(crate) fn to_vk(self) -> vk::SamplerAddressMode {
        match self {
            SamplerAddressMode::Repeat => vk::SamplerAddressMode::REPEAT,
            SamplerAddressMode::MirroredRepeat => vk::SamplerAddressMode::MIRRORED_REPEAT,
            SamplerAddressMode::ClampToEdge => vk::SamplerAddressMode::CLAMP_TO_EDGE,
            SamplerAddressMode::ClampToBorder => vk::SamplerAddressMode::CLAMP_TO_BORDER,
        }
    }
}

/// Border colors for ClampToBorder
#[derive(Clone, Copy, Debug)]
pub enum BorderColor {
    FloatTransparentBlack,
    IntTransparentBlack,
    FloatOpaqueBlack,
    IntOpaqueBlack,
    FloatOpaqueWhite,
    IntOpaqueWhite,
}
impl BorderColor {
    pub(crate) fn to_vk(self) -> vk::BorderColor {
        match self {
            BorderColor::FloatTransparentBlack => vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            BorderColor::IntTransparentBlack => vk::BorderColor::INT_TRANSPARENT_BLACK,
            BorderColor::FloatOpaqueBlack => vk::BorderColor::FLOAT_OPAQUE_BLACK,
            BorderColor::IntOpaqueBlack => vk::BorderColor::INT_OPAQUE_BLACK,
            BorderColor::FloatOpaqueWhite => vk::BorderColor::FLOAT_OPAQUE_WHITE,
            BorderColor::IntOpaqueWhite => vk::BorderColor::INT_OPAQUE_WHITE,
        }
    }
}

/// Optional compare operation for depth samplers
#[derive(Clone, Copy, Debug)]
pub enum CompareOp {
    Never,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always,
}
impl CompareOp {
    pub(crate) fn to_vk(self) -> vk::CompareOp {
        match self {
            CompareOp::Never => vk::CompareOp::NEVER,
            CompareOp::Less => vk::CompareOp::LESS,
            CompareOp::Equal => vk::CompareOp::EQUAL,
            CompareOp::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
            CompareOp::Greater => vk::CompareOp::GREATER,
            CompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
            CompareOp::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
            CompareOp::Always => vk::CompareOp::ALWAYS,
        }
    }
}

pub struct SamplerDescription {
    pub mag_filter: Filter,
    pub min_filter: Filter,
    pub mipmap_mode: SamplerMipmapMode,
    pub address_mode_u: SamplerAddressMode,
    pub address_mode_v: SamplerAddressMode,
    pub address_mode_w: SamplerAddressMode,
    pub mip_lod_bias: f32,
    pub max_anisotropy: Option<f32>,
    pub compare_op: Option<CompareOp>,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border_color: BorderColor,
    pub unnormalized_coordinates: bool,
}

impl Default for SamplerDescription {
    fn default() -> Self {
        Self {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            mipmap_mode: SamplerMipmapMode::Linear,
            address_mode_u: SamplerAddressMode::Repeat,
            address_mode_v: SamplerAddressMode::Repeat,
            address_mode_w: SamplerAddressMode::Repeat,
            mip_lod_bias: 0.0,
            max_anisotropy: None,
            compare_op: None,
            min_lod: 0.0,
            max_lod: 1000.0,
            border_color: BorderColor::IntOpaqueBlack,
            unnormalized_coordinates: false,
        }
    }
}
