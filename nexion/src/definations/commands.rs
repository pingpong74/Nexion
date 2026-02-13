use ash::vk;
use std::u64;

use crate::*;

use crate::{BufferId, ExecutableCommandBuffer, Fence, ImageId, ImageViewId, Semaphore};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QueueType {
    Graphics,
    Transfer,
    Compute,
    None,
}

#[derive(Debug, Clone, Copy)]
pub enum CommandBufferUsage {
    OneTimeSubmit,
    RenderPassContinue,
    SimultaneousUse,
}

impl CommandBufferUsage {
    pub(crate) const fn to_vk_flags(&self) -> vk::CommandBufferUsageFlags {
        match self {
            Self::OneTimeSubmit => vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            Self::RenderPassContinue => vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
            Self::SimultaneousUse => vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IndexType {
    Uint32,
    Uint16,
}

impl IndexType {
    pub(crate) const fn to_vk_flag(&self) -> vk::IndexType {
        match self {
            Self::Uint32 => vk::IndexType::UINT32,
            Self::Uint16 => vk::IndexType::UINT16,
        }
    }
}

// Render begin info
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderArea {
    pub offset: Offset2D,
    pub extent: Extent2D,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LoadOp {
    Load,
    Clear,
    DontCare,
}

impl LoadOp {
    #[inline]
    pub(crate) const fn to_vk(&self) -> vk::AttachmentLoadOp {
        match self {
            Self::Load => vk::AttachmentLoadOp::LOAD,
            Self::Clear => vk::AttachmentLoadOp::CLEAR,
            Self::DontCare => vk::AttachmentLoadOp::DONT_CARE,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StoreOp {
    Store,
    DontCare,
    None,
}

impl StoreOp {
    #[inline]
    pub(crate) const fn to_vk(&self) -> vk::AttachmentStoreOp {
        match self {
            Self::Store => vk::AttachmentStoreOp::STORE,
            Self::DontCare => vk::AttachmentStoreOp::DONT_CARE,
            Self::None => vk::AttachmentStoreOp::NONE,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResolveMode {
    None,
    SampleZero,
    Average,
    Min,
    Max,
}

impl ResolveMode {
    #[inline]
    pub(crate) const fn to_vk(&self) -> vk::ResolveModeFlags {
        match self {
            Self::None => vk::ResolveModeFlags::NONE,
            Self::SampleZero => vk::ResolveModeFlags::SAMPLE_ZERO,
            Self::Average => vk::ResolveModeFlags::AVERAGE,
            Self::Min => vk::ResolveModeFlags::MIN,
            Self::Max => vk::ResolveModeFlags::MAX,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ClearValue {
    ColorFloat([f32; 4]),
    ColorInt([i32; 4]),
    ColorUint([u32; 4]),
    DepthStencil { depth: f32, stencil: u32 },
}

impl ClearValue {
    #[inline]
    pub(crate) const fn to_vk(&self) -> vk::ClearValue {
        match self {
            Self::ColorFloat(v) => vk::ClearValue { color: vk::ClearColorValue { float32: *v } },
            Self::ColorInt(v) => vk::ClearValue { color: vk::ClearColorValue { int32: *v } },
            Self::ColorUint(v) => vk::ClearValue { color: vk::ClearColorValue { uint32: *v } },
            Self::DepthStencil { depth, stencil } => vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: *depth, stencil: *stencil },
            },
        }
    }

    /// Common helper for zero clear
    pub const fn black() -> Self {
        Self::ColorFloat([0.0, 0.0, 0.0, 1.0])
    }

    /// Common helper for white clear
    pub const fn white() -> Self {
        Self::ColorFloat([1.0, 1.0, 1.0, 1.0])
    }

    /// Common helper for depth clear
    pub const fn depth_one() -> Self {
        Self::DepthStencil { depth: 1.0, stencil: 0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderingAttachment {
    pub image_view: ImageViewId,
    pub image_layout: ImageLayout,
    pub resolve_mode: ResolveMode,
    pub resolve_image_view: Option<ImageViewId>,
    pub resolve_image_layout: ImageLayout,
    pub load_op: LoadOp,
    pub store_op: StoreOp,
    pub clear_value: ClearValue,
}

impl Default for RenderingAttachment {
    fn default() -> Self {
        Self {
            image_view: ImageViewId { id: u64::max_value() },
            image_layout: ImageLayout::Undefined,
            resolve_image_view: None,
            resolve_image_layout: ImageLayout::Undefined,
            load_op: LoadOp::Clear,
            store_op: StoreOp::Store,
            resolve_mode: ResolveMode::None,
            clear_value: ClearValue::ColorFloat([0.0, 0.0, 0.0, 0.0]),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum RenderingFlags {
    None,
    ContentsSecondaryCommandBuffers,
    Suspending,
    Resuming,
}

impl RenderingFlags {
    #[inline]
    pub(crate) const fn to_vk(&self) -> vk::RenderingFlags {
        match self {
            Self::None => vk::RenderingFlags::empty(),
            Self::ContentsSecondaryCommandBuffers => vk::RenderingFlags::CONTENTS_SECONDARY_COMMAND_BUFFERS,
            Self::Suspending => vk::RenderingFlags::SUSPENDING,
            Self::Resuming => vk::RenderingFlags::RESUMING,
        }
    }
}

pub struct RenderingBeginInfo<'a> {
    pub render_area: RenderArea,
    pub rendering_flags: RenderingFlags,
    pub view_mask: u32,
    pub layer_count: u32,
    pub color_attachments: &'a [RenderingAttachment],
    pub depth_attachment: Option<RenderingAttachment>,
    pub stencil_attachment: Option<RenderingAttachment>,
}

impl<'a> Default for RenderingBeginInfo<'a> {
    fn default() -> Self {
        Self {
            render_area: RenderArea {
                offset: Offset2D { x: 0, y: 0 },
                extent: Extent2D { width: 0, height: 0 },
            },
            rendering_flags: RenderingFlags::None,
            view_mask: 0,
            layer_count: 0,
            color_attachments: &[],
            depth_attachment: None,
            stencil_attachment: None,
        }
    }
}

// Indirect draw

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DrawIndirectCommand {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawIndirectInfo {
    pub buffer: BufferId,
    pub offset: u64,
    pub draw_count: u32,
    pub stride: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawIndexedIndirectInfo {
    pub buffer: BufferId,
    pub offset: u64,
    pub draw_count: u32,
    pub stride: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawIndirectCountInfo {
    pub buffer: BufferId,
    pub offset: u64,
    pub count_buffer: BufferId,
    pub count_offset: u64,
    pub max_draw_count: u32,
    pub stride: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DrawIndexedIndirectCountInfo {
    pub buffer: BufferId,
    pub offset: u64,
    pub count_buffer: BufferId,
    pub count_offset: u64,
    pub max_draw_count: u32,
    pub stride: u32,
}

// Compute
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DispatchInfo {
    pub group_count_x: u32,
    pub group_count_y: u32,
    pub group_count_z: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DispatchIndirectInfo {
    pub buffer: BufferId,
    pub offset: u64,
}

// Copy commands
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CopyRegion {
    pub src_offset: u64,
    pub dst_offset: u64,
    pub size: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BufferCopyInfo<'a> {
    pub src_buffer: BufferId,
    pub dst_buffer: BufferId,
    pub regions: &'a [CopyRegion],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BufferFillInfo {
    pub buffer: BufferId,
    pub offset: u64,
    pub size: u64,
    pub data: u32,
}

pub struct BufferUpdateInfo<'a, T: Copy> {
    pub buffer: BufferId,
    pub offset: u64,
    pub data: &'a [T],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BufferImageCopyInfo {
    pub buffer: BufferId,
    pub image: ImageId,
    pub dst_image_layout: ImageLayout,
    pub region: BufferImageCopyRegion,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BufferImageCopyRegion {
    pub buffer_offset: u64,
    pub buffer_row_length: u32,
    pub buffer_image_height: u32,
    pub image_subresource: ImageSubresources,
    pub image_offset: Offset3D,
    pub image_extent: Extent3D,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageCopyInfo {
    pub src_image: ImageId,
    pub src_image_layout: ImageLayout,
    pub dst_image: ImageId,
    pub dst_image_layout: ImageLayout,
    pub region: ImageCopyRegion,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageCopyRegion {
    pub src_subresource: ImageSubresources,
    pub src_offset: Offset3D,
    pub dst_subresource: ImageSubresources,
    pub dst_offset: Offset3D,
    pub extent: Extent3D,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlitInfo<'a> {
    pub src_image: ImageId,
    pub src_layout: ImageLayout,
    pub dst_image: ImageId,
    pub dst_layout: ImageLayout,
    pub regions: &'a [BlitRegion],
    pub filter: Filter,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlitRegion {
    pub src_subresource: ImageSubresources,
    pub src_offsets: [Offset3D; 2],
    pub dst_subresource: ImageSubresources,
    pub dst_offsets: [Offset3D; 2],
}

// Memory barriers
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PipelineStage {
    None,
    TopOfPipe,
    BottomOfPipe,
    DrawIndirect,
    VertexInput,
    VertexShader,
    TessellationControlShader,
    TessellationEvaluationShader,
    GeometryShader,
    FragmentShader,
    EarlyFragmentTests,
    LateFragmentTests,
    ColorAttachmentOutput,
    ComputeShader,
    AllTransfer,
    Transfer,
    Copy,
    Resolve,
    Blit,
    Clear,
    RayTracingShader,
    AccelerationStructureBuild,
    AccelerationStructureCopy,
    Host,
    AllGraphics,
    AllCommands,
}

impl PipelineStage {
    pub const fn to_vk(&self) -> vk::PipelineStageFlags2 {
        match self {
            PipelineStage::None => vk::PipelineStageFlags2::NONE,
            PipelineStage::TopOfPipe => vk::PipelineStageFlags2::TOP_OF_PIPE,
            PipelineStage::BottomOfPipe => vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            PipelineStage::DrawIndirect => vk::PipelineStageFlags2::DRAW_INDIRECT,
            PipelineStage::VertexInput => vk::PipelineStageFlags2::VERTEX_INPUT,
            PipelineStage::VertexShader => vk::PipelineStageFlags2::VERTEX_SHADER,
            PipelineStage::TessellationControlShader => vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER,
            PipelineStage::TessellationEvaluationShader => vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER,
            PipelineStage::GeometryShader => vk::PipelineStageFlags2::GEOMETRY_SHADER,
            PipelineStage::FragmentShader => vk::PipelineStageFlags2::FRAGMENT_SHADER,
            PipelineStage::EarlyFragmentTests => vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
            PipelineStage::LateFragmentTests => vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            PipelineStage::ColorAttachmentOutput => vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            PipelineStage::ComputeShader => vk::PipelineStageFlags2::COMPUTE_SHADER,

            PipelineStage::AllTransfer => vk::PipelineStageFlags2::ALL_TRANSFER,
            PipelineStage::Transfer => vk::PipelineStageFlags2::TRANSFER,
            PipelineStage::Copy => vk::PipelineStageFlags2::COPY,
            PipelineStage::Resolve => vk::PipelineStageFlags2::RESOLVE,
            PipelineStage::Blit => vk::PipelineStageFlags2::BLIT,
            PipelineStage::Clear => vk::PipelineStageFlags2::CLEAR,

            PipelineStage::RayTracingShader => vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR,
            PipelineStage::AccelerationStructureBuild => vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR,
            PipelineStage::AccelerationStructureCopy => vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_COPY_KHR,

            PipelineStage::Host => vk::PipelineStageFlags2::HOST,
            PipelineStage::AllGraphics => vk::PipelineStageFlags2::ALL_GRAPHICS,
            PipelineStage::AllCommands => vk::PipelineStageFlags2::ALL_COMMANDS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AccessType {
    None,
    Indirect,
    IndexRead,
    VertexRead,
    UniformRead,
    ShaderRead,
    ShaderWrite,
    ColorAttachmentRead,
    ColorAttachmentWrite,
    DepthStencilRead,
    DepthStencilWrite,
    TransferRead,
    TransferWrite,
}

impl AccessType {
    pub(crate) const fn to_vk(&self) -> vk::AccessFlags2 {
        match self {
            AccessType::None => vk::AccessFlags2::empty(),
            AccessType::Indirect => vk::AccessFlags2::INDIRECT_COMMAND_READ,
            AccessType::IndexRead => vk::AccessFlags2::INDEX_READ,
            AccessType::VertexRead => vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            AccessType::UniformRead => vk::AccessFlags2::UNIFORM_READ,
            AccessType::ShaderRead => vk::AccessFlags2::SHADER_READ,
            AccessType::ShaderWrite => vk::AccessFlags2::SHADER_WRITE,
            AccessType::ColorAttachmentRead => vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            AccessType::ColorAttachmentWrite => vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            AccessType::DepthStencilRead => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            AccessType::DepthStencilWrite => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            AccessType::TransferRead => vk::AccessFlags2::TRANSFER_READ,
            AccessType::TransferWrite => vk::AccessFlags2::TRANSFER_WRITE,
        }
    }

    pub(crate) fn is_write(&self) -> bool {
        match self {
            AccessType::ShaderWrite => true,
            AccessType::ColorAttachmentWrite => true,
            AccessType::DepthStencilWrite => true,
            AccessType::TransferWrite => true,
            _ => false,
        }
    }

    pub(crate) fn is_read(&self) -> bool {
        !self.is_write()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryBarrier {
    pub src_stage: PipelineStage,
    pub dst_stage: PipelineStage,
    pub src_access: AccessType,
    pub dst_access: AccessType,
}

impl Default for MemoryBarrier {
    fn default() -> Self {
        return MemoryBarrier {
            src_stage: PipelineStage::TopOfPipe,
            dst_stage: PipelineStage::BottomOfPipe,
            src_access: AccessType::ColorAttachmentRead,
            dst_access: AccessType::ColorAttachmentRead,
        };
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageBarrier {
    pub image: ImageId,
    pub old_layout: ImageLayout,
    pub new_layout: ImageLayout,
    pub src_stage: PipelineStage,
    pub dst_stage: PipelineStage,
    pub src_access: AccessType,
    pub dst_access: AccessType,
    pub src_queue: QueueType,
    pub dst_queue: QueueType,
    pub subresources: ImageSubresources,
}

impl Default for ImageBarrier {
    fn default() -> Self {
        return ImageBarrier {
            image: ImageId::null(),
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::Undefined,
            src_stage: PipelineStage::TopOfPipe,
            dst_stage: PipelineStage::BottomOfPipe,
            src_access: AccessType::ColorAttachmentRead,
            dst_access: AccessType::ColorAttachmentRead,
            src_queue: QueueType::None,
            dst_queue: QueueType::None,
            subresources: ImageSubresources::default(),
        };
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BufferBarrier {
    pub buffer: BufferId,
    pub src_stage: PipelineStage,
    pub dst_stage: PipelineStage,
    pub src_access: AccessType,
    pub dst_access: AccessType,
    pub src_queue: QueueType,
    pub dst_queue: QueueType,
    pub offset: u64,
    pub size: u64,
}

impl Default for BufferBarrier {
    fn default() -> Self {
        return BufferBarrier {
            buffer: BufferId { id: u64::MAX },
            src_stage: PipelineStage::TopOfPipe,
            dst_stage: PipelineStage::BottomOfPipe,
            src_access: AccessType::ColorAttachmentRead,
            dst_access: AccessType::ColorAttachmentRead,
            src_queue: QueueType::None,
            dst_queue: QueueType::None,
            offset: 0,
            size: vk::WHOLE_SIZE,
        };
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Barrier {
    Memory(MemoryBarrier),
    Image(ImageBarrier),
    Buffer(BufferBarrier),
}

// Mesh shaders

pub struct DrawMeshTasksIndirect {}

//Submit info
pub struct SemaphoreInfo {
    pub semaphore: Semaphore,
    pub pipeline_stage: PipelineStage,
    pub value: Option<u64>,
}

pub struct QueueSubmitInfo<'a> {
    pub fence: Option<Fence>,
    pub command_buffers: &'a [ExecutableCommandBuffer],
    pub wait_semaphores: &'a [SemaphoreInfo],
    pub signal_semaphores: &'a [SemaphoreInfo],
}
