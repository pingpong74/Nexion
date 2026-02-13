use crate::*;

pub struct TextureWriteInfo {
    pub stg_buffer: BufferId,
    pub buffer_offset: u64,
    pub width: u32,
    pub height: u32,
    pub src_queue: QueueType,
    pub dst_queue: QueueType,
}

pub struct TextureDescription {}

#[derive(Clone, Copy)]
pub struct Texture {
    pub image: ImageId,
    pub image_view: ImageViewId,
}

impl Texture {
    #[inline]
    pub fn write(&self, recorder: &mut CommandRecorder, texture_write_info: &TextureWriteInfo) {
        recorder.pipeline_barrier(&[Barrier::Image(ImageBarrier {
            image: self.image,
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::TransferDst,
            src_access: AccessType::None,
            dst_access: AccessType::TransferWrite,
            dst_stage: PipelineStage::Transfer,
            ..Default::default()
        })]);
        recorder.copy_buffer_to_image(&BufferImageCopyInfo {
            buffer: texture_write_info.stg_buffer,
            image: self.image,
            dst_image_layout: ImageLayout::TransferDst,
            region: BufferImageCopyRegion {
                buffer_offset: texture_write_info.buffer_offset,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: ImageSubresources {
                    aspect: ImageAspect::Color,
                    mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: Offset3D { x: 0, y: 0, z: 0 },
                image_extent: Extent3D { width: texture_write_info.width, height: texture_write_info.height, depth: 1 },
            },
        });
        recorder.pipeline_barrier(&[Barrier::Image(ImageBarrier {
            image: self.image,
            old_layout: ImageLayout::TransferDst,
            new_layout: ImageLayout::ShaderReadOnly,
            src_stage: PipelineStage::Transfer,
            dst_stage: PipelineStage::None,
            src_access: AccessType::TransferWrite,
            dst_access: AccessType::None,
            src_queue: texture_write_info.src_queue,
            dst_queue: texture_write_info.dst_queue,
            ..Default::default()
        })]);
    }
}
