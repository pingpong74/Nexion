use nexion::{utils::vulkan_context::*, *};
use std::sync::Arc;
use winit::{dpi::PhysicalSize, window::Window};

use crate::camera::Camera;

const FRAMES_IN_FLIGHT: usize = 2;

#[repr(C)]
#[derive(Copy, Clone)]
struct MyPushConstants {
    view_proj_mat: [[f32; 4]; 4],
    pos: [f32; 3],
    width: u32,
    height: u32,
    time: f32,
}

pub struct Renderer {
    vk_context: VulkanContext,
    raster_pipeline: Pipeline,
    frame_data: [CommandRecorder; FRAMES_IN_FLIGHT],
}

impl Renderer {
    pub fn new(window: Arc<Window>) -> Renderer {
        let size = window.inner_size();

        let vk_context = VulkanContext::new(
            &window,
            &InstanceDescription {
                api_version: ApiVersion::VkApi1_3,
                enable_validation_layers: false,
            },
            &DeviceDescription {
                use_compute_queue: true,
                use_transfer_queue: true,
                ..Default::default()
            },
            &SwapchainDescription {
                image_count: 5,
                frames_in_flight: FRAMES_IN_FLIGHT,
                width: size.width,
                height: size.height,
            },
        );

        let pipeline =
            vk_context.create_rasterization_pipeline(&RasterizationPipelineDescription {
                geometry: GeometryStage::Classic {
                    vertex_input: VertexInputDescription::default(),
                    topology: InputTopology::TriangleList,
                    vertex_shader: "shaders/vertex.slang",
                },
                fragment_shader_path: "shaders/fragment.slang",
                cull_mode: CullMode::Back,
                front_face: FrontFace::Clockwise,
                push_constants: PushConstantsDescription {
                    stage_flags: ShaderStages::FRAGMENT,
                    offset: 0,
                    size: size_of::<MyPushConstants>() as u32,
                },
                outputs: PipelineOutputs {
                    color: &[Format::Rgba16Float],
                    depth: None,
                    stencil: None,
                },
                ..Default::default()
            });

        let frame_data =
            std::array::from_fn(|_| vk_context.create_command_recorder(QueueType::Graphics));

        return Renderer {
            vk_context: vk_context,
            raster_pipeline: pipeline,
            frame_data: frame_data,
        };
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.vk_context.resize(width, height);
    }

    pub fn render(&mut self, camera: &Camera, time: f32, size: PhysicalSize<u32>) {
        let push_constants = MyPushConstants {
            view_proj_mat: camera.get_inv_view_proj(),
            pos: camera.get_pos(),
            width: size.width,
            height: size.height,
            time: time,
        };

        let acquired_image = self.vk_context.acquire_image();
        let curr_frame = acquired_image.curr_frame;

        self.frame_data[curr_frame].reset();

        self.frame_data[curr_frame].begin_recording(CommandBufferUsage::OneTimeSubmit);

        self.frame_data[curr_frame].set_push_constants(&push_constants, self.raster_pipeline);

        self.frame_data[curr_frame].pipeline_barrier(&[Barrier::Image(ImageBarrier {
            image: acquired_image.image,
            old_layout: ImageLayout::Undefined,
            new_layout: ImageLayout::ColorAttachment,
            src_stage: PipelineStage::TopOfPipe,
            dst_stage: PipelineStage::ColorAttachmentOutput,
            src_access: AccessType::None,
            dst_access: AccessType::ColorAttachmentWrite,
            ..Default::default()
        })]);

        self.frame_data[curr_frame].begin_rendering(&RenderingBeginInfo {
            render_area: RenderArea {
                extent: Extent2D {
                    width: size.width,
                    height: size.height,
                },
                offset: Offset2D { x: 0, y: 0 },
            },
            rendering_flags: RenderingFlags::None,
            view_mask: 0,
            layer_count: 1,
            color_attachments: &[RenderingAttachment {
                image_view: acquired_image.view,
                image_layout: ImageLayout::ColorAttachment,
                clear_value: ClearValue::ColorFloat([0.2, 0.2, 0.4, 1.0]),
                ..Default::default()
            }],
            depth_attachment: None,
            stencil_attachment: None,
        });

        self.frame_data[curr_frame].bind_pipeline(self.raster_pipeline);
        self.frame_data[curr_frame].set_viewport_and_scissor(size.width, size.height);
        self.frame_data[curr_frame].draw(3, 1, 0, 0);

        self.frame_data[curr_frame].end_rendering();
        self.frame_data[curr_frame].pipeline_barrier(&[Barrier::Image(ImageBarrier {
            image: acquired_image.image,
            old_layout: ImageLayout::ColorAttachment,
            new_layout: ImageLayout::PresentSrc,
            src_stage: PipelineStage::ColorAttachmentOutput,
            dst_stage: PipelineStage::BottomOfPipe,
            src_access: AccessType::ColorAttachmentWrite,
            dst_access: AccessType::None,
            ..Default::default()
        })]);
        let exec_buffer = self.frame_data[curr_frame].end_recording();

        self.vk_context.submit(&QueueSubmitInfo {
            fence: Some(acquired_image.fence),
            command_buffers: &[exec_buffer],
            wait_semaphores: &[SemaphoreInfo {
                semaphore: acquired_image.image_semaphore,
                pipeline_stage: PipelineStage::ColorAttachmentOutput,
                value: None,
            }],
            signal_semaphores: &[SemaphoreInfo {
                semaphore: acquired_image.present_semaphore,
                pipeline_stage: PipelineStage::BottomOfPipe,
                value: None,
            }],
        });

        self.vk_context.present();
    }
}
