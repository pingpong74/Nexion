use image::ImageReader;
use nexion::{
    utils::texture::{Texture, TextureWriteInfo},
    *,
};
use std::time::Instant;
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::EventLoop, window::Window,
};

use std::sync::Arc;

const FRAME_IN_FLIGHT: usize = 3;

vertex!(MyVertex {
    input_rate: Vertex,
    pos: [f32; 2],
    uv: [f32; 2],
});

struct PushConstants {
    color: u64,
}

#[allow(unused)]
struct VulkanApp {
    window: Arc<Window>,
    instance: Instance,
    device: Device,
    swapchain: Swapchain,
    raster_pipeline: Pipeline,
    vertex_buffer: BufferId,
    color_buffer: BufferId,
    texture: Texture,
    texture_sampler: SamplerId,
    time: f32,
    frame_data: [CommandRecorder; FRAME_IN_FLIGHT],
}

impl VulkanApp {
    fn new(event_loop: &EventLoop<()>) -> VulkanApp {
        let window_attributes = Window::default_attributes();

        let window = Arc::new(
            event_loop
                .create_window(window_attributes)
                .expect("Failed to create window"),
        );

        let size = window.inner_size();

        let instance = Instance::new(
            &window,
            &InstanceDescription {
                api_version: ApiVersion::VkApi1_3,
                enable_validation_layers: true,
            },
        );

        let device = instance.create_device(&DeviceDescription::default());

        let swapchain = device.create_swapchain(
            &window,
            &SwapchainDescription {
                image_count: 5,
                frames_in_flight: FRAME_IN_FLIGHT,
                width: size.width,
                height: size.height,
            },
        );

        let raster_pipeline =
            device.create_rasterization_pipeline(&RasterizationPipelineDescription {
                geometry: GeometryStage::Classic {
                    vertex_input: MyVertex::vertex_input_description(),
                    topology: InputTopology::TriangleList,
                    vertex_shader: "shaders/vertex_shader.slang",
                },
                fragment_shader_path: "shaders/fragment_shader.slang",
                outputs: PipelineOutputs {
                    color: &[Format::Rgba16Float],
                    depth: None,
                    stencil: None,
                },
                ..Default::default()
            });

        let vertex_data = [
            MyVertex {
                pos: [-0.5, -0.5],
                uv: [0.0, 0.0],
            },
            MyVertex {
                pos: [0.5, -0.5],
                uv: [1.0, 0.0],
            },
            MyVertex {
                pos: [0.5, 0.5],
                uv: [1.0, 1.0],
            },
            MyVertex {
                pos: [-0.5, -0.5],
                uv: [0.0, 0.0],
            },
            MyVertex {
                pos: [0.5, 0.5],
                uv: [1.0, 1.0],
            },
            MyVertex {
                pos: [-0.5, 0.5],
                uv: [0.0, 1.0],
            },
        ];

        let img_file = ImageReader::open("textures/potatoe.png")
            .unwrap()
            .decode()
            .unwrap()
            .to_rgba8();
        let (width, height) = img_file.dimensions();
        let bytes = img_file.into_raw();

        let texture = device.create_texture(
            &ImageDescription {
                usage: ImageUsage::SAMPLED | ImageUsage::TRANSFER_DST,
                format: Format::Rgba8Unorm,
                extent: Extent3D {
                    width: width,
                    height: height,
                    depth: 1,
                },
                memory_type: MemoryType::DeviceLocal,
                ..Default::default()
            },
            &ImageViewDescription::default(),
            3,
        );

        let texture_sampler = device.create_sampler(&SamplerDescription {
            max_anisotropy: Some(16.0),
            ..Default::default()
        });

        let staging_buffer = device.create_buffer(&BufferDescription {
            usage: BufferUsage::TRANSFER_SRC,
            size: bytes.len() as u64,
            memory_type: MemoryType::PreferHost,
            create_mapped: true,
        });

        device.write_data_to_buffer(staging_buffer, &vertex_data);

        let vertex_buffer = device.create_buffer(&BufferDescription {
            usage: BufferUsage::TRANSFER_DST | BufferUsage::VERTEX,
            size: 96,
            memory_type: MemoryType::DeviceLocal,
            create_mapped: false,
        });

        let mut recorder = device.create_command_recorder(QueueType::Graphics);
        recorder.begin_recording(CommandBufferUsage::OneTimeSubmit);
        recorder.copy_buffer(&BufferCopyInfo {
            src_buffer: staging_buffer,
            dst_buffer: vertex_buffer,
            regions: &[CopyRegion {
                size: 96,
                src_offset: 0,
                dst_offset: 0,
            }],
        });
        let exec_cmd = recorder.end_recording();
        device.submit(&QueueSubmitInfo {
            fence: None,
            command_buffers: &[exec_cmd],
            wait_semaphores: &[],
            signal_semaphores: &[],
        });
        device.wait_queue(QueueType::Graphics);

        device.write_data_to_buffer(staging_buffer, bytes.as_slice());

        recorder.reset();
        recorder.begin_recording(CommandBufferUsage::OneTimeSubmit);
        texture.write(
            &mut recorder,
            &TextureWriteInfo {
                stg_buffer: staging_buffer,
                buffer_offset: 0,
                width: width,
                height: height,
                src_queue: QueueType::Graphics,
                dst_queue: QueueType::Graphics,
            },
        );

        let exec_buffer = recorder.end_recording();

        device.submit(&QueueSubmitInfo {
            fence: None,
            command_buffers: &[exec_buffer],
            wait_semaphores: &[],
            signal_semaphores: &[],
        });
        device.wait_queue(QueueType::Graphics);

        device.destroy_buffer(staging_buffer);

        let color_buffer = device.create_buffer(&BufferDescription {
            usage: BufferUsage::STORAGE,
            size: 12,
            memory_type: MemoryType::PreferHost,
            create_mapped: true,
        });
        let color_data = [0.1, 0.8, 0.1];
        device.write_data_to_buffer(color_buffer, &color_data);

        device.write_sampler(&SamplerWriteInfo {
            sampler: texture_sampler,
            index: 0,
        });

        return VulkanApp {
            frame_data: [
                device.create_command_recorder(QueueType::Graphics),
                device.create_command_recorder(QueueType::Graphics),
                device.create_command_recorder(QueueType::Graphics),
            ],
            window: window,
            instance: instance,
            device: device,
            swapchain: swapchain,
            raster_pipeline: raster_pipeline,
            vertex_buffer: vertex_buffer,
            color_buffer: color_buffer,
            texture: texture,
            texture_sampler: texture_sampler,
            time: 0.0,
        };
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.device.wait_idle();
        self.swapchain.recreate_swapchain(width, height);
    }

    fn render(&mut self) {
        let size = self.window.inner_size();

        if size.width == 0 || size.height == 0 {
            return;
        }

        let color = {
            // simple hue-based color cycling
            let r = (self.time * 10.0).sin() * 0.5 + 0.5;
            let g = (self.time * 0.7 + std::f32::consts::PI / 2.0).sin() * 0.5 + 0.5;
            let b = (self.time * 1.3 + std::f32::consts::PI).sin() * 0.5 + 0.5;
            [r, g, b]
        };

        self.device
            .write_data_to_buffer(self.color_buffer, &[color]);

        let acquired_image = self.swapchain.acquire_image();
        let curr_frame = acquired_image.curr_frame;

        self.frame_data[curr_frame].reset();

        self.frame_data[curr_frame].begin_recording(CommandBufferUsage::OneTimeSubmit);

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
                offset: Offset2D { x: 0, y: 0 },
                extent: Extent2D {
                    width: size.width,
                    height: size.height,
                },
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

        self.frame_data[curr_frame].set_push_constants(
            &PushConstants {
                color: self.device.get_buffer_address(self.color_buffer),
            },
            self.raster_pipeline,
        );
        self.frame_data[curr_frame].bind_pipeline(self.raster_pipeline);
        self.frame_data[curr_frame].set_viewport_and_scissor(size.width, size.height);
        self.frame_data[curr_frame].bind_vertex_buffer(self.vertex_buffer, 0);
        self.frame_data[curr_frame].draw(6, 1, 0, 0);

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

        self.device.submit(&QueueSubmitInfo {
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

        self.swapchain.present();
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        self.device.wait_idle();
        self.device.destroy_buffer(self.vertex_buffer);
        self.device.destroy_buffer(self.color_buffer);

        self.device.destory_texture(self.texture);
        self.device.destroy_sampler(self.texture_sampler);
    }
}

#[allow(unused)]
impl ApplicationHandler for VulkanApp {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {}

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => self.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                let start = Instant::now();
                self.render();
                let duration = start.elapsed();
                self.time += duration.as_secs_f32();
                //println!("{}", duration.as_millis());

                self.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    add_shader_directory("shaders");

    let event_loop: EventLoop<()> = EventLoop::with_user_event()
        .build()
        .expect("Failed to create event loop");

    let mut app = VulkanApp::new(&event_loop);

    event_loop.run_app(&mut app).expect("Smt?");
}
