use nexion::*;
use std::time::Instant;
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::EventLoop, window::Window,
};

use std::sync::Arc;

const FRAME_IN_FLIGHT: usize = 3;

vertex!(MyVertex {
    input_rate: Vertex,
    pos: [f32; 2],
    color: [f32; 3],
});

struct FrameData {
    command_recorder: CommandRecorder,
    fence: Fence,
}

#[allow(unused)]
struct VulkanApp {
    window: Arc<Window>,
    instance: Instance,
    device: Device,
    swapchain: Swapchain,
    pipeline_manager: PipelineManager,
    raster_pipeline: RasterizationPipeline,
    vertex_buffer: BufferID,
    color_buffer: BufferID,
    time: f32,
    frame_data: [FrameData; FRAME_IN_FLIGHT],
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

        let instance = Instance::new(&InstanceDescription {
            api_version: ApiVersion::VkApi1_3,
            enable_validation_layers: true,
            window: &window,
        });

        let device = instance.create_device(&DeviceDescription {
            use_compute_queue: true,
            use_transfer_queue: true,
            ray_tracing: false,
            atomic_float_operations: false,
        });

        let swapchain = device.create_swapchain(&SwapchainDescription {
            image_count: 8,
            width: size.width,
            height: size.height,
        });

        let pipeline_manager = device.create_pipeline_manager();
        let raster_pipeline =
            pipeline_manager.create_rasterization_pipeline(&RasterizationPipelineDescription {
                vertex_input: MyVertex::vertex_input_description(),
                vertex_shader_path: "shaders/vertex_shader.slang",
                fragment_shader_path: "shaders/fragment_shader.slang",
                alpha_blend_enable: false,
                outputs: PipelineOutputs {
                    color: vec![Format::Rgba16Float],
                    depth: None,
                    stencil: None,
                },
                ..Default::default()
            });

        let vertex_data = [
            MyVertex {
                pos: [0.5, 0.5],
                color: [0.2, 0.2, 0.8],
            },
            MyVertex {
                pos: [-0.5, 0.5],
                color: [0.2, 0.8, 0.2],
            },
            MyVertex {
                pos: [0.0, -0.5],
                color: [0.8, 0.2, 0.2],
            },
        ];

        let staging_buffer = device.create_buffer(&BufferDescription {
            usage: BufferUsage::TRANSFER_SRC,
            size: 60,
            memory_type: MemoryType::PreferHost,
            create_mapped: true,
        });

        device.write_data_to_buffer(staging_buffer, &vertex_data);

        let vertex_buffer = device.create_buffer(&BufferDescription {
            usage: BufferUsage::TRANSFER_DST | BufferUsage::VERTEX,
            size: 60,
            memory_type: MemoryType::DeviceLocal,
            create_mapped: false,
        });

        let mut recorder = device.create_command_recorder(QueueType::Transfer);
        recorder.begin_recording(CommandBufferUsage::OneTimeSubmit);
        recorder.copy_buffer(&BufferCopyInfo {
            src_buffer: staging_buffer,
            dst_buffer: vertex_buffer,
            size: 60,
            src_offset: 0,
            dst_offset: 0,
        });
        let exec_cmd = recorder.end_recording();
        device.submit(&QueueSubmitInfo {
            fence: None,
            command_buffers: vec![exec_cmd],
            wait_semaphores: vec![],
            signal_semaphores: vec![],
        });
        device.wait_queue(QueueType::Transfer);
        device.destroy_buffer(staging_buffer);

        let color_buffer = device.create_buffer(&BufferDescription {
            usage: BufferUsage::STORAGE,
            size: 12,
            memory_type: MemoryType::PreferHost,
            create_mapped: true,
        });
        let color_data = [[0.1, 0.8, 0.1]];
        device.write_data_to_buffer(color_buffer, &color_data);
        device.write_buffer(&BufferWriteInfo {
            buffer: color_buffer,
            offset: 0,
            range: 12,
            index: 0,
        });

        return VulkanApp {
            frame_data: [
                FrameData {
                    command_recorder: device.create_command_recorder(QueueType::Graphics),
                    fence: device.create_fence(true),
                },
                FrameData {
                    command_recorder: device.create_command_recorder(QueueType::Graphics),
                    fence: device.create_fence(true),
                },
                FrameData {
                    command_recorder: device.create_command_recorder(QueueType::Graphics),
                    fence: device.create_fence(true),
                },
            ],
            window: window,
            instance: instance,
            device: device,
            swapchain: swapchain,
            pipeline_manager: pipeline_manager,
            raster_pipeline: raster_pipeline,
            vertex_buffer: vertex_buffer,
            color_buffer: color_buffer,
            time: 0.0,
        };
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.device.wait_idle();
        let new_swapchain = self.device.recreate_swapchain(
            &SwapchainDescription {
                image_count: 3,
                width: width,
                height: height,
            },
            &self.swapchain,
        );
        let old_swapchain = std::mem::replace(&mut self.swapchain, new_swapchain);
        drop(old_swapchain);
    }

    unsafe fn render(&mut self) {
        let size = self.window.inner_size();
        static mut curr_frame: usize = 0;

        if size.width == 0 || size.height == 0 {
            return;
        }

        let color = {
            // simple hue-based color cycling
            let r = (self.time * 0.5).sin() * 0.5 + 0.5;
            let g = (self.time * 0.7 + std::f32::consts::PI / 2.0).sin() * 0.5 + 0.5;
            let b = (self.time * 1.3 + std::f32::consts::PI).sin() * 0.5 + 0.5;
            [r, g, b]
        };

        self.device
            .write_data_to_buffer(self.color_buffer, &[color]);

        self.device.wait_fence(self.frame_data[curr_frame].fence);
        self.device.reset_fence(self.frame_data[curr_frame].fence);

        let (img, img_view, image_semaphore, present_semaphore) = self.swapchain.acquire_image();

        self.frame_data[curr_frame].command_recorder.reset();

        self.frame_data[curr_frame]
            .command_recorder
            .begin_recording(CommandBufferUsage::OneTimeSubmit);

        self.frame_data[curr_frame]
            .command_recorder
            .pipeline_barrier(&[Barrier::Image(ImageBarrier {
                image: img,
                old_layout: ImageLayout::Undefined,
                new_layout: ImageLayout::ColorAttachment,
                src_stage: PipelineStage::TopOfPipe,
                dst_stage: PipelineStage::ColorAttachmentOutput,
                src_access: AccessType::None,
                dst_access: AccessType::ColorAttachmentWrite,
                ..Default::default()
            })]);

        self.frame_data[curr_frame]
            .command_recorder
            .begin_rendering(&RenderingBeginInfo {
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
                color_attachments: vec![RenderingAttachment {
                    image_view: img_view,
                    image_layout: ImageLayout::ColorAttachment,
                    clear_value: ClearValue::ColorFloat([0.2, 0.2, 0.4, 1.0]),
                    ..Default::default()
                }],
                depth_attachment: None,
                stencil_attachment: None,
            });

        self.frame_data[curr_frame]
            .command_recorder
            .bind_pipeline(&self.raster_pipeline);
        self.frame_data[curr_frame]
            .command_recorder
            .set_viewport_and_scissor(size.width, size.height);
        self.frame_data[curr_frame]
            .command_recorder
            .bind_vertex_buffer(self.vertex_buffer, 0);
        self.frame_data[curr_frame]
            .command_recorder
            .draw(3, 1, 0, 0);

        self.frame_data[curr_frame].command_recorder.end_rendering();
        self.frame_data[curr_frame]
            .command_recorder
            .pipeline_barrier(&[Barrier::Image(ImageBarrier {
                image: img,
                old_layout: ImageLayout::ColorAttachment,
                new_layout: ImageLayout::PresentSrc,
                src_stage: PipelineStage::ColorAttachmentOutput,
                dst_stage: PipelineStage::BottomOfPipe,
                src_access: AccessType::ColorAttachmentWrite,
                dst_access: AccessType::None,
                ..Default::default()
            })]);
        let exec_buffer = self.frame_data[curr_frame].command_recorder.end_recording();

        self.device.submit(&QueueSubmitInfo {
            fence: Some(self.frame_data[curr_frame].fence),
            command_buffers: vec![exec_buffer],
            wait_semaphores: vec![SemaphoreInfo {
                semaphore: image_semaphore,
                pipeline_stage: PipelineStage::ColorAttachmentOutput,
                value: None,
            }],
            signal_semaphores: vec![SemaphoreInfo {
                semaphore: present_semaphore,
                pipeline_stage: PipelineStage::BottomOfPipe,
                value: None,
            }],
        });

        self.swapchain.present();

        unsafe {
            curr_frame = (curr_frame + 1) % FRAME_IN_FLIGHT;
        }
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        self.device.wait_idle();
        self.device.destroy_buffer(self.vertex_buffer);
        self.device.destroy_buffer(self.color_buffer);

        for i in 0..FRAME_IN_FLIGHT {
            self.device.destroy_fence(self.frame_data[i].fence);
        }
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
                unsafe {
                    let start = Instant::now();
                    self.render();
                    let duration = start.elapsed();
                    self.time += duration.as_secs_f32()
                    //println!("{}", duration.as_millis());
                }
                self.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop: EventLoop<()> = EventLoop::with_user_event()
        .build()
        .expect("Failed to create event loop");

    let mut app = VulkanApp::new(&event_loop);

    event_loop.run_app(&mut app).expect("Smt?");
}
