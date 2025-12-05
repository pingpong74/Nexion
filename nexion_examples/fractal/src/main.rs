use std::sync::Arc;

use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::EventLoop,
    window::{Window, WindowAttributes},
};

mod camera;
mod renderer;
use camera::Camera;

use crate::{camera::CameraController, renderer::Renderer};

struct Application {
    window: Arc<Window>,
    renderer: Renderer,
    camera_controller: CameraController,
    camera: Camera,
    time: f32,
}

impl Application {
    pub fn new(event_loop: &EventLoop<()>) -> Application {
        let window_attribs = WindowAttributes::default();
        let window = Arc::new(
            event_loop
                .create_window(window_attribs)
                .expect("Failed to create window"),
        );
        let size = window.inner_size();

        return Application {
            window: window.clone(),
            renderer: Renderer::new(window.clone()),
            camera_controller: CameraController::new(1.0, 0.7),
            camera: Camera::new(size.width, size.height),
            time: 0.0,
        };
    }
}

#[allow(unused)]
impl ApplicationHandler for Application {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {}

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        self.camera_controller.process_event(&event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.renderer.resize(size.width, size.height);
                self.camera.resize(size);
            }
            WindowEvent::RedrawRequested => {
                let start = Instant::now();
                self.renderer
                    .render(&self.camera, self.time, self.window.inner_size());
                let duration = start.elapsed();
                self.camera_controller
                    .update_camera(&mut self.camera, duration.as_secs_f32());
                self.time += duration.as_secs_f32();
                //println!("{}", duration.as_millis());

                self.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.camera_controller
                .process_mouse_motion(delta.0, delta.1);
        }
    }
}

fn main() {
    let event_loop: EventLoop<()> = EventLoop::with_user_event()
        .build()
        .expect("Failed to create event loop");

    let mut app = Application::new(&event_loop);

    event_loop.run_app(&mut app).expect("Smt?");
}
