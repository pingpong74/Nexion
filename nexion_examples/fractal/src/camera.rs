use cgmath::{Matrix4, Point3, Vector3, prelude::*};

use winit::{
    dpi::PhysicalSize,
    event::{ElementState, KeyEvent, MouseScrollDelta, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
};

pub struct CameraController {
    speed: f32,
    sensitivity: f32,
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    rotate_horizontal: f32,
    rotate_vertical: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            speed,
            sensitivity,
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
        }
    }

    pub fn process_event(&mut self, event: &WindowEvent) {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key,
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = *state == ElementState::Pressed;
                if let PhysicalKey::Code(keycode) = physical_key {
                    match keycode {
                        KeyCode::KeyW => self.forward = pressed,
                        KeyCode::KeyS => self.backward = pressed,
                        KeyCode::KeyA => self.left = pressed,
                        KeyCode::KeyD => self.right = pressed,
                        KeyCode::Space => self.up = pressed,
                        KeyCode::ShiftLeft => self.down = pressed,
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let MouseScrollDelta::LineDelta(_, y) = delta {
                    self.speed = (self.speed + y * 0.1).max(0.0);
                }
            }
            _ => {}
        }
    }

    pub fn process_mouse_motion(&mut self, delta_x: f64, delta_y: f64) {
        self.rotate_horizontal += delta_x as f32;
        self.rotate_vertical += delta_y as f32;
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: f32) {
        let forward_dir = (camera.target - camera.eye).normalize();
        let right_dir = forward_dir.cross(camera.up).normalize();

        println!("{} {}", self.rotate_horizontal, self.rotate_vertical);

        if self.forward {
            camera.eye += forward_dir * self.speed * dt;
            camera.target += forward_dir * self.speed * dt;
        }
        if self.backward {
            camera.eye -= forward_dir * self.speed * dt;
            camera.target -= forward_dir * self.speed * dt;
        }
        if self.right {
            camera.eye += right_dir * self.speed * dt;
            camera.target += right_dir * self.speed * dt;
        }
        if self.left {
            camera.eye -= right_dir * self.speed * dt;
            camera.target -= right_dir * self.speed * dt;
        }
        if self.up {
            camera.eye += camera.up * self.speed * dt;
            camera.target += camera.up * self.speed * dt;
        }
        if self.down {
            camera.eye -= camera.up * self.speed * dt;
            camera.target -= camera.up * self.speed * dt;
        }

        if self.rotate_horizontal != 0.0 || self.rotate_vertical != 0.0 {
            let yaw = Matrix4::from_axis_angle(
                camera.up,
                cgmath::Rad(-self.rotate_horizontal * self.sensitivity * dt),
            );
            let right = (camera.target - camera.eye).cross(camera.up).normalize();
            let pitch = Matrix4::from_axis_angle(
                right,
                cgmath::Rad(-self.rotate_vertical * self.sensitivity * dt),
            );

            let forward = (camera.target - camera.eye).normalize();
            let rotated_forward = (yaw * pitch).transform_vector(forward);
            camera.target = camera.eye + rotated_forward;

            self.rotate_horizontal = 0.0;
            self.rotate_vertical = 0.0;
        }
    }
}

pub struct Camera {
    pub eye: Point3<f32>,
    pub target: Point3<f32>,
    up: Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    //create a camera at 0,0,0 facing towards +x
    pub fn new(width: u32, height: u32) -> Self {
        return Camera {
            eye: Point3::new(2.0, 0.0, 0.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::unit_y(),
            aspect: width as f32 / height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 1000.0,
        };
    }

    pub fn resize(&mut self, size: PhysicalSize<u32>) {
        self.aspect = size.width as f32 / size.height as f32;
    }

    pub fn get_pos(&self) -> [f32; 3] {
        return self.eye.into();
    }

    pub fn get_inv_view_proj(&self) -> [[f32; 4]; 4] {
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        let inv = (proj * view).invert().unwrap();

        return inv.into();
    }
}
