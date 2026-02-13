use std::sync::Arc;

use crate::{
    Fence, ImageId, ImageViewId, Semaphore, SwapchainDescription,
    backend::swapchain::{InnerSwapchain, Surface},
};

/// Swapchain abstraction
/// Contains image and present semaphores internally.
/// This helps manage frames in flight by eliminating the need
/// for manual selection of semaphores
#[derive(Clone)]
pub struct Swapchain {
    pub(crate) inner: Arc<InnerSwapchain>,
    pub(crate) surface: Arc<Surface>,
}

#[derive(Clone, Copy)]
pub struct AcquiredImage {
    pub image: ImageId,
    pub view: ImageViewId,
    pub image_semaphore: Semaphore,
    pub present_semaphore: Semaphore,
    pub fence: Fence,
    pub curr_frame: usize,
}

impl Swapchain {
    pub fn recreate_swapchain(&mut self, width: u32, height: u32) {
        let old_desc = self.inner.desc.clone();
        let desc = SwapchainDescription {
            image_count: old_desc.image_count,
            frames_in_flight: old_desc.frames_in_flight,
            width: width,
            height: height,
        };
        let new_swapchain = InnerSwapchain::new(self.inner.device.clone(), &self.surface, &desc, Some(self.inner.clone()));
        self.inner = Arc::new(new_swapchain);
    }

    pub fn acquire_image(&self) -> AcquiredImage {
        return self.inner.acquire_image();
    }

    pub fn present(&self) {
        self.inner.present();
    }
}
