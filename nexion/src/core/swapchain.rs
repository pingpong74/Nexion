use std::sync::Arc;

use crate::{
    ImageID, ImageViewID, Semaphore, SwapchainDescription,
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

impl Swapchain {
    pub fn recreate_swapchain(&mut self, width: u32, height: u32) {
        let old_desc = self.inner.desc.clone();
        let desc = SwapchainDescription {
            image_count: old_desc.image_count,
            width: width,
            height: height,
        };
        let new_swapchain = InnerSwapchain::new(self.inner.device.clone(), &self.surface, &desc, Some(self.inner.clone()));
        self.inner = Arc::new(new_swapchain);
    }

    pub fn acquire_image(&self) -> (ImageID, ImageViewID, Semaphore, Semaphore) {
        return self.inner.acquire_image();
    }

    pub fn present(&self) {
        self.inner.present();
    }
}
