use std::sync::Arc;

use crate::{ImageID, ImageViewID, Semaphore, backend::swapchain::InnerSwapchain};

/// Swapchain abstraction
/// Contains image and present semaphores internally.
/// This helps manage frames in flight by eliminating the need
/// for manual selection of semaphores
#[derive(Clone)]
pub struct Swapchain {
    pub(crate) inner: Arc<InnerSwapchain>,
}

impl Swapchain {
    pub fn acquire_image(&self) -> (ImageID, ImageViewID, Semaphore, Semaphore) {
        return self.inner.acquire_image();
    }

    pub fn present(&self) {
        self.inner.present();
    }
}
