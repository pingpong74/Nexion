use ash::vk;
use crossbeam::queue::ArrayQueue;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use crate::{ImageID, ImageViewID, Semaphore};

use crate::backend::device::InnerDevice;

pub(crate) struct InnerSwapchain {
    pub(crate) swapchain_loader: ash::khr::swapchain::Device,
    pub(crate) handle: vk::SwapchainKHR,
    pub(crate) curr_img_indeices: ArrayQueue<u32>,
    pub(crate) images: Vec<ImageID>,
    pub(crate) image_views: Vec<ImageViewID>,
    pub(crate) image_semaphore: Vec<Semaphore>,
    pub(crate) preset_semaphore: Vec<Semaphore>,
    pub(crate) timeline: AtomicUsize,
    pub(crate) device: Arc<InnerDevice>,
}

impl InnerSwapchain {
    pub(crate) fn acquire_image(&self) -> (ImageID, ImageViewID, Semaphore, Semaphore) {
        let timeline_index = self.timeline.load(std::sync::atomic::Ordering::Relaxed);
        let sem = self.image_semaphore[timeline_index];

        let acquire_info = vk::AcquireNextImageInfoKHR::default().swapchain(self.handle).timeout(u64::MAX).semaphore(sem.handle()).device_mask(1);

        let next_timeline_index = (timeline_index + 1) % self.image_semaphore.len();
        self.timeline.store(next_timeline_index, std::sync::atomic::Ordering::Relaxed);

        let (index, _) = unsafe { self.swapchain_loader.acquire_next_image2(&acquire_info).expect("Failed to acquire next image") };

        self.curr_img_indeices.push(index);

        //println!("{} {}", timeline_index, index);

        return (self.images[index as usize], self.image_views[index as usize], sem, self.preset_semaphore[index as usize]);
    }

    pub(crate) fn present(&self) {
        let index = match self.curr_img_indeices.pop() {
            Some(i) => i,
            _ => {
                return;
            }
        };
        let sem = [self.preset_semaphore[index as usize].handle()];
        let handle = [self.handle];
        let index = [index];

        let present_info = vk::PresentInfoKHR::default().swapchains(&handle).image_indices(&index).wait_semaphores(&sem);

        unsafe {
            self.swapchain_loader.queue_present(self.device.graphics_queue, &present_info).expect("Failed to preset image!!");
        }
    }
}

impl Drop for InnerSwapchain {
    fn drop(&mut self) {
        for i in 0..self.image_views.len() {
            self.device.image_pool.write().unwrap().delete(self.images[i].id);

            self.device.destroy_image_view(self.image_views[i]);
        }

        unsafe {
            self.swapchain_loader.destroy_swapchain(self.handle, None);
        };
    }
}
