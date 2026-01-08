use ash::vk;
use crossbeam::queue::ArrayQueue;
use gpu_allocator::vulkan::Allocation;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

use crate::{ImageID, ImageViewID, Semaphore, SwapchainDescription};

use crate::backend::device::InnerDevice;

pub(crate) struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

pub(crate) struct Surface {
    pub(crate) handle: vk::SurfaceKHR,
    pub(crate) loader: ash::khr::surface::Instance,
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_surface(self.handle, None);
        }
    }
}

impl Surface {
    fn get_swapchain_support(&self, physical_device: ash::vk::PhysicalDevice) -> Option<SwapchainSupport> {
        unsafe {
            let capabilities = self.loader.get_physical_device_surface_capabilities(physical_device, self.handle).ok()?;

            let formats = self.loader.get_physical_device_surface_formats(physical_device, self.handle).ok()?;

            let present_modes = self.loader.get_physical_device_surface_present_modes(physical_device, self.handle).ok()?;

            if formats.is_empty() || present_modes.is_empty() {
                return None;
            } else {
                return Some(SwapchainSupport { capabilities, formats, present_modes });
            }
        }
    }
}

pub(crate) struct InnerSwapchain {
    pub(crate) swapchain_loader: ash::khr::swapchain::Device,
    pub(crate) handle: vk::SwapchainKHR,
    pub(crate) desc: SwapchainDescription,
    pub(crate) curr_img_indeices: ArrayQueue<u32>,
    pub(crate) images: Vec<ImageID>,
    pub(crate) image_views: Vec<ImageViewID>,
    pub(crate) image_semaphore: Vec<Semaphore>,
    pub(crate) preset_semaphore: Vec<Semaphore>,
    pub(crate) timeline: AtomicUsize,
    pub(crate) device: Arc<InnerDevice>,
}

impl InnerSwapchain {
    pub(crate) fn new(device: Arc<InnerDevice>, surface: &Surface, swapchain_description: &SwapchainDescription, old_swapchain: Option<Arc<InnerSwapchain>>) -> InnerSwapchain {
        let swapchain_loader = ash::khr::swapchain::Device::new(&device.instance.handle, &device.handle);

        let support = surface.get_swapchain_support(device.physical_device.handle).expect("Swapchain not supported!!");

        let present_mode = {
            if support.present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
                vk::PresentModeKHR::MAILBOX
            } else {
                vk::PresentModeKHR::FIFO
            }
        };

        let surface_format = {
            support
                .formats
                .iter()
                .cloned()
                .find(|f| f.format == vk::Format::R16G16B16A16_SFLOAT && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .unwrap_or_else(|| support.formats[0])
        };

        let extent = {
            if support.capabilities.current_extent.width != u32::MAX {
                support.capabilities.current_extent
            } else {
                vk::Extent2D {
                    width: swapchain_description
                        .width
                        .clamp(support.capabilities.min_image_extent.width, support.capabilities.max_image_extent.width),
                    height: swapchain_description
                        .height
                        .clamp(support.capabilities.min_image_extent.height, support.capabilities.max_image_extent.height),
                }
            }
        };

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.handle)
            .min_image_count(swapchain_description.image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(match old_swapchain {
                Some(s) => s.handle,
                None => vk::SwapchainKHR::null(),
            });

        let swapchain = unsafe { swapchain_loader.create_swapchain(&create_info, None).expect("Failed to create swapchain") };

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain).expect("Failed to get swapchain images") };

        let image_ids: Vec<ImageID> = images
            .iter()
            .map(|&image| {
                let id = device.image_pool.write().unwrap().add(crate::backend::gpu_resources::ImageSlot {
                    handle: image,
                    allocation: Allocation::default(),
                    format: surface_format.format,
                });

                ImageID { id: id }
            })
            .collect();

        let image_views: Vec<ImageViewID> = image_ids.iter().map(|&image_id| device.create_image_view(image_id, &crate::ImageViewDescription::default())).collect();

        let (image_semapgores, present_semaphore) = {
            let mut t: Vec<Semaphore> = vec![];
            let mut n: Vec<Semaphore> = vec![];

            for _ in 0..swapchain_description.image_count {
                t.push(Semaphore::Binary(crate::BinarySemaphore {
                    handle: device.create_binary_semaphore(),
                }));
                n.push(Semaphore::Binary(crate::BinarySemaphore {
                    handle: device.create_binary_semaphore(),
                }));
            }

            (t, n)
        };

        return InnerSwapchain {
            handle: swapchain,
            swapchain_loader: swapchain_loader,
            desc: swapchain_description.clone(),
            curr_img_indeices: ArrayQueue::new(swapchain_description.image_count as usize),
            image_views: image_views,
            images: image_ids,
            image_semaphore: image_semapgores,
            preset_semaphore: present_semaphore,
            timeline: AtomicUsize::new(0),
            device: device,
        };
    }

    pub(crate) unsafe fn create_surface<W: HasDisplayHandle + HasWindowHandle>(device: &Arc<InnerDevice>, window: &W) -> Surface {
        let raw_window = window.window_handle().unwrap().as_raw();
        let raw_display = window.display_handle().unwrap().as_raw();

        let surface_handle = match (raw_window, raw_display) {
            // ---------------- Windows ----------------
            (RawWindowHandle::Win32(w), RawDisplayHandle::Windows(_)) => {
                let info = ash::vk::Win32SurfaceCreateInfoKHR::default().hinstance(w.hinstance.unwrap().get()).hwnd(w.hwnd.get());
                let loader = ash::khr::win32_surface::Instance::new(&device.instance.entry, &device.instance.handle);
                unsafe { loader.create_win32_surface(&info, None).expect("Failed to create surface") }
            }

            // ---------------- XCB ----------------
            (RawWindowHandle::Xcb(w), RawDisplayHandle::Xcb(d)) => {
                let info = ash::vk::XcbSurfaceCreateInfoKHR::default().connection(d.connection.unwrap().as_ptr()).window(w.window.get());
                let loader = ash::khr::xcb_surface::Instance::new(&device.instance.entry, &device.instance.handle);
                unsafe { loader.create_xcb_surface(&info, None).expect("Failed to create surface") }
            }

            // -------------- Xlib ----------------
            (RawWindowHandle::Xlib(w), RawDisplayHandle::Xlib(d)) => {
                let info = ash::vk::XlibSurfaceCreateInfoKHR::default().dpy(d.display.unwrap().as_ptr() as *mut _).window(w.window);
                let loader = ash::khr::xlib_surface::Instance::new(&device.instance.entry, &device.instance.handle);
                unsafe { loader.create_xlib_surface(&info, None).expect("Failed to create surface") }
            }

            // ---------------- Wayland ----------------
            (RawWindowHandle::Wayland(w), RawDisplayHandle::Wayland(d)) => {
                let info = ash::vk::WaylandSurfaceCreateInfoKHR::default().display(d.display.as_ptr()).surface(w.surface.as_ptr());
                let loader = ash::khr::wayland_surface::Instance::new(&device.instance.entry, &device.instance.handle);
                unsafe { loader.create_wayland_surface(&info, None).expect("Failed to create surface") }
            }

            // ---------------- macOS ----------------
            (RawWindowHandle::AppKit(w), RawDisplayHandle::AppKit(_)) => {
                let info = ash::vk::MetalSurfaceCreateInfoEXT::default().layer(w.ns_view.as_ptr());
                let loader = ash::ext::metal_surface::Instance::new(&device.instance.entry, &device.instance.handle);
                unsafe { loader.create_metal_surface(&info, None).expect("Failed to create surface") }
            }

            // ---------------- Unsupported ----------------
            _ => panic!("Unsupported platform or mismatched window/display handle"),
        };

        return Surface {
            handle: surface_handle,
            loader: ash::khr::surface::Instance::new(&device.instance.entry, &device.instance.handle),
        };
    }
}

impl InnerSwapchain {
    pub(crate) fn acquire_image(&self) -> (ImageID, ImageViewID, Semaphore, Semaphore) {
        let timeline_index = self.timeline.load(std::sync::atomic::Ordering::Relaxed);
        let sem = self.image_semaphore[timeline_index];

        let acquire_info = vk::AcquireNextImageInfoKHR::default().swapchain(self.handle).timeout(u64::MAX).semaphore(sem.handle()).device_mask(1);

        let next_timeline_index = (timeline_index + 1) % self.image_semaphore.len();
        self.timeline.store(next_timeline_index, std::sync::atomic::Ordering::Relaxed);

        let (index, _) = unsafe { self.swapchain_loader.acquire_next_image2(&acquire_info).expect("Failed to acquire next image") };

        self.curr_img_indeices.push(index).unwrap();

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
        }
    }
}
