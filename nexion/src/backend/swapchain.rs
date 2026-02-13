use ash::vk;
use gpu_allocator::vulkan::Allocation;
use std::cell::{Cell, UnsafeCell};
use std::collections::VecDeque;
use std::sync::Arc;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};

use crate::{AcquiredImage, Fence, ImageId, ImageViewId, Semaphore, SwapchainDescription};

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
    pub(crate) curr_img_indeices: UnsafeCell<VecDeque<u32>>,

    // this is for per image
    pub(crate) images: Vec<ImageId>,
    pub(crate) image_views: Vec<ImageViewId>,
    pub(crate) preset_semaphores: Vec<Semaphore>,

    // this is for per frame
    pub(crate) image_semaphores: Vec<Semaphore>,
    pub(crate) fences: Vec<Fence>,
    pub(crate) image_timeline: Cell<usize>,
    pub(crate) frame_timeline: Cell<usize>,
    pub(crate) device: Arc<InnerDevice>,
}

impl InnerSwapchain {
    pub(crate) fn new(device: Arc<InnerDevice>, surface: &Surface, swapchain_description: &SwapchainDescription, old_swapchain: Option<Arc<InnerSwapchain>>) -> InnerSwapchain {
        assert!(swapchain_description.frames_in_flight <= swapchain_description.image_count as usize);

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
                    width: swapchain_description.width.clamp(support.capabilities.min_image_extent.width, support.capabilities.max_image_extent.width),
                    height: swapchain_description.height.clamp(support.capabilities.min_image_extent.height, support.capabilities.max_image_extent.height),
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
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
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

        let image_ids: Vec<ImageId> = images
            .iter()
            .map(|&image| {
                let id = unsafe {
                    (&mut *device.image_pool.get()).add(crate::backend::gpu_resources::ImageSlot {
                        handle: image,
                        allocation: Allocation::default(),
                        format: surface_format.format,
                    })
                };

                ImageId { id: id }
            })
            .collect();

        let image_views: Vec<ImageViewId> = image_ids.iter().map(|&image_id| device.create_image_view(image_id, &crate::ImageViewDescription::default())).collect();

        let present_semaphore = {
            let mut t: Vec<Semaphore> = vec![];

            for _ in 0..swapchain_description.image_count {
                t.push(Semaphore::Binary(crate::BinarySemaphore { handle: device.create_binary_semaphore() }));
            }

            t
        };

        let (image_semaphores, fences) = {
            let mut t: Vec<Semaphore> = vec![];
            let mut m: Vec<Fence> = vec![];

            for _ in 0..swapchain_description.image_count {
                t.push(Semaphore::Binary(crate::BinarySemaphore { handle: device.create_binary_semaphore() }));
                m.push(Fence { handle: device.create_fence(true) });
            }

            (t, m)
        };

        return InnerSwapchain {
            handle: swapchain,
            swapchain_loader: swapchain_loader,
            desc: swapchain_description.clone(),
            curr_img_indeices: UnsafeCell::new(VecDeque::with_capacity(swapchain_description.image_count as usize)),
            image_views: image_views,
            images: image_ids,
            image_semaphores,
            preset_semaphores: present_semaphore,
            fences: fences,
            image_timeline: Cell::new(0),
            frame_timeline: Cell::new(0),
            device: device,
        };
    }

    pub(crate) unsafe fn create_surface<W: HasDisplayHandle + HasWindowHandle>(device: &Arc<InnerDevice>, window: &W) -> Surface {
        let raw_window = window.window_handle().unwrap().as_raw();
        let raw_display = window.display_handle().unwrap().as_raw();

        let surface_handle = match (raw_window, raw_display) {
            (RawWindowHandle::Win32(w), RawDisplayHandle::Windows(_)) => {
                let info = ash::vk::Win32SurfaceCreateInfoKHR::default().hinstance(w.hinstance.unwrap().get()).hwnd(w.hwnd.get());
                let loader = ash::khr::win32_surface::Instance::new(&device.instance.entry, &device.instance.handle);
                unsafe { loader.create_win32_surface(&info, None).expect("Failed to create surface") }
            }
            (RawWindowHandle::Xcb(w), RawDisplayHandle::Xcb(d)) => {
                let info = ash::vk::XcbSurfaceCreateInfoKHR::default().connection(d.connection.unwrap().as_ptr()).window(w.window.get());
                let loader = ash::khr::xcb_surface::Instance::new(&device.instance.entry, &device.instance.handle);
                unsafe { loader.create_xcb_surface(&info, None).expect("Failed to create surface") }
            }
            (RawWindowHandle::Xlib(w), RawDisplayHandle::Xlib(d)) => {
                let info = ash::vk::XlibSurfaceCreateInfoKHR::default().dpy(d.display.unwrap().as_ptr() as *mut _).window(w.window);
                let loader = ash::khr::xlib_surface::Instance::new(&device.instance.entry, &device.instance.handle);
                unsafe { loader.create_xlib_surface(&info, None).expect("Failed to create surface") }
            }
            (RawWindowHandle::Wayland(w), RawDisplayHandle::Wayland(d)) => {
                let info = ash::vk::WaylandSurfaceCreateInfoKHR::default().display(d.display.as_ptr()).surface(w.surface.as_ptr());
                let loader = ash::khr::wayland_surface::Instance::new(&device.instance.entry, &device.instance.handle);
                unsafe { loader.create_wayland_surface(&info, None).expect("Failed to create surface") }
            }
            (RawWindowHandle::AppKit(w), RawDisplayHandle::AppKit(_)) => {
                let info = ash::vk::MetalSurfaceCreateInfoEXT::default().layer(w.ns_view.as_ptr());
                let loader = ash::ext::metal_surface::Instance::new(&device.instance.entry, &device.instance.handle);
                unsafe { loader.create_metal_surface(&info, None).expect("Failed to create surface") }
            }

            _ => panic!("Unsupported platform or mismatched window/display handle"),
        };

        return Surface {
            handle: surface_handle,
            loader: ash::khr::surface::Instance::new(&device.instance.entry, &device.instance.handle),
        };
    }
}

impl InnerSwapchain {
    pub(crate) fn acquire_image(&self) -> AcquiredImage {
        let image_timeline = self.image_timeline.get();
        let frame_timeline = self.frame_timeline.get();

        let image_semaphore = self.image_semaphores[frame_timeline];
        let fence = self.fences[frame_timeline];

        let (index, _) = unsafe {
            self.device.handle.wait_for_fences(&[fence.handle], true, u64::MAX).expect("Failed to wait for in flight fence");
            self.device.handle.reset_fences(&[fence.handle]).expect("Failed to reset in flight fence");

            let acquire_info = vk::AcquireNextImageInfoKHR::default().swapchain(self.handle).timeout(u64::MAX).semaphore(image_semaphore.handle()).device_mask(1);
            self.swapchain_loader.acquire_next_image2(&acquire_info).expect("Failed to acquire next image")
        };

        unsafe {
            (&mut *self.curr_img_indeices.get()).push_back(index);
        }

        let next_timeline_index = (image_timeline + 1) % self.desc.image_count as usize;
        self.image_timeline.replace(next_timeline_index);

        let next_frame_timeline = (frame_timeline + 1) % self.desc.frames_in_flight;
        self.frame_timeline.replace(next_frame_timeline);

        return AcquiredImage {
            image: self.images[index as usize],
            view: self.image_views[index as usize],
            image_semaphore: image_semaphore,
            present_semaphore: self.preset_semaphores[index as usize],
            fence: fence,
            curr_frame: frame_timeline,
        };
    }

    pub(crate) fn present(&self) {
        let index = unsafe {
            match (&mut *self.curr_img_indeices.get()).pop_back() {
                Some(i) => i,
                _ => {
                    return;
                }
            }
        };
        let sem = [self.preset_semaphores[index as usize].handle()];
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
        self.device.wait_idle();
        for i in 0..self.image_views.len() {
            unsafe {
                (&mut *self.device.image_pool.get()).delete(self.images[i].id);
            };
            self.device.destroy_image_view(self.image_views[i]);
            self.device.destroy_semaphore(self.image_semaphores[i]);
            self.device.destroy_semaphore(self.preset_semaphores[i]);
            self.device.destroy_fence(self.fences[i]);
        }

        unsafe {
            self.swapchain_loader.destroy_swapchain(self.handle, None);
        }
    }
}
